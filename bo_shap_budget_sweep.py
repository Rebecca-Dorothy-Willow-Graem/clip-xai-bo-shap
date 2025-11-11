import os, time, math, itertools, random, argparse
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import shap

# ================================================
# Measure how fast BO‑SHAP reaches a target quality
#  - Builds the SAME superpixels from SHAP partition tree
#  - Computes a SHAP baseline metric (deletion/insertion AUC)
#  - Sweeps BO‑SHAP budgets (k, tail_trials) and stops when
#    achieving >= fraction * baseline (insertion ↑, deletion ↓)
#  - Logs model call count and walltime for the first success
# ================================================

# ---- utils ----

def set_seeds(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CallCounter:
    def __init__(self): self.count = 0
    def wrap(self, f):
        def g(x):
            self.count += len(x)
            return f(x)
        return g

# ---- data ----

def load_captions(path: str):
    df = pd.read_csv(path, header=None, names=["image","caption"]) 
    caps_by_img = df.groupby("image")["caption"].apply(list).to_dict()
    all_caps = df["caption"].tolist()
    return caps_by_img, all_caps

def get_image_and_captions(image_dir, caps_by_img, img_name):
    pil = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
    return pil, caps_by_img[img_name]

# ---- model ----

def build_model_fn(clip_model, clip_proc, device: str, target_text: str, negative_texts: List[str]):
    with torch.no_grad():
        tok = clip_proc(text=[target_text] + negative_texts, return_tensors="pt", padding=True).to(device)
        text_feat = clip_model.get_text_features(**tok)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    def f(img_batch: np.ndarray):
        pil_batch = [Image.fromarray(x.astype(np.uint8)).convert("RGB") for x in img_batch]
        inp = clip_proc(images=pil_batch, return_tensors="pt")
        with torch.no_grad():
            img_feat = clip_model.get_image_features(pixel_values=inp["pixel_values"].to(device))
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            logits = img_feat @ text_feat.T
            probs  = torch.softmax(logits, dim=1)
            return probs[:,0:1].cpu().numpy()
    return f

# ---- partition-tree superpixels ----

def build_segments_from_partition_tree(pil: Image.Image, blur_kernel=(16,16), n_segs: int = 24):
    H,W = pil.height, pil.width
    x = np.array(pil)
    masker = shap.maskers.Image(f"blur({blur_kernel[0]},{blur_kernel[1]})", shape=x.shape)
    clustering = np.array(masker.clustering, dtype=float)
    C = x.shape[2]; M = H*W*C
    def leaf_to_yx(idx):
        i = idx // (W*C); rest = idx % (W*C); j = rest // C; return int(i), int(j)
    class Node:
        __slots__=("nid","li","ri","size")
        def __init__(self,nid,li,ri,size): self.nid=nid; self.li=li; self.ri=ri; self.size=int(size)
    nodes={}
    for r in range(clustering.shape[0]):
        li=int(clustering[r,0]); ri=int(clustering[r,1]); size=int(clustering[r,3]); nid=M+r
        nodes[nid]=Node(nid,li,ri,size)
    root=M+(clustering.shape[0]-1)
    groups=[root]
    def is_internal(nid): return nid>=M
    while len(groups)<n_segs:
        intern=[g for g in groups if is_internal(g)]
        if not intern: break
        g=max(intern, key=lambda z:nodes[z].size)
        groups.remove(g); groups.extend([nodes[g].li, nodes[g].ri])
    seg = -np.ones((H,W),dtype=np.int32)
    from collections import deque
    yx_cache={}
    def assign(nid,sid):
        st=[nid]
        while st:
            cur=st.pop()
            if cur<M:
                if cur not in yx_cache: yx_cache[cur]=leaf_to_yx(cur)
                y,x=yx_cache[cur]
                if seg[y,x]<0: seg[y,x]=sid
            else:
                n=nodes[cur]; st.append(n.li); st.append(n.ri)
    for sid,g in enumerate(groups): assign(g,sid)
    q=deque([(y,x) for y in range(H) for x in range(W) if seg[y,x]>=0])
    while q:
        y,x=q.popleft()
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=ny<H and 0<=nx<W and seg[ny,nx]<0:
                seg[ny,nx]=seg[y,x]; q.append((ny,nx))
    return seg

# ---- masking & eval ----

def blur_image(pil: Image.Image, radius: int = 8):
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_subset_mask_segments(pil, seg, subset, baseline):
    arr=np.array(pil).copy(); base=np.array(baseline)
    keep=np.zeros(seg.max()+1,dtype=bool)
    subset=list(subset)
    if len(subset)>0: keep[subset]=True
    mask=keep[seg]; out=base.copy(); out[mask]=arr[mask]
    return out.astype(np.uint8)

# ---- BO‑SHAP exact ≤k + tail ----

def shapley_weight(n,s):
    from math import factorial
    return math.factorial(s)*math.factorial(n-s-1)/math.factorial(n)

def enumerate_subsets_without_i(n,i,max_size):
    idxs=[j for j in range(n) if j!=i]
    out={s:[] for s in range(max_size+1)}
    for s in range(max_size+1):
        out[s]=list(itertools.combinations(idxs,s))
    return out

class EvalCache:
    def __init__(self): self.cache={}
    def key(self,size,subset): return (size,subset)
    def get(self,k): return self.cache.get(k,None)
    def set(self,k,v): self.cache[k]=v

def eval_subset_prob(model_fn,pil,seg,baseline,subset,cache:EvalCache):
    key=cache.key(len(subset),subset); val=cache.get(key)
    if val is not None: return val
    x=apply_subset_mask_segments(pil,seg,subset,baseline)[None,...]
    prob=float(model_fn(x)[0,0]); cache.set(key,prob); return prob

def exact_phi_leq_k(model_fn,pil,seg,k=2,blur_radius=8):
    N=int(seg.max()+1); baseline=blur_image(pil,blur_radius); cache=EvalCache(); phi=np.zeros(N)
    def total_w(n,k):
        acc=0.0
        for s in range(0,k+1): acc+= math.comb(n-1,s)*shapley_weight(n,s)
        return acc
    for i in range(N):
        subsets_by_size=enumerate_subsets_without_i(N,i,k); contrib=0.0
        for s,subs in subsets_by_size.items():
            w_s=shapley_weight(N,s)
            for T in subs:
                T=tuple(sorted(T)); prob_T=eval_subset_prob(model_fn,pil,seg,baseline,T,cache)
                T_plus_i=tuple(sorted(T+(i,))); prob_Ti=eval_subset_prob(model_fn,pil,seg,baseline,T_plus_i,cache)
                contrib+= w_s*(prob_Ti-prob_T)
        phi[i]=contrib
    residual=1.0 - total_w(N,k)
    return phi,{"residual_weight_mass":float(residual)}

def tail_bound_BO(model_fn,pil,seg,k,trials=32,s_max_extra=2,blur_radius=8,seed=0):
    set_seeds(seed); N=int(seg.max()+1); baseline=blur_image(pil,blur_radius); cache=EvalCache()
    def residual_mass(n,k):
        acc=0.0
        for s in range(0,k+1): acc+= math.comb(n-1,s)*shapley_weight(n,s)
        return 1.0-acc
    residual=residual_mass(N,k)
    max_abs=np.zeros(N)
    sizes=list(range(k+1, min(N,k+1+s_max_extra)+1))
    for i in range(N):
        best=0.0; others=[j for j in range(N) if j!=i]
        for _ in range(trials):
            random.shuffle(others)
            for s in sizes:
                if s>len(others): continue
                T=tuple(sorted(others[:s]))
                pT=eval_subset_prob(model_fn,pil,seg,baseline,T,cache)
                pTi=eval_subset_prob(model_fn,pil,seg,baseline,tuple(sorted(T+(i,))),cache)
                d=abs(pTi-pT); best=max(best,d)
        max_abs[i]=best
    return residual*max_abs,{"residual_weight_mass":float(residual)}

# ---- metrics ----

def segment_scores_from_pixel_map(seg, heat):
    N=int(seg.max()+1); out=np.zeros(N)
    for sid in range(N):
        m=(seg==sid); out[sid]=heat[m].mean() if m.sum()>0 else 0.0
    return out

def blur_image_fast(pil, r): return blur_image(pil, r)

def eval_curve(model_fn,pil,seg,scores,mode,steps=10,blur_radius=8):
    assert mode in ("deletion","insertion")
    baseline=blur_image_fast(pil,blur_radius); order=np.argsort(-scores)
    fracs=[]; probs=[]; N=int(seg.max()+1)
    for t in range(steps+1):
        frac=t/steps; k=int(round(frac*N))
        if mode=="deletion": keep=set(range(N)) - set(order[:k])
        else: keep=set(order[:k])
        x=apply_subset_mask_segments(pil,seg,keep,baseline)[None,...]
        prob=float(model_fn(x)[0,0])
        fracs.append(frac); probs.append(prob)
    auc=0.0
    for i in range(len(fracs)-1):
        dx=fracs[i+1]-fracs[i]; auc+=0.5*dx*(probs[i]+probs[i+1])
    return float(auc)

# ---- main sweep ----

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--img', required=True)
    ap.add_argument('--model', default='openai/clip-vit-base-patch32')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_neg', type=int, default=4)
    ap.add_argument('--blur', type=int, default=8)
    ap.add_argument('--n_segs', type=int, default=24)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--max_evals', type=int, default=200, help='SHAP baseline budget')
    ap.add_argument('--target_frac', type=float, default=0.95, help='fraction of SHAP quality to reach')
    ap.add_argument('--sweep_k', type=int, nargs='+', default=[1,2])
    ap.add_argument('--sweep_trials', type=int, nargs='+', default=[0,4,8,16,32,64])
    args=ap.parse_args()

    set_seeds(args.seed)

    DATA_DIR=os.path.join(args.root,'flicker8k')
    CAP=os.path.join(DATA_DIR,'captions.txt')
    IMG_DIR=os.path.join(DATA_DIR,'images')

    caps_by_img, all_caps=load_captions(CAP)
    pil, caps=get_image_and_captions(IMG_DIR, caps_by_img, args.img)
    target = caps[0]
    rng = random.Random(args.seed)
    neg_candidates = [c for c in all_caps if c not in caps]  # exclude any caption of the same image
    if len(neg_candidates) < args.n_neg:
        raise ValueError("Not enough negative candidates in the corpus.")
    neg = rng.sample(neg_candidates, args.n_neg)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    clip_model=CLIPModel.from_pretrained(args.model).to(device).eval()
    clip_proc=CLIPProcessor.from_pretrained(args.model)

    base_model_fn=build_model_fn(clip_model, clip_proc, device, target, neg)

    # common segments
    seg=build_segments_from_partition_tree(pil, blur_kernel=(args.blur,args.blur), n_segs=args.n_segs)

    # SHAP baseline
    cc_s=CallCounter(); model_fn_s=cc_s.wrap(base_model_fn)
    masker=shap.maskers.Image(f"blur({args.blur},{args.blur})", shape=np.array(pil).shape)
    expl=shap.Explainer(model_fn_s, masker)
    sv=expl(np.array(pil)[None,...], max_evals=args.max_evals)
    shap_pixel=sv.values.mean(axis=-1)[0]
    phi_shap=segment_scores_from_pixel_map(seg, shap_pixel)

    del_auc_s=eval_curve(model_fn_s, pil, seg, phi_shap, 'deletion', steps=args.steps, blur_radius=args.blur)
    ins_auc_s=eval_curve(model_fn_s, pil, seg, phi_shap, 'insertion', steps=args.steps, blur_radius=args.blur)

    target_del=del_auc_s * args.target_frac   # smaller is better
    target_ins=ins_auc_s * args.target_frac   # larger is better → we compare using >=

    print("[BASELINE SHAP]")
    print({"calls":cc_s.count, "del_auc":del_auc_s, "ins_auc":ins_auc_s})

    # BO‑SHAP sweep
    best=None
    for k in args.sweep_k:
        for trials in args.sweep_trials:
            cc_b=CallCounter(); model_fn_b=cc_b.wrap(base_model_fn)
            t0=time.time()
            phi_leq_k,_=exact_phi_leq_k(model_fn_b, pil, seg, k=k, blur_radius=args.blur)
            tail,_ = tail_bound_BO(model_fn_b, pil, seg, k=k, trials=trials, s_max_extra=2, blur_radius=args.blur, seed=args.seed)
            del_auc_b=eval_curve(model_fn_b, pil, seg, phi_leq_k, 'deletion', steps=args.steps, blur_radius=args.blur)
            ins_auc_b=eval_curve(model_fn_b, pil, seg, phi_leq_k, 'insertion', steps=args.steps, blur_radius=args.blur)
            dt=time.time()-t0
            ok = (del_auc_b <= target_del) and (ins_auc_b >= target_ins)
            print({"k":k, "trials":trials, "calls":cc_b.count, "time_sec":dt, "del_auc":del_auc_b, "ins_auc":ins_auc_b, "meets_target":ok})
            if ok and (best is None or cc_b.count < best['calls']):
                best={"k":k, "trials":trials, "calls":cc_b.count, "time_sec":dt, "del_auc":del_auc_b, "ins_auc":ins_auc_b}
        
    print("\n[RESULT]")
    if best is None:
        print("No BO‑SHAP setting reached the target fraction. Try relaxing --target_frac or expanding sweep.")
    else:
        print({"target_frac":args.target_frac, "baseline_calls":cc_s.count, "baseline_del":del_auc_s, "baseline_ins":ins_auc_s, "best":best})

if __name__=='__main__':
    main()
