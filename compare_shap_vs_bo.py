# filename: compare_shap_vs_bo.py  (네가 올린 스크립트의 강화판)
import os, math, itertools, random, argparse, time
from typing import List, Tuple, Dict, Iterable

import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

import shap  # official SHAP

# =============================================================
# End-to-end comparison: Original SHAP vs BO-SHAP(≤k)+Tail (CLIP)
#  - Uses SHAP Image masker partition-tree to define the SAME superpixels
#  - Runs original SHAP (pixel-level) then aggregates to superpixels
#  - Runs BO-SHAP on the same superpixels (exact ≤k + tail bound)
#  - Computes Deletion/Insertion AUC, rank corr, L1, bound coverage
#  - Saves visual overlays and metric report
#  - [NEW] 시간/호출 수/고유마스크 수를 섹션별로 집계 출력
# =============================================================

# ----------------------------
# Deterministic seeds
# ----------------------------

def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data loading (Flickr8k-style)
# ----------------------------

def load_captions(caption_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    df = pd.read_csv(caption_path, header=None, names=["image", "caption"])
    caps_by_img = df.groupby("image")["caption"].apply(list).to_dict()
    all_caps = df["caption"].tolist()
    return caps_by_img, all_caps


def get_image_and_captions(image_dir: str, caps_by_img: Dict[str, List[str]], img_name: str):
    pil = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
    return pil, caps_by_img[img_name]

# ----------------------------
# CLIP model + counting wrapper
# ----------------------------

class CallCounter:
    """배치 내 아이템 수(=실제 모델 평가된 이미지 수) 기준으로 호출 수를 누적."""
    def __init__(self):
        self.count = 0
    def wrap(self, f):
        def g(x):
            self.count += len(x)
            return f(x)
        return g


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
            probs = torch.softmax(logits, dim=1)
            return probs[:, 0:1].cpu().numpy()
    return f

# ----------------------------
# SHAP partition-tree → superpixel segments
# ----------------------------

def build_segments_from_partition_tree(pil: Image.Image, blur_kernel=(16,16), n_segs: int = 32) -> np.ndarray:
    """Use shap.maskers.Image to build the hierarchical partition tree, then cut into ~n_segs leaves."""
    H, W = pil.height, pil.width
    x = np.array(pil)
    masker = shap.maskers.Image(f"blur({blur_kernel[0]},{blur_kernel[1]})", shape=x.shape)

    clustering = np.array(masker.clustering, dtype=float)
    C = x.shape[2]
    M = H * W * C

    def leaf_to_yx(leaf_idx: int) -> Tuple[int, int]:
        i = leaf_idx // (W * C)
        rest = leaf_idx % (W * C)
        j = rest // C
        return int(i), int(j)

    class Node:
        __slots__ = ("nid", "li", "ri", "size")
        def __init__(self, nid: int, li: int, ri: int, size: int):
            self.nid = nid; self.li = li; self.ri = ri; self.size = int(size)

    nodes: Dict[int, Node] = {}
    for row in range(clustering.shape[0]):
        li = int(clustering[row, 0]); ri = int(clustering[row, 1]); size = int(clustering[row, 3])
        nid = M + row
        nodes[nid] = Node(nid, li, ri, size)

    root_id = M + (clustering.shape[0] - 1)

    groups = [root_id]
    def children(nid: int) -> Tuple[int, int]:
        n = nodes[nid]; return n.li, n.ri
    def is_internal(nid: int) -> bool:
        return nid >= M

    while len(groups) < n_segs:
        internal_groups = [g for g in groups if is_internal(g)]
        if not internal_groups: break
        g = max(internal_groups, key=lambda nid: nodes[nid].size)
        groups.remove(g)
        li, ri = children(g)
        groups.extend([li, ri])

    seg_map = -np.ones((H, W), dtype=np.int32)

    from collections import deque
    yx_cache: Dict[int, Tuple[int,int]] = {}

    def assign_group_pixels(nid: int, sid: int):
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur < M:
                if cur not in yx_cache:
                    yx_cache[cur] = leaf_to_yx(cur)
                (yy, xx) = yx_cache[cur]
                if 0 <= yy < H and 0 <= xx < W:
                    if seg_map[yy, xx] < 0:
                        seg_map[yy, xx] = sid
                continue
            n = nodes[cur]
            stack.append(n.li); stack.append(n.ri)

    for sid, gid in enumerate(groups):
        assign_group_pixels(gid, sid)

    # Fill any -1 by nearest 4-neighborhood
    q = deque([(y,x) for y in range(H) for x in range(W) if seg_map[y,x] >= 0])
    while q:
        y,x = q.popleft()
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=ny<H and 0<=nx<W and seg_map[ny,nx] < 0:
                seg_map[ny,nx] = seg_map[y,x]
                q.append((ny, nx))

    return seg_map

# ----------------------------
# Masking & evaluation over segments
# ----------------------------

def blur_image(pil: Image.Image, radius: int = 8) -> Image.Image:
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_subset_mask_segments(pil: Image.Image, seg: np.ndarray, subset: Iterable[int], baseline: Image.Image) -> np.ndarray:
    arr = np.array(pil).copy()
    base = np.array(baseline)
    keep = np.zeros(seg.max()+1, dtype=bool)
    subset = list(subset)
    if len(subset) > 0:
        keep[subset] = True
    mask = keep[seg]
    out = base.copy()
    out[mask] = arr[mask]
    return out.astype(np.uint8)

# ----------------------------
# Shapley weights and exact ≤k (BO-SHAP core)
# ----------------------------

def shapley_weight(n: int, s: int) -> float:
    from math import factorial
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def enumerate_subsets_without_i(n: int, i: int, max_size: int) -> Dict[int, List[Tuple[int, ...]]]:
    idxs = [j for j in range(n) if j != i]
    out: Dict[int, List[Tuple[int, ...]]] = {s: [] for s in range(max_size+1)}
    for s in range(max_size+1):
        out[s] = list(itertools.combinations(idxs, s))
    return out


class EvalCache:
    def __init__(self):
        self.cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    def key(self, size: int, subset: Tuple[int, ...]):
        return (size, subset)

    def get(self, k):
        return self.cache.get(k, None)

    def set(self, k, v):
        self.cache[k] = v


def eval_subset_prob(model_fn, pil, seg, baseline, subset: Tuple[int, ...], cache: EvalCache) -> float:
    key = cache.key(len(subset), subset)
    val = cache.get(key)
    if val is not None:
        return val
    x = apply_subset_mask_segments(pil, seg, subset, baseline)[None, ...]
    prob = float(model_fn(x)[0, 0])
    cache.set(key, prob)
    return prob


def exact_phi_leq_k(model_fn, pil: Image.Image, seg: np.ndarray, k: int = 2, blur_radius: int = 8):
    """정확 ≤k 계산 + 메타(고유마스크 수, residual weight mass) 반환."""
    N = int(seg.max() + 1)
    baseline = blur_image(pil, radius=blur_radius)
    cache = EvalCache()

    phi = np.zeros(N, dtype=float)

    def total_weight_upto_k(n: int, k: int) -> float:
        acc = 0.0
        for s in range(0, k+1):
            w = shapley_weight(n, s)
            acc += math.comb(n-1, s) * w
        return acc

    w_upto_k = total_weight_upto_k(N, k)

    for i in range(N):
        subsets_by_size = enumerate_subsets_without_i(N, i, k)
        contrib = 0.0
        for s, subs in subsets_by_size.items():
            w_s = shapley_weight(N, s)
            for T in subs:
                T = tuple(sorted(T))
                prob_T = eval_subset_prob(model_fn, pil, seg, baseline, T, cache)
                T_plus_i = tuple(sorted(T + (i,)))
                prob_Ti = eval_subset_prob(model_fn, pil, seg, baseline, T_plus_i, cache)
                delta = prob_Ti - prob_T
                contrib += w_s * delta
        phi[i] = contrib

    residual_mass = 1.0 - w_upto_k
    meta = {
        "n": float(N),
        "k": float(k),
        "residual_weight_mass": float(residual_mass),
        "unique_masks": int(len(cache.cache)),  # ← 몇 개의 고유 마스크가 실제 평가되었는지
    }
    return phi, meta

# ----------------------------
# Tail (|T|>k) empirical upper bound via BO-like search
# ----------------------------

def tail_bound_BO(model_fn, pil: Image.Image, seg: np.ndarray, k: int, trials: int = 16,
                  s_max_extra: int = 3, blur_radius: int = 8, seed: int = 0):
    set_seeds(seed)
    N = int(seg.max() + 1)
    baseline = blur_image(pil, radius=blur_radius)
    cache = EvalCache()

    def residual_mass_of_tail(n: int, k: int) -> float:
        acc = 0.0
        for s in range(0, k+1):
            w = shapley_weight(n, s)
            acc += math.comb(n-1, s) * w
        return 1.0 - acc

    residual_mass = residual_mass_of_tail(N, k)

    max_abs_delta = np.zeros(N, dtype=float)
    sizes = list(range(k+1, min(N, k+1 + s_max_extra) + 1))

    for i in range(N):
        best = 0.0
        others = [j for j in range(N) if j != i]
        for _ in range(trials):
            random.shuffle(others)
            for s in sizes:
                if s > len(others):
                    continue
                T = tuple(sorted(others[:s]))
                prob_T = eval_subset_prob(model_fn, pil, seg, baseline, T, cache)
                T_plus_i = tuple(sorted(T + (i,)))
                prob_Ti = eval_subset_prob(model_fn, pil, seg, baseline, T_plus_i, cache)
                d = abs(prob_Ti - prob_T)
                if d > best:
                    best = d
            if sizes:
                T = set(others[:sizes[0]])
                improved = True
                while improved and T:
                    improved = False
                    for add in [j for j in others if j not in T and j != i]:
                        rem = random.choice(tuple(T))
                        cand = sorted((T - {rem}) | {add})
                        prob_T = eval_subset_prob(model_fn, pil, seg, baseline, tuple(cand), cache)
                        prob_Ti = eval_subset_prob(model_fn, pil, seg, baseline, tuple(sorted(cand + [i])), cache)
                        d = abs(prob_Ti - prob_T)
                        if d > best:
                            best = d
                            T = set(cand)
                            improved = True
        max_abs_delta[i] = best

    bound = residual_mass * max_abs_delta
    meta = {
        "residual_weight_mass": float(residual_mass),
        "unique_masks": int(len(cache.cache)),  # tail 탐색 중 실제 평가된 고유 마스크 수
        "trials": int(trials),
        "s_max_extra": int(s_max_extra),
    }
    return bound, meta

# ----------------------------
# Visualization helpers
# ----------------------------

def overlay_heatmap(pil: Image.Image, seg: np.ndarray, values: np.ndarray, title: str, out_path: str, blur_sigma: float = 10.0):
    h, w = pil.height, pil.width
    heat = np.zeros((h, w), dtype=float)
    for sid in range(int(seg.max()+1)):
        heat[seg == sid] = values[sid]
    try:
        import cv2
        heat = cv2.GaussianBlur(heat, (0,0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    except Exception:
        pass
    vmin, vmax = float(heat.min()), float(heat.max())
    if vmax - vmin < 1e-12:
        disp = np.zeros_like(heat)
    else:
        disp = (heat - vmin) / (vmax - vmin)
    plt.figure(figsize=(6,6))
    plt.imshow(np.array(pil))
    plt.imshow(disp, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

# ----------------------------
# Metrics: Deletion/Insertion, rank corr, L1, coverage
# ----------------------------

def segment_scores_from_pixel_map(seg, heat):
    N = int(seg.max()+1)
    out = np.zeros(N, dtype=float)
    for sid in range(N):
        m = (seg == sid)
        if m.sum() > 0:
            out[sid] = heat[m].mean()
    return out


def eval_curve(model_fn, pil: Image.Image, seg: np.ndarray, scores: np.ndarray, mode: str, steps: int = 20, blur_radius: int = 8):
    assert mode in ("deletion","insertion")
    baseline = blur_image(pil, radius=blur_radius)
    order = np.argsort(-scores)
    fracs = []
    probs = []
    N = int(seg.max()+1)

    for t in range(steps+1):
        frac = t/steps
        k = int(round(frac*N))
        if mode == "deletion":
            keep = set(range(N)) - set(order[:k])
        else:  # insertion
            keep = set(order[:k])
        x = apply_subset_mask_segments(pil, seg, keep, baseline)[None, ...]
        prob = float(model_fn(x)[0,0])
        fracs.append(frac)
        probs.append(prob)
    # AUC via trapezoid
    auc = 0.0
    for i in range(len(fracs)-1):
        dx = fracs[i+1]-fracs[i]
        auc += 0.5*dx*(probs[i]+probs[i+1])
    return np.array(fracs), np.array(probs), float(auc)

# ----------------------------
# Runner (standalone)
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root (contains flicker8k/{images,captions.txt})")
    parser.add_argument("--img", type=str, required=True, help="Image filename in the images folder")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_neg", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blur", type=int, default=8, help="Gaussian blur radius for baseline")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--n_segs", type=int, default=32, help="Approx number of superpixels cut from SHAP partition tree")
    parser.add_argument("--steps", type=int, default=20, help="Steps for deletion/insertion curve")
    parser.add_argument("--max_evals", type=int, default=800, help="Original SHAP evaluation budget")
    parser.add_argument("--tail_trials", type=int, default=16)
    parser.add_argument("--tail_smax", type=int, default=3)
    args = parser.parse_args()

    set_seeds(args.seed)

    DATA_DIR = os.path.join(args.root, "flicker8k")
    CAPTION_PATH = os.path.join(DATA_DIR, "captions.txt")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")

    if not os.path.exists(CAPTION_PATH):
        raise FileNotFoundError(f"Not found: {CAPTION_PATH}")
    if not os.path.exists(os.path.join(IMAGE_DIR, args.img)):
        raise FileNotFoundError(f"Not found: {os.path.join(IMAGE_DIR, args.img)}")

    # Load captions
    caps_by_img, all_caps = load_captions(CAPTION_PATH)
    pil, caps = get_image_and_captions(IMAGE_DIR, caps_by_img, args.img)

    # Choose target & negatives (랜덤 네거티브: 같은 이미지 캡션 전부 제외)
    target = caps[0]
    rng = random.Random(args.seed)
    neg_candidates = [c for c in all_caps if c not in caps]
    if len(neg_candidates) < args.n_neg:
        raise ValueError("Not enough negative candidates in the corpus.")
    neg = rng.sample(neg_candidates, args.n_neg)

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(args.model).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(args.model)

    base_model_fn = build_model_fn(clip_model, clip_proc, device, target, neg)
    counter = CallCounter()
    model_fn = counter.wrap(base_model_fn)

    # 0) build common superpixels via SHAP partition tree
    seg = build_segments_from_partition_tree(pil, blur_kernel=(args.blur,args.blur), n_segs=args.n_segs)

    # 1) Original SHAP (pixel-level) → aggregate to segments
    t0 = time.time(); calls_before = counter.count
    masker = shap.maskers.Image(f"blur({args.blur},{args.blur})", shape=np.array(pil).shape)
    explainer = shap.Explainer(model_fn, masker)
    x = np.array(pil)[None, ...]
    sv = explainer(x, max_evals=args.max_evals)
    t_shap = time.time() - t0
    calls_shap = counter.count - calls_before

    shap_pixel = sv.values.mean(axis=-1)[0]
    phi_shap = segment_scores_from_pixel_map(seg, shap_pixel)

    print(f"[SHAP] time={t_shap:.2f}s, calls={calls_shap}")

    # 2) BO-SHAP exact ≤k
    t1 = time.time(); calls_before = counter.count
    phi_leq_k, meta_ex = exact_phi_leq_k(model_fn, pil, seg, k=args.k, blur_radius=args.blur)
    t_exact = time.time() - t1
    calls_exact = counter.count - calls_before

    print(f"[BO-EXACT ≤{args.k}] time={t_exact:.2f}s, calls={calls_exact}, unique_masks={meta_ex['unique_masks']}, residual_mass={meta_ex['residual_weight_mass']:.6f}")

    # 2b) Tail bound (BO-like search)
    t2 = time.time(); calls_before = counter.count
    tail_bound, meta_tail = tail_bound_BO(
        model_fn, pil, seg, k=args.k,
        trials=args.tail_trials, s_max_extra=args.tail_smax,
        blur_radius=args.blur, seed=args.seed
    )
    t_tail = time.time() - t2
    calls_tail = counter.count - calls_before

    print(f"[TAIL Λ_{args.k}] time={t_tail:.2f}s, calls={calls_tail}, unique_masks={meta_tail['unique_masks']}, residual_mass={meta_tail['residual_weight_mass']:.6f}, trials={meta_tail['trials']}")

    # 3) Curves & metrics (섹션별 시간/호출수)
    # SHAP curves
    t3 = time.time(); calls_before = counter.count
    fr_del_s, pr_del_s, auc_del_s = eval_curve(model_fn, pil, seg, phi_shap, mode="deletion", steps=args.steps, blur_radius=args.blur)
    fr_ins_s, pr_ins_s, auc_ins_s = eval_curve(model_fn, pil, seg, phi_shap, mode="insertion", steps=args.steps, blur_radius=args.blur)
    t_curves_shap = time.time() - t3
    calls_curves_shap = counter.count - calls_before
    print(f"[CURVES SHAP] time={t_curves_shap:.2f}s, calls={calls_curves_shap}")

    # BO curves (≤k exact)
    t4 = time.time(); calls_before = counter.count
    fr_del_b, pr_del_b, auc_del_b = eval_curve(model_fn, pil, seg, phi_leq_k, mode="deletion", steps=args.steps, blur_radius=args.blur)
    fr_ins_b, pr_ins_b, auc_ins_b = eval_curve(model_fn, pil, seg, phi_leq_k, mode="insertion", steps=args.steps, blur_radius=args.blur)
    t_curves_bo = time.time() - t4
    calls_curves_bo = counter.count - calls_before
    print(f"[CURVES BO-EXACT] time={t_curves_bo:.2f}s, calls={calls_curves_bo}")

    # Rank corr & L1
    from scipy.stats import spearmanr
    rho, _ = spearmanr(phi_shap, phi_leq_k)
    l1 = float(np.mean(np.abs(phi_shap - phi_leq_k)))

    # Bound coverage: does |diff| <= tail_bound ?
    diff = np.abs(phi_shap - phi_leq_k)
    coverage = float(np.mean(diff <= tail_bound))

    # 4) Visuals
    os.makedirs("_compare_out", exist_ok=True)
    core = f"cmp_{os.path.splitext(args.img)[0]}_k{args.k}_K{int(seg.max()+1)}"

    overlay_heatmap(pil, seg, phi_shap, title=f"Original SHAP (seg-avg) — target: {target}", out_path=os.path.join("_compare_out", core + "_shap_overlay.png"))
    overlay_heatmap(pil, seg, phi_leq_k, title=f"BO-SHAP(≤{args.k}) — exact part", out_path=os.path.join("_compare_out", core + "_bo_exact_overlay.png"))
    overlay_heatmap(pil, seg, tail_bound, title=f"Tail bound Λ_{args.k}", out_path=os.path.join("_compare_out", core + "_bo_tail_overlay.png"))

    # Deletion / Insertion plots
    plt.figure(figsize=(5,4))
    plt.plot(fr_del_s, pr_del_s, label="SHAP")
    plt.plot(fr_del_b, pr_del_b, label="BO-SHAP")
    plt.xlabel("fraction removed"); plt.ylabel("target prob")
    plt.title(f"Deletion curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join("_compare_out", core + "_deletion.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(fr_ins_s, pr_ins_s, label="SHAP")
    plt.plot(fr_ins_b, pr_ins_b, label="BO-SHAP")
    plt.xlabel("fraction inserted"); plt.ylabel("target prob")
    plt.title(f"Insertion curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join("_compare_out", core + "_insertion.png"), dpi=150)
    plt.close()

    # 5) Report (섹션별 시간/호출 breakdown 포함)
    report = {
        "device": device,
        "target": target,
        "negatives": neg,
        "n_superpixels": int(seg.max()+1),
        "k": int(args.k),
        "max_evals_SHAP": int(args.max_evals),
        "calls_total_all_sections": int(counter.count),
        "time_sec": {
            "shap_pixel": float(t_shap),
            "bo_exact_leq_k": float(t_exact),
            "bo_tail_search": float(t_tail),
            "curves_shap": float(t_curves_shap),
            "curves_bo_exact": float(t_curves_bo),
        },
        "calls": {
            "shap_pixel": int(calls_shap),
            "bo_exact_leq_k": int(calls_exact),
            "bo_tail_search": int(calls_tail),
            "curves_shap": int(calls_curves_shap),
            "curves_bo_exact": int(calls_curves_bo),
        },
        "bo_meta": {
            "exact_unique_masks": int(meta_ex["unique_masks"]),
            "tail_unique_masks": int(meta_tail["unique_masks"]),
            "residual_weight_mass": float(meta_ex["residual_weight_mass"]),  # same as meta_tail[...] by 정의
            "tail_trials": int(meta_tail["trials"]),
            "tail_s_max_extra": int(meta_tail["s_max_extra"]),
        },
        "metrics": {
            "spearman": float(rho),
            "L1": float(l1),
            "coverage_diff_le_tail": coverage,
            "deletion_auc_SHAP": float(auc_del_s),
            "deletion_auc_BO": float(auc_del_b),
            "insertion_auc_SHAP": float(auc_ins_s),
            "insertion_auc_BO": float(auc_ins_b),
        },
        "outputs": {
            "shap_overlay": os.path.join("_compare_out", core + "_shap_overlay.png"),
            "bo_exact_overlay": os.path.join("_compare_out", core + "_bo_exact_overlay.png"),
            "bo_tail_overlay": os.path.join("_compare_out", core + "_bo_tail_overlay.png"),
            "deletion_curve": os.path.join("_compare_out", core + "_deletion.png"),
            "insertion_curve": os.path.join("_compare_out", core + "_insertion.png"),
        }
    }

    print("\n[COMPARISON REPORT]")
    print(report)
    metrics_path = os.path.join("_compare_out", core + "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {metrics_path}")

if __name__ == "__main__":
    main()
