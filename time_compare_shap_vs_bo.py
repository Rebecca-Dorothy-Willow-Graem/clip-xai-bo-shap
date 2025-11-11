# filename: time_compare_shap_vs_bo.py
import os, time, random, math, itertools, argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
import shap

# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, help="Dataset root containing flicker8k/{images,captions.txt}")
parser.add_argument("--img", required=True, help="Image filename under images/")
parser.add_argument("--n_neg", type=int, default=4, help="number of negative captions")
parser.add_argument("--n_segs", type=int, default=12, help="approx # of superpixels cut from partition tree")
parser.add_argument("--k", type=int, default=1, help="BO-SHAP: exact up to order k (no tail search here)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--blur", type=int, default=8, help="Gaussian blur radius for masked baseline")
parser.add_argument("--max_evals", type=int, default=200, help="SHAP evaluation budget")
# (옵션) 이미지 리사이즈: 0이면 원본 사용
parser.add_argument("--resize", type=int, default=0, help="optional square resize (e.g., 256); 0 = no resize")
args = parser.parse_args()

# =========================
# Determinism
# =========================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Data paths
# =========================
DATA_DIR = os.path.join(args.root, "flicker8k")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.txt")

# =========================
# Load CLIP
# =========================
MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained(MODEL_NAME)

# =========================
# Load captions
# =========================
df = pd.read_csv(CAPTION_PATH, header=None, names=["image", "caption"])
caps_by_img = df.groupby("image")["caption"].apply(list).to_dict()
all_caps = df["caption"].tolist()

# =========================
# Utils
# =========================
def get_image_and_captions(img_name: str):
    pil = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    if args.resize and args.resize > 0:
        pil = pil.resize((args.resize, args.resize), Image.BICUBIC)
    return pil, caps_by_img[img_name]

def get_random_negatives(caps_same_img, n, seed=0):
    """Random negatives from the global corpus, excluding any caption of the same image."""
    rng = random.Random(seed)
    pool = [c for c in all_caps if c not in caps_same_img]
    if len(pool) < n:
        raise ValueError("Not enough negative caption candidates.")
    return rng.sample(pool, n)

def build_model_fn(target_text, negative_texts):
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
            return probs[:, 0:1].cpu().numpy()
    return f

class CallCounter:
    """Wrap a model function that takes a batch of images (N,H,W,C), returns (N,1).
       We count calls as 'number of items processed' (i.e., N per batch)."""
    def __init__(self, f):
        self.f = f
        self.calls = 0
    def __call__(self, x):
        self.calls += len(x)
        return self.f(x)

def blur_image(pil: Image.Image, radius: int = 8) -> Image.Image:
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))

# =========================
# Partition-tree superpixels
# =========================
def build_segments_from_partition_tree(pil: Image.Image, n_segs: int = 12, blur=(16,16)) -> np.ndarray:
    """Use shap.maskers.Image to build the hierarchical partition tree; then cut ~n_segs groups."""
    H, W = pil.height, pil.width
    x = np.array(pil)
    masker = shap.maskers.Image(f"blur({blur[0]},{blur[1]})", shape=x.shape)
    clustering = np.array(masker.clustering, dtype=float)  # (M-1, 4): [left, right, distance, size]
    C = x.shape[2]
    M = H * W * C

    def leaf_to_yx(leaf_idx: int):
        i = leaf_idx // (W * C)
        rest = leaf_idx % (W * C)
        j = rest // C
        return int(i), int(j)

    class Node:
        __slots__ = ("nid", "li", "ri", "size")
        def __init__(self, nid, li, ri, size):
            self.nid = nid; self.li = li; self.ri = ri; self.size = int(size)

    nodes = {}
    for row in range(clustering.shape[0]):
        li = int(clustering[row, 0]); ri = int(clustering[row, 1]); size = int(clustering[row, 3])
        nid = M + row
        nodes[nid] = Node(nid, li, ri, size)

    root_id = M + (clustering.shape[0] - 1)

    # greedy top-down split: split largest internal until ~n_segs groups
    groups = [root_id]
    def is_internal(nid: int) -> bool: return nid >= M
    while len(groups) < n_segs:
        internal_groups = [g for g in groups if is_internal(g)]
        if not internal_groups: break
        g = max(internal_groups, key=lambda z: nodes[z].size)
        groups.remove(g)
        groups.extend([nodes[g].li, nodes[g].ri])

    seg = -np.ones((H, W), dtype=np.int32)

    from collections import deque
    yx_cache = {}

    def assign_group_pixels(nid: int, sid: int):
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur < M:
                if cur not in yx_cache:
                    yx_cache[cur] = leaf_to_yx(cur)
                y, x = yx_cache[cur]
                if 0 <= y < H and 0 <= x < W and seg[y, x] < 0:
                    seg[y, x] = sid
            else:
                n = nodes[cur]
                stack.append(n.li); stack.append(n.ri)

    for sid, gid in enumerate(groups):
        assign_group_pixels(gid, sid)

    # fill any -1 by nearest 4-neighborhood
    q = deque([(y, x) for y in range(H) for x in range(W) if seg[y, x] >= 0])
    while q:
        y, x = q.popleft()
        for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
            if 0 <= ny < H and 0 <= nx < W and seg[ny, nx] < 0:
                seg[ny, nx] = seg[y, x]
                q.append((ny, nx))
    return seg

# =========================
# Subset masking helper
# =========================
def apply_subset_mask_segments(pil: Image.Image, seg: np.ndarray, subset, baseline: Image.Image) -> np.ndarray:
    arr = np.array(pil)
    base = np.array(baseline)
    N = int(seg.max() + 1)
    keep = np.zeros(N, dtype=bool)
    if subset:
        keep[list(subset)] = True
    mask = keep[seg]
    out = base.copy()
    out[mask] = arr[mask]
    return out.astype(np.uint8)

# =========================
# exact ≤k with caching (count calls)
# =========================
class EvalCache:
    def __init__(self): self._m = {}
    def get(self, key): return self._m.get(key, None)
    def set(self, key, val): self._m[key] = val

def shapley_weight(n, s):
    from math import factorial
    return factorial(s) * factorial(n - s - 1) / factorial(n)

def exact_phi_leq_k_counted(model_fn, pil, seg, k=2, blur_radius=8):
    """
    Evaluate all coalitions |T|<=k exactly with caching.
    Returns (phi vector, counted_calls_increment).
    model_fn is a CallCounter-wrapped function (has .calls).
    """
    N = int(seg.max() + 1)
    baseline = blur_image(pil, radius=blur_radius)
    cache = EvalCache()
    phi = np.zeros(N, dtype=float)

    def eval_subset(T):
        key = (len(T), tuple(T))
        got = cache.get(key)
        if got is not None:
            return got
        x = apply_subset_mask_segments(pil, seg, T, baseline)[None, ...]
        prob = float(model_fn(x)[0, 0])  # increments model_fn.calls by len(x)=1
        cache.set(key, prob)
        return prob

    counted_before = model_fn.calls

    all_idxs = list(range(N))
    for i in range(N):
        others = [j for j in all_idxs if j != i]
        for s in range(0, k + 1):
            w_s = shapley_weight(N, s)
            for T in itertools.combinations(others, s):
                T = tuple(sorted(T))
                pT  = eval_subset(T)
                pTi = eval_subset(tuple(sorted(T + (i,))))
                phi[i] += w_s * (pTi - pT)

    counted_after = model_fn.calls
    return phi, (counted_after - counted_before)

# =========================
# Main
# =========================
def main():
    pil, caps = get_image_and_captions(args.img)
    target = caps[0]
    neg = get_random_negatives(caps, args.n_neg, seed=args.seed)
    x = np.array(pil)[None, ...]

    # Build shared superpixels (partition-tree)
    seg = build_segments_from_partition_tree(pil, n_segs=args.n_segs, blur=(args.blur, args.blur))
    N = int(seg.max() + 1)

    # 1) Standard SHAP (PartitionExplainer) timing & calls
    print("\n[1] Standard SHAP (PartitionExplainer)")
    raw_fn = build_model_fn(target, neg)
    counted_fn = CallCounter(raw_fn)
    masker = shap.maskers.Image(f"blur({args.blur},{args.blur})", shape=x[0].shape)
    t0 = time.time()
    explainer = shap.Explainer(counted_fn, masker, algorithm="partition")
    _ = explainer(x, max_evals=args.max_evals)
    t_shap = time.time() - t0
    calls_shap = counted_fn.calls
    print(f"Time: {t_shap:.2f}s, Model calls (counted): {calls_shap}")

    # 2) BO-SHAP ≤k (exact part only; NO tail search) timing & calls
    print(f"\n[2] BO-SHAP ≤{args.k} (exact part only; no tail search)")
    raw_fn2 = build_model_fn(target, neg)
    counted_fn2 = CallCounter(raw_fn2)
    t1 = time.time()
    _phi, calls_boshap = exact_phi_leq_k_counted(
        counted_fn2, pil, seg, k=args.k, blur_radius=args.blur
    )
    t_boshap = time.time() - t1
    print(f"Time: {t_boshap:.2f}s, Model calls (counted): {calls_boshap}")

    # Summary
    print("\n================ SUMMARY ================")
    print(f"Standard SHAP Runtime   : {t_shap:.2f} s")
    print(f"Standard SHAP Calls     : {calls_shap}")
    print("-----------------------------------------")
    print(f"BO-SHAP (≤{args.k}) Runtime : {t_boshap:.2f} s")
    print(f"BO-SHAP (≤{args.k}) Calls   : {calls_boshap}  (N={N})")
    print("-----------------------------------------")
    spd = (t_shap / t_boshap) if t_boshap > 0 else float("inf")
    print(f"Observed speed ratio    : {spd:.2f}×")
    print("=========================================\n")

if __name__ == "__main__":
    main()
