import os, math, itertools, random, argparse
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

import shap  # we will use shap.maskers.Image to get the partition tree

# =============================
# BO‑SHAP(≤k) + Tail Bound for CLIP
# Feature units = SHAP Image masker partition‑tree superpixels (hierarchical axis‑aligned splits)
# =============================
# Usage example (Windows paths):
#   python bo_shap_clip_partition_tree.py \
#       --root C:\Users\32210813\xai_clip \
#       --img 1000268201_693b08cb0e.jpg \
#       --k 2 --n_neg 4 --seed 0 --n_segs 32
# Outputs saved under ./_bo_shap_out/

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
# CLIP model wrapper
# ----------------------------

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
# SHAP partition‑tree → superpixel segments
# ----------------------------

def build_segments_from_partition_tree(pil: Image.Image, n_segs: int = 32) -> np.ndarray:
    """Use shap.maskers.Image to build the hierarchical partition tree, then cut into ~n_segs leaves.
    Returns an (H,W) array of integer segment ids [0..K-1].
    """
    H, W = pil.height, pil.width
    x = np.array(pil)
    masker = shap.maskers.Image("blur(16,16)", shape=x.shape)

    # The masker builds a binary merge tree in masker.clustering with shape (M-1, 4)
    # Columns: left_child, right_child, (unused), group_size. Leaves are indices [0..M-1), internals are shifted by +M.
    clustering = np.array(masker.clustering, dtype=float)
    M = H * W * x.shape[2]  # flatten by channel (as SHAP does); we will collapse channels back to HxW later

    # Helper to map a leaf index (pixel-channel) to (y, x) ignoring channel, then we will vote per pixel.
    def leaf_to_yx(leaf_idx: int) -> Tuple[int, int]:
        # SHAP ensures flatten order x (rows) * W*C + y*C + z; they assert alignment in their JIT (x is height)
        # We'll reconstruct using integer math consistent with _jit_build_partition_tree assertion.
        C = x.shape[2]
        total_z = C
        total_y = W
        # flattened index order: idx = i * (W*C) + j * C + k
        i = leaf_idx // (W * C)
        rest = leaf_idx % (W * C)
        j = rest // C
        return int(i), int(j)

    # Node representation
    class Node:
        __slots__ = ("nid", "li", "ri", "size", "is_leaf")
        def __init__(self, nid: int, li: int, ri: int, size: int):
            self.nid = nid
            self.li = li
            self.ri = ri
            self.size = int(size)
            self.is_leaf = (li < M and ri == 0)  # convention: leaves have only single pixel-channel

    # Build tree list; internal node ids are [M .. M+(M-2)] mapping to rows 0..M-2
    nodes: Dict[int, Node] = {}
    for row in range(clustering.shape[0]):
        li = int(clustering[row, 0])
        ri = int(clustering[row, 1])
        size = int(clustering[row, 3])
        nid = M + row
        nodes[nid] = Node(nid, li, ri, size)

    # Root is the last internal id
    root_id = M + (clustering.shape[0] - 1)

    # Expand nodes into superpixels by greedy splitting the largest node until we reach ~n_segs groups
    groups = [root_id]
    def children(nid: int) -> Tuple[int, int]:
        n = nodes[nid]
        return n.li, n.ri

    def is_internal(nid: int) -> bool:
        return nid >= M

    while len(groups) < n_segs:
        # pick the largest internal node to split
        # if all are leaves (rare), break
        internal_groups = [g for g in groups if is_internal(g)]
        if not internal_groups:
            break
        g = max(internal_groups, key=lambda nid: nodes[nid].size)
        groups.remove(g)
        li, ri = children(g)
        groups.extend([li, ri])

    # Now map each group to a set of pixel (y,x)
    seg_map = -np.ones((H, W), dtype=np.int32)

    # cache for leaf pixel-channel → (y,x)
    yx_cache: Dict[int, Tuple[int,int]] = {}

    def assign_group_pixels(nid: int, sid: int):
        # DFS down to leaves; for leaves: nid<M → pixel-channel
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur < M:
                # pixel-channel leaf
                if cur not in yx_cache:
                    yx_cache[cur] = leaf_to_yx(cur)
                (yy, xx) = yx_cache[cur]
                if 0 <= yy < H and 0 <= xx < W:
                    # assign pixel to this segment (multiple channels may map to same pixel → it's fine)
                    if seg_map[yy, xx] < 0:
                        seg_map[yy, xx] = sid
                continue
            # internal → push children
            n = nodes[cur]
            stack.append(n.li)
            stack.append(n.ri)

    for sid, gid in enumerate(groups):
        assign_group_pixels(gid, sid)

    # Some pixels may remain -1 if channels mapped redundantly; fill by nearest assignment (simple dilation)
    # Quick fix: assign remaining to nearest assigned neighbor by 4-neighborhood pass
    from collections import deque
    q = deque()
    for y in range(H):
        for z in range(W):
            if seg_map[y, z] >= 0:
                q.append((y, z))
    while q:
        y, z = q.popleft()
        for ny, nz in ((y-1,z),(y+1,z),(y,z-1),(y,z+1)):
            if 0 <= ny < H and 0 <= nz < W and seg_map[ny, nz] < 0:
                seg_map[ny, nz] = seg_map[y, z]
                q.append((ny, nz))

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
# Shapley weights and exact ≤k
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
    meta = {"n": float(N), "k": float(k), "residual_weight_mass": float(residual_mass)}
    return phi, meta

# ----------------------------
# Tail (|T|>k) empirical upper bound via BO-like search
# ----------------------------

def tail_bound_BO(model_fn, pil: Image.Image, seg: np.ndarray, k: int, trials: int = 64,
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
            # greedy swap improvement
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
    meta = {"residual_weight_mass": float(residual_mass)}
    return bound, meta

# ----------------------------
# Visualization
# ----------------------------

def visualize_phi_heatmap(pil: Image.Image, seg: np.ndarray, values: np.ndarray, title: str, out_path: str):
    h, w = pil.height, pil.width
    heat = np.zeros((h, w), dtype=float)
    for sid in range(seg.max()+1):
        heat[seg == sid] = values[sid]
    # Smooth to look like superpixel SHAP
    try:
        import cv2
        heat = cv2.GaussianBlur(heat, (0,0), sigmaX=10, sigmaY=10)
    except Exception:
        pass
    vmin, vmax = float(heat.min()), float(heat.max())
    if vmax - vmin < 1e-12:
        disp = np.zeros_like(heat)
    else:
        disp = (heat - vmin) / (vmax - vmin)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(pil))
    plt.imshow(disp, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

# ----------------------------
# Runner (standalone)
# ----------------------------

def choose_negatives(target: str, caps_same_img: List[str], all_caps: List[str], n_neg: int, seed: int) -> List[str]:
    """
    Random negatives from the global corpus, excluding any caption of the current image.
    The 'target' is not separately filtered because 'caps_same_img' already contains it.
    """
    rng = random.Random(seed)
    neg_candidates = [c for c in all_caps if c not in caps_same_img]  # exclude all positives for this image
    if len(neg_candidates) < n_neg:
        raise ValueError("Not enough negative candidates in the corpus.")
    return rng.sample(neg_candidates, n_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root (contains flicker8k/{images,captions.txt})")
    parser.add_argument("--img", type=str, required=True, help="Image filename in the images folder")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_neg", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blur", type=int, default=8, help="Gaussian blur radius for baseline")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--n_segs", type=int, default=32, help="Approximate number of superpixels to cut from SHAP partition tree")
    parser.add_argument("--steps", type=int, default=20, help="steps for deletion/insertion curves")

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

    # Choose target & negatives
    target = caps[0]
    neg = choose_negatives(target, caps, all_caps, n_neg=args.n_neg, seed=args.seed)

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(args.model).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(args.model)

    model_fn = build_model_fn(clip_model, clip_proc, device, target, neg)

    # 0) build SHAP partition‑tree superpixels
    seg = build_segments_from_partition_tree(pil, n_segs=args.n_segs)

    # 1) exact φ≤k over these superpixels
    phi_leq_k, meta_ex = exact_phi_leq_k(model_fn, pil, seg, k=args.k, blur_radius=args.blur)

    # 2) tail bound Λ_k via BO-like search
    tail_bound, meta_tail = tail_bound_BO(model_fn, pil, seg, k=args.k, trials=64, s_max_extra=3, blur_radius=args.blur, seed=args.seed)

    # 3) visuals
    os.makedirs("_bo_shap_out", exist_ok=True)
    core = f"bo_shap_partition_{os.path.splitext(args.img)[0]}_k{args.k}_K{int(seg.max()+1)}"

    visualize_phi_heatmap(pil, seg, phi_leq_k, title=f"BO-SHAP(≤{args.k}) — partition-tree superpixels — target: {target}", out_path=os.path.join("_bo_shap_out", core + "_phi_leq_k.png"))
    visualize_phi_heatmap(pil, seg, tail_bound, title=f"Tail bound Λ_{args.k} (per superpixel)", out_path=os.path.join("_bo_shap_out", core + "_tail_bound.png"))

    conservative = phi_leq_k + np.sign(phi_leq_k) * tail_bound
    visualize_phi_heatmap(pil, seg, conservative, title=f"Conservative φ (φ≤{args.k} ± Λ_{args.k})", out_path=os.path.join("_bo_shap_out", core + "_conservative.png"))

    # 4) print a small report
    print("\n[BO‑SHAP PARTITION REPORT]")
    print({
        "device": device,
        "n_superpixels": int(seg.max() + 1),
        "k": int(args.k),
        "residual_weight_mass": meta_ex["residual_weight_mass"],
        "target": target,
        "negatives": neg,
        "outputs": {
            "phi_leq_k": os.path.join("_bo_shap_out", core + "_phi_leq_k.png"),
            "tail_bound": os.path.join("_bo_shap_out", core + "_tail_bound.png"),
            "conservative": os.path.join("_bo_shap_out", core + "_conservative.png"),
        }
    })


if __name__ == "__main__":
    main()
