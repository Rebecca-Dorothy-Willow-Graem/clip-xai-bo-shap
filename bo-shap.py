import os, math, itertools, random, argparse
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# =============================
# Standalone BO‑SHAP(≤k) + Tail Bound for CLIP (image→target caption prob)
# =============================
# Usage example (Windows paths):
#   python bo_shap_clip_standalone.py \
#       --root C:\Users\32210813\xai_clip \
#       --img 1000268201_693b08cb0e.jpg \
#       --k 2 --S 4 --n_neg 4 --seed 0
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
# Image patching & masking
# ----------------------------

def grid_segments(h: int, w: int, S: int) -> np.ndarray:
    ys = np.linspace(0, h, S+1, dtype=int)
    xs = np.linspace(0, w, S+1, dtype=int)
    seg = -np.ones((h, w), dtype=np.int32)
    idx = 0
    for gy in range(S):
        for gx in range(S):
            seg[ys[gy]:ys[gy+1], xs[gx]:xs[gx+1]] = idx
            idx += 1
    return seg


def blur_image(pil: Image.Image, radius: int = 8) -> Image.Image:
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_subset_mask(pil: Image.Image, seg: np.ndarray, subset: Iterable[int], baseline: Image.Image) -> np.ndarray:
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
    x = apply_subset_mask(pil, seg, subset, baseline)[None, ...]
    prob = float(model_fn(x)[0, 0])
    cache.set(key, prob)
    return prob


def exact_phi_leq_k(model_fn, pil: Image.Image, S: int = 4, k: int = 2, blur_radius: int = 8):
    seg = grid_segments(pil.height, pil.width, S)
    N = S * S
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
    return phi, meta, seg

# ----------------------------
# Tail (|T|>k) empirical upper bound via BO-like search
# ----------------------------

def tail_bound_BO(model_fn, pil: Image.Image, seg: np.ndarray, k: int, trials: int = 64,
                  s_max_extra: int = 3, blur_radius: int = 8, seed: int = 0):
    set_seeds(seed)
    N = seg.max() + 1
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
            T = set(others[:sizes[0]]) if sizes else set()
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
    vmin, vmax = float(heat.min()), float(heat.max())
    if vmax - vmin < 1e-12:
        disp = np.zeros_like(heat)
    else:
        disp = (heat - vmin) / (vmax - vmin)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(pil))
    plt.imshow(disp, alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

# ----------------------------
# Runner (standalone)
# ----------------------------

def choose_negatives(target: str, caps_same_img: List[str], all_caps: List[str], n_neg: int, seed: int) -> List[str]:
    set_seeds(seed)
    pool = [c for c in caps_same_img if c != target]
    if len(pool) < n_neg:
        candidates = [c for c in all_caps if c != target]
        extra = random.sample(candidates, k=n_neg - len(pool)) if len(candidates) >= (n_neg - len(pool)) else candidates[:(n_neg - len(pool))]
        pool += extra
    return pool[:n_neg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root (contains flicker8k/{images,captions.txt})")
    parser.add_argument("--img", type=str, required=True, help="Image filename in the images folder")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--S", type=int, default=4, help="Grid splits per side (SxS patches)")
    parser.add_argument("--n_neg", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blur", type=int, default=8, help="Gaussian blur radius for baseline")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
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

    # 1) exact φ≤k
    phi_leq_k, meta_ex, seg = exact_phi_leq_k(model_fn, pil, S=args.S, k=args.k, blur_radius=args.blur)

    # 2) tail bound Λ_k via BO-like search
    tail_bound, meta_tail = tail_bound_BO(model_fn, pil, seg, k=args.k, trials=64, s_max_extra=3, blur_radius=args.blur, seed=args.seed)

    # 3) visuals
    os.makedirs("_bo_shap_out", exist_ok=True)
    core = f"bo_shap_{os.path.splitext(args.img)[0]}_k{args.k}_S{args.S}"

    visualize_phi_heatmap(pil, seg, phi_leq_k, title=f"BO-SHAP(≤{args.k}) — target: {target}", out_path=os.path.join("_bo_shap_out", core + "_phi_leq_k.png"))
    visualize_phi_heatmap(pil, seg, tail_bound, title=f"Tail bound Λ_{args.k} (per patch)", out_path=os.path.join("_bo_shap_out", core + "_tail_bound.png"))

    conservative = phi_leq_k + np.sign(phi_leq_k) * tail_bound
    visualize_phi_heatmap(pil, seg, conservative, title=f"Conservative φ (φ≤{args.k} ± Λ_{args.k})", out_path=os.path.join("_bo_shap_out", core + "_conservative.png"))

    # 4) print a small report
    print("\n[BO‑SHAP REPORT]")
    print({
        "device": device,
        "n_patches": int(args.S * args.S),
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
