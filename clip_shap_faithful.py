# filename: clip_shap_faithful.py
import os, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

import shap
import matplotlib.pyplot as plt

# ===== 경로 수정됨 =====
ROOT = r"C:\Users\32210813\xai_clip"
DATA_DIR = os.path.join(ROOT, "flicker8k")

CAPTION_PATH = os.path.join(DATA_DIR, "captions.txt")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained(MODEL_NAME)

df = pd.read_csv(CAPTION_PATH, header=None, names=["image", "caption"])
caps_by_img = df.groupby("image")["caption"].apply(list).to_dict()
all_caps = df["caption"].tolist()

def get_image_and_captions(img_name):
    pil = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    return pil, caps_by_img[img_name]

def build_model_fn(target_text, negative_texts):
    with torch.no_grad():
        tok = clip_proc(text=[target_text] + negative_texts, return_tensors="pt", padding=True).to(device)
        text_feat = clip_model.get_text_features(**tok)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

    def f(img_batch):
        pil_batch = [Image.fromarray(x.astype(np.uint8)).convert("RGB") for x in img_batch]
        inp = clip_proc(images=pil_batch, return_tensors="pt")
        with torch.no_grad():
            img_feat = clip_model.get_image_features(pixel_values=inp["pixel_values"].to(device))
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            logits = img_feat @ text_feat.T
            probs  = torch.softmax(logits, dim=1)
            return probs[:, 0:1].cpu().numpy()
    return f

def explain_with_shap(img_name, target_caption_idx=0, n_neg=4, nsamples=500):
    pil, caps = get_image_and_captions(img_name)
    target = caps[target_caption_idx]

    # ===== NEW: random negatives from other images only =====
    rng = random.Random(0)  # 재현성: 필요시 seed 바꿔도 됨
    neg_candidates = [c for c in all_caps if c not in caps]  # 현재 이미지의 모든 캡션 제외
    if len(neg_candidates) < n_neg:
        raise ValueError("Not enough negative caption candidates.")
    neg = rng.sample(neg_candidates, n_neg)
    # =======================================================

    pool = [c for c in caps if c != target]
    if len(pool) < n_neg:
        pool += random.sample([c for c in all_caps if c != target], n_neg - len(pool))
    neg = pool[:n_neg]

    masker = shap.maskers.Image("blur(128,128)", np.array(pil).shape)
    model_fn = build_model_fn(target, neg)

    explainer = shap.Explainer(model_fn, masker)
    x = np.array(pil)[None, ...]
    sv = explainer(x, max_evals=nsamples)

    heat = sv.values.mean(axis=-1)[0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    plt.figure(figsize=(6,6))
    plt.imshow(np.array(pil))
    plt.imshow(heat, alpha=0.5)
    plt.title(f"SHAP for: {target}")
    plt.axis("off")

    out = f"shap_{os.path.splitext(img_name)[0]}_{target_caption_idx}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[SHAP 저장]: {out}")

if __name__ == "__main__":
    explain_with_shap("1000268201_693b08cb0e.jpg", target_caption_idx=0)
