import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM   = 768


def preprocess(img, size=224):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
    return model


def extract(model, img_path, mode):
    img = Image.open(img_path)
    x   = preprocess(img)
    with torch.no_grad():
        out          = model.forward_features(x)
        cls_token    = out["x_norm_clstoken"][0]
        patch_tokens = out["x_norm_patchtokens"][0]
        pooled       = patch_tokens.mean(0)
    if mode == "cls":
        feat = cls_token
    elif mode == "patch":
        feat = pooled
    else:
        feat = (cls_token + pooled) / 2.0
    v  = feat.cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def image_files(folder):
    exts = {".png", ".jpg", ".jpeg"}
    f    = Path(folder)
    return sorted(p for p in f.iterdir() if p.suffix.lower() in exts) if f.exists() else []


def run_ablation(model, train_normals, test_normals, test_anomalies, mode):
    feats = []
    for p in tqdm(train_normals[:69], desc=f"  Building bank [{mode}]", leave=False):
        try:
            feats.append(extract(model, p, mode))
        except Exception:
            pass
    if not feats:
        return float("nan")
    feats = np.stack(feats).astype(np.float32)
    faiss.normalize_L2(feats)
    index = faiss.IndexFlatL2(DIM)
    index.add(feats)

    scores, labels = [], []
    for p in test_normals:
        try:
            v = extract(model, p, mode).reshape(1, -1)
            faiss.normalize_L2(v)
            d, _ = index.search(v, 1)
            scores.append(float(d[0][0]))
            labels.append(0)
        except Exception:
            pass
    for p in test_anomalies:
        try:
            v = extract(model, p, mode).reshape(1, -1)
            faiss.normalize_L2(v)
            d, _ = index.search(v, 1)
            scores.append(float(d[0][0]))
            labels.append(1)
        except Exception:
            pass

    if len(np.unique(labels)) < 2:
        return float("nan")
    return round(roc_auc_score(labels, scores) * 100, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: CLS token vs patch token vs combined."
    )
    parser.add_argument("--data_root", required=True,
                        help="Path to MVTec LOCO root directory.")
    parser.add_argument("--category",  default="pushpins",
                        help="Category name (default: pushpins).")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    root           = Path(args.data_root) / args.category
    train_normals  = image_files(root / "train" / "good")
    test_normals   = image_files(root / "test"  / "good")
    test_anomalies = []
    for sub in (root / "test").iterdir():
        if sub.is_dir() and sub.name != "good":
            test_anomalies.extend(image_files(sub))

    print(f"Category      : {args.category}")
    print(f"Train normals : {len(train_normals)}")
    print(f"Test normals  : {len(test_normals)}")
    print(f"Test anomalies: {len(test_anomalies)}")

    print("\nLoading DINOv2 ViT-B/14 ...")
    model = load_model()

    modes = {
        "cls":      "CLS token only      ",
        "patch":    "Mean patch only     ",
        "combined": "CLS + mean (default)",
    }

    print("\nAblation results:")
    print("-" * 45)
    for mode, label in modes.items():
        auc = run_ablation(model, train_normals, test_normals, test_anomalies, mode)
        print(f"  {label}  AUROC = {auc:.1f}%")
    print("-" * 45)


if __name__ == "__main__":
    main()
