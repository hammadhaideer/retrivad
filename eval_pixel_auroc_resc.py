
import argparse
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import json

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM      = 768
IMG_SIZE = 224
N_PATCHES = 16  # 224/14 = 16

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32)/255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def get_patch_feats(model, path):
    x = preprocess(path)
    with torch.no_grad():
        out = model.forward_features(x)
        patches = out["x_norm_patchtokens"][0]  # (256, 768)
    pf = patches.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(pf)
    return pf

def get_img_feat(model, path):
    x = preprocess(path)
    with torch.no_grad():
        out = model.forward_features(x)
        cls = out["x_norm_clstoken"][0]
        pool = out["x_norm_patchtokens"][0].mean(0)
        feat = (cls + pool) / 2.0
    v = feat.cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def imgs(folder):
    exts = {'.png','.jpg','.jpeg'}
    f = Path(folder)
    if not f.exists(): return []
    return sorted(p for p in f.iterdir() if p.suffix.lower() in exts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--max_ref', type=int, default=69)
    args = parser.parse_args()

    root = Path(args.data_root)

    # RESC structure: train/good, test/good/img, test/ungood/img
    train_normals  = imgs(root / 'train' / 'good')[:args.max_ref]
    test_normals   = imgs(root / 'test'  / 'good' / 'img')
    test_anomalies = imgs(root / 'test'  / 'ungood' / 'img')

    # Check for pixel masks
    mask_dir = root / 'test' / 'ungood' / 'mask'
    has_masks = mask_dir.exists() and len(list(mask_dir.glob('*'))) > 0

    print(f"Train normals:  {len(train_normals)}")
    print(f"Test normals:   {len(test_normals)}")
    print(f"Test anomalies: {len(test_anomalies)}")
    print(f"Pixel masks:    {'YES' if has_masks else 'NO (image-level only)'}")

    print("Loading DINOv2...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()

    # Build patch memory bank
    print("Building patch memory bank...")
    bank_feats = []
    for p in tqdm(train_normals, desc="encoding"):
        try:
            pf = get_patch_feats(model, p)
            bank_feats.append(pf)
        except: pass

    bank = np.concatenate(bank_feats, axis=0).astype(np.float32)
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatL2(DIM)
    index.add(bank)

    # --- IMAGE-LEVEL AUROC ---
    img_scores, img_labels = [], []

    img_bank_feats = []
    for p in train_normals:
        try:
            img_bank_feats.append(get_img_feat(model, p))
        except: pass
    img_bank = np.stack(img_bank_feats).astype(np.float32)
    faiss.normalize_L2(img_bank)
    img_index = faiss.IndexFlatL2(DIM)
    img_index.add(img_bank)

    for p in tqdm(test_normals,   desc="img score normals"):
        try:
            v = get_img_feat(model, p).reshape(1,-1)
            faiss.normalize_L2(v)
            d, _ = img_index.search(v, 1)
            img_scores.append(float(d[0][0])); img_labels.append(0)
        except: pass
    for p in tqdm(test_anomalies, desc="img score anomalies"):
        try:
            v = get_img_feat(model, p).reshape(1,-1)
            faiss.normalize_L2(v)
            d, _ = img_index.search(v, 1)
            img_scores.append(float(d[0][0])); img_labels.append(1)
        except: pass

    img_auroc = roc_auc_score(img_labels, img_scores) * 100
    print(f"\nRESC Image-AUROC  = {img_auroc:.2f}%")

    # --- PIXEL-LEVEL AUROC (if masks available) ---
    pixel_auroc = None
    if has_masks:
        masks = sorted(mask_dir.glob('*'))
        all_scores, all_labels = [], []

        for p, m in tqdm(zip(test_anomalies, masks),
                         total=min(len(test_anomalies), len(masks)),
                         desc="pixel scoring anomalies"):
            try:
                pf = get_patch_feats(model, p)
                D, _ = index.search(pf, 1)
                patch_scores = D[:,0].reshape(N_PATCHES, N_PATCHES)
                t = torch.from_numpy(patch_scores).unsqueeze(0).unsqueeze(0)
                upsampled = F.interpolate(t, size=(IMG_SIZE,IMG_SIZE),
                                          mode='bilinear', align_corners=False)
                score_map = upsampled.squeeze().numpy().flatten()

                mask_img = Image.open(m).convert('L').resize((IMG_SIZE,IMG_SIZE))
                mask_arr = (np.array(mask_img) > 127).astype(int).flatten()

                all_scores.append(score_map)
                all_labels.append(mask_arr)
            except: pass

        if all_scores:
            flat_scores = np.concatenate(all_scores)
            flat_labels = np.concatenate(all_labels)
            pixel_auroc = roc_auc_score(flat_labels, flat_scores) * 100
            print(f"RESC Pixel-AUROC  = {pixel_auroc:.2f}%")
    else:
        print("No pixel masks found — pixel-AUROC not computed.")

    result = {
        "image_auroc": round(img_auroc, 2),
        "pixel_auroc": round(pixel_auroc, 2) if pixel_auroc else "N/A"
    }
    out = Path('results') / 'resc_pixel_auroc.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()