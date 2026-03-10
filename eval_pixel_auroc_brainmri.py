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
import csv

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM      = 768
IMG_SIZE = 224
N_PATCHES = 16

VISA_CATEGORIES = [
    'candle','capsules','cashew','chewinggum',
    'fryum','macaroni1','macaroni2',
    'pcb1','pcb2','pcb3','pcb4','pipe_fryum'
]

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32)/255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def get_patch_feats(model, path):
    x = preprocess(path)
    with torch.no_grad():
        out = model.forward_features(x)
        patches = out["x_norm_patchtokens"][0]
    pf = patches.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(pf)
    return pf

def get_img_feat(model, path):
    x = preprocess(path)
    with torch.no_grad():
        out = model.forward_features(x)
        cls  = out["x_norm_clstoken"][0]
        pool = out["x_norm_patchtokens"][0].mean(0)
        feat = (cls + pool) / 2.0
    v = feat.cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def load_split(data_root, category, max_ref=69):
    root     = Path(data_root) / category
    csv_path = root / 'image_anno.csv'
    img_dir  = root / 'Data' / 'Images'

    train_normals, test_normals, test_anomalies, test_masks = [], [], [], []

    if not csv_path.exists():
        train_normals  = sorted((img_dir / 'Normal').glob('*'))[:max_ref]
        test_normals   = sorted((img_dir / 'Normal').glob('*'))[max_ref:max_ref+50]
        test_anomalies = sorted((img_dir / 'Anomaly').glob('*'))
        return train_normals, test_normals, test_anomalies, []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split         = row.get('split','').strip().lower()
            label         = row.get('label','').strip().lower()
            img_path_rel  = row.get('image','').strip()
            mask_path_rel = row.get('mask','').strip()
            img_path  = root / img_path_rel  if img_path_rel  else None
            mask_path = root / mask_path_rel if mask_path_rel else None
            if img_path and img_path.exists():
                if split == 'train':
                    train_normals.append(img_path)
                elif split == 'test':
                    if label == 'normal':
                        test_normals.append(img_path)
                    else:
                        test_anomalies.append(img_path)
                        test_masks.append(mask_path if (mask_path and mask_path.exists()) else None)

    return train_normals[:max_ref], test_normals, test_anomalies, test_masks

def run_category(model, data_root, category, max_ref=69):
    train_normals, test_normals, test_anomalies, test_masks = load_split(data_root, category, max_ref)
    print(f"  [{category}] train={len(train_normals)} test_n={len(test_normals)} test_a={len(test_anomalies)}")
    if len(train_normals) == 0 or len(test_anomalies) == 0:
        print(f"  [{category}] insufficient data, skipping")
        return None, None

    bank_feats = []
    for p in tqdm(train_normals, desc=f"  [{category}] bank", leave=False):
        try: bank_feats.append(get_patch_feats(model, p))
        except: pass
    if not bank_feats: return None, None

    bank = np.concatenate(bank_feats, axis=0).astype(np.float32)
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatL2(DIM)
    index.add(bank)

    img_feats = []
    for p in train_normals:
        try: img_feats.append(get_img_feat(model, p))
        except: pass
    img_bank = np.stack(img_feats).astype(np.float32)
    faiss.normalize_L2(img_bank)
    img_idx = faiss.IndexFlatL2(DIM)
    img_idx.add(img_bank)

    img_scores, img_labels = [], []
    for p in test_normals:
        try:
            v = get_img_feat(model, p).reshape(1,-1); faiss.normalize_L2(v)
            d,_ = img_idx.search(v,1)
            img_scores.append(float(d[0][0])); img_labels.append(0)
        except: pass
    for p in test_anomalies:
        try:
            v = get_img_feat(model, p).reshape(1,-1); faiss.normalize_L2(v)
            d,_ = img_idx.search(v,1)
            img_scores.append(float(d[0][0])); img_labels.append(1)
        except: pass

    img_auroc = roc_auc_score(img_labels, img_scores)*100 if len(set(img_labels))==2 else None

    pixel_auroc = None
    if any(m is not None for m in test_masks):
        all_s, all_l = [], []
        for p, m in tqdm(zip(test_anomalies, test_masks), total=len(test_anomalies),
                         desc=f"  [{category}] pixel", leave=False):
            if m is None: continue
            try:
                pf = get_patch_feats(model, p)
                D,_ = index.search(pf,1)
                ps = D[:,0].reshape(N_PATCHES,N_PATCHES)
                t  = torch.from_numpy(ps).unsqueeze(0).unsqueeze(0)
                up = F.interpolate(t, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)
                sm = up.squeeze().numpy().flatten()
                mk = Image.open(m).convert('L').resize((IMG_SIZE,IMG_SIZE))
                ma = (np.array(mk)>127).astype(int).flatten()
                if ma.sum() > 0:
                    all_s.append(sm); all_l.append(ma)
            except: pass
        if all_s:
            fs = np.concatenate(all_s); fl = np.concatenate(all_l)
            if fl.sum() > 0:
                pixel_auroc = roc_auc_score(fl, fs)*100

    print(f"  [{category}] Img={img_auroc:.2f}%  Pixel={pixel_auroc:.2f}%" if (img_auroc and pixel_auroc) else f"  [{category}] Img={img_auroc}  Pixel={pixel_auroc}")
    return img_auroc, pixel_auroc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--max_ref', type=int, default=69)
    parser.add_argument('--categories', nargs='+', default=VISA_CATEGORIES)
    args = parser.parse_args()

    print("Loading DINOv2...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()

    img_res, pix_res = {}, {}
    for cat in args.categories:
        ia, pa = run_category(model, args.data_root, cat, args.max_ref)
        if ia is not None: img_res[cat] = round(ia,2)
        if pa is not None: pix_res[cat] = round(pa,2)

    print("\n=== VisA Results ===")
    if img_res:
        m = np.mean(list(img_res.values())); img_res['mean'] = round(m,2)
        print(f"Image-AUROC mean: {m:.2f}%")
    if pix_res:
        m = np.mean(list(pix_res.values())); pix_res['mean'] = round(m,2)
        print(f"Pixel-AUROC mean: {m:.2f}%")

    out = Path('results') / 'visa_pixel_auroc.json'
    out.parent.mkdir(exist_ok=True)
    with open(out,'w') as f: json.dump({'image': img_res, 'pixel': pix_res}, f, indent=2)
    print(f"Saved to {out}")

if __name__ == '__main__':
    main()