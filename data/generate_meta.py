"""
Generate meta.json for datasets in UniVAD-compatible format.

Usage:
    python data/generate_meta.py --dataset mvtec --data_root ./data/mvtec
"""

import argparse
import json
import os
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(str(p.relative_to(folder.parent.parent.parent))
                  for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def generate_mvtec_meta(data_root):
    """Generate meta.json for MVTec-AD."""
    root = Path(data_root)
    categories = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and (d / "train").exists())

    meta = {"train": {}, "test": {}}

    for cat in categories:
        # Train
        train_dir = root / cat / "train" / "good"
        meta["train"][cat] = []
        for img in sorted(train_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                meta["train"][cat].append({
                    "img_path": str(img.relative_to(root)),
                    "mask_path": "",
                    "cls_name": cat,
                    "specie_name": "good",
                    "anomaly": 0,
                })

        # Test
        meta["test"][cat] = []
        test_dir = root / cat / "test"
        gt_dir = root / cat / "ground_truth"
        for defect_type in sorted(os.listdir(test_dir)):
            defect_dir = test_dir / defect_type
            if not defect_dir.is_dir():
                continue
            for img in sorted(defect_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                is_anomaly = defect_type != "good"
                mask_path = ""
                if is_anomaly:
                    mask_name = img.stem + "_mask.png"
                    mp = gt_dir / defect_type / mask_name
                    if mp.exists():
                        mask_path = str(mp.relative_to(root))

                meta["test"][cat].append({
                    "img_path": str(img.relative_to(root)),
                    "mask_path": mask_path,
                    "cls_name": cat,
                    "specie_name": defect_type,
                    "anomaly": 1 if is_anomaly else 0,
                })

    out = root / "meta.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out} ({len(categories)} categories)")


def generate_medical_meta(data_root, normal_name="normal"):
    """Generate meta.json for medical datasets with train/test/normal/anomaly structure."""
    root = Path(data_root)
    meta = {"train": {}, "test": {}}
    cat = root.name.lower()

    # Train normals
    meta["train"][cat] = []
    for try_name in [normal_name, "NORMAL", "good"]:
        train_dir = root / "train" / try_name
        if train_dir.exists():
            for img in sorted(train_dir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    meta["train"][cat].append({
                        "img_path": str(img.relative_to(root)),
                        "mask_path": "",
                        "cls_name": cat,
                        "specie_name": "normal",
                        "anomaly": 0,
                    })
            break

    # Test
    meta["test"][cat] = []
    test_dir = root / "test"
    if test_dir.exists():
        for subdir in sorted(os.listdir(test_dir)):
            sub_path = test_dir / subdir
            if not sub_path.is_dir():
                continue
            is_normal = subdir.lower() in {"normal", "good"}
            for img in sorted(sub_path.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                mask_path = ""
                gt_dir = root / "ground_truth" / subdir
                if not is_normal:
                    for ext in [".png", ".bmp"]:
                        mp = gt_dir / (img.stem + "_mask" + ext)
                        if mp.exists():
                            mask_path = str(mp.relative_to(root))
                            break
                        mp2 = gt_dir / (img.stem + ext)
                        if mp2.exists():
                            mask_path = str(mp2.relative_to(root))
                            break

                meta["test"][cat].append({
                    "img_path": str(img.relative_to(root)),
                    "mask_path": mask_path,
                    "cls_name": cat,
                    "specie_name": subdir,
                    "anomaly": 0 if is_normal else 1,
                })

    out = root / "meta.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    n_train = len(meta["train"][cat])
    n_test = len(meta["test"][cat])
    print(f"Saved {out} ({n_train} train, {n_test} test)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["mvtec", "visa", "mvtec_loco",
                                 "brainmri", "liverct", "resc",
                                 "his", "chestxray", "oct17"])
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    if args.dataset == "mvtec":
        generate_mvtec_meta(args.data_root)
    else:
        normal_name = "NORMAL" if args.dataset in ("chestxray", "oct17") else "normal"
        generate_medical_meta(args.data_root, normal_name=normal_name)
