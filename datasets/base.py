"""
Unified dataset loading for RetriVAD.

Supports 9 datasets across 3 domains:
  Industrial: MVTec-AD, VisA
  Logical:    MVTec LOCO
  Medical:    BrainMRI, LiverCT, RESC, HIS, ChestXray, OCT17

Two loading modes:
  1. meta.json (UniVAD-compatible): uses the same data format as UniVAD
  2. Directory scan: directly scans train/test folder structures

All datasets return (train_normal_paths, test_paths, test_labels, test_masks)
where test_masks may be None for datasets without pixel-level GT.
"""

import json
import os
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SUPPORTED_DATASETS = [
    "mvtec", "visa", "mvtec_loco",
    "brainmri", "liverct", "resc", "his", "chestxray", "oct17",
]

# Category definitions
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

LOCO_CATEGORIES = [
    "pushpins", "breakfast_box", "juice_bottle", "screw_bag",
    "splicing_connectors",
]


def _list_images(folder):
    """List all image files in a folder, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _list_images_recursive(folder):
    """List all image files recursively, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


# ---------------------------------------------------------------------------
#  meta.json loading (UniVAD-compatible)
# ---------------------------------------------------------------------------

def _load_from_meta(root, k_shot=1, seed=42):
    """
    Load dataset from UniVAD-style meta.json.

    Returns dict: {category_name: {train_paths, test_paths, test_labels, test_masks}}
    """
    import torch
    root = Path(root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None

    meta = json.load(open(meta_path, "r"))
    train_meta = meta.get("train", {})
    test_meta = meta.get("test", {})

    torch.manual_seed(seed)
    categories = {}

    for cls_name in test_meta.keys():
        # Train: sample k_shot normals
        train_items = train_meta.get(cls_name, [])
        if len(train_items) > 0:
            indices = torch.randint(0, len(train_items), (min(k_shot, len(train_items)),))
            train_paths = [root / train_items[i]["img_path"] for i in indices]
        else:
            train_paths = []

        # Test: all items
        test_paths = []
        test_labels = []
        test_masks = []
        for item in test_meta[cls_name]:
            test_paths.append(root / item["img_path"])
            test_labels.append(item["anomaly"])
            mask_path = item.get("mask_path", "")
            if mask_path and mask_path != "":
                test_masks.append(root / mask_path)
            else:
                test_masks.append(None)

        categories[cls_name] = {
            "train_paths": train_paths,
            "test_paths": test_paths,
            "test_labels": test_labels,
            "test_masks": test_masks,
        }

    return categories


# ---------------------------------------------------------------------------
#  Directory-structure loading (fallback)
# ---------------------------------------------------------------------------

def _load_mvtec_category(root, cat):
    """Load a single MVTec-AD category from directory structure."""
    root = Path(root)
    train_paths = _list_images(root / cat / "train" / "good")

    test_paths, test_labels, test_masks = [], [], []

    # Test normals
    for p in _list_images(root / cat / "test" / "good"):
        test_paths.append(p)
        test_labels.append(0)
        test_masks.append(None)

    # Test anomalies
    test_dir = root / cat / "test"
    gt_dir = root / cat / "ground_truth"
    if test_dir.exists():
        for defect_type in sorted(os.listdir(test_dir)):
            if defect_type == "good":
                continue
            defect_dir = test_dir / defect_type
            if not defect_dir.is_dir():
                continue
            for p in _list_images(defect_dir):
                test_paths.append(p)
                test_labels.append(1)
                # Try to find corresponding mask
                mask_name = p.stem + "_mask" + ".png"
                mask_path = gt_dir / defect_type / mask_name
                if mask_path.exists():
                    test_masks.append(mask_path)
                else:
                    # Try without _mask suffix
                    mask_path2 = gt_dir / defect_type / (p.stem + ".png")
                    test_masks.append(mask_path2 if mask_path2.exists() else None)

    return {
        "train_paths": train_paths,
        "test_paths": test_paths,
        "test_labels": test_labels,
        "test_masks": test_masks,
    }


def _load_loco_category(root, cat):
    """Load a single MVTec LOCO category."""
    root = Path(root)
    train_paths = _list_images(root / cat / "train" / "good")

    test_paths, test_labels, test_masks = [], [], []

    for p in _list_images(root / cat / "test" / "good"):
        test_paths.append(p)
        test_labels.append(0)
        test_masks.append(None)

    for anomaly_type in ["logical_anomalies", "structural_anomalies"]:
        adir = root / cat / "test" / anomaly_type
        if adir.exists():
            for p in _list_images(adir):
                test_paths.append(p)
                test_labels.append(1)
                # LOCO masks
                gt_dir = root / cat / "ground_truth" / anomaly_type
                mask_path = gt_dir / (p.stem + ".png")
                test_masks.append(mask_path if mask_path.exists() else None)

    return {
        "train_paths": train_paths,
        "test_paths": test_paths,
        "test_labels": test_labels,
        "test_masks": test_masks,
    }


def _load_medical_binary(root, normal_name="normal", anomaly_names=None):
    """
    Load a medical dataset with binary normal/anomaly structure.
    Handles common layouts:
      root/train/normal/   root/test/normal/   root/test/<anomaly>/
    """
    root = Path(root)

    # Try multiple common layouts
    train_paths = []
    for try_dir in [
        root / "train" / "good",
        root / "train" / normal_name,
        root / "train" / "NORMAL",
        root / "train",
    ]:
        paths = _list_images(try_dir)
        if paths:
            train_paths = paths
            break

    test_paths, test_labels, test_masks = [], [], []

    # Test normals
    for try_dir in [
        root / "test" / "good",
        root / "test" / normal_name,
        root / "test" / "NORMAL",
    ]:
        for p in _list_images(try_dir):
            test_paths.append(p)
            test_labels.append(0)
            test_masks.append(None)

    # Test anomalies
    test_dir = root / "test"
    if test_dir.exists():
        normal_dirs = {"good", normal_name, "NORMAL", normal_name.lower(), normal_name.upper()}
        for subdir in sorted(os.listdir(test_dir)):
            sub_path = test_dir / subdir
            if not sub_path.is_dir() or subdir in normal_dirs:
                continue
            if anomaly_names and subdir not in anomaly_names:
                continue
            for p in _list_images(sub_path):
                test_paths.append(p)
                test_labels.append(1)
                # Check for masks
                gt_dir = root / "ground_truth" / subdir
                mask_path = gt_dir / (p.stem + "_mask.png")
                if mask_path.exists():
                    test_masks.append(mask_path)
                else:
                    mask_path2 = gt_dir / p.name
                    test_masks.append(mask_path2 if mask_path2.exists() else None)

    return {
        "train_paths": train_paths,
        "test_paths": test_paths,
        "test_labels": test_labels,
        "test_masks": test_masks,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_dataset(dataset_name, data_path, k_shot=69, seed=42):
    """
    Load a dataset, returning per-category data.

    Args:
        dataset_name: one of SUPPORTED_DATASETS
        data_path: root directory of the dataset
        k_shot: number of normal reference images per category
        seed: random seed for reference selection

    Returns:
        dict: {category_name: {train_paths, test_paths, test_labels, test_masks}}
    """
    root = Path(data_path)

    # Try meta.json first (UniVAD-compatible)
    meta_result = _load_from_meta(root, k_shot=k_shot, seed=seed)
    if meta_result is not None:
        return meta_result

    # Fallback: directory scanning
    if dataset_name == "mvtec":
        return {cat: _load_mvtec_category(root, cat) for cat in MVTEC_CATEGORIES
                if (root / cat).exists()}

    elif dataset_name == "visa":
        results = {}
        for cat in VISA_CATEGORIES:
            cat_dir = root / cat
            if not cat_dir.exists():
                # Try 1cls format
                cat_dir = root / "1cls" / cat
            if cat_dir.exists():
                results[cat] = _load_mvtec_category(root if (root / cat).exists() else root / "1cls", cat)
        return results

    elif dataset_name == "mvtec_loco":
        return {cat: _load_loco_category(root, cat) for cat in LOCO_CATEGORIES
                if (root / cat).exists()}

    elif dataset_name == "brainmri":
        return {"brain": _load_medical_binary(root, normal_name="normal")}

    elif dataset_name == "liverct":
        return {"liver": _load_medical_binary(root, normal_name="normal")}

    elif dataset_name == "resc":
        return {"retina": _load_medical_binary(root, normal_name="normal")}

    elif dataset_name == "his":
        return {"his": _load_medical_binary(root, normal_name="normal")}

    elif dataset_name == "chestxray":
        return {"chest": _load_medical_binary(root, normal_name="NORMAL")}

    elif dataset_name == "oct17":
        return {"oct": _load_medical_binary(root, normal_name="NORMAL")}

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: {SUPPORTED_DATASETS}")
