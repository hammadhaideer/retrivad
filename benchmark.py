import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from models.retrivad import RetriVAD
from utils.metrics import image_auroc, pixel_auroc, upsample_map, combine_masks

os.makedirs(ROOT / "results", exist_ok=True)

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
    "pushpins", "breakfast_box", "juice_bottle", "screw_bag", "splicing_connectors",
]

UNIVAD_TABLE1 = {
    "MVTec-AD":  97.8,
    "VisA":      93.5,
    "MVTecLOCO": 71.0,
    "BrainMRI":  80.2,
    "LiverCT":   70.0,
    "RESC":      85.5,
    "HIS":       72.6,
    "ChestXray": 72.2,
    "OCT17":     82.1,
}


def image_files(folder):
    if not Path(folder).exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)


def run_category(category, train_dir, test_normal_dir, test_anomaly_dirs,
                 max_ref=69, k=1, device="cpu"):
    normal_paths = image_files(train_dir)
    if not normal_paths:
        print(f"  Skipping {train_dir}: no training images found")
        return None

    model = RetriVAD(k=k, device=device)
    model.build_memory_bank(normal_paths, max_ref=max_ref)

    scores, labels = [], []
    t0 = time.time()

    for p in image_files(test_normal_dir):
        scores.append(model.predict(p)); labels.append(0)

    for d in test_anomaly_dirs:
        for p in image_files(d):
            scores.append(model.predict(p)); labels.append(1)

    elapsed = time.time() - t0

    if len(np.unique(labels)) < 2:
        print(f"  Skipping {category}: requires both normal and anomaly samples")
        return None

    auc = roc_auc_score(labels, scores)
    r   = {
        "category":    category,
        "image_auroc": round(auc * 100, 2),
        "n_normal":    labels.count(0),
        "n_anomaly":   labels.count(1),
        "latency_s":   round(elapsed / max(len(scores), 1), 3),
    }
    print(f"  {category:30s}  img-AUC={r['image_auroc']:.1f}%"
          f"  ({r['n_normal']}N+{r['n_anomaly']}A)")
    return r


def run_mvtec(data_root, max_ref=69, k=1, device="cpu"):
    print("\n=== MVTec-AD ===")
    root    = Path(data_root)
    results = []
    for cat in MVTEC_CATEGORIES:
        defect_dirs = [
            root / cat / "test" / d
            for d in os.listdir(root / cat / "test")
            if (root / cat / "test" / d).is_dir() and d != "good"
        ]
        r = run_category(
            cat,
            root / cat / "train" / "good",
            root / cat / "test"  / "good",
            defect_dirs,
            max_ref, k, device,
        )
        if r:
            results.append(r)
    return results


def run_visa(data_root, max_ref=69, k=1, device="cpu"):
    print("\n=== VisA ===")
    root    = Path(data_root)
    results = []
    for cat in VISA_CATEGORIES:
        normal_dir  = root / cat / "Data" / "Images" / "Normal"
        anomaly_dir = root / cat / "Data" / "Images" / "Anomaly"
        if not normal_dir.exists():
            normal_dir  = root / cat / "train" / "good"
            anomaly_dir = root / cat / "test"  / "bad"

        normal_paths = image_files(normal_dir)
        if not normal_paths:
            print(f"  Skipping {cat}: no normal images found")
            continue

        split       = max(1, int(len(normal_paths) * 0.8))
        train_paths = normal_paths[:split]
        test_normal = normal_paths[split:]

        model = RetriVAD(k=k, device=device)
        model.build_memory_bank(train_paths, max_ref=max_ref)

        scores, labels = [], []
        for p in test_normal:
            scores.append(model.predict(p)); labels.append(0)
        for p in image_files(anomaly_dir):
            scores.append(model.predict(p)); labels.append(1)

        if len(np.unique(labels)) < 2:
            continue

        auc = roc_auc_score(labels, scores)
        r   = {
            "category":    cat,
            "image_auroc": round(auc * 100, 2),
            "n_normal":    labels.count(0),
            "n_anomaly":   labels.count(1),
        }
        print(f"  {cat:30s}  img-AUC={r['image_auroc']:.1f}%")
        results.append(r)
    return results


def run_loco(data_root, max_ref=69, k=1, device="cpu"):
    print("\n=== MVTec LOCO ===")
    root    = Path(data_root)
    results = []
    for cat in LOCO_CATEGORIES:
        r = run_category(
            cat,
            root / cat / "train" / "good",
            root / cat / "test"  / "good",
            [
                root / cat / "test" / "logical_anomalies",
                root / cat / "test" / "structural_anomalies",
            ],
            max_ref, k, device,
        )
        if r:
            results.append(r)
    return results


def print_comparison(all_results):
    print("\n" + "=" * 75)
    print("RetriVAD vs UniVAD (CVPR 2025) — Table 1 Comparison")
    print("=" * 75)
    print(f"{'Dataset':<18} {'UniVAD':>8} {'RetriVAD':>10} {'Gap':>7}  Notes")
    print("-" * 75)
    for ds, results in all_results.items():
        if not results:
            continue
        mean_auc = np.mean([r["image_auroc"] for r in results])
        univad   = UNIVAD_TABLE1.get(ds)
        gap_str  = f"{mean_auc - univad:+.1f}%" if univad else "  n/a"
        uni_str  = f"{univad:.1f}" if univad else " n/a"
        flag     = "✓" if (univad and mean_auc >= univad) else "✗ gap"
        print(f"  {ds:<16} {uni_str:>8} {mean_auc:>9.1f}% {gap_str:>7}  ")
    print("=" * 75)


def save_results(name, results):
    if not results:
        return
    mean_auc = np.mean([r["image_auroc"] for r in results])
    payload  = {
        "dataset":          name,
        "mean_image_auroc": round(mean_auc, 2),
        "categories":       results,
    }
    fname = ROOT / "results" / f"{name.lower().replace(' ','_').replace('-','_')}_retrivad.json"
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {fname}")


def save_csv(all_results):
    rows = ["Dataset,Category,ImageAUROC,PixelAUROC,N_Normal,N_Anomaly"]
    for ds, results in all_results.items():
        for r in results:
            rows.append(
                f"{ds},{r['category']},{r.get('image_auroc','')},"
                f"{r.get('pixel_auroc','')},{r.get('n_normal','')},{r.get('n_anomaly','')}"
            )
    path = ROOT / "results" / "summary_retrivad.csv"
    path.write_text("\n".join(rows))
    print(f"\n  Summary CSV: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="loco",
                   choices=["mvtec", "visa", "loco", "all"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--max_ref",   type=int, default=69)
    p.add_argument("--k",         type=int, default=1)
    p.add_argument("--device",    default="cpu")
    return p.parse_args()


def main():
    args        = parse_args()
    root        = Path(args.data_root)
    all_results = {}

    if args.dataset in ("mvtec", "all"):
        d = root if args.dataset == "mvtec" else root / "mvtec"
        r = run_mvtec(d, args.max_ref, args.k, args.device)
        all_results["MVTec-AD"] = r
        save_results("MVTec-AD", r)

    if args.dataset in ("visa", "all"):
        d = root if args.dataset == "visa" else root / "visa"
        r = run_visa(d, args.max_ref, args.k, args.device)
        all_results["VisA"] = r
        save_results("VisA", r)

    if args.dataset in ("loco", "all"):
        d = root if args.dataset == "loco" else root / "mvtec_loco"
        r = run_loco(d, args.max_ref, args.k, args.device)
        all_results["MVTecLOCO"] = r
        save_results("MVTecLOCO", r)

    print_comparison(all_results)
    save_csv(all_results)


if __name__ == "__main__":
    main()
