"""
RetriVAD Evaluation Script.

Evaluates image-level AUROC and pixel-level AUROC across all supported datasets.
Outputs per-category and per-dataset results with comparison to UniVAD (CVPR 2025).

Usage:
    python test_retrivad.py --dataset mvtec --data_path ./data/mvtec --k_shot 69
    python test_retrivad.py --dataset all --data_path ./data --k_shot 69
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image

from RetriVAD import RetriVAD
from datasets.base import get_dataset, SUPPORTED_DATASETS
from utils.metrics import compute_pixel_auroc

# UniVAD (CVPR 2025) reported numbers for comparison
UNIVAD_IMAGE_AUROC = {
    "mvtec": 97.8, "visa": 93.5, "mvtec_loco": 71.0,
    "brainmri": 80.2, "liverct": 70.0, "resc": 85.5,
    "his": 72.6, "chestxray": 72.2, "oct17": 82.1,
}


def evaluate_dataset(model, dataset_dict, compute_pixel=True):
    """
    Evaluate RetriVAD on a loaded dataset.

    Args:
        model: RetriVAD instance (encoder loaded, no memory bank yet)
        dataset_dict: {category: {train_paths, test_paths, test_labels, test_masks}}
        compute_pixel: whether to compute pixel-level AUROC

    Returns:
        list of per-category result dicts
    """
    results = []

    for cat_name, cat_data in dataset_dict.items():
        train_paths = cat_data["train_paths"]
        test_paths = cat_data["test_paths"]
        test_labels = cat_data["test_labels"]
        test_masks = cat_data["test_masks"]

        if len(train_paths) == 0:
            print(f"  Skipping {cat_name}: no training images")
            continue
        if len(test_paths) == 0 or len(set(test_labels)) < 2:
            print(f"  Skipping {cat_name}: insufficient test data")
            continue

        print(f"  {cat_name}: {len(train_paths)} refs, "
              f"{test_labels.count(0)}N+{test_labels.count(1)}A test")

        # Build memory bank
        model.build(train_paths, max_ref=args.k_shot, use_coreset=args.use_coreset)

        # Score all test images
        t0 = time.time()
        if compute_pixel and any(m is not None for m in test_masks):
            scores, maps = model.predict_batch(test_paths, return_maps=True)
        else:
            scores = model.predict_batch(test_paths, return_maps=False)
            maps = None
        elapsed = time.time() - t0

        # Image-level AUROC
        img_auroc = roc_auc_score(test_labels, scores)

        # Pixel-level AUROC (if masks available)
        pix_auroc = None
        if compute_pixel and maps is not None:
            pix_auroc = compute_pixel_auroc(test_paths, test_labels, test_masks, maps)

        r = {
            "category": cat_name,
            "image_auroc": round(img_auroc * 100, 1),
            "pixel_auroc": round(pix_auroc * 100, 1) if pix_auroc is not None else None,
            "n_train": len(train_paths),
            "n_test": len(test_paths),
            "n_normal": test_labels.count(0),
            "n_anomaly": test_labels.count(1),
            "latency_s": round(elapsed / max(len(test_paths), 1), 3),
        }

        pix_str = f"  pix={r['pixel_auroc']:.1f}%" if r["pixel_auroc"] else ""
        print(f"    -> img={r['image_auroc']:.1f}%{pix_str}"
              f"  ({r['latency_s']}s/img)")

        results.append(r)

    return results


def print_comparison(all_results):
    """Print comparison table against UniVAD."""
    print("\n" + "=" * 80)
    print("RetriVAD vs UniVAD (CVPR 2025) — Image-Level AUROC Comparison")
    print("=" * 80)
    print(f"{'Dataset':<15} {'UniVAD':>8} {'RetriVAD':>10} {'Gap':>8}  {'Hardware':<10}")
    print("-" * 80)

    retrivad_means = []

    for ds_name, results in all_results.items():
        if not results:
            continue
        mean_auc = np.mean([r["image_auroc"] for r in results])
        retrivad_means.append(mean_auc)

        univad = UNIVAD_IMAGE_AUROC.get(ds_name)
        if univad:
            gap = mean_auc - univad
            print(f"  {ds_name:<13} {univad:>7.1f}% {mean_auc:>9.1f}% {gap:>+7.1f}%  CPU")
        else:
            print(f"  {ds_name:<13}     n/a  {mean_auc:>9.1f}%      n/a  CPU")

    print("-" * 80)
    if retrivad_means:
        overall = np.mean(retrivad_means)
        univad_overall = np.mean([v for k, v in UNIVAD_IMAGE_AUROC.items()
                                   if k in all_results])
        gap = overall - univad_overall if univad_overall else 0
        print(f"  {'Mean':<13} {univad_overall:>7.1f}% {overall:>9.1f}% {gap:>+7.1f}%  CPU")
    print("=" * 80)

    # Pixel-level table
    has_pixel = any(r.get("pixel_auroc") is not None
                    for results in all_results.values() for r in results)
    if has_pixel:
        print("\nPixel-Level AUROC:")
        print(f"{'Dataset':<15} {'Category':<20} {'Pixel AUROC':>12}")
        print("-" * 50)
        for ds_name, results in all_results.items():
            for r in results:
                if r.get("pixel_auroc") is not None:
                    print(f"  {ds_name:<13} {r['category']:<20} {r['pixel_auroc']:>11.1f}%")


def save_results(all_results, save_dir):
    """Save results to JSON and CSV."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(save_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV
    rows = ["dataset,category,image_auroc,pixel_auroc,n_train,n_test,latency_s"]
    for ds_name, results in all_results.items():
        for r in results:
            pix = r.get("pixel_auroc", "")
            rows.append(f"{ds_name},{r['category']},{r['image_auroc']},"
                        f"{pix},{r['n_train']},{r['n_test']},{r['latency_s']}")

    (save_dir / "results.csv").write_text("\n".join(rows))
    print(f"\nResults saved to {save_dir}/")


# Dataset paths relative to --data_path
DATASET_SUBDIRS = {
    "mvtec": "mvtec",
    "visa": "VisA_pytorch/1cls",
    "mvtec_loco": "mvtec_loco_caption",
    "brainmri": "BrainMRI",
    "liverct": "LiverCT",
    "resc": "RESC",
    "his": "HIS",
    "chestxray": "ChestXray",
    "oct17": "OCT17",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RetriVAD Evaluation")
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=SUPPORTED_DATASETS + ["all"],
                        help="Dataset to evaluate")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root data directory")
    parser.add_argument("--k_shot", type=int, default=69,
                        help="Number of normal reference images per category")
    parser.add_argument("--k_nn", type=int, default=1,
                        help="Number of nearest neighbours for scoring")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--use_coreset", action="store_true", default=True,
                        help="Use greedy coreset selection")
    parser.add_argument("--no_coreset", action="store_true",
                        help="Disable coreset (use first-M references)")
    parser.add_argument("--no_pixel", action="store_true",
                        help="Skip pixel-level evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results")

    args = parser.parse_args()

    if args.no_coreset:
        args.use_coreset = False

    np.random.seed(args.seed)

    # Initialise model (encoder only — memory bank built per category)
    model = RetriVAD(k=args.k_nn, device=args.device)

    # Determine which datasets to evaluate
    if args.dataset == "all":
        datasets_to_run = SUPPORTED_DATASETS
    else:
        datasets_to_run = [args.dataset]

    all_results = {}

    for ds_name in datasets_to_run:
        # Resolve data path
        subdir = DATASET_SUBDIRS.get(ds_name, ds_name)
        ds_path = Path(args.data_path) / subdir
        if not ds_path.exists():
            # Try direct path
            ds_path = Path(args.data_path)
            if not ds_path.exists():
                print(f"\nSkipping {ds_name}: path not found ({ds_path})")
                continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_path})")
        print(f"{'='*60}")

        # Load dataset
        dataset_dict = get_dataset(ds_name, ds_path, k_shot=args.k_shot, seed=args.seed)

        if not dataset_dict:
            print(f"  No data found for {ds_name}")
            continue

        # Evaluate
        results = evaluate_dataset(
            model, dataset_dict, compute_pixel=not args.no_pixel
        )
        all_results[ds_name] = results

    # Print comparison and save
    print_comparison(all_results)
    save_results(all_results, args.save_dir)
