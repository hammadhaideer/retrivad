"""
Multi-seed evaluation for RetriVAD.

Runs evaluation 5 times with different random seeds for reference selection,
computes mean ± std for robust reporting.

Usage:
    python run_statistics.py --data_path ./data --dataset mvtec
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image

from models.encoder import DINOv2Encoder
from models.memory_bank import MemoryBank
from models.scoring import image_score
from datasets.base import get_dataset

SEEDS = [42, 123, 456, 789, 1337]


def run_multi_seed(args):
    encoder = DINOv2Encoder(device=args.device)

    all_seed_results = {s: {} for s in SEEDS}

    for seed in SEEDS:
        print(f"\n{'='*40} Seed {seed} {'='*40}")
        np.random.seed(seed)

        dataset = get_dataset(args.dataset, args.data_path, k_shot=69, seed=seed)

        for cat_name, cat_data in dataset.items():
            if len(set(cat_data["test_labels"])) < 2:
                continue

            bank = MemoryBank()
            bank.build(cat_data["train_paths"], encoder, max_ref=69, use_coreset=True)

            scores = []
            for p in cat_data["test_paths"]:
                try:
                    img = Image.open(p).convert("RGB")
                    desc, _ = encoder.encode_image(img)
                    scores.append(image_score(desc, bank.image_index, k=1))
                except Exception:
                    scores.append(0.0)

            auroc = roc_auc_score(cat_data["test_labels"], scores)
            all_seed_results[seed][cat_name] = round(auroc * 100, 1)
            print(f"  {cat_name}: {auroc*100:.1f}%")

    # Aggregate
    categories = set()
    for seed_res in all_seed_results.values():
        categories.update(seed_res.keys())

    print(f"\n{'='*60}")
    print(f"{'Category':<20} {'Mean±Std':>15} {'Seeds':>40}")
    print("-" * 60)

    rows = []
    for cat in sorted(categories):
        vals = [all_seed_results[s].get(cat) for s in SEEDS if cat in all_seed_results[s]]
        if vals:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            vals_str = ", ".join(f"{v:.1f}" for v in vals)
            print(f"  {cat:<18} {mean_v:>6.1f}±{std_v:<5.1f} [{vals_str}]")
            rows.append({
                "dataset": args.dataset,
                "category": cat,
                "mean": round(mean_v, 1),
                "std": round(std_v, 1),
                **{f"seed_{s}": all_seed_results[s].get(cat, "") for s in SEEDS}
            })

    # Overall mean
    all_means = []
    for seed in SEEDS:
        seed_vals = list(all_seed_results[seed].values())
        if seed_vals:
            all_means.append(np.mean(seed_vals))
    if all_means:
        print(f"\n  {'Overall':<18} {np.mean(all_means):>6.1f}±{np.std(all_means):<5.1f}")

    # Save
    save_path = Path(args.save_dir) / "multi_seed_results.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RetriVAD Multi-Seed")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    run_multi_seed(args)
