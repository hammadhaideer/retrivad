"""
Ablation experiments for RetriVAD.

Experiments:
  1. Feature configuration: CLS-only vs patch-only vs CLS+patch
  2. Shot count: M = 1, 2, 5, 10, 20, 50, 69 (coreset vs random)
  3. Nearest neighbour count: k = 1, 3, 5
  4. Encoder size: ViT-S vs ViT-B vs ViT-L
  5. Component ablation: baseline vs +FPL vs +FPL+coreset

Usage:
    python run_ablation.py --experiment shots --data_path ./data --dataset mvtec
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from PIL import Image

from models.encoder import DINOv2Encoder, preprocess
from models.memory_bank import MemoryBank, greedy_coreset
from models.scoring import image_score, fast_patch_localisation
from datasets.base import get_dataset


def ablation_shots(args):
    """Shot count ablation: coreset vs random at different M values."""
    shot_counts = [1, 2, 5, 10, 20, 50, 69]
    methods = ["random", "coreset"]

    encoder = DINOv2Encoder(device=args.device)
    dataset = get_dataset(args.dataset, args.data_path, k_shot=999, seed=args.seed)

    results = []
    for cat_name, cat_data in dataset.items():
        if len(cat_data["test_paths"]) == 0 or len(set(cat_data["test_labels"])) < 2:
            continue

        print(f"\n  Category: {cat_name}")
        for M in shot_counts:
            for method in methods:
                bank = MemoryBank()
                bank.build(
                    cat_data["train_paths"], encoder,
                    max_ref=M, use_coreset=(method == "coreset")
                )

                scores = []
                for p in cat_data["test_paths"]:
                    try:
                        img = Image.open(p).convert("RGB")
                        desc, _ = encoder.encode_image(img)
                        scores.append(image_score(desc, bank.image_index, k=1))
                    except Exception:
                        scores.append(0.0)

                auroc = roc_auc_score(cat_data["test_labels"], scores)
                r = {
                    "dataset": args.dataset,
                    "category": cat_name,
                    "M": M,
                    "method": method,
                    "image_auroc": round(auroc * 100, 1),
                }
                results.append(r)
                print(f"    M={M:3d} {method:8s} -> {r['image_auroc']:.1f}%")

    # Save
    save_path = Path(args.save_dir) / "shot_ablation.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {save_path}")


def ablation_features(args):
    """Feature configuration ablation: CLS-only, patch-only, CLS+patch."""
    import torch.nn.functional as F

    encoder = DINOv2Encoder(device=args.device)
    dataset = get_dataset(args.dataset, args.data_path, k_shot=69, seed=args.seed)

    configs = ["cls_only", "patch_only", "cls_patch"]
    results = []

    for cat_name, cat_data in dataset.items():
        if len(set(cat_data["test_labels"])) < 2:
            continue
        print(f"\n  Category: {cat_name}")

        for config in configs:
            # Encode normals
            import faiss
            descs = []
            for p in cat_data["train_paths"][:69]:
                try:
                    img = Image.open(p).convert("RGB")
                    x = preprocess(img).to(args.device)
                    cls_tok, patch_toks = encoder.extract(x)

                    if config == "cls_only":
                        d = F.normalize(cls_tok, dim=0)
                    elif config == "patch_only":
                        d = F.normalize(patch_toks.mean(dim=0), dim=0)
                    else:
                        d = F.normalize(0.5 * (cls_tok + patch_toks.mean(dim=0)), dim=0)

                    descs.append(d.cpu().numpy().astype(np.float32))
                except Exception:
                    continue

            if not descs:
                continue

            index = faiss.IndexFlatL2(768)
            index.add(np.stack(descs))

            # Score test
            scores = []
            for p in cat_data["test_paths"]:
                try:
                    img = Image.open(p).convert("RGB")
                    x = preprocess(img).to(args.device)
                    cls_tok, patch_toks = encoder.extract(x)

                    if config == "cls_only":
                        d = F.normalize(cls_tok, dim=0)
                    elif config == "patch_only":
                        d = F.normalize(patch_toks.mean(dim=0), dim=0)
                    else:
                        d = F.normalize(0.5 * (cls_tok + patch_toks.mean(dim=0)), dim=0)

                    q = d.cpu().numpy().astype(np.float32).reshape(1, -1)
                    dist, _ = index.search(q, 1)
                    scores.append(float(dist[0, 0]))
                except Exception:
                    scores.append(0.0)

            auroc = roc_auc_score(cat_data["test_labels"], scores)
            r = {
                "dataset": args.dataset,
                "category": cat_name,
                "config": config,
                "image_auroc": round(auroc * 100, 1),
            }
            results.append(r)
            print(f"    {config:12s} -> {r['image_auroc']:.1f}%")

    save_path = Path(args.save_dir) / "feature_ablation.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RetriVAD Ablation")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["shots", "features"])
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    if args.experiment == "shots":
        ablation_shots(args)
    elif args.experiment == "features":
        ablation_features(args)
