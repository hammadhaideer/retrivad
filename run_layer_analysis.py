"""
Cross-Domain Layer Analysis for RetriVAD.

Tests which DINOv2 ViT-B/14 layers (0-11) produce the best features
for different anomaly domain types (industrial, logical, medical).

Usage:
    python run_layer_analysis.py --data_path ./data --dataset mvtec
"""

import argparse
import csv
from pathlib import Path

import faiss
import numpy as np
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from models.encoder import DINOv2Encoder, preprocess
from datasets.base import get_dataset


def run_layer_analysis(args):
    """Evaluate image-level AUROC at different ViT layers."""
    layers = [3, 5, 7, 9, 10, 11]  # 0-indexed (11 = last layer in ViT-B)
    encoder = DINOv2Encoder(device=args.device)
    dataset = get_dataset(args.dataset, args.data_path, k_shot=69, seed=args.seed)

    results = []

    for cat_name, cat_data in dataset.items():
        if len(set(cat_data["test_labels"])) < 2:
            continue
        print(f"\n  Category: {cat_name}")

        for layer_idx in layers:
            # Encode normals at this layer
            descs = []
            for p in cat_data["train_paths"][:69]:
                try:
                    img = Image.open(p).convert("RGB")
                    desc, _ = encoder.encode_image(img, layer_idx=layer_idx)
                    descs.append(desc)
                except Exception:
                    continue

            if not descs:
                continue

            index = faiss.IndexFlatL2(768)
            index.add(np.stack(descs).astype(np.float32))

            # Score test
            scores = []
            for p in cat_data["test_paths"]:
                try:
                    img = Image.open(p).convert("RGB")
                    desc, _ = encoder.encode_image(img, layer_idx=layer_idx)
                    q = desc.reshape(1, -1).astype(np.float32)
                    dist, _ = index.search(q, 1)
                    scores.append(float(dist[0, 0]))
                except Exception:
                    scores.append(0.0)

            auroc = roc_auc_score(cat_data["test_labels"], scores)
            r = {
                "dataset": args.dataset,
                "category": cat_name,
                "layer": layer_idx,
                "image_auroc": round(auroc * 100, 1),
            }
            results.append(r)
            print(f"    layer={layer_idx:2d} -> {r['image_auroc']:.1f}%")

    # Save
    save_path = Path(args.save_dir) / "layer_analysis.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RetriVAD Layer Analysis")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    run_layer_analysis(args)
