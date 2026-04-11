"""
Evaluation metrics for RetriVAD.

- image_auroc: standard AUROC from image-level scores
- compute_pixel_auroc: pixel-level AUROC from 16x16 anomaly maps vs GT masks
"""

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from models.scoring import upsample_map


def image_auroc(labels, scores):
    """Compute image-level AUROC."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def compute_pixel_auroc(test_paths, test_labels, test_masks, anomaly_maps):
    """
    Compute pixel-level AUROC from patch-level anomaly maps.

    Args:
        test_paths: list of image paths
        test_labels: list of 0/1 labels
        test_masks: list of mask paths (or None for images without masks)
        anomaly_maps: list of (16, 16) np.float32 anomaly maps

    Returns:
        pixel_auroc: float, or None if no masks available
    """
    all_gt = []
    all_pred = []

    for i, (path, label, mask_path, amap) in enumerate(
        zip(test_paths, test_labels, test_masks, anomaly_maps)
    ):
        if mask_path is None or not mask_path.exists():
            if label == 0:
                # Normal image: GT mask is all zeros
                gt = np.zeros((224, 224), dtype=np.uint8)
            else:
                # Anomalous but no mask: skip
                continue
        else:
            gt = np.array(
                Image.open(mask_path).convert("L").resize((224, 224), Image.NEAREST)
            )
            gt = (gt > 127).astype(np.uint8)

        pred = upsample_map(amap, target_size=224)

        all_gt.append(gt.flatten())
        all_pred.append(pred.flatten())

    if not all_gt:
        return None

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    if all_gt.sum() == 0:
        return None

    return roc_auc_score(all_gt, all_pred)
