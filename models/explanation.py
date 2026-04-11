"""
Retrieval Explanation Module (REM) for RetriVAD.

For each anomalous query, retrieves the K most similar normal references
and computes a patch-level difference map as visual explanation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.encoder import DINOv2Encoder, preprocess, GRID_SIZE
from models.scoring import fast_patch_localisation, upsample_map


def retrieve_nearest_normals(img_desc, image_index, normal_paths, K=3):
    """
    Retrieve K most similar normal reference images.

    Args:
        img_desc: (768,) query image descriptor
        image_index: FAISS index of normal image descriptors
        normal_paths: list of paths to normal reference images (same order as index)
        K: number of neighbours to retrieve

    Returns:
        nearest_paths: list of K file paths
        distances: list of K distances
    """
    q = img_desc.reshape(1, -1).astype(np.float32)
    distances, indices = image_index.search(q, K)
    nearest_paths = [normal_paths[i] for i in indices[0]]
    return nearest_paths, distances[0].tolist()


def patch_difference_map(query_patches, ref_patches):
    """
    Compute per-patch L2 distance between query and a single reference.

    Args:
        query_patches: (256, 768) query patch tokens
        ref_patches:   (256, 768) reference patch tokens

    Returns:
        diff_map: (16, 16) per-patch L2 distance
    """
    diffs = np.linalg.norm(query_patches - ref_patches, axis=1)  # (256,)
    return diffs.reshape(GRID_SIZE, GRID_SIZE)


def get_explanation(query_img, encoder, memory_bank, normal_paths, K=3):
    """
    Full explanation pipeline.

    Args:
        query_img: PIL Image (query / potentially anomalous)
        encoder: DINOv2Encoder
        memory_bank: MemoryBank
        normal_paths: list of paths used to build the bank (in order)
        K: number of nearest normals to retrieve

    Returns:
        dict with keys:
            nearest_paths: list of K nearest normal image paths
            distances: list of K distances
            anomaly_map: (16, 16) anomaly score map
            anomaly_map_up: (224, 224) upsampled anomaly map
    """
    img_desc, query_patches = encoder.encode_image(query_img)

    nearest_paths, distances = retrieve_nearest_normals(
        img_desc, memory_bank.image_index, normal_paths, K=K
    )

    anomaly_map = fast_patch_localisation(
        query_patches, memory_bank.patch_index, k=1
    )
    anomaly_map_up = upsample_map(anomaly_map, 224)

    return {
        "nearest_paths": nearest_paths,
        "distances": distances,
        "anomaly_map": anomaly_map,
        "anomaly_map_up": anomaly_map_up,
    }


def visualise_explanation(query_path, explanation, save_path, gt_mask_path=None):
    """
    Create a visualisation figure showing query, nearest normals, and anomaly map.

    Layout:
      Row 1: Query | Normal #1 | Normal #2 | Normal #3
      Row 2: Anomaly heatmap | (optional GT mask) | overlay

    Args:
        query_path: path to query image
        explanation: dict from get_explanation()
        save_path: output file path
        gt_mask_path: optional ground-truth mask path
    """
    nearest = explanation["nearest_paths"]
    amap = explanation["anomaly_map_up"]
    K = len(nearest)

    ncols = max(K + 1, 3)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    # Row 1: query + nearest normals
    query_img = Image.open(query_path).convert("RGB").resize((224, 224))
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query", fontsize=11, fontweight="bold")

    for i, np_ in enumerate(nearest):
        ref = Image.open(np_).convert("RGB").resize((224, 224))
        axes[0, i + 1].imshow(ref)
        axes[0, i + 1].set_title(f"Normal #{i+1}\nd={explanation['distances'][i]:.4f}",
                                  fontsize=9)

    # Row 2: anomaly heatmap + overlay
    axes[1, 0].imshow(amap, cmap="hot", interpolation="bilinear")
    axes[1, 0].set_title("Anomaly map", fontsize=11)

    # Overlay
    overlay = np.array(query_img).astype(np.float32) / 255.0
    heatmap = plt.cm.hot(amap / (amap.max() + 1e-8))[:, :, :3]
    blended = 0.6 * overlay + 0.4 * heatmap
    axes[1, 1].imshow(np.clip(blended, 0, 1))
    axes[1, 1].set_title("Overlay", fontsize=11)

    if gt_mask_path is not None:
        gt = np.array(Image.open(gt_mask_path).convert("L").resize((224, 224)))
        axes[1, 2].imshow(gt, cmap="gray")
        axes[1, 2].set_title("Ground truth", fontsize=11)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
