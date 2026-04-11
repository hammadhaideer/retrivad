"""
Anomaly Scoring for RetriVAD.

Image-level: L2 distance to k=1 nearest neighbour in image bank.
Pixel-level:  Fast Patch-Token Localisation (FPL) — uses 256 patch tokens
              from the SAME single forward pass (zero extra encoder calls).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.encoder import DINOv2Encoder, preprocess, GRID_SIZE, NUM_PATCHES


def image_score(img_desc, image_index, k=1):
    """
    Compute image-level anomaly score.

    Args:
        img_desc: (768,) L2-normalised descriptor
        image_index: FAISS IndexFlatL2 with normal reference descriptors
        k: number of nearest neighbours (default: 1)

    Returns:
        score: float, mean L2 distance to k nearest normals
    """
    q = img_desc.reshape(1, -1).astype(np.float32)
    distances, _ = image_index.search(q, k)
    return float(np.mean(distances[0]))


def fast_patch_localisation(patch_tokens, patch_index, k=1):
    """
    Fast Patch-Token Localisation (FPL).

    Uses the 256 patch tokens already extracted in the single DINOv2 forward
    pass to produce a 16x16 spatial anomaly map. No additional encoder calls.

    Args:
        patch_tokens: (256, 768) L2-normalised patch tokens from encoder
        patch_index: FAISS IndexFlatL2 with reference patch tokens
        k: number of nearest neighbours (default: 1)

    Returns:
        anomaly_map: (16, 16) np.float32, patch-level anomaly scores
    """
    q = patch_tokens.astype(np.float32)
    distances, _ = patch_index.search(q, k)  # (256, k)
    scores = np.mean(distances, axis=1)  # (256,)
    anomaly_map = scores.reshape(GRID_SIZE, GRID_SIZE)
    return anomaly_map


def upsample_map(anomaly_map, target_size=224):
    """
    Bilinearly upsample a 16x16 anomaly map to target resolution.

    Args:
        anomaly_map: (16, 16) np.float32
        target_size: output spatial size (default: 224)

    Returns:
        upsampled: (target_size, target_size) np.float32
    """
    t = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(target_size, target_size),
                       mode="bilinear", align_corners=False)
    return up.squeeze().numpy()


def score_image(img, encoder, memory_bank, k=1):
    """
    Full scoring pipeline for a single image.

    Single DINOv2 forward pass yields both image-level score and pixel map.

    Args:
        img: PIL Image
        encoder: DINOv2Encoder
        memory_bank: MemoryBank with image_index and patch_index
        k: nearest neighbour count

    Returns:
        img_score: float, image-level anomaly score
        anomaly_map: (16, 16) np.float32, patch-level scores
    """
    img_desc, patch_tokens = encoder.encode_image(img)

    img_score = image_score(img_desc, memory_bank.image_index, k=k)
    anomaly_map = fast_patch_localisation(
        patch_tokens, memory_bank.patch_index, k=k
    )

    return img_score, anomaly_map
