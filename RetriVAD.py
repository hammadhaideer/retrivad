"""
RetriVAD: Training-Free Unified Visual Anomaly Detection
via Single-Encoder Retrieval with Fast Patch-Token Localisation.

Usage:
    from RetriVAD import RetriVAD

    model = RetriVAD(device="cpu")
    model.build(normal_paths, max_ref=69, use_coreset=True)
    img_score, anomaly_map = model.predict(query_image)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.encoder import DINOv2Encoder, GRID_SIZE
from models.memory_bank import MemoryBank
from models.scoring import (
    image_score,
    fast_patch_localisation,
    upsample_map,
    score_image,
)
from models.explanation import get_explanation, visualise_explanation


class RetriVAD:
    """
    Training-free unified visual anomaly detection via single-encoder retrieval.

    Architecture:
        - Encoder: Frozen DINOv2 ViT-B/14 (86.6M params)
        - Image bank: FAISS IndexFlatL2 with M<=69 image descriptors (768-d)
        - Patch bank: FAISS IndexFlatL2 with M*256 patch tokens (768-d)
        - Image scoring: L2 distance to k=1 nearest normal descriptor
        - Pixel localisation: Fast Patch-Token Localisation (single forward pass)
    """

    def __init__(self, k=1, device="cpu", model_name="dinov2_vitb14"):
        """
        Args:
            k: number of nearest neighbours for scoring (default: 1)
            device: "cpu" or "cuda"
            model_name: DINOv2 model variant (default: dinov2_vitb14)
        """
        self.k = k
        self.device = device
        self.encoder = DINOv2Encoder(model_name=model_name, device=device)
        self.memory_bank = MemoryBank()
        self.normal_paths = []  # stored for explanation module

    def build(self, normal_paths, max_ref=69, use_coreset=True):
        """
        Build memory banks from normal reference images.

        Args:
            normal_paths: list of file paths to normal training images
            max_ref: maximum number of references per category (default: 69)
            use_coreset: use greedy coreset selection (default: True)
        """
        self.normal_paths = list(normal_paths)[:max_ref]
        self.memory_bank.build(
            normal_paths, self.encoder, max_ref=max_ref, use_coreset=use_coreset
        )

    def predict(self, img_or_path):
        """
        Predict anomaly score and pixel-level anomaly map for a single image.

        Single DINOv2 forward pass yields both outputs simultaneously.

        Args:
            img_or_path: PIL Image or file path

        Returns:
            img_score: float, image-level anomaly score (higher = more anomalous)
            anomaly_map: (16, 16) np.float32, patch-level anomaly scores
        """
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path

        return score_image(img, self.encoder, self.memory_bank, k=self.k)

    def predict_image_only(self, img_or_path):
        """
        Predict image-level anomaly score only (slightly faster, skips patch bank).

        Args:
            img_or_path: PIL Image or file path

        Returns:
            score: float
        """
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path

        img_desc, _ = self.encoder.encode_image(img)
        return image_score(img_desc, self.memory_bank.image_index, k=self.k)

    def predict_batch(self, paths, return_maps=False):
        """
        Predict on a batch of image paths.

        Args:
            paths: list of file paths
            return_maps: if True, also return pixel maps

        Returns:
            scores: list of float
            maps: list of (16,16) arrays (only if return_maps=True)
        """
        scores = []
        maps = []
        for p in tqdm(paths, desc="  Scoring", leave=False):
            try:
                s, m = self.predict(p)
                scores.append(s)
                if return_maps:
                    maps.append(m)
            except Exception:
                scores.append(0.0)
                if return_maps:
                    maps.append(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))

        if return_maps:
            return scores, maps
        return scores

    def explain(self, img_or_path, K=3):
        """
        Generate retrieval-based explanation for an anomalous query.

        Args:
            img_or_path: PIL Image or file path
            K: number of nearest normals to retrieve

        Returns:
            dict with nearest_paths, distances, anomaly_map, anomaly_map_up
        """
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path

        return get_explanation(
            img, self.encoder, self.memory_bank, self.normal_paths, K=K
        )

    def save(self, save_dir):
        """Save memory bank indexes to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.memory_bank.save(
            save_dir / "image_index.faiss",
            save_dir / "patch_index.faiss",
        )

    def load(self, save_dir):
        """Load memory bank indexes from directory."""
        save_dir = Path(save_dir)
        self.memory_bank.load(
            save_dir / "image_index.faiss",
            save_dir / "patch_index.faiss",
        )
