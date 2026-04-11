"""
Memory Bank for RetriVAD.

Two FAISS indexes per category:
  - Image-level bank: M descriptors (768-d each) for image-level scoring
  - Patch-level bank: M*256 patch tokens (768-d each) for pixel-level localisation

Supports greedy coreset selection for diversity-maximising reference sampling.
"""

import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image

from models.encoder import DINOv2Encoder, preprocess, DIM, NUM_PATCHES


def greedy_coreset(descriptors, M):
    """
    Greedy maximum-coverage coreset selection (same principle as PatchCore).

    Iteratively selects the descriptor farthest from all already-selected
    descriptors, maximising coverage of the normal distribution.

    Args:
        descriptors: (N, 768) np.array of all available normal descriptors
        M: target bank size

    Returns:
        selected: list of M indices
    """
    N = len(descriptors)
    if N <= M:
        return list(range(N))

    # Start with the descriptor closest to the centroid
    centroid = descriptors.mean(axis=0, keepdims=True)
    dists_to_centroid = np.linalg.norm(descriptors - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))

    selected = [first]
    # min_dists[i] = distance from descriptor i to nearest selected
    min_dists = np.linalg.norm(descriptors - descriptors[first], axis=1)

    for _ in range(M - 1):
        # Pick the candidate with the largest min-distance to any selected
        best = int(np.argmax(min_dists))
        selected.append(best)
        # Update min distances
        new_dists = np.linalg.norm(descriptors - descriptors[best], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected] = -1  # already selected

    return selected


class MemoryBank:
    """Dual FAISS memory bank (image-level + patch-level) for a single category."""

    def __init__(self):
        self.image_index = None  # FAISS IndexFlatL2 for image descriptors
        self.patch_index = None  # FAISS IndexFlatL2 for patch tokens
        self.n_refs = 0

    def build(self, normal_paths, encoder, max_ref=69, use_coreset=True):
        """
        Build both image-level and patch-level memory banks.

        Args:
            normal_paths: list of Path objects to normal reference images
            encoder: DINOv2Encoder instance
            max_ref: maximum number of reference images (default: 69)
            use_coreset: whether to use greedy coreset selection
        """
        paths = list(normal_paths)

        # Phase 1: Encode all available normals
        all_img_descs = []
        all_patch_tokens = []

        for p in tqdm(paths, desc="  Encoding normals", leave=False):
            try:
                img = Image.open(p).convert("RGB")
                img_desc, patch_tok = encoder.encode_image(img)
                all_img_descs.append(img_desc)
                all_patch_tokens.append(patch_tok)
            except Exception as e:
                continue

        if not all_img_descs:
            raise ValueError("No features extracted from normal images.")

        all_img_descs = np.stack(all_img_descs)  # (N, 768)

        # Phase 2: Select references (coreset or first-M)
        if use_coreset and len(all_img_descs) > max_ref:
            indices = greedy_coreset(all_img_descs, max_ref)
        else:
            indices = list(range(min(len(all_img_descs), max_ref)))

        selected_img = all_img_descs[indices]
        selected_patches = np.concatenate(
            [all_patch_tokens[i] for i in indices], axis=0
        )  # (M*256, 768)

        self.n_refs = len(indices)

        # Phase 3: Build FAISS indexes
        self.image_index = faiss.IndexFlatL2(DIM)
        self.image_index.add(selected_img.astype(np.float32))

        self.patch_index = faiss.IndexFlatL2(DIM)
        self.patch_index.add(selected_patches.astype(np.float32))

    def save(self, img_path, patch_path):
        """Save both indexes to disk."""
        faiss.write_index(self.image_index, str(img_path))
        faiss.write_index(self.patch_index, str(patch_path))

    def load(self, img_path, patch_path):
        """Load both indexes from disk."""
        self.image_index = faiss.read_index(str(img_path))
        self.patch_index = faiss.read_index(str(patch_path))
        self.n_refs = self.image_index.ntotal
