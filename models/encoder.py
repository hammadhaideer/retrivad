"""
DINOv2 Encoder Wrapper for RetriVAD.

Supports:
  - Image-level descriptor: fused CLS + mean-patch token (768-d)
  - Patch-level descriptors: 256 individual patch tokens (256 x 768)
  - Multi-layer feature extraction for layer analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIM = 768
PATCH_SIZE = 14
IMG_SIZE = 224
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 256
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 16


def preprocess(img, size=IMG_SIZE):
    """Resize and normalise a PIL image to a tensor."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class DINOv2Encoder:
    """Frozen DINOv2 ViT-B/14 encoder with CLS + patch token extraction."""

    def __init__(self, model_name="dinov2_vitb14", device="cpu"):
        self.device = device
        self.model = (
            torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
            .eval()
            .to(device)
        )
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, img_tensor):
        """
        Extract CLS token and patch tokens from a single image tensor.

        Args:
            img_tensor: (1, 3, 224, 224) preprocessed tensor

        Returns:
            cls_token:    (768,) CLS token
            patch_tokens: (256, 768) patch tokens in spatial order
        """
        x = img_tensor.to(self.device)
        out = self.model.forward_features(x)
        cls_token = out["x_norm_clstoken"][0]  # (768,)
        patch_tokens = out["x_norm_patchtokens"][0]  # (256, 768)
        return cls_token, patch_tokens

    @torch.no_grad()
    def extract_at_layer(self, img_tensor, layer_idx=11):
        """
        Extract features from a specific transformer layer.

        Args:
            img_tensor: (1, 3, 224, 224) preprocessed tensor
            layer_idx:  0-indexed layer (0..11 for ViT-B)

        Returns:
            cls_token:    (768,)
            patch_tokens: (256, 768)
        """
        x = img_tensor.to(self.device)
        features = self.model.get_intermediate_layers(
            x, n=[layer_idx], return_class_token=True
        )
        patch_tokens = features[0][0][0]  # (256, 768)
        cls_token = features[0][1][0]  # (768,)
        return cls_token, patch_tokens

    def image_descriptor(self, cls_token, patch_tokens):
        """
        Fuse CLS and mean-patch tokens into a single L2-normalised descriptor.

        Args:
            cls_token:    (768,) CLS token
            patch_tokens: (256, 768) patch tokens

        Returns:
            descriptor: (768,) L2-normalised image descriptor
        """
        desc = 0.5 * (cls_token + patch_tokens.mean(dim=0))
        desc = F.normalize(desc, dim=0)
        return desc

    def encode_image(self, img, layer_idx=None):
        """
        Full pipeline: PIL image -> L2-normalised image descriptor.

        Returns:
            img_desc:     (768,) np.float32
            patch_tokens: (256, 768) L2-normalised np.float32
        """
        x = preprocess(img).to(self.device)

        if layer_idx is not None:
            cls_token, patch_tokens = self.extract_at_layer(x, layer_idx)
        else:
            cls_token, patch_tokens = self.extract(x)

        img_desc = self.image_descriptor(cls_token, patch_tokens)
        patch_norm = F.normalize(patch_tokens, dim=-1)

        return (
            img_desc.cpu().numpy().astype(np.float32),
            patch_norm.cpu().numpy().astype(np.float32),
        )
