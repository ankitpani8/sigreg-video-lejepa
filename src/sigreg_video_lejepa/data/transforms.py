from __future__ import annotations

import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class UCF101Transform:
    """V-JEPA 2 spatial augmentation pipeline for video clips.

    Parameters match app/vjepa/transforms.py defaults exactly. No color jitter
    or Gaussian blur — V-JEPA 2 deliberately omits them so that the same spatial
    augmentation can be applied identically across all frames, preserving motion
    coherence. Adding per-frame color noise would break temporal consistency.

    Input:  (T, H, W, C) uint8 numpy array
    Output: (C, T, H, W) float32 tensor, ImageNet-normalized

    Transforms are deterministic under torch.manual_seed() / random.seed().
    """

    def __init__(
        self,
        crop_size: int = 224,
        scale: tuple[float, float] = (0.3, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        flip_p: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.flip_p = flip_p
        self.mean = list(mean)
        self.std = list(std)

    def _get_crop_params(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Sample (i, j, h, w) for RandomResizedCrop using V-JEPA 2 scale/ratio."""
        H, W = height, width
        area = H * W
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= W and 0 < h <= H:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                return i, j, h, w

        # Fallback: center crop at the largest valid aspect ratio
        in_ratio = W / H
        if in_ratio < self.ratio[0]:
            w = W
            h = int(round(W / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            h = H
            w = int(round(H * self.ratio[1]))
        else:
            h, w = H, W
        i = (H - h) // 2
        j = (W - w) // 2
        return i, j, h, w

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W, C) uint8 numpy array
        Returns:
            (C, T, H, W) float32 normalized tensor
        """
        T, H, W, C = frames.shape

        # Sample augmentation params once — applied identically to all T frames
        i, j, h, w = self._get_crop_params(H, W)  # noqa: N806 — T,H,W,C are standard spatial dims
        do_flip = random.random() < self.flip_p

        out_frames: list[torch.Tensor] = []
        for t in range(T):
            img = Image.fromarray(frames[t])                          # (H, W, C) → PIL
            img = TF.resized_crop(img, i, j, h, w, (self.crop_size, self.crop_size), TF.InterpolationMode.BILINEAR)
            if do_flip:
                img = TF.hflip(img)
            tensor = TF.to_tensor(img)                                # (C, H, W) float32 [0,1]
            tensor = TF.normalize(tensor, self.mean, self.std)
            out_frames.append(tensor)

        return torch.stack(out_frames, dim=1)                         # (C, T, H, W)
