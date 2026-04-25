from __future__ import annotations

import timm
import torch
import torch.nn as nn


class VideoViTEncoder(nn.Module):
    """ViT encoder for video clips.

    Accepts (B, C, T, H, W) and returns flat patch tokens (B, T*N, D).
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        embed_dim: int = 192,
        img_size: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # return token sequence, not logits
            img_size=img_size,
        )
        patch_size: int = self.vit.patch_embed.patch_size[0]
        self.num_patches_per_frame: int = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            tokens: (B, T * num_patches_per_frame, embed_dim)
        """
        B, C, T, H, W = x.shape
        # TODO(phase-1): replace with tubelet embedding (V-JEPA standard, 3D patches across time)
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        all_tokens = self.vit.forward_features(frames)  # (B*T, N+1, D) — CLS at index 0
        patch_tokens = all_tokens[:, 1:, :]             # drop CLS → (B*T, N, D)
        return patch_tokens.reshape(B, T * self.num_patches_per_frame, self.embed_dim)
