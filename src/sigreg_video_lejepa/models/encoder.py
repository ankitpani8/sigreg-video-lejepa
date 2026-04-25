from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TubeletEmbed(nn.Module):
    """3D patch embedding via Conv3d — the V-JEPA/VideoMAE standard."""

    def __init__(
        self,
        t_patch: int = 2,
        h_patch: int = 16,
        w_patch: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(t_patch, h_patch, w_patch),
            stride=(t_patch, h_patch, w_patch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → (B, N_tubes, embed_dim)"""
        x = self.proj(x)                     # (B, D, nT, nH, nW)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class VideoViTEncoder(nn.Module):
    """Video ViT encoder using tubelet (3D patch) embedding.

    Accepts (B, C, T, H, W) and returns flat tube tokens (B, N_tubes, D).
    Optionally selects a subset of tokens by index for masked encoding.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        embed_dim: int = 192,
        img_size: int = 224,
        num_frames: int = 16,
        t_patch: int = 2,
        h_patch: int = 16,
        w_patch: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._num_tubelets = (num_frames // t_patch) * (img_size // h_patch) * (img_size // w_patch)

        self.tubelet_embed = TubeletEmbed(
            t_patch=t_patch, h_patch=h_patch, w_patch=w_patch, embed_dim=embed_dim
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self._num_tubelets, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Borrow transformer blocks and final norm from timm; skip its patch_embed and CLS logic.
        _vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.blocks = _vit.blocks
        self.norm = _vit.norm

    @property
    def num_tubelets(self) -> int:
        return self._num_tubelets

    def forward(self, x: torch.Tensor, token_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:             (B, C, T, H, W)
            token_indices: optional 1D int tensor — shared mask, selects a subset of tubes
        Returns:
            tokens: (B, N_tubes, D) or (B, len(token_indices), D)
        """
        tokens = self.tubelet_embed(x) + self.pos_embed  # (B, N, D)
        if token_indices is not None:
            tokens = tokens[:, token_indices, :]          # (B, N_sel, D)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
