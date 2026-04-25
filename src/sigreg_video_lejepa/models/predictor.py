from __future__ import annotations

import torch
import torch.nn as nn


class VideoJEPAPredictor(nn.Module):
    """Shallow Transformer that predicts target tokens at masked positions.

    Takes context tokens (visible positions) and mask tokens (query positions),
    concatenates them, runs through Transformer layers, and returns predictions
    for the mask positions only.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable, V-JEPA convention)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )

    def forward(self, context_tokens: torch.Tensor, mask_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_tokens: (B, N_ctx, D)
            mask_tokens:    (B, N_mask, D)
        Returns:
            predictions:    (B, N_mask, D)
        """
        N_mask = mask_tokens.size(1)
        seq = torch.cat([context_tokens, mask_tokens], dim=1)   # (B, N_ctx+N_mask, D)
        seq = self.transformer(seq)
        return seq[:, -N_mask:, :]                              # (B, N_mask, D)
