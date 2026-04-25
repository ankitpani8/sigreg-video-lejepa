from __future__ import annotations

import torch
import torch.nn as nn


class SIGRegProjector(nn.Module):
    """MLP projector head that maps encoder tokens to the SIGReg regularization space.

    Separate from the predictor head — shares only the context encoder backbone.
    Architecture mirrors LeJEPA reference: D → 2048 → 2048 → proj_dim,
    with BatchNorm1d + GELU after each hidden layer, no normalization on output.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        hidden_dim: int = 2048,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — encoder tokens for context (unmasked) positions
        Returns:
            (B, N, proj_dim)
        """
        B, N, D = x.shape
        flat = x.reshape(B * N, D)
        projected = self.mlp(flat)          # BatchNorm1d works on (B*N, D)
        return projected.reshape(B, N, -1)
