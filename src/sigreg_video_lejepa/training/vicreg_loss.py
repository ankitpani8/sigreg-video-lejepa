"""VICReg-VC loss — variance and covariance regularizer (no invariance term).

The JEPA prediction loss serves as the invariance term architecturally, so only
the variance (V) and covariance (C) terms are needed here.

Paper: VICReg — Bardes, Ponce & LeCun, arXiv 2105.04906.

For TPU SPMD: the input z must be all-gathered across SPMD cores before this call,
identical to the SIGRegLoss pattern. Both variance and covariance are full-batch
statistics; per-shard computation would produce biased estimates. See pretrain_tpu.py
_all_gather_proj for the all-gather pattern.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Variance + Covariance regularizer applied to projected encoder embeddings.

    forward() expects (B, N, D) projected tokens (same shape as SIGRegLoss).
    Tokens are flattened to (B*N, D) before computing statistics.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        mu_v: float = 25.0,
        mu_c: float = 1.0,
    ) -> None:
        """
        Args:
            gamma: Target standard deviation for the variance term (default 1.0).
            mu_v:  Weight for the variance loss (VICReg paper default: 25).
            mu_c:  Weight for the covariance loss (VICReg paper default: 1).
        """
        super().__init__()
        self.gamma = gamma
        self.mu_v = mu_v
        self.mu_c = mu_c

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) — projector output for all context tokens.
               Must be all-gathered before this call under SPMD data parallelism.
        Returns:
            scalar loss: mu_v * V + mu_c * C
        """
        B, N, D = z.shape
        z_flat = z.reshape(B * N, D)  # (S, D)

        # Variance term: push per-dimension std toward gamma
        std = z_flat.std(dim=0)                                      # (D,)
        variance_loss = F.relu(self.gamma - std).mean()              # mean over D

        # Covariance term: penalize off-diagonal elements of the normalized cov matrix
        z_centered = z_flat - z_flat.mean(dim=0)                     # (S, D)
        cov = (z_centered.T @ z_centered) / (z_flat.size(0) - 1)    # (D, D) unbiased
        # Off-diagonal squared sum, normalized by D (VICReg paper §3.2)
        # Avoids in-place fill_ on a grad tensor by subtracting the diagonal directly.
        cov_sq = cov.pow(2)
        covariance_loss = (cov_sq.sum() - cov_sq.diagonal().sum()) / D

        return self.mu_v * variance_loss + self.mu_c * covariance_loss
