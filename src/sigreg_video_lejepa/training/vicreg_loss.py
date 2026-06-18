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
            z: (B, N, D) projector output, OR (S, D) pre-flattened.
               Under SPMD: z must be DATA-SHARDED (not all-gathered). Statistics are
               computed via reductions over the sharded batch axis, which makes XLA
               place the collective on the small (D,) and (D,D) outputs rather than
               gathering the full batch. This avoids the replicated-batch contraction
               that otherwise produces ~14GB intermediates and stalls the SPMD partitioner.
        Returns:
            scalar loss: mu_v * V + mu_c * C
        """
        if z.dim() == 3:
            B, N, D = z.shape
            z_flat = z.reshape(B * N, D)
        else:
            z_flat = z
            D = z_flat.shape[1]

        N_samples = z_flat.shape[0]  # logical global count; XLA reduces across shards

        # Global mean via sum over the (sharded) batch axis.
        mean = z_flat.sum(dim=0) / N_samples              # (D,) — cross-shard reduce on output
        z_centered = z_flat - mean                        # stays sharded

        # Variance term: per-dim std toward gamma. Manual var (XLA-safe, no aten::std).
        var = z_centered.pow(2).sum(dim=0) / N_samples    # (D,) — cross-shard reduce
        std = torch.sqrt(var + 1e-4)
        variance_loss = torch.mean(F.relu(self.gamma - std))

        # Covariance: contraction over the sharded batch axis puts the collective on
        # the small (D, D) output, not the large (S, D) input.
        cov = (z_centered.T @ z_centered) / (N_samples - 1)   # (D, D)
        cov_sq = cov.pow(2)
        covariance_loss = (cov_sq.sum() - cov_sq.diagonal().sum()) / D

        return self.mu_v * variance_loss + self.mu_c * covariance_loss
