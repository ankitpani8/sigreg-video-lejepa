"""SIGReg loss — Epps-Pulley characteristic function Gaussianity test via random projections.

Implementation follows docs/sigreg-spec.md exactly. Do not modify without updating the spec.
Paper: Balestriero & LeCun, arXiv 2511.08544.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """Sliced isotropic Gaussian regularizer using the Epps-Pulley test statistic.

    Precomputes integration buffers at init (device-portable via register_buffer).
    forward() expects projected encoder tokens (B, N, proj_dim) and returns a scalar.
    """

    def __init__(
        self,
        num_projections: int = 256,
        knots: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        assert knots % 2 == 1, "knots must be odd for trapezoidal endpoint correction"
        self.num_projections = num_projections

        # Spec §3.3: trapezoidal weights on [0, t_max] with symmetry trick
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt     # half-weight at t=0
        weights[-1] = dt    # half-weight at t=t_max

        phi = torch.exp(-t.square() / 2.0)  # standard Gaussian CF: exp(-t²/2)

        self.register_buffer("t", t)                    # (K,)
        self.register_buffer("phi", phi)                # (K,)
        self.register_buffer("weights_buf", weights * phi)  # (K,) precomputed product

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (B, N, proj_dim) — projector output for all context tokens
        Returns:
            scalar EP test statistic (mean over projection directions)
        """
        B, N, D = proj.shape
        z = proj.reshape(B * N, D)      # (S, D) — treat all tokens as samples
        S = z.size(0)

        with torch.no_grad():
            A = torch.randn(D, self.num_projections, device=z.device, dtype=z.dtype)
            A = A / A.norm(p=2, dim=0)  # unit-norm columns; (D, M)

        # (S, M, 1) * (K,) → (S, M, K)
        x_t = (z @ A).unsqueeze(-1) * self.t

        cos_mean = x_t.cos().mean(0)    # (M, K) — mean over S samples
        sin_mean = x_t.sin().mean(0)    # (M, K)

        # Spec §3.2: squared distance from standard Gaussian CF
        err = (cos_mean - self.phi).square() + sin_mean.square()   # (M, K)

        # Spec §3.2: integrate, scale by S (EP formula's leading N)
        statistic = (err @ self.weights_buf) * S    # (M,)
        return statistic.mean()                     # scalar


def sigreg_video_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    proj: torch.Tensor,
    sigreg: SIGRegLoss,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine prediction loss and SIGReg per the spec's total loss formula.

    L = (1 - lam) * L_pred + lam * L_SIGReg

    Returns (total, l_pred, l_sigreg) for logging.
    """
    l_pred = F.mse_loss(pred, target)
    l_sigreg = sigreg(proj)
    total = (1.0 - lam) * l_pred + lam * l_sigreg
    return total, l_pred, l_sigreg
