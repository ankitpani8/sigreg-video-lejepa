from __future__ import annotations

import torch
import torch.nn as nn


class StochasticVideoJEPAPredictor(nn.Module):
    """Stochastic JEPA predictor: outputs a Gaussian (mu, log_var) per target token.

    Architecture mirrors VideoJEPAPredictor (mask token + TransformerEncoder) with an
    additional linear head projecting D → 2D. The forward pass returns a dict with:
      "sample":  reparameterized sample z = mu + exp(0.5*log_var) * eps   (B, N_tgt, D)
      "mu":      mean of the variational posterior                         (B, N_tgt, D)
      "log_var": log variance (clamped to [-10, 10] for numerical safety)  (B, N_tgt, D)

    The mask_token attribute is preserved so _forward_representations can call
    self.predictor.mask_token.expand(...) without branching on predictor type.
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
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        # Projects transformer output D → 2D; first D = mu, last D = log_var
        self.head = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(
        self, context_tokens: torch.Tensor, mask_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            context_tokens: (B, N_ctx, D)
            mask_tokens:    (B, N_mask, D)
        Returns:
            dict with keys "sample", "mu", "log_var", each (B, N_mask, D)
        """
        N_mask = mask_tokens.size(1)
        seq = torch.cat([context_tokens, mask_tokens], dim=1)
        seq = self.transformer(seq)
        out = self.head(seq[:, -N_mask:, :])              # (B, N_mask, 2D)

        D = out.size(-1) // 2
        mu = out[..., :D]
        log_var = out[..., D:].clamp(min=-10.0, max=10.0)
        sample = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return {"sample": sample, "mu": mu, "log_var": log_var}
