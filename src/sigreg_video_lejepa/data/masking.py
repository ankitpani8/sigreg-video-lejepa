from __future__ import annotations

import torch


class TubeMasker:
    """Samples a random tube mask shared across all items in a batch.

    Returns sorted ctx and tgt index tensors (1D) on the requested device.
    Both tensors are sorted so positional ordering is preserved in the encoder.

    Phase 2 TODO: upgrade to per-sample independent masks (requires torch.gather
    on a (B, N_sel) index tensor instead of simple slice indexing).
    """

    def __init__(self, mask_ratio: float = 0.75) -> None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio

    def __call__(self, num_tubelets: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_tubelets: total number of tubes N
            device:       device for the returned tensors
        Returns:
            ctx_idx: 1D int tensor, len = N_ctx (unmasked)
            tgt_idx: 1D int tensor, len = N_tgt (masked / to predict)
        """
        num_tgt = max(1, int(num_tubelets * self.mask_ratio))
        num_ctx = num_tubelets - num_tgt
        if num_ctx < 1:
            raise ValueError(
                f"mask_ratio={self.mask_ratio} leaves no context tokens for {num_tubelets} tubes"
            )
        perm = torch.randperm(num_tubelets, device=device)
        tgt_idx = perm[:num_tgt].sort().values
        ctx_idx = perm[num_tgt:].sort().values
        return ctx_idx, tgt_idx
