from __future__ import annotations

import torch


class CausalTubeMasker:
    """Deterministic causal mask: context frames | gap frames | target frames.

    Indices are computed once at __init__ from frame counts and spatial geometry.
    Every call returns the same CPU-precomputed tensors moved to the requested device.
    This is a drop-in replacement for TubeMasker: identical __call__ return signature.

    Phase 6 design: C=6 context, G=4 gap, T=6 target on 16-frame clips with t_patch=2
    gives 3 ctx temporal positions [0,1,2], 2 gap positions [3,4] (excluded from both
    ctx_idx and tgt_idx), and 3 target positions [5,6,7]. The gap breaks the temporal-
    copy shortcut that random tube masking fails to prevent.
    """

    def __init__(
        self,
        context_frames: int,
        gap_frames: int,
        target_frames: int,
        t_patch: int,
        tubelets_per_tpos: int,
    ) -> None:
        """
        Args:
            context_frames:    Number of source video frames in the context window.
            gap_frames:        Number of source video frames in the gap (excluded).
            target_frames:     Number of source video frames in the target window.
            t_patch:           Temporal patch size; must divide each frame count evenly.
            tubelets_per_tpos: Spatial tubelets per temporal position (= H/h_patch × W/w_patch).
        """
        for name, val in [
            ("context_frames", context_frames),
            ("gap_frames", gap_frames),
            ("target_frames", target_frames),
        ]:
            if val <= 0 or val % t_patch != 0:
                raise ValueError(
                    f"{name}={val} must be positive and divisible by t_patch={t_patch}"
                )

        ctx_tpos = context_frames // t_patch
        gap_tpos = gap_frames // t_patch
        tgt_tpos = target_frames // t_patch
        total_tpos = ctx_tpos + gap_tpos + tgt_tpos

        ctx_start, ctx_end = 0, ctx_tpos
        tgt_start = ctx_tpos + gap_tpos
        tgt_end = total_tpos

        # Build flat tubelet indices: temporal_pos * tubelets_per_tpos + spatial_idx
        def _flat_indices(tpos_start: int, tpos_end: int) -> torch.Tensor:
            tpos = torch.arange(tpos_start, tpos_end, dtype=torch.int32)
            spatial = torch.arange(tubelets_per_tpos, dtype=torch.int32)
            # outer product then flatten: (n_tpos, tubelets_per_tpos) → (n_tpos*tubelets_per_tpos,)
            return (tpos.unsqueeze(1) * tubelets_per_tpos + spatial.unsqueeze(0)).reshape(-1)

        self._ctx_idx: torch.Tensor = _flat_indices(ctx_start, ctx_end)
        self._tgt_idx: torch.Tensor = _flat_indices(tgt_start, tgt_end)
        self._total_tubelets: int = total_tpos * tubelets_per_tpos

        self.context_frames = context_frames
        self.gap_frames = gap_frames
        self.target_frames = target_frames

    def __call__(
        self, num_tubelets: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ctx_idx, tgt_idx) as sorted int32 tensors on `device`.

        Args:
            num_tubelets: total tubelet count N; must match (C+G+T)/t_patch × tubelets_per_tpos.
            device:       target device for the returned tensors.
        Returns:
            ctx_idx: 1D int32 tensor, len = N_ctx (context tubelets)
            tgt_idx: 1D int32 tensor, len = N_tgt (target tubelets)
        """
        if num_tubelets != self._total_tubelets:
            raise ValueError(
                f"num_tubelets={num_tubelets} does not match the masker's expected "
                f"{self._total_tubelets} (context={self.context_frames} + "
                f"gap={self.gap_frames} + target={self.target_frames} frames)"
            )
        return self._ctx_idx.to(device), self._tgt_idx.to(device)
