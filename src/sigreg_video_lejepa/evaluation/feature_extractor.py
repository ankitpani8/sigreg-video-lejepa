from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FeatureExtractor:
    """Extract encoder features from a DataLoader, with no-grad and eval mode.

    Handles two batch shapes:
      - 5D (B, C, T, H, W): single-clip train split → encode directly
      - 6D (B, 4, C, T, H, W): multi-clip eval split → encode each clip, average
    Single-clip for train, multi-clip for test is the standard SSL linear-probe protocol:
    train features prioritize throughput; test features average over temporal jitter.
    """

    def __init__(self, encoder: nn.Module, device: torch.device) -> None:
        self.encoder = encoder.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder over all batches and return (N, D) features and (N,) labels.

        Args:
            loader: DataLoader yielding (clips, labels). clips is either
                    (B, C, T, H, W) for single-clip or (B, n_clips, C, T, H, W) for multi-clip.
        Returns:
            Tuple of feature tensor (N, D) and label tensor (N,), both on CPU.
        """
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for clips, labels in loader:
            clips = clips.to(self.device)

            if clips.ndim == 6:  # (B, n_clips, C, T, H, W) — multi-clip
                B, n_clips, C, T, H, W = clips.shape
                clips_flat = clips.view(B * n_clips, C, T, H, W)
                tokens = self.encoder(clips_flat)                    # (B*n_clips, N, D)
                pooled = tokens.mean(dim=1)                          # (B*n_clips, D)
                features = pooled.view(B, n_clips, -1).mean(dim=1)  # (B, D)
            else:  # (B, C, T, H, W) — single-clip
                tokens = self.encoder(clips)   # (B, N, D)
                features = tokens.mean(dim=1)  # (B, D)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
