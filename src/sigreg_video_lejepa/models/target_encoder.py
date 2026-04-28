from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SharedTargetEncoder:
    """Target encoder that shares weights with the context encoder (LeJEPA mode).

    encode() detaches gradients so the prediction loss does not flow into the
    target side. update() is a deliberate no-op — called unconditionally by the
    Lightning module; duck typing keeps the training loop free of isinstance checks.
    """

    def encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return encoder(x).detach()

    def update(self, encoder: nn.Module, decay: float | None) -> None:
        pass


class EMATargetEncoder(nn.Module):
    """Target encoder with exponential moving average weights (V-JEPA mode).

    Owns a shadow copy of the encoder. Shadow weights are updated each step via:
        shadow = decay * shadow + (1 - decay) * encoder
    encode() always runs on the shadow weights with gradients detached.

    Must call initialize_from(encoder) before first use. This two-phase init
    allows Hydra to instantiate EMATargetEncoder with no constructor args, with
    the encoder passed in pretrain.py after both objects are built.
    """

    def __init__(self) -> None:
        super().__init__()
        self._shadow: nn.Module | None = None

    def initialize_from(self, encoder: nn.Module) -> None:
        self._shadow = copy.deepcopy(encoder)
        for p in self._shadow.parameters():
            p.requires_grad_(False)

    def encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._shadow is None:
            raise RuntimeError("EMATargetEncoder: call initialize_from(encoder) before use")
        return self._shadow(x).detach()

    @torch.no_grad()
    def update(self, encoder: nn.Module, decay: float | None) -> None:
        if self._shadow is None:
            raise RuntimeError("EMATargetEncoder: call initialize_from(encoder) before use")
        if decay is None:
            raise ValueError("EMATargetEncoder.update requires a float decay, got None")
        for shadow_p, enc_p in zip(self._shadow.parameters(), encoder.parameters()):
            shadow_p.data.mul_(decay).add_(enc_p.data, alpha=1.0 - decay)
