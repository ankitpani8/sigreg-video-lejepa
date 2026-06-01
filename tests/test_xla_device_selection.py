"""Unit tests for XLA-aware device selection in extract_features._select_device().

Tests use monkeypatching only — no real XLA runtime required.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import scripts.extract_features as ef


def test_cuda_selected_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = ef._select_device()
    assert device.type == "cuda"


def test_xla_selected_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    sentinel = object()
    fake_xm = types.ModuleType("torch_xla.core.xla_model")
    fake_xm.xla_device = lambda: sentinel  # type: ignore[attr-defined]
    fake_core = types.ModuleType("torch_xla.core")
    fake_xla = types.ModuleType("torch_xla")
    fake_xla.core = fake_core  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", fake_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", fake_xm)

    result = ef._select_device()
    assert result is sentinel


def test_cpu_selected_when_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Setting a module entry to None causes any `import` of it to raise ImportError.
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", None)

    device = ef._select_device()
    assert device == torch.device("cpu")


def test_cuda_takes_priority_over_xla(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA should win even when torch_xla is importable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    sentinel = object()
    fake_xm = types.ModuleType("torch_xla.core.xla_model")
    fake_xm.xla_device = lambda: sentinel  # type: ignore[attr-defined]
    fake_core = types.ModuleType("torch_xla.core")
    fake_xla = types.ModuleType("torch_xla")
    fake_xla.core = fake_core  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", fake_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", fake_xm)

    device = ef._select_device()
    assert device.type == "cuda"
    assert device is not sentinel
