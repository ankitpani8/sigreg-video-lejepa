"""Pytest fixtures for UCF101 tests.

Builds a minimal UCF101-shaped directory tree using synthetic .avi files written
via PyAV, so tests run without real UCF101 data.
"""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

# Fixture parameters
_CLASSES = ["ApplyEyeMakeup", "Archery", "BabyCrawling"]
_CLIPS_PER_CLASS = 3
_NUM_FRAMES = 32      # 16 × stride-6 = 96 source frames needed; 32 < 96 triggers looping
_HEIGHT = 64
_WIDTH = 64
_FPS = 25

# UCF101Dataset validates ≥100 class subdirectories. We pad with empty dirs so the
# fixture matches the expected structure without writing extra videos.
_TOTAL_FAKE_CLASSES = 101


def _write_avi(path: Path, num_frames: int, height: int, width: int, fps: int) -> None:
    """Write a synthetic .avi file with mpeg4 codec using PyAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=abs(hash(path.name)) % (2**31))

    with av.open(str(path), "w", format="avi") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"qscale:v": "5"}

        for _ in range(num_frames):
            array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="session")
def ucf101_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a minimal UCF101 directory tree with synthetic .avi files.

    Structure::

        root/
          videos/              ← data_root (UCF-101/)
            ApplyEyeMakeup/
              v_ApplyEyeMakeup_g01_c01.avi  (32 frames, 64×64)
              v_ApplyEyeMakeup_g01_c02.avi
              v_ApplyEyeMakeup_g01_c03.avi
            Archery/  ...
            BabyCrawling/  ...
          splits/              ← split_root (ucfTrainTestlist/)
            classInd.txt
            trainlist01.txt    (first 2 clips per class = 6 entries)
            testlist01.txt     (last clip per class = 3 entries, no labels)
    """
    root = tmp_path_factory.mktemp("ucf101")
    data_root = root / "videos"
    split_root = root / "splits"
    split_root.mkdir()

    train_lines: list[str] = []
    test_lines: list[str] = []
    classind_lines: list[str] = []

    for cls_idx, cls_name in enumerate(_CLASSES, start=1):
        classind_lines.append(f"{cls_idx} {cls_name}")
        cls_dir = data_root / cls_name
        cls_dir.mkdir(parents=True)

        for clip_num in range(1, _CLIPS_PER_CLASS + 1):
            fname = f"v_{cls_name}_g01_c{clip_num:02d}.avi"
            fpath = cls_dir / fname
            _write_avi(fpath, _NUM_FRAMES, _HEIGHT, _WIDTH, _FPS)

            rel = f"{cls_name}/{fname}"
            if clip_num <= 2:
                train_lines.append(f"{rel} {cls_idx}")
            else:
                test_lines.append(rel)

    # Pad with empty class directories so the ≥100 validation in UCF101Dataset passes.
    for pad_idx in range(len(_CLASSES) + 1, _TOTAL_FAKE_CLASSES + 1):
        (data_root / f"PaddingClass{pad_idx:03d}").mkdir()

    (split_root / "classInd.txt").write_text("\n".join(classind_lines) + "\n")
    (split_root / "trainlist01.txt").write_text("\n".join(train_lines) + "\n")
    (split_root / "testlist01.txt").write_text("\n".join(test_lines) + "\n")

    return root
