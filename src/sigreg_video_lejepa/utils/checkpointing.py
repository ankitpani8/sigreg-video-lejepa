"""HuggingFace Hub checkpoint persistence for cross-session Kaggle training.

Checkpoints are stored in a single shared repo under
    checkpoints/{experiment_name}/step_{step:07d}.ckpt
so multiple runs coexist without collision.
"""
from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def save_checkpoint_to_hf(
    ckpt_path: Path,
    repo_id: str,
    experiment_name: str,
    step: int,
    token: str,
) -> None:
    """Upload ckpt_path to HF Hub under checkpoints/{experiment_name}/step_{step:07d}.ckpt."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    path_in_repo = f"checkpoints/{experiment_name}/step_{step:07d}.ckpt"
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def list_checkpoints_on_hf(repo_id: str, experiment_name: str, token: str) -> list[int]:
    """Return sorted list of step numbers available on HF Hub for experiment_name."""
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="model", token=token)
    except Exception:
        return []
    prefix = f"checkpoints/{experiment_name}/step_"
    steps: list[int] = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".ckpt"):
            m = re.search(r"step_(\d+)\.ckpt$", f)
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def load_latest_checkpoint_from_hf(
    repo_id: str,
    experiment_name: str,
    token: str,
    local_dir: Path,
) -> Path | None:
    """Download the highest-numbered checkpoint for experiment_name. Returns None if none exist."""
    steps = list_checkpoints_on_hf(repo_id, experiment_name, token)
    if not steps:
        return None
    latest_step = steps[-1]
    path_in_repo = f"checkpoints/{experiment_name}/step_{latest_step:07d}.ckpt"
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="model",
        token=token,
        local_dir=str(local_dir),
    )
    return Path(local_path)
