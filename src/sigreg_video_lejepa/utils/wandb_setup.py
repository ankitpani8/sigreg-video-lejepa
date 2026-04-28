"""WandB run ID lookup for seamless multi-session resume.

Returns an existing run's ID so that re-running a training cell in Kaggle
appends steps to the same WandB run instead of creating a duplicate.
"""
from __future__ import annotations


def get_or_create_run_id(
    project: str,
    entity: str,
    run_name: str,
    wandb_token: str | None,
) -> str | None:
    """Return existing WandB run ID matching run_name, or None to start a new run.

    Returns None silently when wandb is not available or the token is absent
    (e.g. local CPU smoke tests).
    """
    if not wandb_token:
        return None
    try:
        import wandb

        api = wandb.Api(api_key=wandb_token)
        runs = api.runs(f"{entity}/{project}", filters={"displayName": run_name})
        for run in runs:
            return run.id  # return ID of first (most recent) matching run
    except Exception:
        pass
    return None
