from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_cached(source: Path, cache: Path) -> Path:
    """Copy source tree to cache if not already present. Idempotent.

    Copies source → cache using shutil.copytree(dirs_exist_ok=True).
    Skips the copy if cache already exists and is non-empty.
    Logs progress so the user knows a large copy is in progress.

    Call from the main process before DataLoader workers fork.
    """
    source = Path(source)
    cache = Path(cache)

    if cache.exists() and any(cache.iterdir()):
        logger.info("Cache hit: %s already populated, skipping copy.", cache)
        return cache

    logger.info("Cache miss: copying %s → %s (this may take several minutes).", source, cache)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src=source, dst=cache, dirs_exist_ok=True)
    logger.info("Cache copy complete: %s", cache)
    return cache
