"""Canonical project and artifact locations."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    """Repository root, so a fresh checkout works with no environment set."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pixi.toml").exists():
            return parent
    return Path.cwd()


DEFAULT_ARTIFACT_ROOT = _repo_root() / "artifacts"
ARTIFACT_ROOT = Path(
    os.environ.get("MUTATION_DDG_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
).expanduser()
DATA_ROOT = ARTIFACT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
PREDICTIONS_ROOT = DATA_ROOT / "predictions"
RESULTS_ROOT = ARTIFACT_ROOT / "results"
