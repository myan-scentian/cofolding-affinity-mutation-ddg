"""Loading the tables a claim script needs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mutation_ddg.paths import PREDICTIONS_ROOT, PROCESSED_ROOT

PROCESSED = PROCESSED_ROOT
PREDICTIONS = PREDICTIONS_ROOT

RT_KCAL = 1.3642


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with the driver in tools/ that produces it; see tools/README.md."
        )
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def pairs(source: str) -> pd.DataFrame:
    if source != "hiqbind":
        raise KeyError("this release contains only the HiQBind evaluation")
    return _read(PROCESSED / "ddg_bench_v2.parquet")


def predictions(model: str, source: str) -> pd.DataFrame:
    """Per-complex, per-seed predictions."""
    return _read(PREDICTIONS / f"{model}_{source}.parquet")
