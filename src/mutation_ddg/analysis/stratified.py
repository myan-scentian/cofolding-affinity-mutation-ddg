"""Preparation and equal-count magnitude bins used by Figure 2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Select complete values and identify protein-ligand systems."""
    output = frame[["uniprot", "het", "ddg_kcal", "pred_ddg"]].dropna().copy()
    output["system"] = output.uniprot.astype(str) + "|" + output.het.astype(str)
    return output.reset_index(drop=True)


def quantile_edges(values, parts: int = 5) -> list[float]:
    """Return equal-count bin edges for absolute values."""
    values = np.abs(np.asarray(values, float))
    return [0.0, *np.quantile(values, np.arange(1, parts) / parts), np.inf]
