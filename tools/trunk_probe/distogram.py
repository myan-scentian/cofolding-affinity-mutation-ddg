"""Boltz-2 distogram discretisation and token matching."""

from __future__ import annotations

import numpy as np
import probe

NUM_BINS = 64
BOUNDARIES = np.linspace(2.0, 22.0, NUM_BINS - 1)
WANTED = (
    "disto_coords",
    "affinity_pred_value",
    "asym_id",
    "res_idx",
    "res_type",
    "receptor_mask",
    "mol_type",
)


def bins(points: np.ndarray) -> np.ndarray:
    """Discretise representative-atom distances using Boltz-2's 63 thresholds."""
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return np.searchsorted(BOUNDARIES, distances, side="left").astype(np.uint8)


def shared_rows(reference: dict, variant: dict) -> np.ndarray | None:
    """Match receptor tokens by identity and ligand atoms by their stored order."""
    shared, _ = probe.matched_positions(reference, variant)
    left = np.flatnonzero(~reference["receptor_mask"].astype(bool))
    right = np.flatnonzero(~variant["receptor_mask"].astype(bool))
    if len(left) != len(right):
        return None
    if (
        len(left)
        and not (reference["mol_type"][left] == variant["mol_type"][right]).all()
    ):
        return None
    if len(left):
        shared = np.vstack([shared, np.stack([left, right], axis=1)])
    return shared if len(shared) >= 3 else None
