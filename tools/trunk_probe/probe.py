"""Shared operations for the Boltz-2 representation probes."""

from __future__ import annotations

import numpy as np
from scipy import stats

from mutation_ddg.analysis import pairing
from mutation_ddg.io import tables

ALPHAS = 10.0 ** np.arange(-2, 7)
PAIR_SET_ATTR = "pair_set"


def canonical_pair_frame():
    """Load HiQBind and attach the archived Boltz-2 mutation effect."""
    pairs = tables.pairs("hiqbind")
    covered = pairing.to_pairs(pairs, tables.predictions("boltz2_v2", "hiqbind"))
    keys = ["pdb_a", "pdb_b", "het"]
    extra = [column for column in covered if column not in pairs]
    frame = pairs.merge(covered[keys + extra], on=keys, how="left")
    frame.attrs[PAIR_SET_ATTR] = {"source": "HiQBind", "pairs": len(frame)}
    return frame


def pair_set_provenance(frame) -> dict:
    """Return the pair-set description attached by canonical_pair_frame."""
    if PAIR_SET_ATTR not in frame.attrs:
        raise ValueError("pair-set provenance is missing")
    return dict(frame.attrs[PAIR_SET_ATTR])


def matched_positions(wild: dict, variant: dict) -> tuple[np.ndarray, np.ndarray]:
    """Match receptor tokens by chain and residue index and locate substitutions."""

    def keyed(side):
        return {
            (int(chain), int(residue)): index
            for index, (chain, residue) in enumerate(
                zip(side["asym_id"], side["res_idx"])
            )
            if side["receptor_mask"][index]
        }

    left = keyed(wild)
    right = keyed(variant)
    shared = np.array(
        [[left[key], right[key]] for key in sorted(set(left) & set(right))],
        dtype=int,
    ).reshape(-1, 2)
    if not len(shared):
        return shared, shared
    changed = shared[
        [
            wild["res_type"][reference] != variant["res_type"][mutant]
            for reference, mutant in shared
        ]
    ]
    return shared, changed


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine distance in float64."""
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(1 - (left @ right) / denominator) if denominator else np.nan


def _scaled(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = np.sqrt((train**2).mean(axis=0))
    spread[spread == 0] = 1.0
    return train / spread, test / spread


def ridge_path(x_train, y_train, x_test, alphas=ALPHAS) -> np.ndarray:
    """Return through-origin ridge predictions for every regularisation value."""
    train, test = _scaled(x_train, x_test)
    if not len(train):
        return np.zeros((len(alphas), len(test)))
    left, singular, right = np.linalg.svd(train, full_matrices=False)
    projected = test @ right.T
    weighted = left.T @ y_train
    return np.array(
        [
            projected @ ((singular / (singular**2 + alpha)) * weighted)
            for alpha in alphas
        ]
    )


def _grouped_folds(groups: np.ndarray, folds: int = 5) -> list[np.ndarray]:
    unique = np.unique(groups)
    assignment = {name: index % folds for index, name in enumerate(unique)}
    block = np.array([assignment[name] for name in groups])
    return [block == index for index in range(folds) if (block == index).any()]


def out_of_fold(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas=ALPHAS,
) -> tuple[np.ndarray, list[float]]:
    """Fit leave-one-protein-out ridge probes with grouped inner selection."""
    predicted = np.empty(len(target))
    chosen = []
    for held_out in np.unique(groups):
        test = groups == held_out
        train = ~test
        x_train = features[train]
        y_train = target[train]
        inner = np.empty((len(alphas), len(y_train)))
        for mask in _grouped_folds(groups[train]):
            inner[:, mask] = ridge_path(
                x_train[~mask], y_train[~mask], x_train[mask], alphas
            )
        best = int(np.argmin(((inner - y_train) ** 2).mean(axis=1)))
        predicted[test] = ridge_path(x_train, y_train, features[test], alphas)[best]
        chosen.append(float(alphas[best]))
    return predicted, chosen


def score(measured: np.ndarray, predicted: np.ndarray) -> dict:
    """Return the correlation and slope used by the probe analysis."""
    if predicted.std() == 0:
        return {"pearson": float("nan"), "slope": 0.0}
    return {
        "pearson": float(stats.pearsonr(measured, predicted).statistic),
        "slope": float(stats.linregress(measured, predicted).slope),
    }
