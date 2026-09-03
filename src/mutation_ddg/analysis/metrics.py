"""System-balanced metrics and clustered intervals used in the manuscript."""

from __future__ import annotations

import numpy as np

BOOTSTRAP_DRAWS = 2000


def _system_weights(systems) -> np.ndarray:
    """Assign every system total weight 1/S and divide it across its pairs."""
    systems = np.asarray(systems)
    unique, counts = np.unique(systems, return_counts=True)
    per_system = dict(zip(unique, counts))
    return np.array([1.0 / (len(unique) * per_system[system]) for system in systems])


def skill_against_zero(measured, predicted, systems) -> float:
    """Compute system-balanced R2 against predicting zero change."""
    measured = np.asarray(measured, float)
    predicted = np.asarray(predicted, float)
    weights = _system_weights(systems)
    model_error = float(np.sum(weights * (predicted - measured) ** 2))
    null_error = float(np.sum(weights * measured**2))
    return 1.0 - model_error / null_error if null_error else float("nan")


def balanced_mae(measured, predicted, systems) -> float:
    """Compute system-balanced mean absolute error."""
    measured = np.asarray(measured, float)
    predicted = np.asarray(predicted, float)
    return float(np.sum(_system_weights(systems) * np.abs(predicted - measured)))


def within_system_association(
    frame, measured: str, predicted: str, group: tuple = ("uniprot", "het")
) -> tuple[float, int, int]:
    """Correlate deviations after centring both quantities within each system."""
    counts = frame.groupby(list(group))[measured].transform("size")
    kept = frame[counts >= 2]
    if len(kept) < 3:
        return float("nan"), 0, 0
    measured_c = kept[measured] - kept.groupby(list(group))[measured].transform("mean")
    predicted_c = kept[predicted] - kept.groupby(list(group))[predicted].transform(
        "mean"
    )
    if measured_c.std() == 0 or predicted_c.std() == 0:
        return float("nan"), len(kept), 0
    correlation = float(np.corrcoef(measured_c, predicted_c)[0, 1])
    return correlation, len(kept), int(kept.groupby(list(group)).ngroups)


def clustered_interval(
    frame,
    statistic,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    group: tuple = ("uniprot", "het"),
) -> tuple[float, float]:
    """Return a percentile interval from resampling complete systems."""
    groups = [part for _, part in frame.groupby(list(group), dropna=False)]
    positions = {label: position for position, label in enumerate(frame.index)}
    offsets = [np.array([positions[label] for label in part.index]) for part in groups]
    index_values = frame.index.to_numpy()
    sizes = np.array([len(offset) for offset in offsets])

    def constant_within_system(column: str) -> bool:
        try:
            counts = frame.groupby(list(group), dropna=False)[column].nunique(
                dropna=False
            )
        except TypeError:
            return False
        return bool(counts.max() <= 1)

    labels = list(
        dict.fromkeys(
            list(group)
            + [column for column in frame.columns if constant_within_system(column)]
        )
    )
    as_text = {
        column: frame[column].astype(object).fillna("NA").astype(str).to_numpy()
        for column in labels
    }
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        chosen = generator.integers(0, len(groups), len(groups))
        take = np.concatenate([offsets[index] for index in chosen])
        copies = np.repeat(np.arange(len(chosen)), sizes[chosen]).astype(str)
        drawn = frame.loc[index_values[take]].copy()
        for column in labels:
            drawn[column] = as_text[column][take] + "#" + copies
        value = statistic(drawn)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
