"""Construct mutation-pair predictions from per-complex affinity outputs."""

from __future__ import annotations

import pandas as pd

from mutation_ddg.io.tables import RT_KCAL

MODELS = (
    ("Boltz-2", "boltz2_v2"),
    ("AQAffinity", "aqaffinity_v2"),
    ("AuroBind", "aurobind"),
)
MODEL_NAMES = tuple(name for name, _ in MODELS)
THERMODYNAMIC_MEASUREMENTS = ("kd", "ki")
PRIMARY_FILTERS = (
    "single_substitution",
    "kd_or_ki_only",
    "complete_predictions_from_all_three_heads",
)


class PairSet(frozenset):
    """Pair keys with the filters that define their cohort."""

    screens: tuple[str, ...]

    def __new__(cls, keys, screens=()):
        instance = super().__new__(cls, keys)
        instance.screens = tuple(screens)
        return instance

    def __init__(self, keys, screens=()):
        super().__init__()


def average_seeds(
    predictions: pd.DataFrame, value: str = "affinity_pred_value"
) -> pd.DataFrame:
    """Average scalar predictions over seeds for each complex."""
    grouped = predictions.dropna(subset=[value]).groupby(["pdb", "het"])
    return grouped.agg(
        predicted=(value, "mean"),
        predicted_sd=(value, lambda column: column.std(ddof=1)),
        n_seeds=(value, "size"),
        n_residues=("nres", "first"),
    ).reset_index()


def to_pairs(
    pairs: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_column: str = "pdb_a",
    variant_column: str = "pdb_b",
    ligand_column: str = "het",
) -> pd.DataFrame:
    """Attach predicted mutation effects to pairs with both predictions present."""
    per_complex = average_seeds(predictions).set_index(["pdb", "het"])
    output = pairs.copy()
    for side, column in (("ref", reference_column), ("var", variant_column)):
        keys = pd.MultiIndex.from_arrays(
            [output[column].str.lower(), output[ligand_column]]
        )
        output[f"predicted_{side}"] = per_complex.predicted.reindex(keys).to_numpy()
        output[f"n_seeds_{side}"] = per_complex.n_seeds.reindex(keys).to_numpy()
    output = output.dropna(subset=["predicted_ref", "predicted_var"]).copy()
    output["pred_ddg"] = RT_KCAL * (output.predicted_var - output.predicted_ref)
    return output


def pair_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(zip(frame.pdb_a, frame.pdb_b, frame.het))


def canonical_frames(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build pair-level predictions for the paper's three affinity heads."""
    from mutation_ddg.io import tables

    return {
        name: to_pairs(pairs, tables.predictions(table_name, "hiqbind"))
        for name, table_name in MODELS
    }


def coverage_intersection(pairs: pd.DataFrame) -> set[tuple[str, str, str]]:
    frames = canonical_frames(pairs)
    return set.intersection(*(set(pair_keys(frames[name])) for name in MODEL_NAMES))


def single_substitution_kd_ki(pairs: pd.DataFrame) -> pd.Series:
    """Select one-substitution pairs with Kd or Ki on both sides."""
    if "min_mut_lig_dist" not in pairs or not (pairs.min_mut_lig_dist <= 5).all():
        raise ValueError("the HiQBind input must already satisfy the 5 A filter")
    return (
        pairs.n_sub.eq(1)
        & pairs.meas_a.str.lower().isin(THERMODYNAMIC_MEASUREMENTS)
        & pairs.meas_b.str.lower().isin(THERMODYNAMIC_MEASUREMENTS)
    )


def primary_pairs(pairs: pd.DataFrame) -> PairSet:
    """Return the complete 163-pair cohort used throughout the paper."""
    eligible = set(pair_keys(pairs[single_substitution_kd_ki(pairs)]))
    return PairSet(eligible & coverage_intersection(pairs), screens=PRIMARY_FILTERS)


def restrict(frame: pd.DataFrame, keep: set) -> pd.DataFrame:
    selected = [key in keep for key in pair_keys(frame)]
    return frame[selected].copy()
