"""Reproduce the tabulated affinity-head results reported in the paper."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "4")

import numpy as np
import pandas as pd
from scipy import stats

from mutation_ddg.analysis import metrics, pairing
from mutation_ddg.io import tables
from mutation_ddg.paths import ARTIFACT_ROOT

MODELS = (
    ("Boltz-2", "boltz2_v2"),
    ("AQAffinity", "aqaffinity_v2"),
    ("AuroBind", "aurobind"),
)
BOOTSTRAP_SEED = 0
MAGNITUDE_BINS = 6


def finite(value):
    """Replace non-finite floating-point values with JSON nulls."""
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Select complete measured and predicted changes and identify each system."""
    output = frame[["uniprot", "het", "ddg_kcal", "pred_ddg"]].dropna().copy()
    output["system"] = output.uniprot.astype(str) + "|" + output.het.astype(str)
    return output.reset_index(drop=True)


def r2_zero(frame: pd.DataFrame) -> float:
    return metrics.skill_against_zero(frame.ddg_kcal, frame.pred_ddg, frame.system)


def balanced_mae(frame: pd.DataFrame) -> float:
    return metrics.balanced_mae(frame.ddg_kcal, frame.pred_ddg, frame.system)


def slope(frame: pd.DataFrame) -> float:
    return float(stats.linregress(frame.ddg_kcal, frame.pred_ddg).slope)


def within_system_r(frame: pd.DataFrame) -> float:
    return metrics.within_system_association(frame, "ddg_kcal", "pred_ddg")[0]


def interval(frame: pd.DataFrame, statistic, draws: int) -> list[float]:
    return list(
        metrics.clustered_interval(frame, statistic, draws=draws, seed=BOOTSTRAP_SEED)
    )


def counts(frame: pd.DataFrame) -> dict:
    return {
        "pairs": len(frame),
        "systems": frame.groupby(["uniprot", "het"]).ngroups,
        "proteins": frame.uniprot.nunique(),
    }


def absolute_frame(
    pairs: pd.DataFrame,
    pair_frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    measured = pd.concat(
        [
            pairs[[side, "het", "uniprot", affinity]].rename(
                columns={side: "pdb", affinity: "measured"}
            )
            for side, affinity in (("pdb_a", "logaff_a"), ("pdb_b", "logaff_b"))
        ]
    )
    measured["pdb"] = measured.pdb.str.lower()
    measured = measured.drop_duplicates(["pdb", "het"])
    keys = set(zip(pair_frame.pdb_a.str.lower(), pair_frame.het)) | set(
        zip(pair_frame.pdb_b.str.lower(), pair_frame.het)
    )
    selected = pd.DataFrame(sorted(keys), columns=["pdb", "het"])
    return (
        selected.merge(measured, on=["pdb", "het"], how="inner")
        .merge(pairing.average_seeds(predictions), on=["pdb", "het"], how="inner")
        .dropna(subset=["measured", "predicted"])
    )


def absolute_metrics(frame: pd.DataFrame) -> dict:
    return {
        "complexes": len(frame),
        "systems": frame.groupby(["uniprot", "het"]).ngroups,
        "pooled_r": float(stats.pearsonr(frame.measured, frame.predicted).statistic),
        "within_system_r": metrics.within_system_association(
            frame, "measured", "predicted"
        )[0],
    }


def magnitude_results(frame: pd.DataFrame, edges: np.ndarray) -> list[dict]:
    magnitude = np.abs(frame.ddg_kcal.to_numpy(float))
    membership = np.searchsorted(edges[1:-1], magnitude, side="right")
    output = []
    for index in range(len(edges) - 1):
        part = frame[membership == index]
        output.append(
            {
                "lower_kcal": float(edges[index]),
                "upper_kcal": None
                if np.isinf(edges[index + 1])
                else float(edges[index + 1]),
                "pairs": len(part),
                "r2_zero": r2_zero(part),
                "measured_sd_kcal": float(part.ddg_kcal.std(ddof=1)),
                "predicted_sd_kcal": float(part.pred_ddg.std(ddof=1)),
            }
        )
    return output


def direction_results(frame: pd.DataFrame, draws: int) -> dict:
    output = {}
    for name, selected in (
        ("destabilising", frame.ddg_kcal > 0),
        ("stabilising", frame.ddg_kcal < 0),
    ):
        part = frame[selected].copy()
        output[name] = {
            "pairs": len(part),
            "r2_zero": r2_zero(part),
            "r2_zero_interval": interval(part, r2_zero, draws),
            "mae_kcal": balanced_mae(part),
            "slope": slope(part),
        }
    return output


def leave_one_protein_out_mean(measured, proteins) -> np.ndarray:
    """Predict each pair using the mean among all other proteins."""
    measured = np.asarray(measured, float)
    proteins = np.asarray(proteins)
    if len(measured) != len(proteins):
        raise ValueError("measurements and protein labels must have the same length")
    predicted = np.empty(len(measured))
    for protein in np.unique(proteins):
        held_out = proteins == protein
        if held_out.all():
            raise ValueError(
                "leave-one-protein-out prediction needs at least two proteins"
            )
        predicted[held_out] = measured[~held_out].mean()
    return predicted


def lopo_results(frame: pd.DataFrame, draws: int) -> dict:
    measured = frame.ddg_kcal.to_numpy(float)
    proteins = frame.uniprot.astype(str).to_numpy()
    systems = frame.system.astype(str).to_numpy()
    predicted = leave_one_protein_out_mean(measured, proteins)
    point_frame = frame.assign(pred_ddg=predicted)
    blocks = [part.index.to_numpy() for _, part in frame.groupby(["uniprot", "het"])]
    sizes = np.array([len(block) for block in blocks])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = []
    for _ in range(draws):
        chosen = generator.integers(0, len(blocks), len(blocks))
        take = np.concatenate([blocks[index] for index in chosen])
        copies = np.repeat(np.arange(len(chosen)), sizes[chosen]).astype(str)
        drawn_measured = measured[take]
        drawn_predicted = leave_one_protein_out_mean(drawn_measured, proteins[take])
        drawn_systems = systems[take] + "#" + copies
        sampled.append(
            metrics.skill_against_zero(drawn_measured, drawn_predicted, drawn_systems)
        )
    return {
        "definition": "mean measured ddG among pairs from all other proteins",
        "fit": "refit within each system-clustered bootstrap draw",
        "pairs": len(frame),
        "r2_zero": r2_zero(point_frame),
        "r2_zero_interval": list(np.percentile(sampled, [2.5, 97.5])),
        "mae_kcal": balanced_mae(point_frame),
    }


def build(draws: int) -> dict:
    pairs = tables.pairs("hiqbind")
    frames = pairing.canonical_frames(pairs)
    primary = pairing.primary_pairs(pairs)
    eligible = int(pairing.single_substitution_kd_ki(pairs).sum())
    if eligible != 170 or len(primary) != 163:
        raise RuntimeError(
            f"expected 170 eligible and 163 common-complete pairs; got "
            f"{eligible} and {len(primary)}"
        )

    prepared = {
        name: prepare(pairing.restrict(frames[name], primary)) for name, _ in MODELS
    }
    reference = prepared["Boltz-2"]
    reference_keys = pairing.pair_keys(pairing.restrict(frames["Boltz-2"], primary))
    for name, frame in prepared.items():
        source_keys = pairing.pair_keys(pairing.restrict(frames[name], primary))
        if source_keys != reference_keys:
            raise RuntimeError(f"{name} is not aligned with the common cohort")
        if not np.allclose(frame.ddg_kcal, reference.ddg_kcal):
            raise RuntimeError(f"{name} carries different measurements")

    edges = np.concatenate(
        (
            [0.0],
            np.quantile(
                np.abs(reference.ddg_kcal),
                np.arange(1, MAGNITUDE_BINS) / MAGNITUDE_BINS,
            ),
            [np.inf],
        )
    )
    report = {
        "generated_by": "scripts/main_results.py",
        "bootstrap": {
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "cluster": ["uniprot", "het"],
        },
        "cohort": {
            "hiqbind_pairs_after_initial_filters": len(pairs),
            "single_substitution_kd_ki_pairs": eligible,
            "common_complete_pairs": len(primary),
            "systems": reference.groupby(["uniprot", "het"]).ngroups,
            "proteins": reference.uniprot.nunique(),
            "filters": list(pairing.PRIMARY_FILTERS),
            "models_required": [name for name, _ in MODELS],
        },
        "zero_change_null": {
            "r2_zero": 0.0,
            "mae_kcal": metrics.balanced_mae(
                reference.ddg_kcal, np.zeros(len(reference)), reference.system
            ),
        },
        "lopo_mean": lopo_results(reference, draws),
        "magnitude_bin_edges_kcal": [
            None if np.isinf(edge) else float(edge) for edge in edges
        ],
        "models": {},
    }

    for name, table_name in MODELS:
        predictions = tables.predictions(table_name, "hiqbind")
        seed_counts = (
            predictions.dropna(subset=["affinity_pred_value"])
            .groupby(["pdb", "het"])
            .size()
        )
        if set(seed_counts.unique()) != {5}:
            raise RuntimeError(f"{name} does not have five predictions per complex")
        frame = prepared[name]
        source = pairing.restrict(frames[name], primary)
        report["models"][name] = {
            "prediction_table": f"{table_name}_hiqbind.parquet",
            "seeds": [1, 2, 3, 4, 5],
            "counts": counts(frame),
            "r2_zero": r2_zero(frame),
            "r2_zero_interval": interval(frame, r2_zero, draws),
            "mae_kcal": balanced_mae(frame),
            "slope": slope(frame),
            "within_system_r": within_system_r(frame),
            "within_system_r_interval": interval(frame, within_system_r, draws),
            "absolute_affinity": absolute_metrics(
                absolute_frame(pairs, source, predictions)
            ),
            "magnitude_bins": magnitude_results(frame, edges),
            "direction": direction_results(frame, draws),
        }
        print(f"{name}: {report['models'][name]['r2_zero']:+.3f}", flush=True)
    return finite(report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=metrics.BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_ROOT / "work" / "main_results.json",
    )
    arguments = parser.parse_args(argv)
    report = build(arguments.draws)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
