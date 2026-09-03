"""Run the nested linear probes and simultaneous 14-cut-point permutation test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common
import probe

RT_KCAL = 1.98720425e-3 * 298.15 * np.log(10)
MODULES = {
    1: ("z_aff1_block", "affinity_pred_value1"),
    2: ("z_aff2_block", "affinity_pred_value2"),
}


def load(directory: str, seed: int) -> dict:
    """Load one captured record per complex for the requested seed."""
    records = {}
    for path in sorted((common.WORK / directory).glob(f"*_s{seed}.npz")):
        stem = path.stem[: -len(f"_s{seed}")]
        pdb, ligand = stem.split("_", 1)
        records[(pdb, ligand)] = dict(np.load(path, allow_pickle=False))
    return records


def read_subset(path: Path) -> set[tuple[str, str, str]]:
    with path.open() as handle:
        return {
            (row["pdb_a"], row["pdb_b"], row["het"]) for row in csv.DictReader(handle)
        }


def _decompose(train: np.ndarray, test: np.ndarray):
    if not len(train):
        return None, len(test)
    spread = np.sqrt((train**2).mean(axis=0))
    spread[spread == 0] = 1.0
    left, singular, right = np.linalg.svd(train / spread, full_matrices=False)
    return left, singular, right, (test / spread) @ right.T


def _ridge_path(block, target: np.ndarray, alphas) -> np.ndarray:
    if len(block) == 2:
        return np.zeros((len(alphas), block[1]))
    left, singular, _right, projected = block
    weighted = left.T @ target
    return np.asarray(
        [
            projected @ ((singular / (singular**2 + alpha)) * weighted)
            for alpha in alphas
        ]
    )


def _prepare(_probe, features: np.ndarray, groups: np.ndarray) -> list:
    """Cache label-independent decompositions for all outer and inner folds."""
    prepared = []
    for held_out in np.unique(groups):
        test = groups == held_out
        train = ~test
        train_features = features[train]
        inner = [
            (mask, _decompose(train_features[~mask], train_features[mask]))
            for mask in probe._grouped_folds(groups[train])
        ]
        prepared.append(
            (test, train, inner, _decompose(train_features, features[test]))
        )
    return prepared


def _out_of_fold(prepared: list, target: np.ndarray, alphas=None) -> np.ndarray:
    """Fit the nested through-origin ridge probe using cached decompositions."""
    alphas = probe.ALPHAS if alphas is None else alphas
    predicted = np.empty(len(target))
    for test, train, inner_blocks, final in prepared:
        train_target = target[train]
        inner_predictions = np.empty((len(alphas), len(train_target)))
        for mask, block in inner_blocks:
            inner_predictions[:, mask] = _ridge_path(block, train_target[~mask], alphas)
        best = int(np.argmin(((inner_predictions - train_target) ** 2).mean(axis=1)))
        predicted[test] = _ridge_path(final, train_target, alphas)[best]
    return predicted


def permutations(
    groups: np.ndarray, repeats: int, seed: int, within: bool
) -> list[np.ndarray]:
    """Generate global or within-protein label-index permutations."""
    generator = np.random.default_rng(seed)
    index = np.arange(len(groups))
    if not within:
        return [generator.permutation(index) for _ in range(repeats)]
    blocks = [index[groups == group] for group in np.unique(groups)]
    output = []
    for _ in range(repeats):
        order = index.copy()
        for block in blocks:
            if len(block) > 1:
                order[block] = generator.permutation(block)
        output.append(order)
    return output


def collect(records: dict, frame, subset: set) -> dict:
    """Assemble aligned labels, outputs, and per-cut-point feature differences."""
    rows = []
    for row in frame.itertuples():
        key = (row.pdb_a, row.pdb_b, row.het)
        reference = records.get((str(row.pdb_a).lower(), row.het))
        variant = records.get((str(row.pdb_b).lower(), row.het))
        if key in subset and reference is not None and variant is not None:
            rows.append((row, reference, variant))
    if len(rows) != len(subset):
        raise RuntimeError(f"expected {len(subset)} complete pairs, found {len(rows)}")
    measured = np.asarray([float(row.ddg_kcal) for row, _, _ in rows])
    proteins = np.asarray([str(row.uniprot) for row, _, _ in rows])
    systems = np.asarray([f"{row.uniprot}|{row.het}" for row, _, _ in rows])
    combined = np.asarray(
        [
            RT_KCAL
            * float(variant["affinity_pred_value"] - reference["affinity_pred_value"])
            for _, reference, variant in rows
        ]
    )
    modules = {}
    for number, (prefix, control_key) in MODULES.items():
        blocks = sorted(key for key in rows[0][1] if key.startswith(prefix))
        control = np.asarray(
            [
                RT_KCAL * float(variant[control_key] - reference[control_key])
                for _, reference, variant in rows
            ]
        )
        modules[number] = {
            "blocks": blocks,
            "control": control,
            "features": {
                block: np.asarray(
                    [
                        variant[block].astype(np.float64)
                        - reference[block].astype(np.float64)
                        for _, reference, variant in rows
                    ]
                )
                for block in blocks
            },
        }
    return {
        "measured": measured,
        "proteins": proteins,
        "systems": systems,
        "combined": combined,
        "modules": modules,
    }


def analyse(data: dict, repeats: int) -> dict:
    """Fit observed and permuted probes with one shared permutation family."""
    measured = data["measured"]
    proteins = data["proteins"]
    global_orders = permutations(proteins, repeats, seed=0, within=False)
    within_orders = permutations(proteins, repeats, seed=1, within=True)
    global_nulls = []
    within_nulls = []
    modules = {}
    observed = []
    locations = []

    for number, values in data["modules"].items():
        depths = {}
        for block in values["blocks"]:
            prepared = _prepare(probe, values["features"][block], proteins)
            measured_r = probe.score(measured, _out_of_fold(prepared, measured))[
                "pearson"
            ]
            control_r = probe.score(
                values["control"], _out_of_fold(prepared, values["control"])
            )["pearson"]
            global_at_depth = np.asarray(
                [
                    probe.score(
                        measured[order], _out_of_fold(prepared, measured[order])
                    )["pearson"]
                    for order in global_orders
                ]
            )
            within_at_depth = np.asarray(
                [
                    probe.score(
                        measured[order], _out_of_fold(prepared, measured[order])
                    )["pearson"]
                    for order in within_orders
                ]
            )
            label = block.replace(MODULES[number][0], "")
            depths[label] = {"r_to_ddg": measured_r, "r_to_head": control_r}
            observed.append(measured_r)
            locations.append((f"module{number}", label))
            global_nulls.append(global_at_depth)
            within_nulls.append(within_at_depth)
        modules[f"module{number}"] = {
            "n_cut_points": len(depths),
            "depths": depths,
        }

    observed = np.asarray(observed)
    peak_index = int(np.argmax(observed))
    global_max = np.max(np.vstack(global_nulls), axis=0)
    within_max = np.max(np.vstack(within_nulls), axis=0)
    peak = float(observed[peak_index])
    global_exceedances = int(np.sum(global_max >= peak))
    within_exceedances = int(np.sum(within_max >= peak))
    module, block = locations[peak_index]
    combined = data["combined"]
    module_outputs = [data["modules"][number]["control"] for number in MODULES]
    return {
        "generated_by": "tools/trunk_probe/layer_analysis.py",
        "method": (
            "ridge through the origin on five-seed-averaged pooled representation "
            "differences; protein-grouped inner selection; leave-one-protein-out; "
            "shared permutations and a maximum over all 14 cut points"
        ),
        "cohort_definition": (
            "single substitution; Kd or Ki on both sides; complete predictions from "
            "Boltz-2, AQAffinity, and AuroBind"
        ),
        "permutations": repeats,
        "cohort": {
            "counts": {
                "pairs": len(measured),
                "proteins": len(np.unique(data["proteins"])),
                "systems": len(np.unique(data["systems"])),
                "singleton_proteins": int(
                    np.sum(np.unique(data["proteins"], return_counts=True)[1] == 1)
                ),
            },
            "modules": modules,
            "joint": {
                "n_cut_points": len(observed),
                "peak": peak,
                "peak_module": module,
                "peak_at": block,
                "envelope_global_p95": float(np.percentile(global_max, 95)),
                "envelope_within_protein_p95": float(np.percentile(within_max, 95)),
                "p_global": global_exceedances / repeats,
                "p_within": within_exceedances / repeats,
                "global_exceedances": global_exceedances,
                "within_protein_exceedances": within_exceedances,
            },
            "head_output": {
                "combined_r_vs_measured": float(
                    stats.pearsonr(measured, combined).statistic
                ),
                "combined_spearman": float(
                    stats.spearmanr(measured, combined).statistic
                ),
                "combined_slope": float(stats.linregress(measured, combined).slope),
                "module1_r_vs_measured": float(
                    stats.pearsonr(measured, module_outputs[0]).statistic
                ),
                "module2_r_vs_measured": float(
                    stats.pearsonr(measured, module_outputs[1]).statistic
                ),
                "destabilising_share": float(np.mean(measured > 0)),
            },
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", default="layers_unified_mean")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pairs", type=Path, default=common.WORK / "primary_cohort_163.csv"
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=common.WORK / "probe_results.json")
    arguments = parser.parse_args(argv)
    records = load(arguments.representations, arguments.seed)
    data = collect(records, probe.canonical_pair_frame(), read_subset(arguments.pairs))
    report = analyse(data, arguments.permutations)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
