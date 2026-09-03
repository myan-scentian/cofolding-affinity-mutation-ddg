"""Compare Boltz-2 distogram changes after substitution and stochastic re-folding."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import distogram
import probe

from mutation_ddg.analysis import pairing
from mutation_ddg.io import tables

SHELLS = (
    (0.0, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 24.0),
)
STRATA = ("touched",) + tuple(f"shell_{lower:g}_{upper:g}" for lower, upper in SHELLS)
KEEP = (*distogram.WANTED, "ligand_mask", "n_cross_entries")
BOOTSTRAP_DRAWS = 2000


def artifact_root() -> Path:
    override = os.environ.get("MUTATION_DDG_ARTIFACT_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "artifacts"


def load(root: Path, seed: int) -> dict[tuple[str, str], dict]:
    """Load the arrays needed for one inference seed."""
    output = {}
    directory = root / f"s{seed}" / "predictions"
    if not directory.is_dir():
        return output
    for record in sorted(directory.iterdir()):
        path = record / f"representations_{record.name}.npz"
        if not path.exists():
            continue
        pdb, ligand = record.name.split("_", 1)
        data = np.load(path, allow_pickle=False)
        output[(pdb.lower(), ligand)] = {
            key: data[key] for key in KEEP if key in data.files
        }
    return output


def token_keys(record: dict) -> list[tuple[int, int]]:
    return [
        (int(chain), int(residue))
        for chain, residue in zip(record["asym_id"], record["res_idx"])
    ]


def comparison(
    reference: dict,
    variant: dict,
    substituted: set[tuple[int, int]],
) -> dict[str, tuple[int, int, int]] | None:
    """Count changed affinity-pooled entries at the site and in distance shells."""
    rows = distogram.shared_rows(reference, variant)
    if rows is None:
        return None
    left, right = rows[:, 0], rows[:, 1]
    receptor = reference["receptor_mask"][left].astype(bool)
    ligand = reference["ligand_mask"][left].astype(bool)
    if not np.array_equal(receptor, variant["receptor_mask"][right].astype(bool)):
        return None
    if not np.array_equal(ligand, variant["ligand_mask"][right].astype(bool)):
        return None

    changed = (
        distogram.bins(reference["disto_coords"])[np.ix_(left, left)]
        != distogram.bins(variant["disto_coords"])[np.ix_(right, right)]
    )
    off_diagonal = ~np.eye(len(left), dtype=bool)
    pooled = off_diagonal & (
        (ligand[:, None] & receptor[None, :])
        | (receptor[:, None] & ligand[None, :])
        | (ligand[:, None] & ligand[None, :])
    )
    keys = token_keys(reference)
    is_substituted = np.array([keys[index] in substituted for index in left]) & receptor
    if not is_substituted.any():
        return None

    touched = is_substituted[:, None] | is_substituted[None, :]
    coordinates = reference["disto_coords"].astype(np.float64)[left]
    to_substitution = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, is_substituted, :], axis=-1
    ).min(axis=1)
    entry_distance = np.minimum(to_substitution[:, None], to_substitution[None, :])
    masks = {"touched": pooled & touched}
    for lower, upper in SHELLS:
        masks[f"shell_{lower:g}_{upper:g}"] = (
            pooled & ~touched & (entry_distance > lower) & (entry_distance <= upper)
        )
    scope = int(pooled.sum())
    return {
        f"cross/{name}": (int((changed & mask).sum()), int(mask.sum()), scope)
        for name, mask in masks.items()
    }


def rate_statistics(records: list[tuple[float, int, int, int]]) -> dict:
    if not records:
        return {"n_comparisons": 0}
    rates = np.array([rate for rate, _, _, _ in records])
    return {
        "n_comparisons": len(records),
        "median": float(np.median(rates)),
    }


def paired_statistics(
    mutation: dict[tuple, dict[str, list[float]]],
    repeat: dict[tuple, dict[str, list[float]]],
    systems: dict[tuple, str],
    name: str,
) -> dict | None:
    left = []
    right = []
    keys = []
    for key in sorted(set(mutation) & set(repeat)):
        mutation_values = mutation[key].get(name)
        repeat_values = repeat[key].get(name)
        if mutation_values and repeat_values:
            keys.append(key)
            left.append(float(np.median(mutation_values)))
            right.append(float(np.median(repeat_values)))
    if len(left) < 10:
        return None
    left = np.array(left)
    right = np.array(right)
    difference = left - right
    labels = np.array([systems[key] for key in keys])
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    generator = np.random.default_rng(0)
    sampled = np.array(
        [
            np.median(
                difference[
                    np.concatenate(
                        [
                            clusters[index]
                            for index in generator.integers(
                                0, len(clusters), len(clusters)
                            )
                        ]
                    )
                ]
            )
            for _ in range(BOOTSTRAP_DRAWS)
        ]
    )
    return {
        "n_pairs": len(left),
        "n_systems": len(clusters),
        "median_delta": float(np.median(difference)),
        "delta_ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
    }


def main(argv=None) -> int:
    root = artifact_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=root / "results" / "boltz_hiqbind_unified",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--out",
        type=Path,
        default=root
        / "work"
        / "trunk_probe"
        / "distogram_locality_unified_primary163_s123.json",
    )
    arguments = parser.parse_args(argv)
    seeds = [int(seed) for seed in arguments.seeds.split(",")]

    pairs = tables.pairs("hiqbind")
    primary = pairing.primary_pairs(pairs)
    frame = pairing.restrict(probe.canonical_pair_frame(), primary)
    recorded = {seed: load(arguments.root, seed) for seed in seeds}
    if any(not records for records in recorded.values()):
        raise FileNotFoundError("one or more representation-capture seeds are missing")

    substituted = {}
    first = recorded[seeds[0]]
    for row in frame.itertuples():
        key = (str(row.pdb_a).lower(), str(row.pdb_b).lower(), row.het)
        reference = first.get((key[0], row.het))
        variant = first.get((key[1], row.het))
        if reference is None or variant is None:
            continue
        _, changed = probe.matched_positions(reference, variant)
        reference_keys = token_keys(reference)
        substituted[key] = {reference_keys[index] for index, _ in changed}

    per_comparison = {
        contrast: {f"cross/{name}": [] for name in STRATA}
        for contrast in ("mutation", "repeat")
    }
    per_pair = {"mutation": {}, "repeat": {}}
    systems = {
        (
            str(row.pdb_a).lower(),
            str(row.pdb_b).lower(),
            row.het,
        ): f"{row.uniprot}|{row.het}"
        for row in frame.itertuples()
    }
    for row in frame.itertuples():
        key = (str(row.pdb_a).lower(), str(row.pdb_b).lower(), row.het)
        if not substituted.get(key):
            continue
        jobs = [
            ("mutation", (key[0], row.het), (key[1], row.het), seed, seed)
            for seed in seeds
        ]
        jobs.extend(
            (
                "repeat",
                (side, row.het),
                (side, row.het),
                first_seed,
                second_seed,
            )
            for side in (key[0], key[1])
            for first_seed, second_seed in itertools.combinations(seeds, 2)
        )
        for contrast, left_key, right_key, left_seed, right_seed in jobs:
            left = recorded[left_seed].get(left_key)
            right = recorded[right_seed].get(right_key)
            if left is None or right is None:
                continue
            result = comparison(left, right, substituted[key])
            if result is None:
                continue
            for name, (changed, total, scope) in result.items():
                if not total:
                    continue
                rate = changed / total
                per_comparison[contrast][name].append((rate, changed, total, scope))
                per_pair[contrast].setdefault(key, {}).setdefault(name, []).append(rate)

    strata = {}
    for name in per_comparison["mutation"]:
        mutation = rate_statistics(per_comparison["mutation"][name])
        repeat = rate_statistics(per_comparison["repeat"][name])
        paired = paired_statistics(
            per_pair["mutation"], per_pair["repeat"], systems, name
        )
        strata[name] = {
            "mutation": mutation,
            "repeat": repeat,
            "paired": paired,
        }

    report = {
        "generated_by": "tools/trunk_probe/locality_decomposition.py",
        "root": "artifacts/results/boltz_hiqbind_unified",
        "seeds": seeds,
        "pair_set": {
            "cohort": "primary",
            "pairs": len(frame),
            "filters": list(primary.screens),
        },
        "head_pooling": "off-diagonal protein-ligand, ligand-protein, and ligand-ligand entries",
        "strata": strata,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
