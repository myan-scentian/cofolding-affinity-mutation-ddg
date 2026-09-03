"""Fit the Boltz-2 linear probes separately at seeds 1--5."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common
import probe
from layer_analysis import MODULES, RT_KCAL, _out_of_fold, _prepare, load


def collect(records, frame, subset, blocks, control_key):
    """Fit one ensemble member on one seed's aligned primary-cohort records."""
    pairs, measured, proteins, head = [], [], [], []
    for row in frame.itertuples():
        key = (row.pdb_a, row.pdb_b, row.het)
        reference = records.get((str(row.pdb_a).lower(), row.het))
        variant = records.get((str(row.pdb_b).lower(), row.het))
        if key not in subset or reference is None or variant is None:
            continue
        pairs.append((reference, variant))
        measured.append(float(row.ddg_kcal))
        proteins.append(str(row.uniprot))
        head.append(RT_KCAL * float(variant[control_key] - reference[control_key]))
    measured = np.asarray(measured)
    proteins = np.asarray(proteins)
    head = np.asarray(head)
    curve = []
    control = []
    for block in blocks:
        features = np.asarray(
            [
                variant[block].astype(np.float64) - reference[block].astype(np.float64)
                for reference, variant in pairs
            ]
        )
        prepared = _prepare(probe, features, proteins)
        curve.append(probe.score(measured, _out_of_fold(prepared, measured))["pearson"])
        control.append(probe.score(head, _out_of_fold(prepared, head))["pearson"])
    return {
        "pairs": len(pairs),
        "curve": np.asarray(curve),
        "control": np.asarray(control),
    }


def section(number, header, seeds, per_seed, averaged, labels) -> list[str]:
    """Format the rows consumed by the manuscript figure."""
    matrix = np.asarray([per_seed[seed]["curve"] for seed in seeds])
    lines = [f"================ {header}, copy {number} ================"]
    for index, label in enumerate(labels):
        values = matrix[:, index]
        lines.append(
            f"{label} "
            + " ".join(f"{value:+.6f}" for value in values)
            + f" {values.mean():+.6f} {values.std(ddof=1):.6f} "
            + f"{np.ptp(values):.6f} {averaged['curve'][index]:+.6f}"
        )
    lines.append("")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", default="layers_unified")
    parser.add_argument("--averaged", default="layers_unified_mean")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--copies", default="1,2")
    parser.add_argument(
        "--pairs", type=Path, default=common.WORK / "primary_cohort_163.csv"
    )
    parser.add_argument("--out", type=Path, default=common.WORK / "seed_spread_163.log")
    arguments = parser.parse_args(argv)
    seeds = [int(seed) for seed in arguments.seeds.split(",")]
    copies = [int(copy) for copy in arguments.copies.split(",")]
    with arguments.pairs.open() as handle:
        subset = {
            (row["pdb_a"], row["pdb_b"], row["het"]) for row in csv.DictReader(handle)
        }
    frame = probe.canonical_pair_frame()
    per_seed = {number: {} for number in copies}
    blocks = {}
    for seed in seeds:
        records = load(arguments.representations, seed)
        for number in copies:
            prefix, control_key = MODULES[number]
            blocks.setdefault(
                number,
                sorted(
                    key
                    for key in next(iter(records.values()))
                    if key.startswith(prefix)
                ),
            )
            per_seed[number][seed] = collect(
                records, frame, subset, blocks[number], control_key
            )
    averaged_records = load(arguments.averaged, 1)
    averaged = {
        number: collect(
            averaged_records, frame, subset, blocks[number], MODULES[number][1]
        )
        for number in copies
    }
    counts = {
        result["pairs"]
        for number in copies
        for result in (*per_seed[number].values(), averaged[number])
    }
    if counts != {len(subset)}:
        raise RuntimeError(
            f"the probe fits do not all contain {len(subset)} pairs: {counts}"
        )
    header = f"{len(subset)} pairs (primary cohort)"
    lines = []
    for number in copies:
        labels = [block.replace(MODULES[number][0], "") for block in blocks[number]]
        lines.extend(
            section(number, header, seeds, per_seed[number], averaged[number], labels)
        )
    arguments.out.write_text("\n".join(lines))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
