"""Collect Boltz-2 HiQBind outputs into the analysis prediction table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from run_boltz2 import load_complexes, safe_name

from mutation_ddg import config as configuration
from mutation_ddg.paths import PREDICTIONS_ROOT

SCHEMA = ["pdb", "het", "seed", "nres", "affinity_pred_value"]
REQUIRED_REPRESENTATIONS = [
    *(f"z_aff1_block{block:02d}" for block in range(9)),
    *(f"z_aff2_block{block:02d}" for block in range(5)),
]
ENSEMBLE_TOLERANCE = 1e-6


def planned_complexes(settings: dict) -> tuple[dict, dict]:
    """Map output-directory names to the planned HiQBind complexes."""
    pairs, plan, runnable, oversize = load_complexes("hiqbind", settings)
    keyed = {}
    for complex_ in plan.complexes:
        pdb, ligand = complex_.name.rsplit("_", 1)
        keyed[safe_name(complex_.name)] = (pdb.lower(), ligand, complex_.n_residues)
    report = {
        "pairs": len(pairs),
        "complexes_planned": len(plan.complexes),
        "complexes_runnable": len(runnable),
        "complexes_oversize": len(oversize),
        "pairs_refused": [
            {
                "reference": refusal.wild_type,
                "variant": refusal.mutant,
                "ligand": refusal.ligand,
                "reason": refusal.reason,
            }
            for refusal in plan.refusals
        ],
        "oversize": [complex_.name for complex_ in oversize],
    }
    return keyed, report


def check_instrumented(record: str, seed: int, values: dict, path: Path) -> None:
    """Verify that each probe record and released scalar came from one forward pass."""
    data = np.load(path, allow_pickle=False)
    missing = [key for key in REQUIRED_REPRESENTATIONS if key not in data.files]
    if missing:
        raise RuntimeError(f"{record} seed {seed} is missing {missing}")
    recorded = float(np.asarray(data["affinity_pred_value_exact"]).flat[0])
    released = float(values["affinity_pred_value"])
    if recorded != released:
        raise RuntimeError(
            f"{record} seed {seed}: representation scalar {recorded} != released {released}"
        )
    if int(np.asarray(data["prediction_match"]).flat[0]) != 1:
        raise RuntimeError(f"{record} seed {seed} failed its prediction identity check")


def collect(root: Path, seeds: int, instrumented: bool) -> tuple[pd.DataFrame, dict]:
    """Collect one row for every completed complex and seed."""
    settings = configuration.run("hiqbind")
    keyed, report = planned_complexes(settings)
    rows = []
    found: dict[str, set[int]] = {}
    for seed in range(1, seeds + 1):
        directory = root / f"s{seed}" / "predictions"
        if not directory.is_dir():
            continue
        for record_directory in sorted(directory.iterdir()):
            record = record_directory.name
            if not record_directory.is_dir() or record not in keyed:
                continue
            prediction = record_directory / f"affinity_{record}.json"
            if not prediction.exists():
                continue
            values = json.loads(prediction.read_text())
            if values.get("affinity_pred_value") is None:
                continue
            if instrumented:
                representation = record_directory / f"representations_{record}.npz"
                if not representation.exists():
                    raise RuntimeError(
                        f"{record} seed {seed} has no representation file"
                    )
                check_instrumented(record, seed, values, representation)
            if all(
                key in values
                for key in ("affinity_pred_value1", "affinity_pred_value2")
            ):
                ensemble_mean = (
                    float(values["affinity_pred_value1"])
                    + float(values["affinity_pred_value2"])
                ) / 2
                if (
                    abs(ensemble_mean - float(values["affinity_pred_value"]))
                    > ENSEMBLE_TOLERANCE
                ):
                    raise RuntimeError(f"{record} seed {seed}: ensemble mean mismatch")
            pdb, ligand, residues = keyed[record]
            row = {
                "pdb": pdb,
                "het": ligand,
                "seed": seed,
                "nres": residues,
                "affinity_pred_value": float(values["affinity_pred_value"]),
            }
            rows.append(row)
            found.setdefault(record, set()).add(seed)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table[SCHEMA].sort_values(["pdb", "het", "seed"]).reset_index(drop=True)
    report.update(
        {
            "source": "instrumented" if instrumented else "released_writer",
            "seeds": list(range(1, seeds + 1)),
            "rows": len(table),
            "complexes_with_any_seed": len(found),
            "complexes_with_all_seeds": sum(
                len(seed_set) == seeds for seed_set in found.values()
            ),
            "seed_counts": {
                str(seed): sum(seed in seed_set for seed_set in found.values())
                for seed in range(1, seeds + 1)
            },
            "settings": {
                "boltz_version": "2.2.1",
                "recycling_steps": settings["recycling_steps"],
                "structure_diffusion_samples": settings["diffusion_samples"],
                "affinity_diffusion_samples": settings["affinity_samples"],
            },
        }
    )
    return table, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["hiqbind"], default="hiqbind")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--from-unified", type=Path)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)

    settings = configuration.run(arguments.run)
    default_root = Path(str(configuration.resolve(settings, "results")) + "_v2")
    root = arguments.from_unified or arguments.results or default_root
    table, report = collect(root, arguments.seeds, arguments.from_unified is not None)
    output = arguments.out or PREDICTIONS_ROOT / "boltz2_v2_hiqbind.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"wrote {len(table)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
