"""Convert AQAffinity outputs to the common per-complex prediction table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import JOBS_JSON, RAW_VALUES, SEEDS

from mutation_ddg.paths import PREDICTIONS_ROOT


def build(jobs: list[dict], values: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        for seed, raw in sorted(
            values.get(job["name"], {}).items(), key=lambda item: int(item[0])
        ):
            rows.append(
                {
                    "pdb": job["pdb"].lower(),
                    "het": job["het"],
                    "seed": int(seed),
                    "nres": job["nres"],
                    "affinity_pred_value": -float(raw),
                }
            )
    return pd.DataFrame(rows).sort_values(["pdb", "het", "seed"]).reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=JOBS_JSON)
    parser.add_argument("--raw", type=Path, default=RAW_VALUES)
    parser.add_argument(
        "--out", type=Path, default=PREDICTIONS_ROOT / "aqaffinity_v2_hiqbind.parquet"
    )
    arguments = parser.parse_args(argv)
    jobs = json.loads(arguments.jobs.read_text())["hiqbind"]
    values = json.loads(arguments.raw.read_text())
    table = build(jobs, values)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(arguments.out, index=False)
    missing = [
        f"{job['name']}_seed-{seed}"
        for job in jobs
        for seed in SEEDS
        if str(seed) not in values.get(job["name"], {})
    ]
    arguments.out.with_suffix(".provenance.json").write_text(
        json.dumps({"seeds": list(SEEDS), "missing": missing}, indent=2)
    )
    print(f"wrote {len(table)} rows to {arguments.out}")
    if missing:
        print(f"missing {len(missing)} predictions; first entries: {missing[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
