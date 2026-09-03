"""Run the AuroBind affinity-only path for seeds 1--5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd
from common import (
    CACHE_DATA,
    INPUTS,
    OUT,
    PYTHON,
    SEEDS,
    SOURCE,
    affinity_path,
    environment,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args(argv)
    manifest = pd.read_csv(INPUTS / "manifest_hiqbind.csv")
    records = manifest.loc[manifest.status == "written", "record"].tolist()
    command = [
        str(PYTHON),
        "run_aurobind.py",
        str(INPUTS / "hiqbind"),
        "--out_dir",
        str(OUT),
        "--cache",
        str(CACHE_DATA),
        "--recycling_iters",
        "10",
        "--sampling_steps",
        "200",
        "--num_diffusion_samples",
        "1",
        "--num_workers",
        str(arguments.workers),
        "--affinity_only",
    ]
    returncodes = {}
    for seed in SEEDS:
        result = subprocess.run(
            command + ["--seed", str(seed)],
            cwd=SOURCE,
            env=environment(arguments.gpu),
            check=False,
        )
        returncodes[str(seed)] = result.returncode
    missing = {
        str(seed): [
            record for record in records if not affinity_path(record, seed).exists()
        ]
        for seed in SEEDS
    }
    report = {
        "seeds": list(SEEDS),
        "recycling_iterations": 10,
        "affinity_only": True,
        "records": len(records),
        "missing": missing,
        "returncodes": returncodes,
    }
    path = OUT / "run_hiqbind.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return 1 if any(missing.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
