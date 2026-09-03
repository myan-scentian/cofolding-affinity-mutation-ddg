"""Run AQAffinity for seeds 1--5 and collect its released scalar output."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from common import (
    ALIGNMENT_MAP,
    AQAFFINITY_BIN,
    BATCH,
    CHAIN_LIGAND,
    CHAIN_PROTEIN,
    HEAD_CKPT,
    JOBS_JSON,
    RAW_VALUES,
    RUNNER_YAML_EXTRA,
    RUNS,
    SEEDS,
    TRUNK_CKPT,
    venv_env,
)


def write_query_json(jobs: list[dict], alignments: dict, path: Path) -> None:
    queries = {}
    for job in jobs:
        bundle = alignments[job["seq"]]
        queries[job["name"]] = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": CHAIN_PROTEIN,
                    "sequence": job["seq"],
                    "main_msa_file_paths": bundle["main_msa_file_paths"],
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": CHAIN_LIGAND,
                    "smiles": job["smiles"],
                },
            ]
        }
    path.write_text(json.dumps({"queries": queries}, indent=2) + "\n")


def write_runner_yaml(path: Path, seed: int) -> None:
    path.write_text(
        "pl_trainer_args:\n"
        "  devices: 1\n"
        "  num_nodes: 1\n"
        "experiment_settings:\n"
        f"  seeds: [{seed}]\n"
        "data_module_args:\n"
        "  num_workers: 0\n" + RUNNER_YAML_EXTRA
    )


def run_batch(
    jobs: list[dict], alignments: dict, directory: Path, seed: int, timeout: int
) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "out"
    if output.exists():
        shutil.rmtree(output)
    write_query_json(jobs, alignments, directory / "query.json")
    write_runner_yaml(directory / "runner.yaml", seed)
    command = [
        str(AQAFFINITY_BIN),
        "predict",
        "--query_json",
        str(directory / "query.json"),
        "--inference_ckpt_path",
        str(TRUNK_CKPT),
        "--runner_yaml",
        str(directory / "runner.yaml"),
        "--use_msa_server",
        "false",
        "--use_templates",
        "false",
        "--output_dir",
        str(output),
        "--binding_affinity_ckpt_path",
        str(HEAD_CKPT),
    ]
    with (directory / "run.log").open("w") as log:
        completed = subprocess.run(
            command,
            cwd=directory,
            env=venv_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def harvest(jobs: list[dict], directory: Path, seed: int) -> dict[str, float]:
    values = {}
    for job in jobs:
        seed_directory = directory / "out" / job["name"] / f"seed_{seed}"
        files = sorted(seed_directory.glob("*_binding_head.txt"))
        if files:
            values[job["name"]] = float(files[0].read_text().strip())
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--jobs", type=Path, default=JOBS_JSON)
    parser.add_argument("--alignments", type=Path, default=ALIGNMENT_MAP)
    parser.add_argument("--out", type=Path, default=RAW_VALUES)
    arguments = parser.parse_args(argv)
    jobs = json.loads(arguments.jobs.read_text())["hiqbind"]
    alignments = json.loads(arguments.alignments.read_text())
    ready = [
        job for job in jobs if alignments.get(job["seq"], {}).get("main_msa_file_paths")
    ]
    collected = json.loads(arguments.out.read_text()) if arguments.out.exists() else {}
    started = time.time()
    for seed in SEEDS:
        pending = [
            job for job in ready if str(seed) not in collected.get(job["name"], {})
        ]
        for start in range(0, len(pending), arguments.batch):
            batch = pending[start : start + arguments.batch]
            directory = RUNS / f"seed_{seed}_{start:04d}"
            run_batch(batch, alignments, directory, seed, arguments.timeout)
            for name, value in harvest(batch, directory, seed).items():
                collected.setdefault(name, {})[str(seed)] = value
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            arguments.out.write_text(json.dumps(collected, indent=2) + "\n")
            print(
                f"seed {seed}: {start + len(batch)}/{len(pending)} in "
                f"{time.time() - started:.0f}s"
            )
    missing = [
        f"{job['name']}_seed-{seed}"
        for job in jobs
        for seed in SEEDS
        if str(seed) not in collected.get(job["name"], {})
    ]
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "run_hiqbind.json").write_text(
        json.dumps({"seeds": list(SEEDS), "missing": missing}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
