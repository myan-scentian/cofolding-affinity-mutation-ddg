"""Shared paths and settings for AQAffinity v0.0.1 with OpenFold3 v0.3.1."""

from __future__ import annotations

import os
from pathlib import Path

from mutation_ddg.paths import ARTIFACT_ROOT

WORK = Path(
    os.environ.get("AQAFFINITY_WORK", ARTIFACT_ROOT / "work" / "aqaffinity")
).expanduser()
REPO = Path(os.environ.get("AQAFFINITY_SOURCE", WORK / "repo")).expanduser()
OF3_CACHE = WORK / "of3_cache"
MSA_STORE = WORK / "msa_store"
JOBS_JSON = WORK / "jobs.json"
ALIGNMENT_MAP = WORK / "seq_alignments.json"
VENV = Path(os.environ.get("AQAFFINITY_VENV", WORK / ".venv-ft3")).expanduser()
TRUNK_CKPT = Path(
    os.environ.get("AQAFFINITY_TRUNK", OF3_CACHE / "of3_ft3_v1.pt")
).expanduser()
HEAD_CKPT = REPO / "model_weights" / "model_weights_only.safetensors"
RUNS = Path(os.environ.get("AQAFFINITY_RUNS", WORK / "runs")).expanduser()
RAW_VALUES = Path(
    os.environ.get("AQAFFINITY_RAW_VALUES", WORK / "raw_values.json")
).expanduser()
AQAFFINITY_BIN = VENV / "bin" / "aqaffinity"
GPU = os.environ.get("AQAFFINITY_GPU", "0")
SEEDS = tuple(range(1, 6))
BATCH = 12
CHAIN_PROTEIN = "A"
CHAIN_LIGAND = "B"
RUNNER_YAML_EXTRA = (
    "settings:\n  memory:\n    eval:\n      use_deepspeed_evo_attention: false\n"
)


def venv_env() -> dict:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = GPU
    environment["OPENFOLD_CACHE"] = str(OF3_CACHE)
    environment["PATH"] = f"{VENV / 'bin'}:{environment.get('PATH', '')}"
    return environment
