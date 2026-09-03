"""Shared paths and settings for AuroBind inference."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "matched_inputs"))

from mutation_ddg.paths import ARTIFACT_ROOT, RESULTS_ROOT

WORK = Path(os.environ.get("AUROBIND_WORK", ARTIFACT_ROOT / "work" / "aurobind"))
REPO = WORK / "repo"
SOURCE = REPO / "src"
PYTHON = WORK / ".venv" / "bin" / "python"
CACHE_DATA = WORK / "cache_data"
INPUTS = WORK / "inputs"
OUT = WORK / "out"
SEEDS = tuple(range(1, 6))
ALIGNMENT_CACHES = (RESULTS_ROOT / "msa_cache", WORK / "msa_extra")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


def sequence_key(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()[:16]


def cached_alignment(sequence: str) -> Path | None:
    key = sequence_key(sequence)
    for directory in ALIGNMENT_CACHES:
        for suffix in (".csv", ".a3m"):
            path = directory / f"{key}{suffix}"
            if path.exists() and path.stat().st_size:
                return path
    return None


def affinity_path(record: str, seed: int) -> Path:
    return (
        OUT / "hiqbind" / "predictions" / record / f"{record}_seed-{seed}_affinity.json"
    )


def environment(gpu: str) -> dict[str, str]:
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
