"""Precompute ColabFold MSAs for the AQAffinity HiQBind sequences."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import ALIGNMENT_MAP, JOBS_JSON, MSA_STORE
from openfold3.core.data.tools.colabfold_msa_server import (
    MsaComputationSettings,
    preprocess_colabfold_msas,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


def fetch(sequences: list[str]) -> dict[str, dict]:
    query_set = InferenceQuerySet(
        seeds=[1],
        queries={
            f"q{index}": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": "A",
                        "sequence": sequence,
                    }
                ]
            }
            for index, sequence in enumerate(sequences)
        },
    )
    raw = MSA_STORE / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    MSA_STORE.mkdir(parents=True, exist_ok=True)
    query_set = preprocess_colabfold_msas(
        inference_query_set=query_set,
        compute_settings=MsaComputationSettings(
            msa_output_directory=MSA_STORE,
            cleanup_msa_dir=False,
            msa_file_format="npz",
        ),
    )
    output = {}
    for index, sequence in enumerate(sequences):
        chain = query_set.queries[f"q{index}"].chains[0]
        output[sequence] = {
            "main_msa_file_paths": [
                str(path) for path in chain.main_msa_file_paths or []
            ]
        }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--jobs", type=Path, default=JOBS_JSON)
    parser.add_argument("--out", type=Path, default=ALIGNMENT_MAP)
    arguments = parser.parse_args(argv)
    jobs = json.loads(arguments.jobs.read_text())["hiqbind"]
    sequences = list(dict.fromkeys(job["seq"] for job in jobs))
    have = json.loads(arguments.out.read_text()) if arguments.out.exists() else {}
    missing = [
        sequence
        for sequence in sequences
        if not have.get(sequence, {}).get("main_msa_file_paths")
    ]
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), arguments.chunk):
        batch = missing[start : start + arguments.chunk]
        have.update(fetch(batch))
        arguments.out.write_text(json.dumps(have, indent=2) + "\n")
    complete = sum(
        bool(have.get(sequence, {}).get("main_msa_file_paths"))
        for sequence in sequences
    )
    print(f"{complete}/{len(sequences)} sequences have MSAs")
    return int(complete != len(sequences))


if __name__ == "__main__":
    raise SystemExit(main())
