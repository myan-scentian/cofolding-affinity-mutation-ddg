"""Fetch the AuroBind MSAs absent from the shared cache."""

from __future__ import annotations

import json
import subprocess
import sys

from common import INPUTS, PYTHON, SOURCE, WORK, environment, sequence_key

WORKER = """
import json
import pathlib
import sys
sys.path.insert(0, ".")
from aurobind.data.msa.mmseqs2 import run_mmseqs2
sequences = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
alignments = run_mmseqs2(sequences, prefix=sys.argv[3], use_env=True,
                         use_filter=True, use_pairing=False)
for sequence, alignment in zip(sequences, alignments):
    key = __import__("hashlib").md5(sequence.encode()).hexdigest()[:16]
    (out / f"{key}.a3m").write_text(alignment)
"""


def main() -> int:
    source = INPUTS / "missing_sequences_hiqbind.json"
    if not source.exists():
        raise SystemExit("run build_inputs.py first")
    wanted = json.loads(source.read_text())
    out = WORK / "msa_extra"
    scratch = WORK / "msa_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    todo = [
        sequence
        for sequence in wanted
        if not (out / f"{sequence_key(sequence)}.a3m").exists()
    ]
    failures = []
    for index in range(0, len(todo), 5):
        chunk = todo[index : index + 5]
        batch = scratch / f"batch_{index:04d}.json"
        batch.write_text(json.dumps(chunk))
        try:
            result = subprocess.run(
                [
                    str(PYTHON),
                    "-c",
                    WORKER,
                    str(batch),
                    str(out),
                    str(scratch / f"b{index}"),
                ],
                cwd=SOURCE,
                env=environment(""),
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.extend(sequence_key(sequence) for sequence in chunk)
            continue
        if result.returncode:
            failures.extend(sequence_key(sequence) for sequence in chunk)
    print(f"recovered {len(todo) - len(failures)}/{len(todo)} sequences")
    if failures:
        print(f"failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
