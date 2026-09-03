"""Run Boltz-2 on the matched HiQBind inputs for seeds 1--5."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mutation_ddg import config as configuration
from mutation_ddg.datasets import jobs as jobs_module
from mutation_ddg.paths import RESULTS_ROOT
from mutation_ddg.predict.boltz2 import CHAIN_IDS, sequence_key

SHARED_CACHE = RESULTS_ROOT / "msa_cache"
DEFAULT_EXECUTABLE = "boltz"


INSTRUMENTED = Path(__file__).resolve().parents[1] / "trunk_probe" / "unified.py"


def load_complexes(run: str, settings: dict):
    """The matched-construct complexes for one run, plus what it refused."""
    pair_path = configuration.resolve(settings, "pairs")
    pairs = (
        pd.read_parquet(pair_path)
        if pair_path.suffix == ".parquet"
        else pd.read_csv(pair_path)
    )
    sequence_file = configuration.resolve(settings, "sequences")
    ids = set()
    for column in settings["pdb_columns"]:
        ids |= set(pairs[column].dropna().astype(str).str.lower())
    sequences = jobs_module.read_seqres(sequence_file, ids)
    made = jobs_module.plan(pairs, settings, sequences)
    runnable, oversize = jobs_module.partition(made.complexes, settings["max_residues"])
    return pairs, made, runnable, oversize


def safe_name(name: str) -> str:
    """A record id safe as a directory name. Some ligand codes carry a separator."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


class Alignments:
    """A per-process view of the shared alignment cache."""

    def __init__(self, private: Path):
        self.private = private
        self.private.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        SHARED_CACHE.mkdir(parents=True, exist_ok=True)
        for found in SHARED_CACHE.glob("*.csv"):
            link = self.private / found.name
            if not link.exists():
                try:
                    link.symlink_to(found)
                except (FileExistsError, OSError):
                    pass

    def path(self, sequence: str) -> Path | None:
        found = self.private / f"{sequence_key(sequence)}.csv"
        return found if found.exists() and found.stat().st_size > 0 else None

    def harvest(self, run_directory: Path, records: dict[str, list[str]]) -> int:
        """Keep the alignments this chunk fetched, then publish them."""
        stored = 0
        for found in run_directory.glob("**/msa/*.csv"):
            match = re.match(r"^(.*)_(\d+)\.csv$", found.name)
            if not match:
                continue
            chains = records.get(match.group(1))
            index = int(match.group(2))
            if chains is None or index >= len(chains):
                continue
            destination = self.private / f"{sequence_key(chains[index])}.csv"
            if destination.exists():
                continue
            shutil.copyfile(found, destination)
            stored += 1
        self.publish()
        return stored

    def publish(self) -> int:
        """Copy anything this process fetched into the shared cache, atomically."""
        published = 0
        for found in self.private.glob("*.csv"):
            if found.is_symlink():
                continue
            destination = SHARED_CACHE / found.name
            if destination.exists():
                continue
            staging = SHARED_CACHE / f".{found.name}.{os.getpid()}"
            staging.write_bytes(found.read_bytes())
            os.replace(staging, destination)
            published += 1
        return published


def write_input(path: Path, complex_, alignments: Alignments) -> None:
    """Write one Boltz input, using cached MSAs only when every chain has one."""
    cached = [alignments.path(sequence) for sequence in complex_.chains]
    if not all(found is not None for found in cached):
        cached = [None] * len(complex_.chains)
    lines = ["version: 1", "sequences:"]
    for index, sequence in enumerate(complex_.chains):
        lines += [
            "  - protein:",
            f"      id: {CHAIN_IDS[index]}",
            f"      sequence: {sequence}",
        ]
        if cached[index] is not None:
            lines.append(f"      msa: {cached[index]}")
    binder = CHAIN_IDS[len(complex_.chains)]
    lines += [
        "  - ligand:",
        f"      id: {binder}",
        f"      smiles: '{complex_.ligand_smiles}'",
        "properties:",
        "  - affinity:",
        f"      binder: {binder}",
    ]
    path.write_text("\n".join(lines) + "\n")


def affinity_path(root: Path, seed: int, record: str) -> Path:
    """Where this runner keeps the affinity for one complex at one seed."""
    return root / f"s{seed}" / "predictions" / record / f"affinity_{record}.json"


def structure_path(root: Path, seed: int, record: str) -> Path:
    """The structure stage's own output, which the affinity stage reads back."""
    return root / f"s{seed}" / "predictions" / record / f"pre_affinity_{record}.npz"


def representation_path(root: Path, seed: int, record: str) -> Path:
    """What `tools/trunk_probe/unified.py` writes beside the affinity JSON."""
    return root / f"s{seed}" / "predictions" / record / f"representations_{record}.npz"


def settings_provenance(settings: dict) -> dict:
    """Record the inference settings that determine the reported predictions."""
    return {
        "structure_recycling_steps": int(settings["recycling_steps"]),
        "affinity_recycling_steps": 5,
        "structure_diffusion_samples": int(settings.get("diffusion_samples", 1)),
        "affinity_diffusion_samples": int(settings.get("affinity_samples", 5)),
        "num_workers": int(settings.get("num_workers", 0)),
        "preprocessing_threads": int(settings.get("preprocessing_threads", 4)),
        "no_kernels": bool(settings.get("no_kernels", True)),
    }


def mismatch_path(root: Path, seed: int, record: str) -> Path:
    """The sentinel `unified.py` leaves when a value failed its own equality check."""
    return root / f"s{seed}" / "predictions" / record / f"MISMATCH_{record}.json"


def boltz_interpreter(executable: str) -> str:
    """Resolve the Python interpreter from the Boltz console-script shebang."""
    with open(executable, "rb") as handle:
        first = handle.readline().decode("utf-8", "replace").strip()
    if first.startswith("#!"):
        return first[2:].strip().split()[0]
    msg = f"{executable} has no shebang; cannot find the interpreter boltz runs under"
    raise RuntimeError(msg)


class Runner:
    def __init__(
        self, complexes, root, settings, gpu, chunk, worker, instrument: bool = False
    ):
        self.complexes = {safe_name(c.name): c for c in complexes}
        self.root = root
        self.settings = settings
        self.gpu = gpu
        self.chunk = chunk
        self.worker = worker
        self.instrument = instrument
        self.executable = os.path.expanduser(
            settings.get("executable") or DEFAULT_EXECUTABLE
        )
        self.alignments = Alignments(root / f"msa_worker_{worker}")
        self.started = time.time()

    def done(self, seed: int, record: str) -> bool:
        """Return whether expected scalar and optional representation outputs exist."""
        if affinity_path(self.root, seed, record).exists():
            if self.instrument:
                return (
                    representation_path(self.root, seed, record).exists()
                    and not mismatch_path(self.root, seed, record).exists()
                )
            return True
        return False

    def missing(self, seed: int) -> list[str]:
        return [
            record for record in sorted(self.complexes) if not self.done(seed, record)
        ]

    def fold(self, records: list[str], seed: int, label: str) -> int:
        """One Boltz-2 invocation over a chunk. Returns how many came back."""

        work = self.root / f"s{seed}" / "work" / f"w{self.worker}_{label}"
        if work.exists():
            shutil.rmtree(work)
        inputs = work / label
        inputs.mkdir(parents=True, exist_ok=True)
        for record in records:
            write_input(
                inputs / f"{record}.yaml", self.complexes[record], self.alignments
            )

        if self.instrument:
            head = [boltz_interpreter(self.executable), str(INSTRUMENTED)]
        else:
            head = [self.executable]
        command = [
            *head,
            "predict",
            str(inputs),
            "--out_dir",
            str(work),
            "--diffusion_samples",
            str(self.settings.get("diffusion_samples", 1)),
            "--diffusion_samples_affinity",
            str(self.settings.get("affinity_samples", 5)),
            "--recycling_steps",
            str(self.settings["recycling_steps"]),
            "--no_kernels",
            "--seed",
            str(seed),
            "--use_msa_server",
            "--num_workers",
            str(self.settings.get("num_workers", 0)),
            "--preprocessing-threads",
            str(self.settings.get("preprocessing_threads", 4)),
        ]
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=self.gpu,
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )

        budget = 600 + 300 * len(records)
        finished = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=budget,
            check=False,
        )
        output = (finished.stdout or "")[-20000:] + (finished.stderr or "")[-20000:]
        (work / "boltz.log").write_text(output)

        if finished.returncode:
            note = {
                2: "PREDICTION MISMATCH",
                3: "record recorded but unwritten",
                4: "boltz scored but nothing was instrumented",
                5: "record count does not reconcile",
            }.get(finished.returncode, "")
            print(
                f"  seed {seed} {label}: boltz exited {finished.returncode}"
                f"{' -- ' + note if note else ''}",
                flush=True,
            )
            for line in output.strip().splitlines()[-12:]:
                print(f"    | {line}", flush=True)

        self.alignments.harvest(
            work, {record: self.complexes[record].chains for record in records}
        )

        landed = 0
        destination_root = self.root / f"s{seed}" / "predictions"
        destination_root.mkdir(parents=True, exist_ok=True)
        for produced in work.glob("**/predictions/*"):
            if not produced.is_dir():
                continue
            destination = destination_root / produced.name
            if destination.exists():
                if self.done(seed, produced.name):
                    shutil.rmtree(produced)
                    continue
                shutil.rmtree(destination)
            shutil.move(str(produced), str(destination))
            landed += 1

        for leftover in work.iterdir():
            if leftover.name != "boltz.log":
                shutil.rmtree(leftover, ignore_errors=True)
        return landed

    def pass_over(self, seed: int, chunk: int, label: str) -> list[str]:
        """Fold everything still missing at this seed, in chunks of `chunk`."""
        pending = self.missing(seed)
        for start in range(0, len(pending), chunk):
            batch = pending[start : start + chunk]
            began = time.time()
            try:
                landed = self.fold(batch, seed, f"{label}_{start:05d}")
            except subprocess.TimeoutExpired:
                landed = 0
                print(f"  seed {seed} {label} {start}: timed out", flush=True)
            done = len(self.complexes) - len(self.missing(seed))
            print(
                f"  seed {seed} {label}: {start + len(batch)}/{len(pending)} sent, "
                f"{landed} back in {time.time() - began:.0f}s, "
                f"{done}/{len(self.complexes)} complete "
                f"(elapsed {(time.time() - self.started) / 60:.0f}m)",
                flush=True,
            )
        return self.missing(seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a name in configuration/datasets")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        help="index of this process, so its alignment cache is its own",
    )
    parser.add_argument(
        "--shard", default="0/1", help="i/n: fold only every n-th complex, offset i"
    )
    parser.add_argument("--chunk", type=int, default=48)
    parser.add_argument("--results", default=None)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="run boltz through tools/trunk_probe/unified.py, which "
        "installs hooks on the affinity model and writes the "
        "representations that produced each affinity value beside "
        "the value itself. Nothing is replayed afterwards, so no "
        "settings can diverge between the two.",
    )
    arguments = parser.parse_args(argv)

    settings = configuration.run(arguments.run)
    pairs, made, runnable, oversize = load_complexes(arguments.run, settings)

    planned = len(runnable)

    root = (
        Path(arguments.results)
        if arguments.results
        else Path(str(configuration.resolve(settings, "results")) + "_v2")
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest = json.dumps(
        {
            "run": arguments.run,
            "pairs": len(pairs),
            "complexes": [
                {
                    "name": c.name,
                    "n_residues": c.n_residues,
                    "chains": len(c.chains),
                    "lengths": [len(x) for x in c.chains],
                }
                for c in made.complexes
            ],
            "oversize": [c.name for c in oversize],
            "refusals": [
                {
                    "wild_type": r.wild_type,
                    "mutant": r.mutant,
                    "ligand": r.ligand,
                    "reason": r.reason,
                }
                for r in made.refusals
            ],
            "seeds": arguments.seeds,
            "recycling_steps": int(settings["recycling_steps"]),
            "diffusion_samples": int(settings.get("diffusion_samples", 1)),
            "affinity_samples": int(settings.get("affinity_samples", 5)),
            "max_residues": int(settings["max_residues"]),
            "instrument": bool(arguments.instrument),
            "settings_provenance": settings_provenance(settings),
        },
        indent=2,
    )
    if arguments.worker == 0:
        (root.parent / f"{root.name}_manifest.json").write_text(manifest)

        (root / "manifest.json").write_text(manifest)

    index, count = (int(x) for x in arguments.shard.split("/"))

    mine = jobs_module.shard(sorted(runnable, key=lambda c: c.name), index, count)
    print(
        f"{arguments.run}: {len(pairs)} pairs, {len(made.complexes)} complexes, "
        f"{len(runnable)} runnable of {planned} planned, "
        f"{len(oversize)} oversize, "
        f"{len(made.refusals)} pairs refused | shard {arguments.shard} takes "
        f"{len(mine)} | seeds {arguments.first_seed}..{arguments.seeds} | "
        f"recycling {settings['recycling_steps']} | GPU {arguments.gpu}",
        flush=True,
    )

    runner = Runner(
        mine,
        root,
        settings,
        arguments.gpu,
        arguments.chunk,
        arguments.worker,
        instrument=arguments.instrument,
    )
    for seed in range(arguments.first_seed, arguments.seeds + 1):
        runner.alignments.refresh()
        left = runner.pass_over(seed, arguments.chunk, "pass1")

        if left:
            left = runner.pass_over(seed, max(1, arguments.chunk // 6), "pass2")
        if left:
            left = runner.pass_over(seed, 1, "pass3")
        print(
            f"seed {seed} done: {len(runner.complexes) - len(left)}/"
            f"{len(runner.complexes)} complete, {len(left)} lost: {left[:20]}",
            flush=True,
        )
    runner.alignments.publish()

    if arguments.instrument:
        sentinels = sorted(root.glob("s*/predictions/*/MISMATCH_*.json"))
        kept = sorted(root.glob("s*/predictions/*/representations_*.npz"))
        print(
            f"instrumented: {len(kept)} records kept, "
            f"{len(sentinels)} prediction mismatches",
            flush=True,
        )
        if sentinels:
            for found in sentinels[:10]:
                print(
                    f"  MISMATCH {found.parent.name}: {found.read_text()[:400]}",
                    flush=True,
                )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
