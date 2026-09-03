"""Average the fixed-width pooled probe vectors over five inference seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _artifact_root():
    """Artifact root: $MUTATION_DDG_ARTIFACT_ROOT, else <repo>/artifacts."""
    import os
    from pathlib import Path as _P

    override = os.environ.get("MUTATION_DDG_ARTIFACT_ROOT")
    if override:
        return _P(override)
    for parent in _P(__file__).resolve().parents:
        if (parent / "pixi.toml").exists():
            return parent / "artifacts"
    return _P.cwd() / "artifacts"


RESULTS = Path(str(_artifact_root() / "results/boltz_hiqbind_unified"))
WORK = Path(str(_artifact_root() / "work/trunk_probe"))

KEYS = (
    [f"z_aff1_block{i:02d}" for i in range(9)]
    + [f"z_aff2_block{i:02d}" for i in range(5)]
    + ["affinity_pred_value", "affinity_pred_value1", "affinity_pred_value2"]
)


def discover(results: Path) -> list[int]:
    """Seeds actually on disk, so a 5-seed and a 35-seed tree both just work."""
    return sorted(
        int(d.name[1:])
        for d in results.glob("s*")
        if d.is_dir() and d.name[1:].isdigit()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated subset; default is every seed present",
    )
    parser.add_argument(
        "--out", required=True, help="directory name under work/trunk_probe"
    )
    arguments = parser.parse_args(argv)

    seeds = (
        [int(s) for s in arguments.seeds.split(",")]
        if arguments.seeds
        else discover(arguments.results)
    )
    if not seeds:
        print(f"no seed directories under {arguments.results}")
        return 1
    destination = WORK / arguments.out
    destination.mkdir(parents=True, exist_ok=True)

    present = [
        {
            d.name
            for d in (arguments.results / f"s{s}" / "predictions").iterdir()
            if d.is_dir()
        }
        for s in seeds
    ]
    names = sorted(set.intersection(*present))
    dropped = sorted(set.union(*present) - set(names))
    print(
        f"seeds {seeds}; {len(names)} complexes at all of them"
        + (
            f"; {len(dropped)} dropped for uneven coverage: {dropped[:5]}"
            if dropped
            else ""
        )
    )

    for name in names:
        stack: dict[str, list] = {}
        for seed in seeds:
            path = (
                arguments.results
                / f"s{seed}"
                / "predictions"
                / name
                / f"representations_{name}.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                for key in KEYS:
                    if key in data.files:
                        stack.setdefault(key, []).append(
                            np.asarray(data[key], np.float64)
                        )
        np.savez_compressed(
            destination / f"{name}_s1.npz",
            **{k: np.mean(v, axis=0).astype(np.float32) for k, v in stack.items()},
        )
    print(f"wrote {len(names)} averaged records to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
