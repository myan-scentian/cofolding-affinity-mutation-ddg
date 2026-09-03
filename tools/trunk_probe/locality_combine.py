"""Create the compact locality table consumed by the paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(name: str, stratum: dict) -> dict:
    paired = stratum["paired"]
    mutation = stratum["mutation"]
    repeat = stratum["repeat"]
    return {
        "stratum": name,
        "mutation_median": mutation["median"],
        "repeat_median": repeat["median"],
        "paired_median_delta_pp": 100 * paired["median_delta"],
        "paired_delta_ci95_pp": [100 * value for value in paired["delta_ci95"]],
        "paired_n_pairs": paired["n_pairs"],
        "paired_n_systems": paired["n_systems"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arms = {}
    for specification in arguments.arm:
        name, separator, source = specification.partition("=")
        if not separator:
            raise ValueError("--arm requires NAME=PATH")
        path = Path(source)
        arms[name] = (path, json.loads(path.read_text()))

    first = next(iter(arms.values()))[1]
    for name, (_, result) in arms.items():
        if result["seeds"] != first["seeds"]:
            raise ValueError(f"{name} used a different seed set")
        if result["pair_set"] != first["pair_set"]:
            raise ValueError(f"{name} used a different pair set")
    report = {
        "generated_by": "tools/trunk_probe/locality_combine.py",
        "inputs": {
            "seeds": first["seeds"],
            "pair_set": first["pair_set"],
            "per_arm_files": {name: path.name for name, (path, _) in arms.items()},
        },
        "head_pooling": first["head_pooling"],
        "table": {
            name: [row(key, block) for key, block in result["strata"].items()]
            for name, (_, result) in arms.items()
        },
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
