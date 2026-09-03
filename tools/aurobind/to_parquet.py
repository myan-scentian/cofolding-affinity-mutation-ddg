"""Convert AuroBind pXC50-like scores to the common prediction scale."""

from __future__ import annotations

import json
import sys

import pandas as pd
from common import INPUTS, SEEDS, affinity_path

from mutation_ddg.paths import PREDICTIONS_ROOT


def main() -> int:
    manifest = pd.read_csv(INPUTS / "manifest_hiqbind.csv")
    rows = []
    missing = []
    for entry in manifest[manifest.status == "written"].itertuples():
        for seed in SEEDS:
            path = affinity_path(entry.record, seed)
            if not path.exists():
                missing.append(f"{entry.record}_seed-{seed}")
                continue
            raw = float(json.loads(path.read_text())["affinity"])
            rows.append(
                {
                    "pdb": str(entry.pdb).lower(),
                    "het": str(entry.het),
                    "seed": seed,
                    "nres": int(entry.nres),
                    "affinity_pred_value": 6.0 - raw,
                }
            )
    table = pd.DataFrame(rows).sort_values(["pdb", "het", "seed"])
    if table.duplicated(["pdb", "het", "seed"]).any():
        raise SystemExit("duplicate complex/seed rows")
    out = PREDICTIONS_ROOT / "aurobind_hiqbind.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    out.with_suffix(".provenance.json").write_text(
        json.dumps({"seeds": list(SEEDS), "missing": missing}, indent=2)
    )
    print(f"{out}: {len(table)} rows")
    if missing:
        print(f"missing {len(missing)} predictions; first entries: {missing[:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
