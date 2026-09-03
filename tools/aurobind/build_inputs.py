"""Build AuroBind inputs for the matched HiQBind complexes."""

from __future__ import annotations

import json
import sys

import pandas as pd
import yaml
from common import INPUTS, cached_alignment, safe_name, sequence_key
from run_boltz2 import load_complexes

from mutation_ddg import config as configuration
from mutation_ddg.predict.boltz2 import CHAIN_IDS


def write_input(path, complex_, alignments) -> None:
    sequences = [
        {
            "protein": {
                "id": CHAIN_IDS[index],
                "sequence": sequence,
                "msa": str(alignments[index]),
            }
        }
        for index, sequence in enumerate(complex_.chains)
    ]
    sequences.append(
        {
            "ligand": {
                "id": CHAIN_IDS[len(complex_.chains)],
                "smiles": complex_.ligand_smiles,
            }
        }
    )
    path.write_text(
        yaml.safe_dump({"version": 1, "sequences": sequences}, sort_keys=False)
    )


def main() -> int:
    settings = configuration.run("hiqbind")
    _, planned, runnable, oversize = load_complexes("hiqbind", settings)
    allowed = {complex_.name for complex_ in runnable}
    directory = INPUTS / "hiqbind"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = {}
    for complex_ in planned.complexes:
        pdb, ligand = complex_.name.rsplit("_", 1)
        record = safe_name(complex_.name)
        alignments = [cached_alignment(sequence) for sequence in complex_.chains]
        status = "written"
        reason = ""
        if complex_.name not in allowed:
            status = "oversize"
            reason = f"over the {settings['max_residues']}-residue ceiling"
        elif any(path is None for path in alignments):
            status = "no_alignment"
            reason = "no cached MSA for a chain"
            for sequence, path in zip(complex_.chains, alignments):
                if path is None:
                    missing[sequence_key(sequence)] = sequence
        else:
            write_input(directory / f"{record}.yaml", complex_, alignments)
        rows.append(
            {
                "pdb": pdb.lower(),
                "het": ligand,
                "record": record,
                "nres": complex_.n_residues,
                "chains": len(complex_.chains),
                "status": status,
                "reason": reason,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["pdb", "het"])
    manifest.to_csv(INPUTS / "manifest_hiqbind.csv", index=False)
    (INPUTS / "missing_sequences_hiqbind.json").write_text(
        json.dumps(sorted(missing.values()), indent=1)
    )
    print(manifest.status.value_counts().to_string())
    print(f"pair refusals: {len(planned.refusals)}; oversize: {len(oversize)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
