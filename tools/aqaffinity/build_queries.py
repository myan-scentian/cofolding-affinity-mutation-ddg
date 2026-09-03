"""Build the single-chain AQAffinity queries for the HiQBind complexes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "matched_inputs"))

from common import JOBS_JSON
from run_boltz2 import load_complexes

from mutation_ddg import config as configuration
from mutation_ddg.io import tables

MATCHED = "matched_single_chain"
LIGAND_CHAIN = "ligand_bearing_chain"


def matched_constructs(settings: dict) -> dict[tuple[str, str], list[str]]:
    """Return the pair-matched construct planned for each complex."""
    _, plan, _, _ = load_complexes("hiqbind", settings)
    output = {}
    for complex_ in plan.complexes:
        pdb, ligand = complex_.name.rsplit("_", 1)
        output[(pdb, ligand)] = list(complex_.chains)
    return output


def read_seqres(path: Path, pdb_ids: set[str]) -> dict[str, dict[str, str]]:
    """Read protein SEQRES records by PDB and chain identifier."""
    wanted = {pdb.lower() for pdb in pdb_ids}
    output: dict[str, dict[str, str]] = {}
    keep = False
    pdb = None
    chain = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                tag = line[1:].split()[0]
                pdb = tag.split("_")[0].lower()
                chain = tag.split("_", 1)[1] if "_" in tag else "A"
                keep = "mol:protein" in line and pdb in wanted
            elif keep:
                output.setdefault(pdb, {}).setdefault(chain, "")
                output[pdb][chain] += line.strip()
    return output


def ligand_chains(settings: dict) -> dict[tuple[str, str], str]:
    """Read ligand-bearing chains from the included site table."""
    output = {}
    with configuration.resolve(settings, "ligand_sites").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                continue
            fields = row["site"].split("_")
            key = (row["pdb"].lower(), row["het"])
            if len(fields) < 4 or fields[:2] != list(key):
                raise ValueError(f"malformed ligand site {row['site']!r}")
            output[key] = fields[2]
    return output


def build() -> tuple[list[dict], list[dict]]:
    settings = configuration.run("hiqbind")
    pairs = tables.pairs("hiqbind")
    wanted = {}
    for row in pairs.itertuples():
        for pdb, smiles in ((row.pdb_a, row.smiles_a), (row.pdb_b, row.smiles_b)):
            if isinstance(smiles, str) and smiles.strip():
                wanted.setdefault((str(pdb).lower(), str(row.het)), smiles)
    seqres = read_seqres(
        configuration.resolve(settings, "sequences"), {pdb for pdb, _ in wanted}
    )
    planned = matched_constructs(settings)
    sites = ligand_chains(settings)
    jobs = []
    refused = []
    for (pdb, ligand), smiles in sorted(wanted.items()):
        construct = planned.get((pdb, ligand))
        if construct is not None and len(construct) == 1:
            sequence = construct[0]
            source = MATCHED
            chain = None
        else:
            available = seqres.get(pdb, {})
            chain = sites.get((pdb, ligand))
            if chain not in available:
                refused.append({"pdb": pdb, "het": ligand, "reason": "no_ligand_chain"})
                continue
            sequence = available[chain]
            source = LIGAND_CHAIN
        jobs.append(
            {
                "name": f"{pdb}__{ligand}",
                "pdb": pdb,
                "het": ligand,
                "chain": chain,
                "construct": source,
                "seq": sequence,
                "smiles": smiles,
                "nres": len(sequence),
            }
        )
    return jobs, refused


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=JOBS_JSON)
    arguments = parser.parse_args(argv)
    jobs, refused = build()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps({"hiqbind": jobs, "refused": refused}, indent=2) + "\n"
    )
    print(f"wrote {len(jobs)} queries to {arguments.out}; {len(refused)} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
