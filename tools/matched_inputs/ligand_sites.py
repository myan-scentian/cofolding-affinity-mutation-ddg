"""How many copies of a chain line each HiQBind ligand site."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mutation_ddg import config as configuration
from mutation_ddg.datasets.jobs import read_seqres_chains
from mutation_ddg.paths import PROCESSED_ROOT

CONTACT_RADIUS = 4.5
MIN_RESIDUES = 3


def heavy_atoms(path: Path, record: str) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Coordinates and (chain, residue) labels for one record type."""
    points, labels = [], []
    for line in path.read_text().splitlines():
        if line[:6].strip() != record or line[76:78].strip() == "H":
            continue
        try:
            points.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        labels.append((line[21], line[22:27].strip()))
    return np.array(points).reshape(-1, 3), labels


def site_chains(directory: Path) -> dict[str, int] | None:
    """Residues per chain lining the ligand in one curated site directory."""
    ligand = directory / f"{directory.name}_ligand.pdb"
    protein = directory / f"{directory.name}_protein.pdb"
    if not ligand.exists() or not protein.exists():
        return None
    ligand_points, _ = heavy_atoms(ligand, "HETATM")
    if not len(ligand_points):
        ligand_points, _ = heavy_atoms(ligand, "ATOM")
    protein_points, protein_labels = heavy_atoms(protein, "ATOM")
    if not len(ligand_points) or not len(protein_points):
        return None
    tree = cKDTree(protein_points)
    residues: dict[str, set[str]] = defaultdict(set)
    for neighbours in tree.query_ball_point(ligand_points, r=CONTACT_RADIUS):
        for index in neighbours:
            chain, residue = protein_labels[index]
            residues[chain].add(residue)
    return {chain: len(found) for chain, found in residues.items()}


def copies_for(counts: dict[str, int], chain_sequences: dict[str, str]) -> int:
    """Count equivalent chain copies represented at the ligand site."""
    lining = [chain for chain, n in counts.items() if n >= MIN_RESIDUES]
    per_sequence: dict[str, int] = defaultdict(int)
    for chain in lining:
        per_sequence[chain_sequences.get(chain, chain)] += 1
    return max(per_sequence.values(), default=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument(
        "--pairs", type=Path, default=PROCESSED_ROOT / "ddg_bench_v2.parquet"
    )
    parser.add_argument(
        "--seqres",
        type=Path,
        default=configuration.resolve(configuration.run("hiqbind"), "sequences"),
    )
    parser.add_argument(
        "--out", default=str(PROCESSED_ROOT / "hiqbind_ligand_site_chains.csv")
    )
    arguments = parser.parse_args(argv)

    pairs = (
        pd.read_parquet(arguments.pairs)
        if arguments.pairs.suffix == ".parquet"
        else pd.read_csv(arguments.pairs)
    )
    wanted = sorted(
        {(str(row.pdb_a).lower(), row.het) for row in pairs.itertuples()}
        | {(str(row.pdb_b).lower(), row.het) for row in pairs.itertuples()}
    )
    by_chain = read_seqres_chains(arguments.seqres, {pdb for pdb, _ in wanted})

    rows = []
    for pdb, het in wanted:
        entry = arguments.sites / pdb
        candidates = (
            sorted(
                d for d in entry.iterdir() if d.is_dir() and d.name.split("_")[1] == het
            )
            if entry.is_dir()
            else []
        )
        best, best_counts = None, None
        for directory in candidates:
            counts = site_chains(directory)
            if counts and (
                best_counts is None or sum(counts.values()) > sum(best_counts.values())
            ):
                best, best_counts = directory.name, counts
        if best_counts is None:
            rows.append(
                {
                    "pdb": pdb,
                    "het": het,
                    "status": "no_site",
                    "site": None,
                    "copies": None,
                    "detail": None,
                }
            )
            continue
        rows.append(
            {
                "pdb": pdb,
                "het": het,
                "status": "ok",
                "site": best,
                "copies": copies_for(best_counts, by_chain.get(pdb, {})),
                "detail": ";".join(f"{c}:{n}" for c, n in sorted(best_counts.items())),
            }
        )

    table = pd.DataFrame(rows)
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.out, index=False)
    print(f"{len(table)} entries -> {arguments.out}")
    print(table.status.value_counts().to_string())
    resolved = table[table.status == "ok"]
    print("copies at the ligand site:")
    print(resolved.copies.value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
