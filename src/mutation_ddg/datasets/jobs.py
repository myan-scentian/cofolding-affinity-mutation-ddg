"""Build matched protein constructs for the HiQBind mutation pairs."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mutation_ddg.predict.boltz2 import Complex

MIN_CORE_FRACTION = 0.9
MIN_OVERLAP_FRACTION = 0.6


@dataclass(frozen=True)
class Refusal:
    wild_type: str
    mutant: str
    ligand: str
    reason: str


@dataclass(frozen=True)
class Plan:
    complexes: list[Complex]
    refusals: list[Refusal]


def _lines(path: str | Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        yield from handle


def read_seqres_chains(path: str | Path, pdb_ids) -> dict[str, dict[str, str]]:
    wanted = {str(pdb).lower() for pdb in pdb_ids}
    found: dict[str, dict[str, str]] = {}
    keep = False
    code = None
    chain = None
    for line in _lines(path):
        if line.startswith(">"):
            name = line[1:].split()[0]
            code = name.split("_")[0].lower()
            chain = name.split("_", 1)[1] if "_" in name else "A"
            keep = "mol:protein" in line and code in wanted
        elif keep and code is not None and chain is not None:
            current = found.setdefault(code, {}).get(chain, "")
            found[code][chain] = current + line.strip()
    return found


def read_seqres(path: str | Path, pdb_ids) -> dict[str, list[str]]:
    return {
        entry: list(chains.values())
        for entry, chains in read_seqres_chains(path, pdb_ids).items()
    }


def read_site_copies(path: str | Path) -> dict[tuple[str, str], int]:
    table = pd.read_csv(path)
    return {
        (str(row.pdb).lower(), str(row.het)): int(row.copies)
        for row in table.itertuples()
        if pd.notna(row.copies)
    }


def distinct(chains) -> list[str]:
    output = []
    for chain in chains:
        if chain not in output:
            output.append(chain)
    return output


def _aligner():
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    for names in (
        ("end_insertion_score", "target_end_gap_score"),
        ("end_deletion_score", "query_end_gap_score"),
    ):
        for name in names:
            try:
                setattr(aligner, name, 0.0)
                break
            except AttributeError:
                continue
    return aligner


def aligned_span(reference: str, other: str) -> tuple[int, int, int, int] | None:
    if reference == other:
        return 0, len(reference), 0, len(other)
    blocks = _aligner().align(reference, other)[0].aligned
    if len(blocks[0]) != 1:
        return None
    (reference_start, reference_end), (other_start, other_end) = (
        blocks[0][0],
        blocks[1][0],
    )
    return int(reference_start), int(reference_end), int(other_start), int(other_end)


def _alignable(reference: str, sequences: dict[str, str]) -> dict[str, tuple]:
    spans = {}
    for key, sequence in sequences.items():
        span = aligned_span(reference, sequence)
        if span is None:
            continue
        covered = span[1] - span[0]
        shorter, longer = sorted((len(reference), len(sequence)))
        if (
            covered >= MIN_CORE_FRACTION * shorter
            and covered >= MIN_OVERLAP_FRACTION * longer
        ):
            spans[key] = span
    return spans


def common_construct(sequences: dict[str, str]) -> dict[str, str]:
    if len({len(sequence) for sequence in sequences.values()}) == 1:
        return dict(sequences)
    best = None
    for candidate in sorted(sequences, key=lambda key: (len(sequences[key]), key)):
        spans = _alignable(sequences[candidate], sequences)
        low = max(span[0] for span in spans.values())
        high = min(span[1] for span in spans.values())
        if high <= low:
            continue
        score = len(spans), high - low
        if best is None or score > best[0]:
            best = score, spans, low, high
    if best is None:
        return dict(sequences)
    _, spans, low, high = best
    trimmed = dict(sequences)
    for key, (reference_start, _, other_start, _) in spans.items():
        offset = other_start - reference_start
        trimmed[key] = sequences[key][low + offset : high + offset]
    return trimmed


def _order_like(reference: list[str], chains: list[str]) -> list[str] | None:
    if len(reference) != len(chains):
        return None
    if len(chains) == 1:
        return list(chains)
    aligner = _aligner()
    remaining = list(chains)
    ordered = []
    for wanted in reference:
        best = max(
            remaining,
            key=lambda candidate: (aligner.score(wanted, candidate), candidate),
        )
        remaining.remove(best)
        ordered.append(best)
    return ordered


def components(edges: list[tuple]) -> list[list]:
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    order = []
    for left, right in edges:
        for node in (left, right):
            if node not in parent:
                parent[node] = node
                order.append(node)
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
    grouped = {}
    for node in order:
        grouped.setdefault(find(node), []).append(node)
    return list(grouped.values())


def _expected_substitutions(row, config: dict) -> int | None:
    text = row.get(config.get("mutations_column", "mutations"))
    if isinstance(text, str) and text.strip():
        return len([token for token in text.split(";") if token.strip()])
    count = row.get("n_sub")
    return int(count) if pd.notna(count) else None


def _substitutions(left: list[str], right: list[str]) -> int:
    return sum(
        first != second
        for chain_left, chain_right in zip(distinct(left), distinct(right))
        for first, second in zip(chain_left, chain_right)
    )


def _build_component(component, sequences, sites) -> dict[tuple[str, str], list[str]]:
    protomers = {node: distinct(sequences[node[0]]) for node in component}
    reference = protomers[min(component, key=lambda node: (len(protomers[node]), node))]
    ordered = {}
    for node, chains in protomers.items():
        placed = _order_like(reference, chains)
        if placed is None:
            return {}
        ordered[node] = placed
    copies = max((sites.get(node, 1) for node in component), default=1)
    trimmed = {node: [] for node in component}
    for index in range(len(reference)):
        slot = {node: chains[index] for node, chains in ordered.items()}
        for node, sequence in common_construct(slot).items():
            trimmed[node].extend([sequence] * copies)
    return trimmed


def _refusal_reason(left, right, expected: int | None) -> str | None:
    if left is None or right is None:
        return "no_matched_construct"
    if len(left) != len(right):
        return "chain_count_differs"
    if [len(chain) for chain in left] != [len(chain) for chain in right]:
        return "chain_length_differs"
    found = _substitutions(left, right)
    if found == 0:
        return "sides_identical_after_matching"
    if expected is not None and found != expected:
        return f"substitutions_{found}_expected_{expected}"
    return None


def plan(
    pairs: pd.DataFrame,
    config: dict,
    sequences: dict[str, list[str]],
    sites: dict[tuple[str, str], int] | None = None,
) -> Plan:
    columns = config["pdb_columns"]
    if any(column not in pairs for column in columns):
        raise KeyError(f"pair table lacks one of {columns}")
    if sites is None:
        from mutation_ddg import config as configuration

        sites = read_site_copies(configuration.resolve(config, "ligand_sites"))
    rows = []
    for _, row in pairs.iterrows():
        smiles = row.get(config["smiles_column"])
        keys = [str(row[column]).lower() for column in columns]
        if (
            not isinstance(smiles, str)
            or not smiles.strip()
            or any(not sequences.get(key) for key in keys)
        ):
            continue
        rows.append((row, keys, str(row[config["ligand_column"]]), smiles))
    edges = [((keys[0], ligand), (keys[1], ligand)) for _, keys, ligand, _ in rows]
    built = {}
    for component in components(edges):
        built.update(_build_component(component, sequences, sites))
    complexes = {}
    refusals = []
    for row, keys, ligand, smiles in rows:
        left = built.get((keys[0], ligand))
        right = built.get((keys[1], ligand))
        reason = _refusal_reason(left, right, _expected_substitutions(row, config))
        if reason:
            refusals.append(Refusal(keys[0], keys[1], ligand, reason))
            continue
        for key, chains in ((keys[0], left), (keys[1], right)):
            name = f"{key}_{ligand}"
            complexes.setdefault(name, Complex(name, list(chains), smiles))
    return Plan(list(complexes.values()), refusals)


def partition(
    jobs: list[Complex], max_residues: int
) -> tuple[list[Complex], list[Complex]]:
    return (
        [job for job in jobs if job.n_residues <= max_residues],
        [job for job in jobs if job.n_residues > max_residues],
    )


def shard(jobs: list[Complex], index: int, count: int) -> list[Complex]:
    if not 0 <= index < count:
        raise ValueError(f"shard {index} is not one of {count}")
    return jobs[index::count]
