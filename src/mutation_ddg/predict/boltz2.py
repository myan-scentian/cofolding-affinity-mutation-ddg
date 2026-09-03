"""Shared definitions for the Boltz-2 input builders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

CHAIN_IDS = "ABCDEFGHIJKLMNOP"


@dataclass(frozen=True)
class Complex:
    name: str
    chains: list[str]
    ligand_smiles: str

    @property
    def n_residues(self) -> int:
        return sum(map(len, self.chains))


def sequence_key(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()[:16]
