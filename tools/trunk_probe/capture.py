"""Capture the metadata and coordinates used by the Boltz-2 affinity pass."""

from __future__ import annotations

import numpy as np
import torch


def masks_from(features: dict) -> dict[str, torch.Tensor]:
    pad = features["token_pad_mask"][0].float()
    receptor = (features["mol_type"][0] == 0).float() * pad
    ligand = features["affinity_token_mask"][0].float() * pad
    cross = (
        ligand[:, None] * receptor[None, :]
        + receptor[:, None] * ligand[None, :]
        + ligand[:, None] * ligand[None, :]
    )
    cross *= 1 - torch.eye(len(pad), device=pad.device)
    return {"pad": pad, "receptor": receptor, "ligand": ligand, "cross": cross}


def pooled(pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (pair * mask[..., None]).sum(dim=(0, 1)) / (mask.sum() + 1e-7)


class Recorder:
    def __init__(self, _model) -> None:
        self.masks = None

    def start(self, features: dict) -> None:
        self.masks = masks_from(features)

    def finish(self, features: dict, output: dict) -> dict:
        masks = self.masks
        record = {
            "res_idx": features["residue_index"][0].cpu().numpy().astype(np.int32),
            "asym_id": features["asym_id"][0].cpu().numpy().astype(np.int32),
            "mol_type": features["mol_type"][0].cpu().numpy().astype(np.int32),
            "res_type": features["res_type"][0]
            .argmax(-1)
            .cpu()
            .numpy()
            .astype(np.int32),
            "token_pad_mask": masks["pad"].cpu().numpy().astype(np.int8),
            "ligand_mask": masks["ligand"].cpu().numpy().astype(np.int8),
            "receptor_mask": masks["receptor"].cpu().numpy().astype(np.int8),
            "n_cross_entries": np.float32(masks["cross"].sum().item()),
        }
        if "sample_atom_coords" in output:
            best = 0
            if "iptm" in output:
                scores = output["iptm"].detach().float().flatten()
                best = int(torch.argmax(scores).item())
                record["scored_sample_index"] = np.int32(best)
            coordinates = output["sample_atom_coords"].detach().float()[best]
            mapping = features["token_to_rep_atom"][0].float()
            record["disto_coords"] = (
                (mapping @ coordinates).cpu().numpy().astype(np.float32)
            )
        self.masks = None
        return record
