"""Capture the pooled input and every block output of both affinity stacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture


class BlockRecorder:
    """Record the released head's pooled vector at all 14 cut points."""

    def __init__(self, model) -> None:
        self.buffer: dict[str, np.ndarray] = {}
        self.masks: dict[str, torch.Tensor] | None = None
        for index in (1, 2):
            stack = getattr(model, f"affinity_module{index}").pairformer_stack
            stack.register_forward_pre_hook(self._entry(index))
            for block, layer in enumerate(stack.layers):
                layer.register_forward_hook(self._block(index, block))

    def _store(self, name: str, pair: torch.Tensor) -> None:
        if self.masks is None:
            return
        pooled = capture.pooled(pair[0].float(), self.masks["cross"])
        self.buffer[name] = pooled.detach().float().cpu().numpy().astype(np.float32)

    def _entry(self, index: int):
        def hook(_module, arguments) -> None:
            self._store(f"z_aff{index}_block00", arguments[0])

        return hook

    def _block(self, index: int, block: int):
        def hook(_module, _arguments, output) -> None:
            self._store(f"z_aff{index}_block{block + 1:02d}", output)

        return hook

    def start(self, batch: dict) -> None:
        self.buffer = {}
        self.masks = capture.masks_from(batch)

    def finish(self, output: dict) -> dict:
        record = dict(self.buffer)
        for key in (
            "affinity_pred_value",
            "affinity_pred_value1",
            "affinity_pred_value2",
        ):
            if key in output:
                record[key] = np.float32(
                    output[key].detach().float().flatten()[0].item()
                )
        self.masks = None
        return record
