"""Run Boltz-2 while recording all 14 affinity-stack cut points."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture
import layer_capture

PINNED = {
    "--diffusion_samples_affinity": "5",
    "--sampling_steps_affinity": "200",
}
CHECKED = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
)


class Mismatch(RuntimeError):
    pass


class Instrument:
    def __init__(self) -> None:
        self.recorder = None
        self.blocks = None
        self.pending = {}
        self.provenance = {}
        self.forwards = 0
        self.writes = 0
        self.written = 0
        self.mismatched = []
        self.started = time.time()

    def attach(self, model) -> None:
        if self.recorder is not None:
            return
        if not getattr(model, "affinity_prediction", False):
            raise RuntimeError("instrumentation reached a non-affinity model")
        for name in (
            "pairformer_module",
            "msa_module",
            "affinity_module1",
            "affinity_module2",
        ):
            module = getattr(model, name, None)
            if module is not None and hasattr(module, "_orig_mod"):
                raise RuntimeError(f"{name} is compiled; hooks cannot see its calls")
        if not getattr(model, "affinity_ensemble", False):
            raise RuntimeError("the affinity checkpoint is not an ensemble")
        self.recorder = capture.Recorder(model)
        self.blocks = layer_capture.BlockRecorder(model)
        predict_args = dict(getattr(model, "predict_args", None) or {})
        self.provenance = {
            "argv": " ".join(sys.argv),
            "predict_args_affinity": json.dumps(predict_args, sort_keys=True),
            "use_kernels": str(bool(getattr(model, "use_kernels", False))),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        }
        try:
            import boltz

            self.provenance["boltz"] = getattr(boltz, "__version__", "unknown")
        except ImportError:
            self.provenance["boltz"] = "unknown"

    def start(self, features: dict) -> None:
        self.recorder.start(features)
        self.blocks.start(features)

    def abandon(self) -> None:
        if self.recorder is not None:
            self.recorder.masks = None
        if self.blocks is not None:
            self.blocks.masks = None

    def finish(self, features: dict, output: dict) -> None:
        with (
            torch.autocast("cuda", enabled=False),
            torch.autocast("cpu", enabled=False),
        ):
            record = self.recorder.finish(features, output)
            record.update(self.blocks.finish(output))
        expected = [
            *(f"z_aff1_block{index:02d}" for index in range(9)),
            *(f"z_aff2_block{index:02d}" for index in range(5)),
        ]
        missing = [key for key in expected if key not in record]
        if missing:
            raise Mismatch(f"missing affinity cut points: {missing}")
        for key in CHECKED:
            if key in output:
                record[f"{key}_exact"] = np.float64(
                    output[key].detach().float().flatten()[0].item()
                )
        self.pending[features["record"][0].id] = record

    def verify_and_write(self, record_id: str, directory: Path) -> None:
        record = self.pending.pop(record_id, None)
        if record is None:
            raise Mismatch(f"{record_id}: writer had no instrumented forward")
        path = directory / f"affinity_{record_id}.json"
        written = json.loads(path.read_text())
        disagreements = []
        for key in CHECKED:
            if key in written and float(record[f"{key}_exact"]) != float(written[key]):
                disagreements.append(
                    {
                        "key": key,
                        "recorded": float(record[f"{key}_exact"]),
                        "written": float(written[key]),
                    }
                )
        if disagreements:
            self.mismatched.append({"record": record_id, "keys": disagreements})
            path.unlink(missing_ok=True)
            raise Mismatch(f"{record_id}: recorded values differ from Boltz output")
        record["prediction_match"] = np.int8(1)
        record["record_id"] = np.array(record_id)
        for key, value in self.provenance.items():
            record[f"provenance_{key}"] = np.array(value)
        np.savez_compressed(directory / f"representations_{record_id}.npz", **record)
        self.written += 1
        print(
            f"[instrument] {record_id}: {self.written} records in "
            f"{(time.time() - self.started) / 60:.1f} min",
            flush=True,
        )


INSTRUMENT = Instrument()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from boltz.data.write.writer import BoltzAffinityWriter
    from boltz.model.models.boltz2 import Boltz2

    original_forward = Boltz2.forward

    def forward(self, features, *args, **kwargs):
        if not getattr(self, "affinity_prediction", False):
            return original_forward(self, features, *args, **kwargs)
        INSTRUMENT.attach(self)
        INSTRUMENT.forwards += 1
        INSTRUMENT.start(features)
        try:
            output = original_forward(self, features, *args, **kwargs)
        except Exception:
            INSTRUMENT.abandon()
            raise
        INSTRUMENT.finish(features, output)
        return output

    Boltz2.forward = forward
    original_write = BoltzAffinityWriter.write_on_batch_end

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        original_write(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx,
        )
        record_id = batch["record"][0].id
        if prediction.get("exception"):
            INSTRUMENT.pending.pop(record_id, None)
            return
        INSTRUMENT.writes += 1
        INSTRUMENT.verify_and_write(record_id, self.output_dir / record_id)

    BoltzAffinityWriter.write_on_batch_end = write_on_batch_end


def pin(arguments: list[str]) -> list[str]:
    output = list(arguments)
    for flag, value in PINNED.items():
        if flag not in output:
            output += [flag, value]
    return output


def main(arguments: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if arguments is None else arguments)
    if argv and argv[0] == "predict":
        argv = [argv[0], *pin(argv[1:])]
    install()
    from boltz.main import cli

    try:
        cli.main(args=argv, prog_name="boltz", standalone_mode=False)
    except SystemExit as exit_:
        code = int(exit_.code or 0)
        if code:
            return code
    except Mismatch as error:
        print(f"[instrument] stopped: {error}", flush=True)
        return 2
    if INSTRUMENT.mismatched:
        return 2
    if INSTRUMENT.pending:
        print(f"recorded but unwritten: {sorted(INSTRUMENT.pending)[:10]}")
        return 3
    if INSTRUMENT.writes and not INSTRUMENT.written:
        return 4
    if INSTRUMENT.written != INSTRUMENT.writes:
        return 5
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
