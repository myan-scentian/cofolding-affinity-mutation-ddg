"""Add the reference-attention and affinity-only AuroBind paths."""

from __future__ import annotations

import sys

from common import SOURCE

PATCHES = [
    (
        SOURCE / "aurobind" / "openfold" / "model" / "heads.py",
        "                use_deepspeed_evo_attention=True,",
        "                use_deepspeed_evo_attention=False,",
    ),
    (
        SOURCE / "run_aurobind.py",
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'",
        "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')",
    ),
    (
        SOURCE / "aurobind" / "openfold" / "model" / "model.py",
        "    def forward(self, input_features, diffusion_batch_size=1):",
        "    def forward(self, input_features, diffusion_batch_size=1, affinity_only=False):",
    ),
    (
        SOURCE / "aurobind" / "openfold" / "model" / "model.py",
        "        x_predicted = self.sample_diffusion(input_features,",
        "        if affinity_only:\n            return outputs\n\n        x_predicted = self.sample_diffusion(input_features,",
    ),
    (
        SOURCE / "run_aurobind.py",
        "    finished = check_outputs(record, struct_dir, seed, args.num_diffusion_samples, args.output_format)",
        """    if args.affinity_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        struct_dir.mkdir(exist_ok=True)
        output_path = struct_dir / f"{record.id}_seed-{seed}_affinity.json"
        if output_path.exists() and not args.override:
            return struct_dir
        with torch.no_grad():
            outputs = model(input_features,
                            diffusion_batch_size=args.num_diffusion_samples,
                            affinity_only=True)
        with output_path.open("w") as handle:
            json.dump({"affinity": outputs["affinity_logits"].cpu().item()}, handle)
        return struct_dir

    finished = check_outputs(record, struct_dir, seed, args.num_diffusion_samples, args.output_format)""",
    ),
    (
        SOURCE / "run_aurobind.py",
        '    parser.add_argument(\n        "--save_features",',
        """    parser.add_argument(
        "--affinity_only",
        action="store_true",
    )
    parser.add_argument(
        "--save_features",""",
    ),
]


def main() -> int:
    failed = False
    for path, before, after in PATCHES:
        text = path.read_text()
        if after in text:
            print(f"applied {path.relative_to(SOURCE.parent)}")
            continue
        if text.count(before) != 1:
            print(f"missing patch anchor in {path}", file=sys.stderr)
            failed = True
            continue
        path.write_text(text.replace(before, after))
        print(f"applied {path.relative_to(SOURCE.parent)}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
