#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${AQAFFINITY_WORK:-${PROJECT_ROOT}/artifacts/work/aqaffinity}"
mkdir -p "$WORK"

hf auth whoami >/dev/null
[ -f "$WORK/repo/model_weights/model_weights_only.safetensors" ] || \
  hf download SandboxAQ/AQAffinity --local-dir "$WORK/repo" --exclude "data/*"

uv venv --python 3.11 "$WORK/.venv-ft3"
uv pip install --python "$WORK/.venv-ft3/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install --python "$WORK/.venv-ft3/bin/python" "openfold3==0.3.1" safetensors
uv pip install --python "$WORK/.venv-ft3/bin/python" --no-deps "$WORK/repo"

RUNNER="$(find "$WORK/.venv-ft3" -path '*aqaffinity/modules/model/runner.py' | head -1)"
sed -i 's/use_deepspeed_evo_attention=True,/use_deepspeed_evo_attention=False,/' "$RUNNER"

mkdir -p "$WORK/of3_cache"
[ -f "$WORK/of3_cache/of3_ft3_v1.pt" ] || \
  curl -fL --retry 3 -o "$WORK/of3_cache/of3_ft3_v1.pt" \
    "https://openfold.s3.amazonaws.com/openfold3_params/of3_ft3_v1.pt"

"$WORK/.venv-ft3/bin/python" -c \
  "import importlib.metadata as m; print('openfold3', m.version('openfold3'))"
