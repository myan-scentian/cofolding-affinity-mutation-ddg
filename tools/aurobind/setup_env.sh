#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${AUROBIND_WORK:-${PROJECT_ROOT}/artifacts/work/aurobind}"
REVISION="ba07ce02621e99797b60492c61b4d320825b8f0a"
mkdir -p "$WORK"

if [ ! -d "$WORK/repo/.git" ]; then
  git clone https://github.com/GENTEL-lab/AuroBind.git "$WORK/repo"
fi
git -C "$WORK/repo" fetch --quiet origin
git -C "$WORK/repo" checkout --quiet "$REVISION"

if [ ! -x "$WORK/.venv/bin/python" ]; then
  uv venv --python 3.11 "$WORK/.venv"
fi
uv pip install --python "$WORK/.venv/bin/python" \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$WORK/.venv/bin/python" \
  numpy==1.26.4 scipy==1.14.1 pandas==2.2.3 numba==0.61.0 rdkit==2024.3.2 \
  biopython==1.85 einops==0.8.0 ml-collections==1.0.0 mashumaro==3.14 \
  modelcif==1.2 ihm==2.5 pyyaml==6.0.2 requests==2.32.3 tqdm==4.67.1 \
  click==8.1.8 accelerate==1.1.1

mkdir -p "$WORK/cache_data"
if [ ! -f "$WORK/cache_data/model.pt" ] || [ ! -f "$WORK/cache_data/ccd.pkl" ]; then
  uv tool run --from gdown gdown --folder \
    "https://drive.google.com/drive/folders/1DNZDXewqFibbP0erslQlm6Sp0qVQG1Ad" \
    -O "$WORK/cache_data"
fi

test -f "$WORK/cache_data/model.pt"
test -f "$WORK/cache_data/ccd.pkl"
pixi run --manifest-path "$PROJECT_ROOT/pixi.toml" python "$PROJECT_ROOT/tools/aurobind/patch_repo.py"
