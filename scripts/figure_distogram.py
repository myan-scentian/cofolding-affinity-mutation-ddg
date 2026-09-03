"""Figure: signed distogram-bin shift under the substitution contrast, one pair."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


def _artifact_root():
    """Artifact root: $MUTATION_DDG_ARTIFACT_ROOT, else <repo>/artifacts."""
    import os
    from pathlib import Path as _P

    override = os.environ.get("MUTATION_DDG_ARTIFACT_ROOT")
    if override:
        return _P(override)
    for parent in _P(__file__).resolve().parents:
        if (parent / "pixi.toml").exists():
            return parent / "artifacts"
    return _P.cwd() / "artifacts"


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "trunk_probe"))
import distogram as pooled

from mutation_ddg.analysis import style

FOLDED = Path(str(_artifact_root() / "results/boltz_hiqbind_unified"))
REFERENCE, VARIANT = "3nu3_478", "3nu4_478"
SEED = 1
LINEWIDTH_IN = 5.5
DISTOGRAM_INCLUDED_AT = 0.62
SHIFT_CLIP = 4
OUT = _artifact_root() / "results" / "figures" / "distogram.pdf"


def record(seed: int, tag: str) -> dict:
    path = FOLDED / f"s{seed}" / "predictions" / tag / f"representations_{tag}.npz"
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in (*pooled.WANTED, "ligand_mask") if k in data.files}


def signed_shift(a: dict, b: dict):
    """Compare bins over shared identity-keyed tokens."""
    rows = pooled.shared_rows(a, b)
    if rows is None:
        raise SystemExit(f"{REFERENCE}/{VARIANT}: the two crops share too few tokens")
    left, right = rows[:, 0], rows[:, 1]
    ligand = a["ligand_mask"][left].astype(bool)
    substituted = np.array(
        [x != y for x, y in zip(a["res_type"][left], b["res_type"][right])]
    )
    order = np.argsort(~ligand)
    before = pooled.bins(a["disto_coords"])[np.ix_(left, left)][np.ix_(order, order)]
    after = pooled.bins(b["disto_coords"])[np.ix_(right, right)][np.ix_(order, order)]
    shift = np.clip(
        after.astype(int) - before.astype(int), -SHIFT_CLIP, SHIFT_CLIP
    ).astype(float)
    shift[before == after] = np.nan
    return shift, int(ligand.sum()), substituted[order]


def load() -> dict:
    """Load the one substitution contrast drawn by both figure artifacts."""
    shift, n_ligand, substituted = signed_shift(
        record(SEED, REFERENCE), record(SEED, VARIANT)
    )
    return {
        "shift": shift,
        "n_ligand": n_ligand,
        "substituted": substituted,
    }


def draw(ax, data: dict):
    """Draw the signed matrix on an existing axes and return its colour-mapped image."""
    shift = data["shift"]
    n_ligand = data["n_ligand"]
    substituted = data["substituted"]
    image = ax.imshow(
        shift, cmap="RdBu_r", vmin=-SHIFT_CLIP, vmax=SHIFT_CLIP, interpolation="nearest"
    )
    for index in np.flatnonzero(substituted):
        ax.axhline(index, color=style.MUTATION, lw=0.5, alpha=0.85)
        ax.axvline(index, color=style.MUTATION, lw=0.5, alpha=0.85)
    ax.axhline(n_ligand - 0.5, color=style.MUTED, lw=style.SPINE, alpha=0.5)
    ax.axvline(n_ligand - 0.5, color=style.MUTED, lw=style.SPINE, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    style.spines(ax, keep=("left", "right", "top", "bottom"))
    return image


def add_colorbar(fig, image, *, ax=None, cax=None):
    """Apply the shared signed-shift scale to a standalone or composed panel."""
    bar = fig.colorbar(image, ax=ax, cax=cax, fraction=0.046, pad=0.02)
    bar.set_label("Signed distogram change", fontsize=style.LABEL, color=style.INK)
    bar.ax.tick_params(
        labelsize=style.TICK,
        length=style.TICK_LENGTH,
        width=style.SPINE,
        colors=style.MUTED,
        pad=2,
    )
    for label in bar.ax.get_yticklabels():
        label.set_color(style.INK)
    bar.outline.set_color(style.MUTED)
    bar.outline.set_linewidth(style.SPINE)
    return bar


def main() -> None:
    style.apply()
    data = load()
    width = DISTOGRAM_INCLUDED_AT * LINEWIDTH_IN
    fig, ax = plt.subplots(figsize=(width, width * 0.94))
    fig.subplots_adjust(left=0.045, right=0.86, bottom=0.035, top=0.98)
    image = draw(ax, data)
    add_colorbar(fig, image, ax=ax)
    style.save(fig, OUT.stem, OUT.parent)
    plt.close(fig)
    changed = int(np.isfinite(data["shift"]).sum())
    print(
        f"  wrote {OUT}  ({changed} of {data['shift'].size} entries changed bin, "
        f"{data['n_ligand']} ligand tokens, {int(data['substituted'].sum())} substituted)"
    )


if __name__ == "__main__":
    main()
