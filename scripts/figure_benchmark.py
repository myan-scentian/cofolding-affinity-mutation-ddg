"""Figure 1: one matched structural pair in its measured-effect cohort."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mutation_ddg.analysis import pairing, style
from mutation_ddg.io import tables
from mutation_ddg.paths import ARTIFACT_ROOT

WORK = ARTIFACT_ROOT / "work" / "figures"
OUT = ARTIFACT_ROOT / "results" / "figures"

FIGSIZE = (5.50, 2.02)
INCLUDED_AT = 1.0
ADJUST = {"left": 0.015, "right": 0.995, "top": 0.825, "bottom": 0.205, "wspace": 0.018}
STRUCTURE_BOTTOM = 0.075
STRUCTURE_PANEL_Y = 1.13
PANEL_TITLE = style.TITLE

PANELS = [
    ("cofolded_1czh.png", "Flavodoxin--FMN\nGly58"),
    ("cofolded_1czl.png", "Flavodoxin--FMN\nAsn58"),
]


def measured_effects() -> np.ndarray:
    """Measured effects for the primary cohort."""
    pairs = tables.pairs("hiqbind")
    keep = pairing.primary_pairs(pairs)
    frame = pairing.restrict(pairing.canonical_frames(pairs)["Boltz-2"], keep)
    if len(frame) != 163:
        raise RuntimeError(f"expected 163 primary pairs; got {len(frame)}")
    return frame["ddg_kcal"].to_numpy(float)


def _effect_distribution(ax, effects: np.ndarray) -> None:
    """Show the measured-effect distribution without highlighting an example."""
    counts, _, _ = ax.hist(
        effects,
        bins=np.arange(-5.0, 8.0, 1.0),
        color=style.STRATUM[1],
        edgecolor=style.WHITE,
        linewidth=0.5,
    )
    top = float(counts.max())
    ax.axvline(0.0, color=style.NULL, lw=style.RULE, ls=style.DOT, zorder=2)

    style.panel_label(ax, "c", x=-0.15, y=1.16)
    ax.set_title(
        "Distribution of measured\n" r"$\Delta\Delta G$",
        fontsize=PANEL_TITLE,
        color=style.INK,
        pad=2.5,
        linespacing=1.18,
    )
    ax.set_xlim(-5.0, 7.0)
    ax.set_ylim(0.0, 1.08 * top)
    ax.set_xticks([-5.0, 0.0, 5.0])
    ax.set_yticks([])
    ax.set_xlabel(
        r"Measured $\Delta\Delta G$ (kcal/mol)",
        fontsize=style.LABEL,
        color=style.INK,
        labelpad=2,
    )
    style.spines(ax, keep=("bottom",))


def main() -> int:
    style.apply()
    effects = measured_effects()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.subplots_adjust(**ADJUST)

    for letter, ax, (name, title) in zip("ab", axes[:2], PANELS):
        path = WORK / name
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run tools/figures/render_cofolded.py")
        position = ax.get_position(original=True)
        ax.set_position(
            [
                position.x0,
                STRUCTURE_BOTTOM,
                position.width,
                ADJUST["top"] - STRUCTURE_BOTTOM,
            ]
        )
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(
            title, fontsize=PANEL_TITLE, color=style.INK, pad=2.5, linespacing=1.18
        )
        style.panel_label(ax, letter, x=0.03, y=STRUCTURE_PANEL_Y)

    _effect_distribution(axes[2], effects)

    style.save(fig, "benchmark", OUT, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / 'benchmark.pdf'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
