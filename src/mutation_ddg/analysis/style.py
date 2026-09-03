"""Shared visual settings for the manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MODELS = ("Boltz-2", "AQAffinity", "AuroBind")
MODEL = {
    "Boltz-2": "#1D4B73",
    "AQAffinity": "#C65F25",
    "AuroBind": "#9295A6",
}
MODEL_MARKER = {"Boltz-2": "o", "AQAffinity": "^", "AuroBind": "s"}
INK = "#1B1B1B"
MUTED = "#6B6B6B"
HAIRLINE = "#D3D3D3"
NULL = "#5F5F5F"
WHITE = "#FFFFFF"
MUTATION = "#B2182B"
STRATUM = ("#4A4A4A", "#B4B4B4")
TITLE = 8.0
TITLE_SMALL = 7.0
LABEL = 8.0
TICK = 7.0
LEGEND = 7.5
ANNOTATION = 6.5
MARKER = 4.2
LINE = 1.4
RULE = 0.9
SPINE = 0.7
TICK_LENGTH = 2.0
BAND_ALPHA = 0.30
DASH = (0, (5, 2.5))
DOT = (0, (1, 2))
DIAGONAL = (0, (4, 3))


def colour(model: str) -> str:
    return MODEL[model]


def marker(model: str) -> str:
    return MODEL_MARKER[model]


def binary_map() -> ListedColormap:
    return ListedColormap([WHITE, MODEL["Boltz-2"]])


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": LABEL,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.cal": "Liberation Sans",
            "mathtext.sf": "Liberation Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": LABEL,
            "axes.titlesize": TITLE,
            "axes.titlecolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": SPINE,
            "axes.facecolor": WHITE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": WHITE,
            "grid.color": HAIRLINE,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": LEGEND,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.3,
            "lines.linewidth": LINE,
            "lines.markersize": MARKER,
            "savefig.facecolor": WHITE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "xtick.major.width": SPINE,
            "ytick.major.width": SPINE,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def spines(ax, keep: tuple = ("left", "bottom"), tick_size: float = TICK) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(MUTED)
        spine.set_linewidth(SPINE)
    ax.tick_params(
        axis="both",
        length=TICK_LENGTH,
        width=SPINE,
        colors=MUTED,
        labelsize=tick_size,
        pad=2,
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(INK)


def reference_rule(
    ax,
    level: float,
    label: str = "",
    *,
    dash=DOT,
    line_colour: str = NULL,
    at: float = 0.0,
    offset: tuple = (0, 3),
    ha: str = "left",
    va: str = "baseline",
    zorder: float = 2.0,
    patch: bool = False,
):
    line = ax.axhline(level, color=line_colour, lw=RULE, ls=dash, zorder=zorder)
    if label:
        box = {"facecolor": WHITE, "edgecolor": "none", "pad": 1.0} if patch else None
        ax.annotate(
            label,
            (at, level),
            xytext=offset,
            textcoords="offset points",
            fontsize=ANNOTATION,
            color=line_colour,
            ha=ha,
            va=va,
            zorder=zorder + 0.1,
            bbox=box,
        )
    return line


def column_title(
    ax, text: str, *, pad: float = 3.5, size: float = TITLE, mark: str = INK
):
    return ax.set_title(text, fontsize=size, color=mark, loc="center", pad=pad)


def panel_title(
    ax,
    text: str,
    *,
    pad: float = 3.0,
    size: float = TITLE,
    mark: str = INK,
    right: str = "",
):
    title = ax.set_title(text, fontsize=size, color=mark, loc="left", pad=pad)
    if right:
        ax.set_title(right, fontsize=ANNOTATION, color=MUTED, loc="right", pad=pad)
    return title


def panel_label(
    ax, label: str, *, x: float = -0.16, y: float = 1.08, size: float = TITLE
):
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def overflow(fig, tolerance: float = 0.01) -> tuple:
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    sides = (-box.x0, -box.y0, box.x1 - width, box.y1 - height)
    return tuple(round(side, 4) if side > tolerance else 0.0 for side in sides)


def save(
    fig,
    name: str,
    out_dir: Path,
    *,
    dpi: int = 400,
    suffixes: tuple = ("pdf", "png"),
    allow_overflow: bool = False,
) -> None:
    spill = overflow(fig)
    if any(spill) and not allow_overflow:
        raise SystemExit(f"{name}: drawn ink leaves the canvas by {spill} inches")
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=dpi)
