"""Predicted against measured ddG and spread by measured-effect sextile."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mutation_ddg.analysis import pairing, stratified, style
from mutation_ddg.io import tables
from mutation_ddg.paths import ARTIFACT_ROOT

OUT = ARTIFACT_ROOT / "results" / "figures"

FIGSIZE = (5.50, 1.55)
INCLUDED_AT = 1.0
ADJUST = {"left": 0.055, "right": 0.99, "top": 0.90, "bottom": 0.18}
SCATTER_WSPACE = 0.10
GROUP_WSPACE = 0.13
GROUP_WIDTH_RATIOS = (3.2, 1.65)
SPREAD_BOX_RATIO = 1.65
LIMIT = 7.5
BOTTOM_LABEL_Y = 0.045
SPREAD_PARTS = 6
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def _spread_summary(frames, keep) -> tuple[np.ndarray, ...]:
    """SDs and paired spread-ratio intervals in primary-cohort sextiles."""
    prepared = {}
    reference_keys = None
    for name in style.MODELS:
        restricted = pairing.restrict(frames[name], keep)
        keys = pairing.pair_keys(restricted)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(f"{name} rows are not aligned with the primary cohort")
        prepared[name] = stratified.prepare(restricted)

    reference = prepared["Boltz-2"]
    for name, frame in prepared.items():
        if not np.array_equal(frame.system.to_numpy(), reference.system.to_numpy()):
            raise RuntimeError(f"{name} systems are not aligned with Boltz-2")
        if not np.allclose(frame.ddg_kcal, reference.ddg_kcal):
            raise RuntimeError(f"{name} measurements differ from Boltz-2")

    values = np.column_stack(
        [
            reference.ddg_kcal.to_numpy(float),
            *(prepared[name].pred_ddg.to_numpy(float) for name in style.MODELS),
        ]
    )
    edges = stratified.quantile_edges(reference.ddg_kcal, parts=SPREAD_PARTS)
    membership = np.searchsorted(
        np.asarray(edges[1:-1]), np.abs(values[:, 0]), side="right"
    )
    counts = np.bincount(membership, minlength=SPREAD_PARTS)
    centres = np.array(
        [
            np.median(np.abs(values[membership == part, 0]))
            for part in range(SPREAD_PARTS)
        ]
    )
    point = np.stack(
        [values[membership == part].std(axis=0, ddof=1) for part in range(SPREAD_PARTS)]
    )

    systems = reference.system.to_numpy()
    groups = [np.flatnonzero(systems == system) for system in np.unique(systems)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.full((BOOTSTRAP_DRAWS, SPREAD_PARTS, values.shape[1]), np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), len(groups))
        take = np.concatenate([groups[index] for index in chosen])
        for part in range(SPREAD_PARTS):
            rows = take[membership[take] == part]
            if len(rows) > 1:
                sampled[draw, part] = values[rows].std(axis=0, ddof=1)

    ratio_point = point[:, 1:] / point[:, [0]]
    ratio_sampled = sampled[:, :, 1:] / sampled[:, :, [0]]
    ratio_low = np.nanpercentile(ratio_sampled, 2.5, axis=0)
    ratio_high = np.nanpercentile(ratio_sampled, 97.5, axis=0)
    if not all(
        np.isfinite(array).all()
        for array in (centres, point, ratio_point, ratio_low, ratio_high)
    ):
        raise RuntimeError("spread bootstrap produced a non-finite interval")
    return counts, centres, point, ratio_point, ratio_low, ratio_high


def _spread_calibration_panel(ax, frames, keep) -> tuple[np.ndarray, np.ndarray]:
    """Draw paired spread ratios as a Cleveland point-range plot."""
    counts, centres, spread, point, low, high = _spread_summary(frames, keep)
    offsets = np.linspace(-0.045, 0.045, len(style.MODELS))
    for column, (name, offset) in enumerate(zip(style.MODELS, offsets)):
        mark = style.colour(name)
        error = np.stack(
            [point[:, column] - low[:, column], high[:, column] - point[:, column]]
        )
        ax.errorbar(
            centres + offset,
            point[:, column],
            yerr=error,
            color=mark,
            marker=style.marker(name),
            linestyle="none",
            markersize=style.MARKER,
            elinewidth=0.7,
            capsize=1.7,
            capthick=0.7,
            alpha=0.95,
            zorder=3,
        )

    ax.axhline(
        1.0, color=style.NULL, linewidth=style.RULE, linestyle=style.DIAGONAL, zorder=1
    )
    style.column_title(ax, "Spread ratio vs magnitude", size=style.TITLE_SMALL, pad=4)
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(0.0, 2.1)
    ax.set_box_aspect(1 / SPREAD_BOX_RATIO)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel(
        r"$\mathrm{SD}_{pred}/\mathrm{SD}_{meas}$",
        fontsize=style.LABEL,
        color=style.INK,
        labelpad=2,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=style.HAIRLINE, linewidth=0.5)
    style.spines(ax)
    return counts, spread


def main() -> int:
    style.apply()
    pairs = tables.pairs("hiqbind")
    frames = pairing.canonical_frames(pairs)
    keep = pairing.primary_pairs(pairs)
    assert keep.screens == pairing.PRIMARY_FILTERS, keep.screens

    fig = plt.figure(figsize=FIGSIZE)
    groups = fig.add_gridspec(
        1, 2, width_ratios=GROUP_WIDTH_RATIOS, wspace=GROUP_WSPACE
    )
    scatter_grid = groups[0, 0].subgridspec(1, len(style.MODELS), wspace=SCATTER_WSPACE)
    axes = [fig.add_subplot(scatter_grid[0, 0])]
    axes.extend(
        fig.add_subplot(scatter_grid[0, index], sharex=axes[0], sharey=axes[0])
        for index in range(1, len(style.MODELS))
    )
    spread_ax = fig.add_subplot(groups[0, 1])
    fig.subplots_adjust(**ADJUST)
    key = [
        Patch(facecolor=style.colour(name), edgecolor="none", label=name)
        for name in style.MODELS
    ]
    fig.legend(
        handles=key,
        loc="upper center",
        bbox_to_anchor=(0.46, 0.995),
        ncol=len(key),
        frameon=False,
        fontsize=style.LEGEND,
        handlelength=1.4,
        handleheight=0.7,
        handletextpad=0.45,
        columnspacing=1.5,
        borderpad=0.0,
    )

    for index, (ax, name) in enumerate(zip(axes, style.MODELS)):
        frame = pairing.restrict(frames[name], keep)
        measured = frame.ddg_kcal.to_numpy(float)
        predicted = frame.pred_ddg.to_numpy(float)
        mark = style.colour(name)

        ax.plot(
            [-LIMIT, LIMIT],
            [-LIMIT, LIMIT],
            color=style.MUTED,
            linewidth=0.8,
            linestyle=style.DIAGONAL,
            zorder=1,
        )
        ax.axhline(
            0, color=style.NULL, linewidth=style.RULE, linestyle=style.DOT, zorder=1
        )
        ax.scatter(
            measured,
            predicted,
            s=12,
            linewidth=0,
            alpha=0.66,
            color=mark,
            marker=style.marker(name),
            zorder=3,
        )

        ok = np.isfinite(measured) & np.isfinite(predicted)
        fit = np.polyfit(measured[ok], predicted[ok], 1)
        ax.plot(
            [-LIMIT, LIMIT],
            np.polyval(fit, [-LIMIT, LIMIT]),
            color=mark,
            linewidth=1.3,
            zorder=4,
        )
        ax.set_xlim(-LIMIT, LIMIT)
        ax.set_ylim(-LIMIT, LIMIT)
        ax.set_aspect("equal")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        style.spines(ax)
        if index:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel(
        r"Predicted $\Delta\Delta G$ (kcal/mol)",
        fontsize=style.LABEL,
        color=style.INK,
        labelpad=0,
    )
    style.panel_label(axes[0], "a")
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    fig.text(
        (left + right) / 2,
        BOTTOM_LABEL_Y,
        r"Measured $\Delta\Delta G$ (kcal/mol)",
        ha="center",
        va="bottom",
        fontsize=style.LABEL,
        color=style.INK,
    )
    spread_counts, spread_point = _spread_calibration_panel(spread_ax, frames, keep)
    style.panel_label(spread_ax, "b", x=-0.08)
    spread_box = spread_ax.get_position()
    fig.text(
        (spread_box.x0 + spread_box.x1) / 2,
        BOTTOM_LABEL_Y,
        r"Median $|\Delta\Delta G|$ (kcal/mol)",
        ha="center",
        va="bottom",
        fontsize=style.LABEL,
        color=style.INK,
    )
    style.save(fig, "scatter", OUT, dpi=400)
    plt.close(fig)

    print(f"wrote {OUT / 'scatter.pdf'} and .png")
    print(f"  effect-magnitude sextiles: {spread_counts.tolist()} pairs")
    print(f"  measured SD: {spread_point[:, 0].round(3).tolist()}")
    for name in style.MODELS:
        frame = pairing.restrict(frames[name], keep)
        measured = frame.ddg_kcal.to_numpy(float)
        predicted = frame.pred_ddg.to_numpy(float)
        print(
            f"  {name:12s} measured sd {measured.std(ddof=1):5.2f}  "
            f"predicted sd {predicted.std(ddof=1):5.2f}  "
            f"ratio {predicted.std(ddof=1) / measured.std(ddof=1):.3f}  "
            f"n {len(measured)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
