"""Draw the 14-cut-point Boltz-2 linear-probe panel."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mutation_ddg.analysis import style
from mutation_ddg.paths import ARTIFACT_ROOT

WORK = ARTIFACT_ROOT / "work" / "trunk_probe"
OUT = ARTIFACT_ROOT / "results" / "figures"
BOLTZ = style.colour("Boltz-2")
TAG = "163"
HEADER = "163 pairs"
PROTEINS = 66
SEEDS = 5
FIGSIZE = (3.05, 2.55)
INCLUDED_AT = FIGSIZE[0] / 5.50
ADJUST = {"left": 0.180, "right": 0.985, "top": 0.770, "bottom": 0.205}
YLABEL = "Pooled out-of-fold Pearson $r$"
XLABEL = "Block within each ensemble"
XLABEL_PAD = 3.0
Y_WINDOW = (0.18, 1.02)
Y_TICKS = (0.2, 0.4, 0.6, 0.8, 1.0)
COPY_GAP = 1.0
COPIES = ("module1", "module2")
COPY_LABELS = {"module1": "ensemble 1", "module2": "ensemble 2"}
LOG_COPY_LABELS = {"module1": "copy 1", "module2": "copy 2"}
JOINT_SOURCE = "probe_results.json"


def cut_points(modules: dict) -> dict[str, list[str]]:
    """Return the input and block-output labels for both ensemble members."""
    missing = [name for name in COPIES if name not in modules]
    if missing:
        raise SystemExit(f"missing probe results for {', '.join(missing)}")
    return {name: sorted(modules[name]["depths"]) for name in COPIES}


def read_json(name: str) -> dict:
    return json.loads((WORK / name).read_text())


def read_seed_table(header: str, name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read the five individual-seed fits and averaged-representation fit."""
    log = (WORK / f"seed_spread_{TAG}.log").read_text()
    sections = re.split(r"^=+ (.+?) =+$", log, flags=re.MULTILINE)
    bodies = {
        sections[index]: sections[index + 1] for index in range(1, len(sections), 2)
    }
    label = LOG_COPY_LABELS[name]
    matched = [key for key in bodies if key.startswith(header) and key.endswith(label)]
    if len(matched) != 1:
        raise SystemExit(f"expected one {header!r} section for {label!r}")
    blocks, per_seed, averaged = [], [], []
    for line in bodies[matched[0]].splitlines():
        fields = line.split()
        if len(fields) != 10 or not re.fullmatch(r"\d{2}", fields[0]):
            continue
        values = [float(value) for value in fields[1:]]
        blocks.append(fields[0])
        per_seed.append(values[:SEEDS])
        averaged.append(values[-1])
    if not blocks:
        raise SystemExit(f"no cut-point rows in {matched[0]!r}")
    return blocks, np.asarray(per_seed).T, np.asarray(averaged)


def joint_envelope(
    cohort: dict, depths: dict[str, list[str]], measured
) -> tuple[float, dict]:
    """Read and verify the simultaneous 14-cut-point permutation result."""
    recorded_values = []
    for name in COPIES:
        recorded = cohort["modules"][name]["depths"]
        if sorted(recorded) != depths[name]:
            raise SystemExit(f"{JOINT_SOURCE} has different cut points for {name}")
        recorded_values.extend(recorded[block]["r_to_ddg"] for block in depths[name])
    if np.max(np.abs(np.asarray(recorded_values) - measured)) > 5e-4:
        raise SystemExit(f"{JOINT_SOURCE} does not describe the plotted probe fit")
    joint = cohort["joint"]
    if joint["n_cut_points"] != len(measured):
        raise SystemExit(
            "the permutation family does not contain all plotted cut points"
        )
    return float(joint["envelope_within_protein_p95"]), joint


def load(tag: str, header: str) -> dict:
    """Load and cross-check the averaged, individual-seed, and permutation results."""
    cohort = read_json(JOINT_SOURCE)["cohort"]
    modules = cohort["modules"]
    depths = cut_points(modules)

    cuts = []
    spans = {}
    start = 0
    for name in COPIES:
        cuts.extend((name, block) for block in depths[name])
        spans[name] = (start, start + len(depths[name]))
        start += len(depths[name])

    measured = np.asarray(
        [modules[name]["depths"][block]["r_to_ddg"] for name, block in cuts]
    )
    seed_columns = []
    for name in COPIES:
        blocks, matrix, logged_average = read_seed_table(header, name)
        if blocks != depths[name]:
            raise SystemExit(f"the seed log has different cut points for {name}")
        expected_average = np.asarray(
            [modules[name]["depths"][block]["r_to_ddg"] for block in blocks]
        )
        if np.max(np.abs(logged_average - expected_average)) > 5e-4:
            raise SystemExit(f"the averaged seed log and JSON disagree for {name}")
        seed_columns.append(matrix)

    threshold, joint = joint_envelope(cohort, depths, measured)
    return {
        "cuts": cuts,
        "depths": depths,
        "spans": spans,
        "control": np.asarray(
            [modules[name]["depths"][block]["r_to_head"] for name, block in cuts]
        ),
        "measured": measured,
        "per_seed": np.hstack(seed_columns),
        "null_mean": threshold,
        "joint": joint,
        "head": cohort["head_output"]["combined_r_vs_measured"],
    }


def x_positions(data: dict) -> np.ndarray:
    """Place a gap between the independently parameterised ensemble members."""
    positions = np.arange(len(data["cuts"]), dtype=float)
    first, _ = data["spans"][COPIES[1]]
    positions[first:] += COPY_GAP
    return positions


def draw(ax, data: dict, *, panel: str | None = None) -> None:
    """Draw both probe targets and the simultaneous permutation threshold."""
    positions = x_positions(data)
    band_low = data["per_seed"].min(axis=0)
    band_high = data["per_seed"].max(axis=0)
    for name in COPIES:
        first, stop = data["spans"][name]
        x = positions[first:stop]
        ax.fill_between(
            x,
            band_low[first:stop],
            band_high[first:stop],
            color=BOLTZ,
            alpha=style.BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            data["control"][first:stop],
            marker="o",
            markersize=3.2,
            linewidth=style.LINE,
            color=style.INK,
            zorder=5,
        )
        ax.plot(
            x,
            data["measured"][first:stop],
            marker="s",
            markersize=3.2,
            linewidth=style.LINE,
            color=BOLTZ,
            zorder=5,
        )
        heading_x = x[-1] if name == COPIES[1] else (x[0] + x[-1]) / 2
        heading_align = "right" if name == COPIES[1] else "center"
        ax.text(
            heading_x,
            0.985,
            COPY_LABELS[name],
            transform=ax.get_xaxis_transform(),
            ha=heading_align,
            va="top",
            fontsize=style.ANNOTATION,
            color=style.MUTED,
        )

    released = style.reference_rule(
        ax, data["head"], dash=style.DASH, line_colour=style.MUTED, zorder=2
    )
    permutation = style.reference_rule(
        ax, data["null_mean"], dash=style.DOT, line_colour=style.NULL, zorder=2
    )
    first_second, _ = data["spans"][COPIES[1]]
    divider = (positions[first_second - 1] + positions[first_second]) / 2
    ax.axvline(divider, color=style.HAIRLINE, linewidth=style.SPINE, zorder=0.5)

    control = plt.Line2D(
        [],
        [],
        marker="o",
        color=style.INK,
        linewidth=style.LINE,
        markersize=3.2,
        label="Ensemble output",
    )
    measured = plt.Line2D(
        [],
        [],
        marker="s",
        color=BOLTZ,
        linewidth=style.LINE,
        markersize=3.2,
        label=r"Measured $\Delta\Delta G$",
    )
    released.set_label("Released head")
    permutation.set_label("Permutation 95%")
    ax.legend(
        handles=[control, measured, released, permutation],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(-0.09, 1.025),
        fontsize=style.ANNOTATION,
        frameon=False,
        borderpad=0,
        handlelength=1.6,
        columnspacing=1.0,
        labelspacing=0.25,
    )
    ax.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
    ax.set_ylim(*Y_WINDOW)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["in" if block == "00" else str(int(block)) for _, block in data["cuts"]]
    )
    ax.set_ylabel(YLABEL, fontsize=style.LABEL, color=style.INK, labelpad=2)
    ax.set_xlabel(XLABEL, fontsize=style.LABEL, color=style.INK, labelpad=XLABEL_PAD)
    style.spines(ax)
    if panel:
        style.panel_label(ax, panel, x=-0.17, y=1.245)


def build(data: dict):
    style.apply()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    draw(axes, data)
    figure.subplots_adjust(**ADJUST)
    return figure


def main() -> int:
    data = load(TAG, HEADER)
    figure = build(data)
    style.save(figure, "decodability", OUT)
    plt.close(figure)
    joint = data["joint"]
    print(f"{len(data['cuts'])} cut points over {PROTEINS} proteins")
    print(
        f"measured peak {joint['peak']:+.4f} at {joint['peak_module']} block {joint['peak_at']}"
    )
    print(f"within-protein permutation threshold {data['null_mean']:+.4f}")
    print(f"released-head correlation {data['head']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
