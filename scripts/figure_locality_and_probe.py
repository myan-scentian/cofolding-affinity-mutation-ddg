"""The signed substitution distogram beside the probe, plus the distance decay."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _limit in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_limit, "4")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "trunk_probe"))
sys.path.insert(0, str(Path(__file__).parent))


import distogram as pooled
import figure_decodability as probe_panels
import figure_distogram as distogram_panel
import probe

from mutation_ddg.analysis import decay, style
from mutation_ddg.paths import ARTIFACT_ROOT

BOLTZ = style.colour("Boltz-2")

WORK = ARTIFACT_ROOT / "work" / "trunk_probe"
FOLDED = ARTIFACT_ROOT / "results" / "boltz_hiqbind_unified"
OUT = ARTIFACT_ROOT / "results" / "figures"

SUMMARY = "distogram_locality_decomposition_primary163_s123.json"
COHORT = "primary"
ARM = "unified"
SCOPE = "cross/touched"


SHELLS = (
    ("cross/shell_0_4", "0–4"),
    ("cross/shell_4_8", "4–8"),
    ("cross/shell_8_12", "8–12"),
    ("cross/shell_12_16", "12–16"),
    ("cross/shell_16_24", "16–24"),
)


WILD, MUTANT = "3nu3_478", "3nu4_478"


REPEAT_SEEDS, MUTATION_SEED = (1, 2), 1

MAPS = (("repeat", "Re-fold"), ("mutation", "Substitution"))


TEXT_WIDTH = 5.50


INCLUDED_AT = 1.0


FIGURE_HEIGHT = 2.20
FIGSIZE = (TEXT_WIDTH * INCLUDED_AT, FIGURE_HEIGHT)


DISTOGRAM_RECT = (0.02, 0.095, 0.318, 0.795)
COLORBAR_RECT = (0.350, 0.095, 0.022, 0.795)
PROBE_RECT = (0.575, 0.15, 0.415, 0.67)


DECAY_INCLUDED_AT = 0.62
DECAY_FIGSIZE = (TEXT_WIDTH * DECAY_INCLUDED_AT, 2.55)


EXAMPLE_FIGSIZE = (4.40, 1.62)
EXAMPLE_INCLUDED_AT = 0.80
EXAMPLE_ADJUST = {
    "left": 0.075,
    "right": 0.975,
    "top": 0.83,
    "bottom": 0.25,
    "wspace": 0.24,
}


SHELL_X = (2.0, 6.0, 10.0, 14.0, 20.0)
SHELL_LABELS = ("0–4", "4–8", "8–12", "12–16", "16–24")


CURVE_POINTS = 200


def rows_by_name(table: list[dict]) -> dict[str, dict]:
    return {row["stratum"]: row for row in table}


def load_locality() -> tuple[dict[str, dict], dict]:
    """Load and cross-check the reported locality statistics."""
    summary = json.loads((WORK / SUMMARY).read_text())
    if (
        summary["inputs"]["pair_set"].get("cohort") != COHORT
        or summary["inputs"]["pair_set"].get("pairs") != 163
    ):
        raise SystemExit(
            f"{SUMMARY} was built on the "
            f"{summary['inputs']['pair_set'].get('cohort', 'unrecorded')} cohort, not "
            f"the 163-pair {COHORT} cohort"
        )

    combined = rows_by_name(summary["table"][ARM])
    named = Path(summary["inputs"]["per_arm_files"][ARM])
    per_arm = json.loads((WORK / named.name).read_text())
    if per_arm["seeds"] != summary["inputs"]["seeds"]:
        raise SystemExit(
            f"{named.name} has seeds {per_arm['seeds']}, "
            f"{SUMMARY} has {summary['inputs']['seeds']}"
        )
    if per_arm["pair_set"] != summary["inputs"]["pair_set"]:
        raise SystemExit(
            f"{named.name} was built on the "
            f"{per_arm['pair_set'].get('cohort', 'unrecorded')} cohort"
        )

    merged: dict[str, dict] = {}
    for name in dict.fromkeys((SCOPE, *(n for n, _ in SHELLS))):
        one, two = combined[name], per_arm["strata"][name]
        for field, left, right in (
            ("mutation median", one["mutation_median"], two["mutation"]["median"]),
            ("repeat median", one["repeat_median"], two["repeat"]["median"]),
            (
                "paired delta",
                one["paired_median_delta_pp"] / 100,
                two["paired"]["median_delta"],
            ),
        ):
            if abs(left - right) > 1e-9:
                raise SystemExit(
                    f"{name}: {field} is {left} in {SUMMARY} "
                    f"and {right} in {named.name}"
                )
        merged[name] = {
            "mutation": one["mutation_median"],
            "repeat": one["repeat_median"],
            "delta": one["paired_median_delta_pp"],
            "ci": one["paired_delta_ci95_pp"],
            "n_pairs": one["paired_n_pairs"],
        }
    return merged, summary


def record(seed: int, tag: str) -> dict:
    path = FOLDED / f"s{seed}" / "predictions" / tag / f"representations_{tag}.npz"
    data = np.load(path, allow_pickle=False)
    return {
        k: data[k]
        for k in (*pooled.WANTED, "ligand_mask", "n_cross_entries")
        if k in data.files
    }


def pooled_strip(a: dict, b: dict, mutated: set[tuple[int, int]]) -> dict:
    """Build a ligand-row view of the symmetric affinity-pooling mask."""
    rows = pooled.shared_rows(a, b)
    if rows is None:
        raise SystemExit(f"{WILD}/{MUTANT}: the two crops share too few tokens")
    left, right = rows[:, 0], rows[:, 1]
    size = len(left)
    differs = (
        pooled.bins(a["disto_coords"])[np.ix_(left, left)]
        != pooled.bins(b["disto_coords"])[np.ix_(right, right)]
    )

    receptor = a["receptor_mask"][left].astype(bool)
    ligand = a["ligand_mask"][left].astype(bool)
    if not (receptor ^ ligand).all():
        raise SystemExit("a token is both receptor and ligand, or neither")
    keys = [(int(x), int(r)) for x, r in zip(a["asym_id"], a["res_idx"])]
    is_mut = np.array([keys[i] in mutated for i in left]) & receptor

    off = ~np.eye(size, dtype=bool)
    cross = (
        (ligand[:, None] & receptor[None, :])
        | (receptor[:, None] & ligand[None, :])
        | (ligand[:, None] & ligand[None, :])
    )
    full = off & cross

    if size == len(a["receptor_mask"]) and "n_cross_entries" in a:
        recorded = int(a["n_cross_entries"])
        if recorded != int(full.sum()):
            raise SystemExit(
                f"cross mask holds {int(full.sum())} entries, the record "
                f"says the head pooled {recorded}"
            )

    lig_rows = np.flatnonzero(ligand)
    keep = full[lig_rows, :]
    changed = differs[lig_rows, :] & keep
    rate = changed.sum() / keep.sum()

    touched = keep & is_mut[None, :]
    both = off & cross & (is_mut[:, None] | is_mut[None, :])
    if int(touched.sum()) * 2 != int(both.sum()):
        raise SystemExit(
            f"the drawn map holds {int(touched.sum())} touching entries "
            f"against {int(both.sum())} in the symmetric mask"
        )
    if (
        abs(
            (changed & touched).sum() / touched.sum()
            - (differs & both).sum() / both.sum()
        )
        > 1e-12
    ):
        raise SystemExit(
            "the drawn map's touching entries are not the source's stratum"
        )
    return {
        "keep": keep,
        "changed": changed,
        "columns": np.flatnonzero(is_mut),
        "ligand_start": int(lig_rows.min()),
        "n_tokens": size,
        "n_ligand": len(lig_rows),
        "rate": float(rate),
        "rate_touched": float((changed & touched).sum() / touched.sum()),
        "n_entries": int(keep.sum()),
        "n_touched": int(touched.sum()),
    }


def load_maps() -> dict[str, dict]:
    """Both contrasts of the drawn example, over one token set."""
    wild = {seed: record(seed, WILD) for seed in (1, 2, 3, 4, 5)}
    mutant = {seed: record(seed, MUTANT) for seed in (1, 2, 3, 4, 5)}

    _, changed = probe.matched_positions(wild[MUTATION_SEED], mutant[MUTATION_SEED])
    keys = [
        (int(x), int(r))
        for x, r in zip(wild[MUTATION_SEED]["asym_id"], wild[MUTATION_SEED]["res_idx"])
    ]
    mutated = {keys[a] for a, _ in changed}
    if not mutated:
        raise SystemExit(f"{WILD} -> {MUTANT}: no substituted row is recorded")

    maps = {
        "repeat": pooled_strip(wild[REPEAT_SEEDS[0]], wild[REPEAT_SEEDS[1]], mutated),
        "mutation": pooled_strip(wild[MUTATION_SEED], mutant[MUTATION_SEED], mutated),
    }
    if maps["repeat"]["keep"].shape != maps["mutation"]["keep"].shape:
        raise SystemExit(
            "the two maps are over different token sets: "
            f"{maps['repeat']['keep'].shape} vs "
            f"{maps['mutation']['keep'].shape}"
        )
    if not np.array_equal(maps["repeat"]["columns"], maps["mutation"]["columns"]):
        raise SystemExit("the two maps mark different substituted columns")
    maps["_records"] = {"wild": wild, "mutant": mutant, "mutated": mutated}
    return maps


def draw_map(ax, panel: dict, label: str, bottom: bool) -> None:
    """Draw one binary changed-bin map."""
    ax.imshow(
        panel["changed"],
        cmap=style.binary_map(),
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax.axvline(panel["ligand_start"] - 0.5, color=style.MUTED, lw=0.45, alpha=0.8)
    style.panel_title(ax, label)
    ax.set_yticks([0, panel["n_ligand"] - 1])
    ax.set_yticklabels(
        ["1", str(panel["n_ligand"])], fontsize=style.TICK, color=style.INK
    )
    ax.set_xticks([0, panel["n_tokens"] - 1])
    if bottom:
        ax.set_xticklabels(
            ["1", str(panel["n_tokens"])], fontsize=style.TICK, color=style.INK
        )

        first_tick, last_tick = ax.get_xticklabels()
        first_tick.set_ha("left")
        last_tick.set_ha("right")

        ax.set_xlabel(
            "All tokens",
            fontsize=style.LABEL,
            color=style.INK,
            labelpad=probe_panels.XLABEL_PAD,
        )
    else:
        ax.set_xticklabels([])

    style.spines(ax, keep=("left", "right", "top", "bottom"))


def fit_decay(data: dict[str, dict]) -> decay.ExponentialFit:
    """Fit ``A exp(-d / lambda)`` to the five plotted distance shells."""
    if tuple(name for name, _ in SHELLS) != tuple(decay.SHELL_EDGES):
        raise SystemExit(
            f"this panel's shells {[n for n, _ in SHELLS]} are not "
            f"analysis.decay's {list(decay.SHELL_EDGES)}"
        )
    if SCOPE != decay.SITE_STRATUM:
        raise SystemExit(
            f"the locality scope is {SCOPE}, but analysis.decay excludes "
            f"{decay.SITE_STRATUM}"
        )

    if SHELL_X != tuple(sum(edges) / 2 for edges in decay.SHELL_EDGES.values()):
        raise SystemExit(
            f"the shells are drawn at {SHELL_X} and fitted at "
            f"{tuple(sum(e) / 2 for e in decay.SHELL_EDGES.values())}"
        )
    rows = {
        name: {
            "paired_median_delta_pp": data[name]["delta"],
            "paired_delta_ci95_pp": data[name]["ci"],
            "paired_n_pairs": data[name]["n_pairs"],
        }
        for name in decay.SHELL_EDGES
    }
    return decay.fit_exponential(decay.shells(rows))


def draw_decay(ax, data: dict[str, dict], fit: decay.ExponentialFit) -> None:
    """Draw paired substitution-minus-refold changed-bin rates by distance."""
    x = np.asarray(SHELL_X)
    delta = np.array([data[name]["delta"] for name, _ in SHELLS])
    ci = np.array([data[name]["ci"] for name, _ in SHELLS])

    style.reference_rule(ax, 0.0, line_colour=style.MUTED)
    grid = np.linspace(SHELL_X[0], SHELL_X[-1], CURVE_POINTS)
    ax.plot(grid, fit.at(grid), lw=style.LINE, color=BOLTZ, zorder=2)

    error = np.stack([delta - ci[:, 0], ci[:, 1] - delta])
    ax.errorbar(
        x,
        delta,
        yerr=error,
        color=BOLTZ,
        marker="o",
        linestyle="none",
        markersize=style.MARKER,
        elinewidth=0.7,
        capsize=1.7,
        capthick=0.7,
        alpha=0.95,
        zorder=3,
    )

    ax.set_xlim(1.0, 21.0)

    ax.set_ylim(-1.5, 15.8)
    ax.set_xticks(SHELL_X)
    ax.set_xticklabels(SHELL_LABELS, fontsize=style.ANNOTATION, color=style.INK)
    ax.set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    ax.set_xlabel(
        "Distance from substitution (Å)",
        fontsize=style.LABEL,
        color=style.INK,
        labelpad=2,
    )
    ax.set_ylabel(
        r"$\Delta$ changed-bin rate (pp)",
        fontsize=style.LABEL,
        color=style.INK,
        labelpad=2,
    )
    style.spines(ax)


def build(locality: dict[str, dict], distogram: dict):
    """The representative signed distogram beside the affinity-module probe."""
    style.apply()
    probe_data = probe_panels.load(probe_panels.TAG, probe_panels.HEADER)
    fig = plt.figure(figsize=FIGSIZE)

    map_ax = fig.add_axes(DISTOGRAM_RECT)
    image = distogram_panel.draw(map_ax, distogram)
    colour_ax = fig.add_axes(COLORBAR_RECT)
    distogram_panel.add_colorbar(fig, image, cax=colour_ax)

    probe_ax = fig.add_axes(PROBE_RECT)
    probe_panels.draw(probe_ax, probe_data)
    return fig


def build_decay(locality: dict[str, dict], fit: decay.ExponentialFit | None = None):
    """Build the standalone distance-decay figure."""
    style.apply()
    fig = plt.figure(figsize=DECAY_FIGSIZE)
    ax = fig.add_axes([0.19, 0.19, 0.77, 0.72])
    draw_decay(ax, locality, fit if fit is not None else fit_decay(locality))
    return fig


def main() -> int:
    locality, summary = load_locality()
    distogram = distogram_panel.load()
    fig = build(locality, distogram)

    style.save(fig, "locality_and_probe", OUT)
    plt.close(fig)
    print(f"wrote {OUT / 'locality_and_probe.pdf'} and .png")

    fitted = fit_decay(locality)
    decay_fig = build_decay(locality, fitted)
    style.save(decay_fig, "distance_decay", OUT)
    plt.close(decay_fig)
    print(f"wrote {OUT / 'distance_decay.pdf'} and .png")

    print(
        f"decay curve drawn over {SHELL_X[0]:.0f}–{SHELL_X[-1]:.0f} Å, "
        f"{decay.SITE_STRATUM} excluded: "
        f"A = {fitted.amplitude:.2f} ± {fitted.amplitude_se:.2f} pp, "
        f"lambda = {fitted.length:.2f} ± {fitted.length_se:.2f} Å, "
        f"half-distance {fitted.half_distance:.2f} Å, "
        f"chi2 = {fitted.chi2:.2f} on {fitted.dof} dof"
    )
    print(
        f"included at {DECAY_INCLUDED_AT:.2f}\\linewidth of a {TEXT_WIDTH:.2f} in text "
        f"block, drawn {DECAY_FIGSIZE[0]:.2f} x {DECAY_FIGSIZE[1]:.2f} in"
    )

    pair_set = summary["inputs"]["pair_set"]
    print(
        f"\n{ARM} arm, seeds {summary['inputs']['seeds']}: "
        f"{pair_set['pairs']} primary-cohort pairs"
    )
    print(
        f"{'stratum':17s} {'subst':>7s} {'re-fold':>8s} "
        f"{'delta':>7s} {'ci95':>16s} {'pairs':>6s}"
    )
    for name in dict.fromkeys((SCOPE, *(n for n, _ in SHELLS))):
        row = locality[name]
        print(
            f"{name:17s} {100 * row['mutation']:6.2f}% {100 * row['repeat']:7.2f}% "
            f"{row['delta']:+7.2f} [{row['ci'][0]:+6.2f},{row['ci'][1]:+6.2f}] "
            f"{row['n_pairs']:6d}"
        )

    shift = distogram["shift"]
    print(
        f"\ndrawn signed example {distogram_panel.REFERENCE} -> "
        f"{distogram_panel.VARIANT}: {shift.shape[0]} shared tokens, "
        f"{distogram['n_ligand']} ligand tokens, "
        f"{int(distogram['substituted'].sum())} substituted, "
        f"{int(np.isfinite(shift).sum())} of {shift.size} entries changed bin"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
