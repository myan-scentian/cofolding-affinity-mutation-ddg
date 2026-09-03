"""Fit the exponential decay reported for the five distogram distance shells."""

from __future__ import annotations

import json
import os
from pathlib import Path

for _limit in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_limit, "4")

from mutation_ddg.analysis import decay
from mutation_ddg.paths import ARTIFACT_ROOT

WORK = ARTIFACT_ROOT / "work" / "trunk_probe"
SUMMARY = "distogram_locality_decomposition_primary163_s123.json"
COHORT = "primary"
ARM = "unified"
OUT = WORK / "distogram_decay_fit_primary163_s123.json"


def load(path: Path) -> tuple[dict[str, dict], dict]:
    summary = json.loads(path.read_text())
    pair_set = summary["inputs"]["pair_set"]
    if pair_set.get("cohort") != COHORT or pair_set.get("pairs") != 163:
        raise SystemExit(f"{path.name} is not the {COHORT} cohort: {pair_set}")
    return {row["stratum"]: row for row in summary["table"][ARM]}, summary


def main() -> int:
    rows, summary = load(WORK / SUMMARY)
    data = decay.shells(rows)
    fit = decay.fit_exponential(data)
    offset, offset_se = decay.fit_offset(data)
    stop = decay.first_indistinguishable_shell(data)

    site = rows[decay.SITE_STRATUM]
    print(
        f"{decay.SITE_STRATUM} (categorical, excluded from the fit): "
        f"{site['paired_median_delta_pp']:+.2f} pp over "
        f"{site['paired_n_pairs']} pairs\n"
    )
    print(
        f"{'shell':18s} {'mid':>5s} {'delta':>7s} {'ci95':>18s} {'sd':>6s} "
        f"{'pairs':>6s} {'fitted':>7s} {'resid/sd':>9s}"
    )
    for index, name in enumerate(data.names):
        print(
            f"{name:18s} {data.distance[index]:5.1f} {data.delta[index]:+7.2f} "
            f"[{data.ci[index][0]:+7.2f},{data.ci[index][1]:+7.2f}] "
            f"{data.sd[index]:6.2f} {data.n_pairs[index]:6d} "
            f"{float(fit.at(data.distance[index])):7.2f} "
            f"{fit.residuals_in_sd[index]:+9.2f}"
        )

    print(
        f"\nA exp(-d/lambda) over {len(data.names)} shells, weighted by the intervals"
    )
    print(
        f"  A       = {fit.amplitude:6.2f} +- {fit.amplitude_se:.2f} pp "
        f"(Monte Carlo 95% [{fit.amplitude_ci[0]:.2f}, {fit.amplitude_ci[1]:.2f}])"
    )
    print(
        f"  lambda  = {fit.length:6.2f} +- {fit.length_se:.2f} A "
        f"(Monte Carlo 95% [{fit.length_ci[0]:.2f}, {fit.length_ci[1]:.2f}], "
        f"{fit.draws} draws)"
    )
    print(f"  half-distance lambda ln2 = {fit.half_distance:.2f} A")
    print(f"  chi2 = {fit.chi2:.2f} on {fit.dof} dof (p = {fit.gof_p:.3f})")
    print(f"  fitted excess under 1 pp beyond {fit.below(1.0):.1f} A")
    print(
        f"  constant term of A exp(-d/lambda) + c: {offset:+.2f} +- {offset_se:.2f} pp"
    )
    if stop is None:
        print("  every shell's interval excludes zero")
    else:
        name, edge = stop
        print(
            f"  innermost shell whose interval covers zero: {name} "
            f"(from {edge:.0f} A), delta "
            f"{rows[name]['paired_median_delta_pp']:+.2f} pp"
        )

    record = {
        "generated_by": "scripts/decay_fit.py",
        "source": SUMMARY,
        "cohort": summary["inputs"]["pair_set"],
        "seeds": summary["inputs"]["seeds"],
        "arm": ARM,
        "excluded": {
            decay.SITE_STRATUM: "categorical: its entries have a substituted residue at one end and "
            "therefore no distance to one"
        },
        "shells": [
            {
                "stratum": name,
                "midpoint_a": float(data.distance[i]),
                "delta_pp": float(data.delta[i]),
                "ci95_pp": [float(data.ci[i][0]), float(data.ci[i][1])],
                "sd_pp": float(data.sd[i]),
                "n_pairs": int(data.n_pairs[i]),
                "fitted_pp": float(fit.at(data.distance[i])),
                "residual_in_sd": float(fit.residuals_in_sd[i]),
            }
            for i, name in enumerate(data.names)
        ],
        "fit": {
            "form": "A * exp(-d / lambda)",
            "amplitude_pp": fit.amplitude,
            "amplitude_se_pp": fit.amplitude_se,
            "amplitude_ci95_pp": list(fit.amplitude_ci),
            "length_a": fit.length,
            "length_se_a": fit.length_se,
            "length_ci95_a": list(fit.length_ci),
            "half_distance_a": fit.half_distance,
            "chi2": fit.chi2,
            "dof": fit.dof,
            "gof_p": fit.gof_p,
            "monte_carlo_draws": fit.draws,
            "distance_below_1pp_a": fit.below(1.0),
            "offset_check_pp": offset,
            "offset_check_se_pp": offset_se,
        },
        "first_indistinguishable_shell": None
        if stop is None
        else {
            "stratum": stop[0],
            "inner_edge_a": stop[1],
            "delta_pp": rows[stop[0]]["paired_median_delta_pp"],
            "ci95_pp": rows[stop[0]]["paired_delta_ci95_pp"],
            "n_pairs": rows[stop[0]]["paired_n_pairs"],
        },
    }
    OUT.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
