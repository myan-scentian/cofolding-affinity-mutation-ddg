"""How fast the distogram's substitution excess falls off with distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

SHELL_EDGES: dict[str, tuple[float, float]] = {
    "cross/shell_0_4": (0.0, 4.0),
    "cross/shell_4_8": (4.0, 8.0),
    "cross/shell_8_12": (8.0, 12.0),
    "cross/shell_12_16": (12.0, 16.0),
    "cross/shell_16_24": (16.0, 24.0),
}

SITE_STRATUM = "cross/touched"
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Shells:
    """The shells a fit reads: one row per stratum, in the order of SHELL_EDGES."""

    names: tuple[str, ...]
    distance: np.ndarray
    delta: np.ndarray
    ci: np.ndarray
    sd: np.ndarray
    n_pairs: np.ndarray


@dataclass(frozen=True)
class ExponentialFit:
    """``A exp(-d / lambda)`` over the shells, with both kinds of interval."""

    amplitude: float
    amplitude_se: float
    amplitude_ci: tuple[float, float]
    length: float
    length_se: float
    length_ci: tuple[float, float]
    half_distance: float
    chi2: float
    dof: int
    gof_p: float
    residuals_in_sd: np.ndarray
    draws: int

    def at(self, distance) -> np.ndarray:
        return self.amplitude * np.exp(-np.asarray(distance, float) / self.length)

    def below(self, level: float) -> float:
        """Distance beyond which the fitted excess is under ``level`` percentage points."""
        return float(self.length * np.log(self.amplitude / level))


def shells(rows: dict[str, dict]) -> Shells:
    """Load distance shells from a locality-decomposition table."""
    missing = [name for name in SHELL_EDGES if name not in rows]
    if missing:
        raise KeyError(f"the decomposition has no {missing}")
    names = tuple(SHELL_EDGES)
    ci = np.array([rows[n]["paired_delta_ci95_pp"] for n in names], float)
    if (ci[:, 1] < ci[:, 0]).any():
        raise ValueError("a bootstrap interval is inverted")
    return Shells(
        names=names,
        distance=np.array([sum(SHELL_EDGES[n]) / 2 for n in names], float),
        delta=np.array([rows[n]["paired_median_delta_pp"] for n in names], float),
        ci=ci,
        sd=(ci[:, 1] - ci[:, 0]) / (2 * Z95),
        n_pairs=np.array([rows[n]["paired_n_pairs"] for n in names], int),
    )


def _exponential(d, amplitude, length):
    return amplitude * np.exp(-d / length)


def fit_exponential(
    data: Shells, *, draws: int = 20000, seed: int = 0
) -> ExponentialFit:
    """Weighted non-linear least squares, with a Monte Carlo interval beside it."""
    if len(data.distance) < 3:
        raise ValueError(
            f"{len(data.distance)} shells cannot support a two-parameter "
            "fit with a residual degree of freedom"
        )
    guess = [float(data.delta[0]), 5.0]
    popt, pcov = optimize.curve_fit(
        _exponential,
        data.distance,
        data.delta,
        p0=guess,
        bounds=(0.0, np.inf),
        sigma=data.sd,
        absolute_sigma=True,
        maxfev=100000,
    )
    perr = np.sqrt(np.diag(pcov))
    residuals = (data.delta - _exponential(data.distance, *popt)) / data.sd
    chi2 = float((residuals**2).sum())
    dof = len(data.distance) - len(popt)

    rng = np.random.default_rng(seed)
    kept = []
    for _ in range(draws):
        drawn = data.delta + rng.normal(0.0, data.sd)
        try:
            got, _ = optimize.curve_fit(
                _exponential,
                data.distance,
                drawn,
                p0=popt,
                bounds=(0.0, np.inf),
                sigma=data.sd,
                absolute_sigma=True,
                maxfev=100000,
            )
        except RuntimeError:
            continue

        if got[0] > 0 and 0 < got[1] < 500:
            kept.append(got)
    sample = np.array(kept)
    lo_a, hi_a = np.percentile(sample[:, 0], [2.5, 97.5])
    lo_l, hi_l = np.percentile(sample[:, 1], [2.5, 97.5])
    return ExponentialFit(
        amplitude=float(popt[0]),
        amplitude_se=float(perr[0]),
        amplitude_ci=(float(lo_a), float(hi_a)),
        length=float(popt[1]),
        length_se=float(perr[1]),
        length_ci=(float(lo_l), float(hi_l)),
        half_distance=float(popt[1] * np.log(2)),
        chi2=chi2,
        dof=dof,
        gof_p=float(1 - stats.chi2.cdf(chi2, dof)),
        residuals_in_sd=residuals,
        draws=len(sample),
    )


def fit_offset(data: Shells) -> tuple[float, float]:
    """Fit the constant in ``A exp(-d / lambda) + c``."""
    popt, pcov = optimize.curve_fit(
        lambda d, a, length, c: a * np.exp(-d / length) + c,
        data.distance,
        data.delta,
        p0=[float(data.delta[0]), 5.0, 0.0],
        bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        sigma=data.sd,
        absolute_sigma=True,
        maxfev=100000,
    )
    return float(popt[2]), float(np.sqrt(np.diag(pcov))[2])


def first_indistinguishable_shell(data: Shells) -> tuple[str, float] | None:
    """The innermost shell whose bootstrap interval covers zero, with its inner edge."""
    for index, name in enumerate(data.names):
        low, high = data.ci[index]
        if low <= 0.0 <= high:
            return name, SHELL_EDGES[name][0]
    return None
