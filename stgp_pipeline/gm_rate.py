"""Canonical loading, plotting, and peak summaries for the Gm rate component."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


@dataclass(frozen=True)
class GmRateSeries:
    """Posterior Gm rate on its native finite-difference time grid."""

    age_bp: np.ndarray
    mean_m_per_yr: np.ndarray
    std_m_per_yr: np.ndarray
    covariance_m2_per_yr2: np.ndarray | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        age = np.asarray(self.age_bp, dtype=float)
        mean = np.asarray(self.mean_m_per_yr, dtype=float)
        std = np.asarray(self.std_m_per_yr, dtype=float)
        if age.ndim != 1 or mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Gm age, mean, and standard deviation must be 1D")
        if not (age.shape == mean.shape == std.shape):
            raise ValueError(
                "Gm age, mean, and standard deviation must have matching shapes"
            )
        if np.any(std < 0):
            raise ValueError("Gm standard deviations must be non-negative")

        covariance = self.covariance_m2_per_yr2
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            expected = (len(age), len(age))
            if covariance.shape != expected:
                raise ValueError(
                    f"Gm covariance shape {covariance.shape} does not match {expected}"
                )

        object.__setattr__(self, "age_bp", age)
        object.__setattr__(self, "mean_m_per_yr", mean)
        object.__setattr__(self, "std_m_per_yr", std)
        object.__setattr__(self, "covariance_m2_per_yr2", covariance)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())

    @property
    def mean_mm_per_yr(self) -> np.ndarray:
        """Sea-level-rise plotting convention: negate m/yr and convert to mm/yr."""
        return -self.mean_m_per_yr * 1000.0

    @property
    def std_mm_per_yr(self) -> np.ndarray:
        return self.std_m_per_yr * 1000.0


def load_gm_rate_netcdf(path: str | Path) -> GmRateSeries:
    """Load the canonical Gm posterior arrays from a sensitivity rate NetCDF."""
    rate_path = Path(path).expanduser().resolve()
    with nc.Dataset(rate_path) as ds:
        required = ("rate_time", "global_rate_mean", "global_rate_std")
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise KeyError(f"{rate_path} is missing variables: {', '.join(missing)}")
        covariance = (
            np.asarray(ds.variables["global_rate_cov"][:], dtype=float)
            if "global_rate_cov" in ds.variables
            else None
        )
        return GmRateSeries(
            age_bp=np.asarray(ds.variables["rate_time"][:], dtype=float),
            mean_m_per_yr=np.asarray(
                ds.variables["global_rate_mean"][:], dtype=float
            ),
            std_m_per_yr=np.asarray(
                ds.variables["global_rate_std"][:], dtype=float
            ),
            covariance_m2_per_yr2=covariance,
            source_path=rate_path,
        )


def plot_gm_rate(
    ax: Any,
    series: GmRateSeries,
    *,
    label: str,
    color: str,
    linestyle: Any = "-",
    linewidth: float = 2.0,
    uncertainty_sigma: float | None = 2.0,
    uncertainty_alpha: float = 0.12,
    age_scale: float = 1.0,
) -> Any:
    """Plot Gm mean and a pointwise Gaussian uncertainty envelope."""
    x = series.age_bp / float(age_scale)
    mean = series.mean_mm_per_yr
    line = ax.plot(
        x,
        mean,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )[0]
    if uncertainty_sigma is not None:
        width = float(uncertainty_sigma) * series.std_mm_per_yr
        ax.fill_between(
            x,
            mean - width,
            mean + width,
            color=color,
            alpha=uncertainty_alpha,
            linewidth=0,
        )
    return line


def pointwise_peak_summary(
    series: GmRateSeries,
    *,
    age_lo_bp: float,
    age_hi_bp: float,
    interval_sigma: float = 2.0,
) -> dict[str, float | list[float] | str]:
    """Maximum of the posterior-mean curve with its pointwise interval."""
    lo, hi = sorted((float(age_lo_bp), float(age_hi_bp)))
    mask = (series.age_bp >= lo) & (series.age_bp <= hi)
    if not np.any(mask):
        raise ValueError(f"No Gm rate coordinates fall inside {lo:g}–{hi:g} BP")

    candidates = np.flatnonzero(mask)
    index = int(candidates[np.argmax(series.mean_mm_per_yr[mask])])
    mean = float(series.mean_mm_per_yr[index])
    sd = float(series.std_mm_per_yr[index])
    half_width = float(interval_sigma) * sd
    return {
        "statistic": "maximum_of_posterior_mean_curve",
        "uncertainty": "pointwise_conditional_gaussian_interval_at_selected_age",
        "window_bp": [lo, hi],
        "age_bp": float(series.age_bp[index]),
        "mean_mm_per_yr": mean,
        "posterior_sd_mm_per_yr": sd,
        "interval_sigma": float(interval_sigma),
        "interval_half_width_mm_per_yr": half_width,
        "interval_lower_mm_per_yr": mean - half_width,
        "interval_upper_mm_per_yr": mean + half_width,
    }


def posterior_peak_summary(
    series: GmRateSeries,
    *,
    age_lo_bp: float,
    age_hi_bp: float,
    draws: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample the joint conditional rate posterior and summarize curve maxima."""
    if series.covariance_m2_per_yr2 is None:
        raise ValueError("The Gm NetCDF does not contain global_rate_cov")
    if draws < 1:
        raise ValueError("draws must be positive")

    lo, hi = sorted((float(age_lo_bp), float(age_hi_bp)))
    mask = (series.age_bp >= lo) & (series.age_bp <= hi)
    if not np.any(mask):
        raise ValueError(f"No Gm rate coordinates fall inside {lo:g}–{hi:g} BP")

    ages = series.age_bp[mask]
    mean = series.mean_mm_per_yr[mask]
    covariance = (
        series.covariance_m2_per_yr2[np.ix_(mask, mask)] * 1_000_000.0
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped_count = int(np.count_nonzero(eigenvalues < 0))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    transform = eigenvectors * np.sqrt(eigenvalues)

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((int(draws), len(ages))) @ transform.T + mean
    peak_indices = np.argmax(samples, axis=1)
    peak_rates = samples[np.arange(int(draws)), peak_indices]
    peak_ages = ages[peak_indices]

    unique_ages, counts = np.unique(peak_ages, return_counts=True)
    age_probabilities = {
        f"{float(age):g}": float(count / draws)
        for age, count in zip(unique_ages, counts)
    }
    return {
        "statistic": "posterior_distribution_of_curve_maximum",
        "conditioning": "optimized_hyperparameters",
        "window_bp": [lo, hi],
        "draws": int(draws),
        "seed": int(seed),
        "mean_mm_per_yr": float(np.mean(peak_rates)),
        "median_mm_per_yr": float(np.median(peak_rates)),
        "sd_mm_per_yr": float(np.std(peak_rates, ddof=1)),
        "credible_interval_95_mm_per_yr": [
            float(np.quantile(peak_rates, 0.025)),
            float(np.quantile(peak_rates, 0.975)),
        ],
        "median_peak_age_bp": float(np.median(peak_ages)),
        "credible_interval_95_peak_age_bp": [
            float(np.quantile(peak_ages, 0.025)),
            float(np.quantile(peak_ages, 0.975)),
        ],
        "peak_age_probabilities": age_probabilities,
        "negative_covariance_eigenvalues_clipped": clipped_count,
    }


def assert_same_gm_series(
    left: GmRateSeries,
    right: GmRateSeries,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Require two figure inputs to use identical Gm coordinates and moments."""
    np.testing.assert_allclose(left.age_bp, right.age_bp, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        left.mean_m_per_yr, right.mean_m_per_yr, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        left.std_m_per_yr, right.std_m_per_yr, rtol=rtol, atol=atol
    )
