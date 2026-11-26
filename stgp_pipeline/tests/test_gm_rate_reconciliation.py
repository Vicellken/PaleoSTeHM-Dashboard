"""Tests for canonical Gm loading, uncertainty, and figure consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.axes
import netCDF4 as nc
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stgp_pipeline.gm_rate import (
    GmRateSeries,
    assert_same_gm_series,
    load_gm_rate_netcdf,
    pointwise_peak_summary,
    posterior_peak_summary,
)
from stgp_pipeline.sensitivity.summary import _write_global_rate_plot


def _write_rate_nc(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    age = np.array([10_900.0, 11_100.0, 11_300.0, 11_500.0])
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("rate_time", len(age))
        ds.createVariable("rate_time", "f8", ("rate_time",))[:] = age
        ds.createVariable("global_rate_mean", "f8", ("rate_time",))[:] = mean
        ds.createVariable("global_rate_std", "f8", ("rate_time",))[:] = std
        ds.createVariable(
            "global_rate_cov", "f8", ("rate_time", "rate_time")
        )[:] = np.diag(std**2)


def test_pointwise_and_joint_peak_summaries(tmp_path: Path):
    path = tmp_path / "rate.nc"
    mean = -np.array([13.0, 14.0, 13.5, 12.0]) / 1000.0
    std = np.full(4, 0.5 / 1000.0)
    _write_rate_nc(path, mean, std)
    series = load_gm_rate_netcdf(path)

    pointwise = pointwise_peak_summary(
        series, age_lo_bp=11_000.0, age_hi_bp=11_500.0
    )
    assert pointwise["age_bp"] == 11_100.0
    assert np.isclose(pointwise["mean_mm_per_yr"], 14.0)
    assert np.isclose(pointwise["interval_lower_mm_per_yr"], 13.0)
    assert np.isclose(pointwise["interval_upper_mm_per_yr"], 15.0)

    posterior = posterior_peak_summary(
        series,
        age_lo_bp=11_000.0,
        age_hi_bp=11_500.0,
        draws=2000,
        seed=2,
    )
    assert posterior["statistic"] == "posterior_distribution_of_curve_maximum"
    assert posterior["draws"] == 2000
    assert posterior["credible_interval_95_mm_per_yr"][0] < 14.0
    assert posterior["credible_interval_95_mm_per_yr"][1] > 14.0


def test_same_series_requires_matching_figure_inputs():
    base = GmRateSeries(
        age_bp=np.array([100.0, 300.0]),
        mean_m_per_yr=np.array([-0.01, -0.02]),
        std_m_per_yr=np.array([0.001, 0.002]),
    )
    same = GmRateSeries(
        age_bp=base.age_bp.copy(),
        mean_m_per_yr=base.mean_m_per_yr.copy(),
        std_m_per_yr=base.std_m_per_yr.copy(),
    )
    assert_same_gm_series(base, same)


def test_sensitivity_plot_shades_two_sigma_for_every_curve(
    tmp_path: Path, monkeypatch
):
    mean = -np.array([13.0, 14.0, 13.5, 12.0]) / 1000.0
    std = np.full(4, 0.5 / 1000.0)
    adopted_path = tmp_path / "adopted" / "rate_adopted_16ka.nc"
    bounds_path = tmp_path / "bounds_m2" / "rate_bounds_m2_16ka.nc"
    _write_rate_nc(adopted_path, mean, std)
    _write_rate_nc(bounds_path, mean * 0.9, std * 1.2)

    fill_widths: list[np.ndarray] = []
    original = matplotlib.axes.Axes.fill_between

    def record_fill(self, x, y1, y2, *args, **kwargs):
        fill_widths.append(np.asarray(y2) - np.asarray(y1))
        return original(self, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", record_fill)
    records = [
        {
            "scenario": "adopted",
            "rate_nc_path": adopted_path,
            "row": {"bounds_multiplier": None},
        },
        {
            "scenario": "bounds_m2",
            "rate_nc_path": bounds_path,
            "row": {"bounds_multiplier": 2.0},
        },
    ]
    outputs = _write_global_rate_plot(records, tmp_path, workflow="16ka")

    assert outputs["global_rate_pdf"].exists()
    assert len(fill_widths) == 2
    assert np.allclose(fill_widths[0], 4.0 * std * 1000.0)
    assert np.allclose(fill_widths[1], 4.0 * std * 1.2 * 1000.0)
