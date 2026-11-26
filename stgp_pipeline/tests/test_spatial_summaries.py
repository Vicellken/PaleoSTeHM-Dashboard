"""Unit tests for stgp_pipeline.spatial_summaries (no torch/GP)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _clean_stgp_pipeline_import():
    pkg = sys.modules.get("stgp_pipeline")
    if pkg is not None and getattr(pkg, "__file__", None) is None:
        for name in [
            k
            for k in list(sys.modules)
            if k == "stgp_pipeline" or k.startswith("stgp_pipeline.")
        ]:
            del sys.modules[name]


def _make_rsl(lat, lon, region, region_name="Test"):
    """Minimal RSL-like object (avoids torch import for pure post-processing tests)."""
    region = np.asarray(region, dtype=int)
    return SimpleNamespace(
        rsl_lat=np.asarray(lat, dtype=float),
        rsl_lon=np.asarray(lon, dtype=float),
        rsl_region=region,
        unique_regions=np.sort(np.unique(region)),
        region_num_to_name={int(r): region_name for r in np.unique(region)},
    )


def _rate_grid_df():
    """Three-pixel grid with known GIA / regional_linear rates."""
    return pd.DataFrame(
        {
            "interval_start_bp": [0, 0, 0],
            "interval_end_bp": [4000, 4000, 4000],
            "midpoint_bp": [2000.0, 2000.0, 2000.0],
            "lat": [20.0, 20.2, 21.0],
            "lon": [110.0, 110.0, 110.0],
            "gia_rate_mm_yr": [1.0, 2.0, 3.0],
            "gia_rate_sd_mm_yr": [0.3, 0.4, 0.5],
            "regional_linear_rate_mm_yr": [0.5, -0.5, 0.0],
            "regional_linear_rate_sd_mm_yr": [0.4, 0.3, 0.2],
            "total_rate_mm_yr": [1.5, 1.5, 3.0],
            "total_rate_sd_mm_yr": [0.5, 0.5, 0.5],
        }
    )


def test_add_gia_plus_regional_linear_sum_and_rss():
    from stgp_pipeline import spatial_summaries as ss

    df = _rate_grid_df()
    out = ss.add_gia_plus_regional_linear(df)
    np.testing.assert_allclose(
        out["gia_plus_regional_linear_rate_mm_yr"],
        [1.5, 1.5, 3.0],
    )
    np.testing.assert_allclose(
        out["gia_plus_regional_linear_rate_sd_mm_yr"],
        [
            np.sqrt(0.3**2 + 0.4**2),
            np.sqrt(0.4**2 + 0.3**2),
            np.sqrt(0.5**2 + 0.2**2),
        ],
    )


def test_filter_sites_near_grid_kept_and_dropped():
    from stgp_pipeline import spatial_summaries as ss

    lat_flat = np.array([20.0, 20.2])
    lon_flat = np.array([110.0, 110.0])
    sites = pd.DataFrame(
        {
            "Name": ["Near", "Far"],
            "Lat": [20.01, 30.0],
            "Lon": [110.01, 120.0],
        }
    )
    filt = ss.filter_sites_near_grid(sites, lat_flat, lon_flat, radius_km=50.0)
    assert list(filt["status"]) == ["kept", "dropped"]
    assert filt.loc[0, "distance_km"] < 50.0
    assert filt.loc[1, "distance_km"] > 50.0
    assert filt.loc[0, "grid_lat"] == pytest.approx(20.0)
    assert np.isnan(filt.loc[1, "grid_lat"])


def test_extract_site_rates_snaps_to_grid():
    from stgp_pipeline import spatial_summaries as ss

    rate = ss.add_gia_plus_regional_linear(_rate_grid_df())
    filt = ss.filter_sites_near_grid(
        pd.DataFrame({"Name": ["A"], "Lat": [20.01], "Lon": [110.01]}),
        rate["lat"].to_numpy(),
        rate["lon"].to_numpy(),
        radius_km=50.0,
    )
    sites = ss.extract_site_rates(rate, filt)
    assert len(sites) == 1
    assert sites.iloc[0]["site"] == "A"
    assert sites.iloc[0]["gia_rate_mm_yr"] == pytest.approx(1.0)
    assert sites.iloc[0]["gia_plus_regional_linear_rate_mm_yr"] == pytest.approx(1.5)


def test_summarize_region_rates_nearest_slip_partitions_grid():
    from stgp_pipeline import spatial_summaries as ss

    rate = ss.add_gia_plus_regional_linear(_rate_grid_df())
    # Two regions: SLIP at 20.0 → region 1; SLIP at 21.0 → region 2.
    # Mid pixel at 20.2 is nearer to 20.0 → region 1.
    rsl = SimpleNamespace(
        rsl_lat=np.array([20.0, 21.0]),
        rsl_lon=np.array([110.0, 110.0]),
        rsl_region=np.array([1, 2]),
        unique_regions=np.array([1, 2]),
        region_num_to_name={1: "West", 2: "East"},
    )
    summary = ss.summarize_region_rates(rate, rsl)
    assert len(summary) == 2
    assert set(summary["mask_mode"]) == {"nearest_slip"}
    west = summary.loc[summary["region"] == 1].iloc[0]
    east = summary.loc[summary["region"] == 2].iloc[0]
    assert west["n_pixels"] == 2
    assert east["n_pixels"] == 1
    assert west["gia_rate_mm_yr_mean"] == pytest.approx(1.5)
    assert west["gia_rate_mm_yr_min"] == pytest.approx(1.0)
    assert west["gia_rate_mm_yr_max"] == pytest.approx(2.0)
    assert west["regional_linear_rate_mm_yr_min"] == pytest.approx(-0.5)
    assert west["regional_linear_rate_mm_yr_max"] == pytest.approx(0.5)
    assert east["gia_rate_mm_yr_mean"] == pytest.approx(3.0)
    assert east["gia_plus_regional_linear_rate_mm_yr_mean"] == pytest.approx(3.0)


def test_summarize_region_rates_single_region_all_pixels():
    from stgp_pipeline import spatial_summaries as ss

    rate = ss.add_gia_plus_regional_linear(_rate_grid_df())
    rsl = _make_rsl([20.0], [110.0], [1], region_name="West")
    summary = ss.summarize_region_rates(rate, rsl)
    row = summary.iloc[0]
    assert row["mask_mode"] == "nearest_slip"
    assert row["n_pixels"] == 3
    assert row["gia_rate_mm_yr_min"] == pytest.approx(1.0)
    assert row["gia_rate_mm_yr_max"] == pytest.approx(3.0)
    assert row["regional_linear_rate_mm_yr_mean"] == pytest.approx(0.0)
    assert row["gia_plus_regional_linear_rate_mm_yr_mean"] == pytest.approx(2.0)
