"""Tests for sensitivity workflow NetCDF helpers and summary rate plots."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stgp_pipeline.sensitivity.summary import _write_regional_linear_rate_plot
from stgp_pipeline.sensitivity.workflow_common import (
    DEFAULT_SLIP_REGION_IDS,
    build_temporal_netcdf_arrays,
    compute_nc_row_to_region_id,
    filter_nc_rows_by_region_ids,
    peak_global_rate_mm_yr,
    peak_rate_age_window,
    write_rate_netcdf,
    workflow_nc_suffix,
)


@dataclass
class _FakeComp:
    mean: np.ndarray
    std: np.ndarray
    rate: np.ndarray
    rate_sd: np.ndarray


@dataclass
class _FakeRegionResult:
    region_name: str
    test_age: np.ndarray
    y_total: np.ndarray
    y_std: np.ndarray
    rsl_time: np.ndarray
    rsl_rate: np.ndarray
    rsl_rate_sd: np.ndarray
    components: dict = field(default_factory=dict)


def _make_fake_temporal_results():
    test_age = np.array([4000.0, 3800.0, 3600.0])
    rate_time = np.array([3900.0, 3700.0])
    global_mean = np.array([0.1, 0.2, 0.3])
    global_rate = np.array([0.01, 0.02])
    rl_mean = np.array([0.05, 0.06, 0.07])
    rl_rate = np.array([0.001, 0.002])

    def region(rid: int, name: str):
        return _FakeRegionResult(
            region_name=name,
            test_age=test_age,
            y_total=np.linspace(1.0, 3.0, 3),
            y_std=np.full(3, 0.1),
            rsl_time=rate_time,
            rsl_rate=np.array([0.02, 0.03]),
            rsl_rate_sd=np.array([0.004, 0.005]),
            components={
                "global": _FakeComp(
                    mean=global_mean,
                    std=np.full(3, 0.01),
                    rate=global_rate,
                    rate_sd=np.full(2, 0.001),
                ),
                "regional_linear": _FakeComp(
                    mean=rl_mean,
                    std=np.full(3, 0.02),
                    rate=rl_rate,
                    rate_sd=np.full(2, 0.0005),
                ),
            },
        )

    return {1: region(1, "North"), 2: region(2, "South")}


def test_build_temporal_netcdf_arrays_includes_regional_linear():
    arrays = build_temporal_netcdf_arrays(_make_fake_temporal_results())
    assert "regional_linear" in arrays["component_rate_mean"]
    assert arrays["component_rate_mean"]["regional_linear"].shape == (2, 2)
    assert np.allclose(
        arrays["global_rate_mean"],
        np.array([0.01, 0.02]),
    )


def test_peak_rate_age_window_by_workflow():
    assert peak_rate_age_window("4ka") == (0.0, 4000.0)
    assert peak_rate_age_window("regional-gia") == (0.0, 5000.0)
    assert peak_rate_age_window("16ka") == (9000.0, 13000.0)


def test_peak_global_rate_mm_yr_4ka_window():
    rate_time = np.array([500.0, 1500.0, 2500.0])
    # Stored rates follow STGP sign convention (negate for mm/yr display).
    global_rate = np.array([-0.01, -0.03, -0.02])
    peak = peak_global_rate_mm_yr(rate_time, global_rate, workflow="4ka")
    assert np.isclose(peak, 30.0)


def test_compute_nc_row_to_region_id_from_csv(tmp_path: Path):
    csv_path = tmp_path / "paleo.csv"
    pd.DataFrame(
        {
            "Region.1": [0, 1, 1, 2, 3],
            "Region": ["all", "A", "A", "B", "C"],
            "Age": [1000, 1100, 1200, 1300, 1400],
            "RSL": [0, 0, 0, 0, 0],
        }
    ).to_csv(csv_path, index=False)

    mapping, summary_rows = compute_nc_row_to_region_id(3, csv_path)
    assert mapping is not None
    assert list(mapping) == [1, 2, 3]
    assert summary_rows == [0, 1, 2]


def test_compute_nc_row_to_region_id_uses_slip_only_when_limiting_present(
    tmp_path: Path,
):
    """Multi-region CSV: many Region.1 codes, but only six SLIP training regions."""
    csv_path = tmp_path / "paleo.csv"
    rows = []
    for code in range(1, 11):
        rows.append(
            {"Region.1": code, "Limiting": 0 if code in (3, 4, 6, 7, 8, 9) else 1}
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mapping, summary_rows = compute_nc_row_to_region_id(6, csv_path)
    assert list(mapping) == list(DEFAULT_SLIP_REGION_IDS)
    assert summary_rows == list(range(6))


def test_workflow_nc_suffix():
    assert workflow_nc_suffix("4ka") == "4ka"
    assert workflow_nc_suffix("16ka") == "16ka"
    assert workflow_nc_suffix("3d-gia") == "3d_gia"
    assert workflow_nc_suffix("regional-gia") == "regional_gia"


def test_write_rsl_netcdf_skips_duplicate_global_component(tmp_path: Path):
    from stgp_pipeline.sensitivity.workflow_common import write_rsl_netcdf

    test_age = np.array([4000.0, 0.0])
    path = tmp_path / "rsl_test.nc"
    write_rsl_netcdf(
        path,
        test_age=test_age,
        region_names=["R1"],
        total_mean=np.zeros((1, 2)),
        total_std=np.zeros((1, 2)),
        global_mean=np.array([0.1, 0.2]),
        global_std=np.array([0.01, 0.02]),
        component_mean={
            "global": np.zeros((1, 2)),
            "regional_linear": np.ones((1, 2)),
        },
        component_std={
            "global": np.zeros((1, 2)),
            "regional_linear": np.full((1, 2), 0.1),
        },
    )
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        var_names = set(ds.variables.keys())
        assert "global_mean" in var_names
        assert "regional_linear_mean" in var_names
        assert "global_mean" not in {
            n for n in var_names if n.startswith("global_") and n != "global_mean"
        }


def test_filter_nc_rows_by_region_ids_orders_by_region_ids():
    mapping = np.array([1, 3, 4, 6, 7, 8, 9, 10], dtype=int)
    non_zero_rows = list(range(7))
    filtered = filter_nc_rows_by_region_ids(
        mapping, non_zero_rows, DEFAULT_SLIP_REGION_IDS
    )
    assert filtered == [1, 2, 3, 4, 5, 6]
    assert [int(mapping[i]) for i in filtered] == list(DEFAULT_SLIP_REGION_IDS)


def test_filter_nc_rows_by_region_ids_without_mapping():
    assert filter_nc_rows_by_region_ids(None, [0, 1, 2], (3, 4)) == [0, 1, 2]


def test_write_regional_linear_rate_plot_4ka_six_slip_panels(tmp_path: Path):
    rate_time = np.array([4000.0, 2000.0, 0.0])
    n_regions = 8
    rl_mean = np.zeros((n_regions, 3))
    rl_std = np.zeros((n_regions, 3))
    rate_nc = tmp_path / "adopted" / "rate_adopted_4ka.nc"
    rate_nc.parent.mkdir(parents=True)
    write_rate_netcdf(
        rate_nc,
        test_age=rate_time,
        rate_time=rate_time,
        region_names=[f"R{i}" for i in range(n_regions)],
        total_rate_mean=rl_mean,
        total_rate_std=rl_std,
        global_rate_mean=np.zeros(3),
        global_rate_std=np.zeros(3),
        component_rate_mean={"regional_linear": rl_mean + 0.01},
        component_rate_std={"regional_linear": rl_std + 0.001},
    )

    paleo_csv = tmp_path / "paleo.csv"
    # Eight NetCDF rows: SLIP codes for panels; extra codes only on limiting rows.
    slip_codes = [1, 3, 4, 5, 6, 7, 8, 9]
    pd.DataFrame(
        {
            "Region.1": slip_codes + [10],
            "Limiting": [0] * len(slip_codes) + [1],
        }
    ).to_csv(paleo_csv, index=False)
    (rate_nc.parent / "config.yaml").write_text(
        f"paths:\n  paleo_rsl: {paleo_csv}\n",
        encoding="utf-8",
    )

    scenario_records = [
        {
            "scenario": "adopted",
            "rate_nc_path": rate_nc,
            "row": {
                "workflow": "4ka",
                "bounds_multiplier": None,
                "output_dir": str(rate_nc.parent),
            },
        }
    ]
    outputs = _write_regional_linear_rate_plot(
        scenario_records, tmp_path, workflow="4ka"
    )
    assert outputs["regional_linear_rate_pdf"].exists()

    index_df = pd.read_csv(tmp_path / "summary_region_index.csv")
    assert index_df["in_linear_rate_panels"].sum() == 6
    assert set(
        index_df.loc[index_df["in_linear_rate_panels"], "region_index"].astype(int)
    ) == set(DEFAULT_SLIP_REGION_IDS)


def test_write_regional_linear_rate_plot(tmp_path: Path):
    rate_time = np.array([4000.0, 2000.0, 0.0])
    rl_mean = np.zeros((2, 3))
    rl_std = np.zeros((2, 3))
    rate_nc = tmp_path / "adopted" / "rate_adopted_4ka.nc"
    rate_nc.parent.mkdir(parents=True)
    write_rate_netcdf(
        rate_nc,
        test_age=rate_time,
        rate_time=rate_time,
        region_names=["R1", "R2"],
        total_rate_mean=rl_mean,
        total_rate_std=rl_std,
        global_rate_mean=np.zeros(3),
        global_rate_std=np.zeros(3),
        component_rate_mean={"regional_linear": rl_mean + 0.01},
        component_rate_std={"regional_linear": rl_std + 0.001},
    )

    scenario_records = [
        {
            "scenario": "adopted",
            "rate_nc_path": rate_nc,
            "row": {
                "workflow": "4ka",
                "bounds_multiplier": None,
                "output_dir": str(rate_nc.parent),
            },
        }
    ]
    outputs = _write_regional_linear_rate_plot(
        scenario_records, tmp_path, workflow="4ka"
    )
    assert "regional_linear_rate_pdf" in outputs
    assert outputs["regional_linear_rate_pdf"].exists()


def test_write_regional_linear_rate_plot_skips_without_variable(tmp_path: Path):
    rate_nc = tmp_path / "adopted" / "rate_adopted_4ka.nc"
    rate_nc.parent.mkdir(parents=True)
    write_rate_netcdf(
        rate_nc,
        test_age=np.array([4000.0, 0.0]),
        rate_time=np.array([2000.0]),
        region_names=["R1"],
        total_rate_mean=np.zeros((1, 1)),
        total_rate_std=np.zeros((1, 1)),
        global_rate_mean=np.zeros(1),
        global_rate_std=np.zeros(1),
    )
    outputs = _write_regional_linear_rate_plot([], tmp_path, workflow="4ka")
    assert outputs == {}
