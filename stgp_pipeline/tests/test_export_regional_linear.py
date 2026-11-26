from __future__ import annotations

import sys
from pathlib import Path

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


def test_regional_linear_rates_to_dataframe_columns_and_mm_yr():
    from stgp_pipeline import export

    df_rates = pd.DataFrame(
        {
            "region": [1, 1, 3],
            "region_name": ["A", "A", "B"],
            "age_bp": [100.0, 300.0, 100.0],
            "regional_linear_rate": [0.001, -0.002, 0.0005],
            "regional_linear_rate_sd": [0.0001, 0.0002, 0.00005],
            "total_rate": [0.0, 0.0, 0.0],
        }
    )
    out = export.regional_linear_rates_to_dataframe(df_rates)
    assert len(out) == 3
    assert list(out.columns) == [
        "region",
        "region_name",
        "age_bp",
        "regional_linear_rate_m_yr",
        "regional_linear_rate_sd_m_yr",
        "regional_linear_rate_mm_yr",
        "regional_linear_rate_sd_mm_yr",
    ]
    assert out.loc[0, "regional_linear_rate_m_yr"] == pytest.approx(0.001)
    assert out.loc[0, "regional_linear_rate_mm_yr"] == pytest.approx(-1.0)
    assert out.loc[0, "regional_linear_rate_sd_mm_yr"] == pytest.approx(0.1)
    assert out.loc[1, "regional_linear_rate_mm_yr"] == pytest.approx(2.0)


def test_regional_linear_rates_to_dataframe_missing_columns():
    from stgp_pipeline import export

    with pytest.raises(KeyError, match="regional_linear"):
        export.regional_linear_rates_to_dataframe(
            pd.DataFrame({"region": [1], "age_bp": [0.0]})
        )


def test_regional_mean_export_reports_one_and_two_sigma():
    from stgp_pipeline import export
    from stgp_pipeline.types import RegionalMeanRateResult

    gia_sd = 0.0002
    component_sd = 0.0003
    combined_sd = float(np.hypot(gia_sd, component_sd))
    result = RegionalMeanRateResult(
        region_num=1,
        region_name="A",
        n_pixels=4,
        mask_mode="nearest_slip",
        weighting="equal_grid_cell",
        younger_bp=0.0,
        older_bp=4000.0,
        midpoint_bp=2000.0,
        gia_rate=0.001,
        gia_rate_sd=gia_sd,
        component_rates={"regional_linear": -0.0005},
        component_rate_sds={"regional_linear": component_sd},
        combined_rate=0.0005,
        combined_rate_sd=combined_sd,
    )

    row = export.regional_mean_rates_to_dataframe([result]).iloc[0]
    assert row["gia_rate_mm_yr"] == pytest.approx(-1.0)
    assert row["gia_rate_sd_of_regional_mean_mm_yr"] == pytest.approx(0.2)
    assert row["gia_rate_2sigma_of_regional_mean_mm_yr"] == pytest.approx(0.4)
    assert row["regional_linear_rate_mm_yr"] == pytest.approx(0.5)
    assert row[
        "regional_linear_rate_2sigma_of_regional_mean_mm_yr"
    ] == pytest.approx(0.6)
    assert row[
        "gia_plus_regional_linear_rate_sd_of_regional_mean_mm_yr"
    ] == pytest.approx(combined_sd * 1000.0)
