"""Tests for sensitivity workflow summary builders."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stgp_pipeline.sensitivity import summary as _SUMMARY

_parameter_row_order = _SUMMARY._parameter_row_order
_adopted_regional_nonlinear_enabled = _SUMMARY._adopted_regional_nonlinear_enabled
_boundary_heatmap_scenario_order = _SUMMARY._boundary_heatmap_scenario_order
_should_drop_no_regional_nonlinear_from_heatmap = (
    _SUMMARY._should_drop_no_regional_nonlinear_from_heatmap
)
build_params_bounds_nill_long = _SUMMARY.build_params_bounds_nill_long
write_params_bounds_nill_markdown = _SUMMARY.write_params_bounds_nill_markdown
LONG_COLUMNS = _SUMMARY._PARAMS_BOUNDS_NILL_LONG_COLUMNS


def _sample_boundary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "workflow": "3d-gia",
                "scenario": "adopted",
                "parameter": "K_G  lengthscale (yr)",
                "value": 15000.0,
                "bound_low": 100.0,
                "bound_high": 30000.0,
                "proximity_pct": 50.0,
                "boundary_hit": False,
                "boundary_side": "",
                "scale": "log",
                "prior_key": "global_temporal.lengthscale_prior",
                "conversion": "identity",
            },
            {
                "workflow": "3d-gia",
                "scenario": "adopted",
                "parameter": "K_LNL temporal variance (m)",
                "value": 8.0,
                "bound_low": 0.001,
                "bound_high": 22.36,
                "proximity_pct": 90.0,
                "boundary_hit": False,
                "boundary_side": "",
                "scale": "log",
                "prior_key": "local_nonlinear.temporal_variance_prior",
                "conversion": "sqrt",
            },
            {
                "workflow": "3d-gia",
                "scenario": "bounds_m2",
                "parameter": "K_G  lengthscale (yr)",
                "value": 14996.0,
                "bound_low": 50.0,
                "bound_high": 60000.0,
                "proximity_pct": 80.0,
                "boundary_hit": False,
                "boundary_side": "",
                "scale": "log",
                "prior_key": "global_temporal.lengthscale_prior",
                "conversion": "identity",
            },
            {
                "workflow": "3d-gia",
                "scenario": "bounds_m2",
                "parameter": "K_WN  variance (m)",
                "value": 0.07,
                "bound_low": 0.0002,
                "bound_high": 1.41,
                "proximity_pct": 96.0,
                "boundary_hit": True,
                "boundary_side": "upper",
                "scale": "log",
                "prior_key": "whitenoise.variance_prior",
                "conversion": "sqrt",
            },
        ]
    )


def _sample_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "adopted", "NILL": 1531.89},
            {"scenario": "bounds_m2", "NILL": 1531.98},
        ]
    )


def test_parameter_row_order_follows_heatmap_order():
    parameters = [
        "K_WN  variance (m)",
        "K_LNL temporal variance (m)",
        "K_G  lengthscale (yr)",
    ]
    ordered = _parameter_row_order(parameters)
    assert ordered == [
        "K_G  lengthscale (yr)",
        "K_LNL temporal variance (m)",
        "K_WN  variance (m)",
    ]


def test_build_params_bounds_nill_long_shape_and_values():
    df_long = build_params_bounds_nill_long(
        _sample_boundary_df(),
        _sample_metrics_df(),
        ["adopted", "bounds_m2"],
    )

    assert list(df_long.columns) == LONG_COLUMNS
    assert len(df_long) == 4

    adopted_kg = df_long[
        (df_long["scenario"] == "adopted")
        & (df_long["parameter"] == "K_G  lengthscale (yr)")
    ].iloc[0]
    assert math.isclose(adopted_kg["optimized"], 15000.0)
    assert math.isclose(adopted_kg["bound_low"], 100.0)
    assert math.isclose(adopted_kg["bound_high"], 30000.0)
    assert math.isclose(adopted_kg["NILL"], 1531.89)

    wn_row = df_long[
        (df_long["scenario"] == "bounds_m2")
        & (df_long["parameter"] == "K_WN  variance (m)")
    ].iloc[0]
    assert math.isclose(wn_row["optimized"], 0.07)
    assert math.isclose(wn_row["NILL"], 1531.98)
    assert bool(wn_row["boundary_hit"])
    assert wn_row["boundary_side"] == "upper"


def test_build_params_bounds_nill_long_sort_order():
    df_long = build_params_bounds_nill_long(
        _sample_boundary_df(),
        _sample_metrics_df(),
        ["adopted", "bounds_m2"],
    )
    assert list(df_long["scenario"]) == [
        "adopted",
        "adopted",
        "bounds_m2",
        "bounds_m2",
    ]
    assert list(df_long["parameter"]) == [
        "K_G  lengthscale (yr)",
        "K_LNL temporal variance (m)",
        "K_G  lengthscale (yr)",
        "K_WN  variance (m)",
    ]


def test_write_params_bounds_nill_markdown(tmp_path: Path):
    df_long = build_params_bounds_nill_long(
        _sample_boundary_df(),
        _sample_metrics_df(),
        ["adopted", "bounds_m2"],
    )
    md_path = tmp_path / "summary_params_bounds_nill.md"
    write_params_bounds_nill_markdown(
        df_long,
        md_path,
        workflow="3d-gia",
    )

    text = md_path.read_text(encoding="utf-8")
    assert md_path.exists()
    assert "Scenario NILL" in text
    assert "Parameter Optimized Values and Bounds" in text
    assert "### adopted" in text
    assert "### bounds_m2" in text
    assert "3d-gia" in text
    assert "1532" in text
    assert "96.00*" in text
    assert "| upper |" in text


def test_build_params_bounds_nill_long_empty_inputs():
    result = build_params_bounds_nill_long(
        pd.DataFrame(),
        pd.DataFrame(columns=["scenario", "NILL"]),
        [],
    )
    assert list(result.columns) == LONG_COLUMNS
    assert result.empty


def test_write_params_bounds_nill_markdown_handles_none_nill(tmp_path: Path):
    df_long = build_params_bounds_nill_long(
        _sample_boundary_df(),
        pd.DataFrame(
            [
                {"scenario": "adopted", "NILL": None},
                {"scenario": "bounds_m2", "NILL": 1531.98},
            ]
        ),
        ["adopted", "bounds_m2"],
    )
    md_path = tmp_path / "summary_params_bounds_nill.md"
    write_params_bounds_nill_markdown(df_long, md_path, workflow="4ka")
    text = md_path.read_text(encoding="utf-8")
    assert "## Scenario NILL" in text
    assert "| adopted |" in text
    assert "1532" in text


def _write_adopted_manifest(tmp_path: Path, *, regional_nonlinear: bool) -> None:
    adopted_dir = tmp_path / "adopted"
    adopted_dir.mkdir(parents=True, exist_ok=True)
    manifest = f"""name: adopted
kernel_enabled_effective:
  regional_nonlinear: {str(regional_nonlinear).lower()}
"""
    (adopted_dir / "manifest.yaml").write_text(manifest, encoding="utf-8")


def test_boundary_heatmap_keeps_no_regional_nonlinear_when_adopted_has_rnl_on(
    tmp_path: Path,
):
    _write_adopted_manifest(tmp_path, regional_nonlinear=True)
    assert _adopted_regional_nonlinear_enabled(tmp_path) is True
    assert _should_drop_no_regional_nonlinear_from_heatmap(tmp_path) is False

    order = [
        "adopted",
        "no_regional_nonlinear",
        "no_local_nonlinear",
        "bounds_m2",
    ]
    plot_order = _boundary_heatmap_scenario_order(
        order, drop_no_regional_nonlinear=False
    )
    assert "no_regional_nonlinear" in plot_order


def test_boundary_heatmap_drops_no_regional_nonlinear_when_adopted_has_rnl_off(
    tmp_path: Path,
):
    _write_adopted_manifest(tmp_path, regional_nonlinear=False)
    assert _adopted_regional_nonlinear_enabled(tmp_path) is False
    assert _should_drop_no_regional_nonlinear_from_heatmap(tmp_path) is True

    order = ["adopted", "no_regional_nonlinear", "bounds_m2"]
    plot_order = _boundary_heatmap_scenario_order(order)
    assert "no_regional_nonlinear" not in plot_order
    assert plot_order == ["adopted", "bounds_m2"]


def test_boundary_heatmap_drops_no_regional_nonlinear_when_manifest_missing(
    tmp_path: Path,
):
    assert _adopted_regional_nonlinear_enabled(tmp_path) is None
    assert _should_drop_no_regional_nonlinear_from_heatmap(tmp_path) is True

    order = ["adopted", "no_regional_nonlinear"]
    plot_order = _boundary_heatmap_scenario_order(order)
    assert "no_regional_nonlinear" not in plot_order
