"""Tests for dashboard sea-level budget config round-trip and figure generation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stgp_dashboard.services.config_builder import (  # noqa: E402
    build_config,
    normalize_budget_age_windows,
    normalize_budget_gmsl_xlsx,
)
from stgp_dashboard.services.config_loader import config_dict_to_state  # noqa: E402
from stgp_dashboard.services.visualize_runner import (  # noqa: E402
    _infer_figure_metadata,
    discover_saved_figures,
    generate_budget_figures,
)
from stgp_dashboard.state import DEFAULT_STATE  # noqa: E402


def _synthetic_rates_csv(path: Path, *, n_regions: int = 2) -> Path:
    ages = np.arange(100.0, 3901.0, 200.0)
    rows = []
    for rn in range(1, n_regions + 1):
        for age in ages:
            rows.append(
                {
                    "region": rn,
                    "region_name": f"Region{rn}",
                    "age_bp": age,
                    "gia_rate": -1.0e-4,
                    "global_rate": -2.0e-4,
                    "regional_linear_rate": 5.0e-4,
                    "regional_nl_rate": 3.0e-4,
                    "local_nl_rate": 1.0e-4,
                    "total_rate": 6.0e-4,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_normalize_budget_age_windows_swaps_and_drops_empty():
    assert normalize_budget_age_windows([[4000, 0], [1000, 1000], None]) == [
        [0.0, 4000.0]
    ]
    assert normalize_budget_age_windows(None) == []


def test_budget_config_round_trip(tmp_path: Path):
    paleo = tmp_path / "paleo.csv"
    paleo.write_text("x\n", encoding="utf-8")
    anu = tmp_path / "ANU.xlsx"
    ice6g = tmp_path / "6g.xlsx"
    anu.write_bytes(b"x")
    ice6g.write_bytes(b"y")
    state = {
        **DEFAULT_STATE,
        "paleo_rsl": str(paleo),
        "output_dir": str(tmp_path / "out"),
        "use_gia": False,
        "budget_age_windows": [[0, 4000], [0, 6000]],
        "budget_merge_gia_global": False,
        "budget_gmsl_xlsx": {"anu": str(anu), "ice6g": str(ice6g)},
    }
    cfg = build_config(state)
    assert cfg["budget"]["age_windows_bp"] == [[0.0, 4000.0], [0.0, 6000.0]]
    assert cfg["budget"]["merge_gia_global"] is False
    assert cfg["budget"]["gmsl_xlsx"] == {"anu": str(anu), "ice6g": str(ice6g)}

    loaded = config_dict_to_state(cfg, base_dir=tmp_path)
    assert loaded["budget_age_windows"] == [[0.0, 4000.0], [0.0, 6000.0]]
    assert loaded["budget_merge_gia_global"] is False
    assert loaded["budget_gmsl_xlsx"] == {
        "anu": str(anu.resolve()),
        "ice6g": str(ice6g.resolve()),
    }


def test_budget_defaults_when_absent_from_yaml():
    loaded = config_dict_to_state({"paths": {"paleo_rsl": "x.csv", "output_dir": "o"}})
    assert loaded["budget_age_windows"] == [[0.0, 4000.0]]
    assert loaded["budget_merge_gia_global"] is True
    assert loaded["budget_gmsl_xlsx"] is None


def test_normalize_budget_gmsl_xlsx():
    assert normalize_budget_gmsl_xlsx(None) is None
    assert normalize_budget_gmsl_xlsx({"anu": "a.xlsx", "ice6g": "b.xlsx"}) == {
        "anu": "a.xlsx",
        "ice6g": "b.xlsx",
    }
    assert normalize_budget_gmsl_xlsx(["a.xlsx", "b.xlsx"]) == {
        "anu": "a.xlsx",
        "ice6g": "b.xlsx",
    }
    assert normalize_budget_gmsl_xlsx({"anu": "", "ice6g": "b.xlsx"}) is None


def test_infer_sea_level_budget_metadata(tmp_path: Path):
    png = tmp_path / "sea_level_budget_all_regions_0-4000_BP.png"
    png.write_bytes(b"")
    (tmp_path / "sea_level_budget_all_regions_0-4000_BP.pdf").write_bytes(b"")
    item = _infer_figure_metadata(png)
    assert item["category"] == "sea_level_budget"
    assert item["interval"] == "0-4000 BP"

    region_png = tmp_path / "sea_level_budget_region_3_west_jiangsu.png"
    region_png.write_bytes(b"")
    region_item = _infer_figure_metadata(region_png)
    assert region_item["category"] == "sea_level_budget"
    assert region_item["region"] == "West Jiangsu"

    # CSV lookup wins over slug title-casing.
    with_lookup = _infer_figure_metadata(
        region_png, region_lookup={3: "West Jiangsu Coast"}
    )
    assert with_lookup["region"] == "West Jiangsu Coast"


def test_generate_budget_figures_writes_artifacts(tmp_path: Path):
    scenario = tmp_path / "run"
    rates = scenario / "temporal_decomposition_rates.csv"
    _synthetic_rates_csv(rates, n_regions=2)
    (scenario.parent / "dashboard_config.yaml").write_text(
        yaml.safe_dump(
            {
                "budget": {
                    "age_windows_bp": [[0, 4000]],
                    "merge_gia_global": True,
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = generate_budget_figures(
        scenario,
        age_windows=[(0.0, 4000.0)],
        merge_gia_global=True,
        recompute=True,
    )
    assert (scenario / "sea_level_budget.csv").is_file()
    assert (scenario / "sea_level_budget.pdf").is_file()
    assert (scenario / "sea_level_budget_field.csv").is_file()
    assert any(o.get("category") == "sea_level_budget" for o in outputs)
    assert any("all_regions" in Path(o["png"]).name for o in outputs)
    assert (scenario / "figures" / "sea_level_budget_all_regions_0-4000_BP.png").is_file()

    discovered = discover_saved_figures(scenario)
    assert any(d["category"] == "sea_level_budget" for d in discovered)


def test_generate_budget_figures_with_gmsl_xlsx(tmp_path: Path):
    from stgp_pipeline import budget as stgp_budget

    gmsl_dir = tmp_path / "gmsl"
    gmsl_dir.mkdir()
    anu = gmsl_dir / "anu_gmsl.xlsx"
    ice6g = gmsl_dir / "ice6g_gmsl.xlsx"
    years = [-4.0, -3.0, -2.0, -1.0, 0.0]
    pd.DataFrame({"Year": years, "RSL": [-2.0, -1.5, -1.0, -0.5, 0.0]}).to_excel(
        anu, index=False
    )
    pd.DataFrame({"Year": years, "RSL": [-1.8, -1.3, -0.8, -0.4, 0.0]}).to_excel(
        ice6g, index=False
    )

    scenario = tmp_path / "run"
    rates_csv = scenario / "temporal_decomposition_rates.csv"
    _synthetic_rates_csv(rates_csv, n_regions=1)
    budget_cfg = {
        "age_windows_bp": [[0, 4000]],
        "merge_gia_global": True,
        "gmsl_xlsx": {"anu": str(anu), "ice6g": str(ice6g)},
    }
    (scenario.parent / "dashboard_config.yaml").write_text(
        yaml.safe_dump({"budget": budget_cfg}),
        encoding="utf-8",
    )

    # Config load + parse path used by generate_budget_figures (skip PDF draw).
    from stgp_dashboard.services.visualize_runner import _load_run_config_with_path

    cfg, cfg_path = _load_run_config_with_path(scenario, None)
    assert cfg_path is not None
    loaded_budget = dict(cfg.get("budget") or {})
    gmsl_pair = stgp_budget.gmsl_xlsx_from_config(
        loaded_budget, base_dir=cfg_path.parent
    )
    assert gmsl_pair is not None
    out = stgp_budget.compute_rate_contributions(
        rates_csv,
        age_windows=stgp_budget.age_windows_from_config(loaded_budget),
        gmsl_xlsx=gmsl_pair,
    )
    components = set(out["component"])
    assert "gmsl" in components
    assert "grd" in components
    assert "global" in components
    assert "gia_gmsl" not in components


def test_generate_budget_custom_window(tmp_path: Path):
    scenario = tmp_path / "run"
    _synthetic_rates_csv(scenario / "temporal_decomposition_rates.csv")
    # Extend ages so 0–6000 is covered
    df = pd.read_csv(scenario / "temporal_decomposition_rates.csv")
    extra = df.copy()
    extra["age_bp"] = extra["age_bp"] + 2000.0
    pd.concat([df, extra], ignore_index=True).to_csv(
        scenario / "temporal_decomposition_rates.csv", index=False
    )

    generate_budget_figures(
        scenario,
        age_windows=[(0.0, 6000.0)],
        merge_gia_global=True,
        recompute=True,
    )
    budget_df = pd.read_csv(scenario / "sea_level_budget.csv")
    assert set(budget_df["window_end_bp"].unique()) == {6000.0}
    assert (
        scenario / "figures" / "sea_level_budget_all_regions_0-6000_BP.png"
    ).is_file()


def test_export_scenario_sea_level_budget_helper(tmp_path: Path):
    from stgp_pipeline import budget

    scenario = tmp_path / "run"
    _synthetic_rates_csv(scenario / "temporal_decomposition_rates.csv")
    out = budget.export_scenario_sea_level_budget(
        scenario,
        {"age_windows_bp": [[0, 4000]], "merge_gia_global": True},
    )
    assert out is not None
    assert not out.empty
    assert (scenario / "sea_level_budget.csv").is_file()


def test_age_windows_from_config_defaults():
    from stgp_pipeline import budget

    assert budget.age_windows_from_config({}) == [(0.0, 4000.0)]
    assert budget.age_windows_from_config(
        {"age_windows_bp": [[0, 4000], [0, 6000]]}
    ) == [(0.0, 4000.0), (0.0, 6000.0)]


def test_run_sea_level_budget_stage_writes_csv(tmp_path: Path):
    from stgp_pipeline.sensitivity.workflow_common import _run_sea_level_budget_stage

    scenario = tmp_path / "run"
    rates = scenario / "temporal_decomposition_rates.csv"
    _synthetic_rates_csv(rates)
    _run_sea_level_budget_stage(
        scenario,
        rates,
        {"budget": {"age_windows_bp": [[0, 4000]], "merge_gia_global": True}},
    )
    assert (scenario / "sea_level_budget.csv").is_file()
    assert (scenario / "sea_level_budget.pdf").is_file()


def test_default_budget_is_single_0_4ka_window():
    windows = DEFAULT_STATE["budget_age_windows"]
    assert windows == [[0, 4000]]

    preset_path = (
        Path(__file__).resolve().parents[1] / "presets" / "defaults.yaml"
    )
    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    assert preset["budget"]["age_windows_bp"] == [[0, 4000]]
    assert preset["budget"]["merge_gia_global"] is True


def test_partition_budget_gallery_items():
    from stgp_dashboard.steps.results import partition_budget_gallery_items

    items = [
        {
            "name": "sea_level_budget_all_regions_0-4000_BP",
            "png": "/tmp/sea_level_budget_all_regions_0-4000_BP.png",
            "category": "sea_level_budget",
        },
        {
            "name": "sea_level_budget_field",
            "png": "/tmp/sea_level_budget_field.png",
            "category": "sea_level_budget",
        },
        {
            "name": "sea_level_budget_region_1_alpha",
            "png": "/tmp/sea_level_budget_region_1_alpha.png",
            "category": "sea_level_budget",
            "region": "Alpha",
        },
        {
            "name": "sea_level_budget_misc",
            "png": "/tmp/sea_level_budget_misc.png",
            "category": "sea_level_budget",
        },
    ]
    parts = partition_budget_gallery_items(items)
    assert len(parts["all_regions"]) == 1
    assert len(parts["field"]) == 1
    assert len(parts["per_region"]) == 1
    assert len(parts["other"]) == 1
    assert parts["per_region"][0]["region"] == "Alpha"


def test_discover_saved_figures_resolves_region_from_csv(tmp_path: Path):
    scenario = tmp_path / "run"
    figures = scenario / "figures"
    figures.mkdir(parents=True)
    png = figures / "sea_level_budget_region_1_e_guangdong_hk.png"
    png.write_bytes(b"")
    pd.DataFrame(
        {
            "region": [1, 1],
            "region_name": ["E Guangdong HK", "E Guangdong HK"],
            "window_start_bp": [0.0, 0.0],
            "window_end_bp": [4000.0, 4000.0],
            "component": ["gia_gmsl", "regional_linear"],
            "abs_contribution_m": [1.0, 1.0],
            "pct": [50.0, 50.0],
        }
    ).to_csv(scenario / "sea_level_budget.csv", index=False)

    discovered = discover_saved_figures(scenario)
    assert len(discovered) == 1
    assert discovered[0]["category"] == "sea_level_budget"
    assert discovered[0]["region"] == "E Guangdong HK"
