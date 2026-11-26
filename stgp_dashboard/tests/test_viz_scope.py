"""Tests for visualization scope (RSL curve only vs full analysis)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stgp_dashboard.services.visualize_runner import (  # noqa: E402
    _infer_figure_metadata,
    discover_saved_figures,
    generate_post_run_figures,
)
from stgp_dashboard.state import DEFAULT_STATE  # noqa: E402


def _synthetic_temporal_csv(path: Path, *, n_regions: int = 2) -> Path:
    ages = np.arange(0.0, 4001.0, 500.0)
    rows = []
    for rn in range(1, n_regions + 1):
        for age in ages:
            rows.append(
                {
                    "region": rn,
                    "region_name": f"Region{rn}",
                    "age_bp": age,
                    "total_mean": -float(age) / 1000.0,
                    "total_std": 0.5,
                    "gia_mean": -float(age) / 2000.0,
                    "regional_linear_mean": 0.1,
                    "regional_linear_std": 0.05,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fake_rsl_data(n_regions: int = 2):
    ages = np.array([1000.0, 2000.0])
    n = n_regions * len(ages)
    region = np.repeat(np.arange(1, n_regions + 1), len(ages))
    names = np.array([f"Region{r}" for r in region], dtype=object)
    X = np.column_stack(
        [np.tile(ages, n_regions), np.zeros(n), np.zeros(n)]
    )
    y = np.zeros(n)
    sigma = np.full(n, 0.2)
    return SimpleNamespace(
        X=X,
        y=y,
        y_sigma=sigma,
        age_sigma=sigma,
        rsl_region=region,
        rsl_region_name=names,
    )


def test_default_viz_scope_is_full():
    assert DEFAULT_STATE["viz_scope"] == "full"


def test_infer_rsl_curve_metadata(tmp_path: Path):
    png = tmp_path / "rsl_curve_all_regions.png"
    png.write_bytes(b"")
    (tmp_path / "rsl_curve_all_regions.pdf").write_bytes(b"")
    item = _infer_figure_metadata(png)
    assert item["category"] == "rsl_curve"
    assert item["name"] == "rsl_curve_all_regions"
    assert item["pdf"]


def test_discover_saved_figures_rsl_curve(tmp_path: Path):
    scenario = tmp_path / "run"
    figures = scenario / "figures"
    figures.mkdir(parents=True)
    (figures / "rsl_curve_all_regions.png").write_bytes(b"")
    discovered = discover_saved_figures(scenario)
    assert len(discovered) == 1
    assert discovered[0]["category"] == "rsl_curve"


def test_generate_post_run_figures_rsl_curve_scope(tmp_path: Path, monkeypatch):
    scenario = tmp_path / "run"
    _synthetic_temporal_csv(scenario / "temporal_decomposition.csv", n_regions=2)
    # Presence of rates/budget artifacts must not produce those categories.
    (scenario / "sea_level_budget.csv").write_text("region\n", encoding="utf-8")
    (scenario / "spatial_predictions.csv").write_text("age_bp\n", encoding="utf-8")
    (scenario / "optimization_track.csv").write_text(
        "iter,loss\n0,1.0\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._load_paleo_rsl",
        lambda cfg: _fake_rsl_data(2),
    )
    # Avoid PSTHM dependency in unit tests.
    monkeypatch.setattr(
        "stgp_pipeline.visualize.plot_temporal_all_regions",
        lambda results, rsl_data: __import__("matplotlib.pyplot").pyplot.figure(),
    )

    outputs = generate_post_run_figures(scenario, scope="rsl_curve")
    assert outputs
    assert all(o.get("category") == "rsl_curve" for o in outputs)
    assert (scenario / "figures" / "rsl_curve_all_regions.png").is_file()
    categories = {o.get("category") for o in outputs}
    assert "sea_level_budget" not in categories
    assert "spatial_decomposition" not in categories
    assert "spatial_rate" not in categories
    assert "temporal_decomposition" not in categories
    assert "optimization" not in categories


def test_generate_post_run_figures_default_scope_is_full(tmp_path: Path, monkeypatch):
    """Default scope remains 'full' and still runs the full suite path."""
    scenario = tmp_path / "run"
    scenario.mkdir(parents=True)
    calls: list[str] = []

    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._load_run_config",
        lambda scenario_path, metrics: {},
    )
    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._load_paleo_rsl",
        lambda cfg: None,
    )

    def _track(name):
        def _inner(*args, **kwargs):
            calls.append(name)
            return []

        return _inner

    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._generate_temporal_decomposition_figures",
        _track("temporal"),
    )
    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner.generate_budget_figures",
        _track("budget"),
    )
    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._generate_spatial_decomposition_figures",
        _track("spatial"),
    )
    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._generate_spatial_rate_figures",
        _track("rates"),
    )
    monkeypatch.setattr(
        "stgp_dashboard.services.visualize_runner._generate_rsl_curve_figures",
        _track("rsl_curve"),
    )

    generate_post_run_figures(scenario)  # default scope="full"
    assert "rsl_curve" not in calls
    assert calls == ["temporal", "budget", "spatial", "rates"]
