from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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


def _synthetic_rates_df(
    *,
    gmsl_rate: float,
    grd_rate: float,
    regional_linear_rate: float = 0.0005,
) -> pd.DataFrame:
    ages = np.arange(100.0, 3901.0, 200.0)
    n = len(ages)
    return pd.DataFrame(
        {
            "region": [1] * n,
            "region_name": ["Test"] * n,
            "age_bp": ages,
            "gmsl_rate": np.full(n, gmsl_rate),
            "grd_rate": np.full(n, grd_rate),
            "global_rate": np.zeros(n),
            "regional_linear_rate": np.full(n, regional_linear_rate),
            "local_nl_rate": np.zeros(n),
        }
    )


def _synthetic_gia_global_rates_df(
    *,
    gia_rate: float,
    global_rate: float,
    regional_linear_rate: float = 0.0005,
) -> pd.DataFrame:
    ages = np.arange(100.0, 3901.0, 200.0)
    n = len(ages)
    return pd.DataFrame(
        {
            "region": [1] * n,
            "region_name": ["Test"] * n,
            "age_bp": ages,
            "gia_rate": np.full(n, gia_rate),
            "global_rate": np.full(n, global_rate),
            "regional_linear_rate": np.full(n, regional_linear_rate),
            "local_nl_rate": np.zeros(n),
        }
    )


def test_merged_gia_gmsl_label_not_external_gmsl():
    from stgp_pipeline import budget

    label = budget._component_label("gia_gmsl")
    assert label == "GIA + Global (Gm)"
    assert label != "GIA and GMSL"
    assert "GMSL" not in label
    assert budget._component_label("gia") == "GIA prior"


def test_regional_nl_and_local_nl_colors_differ():
    from stgp_pipeline import budget

    assert budget.COMPONENT_COLORS["regional_nl"] != budget.COMPONENT_COLORS["local_nl"]


def test_regional_linear_and_regional_nl_colors_differ():
    from stgp_pipeline import budget

    assert (
        budget.COMPONENT_COLORS["regional_linear"]
        != budget.COMPONENT_COLORS["regional_nl"]
    )


def test_contrast_text_color():
    from stgp_pipeline import budget

    assert budget._contrast_text_color("#4477AA") == "white"
    assert budget._contrast_text_color("#CCBB44") == "black"
    assert budget._contrast_text_color("#AA3377") == "white"


def test_gia_and_regional_nl_avoid_red_green_pair():
    """Eustatic (blue) and Regional NL (purple) — not green vs red."""
    from stgp_pipeline import budget

    gia = budget.COMPONENT_COLORS["gia_gmsl"].lower()
    nl = budget.COMPONENT_COLORS["regional_nl"].lower()
    lm = budget.COMPONENT_COLORS["regional_linear"].lower()
    assert gia.startswith("#44") or gia.startswith(
        "#33")  # blue / indigo family
    assert nl.startswith("#aa")  # purple, not red
    assert lm.startswith("#cc")  # gold/yellow, not orange


def test_split_gia_global_when_merge_disabled():
    from stgp_pipeline import budget

    gia_rate = -0.0004
    global_rate = 0.0001
    df = _synthetic_gia_global_rates_df(
        gia_rate=gia_rate, global_rate=global_rate)
    window_length = 4000.0

    out = budget.compute_rate_contributions(
        df, age_windows=[(0, 4000)], merge_gia_global=False
    )
    sub = out[(out["region"] == 1) & (out["window_start_bp"] == 0)]
    components = set(sub["component"])

    assert "gia" in components
    assert "global" in components
    assert "gia_gmsl" not in components

    gia_m = sub.loc[sub["component"] == "gia", "abs_contribution_m"].iloc[0]
    global_m = sub.loc[sub["component"] ==
                       "global", "abs_contribution_m"].iloc[0]
    np.testing.assert_allclose(gia_m, abs(gia_rate) * window_length, rtol=1e-9)
    np.testing.assert_allclose(global_m, abs(
        global_rate) * window_length, rtol=1e-9)


def test_merged_gia_gmsl_still_default():
    from stgp_pipeline import budget

    gia_rate = -0.0004
    global_rate = 0.0001
    df = _synthetic_gia_global_rates_df(
        gia_rate=gia_rate, global_rate=global_rate)
    window_length = 4000.0
    combined_expected = abs(gia_rate + global_rate) * window_length

    out = budget.compute_rate_contributions(df, age_windows=[(0, 4000)])
    sub = out[(out["region"] == 1) & (out["window_start_bp"] == 0)]
    components = set(sub["component"])

    assert components == {"gia_gmsl", "regional_linear", "local_nl"}

    gia_gmsl_m = sub.loc[sub["component"] ==
                         "gia_gmsl", "abs_contribution_m"].iloc[0]
    np.testing.assert_allclose(gia_gmsl_m, combined_expected, rtol=1e-9)


def test_gmsl_grd_split_preserves_combined_gia_contribution():
    from stgp_pipeline import budget

    # Opposite-sign means: raw |mean| sum would inflate GIA side of the budget.
    gmsl_rate = -0.0002
    grd_rate = 0.0006
    df = _synthetic_rates_df(gmsl_rate=gmsl_rate, grd_rate=grd_rate)
    window_length = 4000.0
    combined_expected = abs(gmsl_rate + grd_rate) * window_length

    out = budget.compute_rate_contributions(df, age_windows=[(0, 4000)])
    sub = out[(out["region"] == 1) & (out["window_start_bp"] == 0)]

    gmsl_m = sub.loc[sub["component"] == "gmsl", "abs_contribution_m"].iloc[0]
    grd_m = sub.loc[sub["component"] == "grd", "abs_contribution_m"].iloc[0]
    np.testing.assert_allclose(gmsl_m + grd_m, combined_expected, rtol=1e-9)

    raw_gmsl = abs(gmsl_rate) * window_length
    raw_grd = abs(grd_rate) * window_length
    assert raw_gmsl + raw_grd > combined_expected + 1e-12


def test_regional_linear_pct_matches_combined_gia_denominator():
    from stgp_pipeline import budget

    gmsl_rate = -0.0002
    grd_rate = 0.0006
    lm_rate = 0.0005
    df = _synthetic_rates_df(
        gmsl_rate=gmsl_rate, grd_rate=grd_rate, regional_linear_rate=lm_rate
    )
    window_length = 4000.0

    split = budget.compute_rate_contributions(df, age_windows=[(0, 4000)])
    combined_gia = abs(gmsl_rate + grd_rate) * window_length
    lm_m = abs(lm_rate) * window_length
    expected_lm_pct = lm_m / (combined_gia + lm_m) * 100.0

    lm_pct = split.loc[
        (split["region"] == 1)
        & (split["component"] == "regional_linear")
        & (split["window_start_bp"] == 0),
        "pct",
    ].iloc[0]
    np.testing.assert_allclose(lm_pct, expected_lm_pct, rtol=1e-9)

    # Independent |mean| on GMSL and GRD would lower Lm%.
    inflated_total = (
        abs(gmsl_rate) * window_length + abs(grd_rate) * window_length + lm_m
    )
    inflated_lm_pct = lm_m / inflated_total * 100.0
    assert lm_pct > inflated_lm_pct


def _synthetic_window_rates_df(
    *,
    total_rate: float | np.ndarray,
    regional_linear_rate: float | np.ndarray,
    region: int = 1,
    region_name: str = "Test",
) -> pd.DataFrame:
    ages = np.array([100.0, 2100.0, 3900.0])
    n = len(ages)
    total = (
        np.full(n, total_rate)
        if np.isscalar(total_rate)
        else np.asarray(total_rate, dtype=float)
    )
    rl = (
        np.full(n, regional_linear_rate)
        if np.isscalar(regional_linear_rate)
        else np.asarray(regional_linear_rate, dtype=float)
    )
    return pd.DataFrame(
        {
            "region": [region] * n,
            "region_name": [region_name] * n,
            "age_bp": ages,
            "total_rate": total,
            "regional_linear_rate": rl,
            "gia_rate": np.zeros(n),
            "global_rate": np.zeros(n),
            "local_nl_rate": np.zeros(n),
        }
    )


def test_window_mean_rates_mm_yr_sign_convention():
    from stgp_pipeline import budget

    df = _synthetic_window_rates_df(
        total_rate=-0.001, regional_linear_rate=0.0005)
    out = budget.compute_window_mean_rates(df, age_windows=[(0, 4000)])
    row = out.iloc[0]
    np.testing.assert_allclose(row["total_mean_rate_mm_yr"], 1.0, rtol=1e-9)
    np.testing.assert_allclose(
        row["regional_linear_mean_rate_mm_yr"], -0.5, rtol=1e-9)


def test_window_mean_rates_averages_within_window():
    from stgp_pipeline import budget

    df = _synthetic_window_rates_df(
        total_rate=[-0.001, -0.003, -0.005],
        regional_linear_rate=[0.0004, 0.0005, 0.0006],
    )
    out = budget.compute_window_mean_rates(df, age_windows=[(0, 4000)])
    row = out.iloc[0]
    np.testing.assert_allclose(row["total_mean_rate_mm_yr"], 3.0, rtol=1e-9)
    np.testing.assert_allclose(
        row["regional_linear_mean_rate_mm_yr"], -0.5, rtol=1e-9)


def test_field_window_mean_rates_aggregator_mean():
    from stgp_pipeline import budget

    df_a = _synthetic_window_rates_df(
        total_rate=-0.001,
        regional_linear_rate=0.0005,
        region=1,
        region_name="A",
    )
    df_b = _synthetic_window_rates_df(
        total_rate=-0.003,
        regional_linear_rate=0.0007,
        region=2,
        region_name="B",
    )
    window_rates = budget.compute_window_mean_rates(
        pd.concat([df_a, df_b], ignore_index=True), age_windows=[(0, 4000)]
    )
    field = budget.compute_field_window_mean_rates(
        window_rates, aggregator="mean")
    assert len(field) == 1
    np.testing.assert_allclose(
        field.iloc[0]["total_mean_rate_mm_yr"], 2.0, rtol=1e-9)
    np.testing.assert_allclose(
        field.iloc[0]["regional_linear_mean_rate_mm_yr"], -0.6, rtol=1e-9
    )


def test_export_sea_level_budget_writes_window_rate_csvs(tmp_path: Path):
    from stgp_pipeline import budget

    rates_path = tmp_path / "temporal_decomposition_rates.csv"
    df = _synthetic_window_rates_df(
        total_rate=-0.001, regional_linear_rate=0.0005)
    df.to_csv(rates_path, index=False)

    budget.export_sea_level_budget(
        rates_csv=rates_path,
        pdf_path=tmp_path / "sea_level_budget.pdf",
        csv_path=tmp_path / "sea_level_budget.csv",
        age_windows=[(0, 4000)],
    )

    window_rates_path = tmp_path / "sea_level_budget_window_rates.csv"
    field_window_rates_path = tmp_path / "sea_level_budget_window_rates_field.csv"
    assert window_rates_path.is_file()
    assert field_window_rates_path.is_file()

    window_rates = pd.read_csv(window_rates_path)
    field_window_rates = pd.read_csv(field_window_rates_path)
    assert list(window_rates.columns) == [
        "region",
        "region_name",
        "window_start_bp",
        "window_end_bp",
        "total_mean_rate_mm_yr",
        "regional_linear_mean_rate_mm_yr",
    ]
    assert len(window_rates) == 1
    assert len(field_window_rates) == 1
    np.testing.assert_allclose(
        window_rates.iloc[0]["total_mean_rate_mm_yr"], 1.0, rtol=1e-9
    )


def _synthetic_budget_df(*, local_nl_pct: float = 0.5) -> pd.DataFrame:
    """Two regions, GIA/global split, one 0–4000 BP window."""
    rows = []
    for region, name in ((1, "Alpha"), (2, "Beta")):
        parts = {
            "gia": 50.0,
            "global": 10.0,
            "regional_linear": 20.0,
            "regional_nl": 19.0,
            "local_nl": local_nl_pct,
        }
        total = sum(parts.values())
        for comp, pct in parts.items():
            rows.append(
                {
                    "region": region,
                    "region_name": name,
                    "window_start_bp": 0.0,
                    "window_end_bp": 4000.0,
                    "component": comp,
                    "abs_contribution_m": pct / 100.0 * total,
                    "pct": pct,
                }
            )
    return pd.DataFrame(rows)


def test_plot_budget_regions_grid_smoke():
    from stgp_pipeline import budget

    df = _synthetic_budget_df(local_nl_pct=0.5)
    fig = budget.plot_budget_regions_grid(df, window=(0.0, 4000.0), n_cols=2)
    axes = fig.axes
    assert len(axes) == 2
    assert all(ax.get_visible() for ax in axes)
    legend = fig.legends
    assert len(legend) == 1
    labels = [t.get_text() for t in legend[0].get_texts()]
    assert "Local NL (Ln)" in labels
    assert "GIA prior" in labels

    visible_pct_labels = [
        t
        for ax in axes
        for t in ax.texts
        if t.get_text().strip().endswith("%")
        and "\n" not in t.get_text()
        and t.get_visible()
    ]
    assert visible_pct_labels, "expected at least one on-wedge percentage label"
    assert all(t.get_path_effects() for t in visible_pct_labels)


def test_narrow_wedge_label_moves_to_note():
    from stgp_pipeline import budget

    parts = {
        "gia": 60.0,
        "global": 6.0,
        "regional_linear": 20.0,
        "regional_nl": 13.0,
        "local_nl": 1.0,
    }
    total = sum(parts.values())
    rows = [
        {
            "region": 1,
            "region_name": "Alpha",
            "window_start_bp": 0.0,
            "window_end_bp": 4000.0,
            "component": comp,
            "abs_contribution_m": pct / 100.0 * total,
            "pct": pct,
        }
        for comp, pct in parts.items()
    ]
    df = pd.DataFrame(rows)
    fig = budget.plot_budget_regions_grid(df, window=(0.0, 4000.0), n_cols=1)

    note_text = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "Global (Gm): 6.0%" in note_text
    on_wedge_global = [
        t.get_text()
        for ax in fig.axes
        for t in ax.texts
        if t.get_text().strip() == "6.0%"
    ]
    assert not on_wedge_global


def _write_synthetic_gmsl_xlsx_pair(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    anu = root / "anu_gmsl.xlsx"
    ice6g = root / "ice6g_gmsl.xlsx"
    years = [-4.0, -3.0, -2.0, -1.0, 0.0]
    pd.DataFrame({"Year": years, "RSL": [-2.0, -1.5, -1.0, -0.5, 0.0]}).to_excel(
        anu, index=False
    )
    pd.DataFrame({"Year": years, "RSL": [-1.8, -1.3, -0.8, -0.4, 0.0]}).to_excel(
        ice6g, index=False
    )
    return anu, ice6g


def test_gmsl_xlsx_from_config_mapping_and_list(tmp_path: Path):
    from stgp_pipeline import budget

    anu, ice6g = _write_synthetic_gmsl_xlsx_pair(tmp_path / "src_gmsl")

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    # Relative paths resolve against base_dir.
    rel_anu = Path("gmsl") / "anu.xlsx"
    rel_6g = Path("gmsl") / "6g.xlsx"
    (cfg_dir / "gmsl").mkdir()
    (cfg_dir / rel_anu).write_bytes(anu.read_bytes())
    (cfg_dir / rel_6g).write_bytes(ice6g.read_bytes())

    parsed = budget.gmsl_xlsx_from_config(
        {"gmsl_xlsx": {"anu": str(rel_anu), "ice6g": str(rel_6g)}},
        base_dir=cfg_dir,
    )
    assert parsed is not None
    assert parsed[0].is_file()
    assert parsed[1].is_file()

    abs_parsed = budget.gmsl_xlsx_from_config(
        {"gmsl_xlsx": [str(anu), str(ice6g)]},
        base_dir=cfg_dir,
    )
    assert abs_parsed is not None
    assert abs_parsed[0] == anu.resolve()
    assert abs_parsed[1] == ice6g.resolve()
    assert budget.gmsl_xlsx_from_config({}) is None
    assert budget.gmsl_xlsx_from_config({"gmsl_xlsx": {"anu": "", "ice6g": ""}}) is None


def test_export_scenario_reads_gmsl_xlsx_from_budget_cfg(tmp_path: Path):
    from stgp_pipeline import budget

    anu, ice6g = _write_synthetic_gmsl_xlsx_pair(tmp_path / "src_gmsl")

    scenario = tmp_path / "run"
    scenario.mkdir()
    rates = _synthetic_gia_global_rates_df(
        gia_rate=-0.0004, global_rate=0.0001, regional_linear_rate=0.0005
    )
    rates_csv = scenario / "temporal_decomposition_rates.csv"
    rates.to_csv(rates_csv, index=False)

    budget_cfg = {
        "age_windows_bp": [[0, 4000]],
        "merge_gia_global": True,
        "gmsl_xlsx": {"anu": str(anu), "ice6g": str(ice6g)},
    }
    gmsl_pair = budget.gmsl_xlsx_from_config(budget_cfg, base_dir=scenario.parent)
    assert gmsl_pair is not None
    out = budget.compute_rate_contributions(
        rates_csv,
        age_windows=budget.age_windows_from_config(budget_cfg),
        gmsl_xlsx=gmsl_pair,
        merge_gia_global=True,
    )
    components = set(out["component"])
    assert "gmsl" in components
    assert "grd" in components
    assert "global" in components
    assert "regional_linear" in components
    assert "gia_gmsl" not in components
    assert "gia" not in components
