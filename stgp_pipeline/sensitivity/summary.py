"""
Summary builders for sensitivity scenario outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm

from ..gm_rate import load_gm_rate_netcdf, plot_gm_rate
from .workflow_common import (
    DEFAULT_SLIP_REGION_IDS,
    compute_nc_row_to_region_id,
    filter_nc_rows_by_region_ids,
    peak_global_rate_mm_yr,
    peak_rate_age_window,
    region_display_name,
    workflow_nc_suffix,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_nc_path(path_text: Any, fallback: Path) -> Path | None:
    if isinstance(path_text, str) and path_text.strip():
        path = Path(path_text).expanduser()
        if path.exists():
            return path.resolve()
    if fallback.exists():
        return fallback.resolve()
    return None


def _infer_workflow_name(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    workflow = manifest.get("workflow") or metrics.get("workflow")
    if workflow:
        return str(workflow)
    return "16ka"


def _scenario_nc_fallback(
    scenario_dir: Path,
    scenario_name: str,
    *,
    kind: str,
    workflow: str,
) -> Path:
    suffix = workflow_nc_suffix(workflow)
    return scenario_dir / f"{kind}_{scenario_name}_{suffix}.nc"


def _safe_rate_peak_global(
    rate_nc_path: Path | None,
    *,
    workflow: str = "16ka",
) -> float:
    if rate_nc_path is None or not rate_nc_path.exists():
        return float("nan")
    with nc.Dataset(rate_nc_path) as ds:
        if "rate_time" not in ds.variables or "global_rate_mean" not in ds.variables:
            return float("nan")
        rate_time = np.asarray(ds.variables["rate_time"][:], dtype=float)
        global_rate_mean = np.asarray(ds.variables["global_rate_mean"][:], dtype=float)
    return peak_global_rate_mm_yr(rate_time, global_rate_mean, workflow=workflow)


def _scenario_plot_style(
    scenario: str,
    *,
    bounds_multiplier: Any = None,
) -> Dict[str, Any]:
    """Line style for sensitivity overlays (adopted baseline vs bounds scenarios)."""
    if scenario == "adopted":
        return {
            "color": "#0072B2",
            "linestyle": "-",
            "linewidth": 2.4,
            "is_baseline": True,
        }

    mult = None
    if bounds_multiplier is not None:
        try:
            mult = float(bounds_multiplier)
        except (TypeError, ValueError):
            mult = None

    linestyle_by_mult = {
        1.5: "--",
        2.0: "-.",
        3.0: ":",
    }
    color_by_mult = {1.5: "#E69F00", 2.0: "#009E73", 3.0: "#D55E00"}
    if mult is not None and np.isfinite(mult):
        return {
            "color": color_by_mult.get(mult, "gray"),
            "linestyle": linestyle_by_mult.get(mult, "--"),
            "linewidth": 1.8,
            "is_baseline": False,
        }

    return {"color": None, "linestyle": "--", "linewidth": 1.5, "is_baseline": False}


def _region_labels_from_dataset(ds: nc.Dataset, n_regions: int) -> List[str]:
    try:
        raw = ds.variables["regions"][:]
        labels = []
        for item in raw:
            text = str(item)
            if text.startswith("b'") and text.endswith("'"):
                text = text[2:-1]
            labels.append(text)
        if len(labels) == n_regions:
            return labels
    except Exception:  # noqa: BLE001
        pass
    return [f"R{i+1}" for i in range(n_regions)]


def _boundary_marker(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 5.0:
        return "L"
    if value >= 95.0:
        return "U"
    return ""


def _boundary_heatmap_column_order() -> List[tuple[str, str]]:
    """
    Preferred heatmap column order: (internal parameter name, x-axis display label).
    Matches training.py optimized_params names and workflow_common _PARAM_TO_BOUND keys.
    """
    return [
        ("K_G  lengthscale (yr)", "Gm_temporal"),
        ("K_G  variance (m)", "Gm_variance"),
        ("K_RL spatial s_lengthscale (km)", "Lm_spatial"),
        ("K_RL temporal variance (m/yr)", "Lm_variance"),
        ("K_RNL spatial s_lengthscale (km)", "Rn_spatial"),
        ("K_RNL temporal lengthscale (yr)", "Rn_temporal"),
        ("K_RNL temporal variance (m)", "Rn_variance"),
        ("K_RNL_FAST spatial s_lengthscale (km)", "Rnf_spatial"),
        ("K_RNL_FAST temporal lengthscale (yr)", "Rnf_temporal"),
        ("K_RNL_FAST temporal variance (m)", "Rnf_variance"),
        ("K_MNL spatial s_lengthscale (km)", "Mn_spatial"),
        ("K_MNL temporal lengthscale (yr)", "Mn_temporal"),
        ("K_MNL temporal variance (m)", "Mn_variance"),
        ("K_LNL spatial s_lengthscale (km)", "Ln_spatial"),
        ("K_LNL temporal lengthscale (yr)", "Ln_temporal"),
        ("K_LNL temporal variance (m)", "Ln_variance"),
    ]


def _adopted_regional_nonlinear_enabled(output_root: Path) -> bool | None:
    """
    Return whether adopted keeps regional_nonlinear enabled.

    Reads ``adopted/manifest.yaml`` ``kernel_enabled_effective``. Returns ``None``
    when the manifest is missing or unreadable.
    """
    manifest_path = output_root / "adopted" / "manifest.yaml"
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_yaml(manifest_path)
    except Exception:  # noqa: BLE001
        return None
    effective = manifest.get("kernel_enabled_effective") or {}
    if not isinstance(effective, dict) or "regional_nonlinear" not in effective:
        return None
    return bool(effective["regional_nonlinear"])


def _should_drop_no_regional_nonlinear_from_heatmap(output_root: Path) -> bool:
    """
    Drop ``no_regional_nonlinear`` only when redundant with adopted.

    When adopted already disables regional nonlinear, the ablation
    row duplicates adopted. When adopted keeps it on, keep the row.
    """
    rnl_enabled = _adopted_regional_nonlinear_enabled(output_root)
    if rnl_enabled is None:
        return True
    return not rnl_enabled


def _boundary_heatmap_scenario_order(
    scenario_order: List[str],
    *,
    drop_no_regional_nonlinear: bool = True,
) -> List[str]:
    """Optionally drop ``no_regional_nonlinear`` when redundant with adopted."""
    if "adopted" not in scenario_order or not drop_no_regional_nonlinear:
        return list(scenario_order)
    return [name for name in scenario_order if name != "no_regional_nonlinear"]


def _reorder_boundary_heatmap_pivot(
    pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:
    """Drop K_WN from plot, reorder columns, return (pivot, display_labels)."""
    k_wn = "K_WN  variance (m)"
    if k_wn in pivot.columns:
        pivot = pivot.drop(columns=[k_wn])

    present = set(pivot.columns.astype(str))
    ordered_internal: List[str] = []
    display_by_internal: Dict[str, str] = {}
    for internal, display in _boundary_heatmap_column_order():
        if internal in present:
            ordered_internal.append(internal)
            display_by_internal[internal] = display

    remaining = sorted(present - set(ordered_internal))
    ordered_internal.extend(remaining)
    for name in remaining:
        display_by_internal[name] = name

    pivot = pivot.reindex(columns=ordered_internal)
    labels = [display_by_internal[c] for c in pivot.columns]
    return pivot, labels


def _write_boundary_heatmap(
    df_boundary: pd.DataFrame, output_root: Path, scenario_order: List[str]
) -> Dict[str, Path]:
    if df_boundary.empty:
        return {}
    required = {"scenario", "parameter", "proximity_pct"}
    if not required.issubset(df_boundary.columns):
        return {}

    pivot = df_boundary.pivot_table(
        index="scenario",
        columns="parameter",
        values="proximity_pct",
        aggfunc="mean",
    )
    drop_no_rnl = _should_drop_no_regional_nonlinear_from_heatmap(output_root)
    plot_order = _boundary_heatmap_scenario_order(
        scenario_order, drop_no_regional_nonlinear=drop_no_rnl
    )
    ordered_index = [name for name in plot_order if name in pivot.index]
    if not ordered_index:
        ordered_index = [
            name
            for name in _boundary_heatmap_scenario_order(
                list(pivot.index), drop_no_regional_nonlinear=drop_no_rnl
            )
            if name in pivot.index
        ]
    pivot = pivot.loc[ordered_index]
    if pivot.empty:
        return {}

    pivot, x_display_labels = _reorder_boundary_heatmap_pivot(pivot)
    if pivot.empty or len(pivot.columns) == 0:
        return {}

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(10, len(pivot.columns) * 0.65), max(5, len(pivot.index) * 0.6)),
    )
    heatmap_data = pivot.values
    norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    im = ax.imshow(heatmap_data, aspect="auto", cmap="PuOr", norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(x_display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Boundary Proximity Heatmap (L/U = lower/upper-bound hit)")

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            if not np.isfinite(val):
                continue
            marker = _boundary_marker(float(val))
            text_color = "white" if abs(val - 50.0) >= 30.0 else "black"
            ax.text(
                j,
                i,
                f"{val:.0f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = plt.colorbar(im, ax=ax, label="Position within prior range (%)")
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["Lower bound", "Interior", "Upper bound"])

    fig.tight_layout()
    pdf_path = output_root / "boundary_proximity_heatmap.pdf"
    png_path = output_root / "boundary_proximity_heatmap.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"boundary_heatmap_pdf": pdf_path, "boundary_heatmap_png": png_path}


def _write_global_rate_plot(
    scenario_records: List[Dict[str, Any]],
    output_root: Path,
    *,
    workflow: str = "16ka",
) -> Dict[str, Path]:
    plot_rows = []
    for item in scenario_records:
        scenario = item["scenario"]
        rate_path = item.get("rate_nc_path")
        if rate_path is None or not rate_path.exists():
            continue
        try:
            series = load_gm_rate_netcdf(rate_path)
        except (KeyError, ValueError):
            continue
        plot_rows.append(
            (
                scenario,
                series,
                item.get("row", {}),
            )
        )

    if not plot_rows:
        return {}

    is_4ka = workflow == "4ka"
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    max_age = -np.inf
    min_age = np.inf
    for scenario, series, row in plot_rows:
        max_age = max(max_age, float(np.nanmax(series.age_bp)))
        min_age = min(min_age, float(np.nanmin(series.age_bp)))
        style = _scenario_plot_style(
            scenario, bounds_multiplier=row.get("bounds_multiplier")
        )
        plot_gm_rate(
            ax,
            series,
            label=scenario,
            color=style["color"] or "gray",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            uncertainty_sigma=2.0,
            uncertainty_alpha=0.10,
        )

    if not np.isfinite(max_age) or not np.isfinite(min_age):
        max_age, min_age = (4000.0, 0.0) if is_4ka else (16000.0, 0.0)

    if not is_4ka:
        mwp_lo, mwp_hi = 11000.0, 11500.0
        if mwp_hi >= min_age and mwp_lo <= max_age:
            ax.axvspan(
                mwp_lo,
                mwp_hi,
                facecolor="gray",
                alpha=0.22,
                zorder=0,
                linewidth=0,
            )
            y_bottom, y_top = ax.get_ylim()
            y_span = y_top - y_bottom
            ax.text(
                0.5 * (mwp_lo + mwp_hi),
                y_bottom + 0.05 * y_span,
                "MWP-1B\n(~11.5–11.0 ka BP)",
                ha="center",
                va="bottom",
                fontsize=14,
                color="dimgray",
                zorder=3,
            )

    ax.set_xlim(max_age, min_age)
    ax.set_xlabel("Age (BP)", fontsize=16)
    ax.set_ylabel(r"Rate (mm a$^{-1}$)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_title("Basin-wide Common-Component Rate Comparison", fontsize=16)
    ax.legend(loc="best", fontsize=14, ncol=2)
    fig.tight_layout()

    pdf_path = output_root / "rate_global_comparison.pdf"
    png_path = output_root / "rate_global_comparison.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {"global_rate_pdf": pdf_path, "global_rate_png": png_path}


def _resolve_region_index_for_summary(
    output_root: Path,
    scenario_records: List[Dict[str, Any]],
    *,
    workflow: str | None = None,
) -> tuple[np.ndarray | None, List[int], List[int], Path | None]:
    """Load paleo CSV from first scenario config and build Region.1 row mapping."""
    paleo_csv: Path | None = None
    num_regions = 0
    for item in scenario_records:
        rate_path = item.get("rate_nc_path")
        if rate_path is None or not rate_path.exists():
            continue
        with nc.Dataset(rate_path) as ds:
            num_regions = int(ds.dimensions["region"].size)
        scenario_dir = Path(item.get("row", {}).get("output_dir", ""))
        config_path = scenario_dir / "config.yaml"
        if config_path.exists():
            cfg = _read_yaml(config_path)
            path_text = (cfg.get("paths", {}) or {}).get("paleo_rsl")
            if path_text:
                paleo_path = Path(str(path_text)).expanduser()
                if not paleo_path.is_absolute():
                    paleo_path = (config_path.parent / paleo_path).resolve()
                else:
                    paleo_path = paleo_path.resolve()
                if paleo_path.is_file():
                    paleo_csv = paleo_path
        break

    if num_regions == 0:
        return None, [], [], None

    if paleo_csv is not None:
        nc_row_to_region_id, summary_rows = compute_nc_row_to_region_id(
            num_regions, paleo_csv
        )
    else:
        nc_row_to_region_id, summary_rows = None, list(range(num_regions))

    linear_rate_rows = list(summary_rows)
    if workflow == "4ka":
        linear_rate_rows = filter_nc_rows_by_region_ids(
            nc_row_to_region_id,
            summary_rows,
            DEFAULT_SLIP_REGION_IDS,
        )

    if paleo_csv is not None and nc_row_to_region_id is not None:
        index_rows = []
        for row_idx in range(num_regions):
            index_rows.append(
                {
                    "nc_row": row_idx,
                    "region_index": int(nc_row_to_region_id[row_idx]),
                    "in_summary_panels": row_idx in summary_rows,
                    "in_linear_rate_panels": row_idx in linear_rate_rows,
                }
            )
        index_path = output_root / "summary_region_index.csv"
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        return nc_row_to_region_id, summary_rows, linear_rate_rows, index_path

    return nc_row_to_region_id, summary_rows, linear_rate_rows, None


def _write_regional_linear_rate_plot(
    scenario_records: List[Dict[str, Any]],
    output_root: Path,
    *,
    workflow: str = "16ka",
) -> Dict[str, Path]:
    nc_row_to_region_id, _, plot_region_indices, _ = _resolve_region_index_for_summary(
        output_root, scenario_records, workflow=workflow
    )
    if not plot_region_indices:
        return {}

    plot_rows = []
    for item in scenario_records:
        scenario = item["scenario"]
        rate_path = item.get("rate_nc_path")
        if rate_path is None or not rate_path.exists():
            continue
        with nc.Dataset(rate_path) as ds:
            if (
                "rate_time" not in ds.variables
                or "regional_linear_rate_mean" not in ds.variables
            ):
                continue
            rate_time = np.asarray(ds.variables["rate_time"][:], dtype=float)
            rl_mean = np.asarray(
                ds.variables["regional_linear_rate_mean"][:], dtype=float
            )
            rl_std = (
                np.asarray(ds.variables["regional_linear_rate_std"][:], dtype=float)
                if "regional_linear_rate_std" in ds.variables
                else None
            )
        plot_rows.append((scenario, rate_time, rl_mean, rl_std, item.get("row", {})))

    if not plot_rows:
        return {}

    n_panels = len(plot_region_indices)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols)) if n_panels else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(24, max(10, 3.6 * nrows)),
        constrained_layout=False,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    xlim = (4000, 0) if workflow == "4ka" else None

    for ax_idx, region_i in enumerate(plot_region_indices):
        ax = axes_flat[ax_idx]
        for scenario, rate_time, rl_mean, rl_std, row in plot_rows:
            if region_i >= rl_mean.shape[0]:
                continue
            style = _scenario_plot_style(
                scenario, bounds_multiplier=row.get("bounds_multiplier")
            )
            rmean = rl_mean[region_i]
            plot_kw = {
                "linewidth": style["linewidth"],
                "label": scenario,
                "linestyle": style["linestyle"],
            }
            if style["color"] is not None:
                plot_kw["color"] = style["color"]
            ax.plot(rate_time, -rmean * 1000.0, **plot_kw)
            if (
                style.get("is_baseline")
                and rl_std is not None
                and region_i < rl_std.shape[0]
            ):
                rstd = rl_std[region_i]
                fill_color = style["color"] if style["color"] is not None else "C5"
                ax.fill_between(
                    rate_time,
                    -(rmean - rstd) * 1000.0,
                    -(rmean + rstd) * 1000.0,
                    alpha=0.10,
                    color=fill_color,
                )

        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.set_xticks([4000, 3000, 2000, 1000, 0])
        ax.set_xlabel("Age (BP)", fontsize=11)
        ax.set_ylabel("Regional Linear Rate (mm/yr)", fontsize=10)
        ax.set_title(
            region_display_name(region_i, nc_row_to_region_id),
            fontsize=13,
        )
        ax.tick_params(labelsize=10)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = [], []
    for ax_idx in range(n_panels):
        h, lab = axes_flat[ax_idx].get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        nlab = len(labels)
        ncol = min(6, max(2, (nlab + 3) // 4))
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            fontsize=8,
            borderaxespad=0.5,
            frameon=True,
        )
    fig.subplots_adjust(
        left=0.06,
        right=0.74,
        bottom=0.07,
        top=0.96,
        hspace=0.36,
        wspace=0.28,
    )

    pdf_path = output_root / "rate_regional_linear_comparison.pdf"
    png_path = output_root / "rate_regional_linear_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return {
        "regional_linear_rate_pdf": pdf_path,
        "regional_linear_rate_png": png_path,
    }


def _parameter_row_order(parameters: Iterable[str]) -> List[str]:
    """Order parameter names for summary tables."""
    present = set(parameters)
    ordered: List[str] = []
    for internal, _display in _boundary_heatmap_column_order():
        if internal in present:
            ordered.append(internal)
            present.discard(internal)
    ordered.extend(sorted(present))
    return ordered


_PARAMS_BOUNDS_NILL_LONG_COLUMNS = [
    "workflow",
    "scenario",
    "parameter",
    "optimized",
    "bound_low",
    "bound_high",
    "proximity_pct",
    "boundary_hit",
    "boundary_side",
    "NILL",
    "scale",
    "prior_key",
    "conversion",
]


def _coerce_summary_float(value: Any, default: float = float("nan")) -> float:
    """Convert metric values to float; None and non-numeric become ``default``."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _format_summary_number(value: float, *, kind: str = "param") -> str:
    if not np.isfinite(value):
        return ""
    if kind == "nill":
        return f"{value:.4g}"
    return f"{value:.4g}"


def _proximity_with_marker(proximity: float) -> str:
    if not np.isfinite(proximity):
        return ""
    marker = _boundary_marker(proximity)
    text = f"{proximity:.2f}"
    return f"{text}*" if marker else text


def _boundary_side_label(boundary_hit: Any, boundary_side: Any) -> str:
    if boundary_hit in (True, "True", "true", 1):
        side = str(boundary_side).strip().lower()
        if side in {"lower", "upper"}:
            return side
        return "hit"
    return ""


def build_params_bounds_nill_long(
    df_boundary: pd.DataFrame,
    df_metrics: pd.DataFrame,
    scenario_order: List[str],
) -> pd.DataFrame:
    """
    Build a long table: one row per scenario x parameter with optimized value,
    bounds, proximity, and scenario-level NILL.
    """
    if df_boundary.empty:
        return pd.DataFrame(columns=_PARAMS_BOUNDS_NILL_LONG_COLUMNS)

    df = df_boundary.copy()
    if "value" in df.columns:
        df = df.rename(columns={"value": "optimized"})
    elif "optimized" not in df.columns:
        df["optimized"] = float("nan")

    if not df_metrics.empty and "NILL" in df_metrics.columns:
        nill_by_scenario = df_metrics.set_index("scenario")["NILL"]
        df["NILL"] = pd.to_numeric(
            df["scenario"].map(nill_by_scenario), errors="coerce"
        )
    else:
        df["NILL"] = float("nan")

    boundary_scenarios = set(df["scenario"].unique())
    scenarios = [s for s in scenario_order if s in boundary_scenarios]
    if not scenarios:
        scenarios = sorted(boundary_scenarios)

    param_order = _parameter_row_order(df["parameter"].unique())
    scenario_cat = pd.Categorical(df["scenario"], categories=scenarios, ordered=True)
    param_cat = pd.Categorical(df["parameter"], categories=param_order, ordered=True)
    df = df.assign(_scenario_order=scenario_cat, _param_order=param_cat)
    df = df.sort_values(["_scenario_order", "_param_order"]).drop(
        columns=["_scenario_order", "_param_order"]
    )

    for col in _PARAMS_BOUNDS_NILL_LONG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, _PARAMS_BOUNDS_NILL_LONG_COLUMNS].reset_index(drop=True)


def write_params_bounds_nill_markdown(
    df_long: pd.DataFrame,
    output_path: Path,
    *,
    workflow: str | None = None,
) -> None:
    """Write a markdown report with NILL summary and per-scenario parameter tables."""
    lines: List[str] = [
        "# Sensitivity Summary: Optimized Parameters, Bounds, and NILL",
        "",
    ]
    if workflow:
        lines.extend([f"Workflow: `{workflow}`", ""])

    if df_long.empty:
        lines.append("_No data available._")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    scenario_nill = df_long.drop_duplicates("scenario")[["scenario", "NILL"]]
    scenarios = scenario_nill["scenario"].tolist()

    lines.extend(["## Scenario NILL", "", "| Scenario | NILL |", "| --- | ---: |"])
    for _, row in scenario_nill.iterrows():
        nill_val = _coerce_summary_float(row["NILL"])
        lines.append(
            f"| {row['scenario']} | {_format_summary_number(nill_val, kind='nill')} |"
        )

    lines.extend(
        [
            "",
            "## Parameter Optimized Values and Bounds",
            "",
            "* prox = proximity within prior range (%); * marks boundary hit "
            "(<=5% or >=95%)",
            "",
        ]
    )

    for scenario in scenarios:
        scenario_rows = df_long[df_long["scenario"] == scenario]
        if scenario_rows.empty:
            continue
        lines.extend([f"### {scenario}", ""])
        header = [
            "parameter",
            "optimized",
            "bound_low",
            "bound_high",
            "proximity",
            "boundary",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for _, row in scenario_rows.iterrows():
            prox = float(row.get("proximity_pct", np.nan))
            cells = [
                str(row["parameter"]),
                _format_summary_number(float(row.get("optimized", np.nan))),
                _format_summary_number(float(row.get("bound_low", np.nan))),
                _format_summary_number(float(row.get("bound_high", np.nan))),
                _proximity_with_marker(prox),
                _boundary_side_label(row.get("boundary_hit"), row.get("boundary_side")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            f"- Scenario order: {', '.join(scenarios)}",
            "- Full long table: `summary_params_bounds_nill.csv`",
            "- Bound proximities computed in linear or log space per prior scale.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_workflow_summary(output_root: Path) -> Dict[str, Path]:
    """
    Create aggregated summary CSV files for one workflow output directory.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    df_params_all: List[pd.DataFrame] = []
    df_boundary_all: List[pd.DataFrame] = []

    scenario_records: List[Dict[str, Any]] = []
    for scenario_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        manifest_path = scenario_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = _read_yaml(manifest_path)
        metrics_path = scenario_dir / "metrics.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        scenario_name = manifest.get("name", scenario_dir.name)
        kernel_enabled = manifest.get("kernel_enabled")
        if not isinstance(kernel_enabled, dict):
            kernel_enabled = {}
        regional_linear_enabled = kernel_enabled.get(
            "regional_linear",
            manifest.get("regional_linear_enabled"),
        )
        workflow_name = _infer_workflow_name(manifest, metrics)
        rsl_nc_path = _candidate_nc_path(
            metrics.get("rsl_nc_path"),
            _scenario_nc_fallback(
                scenario_dir, scenario_name, kind="rsl", workflow=workflow_name
            ),
        )
        rate_nc_path = _candidate_nc_path(
            metrics.get("rate_nc_path"),
            _scenario_nc_fallback(
                scenario_dir, scenario_name, kind="rate", workflow=workflow_name
            ),
        )
        peak_global_rate = metrics.get("peak_global_rate_mm_yr")
        if peak_global_rate is None:
            peak_global_rate = _safe_rate_peak_global(
                rate_nc_path, workflow=workflow_name
            )

        row = {
            "workflow": manifest.get("workflow"),
            "scenario": scenario_name,
            "category": manifest.get("category"),
            "status": manifest.get("status"),
            "started_at_utc": manifest.get("started_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc"),
            "bounds_multiplier": manifest.get("bounds_multiplier"),
            "regional_linear_enabled": regional_linear_enabled,
            "merged_nonlinear_enabled": manifest.get("merged_nonlinear_enabled"),
            "negative_log_likelihood": _coerce_summary_float(
                metrics.get("negative_log_likelihood")
            ),
            "NILL": _coerce_summary_float(metrics.get("negative_log_likelihood")),
            "peak_global_rate_mm_yr": peak_global_rate,
            "mwp1b_global_rate_mm_yr": _coerce_summary_float(
                metrics.get("mwp1b_global_rate_mm_yr")
            ),
            "mwp1b_global_rate_age_bp": _coerce_summary_float(
                metrics.get("mwp1b_global_rate_age_bp")
            ),
            "mwp1b_global_rate_pointwise_2sigma_mm_yr": _coerce_summary_float(
                metrics.get("mwp1b_global_rate_pointwise_2sigma_mm_yr")
            ),
            "rate_change_pct": float("nan"),
            "max_rsl_diff_m": float("nan"),
            "n_training_points": metrics.get("n_training_points"),
            "n_hyperparameters": metrics.get("n_hyperparameters"),
            "rsl_nc_path": str(rsl_nc_path) if rsl_nc_path else "",
            "rate_nc_path": str(rate_nc_path) if rate_nc_path else "",
            "output_dir": str(scenario_dir),
        }
        bounds_multiplier = row.get("bounds_multiplier")
        bounds_multiplier_value = None
        if bounds_multiplier is not None:
            try:
                bounds_multiplier_value = float(bounds_multiplier)
            except (TypeError, ValueError):
                bounds_multiplier_value = None
        if (
            bounds_multiplier_value is not None
            and np.isfinite(bounds_multiplier_value)
            and bounds_multiplier_value < 1.0
        ):
            # Skip legacy narrowed-bound scenarios in updated summaries.
            continue
        rows.append(row)
        scenario_records.append(
            {
                "scenario": scenario_name,
                "row": row,
                "rsl_nc_path": rsl_nc_path,
                "rate_nc_path": rate_nc_path,
            }
        )

        params_csv = scenario_dir / "optimized_params.csv"
        if params_csv.exists():
            df = pd.read_csv(params_csv)
            df.insert(0, "scenario", row["scenario"])
            df.insert(0, "workflow", row["workflow"])
            df_params_all.append(df)

        boundary_csv = scenario_dir / "boundary_proximity.csv"
        if boundary_csv.exists():
            df = pd.read_csv(boundary_csv)
            df.insert(0, "scenario", row["scenario"])
            df.insert(0, "workflow", row["workflow"])
            df_boundary_all.append(df)

    record_by_scenario = {item["scenario"]: item for item in scenario_records}
    adopted_record = record_by_scenario.get("adopted")

    adopted_peak = float("nan")
    if adopted_record is not None:
        adopted_peak = float(
            adopted_record["row"].get("peak_global_rate_mm_yr", np.nan)
        )
        if not np.isfinite(adopted_peak):
            adopted_workflow = str(adopted_record["row"].get("workflow") or "16ka")
            adopted_peak = _safe_rate_peak_global(
                adopted_record.get("rate_nc_path"),
                workflow=adopted_workflow,
            )

    if np.isfinite(adopted_peak):
        for row in rows:
            peak = row.get("peak_global_rate_mm_yr")
            if peak is None or not np.isfinite(float(peak)):
                continue
            row["rate_change_pct"] = (float(peak) - adopted_peak) / adopted_peak * 100.0

    max_rsl_diff_by_region_path = output_root / "max_rsl_difference.csv"
    if (
        adopted_record is not None
        and adopted_record.get("rsl_nc_path") is not None
        and adopted_record["rsl_nc_path"].exists()
    ):
        with nc.Dataset(adopted_record["rsl_nc_path"]) as ds:
            adopted_total_mean = ds.variables["total_mean"][:]
            adopted_labels = _region_labels_from_dataset(
                ds, adopted_total_mean.shape[0]
            )

        diff_rows: List[Dict[str, Any]] = []
        for item in scenario_records:
            scenario = item["scenario"]
            if scenario == "adopted":
                continue
            rsl_path = item.get("rsl_nc_path")
            if rsl_path is None or not rsl_path.exists():
                continue
            with nc.Dataset(rsl_path) as ds:
                if "total_mean" not in ds.variables:
                    continue
                total_mean = ds.variables["total_mean"][:]

            n_region = min(total_mean.shape[0], adopted_total_mean.shape[0])
            n_time = min(total_mean.shape[1], adopted_total_mean.shape[1])
            diff = np.abs(
                total_mean[:n_region, :n_time] - adopted_total_mean[:n_region, :n_time]
            )
            item["row"]["max_rsl_diff_m"] = float(np.max(diff))

            for idx in range(n_region):
                diff_rows.append(
                    {
                        "Test": scenario,
                        "Region": (
                            adopted_labels[idx]
                            if idx < len(adopted_labels)
                            else f"R{idx+1}"
                        ),
                        "Max_RSL_diff_m": float(np.max(diff[idx])),
                    }
                )

        if diff_rows:
            df_diff = pd.DataFrame(diff_rows)
            df_diff_pivot = df_diff.pivot(
                index="Region", columns="Test", values="Max_RSL_diff_m"
            )
            df_diff_pivot.to_csv(max_rsl_diff_by_region_path)
        else:
            pd.DataFrame(columns=["Region"]).to_csv(
                max_rsl_diff_by_region_path, index=False
            )
    else:
        pd.DataFrame(columns=["Region"]).to_csv(
            max_rsl_diff_by_region_path, index=False
        )

    peak_rate_comparison_path = output_root / "peak_rate_comparison.csv"
    peak_rows: List[Dict[str, Any]] = []
    summary_workflow = "16ka"
    if scenario_records:
        summary_workflow = str(scenario_records[0]["row"].get("workflow") or "16ka")
    peak_lo, peak_hi = peak_rate_age_window(summary_workflow)
    nc_row_to_region_id, summary_region_rows, _, region_index_path = (
        _resolve_region_index_for_summary(
            output_root, scenario_records, workflow=summary_workflow
        )
    )

    for item in scenario_records:
        scenario = item["scenario"]
        rate_path = item.get("rate_nc_path")
        if rate_path is None or not rate_path.exists():
            continue

        with nc.Dataset(rate_path) as ds:
            if (
                "rate_time" not in ds.variables
                or "total_rate_mean" not in ds.variables
                or "global_rate_mean" not in ds.variables
            ):
                continue
            rate_time = np.asarray(ds.variables["rate_time"][:], dtype=float)
            total_rate_mean = ds.variables["total_rate_mean"][:]
            global_rate_mean = ds.variables["global_rate_mean"][:]
            labels = _region_labels_from_dataset(ds, total_rate_mean.shape[0])

        item_workflow = str(item["row"].get("workflow") or summary_workflow)
        peak_lo, peak_hi = peak_rate_age_window(item_workflow)
        mask = (rate_time >= peak_lo) & (rate_time <= peak_hi)
        if not np.any(mask):
            continue
        region_indices = (
            summary_region_rows
            if summary_region_rows
            else list(range(total_rate_mean.shape[0]))
        )
        for idx in region_indices:
            if idx >= total_rate_mean.shape[0]:
                continue
            region_label = region_display_name(idx, nc_row_to_region_id)
            if idx < len(labels) and nc_row_to_region_id is None:
                region_label = labels[idx]
            peak_rows.append(
                {
                    "Test": scenario,
                    "Region": region_label,
                    "Peak_rate_mm_yr": float(
                        np.max(-total_rate_mean[idx, mask] * 1000.0)
                    ),
                }
            )
        peak_rows.append(
            {
                "Test": scenario,
                "Region": "Global",
                "Peak_rate_mm_yr": float(np.max(-global_rate_mean[mask] * 1000.0)),
            }
        )

    if peak_rows:
        df_peak = pd.DataFrame(peak_rows)
        df_peak_pivot = df_peak.pivot(
            index="Region", columns="Test", values="Peak_rate_mm_yr"
        )
        if "adopted" in df_peak_pivot.columns:
            for col in list(df_peak_pivot.columns):
                if col == "adopted":
                    continue
                df_peak_pivot[f"{col} diff%"] = (
                    (df_peak_pivot[col] - df_peak_pivot["adopted"])
                    / df_peak_pivot["adopted"]
                    * 100.0
                )
        df_peak_pivot.to_csv(peak_rate_comparison_path)
    else:
        pd.DataFrame(columns=["Region"]).to_csv(peak_rate_comparison_path, index=False)

    overview_path = output_root / "summary_overview.csv"
    df_overview = pd.DataFrame(rows)
    df_overview.to_csv(overview_path, index=False)

    metrics_summary_path = output_root / "summary_metrics.csv"
    metrics_cols = [
        "scenario",
        "NILL",
        "peak_global_rate_mm_yr",
        "mwp1b_global_rate_mm_yr",
        "mwp1b_global_rate_age_bp",
        "mwp1b_global_rate_pointwise_2sigma_mm_yr",
        "rate_change_pct",
        "max_rsl_diff_m",
    ]
    missing_cols = [col for col in metrics_cols if col not in df_overview.columns]
    if missing_cols:
        for col in missing_cols:
            df_overview[col] = np.nan
    df_overview.loc[:, metrics_cols].to_csv(metrics_summary_path, index=False)

    params_summary_path = output_root / "summary_optimized_params.csv"
    if df_params_all:
        pd.concat(df_params_all, ignore_index=True).to_csv(
            params_summary_path, index=False
        )
    else:
        pd.DataFrame(columns=["workflow", "scenario", "Parameter", "Value"]).to_csv(
            params_summary_path, index=False
        )

    boundary_summary_path = output_root / "summary_boundary_proximity.csv"
    df_boundary_summary = pd.DataFrame()
    if df_boundary_all:
        df_boundary_summary = pd.concat(df_boundary_all, ignore_index=True)
        df_boundary_summary.to_csv(boundary_summary_path, index=False)
    else:
        df_boundary_summary = pd.DataFrame(
            columns=[
                "workflow",
                "scenario",
                "parameter",
                "value",
                "bound_low",
                "bound_high",
                "proximity_pct",
                "scale",
                "prior_key",
                "conversion",
            ]
        )
        df_boundary_summary.to_csv(boundary_summary_path, index=False)

    scenario_order = [item["scenario"] for item in scenario_records]
    df_metrics = df_overview.loc[:, metrics_cols]
    df_long = build_params_bounds_nill_long(
        df_boundary_summary,
        df_metrics,
        scenario_order,
    )
    wide_path = output_root / "summary_params_bounds_nill.csv"
    md_path = output_root / "summary_params_bounds_nill.md"
    df_long.to_csv(wide_path, index=False)
    workflow_name = None
    if not df_boundary_summary.empty and "workflow" in df_boundary_summary.columns:
        workflow_name = str(df_boundary_summary["workflow"].iloc[0])
    elif not df_overview.empty and "workflow" in df_overview.columns:
        workflow_name = str(df_overview["workflow"].iloc[0])
    write_params_bounds_nill_markdown(df_long, md_path, workflow=workflow_name)

    generated_plots = {}
    generated_plots.update(
        _write_boundary_heatmap(df_boundary_summary, output_root, scenario_order)
    )
    generated_plots.update(
        _write_global_rate_plot(
            scenario_records, output_root, workflow=summary_workflow
        )
    )
    generated_plots.update(
        _write_regional_linear_rate_plot(
            scenario_records, output_root, workflow=summary_workflow
        )
    )

    outputs = {
        "overview": overview_path,
        "metrics": metrics_summary_path,
        "optimized_params": params_summary_path,
        "boundary_proximity": boundary_summary_path,
        "params_bounds_nill": wide_path,
        "params_bounds_nill_md": md_path,
        "peak_rate_comparison": peak_rate_comparison_path,
        "max_rsl_difference": max_rsl_diff_by_region_path,
    }
    if region_index_path is not None:
        outputs["region_index"] = region_index_path
    outputs.update(generated_plots)
    return outputs
