"""Generate post-run figures from scenario outputs (PNG + PDF)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml

from stgp_dashboard.paths import ensure_import_paths

ensure_import_paths()

from stgp_pipeline import budget, data, export, visualize  # noqa: E402


def _save_figure(fig: plt.Figure, out_dir: Path, name: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    # 150 DPI matches typical notebook exports; PDF remains publication-quality.
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _find_netcdf(scenario_dir: Path, kind: str) -> Path | None:
    for pattern in (f"{kind}_*.nc", f"*{kind}*.nc"):
        matches = sorted(scenario_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w]+", "_", str(text).strip().lower())
    return slug.strip("_")[:48] or "region"


def _title_case_slug(slug: str) -> str:
    """Fallback display label when CSV region names are unavailable."""
    parts = [p for p in re.split(r"[_\s]+", str(slug).strip()) if p]
    return " ".join(p.capitalize() for p in parts) if parts else str(slug)


def _load_region_name_lookup(scenario_dir: str | Path) -> Dict[int, str]:
    """Map region id → display name from scenario CSVs (budget, then temporal)."""
    scenario_path = Path(scenario_dir).resolve()
    candidates = (
        scenario_path / "sea_level_budget.csv",
        scenario_path / "temporal_decomposition.csv",
        scenario_path / "temporal_decomposition_rates.csv",
    )
    lookup: Dict[int, str] = {}
    for csv_path in candidates:
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"region", "region_name"})
        except Exception:  # noqa: BLE001
            continue
        if "region" not in df.columns or "region_name" not in df.columns:
            continue
        for _, row in df[["region", "region_name"]].drop_duplicates().iterrows():
            try:
                rn = int(row["region"])
            except (TypeError, ValueError):
                continue
            name = str(row["region_name"]).strip()
            if rn not in lookup and name:
                lookup[rn] = name
        if lookup:
            break
    return lookup


def _region_label_from_stem(
    stem: str,
    *,
    region_lookup: Mapping[int, str] | None = None,
) -> str | None:
    """Resolve a display region name from ``..._region_{n}_{slug}`` stems."""
    marker = "_region_"
    if marker not in stem:
        return None
    after = stem.split(marker, 1)[1]
    parts = after.split("_", 1)
    if not parts:
        return None
    try:
        rn = int(parts[0])
    except ValueError:
        return _title_case_slug(after.replace("_", " "))
    slug = parts[1] if len(parts) > 1 else ""
    if region_lookup and rn in region_lookup:
        return str(region_lookup[rn])
    if slug:
        return _title_case_slug(slug)
    return f"Region {rn}"


def _load_run_config(
    scenario_dir: Path, metrics: Mapping[str, Any] | None
) -> Dict[str, Any]:
    cfg, _ = _load_run_config_with_path(scenario_dir, metrics)
    return cfg


def _load_run_config_with_path(
    scenario_dir: Path, metrics: Mapping[str, Any] | None
) -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates: List[Path] = []
    if metrics:
        cfg_hint = metrics.get("config_path") or metrics.get("saved_config_path")
        if cfg_hint:
            candidates.append(Path(str(cfg_hint)))
    candidates.append(scenario_dir.parent / "dashboard_config.yaml")
    candidates.append(scenario_dir / "config.yaml")
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}, path.resolve()
    return {}, None


def _load_paleo_rsl(cfg: Mapping[str, Any]):
    paths = cfg.get("paths") or {}
    paleo = paths.get("paleo_rsl") or paths.get("dataset")
    if not paleo:
        return None
    max_age_bp = int((cfg.get("analysis") or {}).get("max_age_bp", 16000))
    try:
        return data.load_and_filter_rsl(
            str(paleo), max_age_bp=max_age_bp, config=dict(cfg)
        )
    except Exception:  # noqa: BLE001
        return None


def _infer_temporal_comp_labels(df: pd.DataFrame) -> List[str]:
    found: List[str] = []
    for col in df.columns:
        if col.endswith("_mean") and col not in {
            "total_mean",
            "gp_residual_mean",
            "gia_mean",
            "global_mean",
        }:
            found.append(col[: -len("_mean")])
    ordered = [label for label in visualize.COMP_COLORS if label in found]
    for label in found:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _region_results_from_temporal_csv(
    temporal_csv: Path,
    rates_csv: Path | None,
) -> Tuple[Dict[int, Any], List[str]]:
    """Rebuild lightweight RegionResult-like objects for notebook plotter."""
    df = pd.read_csv(temporal_csv)
    rates = pd.read_csv(rates_csv) if rates_csv and rates_csv.is_file() else None
    comp_labels = _infer_temporal_comp_labels(df)
    results: Dict[int, Any] = {}

    for rn, grp in df.groupby("region"):
        grp = grp.sort_values("age_bp")
        region_name = str(grp["region_name"].iloc[0]) if "region_name" in grp else f"R{rn}"
        test_age = grp["age_bp"].to_numpy(dtype=float)
        y_total = grp["total_mean"].to_numpy(dtype=float)
        y_std = grp["total_std"].to_numpy(dtype=float)
        gia_ts = (
            grp["gia_mean"].to_numpy(dtype=float)
            if "gia_mean" in grp.columns
            else np.zeros_like(y_total)
        )
        components: Dict[str, Dict[str, np.ndarray]] = {}
        for label in comp_labels:
            mean_col, std_col = f"{label}_mean", f"{label}_std"
            if mean_col not in grp.columns:
                continue
            components[label] = {
                "mean": grp[mean_col].to_numpy(dtype=float),
                "std": grp[std_col].to_numpy(dtype=float)
                if std_col in grp.columns
                else np.zeros(len(grp)),
                "rate": np.array([], dtype=float),
                "rate_sd": np.array([], dtype=float),
            }

        rsl_time = test_age[:-1] if len(test_age) > 1 else np.array([], dtype=float)
        rsl_rate = np.zeros_like(rsl_time)
        rsl_rate_sd = np.zeros_like(rsl_time)
        gia_rate = np.zeros_like(rsl_time)

        if rates is not None and not rates.empty:
            rgrp = rates[rates["region"] == rn].sort_values("age_bp")
            if not rgrp.empty:
                rsl_time = rgrp["age_bp"].to_numpy(dtype=float)
                rsl_rate = rgrp["total_rate"].to_numpy(dtype=float)
                rsl_rate_sd = rgrp["total_rate_sd"].to_numpy(dtype=float)
                gia_rate = (
                    rgrp["gia_rate"].to_numpy(dtype=float)
                    if "gia_rate" in rgrp.columns
                    else np.zeros_like(rsl_time)
                )
                for label in comp_labels:
                    rate_col, rate_sd_col = f"{label}_rate", f"{label}_rate_sd"
                    if label not in components:
                        continue
                    if rate_col in rgrp.columns:
                        components[label]["rate"] = rgrp[rate_col].to_numpy(dtype=float)
                        components[label]["rate_sd"] = (
                            rgrp[rate_sd_col].to_numpy(dtype=float)
                            if rate_sd_col in rgrp.columns
                            else np.zeros(len(rgrp))
                        )

        results[int(rn)] = SimpleNamespace(
            region_num=int(rn),
            region_name=region_name,
            test_age=test_age,
            y_total=y_total,
            y_std=y_std,
            gia_ts=gia_ts,
            rsl_time=rsl_time,
            rsl_rate=rsl_rate,
            rsl_rate_sd=rsl_rate_sd,
            gia_rate=gia_rate,
            components=components,
        )
    return results, comp_labels


def _region_labels_from_netcdf(ds: nc.Dataset, n_regions: int) -> List[str]:
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

    try:
        labels = [
            "".join(np.asarray(ds.variables["region_name"][i, :], dtype=str)).strip()
            for i in range(n_regions)
        ]
        if labels:
            return labels
    except Exception:  # noqa: BLE001
        pass

    return [f"R{i + 1}" for i in range(n_regions)]


def _discover_component_labels(rsl_ds: nc.Dataset) -> List[str]:
    found: List[str] = []
    for var in rsl_ds.variables:
        if var.endswith("_mean") and var not in {"total_mean", "global_mean"}:
            found.append(var[: -len("_mean")])
    ordered = [label for label in visualize.COMP_COLORS if label in found]
    for label in found:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _region_results_from_netcdf(
    rsl_nc: Path,
    rate_nc: Path | None,
) -> Tuple[Dict[int, Any], List[str]]:
    """Fallback reconstruction when temporal CSV is absent (e.g. 16ka)."""
    results: Dict[int, Any] = {}
    with nc.Dataset(rsl_nc) as rsl_ds:
        times = np.asarray(rsl_ds.variables["time"][:], dtype=float)
        n_regions = rsl_ds.dimensions["region"].size
        region_names = _region_labels_from_netcdf(rsl_ds, n_regions)
        comp_labels = _discover_component_labels(rsl_ds)
        total_mean = np.asarray(rsl_ds.variables["total_mean"][:], dtype=float)
        total_std = np.asarray(rsl_ds.variables["total_std"][:], dtype=float)
        comp_mean = {
            label: np.asarray(rsl_ds.variables[f"{label}_mean"][:], dtype=float)
            for label in comp_labels
            if f"{label}_mean" in rsl_ds.variables
        }
        comp_std = {
            label: np.asarray(rsl_ds.variables[f"{label}_std"][:], dtype=float)
            for label in comp_labels
            if f"{label}_std" in rsl_ds.variables
        }

    rate_times = times[:-1] if len(times) > 1 else np.array([], dtype=float)
    total_rate = np.zeros((n_regions, len(rate_times)), dtype=float)
    total_rate_sd = np.zeros_like(total_rate)
    comp_rate: Dict[str, np.ndarray] = {}
    comp_rate_sd: Dict[str, np.ndarray] = {}
    if rate_nc and rate_nc.is_file():
        with nc.Dataset(rate_nc) as rate_ds:
            rate_times = np.asarray(rate_ds.variables["rate_time"][:], dtype=float)
            total_rate = np.asarray(rate_ds.variables["total_rate_mean"][:], dtype=float)
            total_rate_sd = np.asarray(
                rate_ds.variables["total_rate_std"][:], dtype=float
            )
            for label in comp_labels:
                key = f"{label}_rate_mean"
                if key in rate_ds.variables:
                    comp_rate[label] = np.asarray(rate_ds.variables[key][:], dtype=float)
                    sd_key = f"{label}_rate_std"
                    comp_rate_sd[label] = (
                        np.asarray(rate_ds.variables[sd_key][:], dtype=float)
                        if sd_key in rate_ds.variables
                        else np.zeros_like(comp_rate[label])
                    )

    for idx, region_name in enumerate(region_names):
        components: Dict[str, Dict[str, np.ndarray]] = {}
        for label in comp_labels:
            if label not in comp_mean:
                continue
            components[label] = {
                "mean": comp_mean[label][idx],
                "std": comp_std[label][idx],
                "rate": comp_rate[label][idx]
                if label in comp_rate
                else np.zeros_like(rate_times),
                "rate_sd": comp_rate_sd[label][idx]
                if label in comp_rate_sd
                else np.zeros_like(rate_times),
            }
        # NetCDF path has no GIA series (16ka). Use NaN so the notebook plotter
        # does not draw a misleading zero GIA curve.
        results[idx + 1] = SimpleNamespace(
            region_num=idx + 1,
            region_name=region_name,
            test_age=times,
            y_total=total_mean[idx],
            y_std=total_std[idx],
            gia_ts=np.full_like(times, np.nan),
            rsl_time=rate_times,
            rsl_rate=total_rate[idx] if idx < len(total_rate) else np.zeros_like(rate_times),
            rsl_rate_sd=total_rate_sd[idx]
            if idx < len(total_rate_sd)
            else np.zeros_like(rate_times),
            gia_rate=np.full_like(rate_times, np.nan),
            components=components,
        )
    return results, comp_labels


def _match_region_number(rsl_data: Any, region_name: str, fallback: int) -> int:
    """Map a region name / NetCDF index to ``rsl_data.rsl_region`` codes when possible."""
    if rsl_data is None:
        return fallback
    names = np.asarray(getattr(rsl_data, "rsl_region_name", []))
    regions = np.asarray(getattr(rsl_data, "rsl_region", []))
    if names.size and regions.size and names.shape[0] == regions.shape[0]:
        mask = names.astype(str) == str(region_name)
        if np.any(mask):
            return int(regions[mask][0])
        # Also try "R{n}" style names
        if str(region_name).startswith("R"):
            try:
                code = int(str(region_name).lstrip("R"))
                if np.any(regions == code):
                    return code
            except ValueError:
                pass
    return fallback


def _load_temporal_region_results(
    scenario_dir: Path,
    metrics: Mapping[str, Any] | None,
) -> Tuple[Dict[int, Any], List[str]]:
    """Load per-region temporal results from CSV or NetCDF artifacts."""
    temporal_csv = scenario_dir / "temporal_decomposition.csv"
    rates_csv = scenario_dir / "temporal_decomposition_rates.csv"

    if temporal_csv.is_file():
        return _region_results_from_temporal_csv(
            temporal_csv, rates_csv if rates_csv.is_file() else None
        )

    rsl_nc = None
    rate_nc = None
    if metrics:
        if metrics.get("rsl_nc_path"):
            rsl_nc = Path(str(metrics["rsl_nc_path"]))
        if metrics.get("rate_nc_path"):
            rate_nc = Path(str(metrics["rate_nc_path"]))
    if rsl_nc is None or not rsl_nc.is_file():
        rsl_nc = _find_netcdf(scenario_dir, "rsl")
    if rate_nc is None or not rate_nc.is_file():
        rate_nc = _find_netcdf(scenario_dir, "rate")
    if rsl_nc and rsl_nc.is_file():
        return _region_results_from_netcdf(rsl_nc, rate_nc)
    return {}, []


def _generate_temporal_decomposition_figures(
    scenario_dir: Path,
    figures_dir: Path,
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    rsl_data = _load_paleo_rsl(cfg)
    results, comp_labels = _load_temporal_region_results(scenario_dir, metrics)

    if not results or rsl_data is None:
        # Without RSL data we cannot draw notebook-style SLIP boxes; skip rather
        # than emit a simplified Total panel.
        if results and rsl_data is None:
            print(
                "Skipping temporal decomposition figures: paleo RSL data unavailable "
                "for SLIP overlay."
            )
        return outputs

    for rn, result in sorted(results.items()):
        plot_rn = _match_region_number(rsl_data, result.region_name, rn)
        # Restrict to labels the notebook plotter knows how to style.
        plot_labels = [
            label
            for label in (comp_labels or list(result.components.keys()))
            if label in visualize.COMP_COLORS and label in result.components
        ]
        if not plot_labels:
            plot_labels = [
                label
                for label in result.components
                if label in visualize.COMP_COLORS
            ]
        fig = visualize.plot_temporal_decomposition(
            plot_rn,
            result,
            rsl_data,
            comp_labels=plot_labels or None,
        )
        slug = _slugify(result.region_name)
        saved = _save_figure(
            fig,
            figures_dir,
            f"temporal_decomposition_region_{rn}_{slug}",
        )
        saved["name"] = f"temporal_decomposition_{result.region_name}"
        saved["category"] = "temporal_decomposition"
        saved["region"] = result.region_name
        outputs.append(saved)
    return outputs


def _generate_rsl_curve_figures(
    scenario_dir: Path,
    figures_dir: Path,
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
) -> List[Dict[str, str]]:
    """Total RSL only: one multi-panel figure with one curve per region."""
    outputs: List[Dict[str, str]] = []
    rsl_data = _load_paleo_rsl(cfg)
    results, _comp_labels = _load_temporal_region_results(scenario_dir, metrics)

    if not results or rsl_data is None:
        if results and rsl_data is None:
            print(
                "Skipping RSL curve figures: paleo RSL data unavailable "
                "for SLIP overlay."
            )
        return outputs

    plot_results: Dict[int, Any] = {}
    for rn, result in sorted(results.items()):
        plot_rn = _match_region_number(rsl_data, result.region_name, rn)
        plot_results[plot_rn] = result

    fig = visualize.plot_temporal_all_regions(plot_results, rsl_data)
    saved = _save_figure(fig, figures_dir, "rsl_curve_all_regions")
    saved["name"] = "rsl_curve_all_regions"
    saved["category"] = "rsl_curve"
    outputs.append(saved)
    return outputs


def _spatial_test_ages(cfg: Mapping[str, Any]) -> List[float]:
    analysis = cfg.get("analysis", {}) or {}
    max_age_bp = int(analysis.get("max_age_bp", 16000))
    spatial_cfg = cfg.get("spatial_test_age", {}) or {}
    ages = np.arange(
        int(spatial_cfg.get("start", 0)),
        int(spatial_cfg.get("end", max_age_bp)) + 1,
        int(spatial_cfg.get("interval", 1000)),
    )
    return [float(x) for x in ages]


def _generate_spatial_decomposition_figures(
    scenario_dir: Path,
    figures_dir: Path,
    cfg: Mapping[str, Any],
    rsl_data: Any = None,
) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    spatial_csv = scenario_dir / "spatial_predictions.csv"
    if not spatial_csv.is_file():
        return outputs

    df_spatial = pd.read_csv(spatial_csv)
    comp_labels = export._infer_spatial_component_labels(df_spatial)
    ages = _spatial_test_ages(cfg)
    slices = export.spatial_slices_from_dataframe(
        df_spatial, ages, comp_labels=comp_labels or None
    )
    for spatial_slice in slices:
        age_bp = int(round(spatial_slice.age_bp))
        fig = visualize.plot_spatial_decomposition(
            spatial_slice,
            comp_labels=comp_labels or None,
            rsl_data=rsl_data,
            cfg=cfg,
        )
        saved = _save_figure(fig, figures_dir, f"spatial_decomposition_{age_bp}bp")
        saved["name"] = f"spatial_decomposition_{age_bp}bp"
        saved["category"] = "spatial_decomposition"
        saved["age_bp"] = age_bp
        outputs.append(saved)
    return outputs


def _generate_spatial_rate_figures(
    scenario_dir: Path,
    figures_dir: Path,
    cfg: Mapping[str, Any],
    rsl_data: Any = None,
) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    rates_csv = scenario_dir / "spatial_rates.csv"
    if not rates_csv.is_file():
        return outputs

    df_rates = pd.read_csv(rates_csv)
    rate_result = export.spatial_rates_from_dataframe(df_rates)
    comp_labels = list(rate_result.rate_comp.keys())
    for interval_idx in range(len(rate_result.rate_midpoints)):
        younger = int(rate_result.spatial_ages[interval_idx])
        older = int(rate_result.spatial_ages[interval_idx + 1])
        fig = visualize.plot_spatial_rate(
            rate_result,
            interval_idx,
            comp_labels=comp_labels or None,
            rsl_data=rsl_data,
            cfg=cfg,
        )
        saved = _save_figure(
            fig,
            figures_dir,
            f"spatial_rate_{younger}_{older}bp",
        )
        saved["name"] = f"spatial_rate_{younger}_{older}bp"
        saved["category"] = "spatial_rate"
        saved["interval"] = f"{younger}-{older} BP"
        outputs.append(saved)
    return outputs


def _infer_figure_metadata(
    png_path: Path,
    *,
    region_lookup: Mapping[int, str] | None = None,
) -> Dict[str, Any]:
    """Rebuild gallery metadata from a saved figure filename."""
    stem = png_path.stem
    item: Dict[str, Any] = {
        "png": str(png_path),
        "pdf": str(png_path.with_suffix(".pdf"))
        if png_path.with_suffix(".pdf").is_file()
        else "",
        "name": stem,
    }
    if stem.startswith("rsl_curve"):
        item["category"] = "rsl_curve"
    elif stem.startswith("temporal_decomposition"):
        item["category"] = "temporal_decomposition"
        region = _region_label_from_stem(stem, region_lookup=region_lookup)
        if region is not None:
            item["region"] = region
        else:
            raw = stem.replace("temporal_decomposition_", "")
            item["region"] = _title_case_slug(raw.replace("_", " "))
    elif stem.startswith("sea_level_budget"):
        item["category"] = "sea_level_budget"
        region = _region_label_from_stem(stem, region_lookup=region_lookup)
        if region is not None:
            item["region"] = region
        win_match = re.search(r"_(\d+)-(\d+)_BP$", stem)
        if win_match:
            item["interval"] = f"{win_match.group(1)}-{win_match.group(2)} BP"
    elif stem.startswith("spatial_decomposition"):
        item["category"] = "spatial_decomposition"
        match = re.search(r"_(\d+)bp$", stem)
        if match:
            item["age_bp"] = int(match.group(1))
    elif stem.startswith("spatial_rate"):
        item["category"] = "spatial_rate"
        match = re.search(r"_(\d+)_(\d+)bp$", stem)
        if match:
            item["interval"] = f"{match.group(1)}-{match.group(2)} BP"
    elif stem.startswith("optimization"):
        item["category"] = "optimization"
    else:
        item["category"] = "other"
    return item


def discover_saved_figures(scenario_dir: str | Path) -> List[Dict[str, Any]]:
    """Scan ``scenario_dir/figures/*.png`` and rebuild gallery metadata."""
    figures_dir = Path(scenario_dir).resolve() / "figures"
    if not figures_dir.is_dir():
        return []
    region_lookup = _load_region_name_lookup(scenario_dir)
    outputs: List[Dict[str, Any]] = []
    for png_path in sorted(figures_dir.glob("*.png")):
        outputs.append(
            _infer_figure_metadata(png_path, region_lookup=region_lookup)
        )
    return outputs


def generate_budget_figures(
    scenario_dir: str | Path,
    *,
    age_windows: Sequence[Tuple[float, float]] | None = None,
    merge_gia_global: bool | None = None,
    metrics: Mapping[str, Any] | None = None,
    recompute: bool = True,
) -> List[Dict[str, Any]]:
    """Export sea-level budget CSVs/PDF (optional) and gallery PNG/PDFs.

    When ``recompute`` is True (default), regenerates ``sea_level_budget.*`` from
    ``temporal_decomposition_rates.csv`` using ``age_windows`` / config / default
    ``0–4000`` BP. Gallery figures always rebuild from the budget CSV when present.
    """
    scenario_path = Path(scenario_dir).resolve()
    figures_dir = scenario_path / "figures"
    rates_csv = scenario_path / "temporal_decomposition_rates.csv"
    budget_csv = scenario_path / "sea_level_budget.csv"
    outputs: List[Dict[str, Any]] = []

    cfg, cfg_path = _load_run_config_with_path(scenario_path, metrics)
    budget_cfg = dict(cfg.get("budget") or {})
    if age_windows is not None:
        budget_cfg["age_windows_bp"] = [
            [float(a), float(b)] for a, b in age_windows
        ]
    if merge_gia_global is not None:
        budget_cfg["merge_gia_global"] = bool(merge_gia_global)
    base_dir = cfg_path.parent if cfg_path is not None else scenario_path.parent

    if recompute:
        if not rates_csv.is_file():
            return outputs
        budget.export_scenario_sea_level_budget(
            scenario_path,
            budget_cfg,
            rates_csv=rates_csv,
            base_dir=base_dir,
        )
    elif not budget_csv.is_file() and rates_csv.is_file():
        budget.export_scenario_sea_level_budget(
            scenario_path,
            budget_cfg,
            rates_csv=rates_csv,
            base_dir=base_dir,
        )

    if not budget_csv.is_file():
        return outputs

    budget_df = pd.read_csv(budget_csv)
    if budget_df.empty:
        return outputs

    windows = [
        (float(a), float(b))
        for a, b in (
            budget_df[["window_start_bp", "window_end_bp"]]
            .drop_duplicates()
            .sort_values("window_start_bp")
            .to_numpy()
        )
    ]

    for lo, hi in windows:
        fig = budget.plot_budget_regions_grid(budget_df, window=(lo, hi))
        name = f"sea_level_budget_all_regions_{int(lo)}-{int(hi)}_BP"
        saved = _save_figure(fig, figures_dir, name)
        saved["name"] = name
        saved["category"] = "sea_level_budget"
        saved["interval"] = f"{int(lo)}-{int(hi)} BP"
        outputs.append(saved)

    field_csv = scenario_path / "sea_level_budget_field.csv"
    if field_csv.is_file():
        field_df = pd.read_csv(field_csv)
        if not field_df.empty:
            fig = budget.plot_field_contributions_pie(field_df)
            saved = _save_figure(fig, figures_dir, "sea_level_budget_field")
            saved["name"] = "sea_level_budget_field"
            saved["category"] = "sea_level_budget"
            outputs.append(saved)

    region_figs = budget.plot_rate_contributions_pie(budget_df)
    region_keys = list(
        budget_df.groupby(["region", "region_name"], sort=True).groups.keys()
    )
    for fig, (rn, region_name) in zip(region_figs, region_keys):
        slug = _slugify(str(region_name))
        name = f"sea_level_budget_region_{int(rn)}_{slug}"
        saved = _save_figure(fig, figures_dir, name)
        saved["name"] = name
        saved["category"] = "sea_level_budget"
        saved["region"] = str(region_name)
        outputs.append(saved)

    return outputs


def generate_post_run_figures(
    scenario_dir: str | Path,
    *,
    metrics: Mapping[str, Any] | None = None,
    mask_enabled: bool = True,
    mask_calibration_factor: float = 2.0,
    scope: str = "full",
) -> List[Dict[str, str]]:
    """
    Build decomposition temporal/spatial plots and optimization track from artifacts.

    Temporal figures use ``visualize.plot_temporal_decomposition`` (SLIP boxes + GIA)
    when paleo RSL and temporal CSV/NetCDF are available. Budget donuts follow
    temporal figures (notebook §5b order) before spatial maps.

    ``scope``:
    - ``"full"`` (default): full suite (unchanged from historical behavior).
    - ``"rsl_curve"``: Total RSL only via ``plot_temporal_all_regions``.
    """
    scenario_path = Path(scenario_dir).resolve()
    figures_dir = scenario_path / "figures"
    outputs: List[Dict[str, str]] = []
    cfg = _load_run_config(scenario_path, metrics)
    cfg["spatial_plot_mask"] = {
        "enabled": bool(mask_enabled),
        "calibration_factor": float(mask_calibration_factor),
    }

    if scope == "rsl_curve":
        outputs.extend(
            _generate_rsl_curve_figures(scenario_path, figures_dir, cfg, metrics)
        )
        return outputs

    rsl_data = _load_paleo_rsl(cfg)

    track_path = scenario_path / "optimization_track.csv"
    if track_path.is_file():
        track_df = pd.read_csv(track_path)
        if "loss" in track_df.columns:
            fig = visualize.plot_optimization_track(track_df)
            saved = _save_figure(fig, figures_dir, "optimization_track")
            saved["name"] = "optimization_track"
            saved["category"] = "optimization"
            outputs.append(saved)

    outputs.extend(
        _generate_temporal_decomposition_figures(
            scenario_path, figures_dir, cfg, metrics
        )
    )
    # Prefer existing budget CSV from postprocess; recompute only if missing.
    outputs.extend(
        generate_budget_figures(
            scenario_path,
            metrics=metrics,
            recompute=not (scenario_path / "sea_level_budget.csv").is_file(),
        )
    )
    outputs.extend(
        _generate_spatial_decomposition_figures(
            scenario_path, figures_dir, cfg, rsl_data=rsl_data
        )
    )
    outputs.extend(
        _generate_spatial_rate_figures(
            scenario_path, figures_dir, cfg, rsl_data=rsl_data
        )
    )

    return outputs
