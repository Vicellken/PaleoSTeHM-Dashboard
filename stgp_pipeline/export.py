"""
DataFrame construction for CSV export.

Pure data transformations — no file I/O. The notebook calls
``ctx.save_dataframe()`` for persistence.

CSV column names match the existing notebook outputs for compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .types import (
    RegionResult,
    RegionalMeanRateResult,
    SpatialSlice,
    SpatialRateResult,
)

_SPATIAL_BASE_MEAN_COLS = frozenset(
    {"total_mean", "gp_residual_mean", "gia_mean"}
)


def temporal_to_dataframes(
    results: Dict[int, RegionResult],
    comp_labels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build temporal prediction and rate DataFrames from region results.

    Returns
    -------
    df_temporal : DataFrame — RSL predictions per region/age
    df_rates : DataFrame — rates per region/age
    """
    rows = []
    rate_rows = []

    for rn in sorted(results.keys()):
        res = results[rn]
        comps = res.components

        for j, t in enumerate(res.test_age):
            row = {
                "region": rn,
                "region_name": res.region_name,
                "age_bp": t,
                "total_mean": res.y_total[j],
                "total_std": res.y_std[j],
                "gp_residual_mean": float(res.y_res_mean[j]),
                "gia_mean": res.gia_ts[j],
                "gia_std": np.sqrt(res.gia_var[j]),
            }
            for label in comp_labels:
                row[f"{label}_mean"] = comps[label]["mean"][j]
                row[f"{label}_std"] = comps[label]["std"][j]
            rows.append(row)

        for j, t in enumerate(res.rsl_time):
            rrow = {
                "region": rn,
                "region_name": res.region_name,
                "age_bp": t,
                "total_rate": res.rsl_rate[j],
                "total_rate_sd": res.rsl_rate_sd[j],
                "gia_rate": res.gia_rate[j],
                "gia_rate_sd": res.gia_rate_sd[j],
            }
            for label in comp_labels:
                rrow[f"{label}_rate"] = comps[label]["rate"][j]
                rrow[f"{label}_rate_sd"] = comps[label]["rate_sd"][j]
            rate_rows.append(rrow)

    return pd.DataFrame(rows), pd.DataFrame(rate_rows)


def sites_to_dataframes(
    results: Dict[str, RegionResult],
    comp_labels: List[str],
    sites: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build temporal prediction and rate DataFrames from site-level results.

    Same columns as ``temporal_to_dataframes``, keyed by site name.
    Pass ``sites`` as (name, lat, lon) tuples to populate lat/lon columns.
    """
    coords: Dict[str, Tuple[float, float]] = {}
    if sites is not None:
        coords = {name: (lat, lon) for name, lat, lon in sites}

    rows = []
    rate_rows = []

    for name in sorted(results.keys()):
        res = results[name]
        comps = res.components
        lat, lon = coords.get(name, (float("nan"), float("nan")))

        for j, t in enumerate(res.test_age):
            row = {
                "site": name,
                "lat": lat,
                "lon": lon,
                "age_bp": t,
                "total_mean": res.y_total[j],
                "total_std": res.y_std[j],
                "gp_residual_mean": float(res.y_res_mean[j]),
                "gia_mean": res.gia_ts[j],
                "gia_std": np.sqrt(res.gia_var[j]),
            }
            for label in comp_labels:
                row[f"{label}_mean"] = comps[label]["mean"][j]
                row[f"{label}_std"] = comps[label]["std"][j]
            rows.append(row)

        for j, t in enumerate(res.rsl_time):
            rrow = {
                "site": name,
                "lat": lat,
                "lon": lon,
                "age_bp": t,
                "total_rate": res.rsl_rate[j],
                "total_rate_sd": res.rsl_rate_sd[j],
                "gia_rate": res.gia_rate[j],
                "gia_rate_sd": res.gia_rate_sd[j],
            }
            for label in comp_labels:
                rrow[f"{label}_rate"] = comps[label]["rate"][j]
                rrow[f"{label}_rate_sd"] = comps[label]["rate_sd"][j]
            rate_rows.append(rrow)

    return pd.DataFrame(rows), pd.DataFrame(rate_rows)


def regional_linear_rates_to_dataframe(df_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Extract regional linear kernel rates per region/age from ``df_rates``.

    Expects columns from ``temporal_to_dataframes`` (m/yr). Adds mm/yr columns
    using the same display sign convention as sensitivity summary plots.
    """
    rate_col = "regional_linear_rate"
    sd_col = "regional_linear_rate_sd"
    if rate_col not in df_rates.columns or sd_col not in df_rates.columns:
        raise KeyError(
            f"df_rates missing {rate_col!r} / {sd_col!r}; "
            "is regional_linear enabled in the kernel stack?"
        )

    out = df_rates[["region", "region_name", "age_bp", rate_col, sd_col]].copy()
    out = out.rename(
        columns={
            rate_col: "regional_linear_rate_m_yr",
            sd_col: "regional_linear_rate_sd_m_yr",
        }
    )
    out["regional_linear_rate_mm_yr"] = -out["regional_linear_rate_m_yr"] * 1000.0
    out["regional_linear_rate_sd_mm_yr"] = out["regional_linear_rate_sd_m_yr"] * 1000.0
    return out


def spatial_to_dataframe(
    slices: List[SpatialSlice],
    comp_labels: List[str],
) -> pd.DataFrame:
    """
    Build spatial predictions DataFrame from a list of SpatialSlice objects.

    Returns
    -------
    pd.DataFrame with one row per (age, lat, lon) grid point.
    """
    rows = []

    for s in slices:
        pm_lat = s.lat_mat.flatten()
        pm_lon = s.lon_mat.flatten()
        gp_mean_flat = s.gp_mean.flatten()
        gp_std_flat = np.sqrt(s.gp_var.flatten())
        gia_mean_flat = s.gia_mean.flatten()
        gia_std_flat = np.sqrt(s.gia_var.flatten())
        total_mean_flat = s.total_mean.flatten()
        total_std_flat = s.total_std.flatten()

        comp_arrays = {}
        for label in comp_labels:
            if label in s.components:
                comp_arrays[label] = {
                    "mean": s.components[label]["mean"].flatten(),
                    "std": s.components[label]["std"].flatten(),
                }

        for k in range(len(pm_lat)):
            row = {
                "age_bp": s.age_bp,
                "lat": pm_lat[k],
                "lon": pm_lon[k],
                "gp_residual_mean": gp_mean_flat[k],
                "gp_residual_std": gp_std_flat[k],
                "gia_mean": gia_mean_flat[k],
                "gia_std": gia_std_flat[k],
                "total_mean": total_mean_flat[k],
                "total_std": total_std_flat[k],
            }
            for label in comp_labels:
                if label in comp_arrays:
                    row[f"{label}_mean"] = comp_arrays[label]["mean"][k]
                    row[f"{label}_std"] = comp_arrays[label]["std"][k]
            rows.append(row)

    return pd.DataFrame(rows)


def _infer_spatial_component_labels(df: pd.DataFrame) -> List[str]:
    """Kernel labels from ``{label}_mean`` / ``{label}_std`` pairs in spatial CSV."""
    labels: List[str] = []
    for col in df.columns:
        if not col.endswith("_mean") or col in _SPATIAL_BASE_MEAN_COLS:
            continue
        std_col = f"{col[:-5]}_std"
        if std_col in df.columns:
            labels.append(col[:-5])
    return labels


def _spatial_age_subset(df: pd.DataFrame, age_bp: float) -> pd.DataFrame:
    """Rows for one snapshot age (tolerant float match)."""
    ages = df["age_bp"].to_numpy(dtype=float)
    mask = np.isclose(ages, float(age_bp), rtol=0.0, atol=0.5)
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(f"No spatial rows for age_bp={age_bp}")
    return sub


def _spatial_field_grid(sub: pd.DataFrame, lat_grid: np.ndarray, lon_grid: np.ndarray, col: str) -> np.ndarray:
    """Pivot one CSV column to (nlat, nlon) aligned with ``lat_grid`` / ``lon_grid``."""
    pivot = sub.pivot(index="lat", columns="lon", values=col)
    pivot = pivot.reindex(index=lat_grid, columns=lon_grid)
    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise ValueError(
            f"Incomplete lat/lon grid for column {col!r}: {missing} missing cells"
        )
    return pivot.to_numpy(dtype=float)


def spatial_slices_from_dataframe(
    df: pd.DataFrame,
    ages_bp: Sequence[float],
    *,
    comp_labels: Optional[List[str]] = None,
) -> List[SpatialSlice]:
    """
    Reconstruct :class:`SpatialSlice` objects from ``spatial_predictions.csv``.

    Inverse of :func:`spatial_to_dataframe` for gridded export columns.
    """
    required = {
        "age_bp",
        "lat",
        "lon",
        "total_mean",
        "total_std",
        "gp_residual_mean",
        "gp_residual_std",
        "gia_mean",
        "gia_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"spatial CSV missing columns: {sorted(missing)}")

    if comp_labels is None:
        comp_labels = _infer_spatial_component_labels(df)

    slices: List[SpatialSlice] = []
    for age in ages_bp:
        sub = _spatial_age_subset(df, float(age))
        lat_grid = np.sort(sub["lat"].unique())
        lon_grid = np.sort(sub["lon"].unique())
        lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)

        gp_std = _spatial_field_grid(sub, lat_grid, lon_grid, "gp_residual_std")
        gia_std = _spatial_field_grid(sub, lat_grid, lon_grid, "gia_std")

        components: Dict[str, Dict[str, np.ndarray]] = {}
        for label in comp_labels:
            mean_col = f"{label}_mean"
            std_col = f"{label}_std"
            if mean_col not in sub.columns or std_col not in sub.columns:
                continue
            components[label] = {
                "mean": _spatial_field_grid(sub, lat_grid, lon_grid, mean_col),
                "std": _spatial_field_grid(sub, lat_grid, lon_grid, std_col),
            }

        slices.append(
            SpatialSlice(
                age_bp=float(age),
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                lat_mat=lat_mat,
                lon_mat=lon_mat,
                total_mean=_spatial_field_grid(sub, lat_grid, lon_grid, "total_mean"),
                total_std=_spatial_field_grid(sub, lat_grid, lon_grid, "total_std"),
                gp_mean=_spatial_field_grid(sub, lat_grid, lon_grid, "gp_residual_mean"),
                gp_var=np.square(gp_std),
                gia_mean=_spatial_field_grid(sub, lat_grid, lon_grid, "gia_mean"),
                gia_var=np.square(gia_std),
                components=components,
            )
        )

    return slices


def rates_to_dataframe(
    rate_result: SpatialRateResult,
    comp_labels: List[str],
) -> pd.DataFrame:
    """
    Build spatial rates DataFrame from a SpatialRateResult.

    Returns
    -------
    pd.DataFrame with one row per (interval, lat, lon) grid point.
    """
    rows = []
    n_intervals = len(rate_result.rate_midpoints)
    n_grid = len(rate_result.lat_flat)

    # Custom windows from spatial_rate_interval store [younger, mid, older]
    # (len = n_intervals + 2); consecutive-age grids store endpoints only.
    ages = rate_result.spatial_ages
    use_window_ends = len(ages) > n_intervals + 1

    for i in range(n_intervals):
        if use_window_ends:
            younger = int(ages[0])
            older = int(ages[-1])
        else:
            younger = int(ages[i])
            older = int(ages[i + 1])
        midpt = rate_result.rate_midpoints[i]

        for k in range(n_grid):
            row = {
                "interval_start_bp": younger,
                "interval_end_bp": older,
                "midpoint_bp": midpt,
                "lat": rate_result.lat_flat[k],
                "lon": rate_result.lon_flat[k],
                "total_rate_mm_yr": -rate_result.rate_total[i, k] * 1000,
                "total_rate_sd_mm_yr": rate_result.sd_total[i, k] * 1000,
                "total_nogia_rate_mm_yr": -rate_result.rate_gp[i, k] * 1000,
                "total_nogia_rate_sd_mm_yr": rate_result.sd_gp[i, k] * 1000,
                "gia_rate_mm_yr": -rate_result.rate_gia[i, k] * 1000,
                "gia_rate_sd_mm_yr": rate_result.sd_gia[i, k] * 1000,
            }
            for label in comp_labels:
                row[f"{label}_rate_mm_yr"] = -rate_result.rate_comp[label][i, k] * 1000
                row[f"{label}_rate_sd_mm_yr"] = rate_result.sd_comp[label][i, k] * 1000
            rows.append(row)

    return pd.DataFrame(rows)


def regional_mean_rates_to_dataframe(
    results: Sequence[RegionalMeanRateResult],
    *,
    component_label: str = "regional_linear",
) -> pd.DataFrame:
    """Export regional-mean rates with explicit one- and two-sigma uncertainty."""
    rows = []
    for result in results:
        if component_label not in result.component_rates:
            raise KeyError(
                f"Regional result missing component {component_label!r}"
            )
        component_rate = result.component_rates[component_label]
        component_sd = result.component_rate_sds[component_label]
        combined_rate = result.combined_rate
        combined_sd = result.combined_rate_sd
        if combined_rate is None or combined_sd is None:
            raise ValueError("Regional result is missing the combined rate")

        row = {
            "region": result.region_num,
            "region_name": result.region_name,
            "n_pixels": result.n_pixels,
            "mask_mode": result.mask_mode,
            "spatial_weighting": result.weighting,
            "interval_start_bp": result.younger_bp,
            "interval_end_bp": result.older_bp,
            "midpoint_bp": result.midpoint_bp,
            "gia_rate_mm_yr": -result.gia_rate * 1000.0,
            "gia_rate_sd_of_regional_mean_mm_yr": result.gia_rate_sd
            * 1000.0,
            "gia_rate_2sigma_of_regional_mean_mm_yr": result.gia_rate_sd
            * 2000.0,
            f"{component_label}_rate_mm_yr": -component_rate * 1000.0,
            f"{component_label}_rate_sd_of_regional_mean_mm_yr": component_sd
            * 1000.0,
            f"{component_label}_rate_2sigma_of_regional_mean_mm_yr": component_sd
            * 2000.0,
            "gia_plus_regional_linear_rate_mm_yr": -combined_rate * 1000.0,
            "gia_plus_regional_linear_rate_sd_of_regional_mean_mm_yr": combined_sd
            * 1000.0,
            "gia_plus_regional_linear_rate_2sigma_of_regional_mean_mm_yr": combined_sd
            * 2000.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


_SPATIAL_BASE_RATE_COLS = frozenset(
    {
        "total_rate_mm_yr",
        "total_rate_sd_mm_yr",
        "total_nogia_rate_mm_yr",
        "total_nogia_rate_sd_mm_yr",
        "gia_rate_mm_yr",
        "gia_rate_sd_mm_yr",
    }
)


def _infer_spatial_rate_component_labels(df: pd.DataFrame) -> List[str]:
    """Kernel labels from ``{label}_rate_mm_yr`` / ``{label}_rate_sd_mm_yr`` pairs."""
    labels: List[str] = []
    for col in df.columns:
        if not col.endswith("_rate_mm_yr") or col in _SPATIAL_BASE_RATE_COLS:
            continue
        label = col[: -len("_rate_mm_yr")]
        sd_col = f"{label}_rate_sd_mm_yr"
        if sd_col in df.columns:
            labels.append(label)
    return labels


def _spatial_interval_subset(
    df: pd.DataFrame, younger_bp: float, older_bp: float
) -> pd.DataFrame:
    """Rows for one rate interval (tolerant float match on endpoints)."""
    y0 = df["interval_start_bp"].to_numpy(dtype=float)
    y1 = df["interval_end_bp"].to_numpy(dtype=float)
    mask = np.isclose(y0, float(younger_bp), rtol=0.0, atol=0.5) & np.isclose(
        y1, float(older_bp), rtol=0.0, atol=0.5
    )
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(
            f"No spatial rate rows for interval {younger_bp:g}--{older_bp:g} BP"
        )
    return sub


def spatial_rates_from_dataframe(
    df: pd.DataFrame,
    *,
    comp_labels: Optional[List[str]] = None,
) -> SpatialRateResult:
    """
    Reconstruct :class:`SpatialRateResult` from ``spatial_rates.csv``.

    Inverse of :func:`rates_to_dataframe` for gridded export columns.
    """
    required = {
        "interval_start_bp",
        "interval_end_bp",
        "midpoint_bp",
        "lat",
        "lon",
        "total_rate_mm_yr",
        "total_rate_sd_mm_yr",
        "total_nogia_rate_mm_yr",
        "total_nogia_rate_sd_mm_yr",
        "gia_rate_mm_yr",
        "gia_rate_sd_mm_yr",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"spatial rates CSV missing columns: {sorted(missing)}")

    if comp_labels is None:
        comp_labels = _infer_spatial_rate_component_labels(df)

    intervals = (
        df[["interval_start_bp", "interval_end_bp"]]
        .drop_duplicates()
        .sort_values(["interval_start_bp", "interval_end_bp"])
    )
    n_intervals = len(intervals)
    if n_intervals == 0:
        raise ValueError("spatial rates CSV contains no intervals")

    first = _spatial_interval_subset(
        df,
        float(intervals.iloc[0]["interval_start_bp"]),
        float(intervals.iloc[0]["interval_end_bp"]),
    )
    lat_grid = np.sort(first["lat"].unique())
    lon_grid = np.sort(first["lon"].unique())
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size

    spatial_ages = np.zeros(n_intervals + 1, dtype=float)
    rate_midpoints = np.zeros(n_intervals, dtype=float)
    rate_total = np.zeros((n_intervals, n_grid))
    sd_total = np.zeros((n_intervals, n_grid))
    rate_gp = np.zeros((n_intervals, n_grid))
    sd_gp = np.zeros((n_intervals, n_grid))
    rate_gia = np.zeros((n_intervals, n_grid))
    sd_gia = np.zeros((n_intervals, n_grid))
    rate_comp: Dict[str, np.ndarray] = {
        label: np.zeros((n_intervals, n_grid)) for label in comp_labels
    }
    sd_comp: Dict[str, np.ndarray] = {
        label: np.zeros((n_intervals, n_grid)) for label in comp_labels
    }

    for i in range(n_intervals):
        younger = float(intervals.iloc[i]["interval_start_bp"])
        older = float(intervals.iloc[i]["interval_end_bp"])
        sub = _spatial_interval_subset(df, younger, older)
        spatial_ages[i] = younger
        if i == n_intervals - 1:
            spatial_ages[i + 1] = older
        mid = sub["midpoint_bp"].iloc[0]
        rate_midpoints[i] = float(mid)

        rate_total[i] = (
            -_spatial_field_grid(sub, lat_grid, lon_grid, "total_rate_mm_yr") / 1000.0
        ).ravel()
        sd_total[i] = (
            _spatial_field_grid(sub, lat_grid, lon_grid, "total_rate_sd_mm_yr") / 1000.0
        ).ravel()
        rate_gp[i] = (
            -_spatial_field_grid(sub, lat_grid, lon_grid, "total_nogia_rate_mm_yr")
            / 1000.0
        ).ravel()
        sd_gp[i] = (
            _spatial_field_grid(sub, lat_grid, lon_grid, "total_nogia_rate_sd_mm_yr")
            / 1000.0
        ).ravel()
        rate_gia[i] = (
            -_spatial_field_grid(sub, lat_grid, lon_grid, "gia_rate_mm_yr") / 1000.0
        ).ravel()
        sd_gia[i] = (
            _spatial_field_grid(sub, lat_grid, lon_grid, "gia_rate_sd_mm_yr") / 1000.0
        ).ravel()

        for label in comp_labels:
            rate_col = f"{label}_rate_mm_yr"
            sd_col = f"{label}_rate_sd_mm_yr"
            if rate_col not in sub.columns or sd_col not in sub.columns:
                continue
            rate_comp[label][i] = (
                -_spatial_field_grid(sub, lat_grid, lon_grid, rate_col) / 1000.0
            ).ravel()
            sd_comp[label][i] = (
                _spatial_field_grid(sub, lat_grid, lon_grid, sd_col) / 1000.0
            ).ravel()

    return SpatialRateResult(
        spatial_ages=spatial_ages,
        rate_midpoints=rate_midpoints,
        lat_flat=lat_mat.ravel(),
        lon_flat=lon_mat.ravel(),
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=rate_total,
        sd_total=sd_total,
        rate_gp=rate_gp,
        sd_gp=sd_gp,
        rate_gia=rate_gia,
        sd_gia=sd_gia,
        rate_comp=rate_comp,
        sd_comp=sd_comp,
    )
