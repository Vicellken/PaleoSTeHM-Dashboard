"""
Point extracts and subregion spatial ranges from STGP spatial-rate grids.

Post-processing only (no training). Typical use: snap city / tide-gauge
coordinates to a spatial-rate CSV within a great-circle radius, and summarize
per-subregion pixel ranges over the whole analysis domain via nearest-SLIP
region assignment.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_RADIUS_KM = 50.0
_EARTH_RADIUS_KM = 6371.0


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Great-circle distance in km from (lat2, lon2) to each (lat1, lon1) point."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


GIA_RATE_COL = "gia_rate_mm_yr"
GIA_SD_COL = "gia_rate_sd_mm_yr"
RL_RATE_COL = "regional_linear_rate_mm_yr"
RL_SD_COL = "regional_linear_rate_sd_mm_yr"
COMBINED_RATE_COL = "gia_plus_regional_linear_rate_mm_yr"
COMBINED_SD_COL = "gia_plus_regional_linear_rate_sd_mm_yr"

THREE_TERM_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("gia", GIA_RATE_COL, GIA_SD_COL),
    ("regional_linear", RL_RATE_COL, RL_SD_COL),
    ("gia_plus_regional_linear", COMBINED_RATE_COL, COMBINED_SD_COL),
)


def add_gia_plus_regional_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GIA + regional_linear mean and RSS SD columns (mm/yr).

    Mean is the sum of component means. SD assumes independent GIA (ensemble)
    and regional_linear (GP) uncertainties: ``sqrt(sd_gia**2 + sd_rl**2)``.
    """
    out = df.copy()
    for col in (GIA_RATE_COL, GIA_SD_COL, RL_RATE_COL, RL_SD_COL):
        if col not in out.columns:
            raise KeyError(f"rate DataFrame missing required column {col!r}")
    out[COMBINED_RATE_COL] = out[GIA_RATE_COL] + out[RL_RATE_COL]
    out[COMBINED_SD_COL] = np.sqrt(
        np.asarray(out[GIA_SD_COL], dtype=float) ** 2
        + np.asarray(out[RL_SD_COL], dtype=float) ** 2
    )
    return out


def _nearest_grid_index(
    lat: float,
    lon: float,
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
) -> Tuple[int, float]:
    """Return (index, distance_km) of the nearest grid node."""
    dist = _haversine_km(lat_flat, lon_flat, float(lat), float(lon) % 360.0)
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def filter_sites_near_grid(
    sites_df: pd.DataFrame,
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    radius_km: float = DEFAULT_RADIUS_KM,
    *,
    name_col: str = "Name",
    lat_col: str = "Lat",
    lon_col: str = "Lon",
) -> pd.DataFrame:
    """
    Keep sites whose nearest grid node is within ``radius_km``.

    Accepts columns ``Name``/``Lat``/``Lon`` (INSAR) or ``site``/``lat``/``lon``.
    Returns one row per input site with ``status`` in {``kept``, ``dropped``},
    plus snap metadata when a nearest node exists.
    """
    lat_flat = np.asarray(lat_flat, dtype=float).ravel()
    lon_flat = np.asarray(lon_flat, dtype=float).ravel()
    if lat_flat.size == 0:
        raise ValueError("lat_flat / lon_flat must be non-empty")

    # Allow either INSAR-style or already-normalized column names.
    colmap = {}
    if name_col in sites_df.columns:
        colmap["site"] = name_col
    elif "site" in sites_df.columns:
        colmap["site"] = "site"
    else:
        raise KeyError("sites_df needs a Name or site column")

    if lat_col in sites_df.columns:
        colmap["lat"] = lat_col
    elif "lat" in sites_df.columns:
        colmap["lat"] = "lat"
    else:
        raise KeyError("sites_df needs a Lat or lat column")

    if lon_col in sites_df.columns:
        colmap["lon"] = lon_col
    elif "lon" in sites_df.columns:
        colmap["lon"] = "lon"
    else:
        raise KeyError("sites_df needs a Lon or lon column")

    rows = []
    for _, row in sites_df.iterrows():
        site = str(row[colmap["site"]]).strip()
        lat = float(row[colmap["lat"]])
        lon = float(row[colmap["lon"]])
        idx, dist_km = _nearest_grid_index(lat, lon, lat_flat, lon_flat)
        kept = dist_km <= float(radius_km)
        rows.append(
            {
                "site": site,
                "lat": lat,
                "lon": lon,
                "status": "kept" if kept else "dropped",
                "grid_lat": float(lat_flat[idx]) if kept else np.nan,
                "grid_lon": float(lon_flat[idx]) if kept else np.nan,
                "distance_km": dist_km,
                "grid_index": idx if kept else -1,
            }
        )
    return pd.DataFrame(rows)


def extract_site_rates(
    rate_df: pd.DataFrame,
    filter_df: pd.DataFrame,
    *,
    rate_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Join kept sites to the spatial-rate grid at snapped cells.

    ``rate_df`` must contain ``lat``/``lon`` (and typically one rate interval).
    When multiple intervals are present, all matching snapped rows are returned.
    """
    if rate_cols is None:
        rate_cols = [
            GIA_RATE_COL,
            GIA_SD_COL,
            RL_RATE_COL,
            RL_SD_COL,
            COMBINED_RATE_COL,
            COMBINED_SD_COL,
            "total_rate_mm_yr",
            "total_rate_sd_mm_yr",
            "interval_start_bp",
            "interval_end_bp",
            "midpoint_bp",
        ]

    kept = filter_df.loc[filter_df["status"] == "kept"].copy()
    if kept.empty:
        return pd.DataFrame()

    rate = rate_df.copy()
    if COMBINED_RATE_COL not in rate.columns:
        rate = add_gia_plus_regional_linear(rate)

    # Index rate rows by rounded lat/lon for exact snap match to grid nodes.
    rate["_key"] = list(
        zip(
            np.round(rate["lat"].to_numpy(dtype=float), 10),
            np.round(rate["lon"].to_numpy(dtype=float), 10),
        )
    )
    key_to_rows = rate.groupby("_key", sort=False)

    out_rows = []
    for _, site in kept.iterrows():
        key = (
            round(float(site["grid_lat"]), 10),
            round(float(site["grid_lon"]), 10),
        )
        if key not in key_to_rows.groups:
            # Fallback: nearest row in rate_df (handles float noise).
            dist = _haversine_km(
                rate["lat"].to_numpy(dtype=float),
                rate["lon"].to_numpy(dtype=float),
                float(site["grid_lat"]),
                float(site["grid_lon"]),
            )
            match = rate.iloc[[int(np.argmin(dist))]]
        else:
            match = rate.loc[key_to_rows.groups[key]]

        for _, rrow in match.iterrows():
            row = {
                "site": site["site"],
                "site_lat": float(site["lat"]),
                "site_lon": float(site["lon"]),
                "grid_lat": float(site["grid_lat"]),
                "grid_lon": float(site["grid_lon"]),
                "distance_km": float(site["distance_km"]),
            }
            for col in rate_cols:
                if col in rrow.index:
                    row[col] = rrow[col]
            out_rows.append(row)

    return pd.DataFrame(out_rows)


def assign_grid_to_nearest_slip_region(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    slip_lat: np.ndarray,
    slip_lon: np.ndarray,
    slip_region: np.ndarray,
) -> np.ndarray:
    """
    Assign each grid pixel to the ``Region.1`` of its nearest SLIP.

    Partitions the full analysis grid into non-overlapping subregion domains
    (Voronoi / nearest-neighbour in great-circle distance).
    """
    grid_lat = np.asarray(grid_lat, dtype=float).ravel()
    grid_lon = np.asarray(grid_lon, dtype=float).ravel()
    slip_lat = np.asarray(slip_lat, dtype=float).ravel()
    slip_lon = np.asarray(slip_lon, dtype=float).ravel() % 360.0
    slip_region = np.asarray(slip_region, dtype=int).ravel()
    if slip_lat.size == 0:
        return np.full(grid_lat.shape, -1, dtype=int)
    if not (slip_lat.size == slip_lon.size == slip_region.size):
        raise ValueError("slip_lat, slip_lon, slip_region must have the same length")

    assigned = np.empty(grid_lat.shape, dtype=int)
    for i, (glat, glon) in enumerate(zip(grid_lat, grid_lon)):
        dist = _haversine_km(slip_lat, slip_lon, float(glat), float(glon) % 360.0)
        assigned[i] = int(slip_region[int(np.argmin(dist))])
    return assigned


def summarize_region_rates(
    rate_df: pd.DataFrame,
    rsl_data: Any,
    *,
    fields: Sequence[Tuple[str, str, str]] = THREE_TERM_FIELDS,
) -> pd.DataFrame:
    """
    Per-subregion mean/min/max of rate fields over the whole-domain region mask.

    For each rate interval, every grid pixel is assigned to the region of its
    nearest SLIP. Statistics are then computed on all pixels belonging to each
    ``Region.1`` (no 50 km buffer). Report mean/min/max of the predicted mean
    field and min/max of the predictive SD for each field in ``fields``
    (default: GIA, regional_linear, GIA+regional_linear).
    """
    rate = rate_df.copy()
    if COMBINED_RATE_COL not in rate.columns:
        rate = add_gia_plus_regional_linear(rate)

    for _, rate_col, sd_col in fields:
        if rate_col not in rate.columns or sd_col not in rate.columns:
            raise KeyError(f"rate DataFrame missing {rate_col!r} / {sd_col!r}")

    interval_cols = ["interval_start_bp", "interval_end_bp", "midpoint_bp"]
    has_intervals = all(c in rate.columns for c in interval_cols)
    if has_intervals:
        intervals = (
            rate[interval_cols]
            .drop_duplicates()
            .sort_values(["interval_start_bp", "interval_end_bp"])
            .reset_index(drop=True)
        )
    else:
        intervals = pd.DataFrame(
            [{"interval_start_bp": np.nan, "interval_end_bp": np.nan, "midpoint_bp": np.nan}]
        )

    slip_lat = np.asarray(rsl_data.rsl_lat, dtype=float)
    slip_lon = np.asarray(rsl_data.rsl_lon, dtype=float)
    slip_region = np.asarray(rsl_data.rsl_region, dtype=int)

    rows: List[dict] = []
    for _, iv in intervals.iterrows():
        if has_intervals:
            sub = rate.loc[
                np.isclose(
                    rate["interval_start_bp"].to_numpy(dtype=float),
                    float(iv["interval_start_bp"]),
                    rtol=0.0,
                    atol=0.5,
                )
                & np.isclose(
                    rate["interval_end_bp"].to_numpy(dtype=float),
                    float(iv["interval_end_bp"]),
                    rtol=0.0,
                    atol=0.5,
                )
            ].copy()
        else:
            sub = rate.copy()

        if sub.empty:
            continue

        assigned = assign_grid_to_nearest_slip_region(
            sub["lat"].to_numpy(dtype=float),
            sub["lon"].to_numpy(dtype=float),
            slip_lat,
            slip_lon,
            slip_region,
        )
        sub["_assigned_region"] = assigned

        for rn in rsl_data.unique_regions:
            rn = int(rn)
            region_name = rsl_data.region_num_to_name.get(rn, f"Region {rn}")
            pix_mask = sub["_assigned_region"].to_numpy(dtype=int) == rn
            n_pix = int(np.count_nonzero(pix_mask))
            row = {
                "region": rn,
                "region_name": region_name,
                "n_pixels": n_pix,
                "mask_mode": "nearest_slip",
                "interval_start_bp": iv["interval_start_bp"],
                "interval_end_bp": iv["interval_end_bp"],
                "midpoint_bp": iv["midpoint_bp"],
            }
            if n_pix == 0:
                for prefix, rate_col, sd_col in fields:
                    row[f"{prefix}_rate_mm_yr_mean"] = np.nan
                    row[f"{prefix}_rate_mm_yr_min"] = np.nan
                    row[f"{prefix}_rate_mm_yr_max"] = np.nan
                    row[f"{prefix}_rate_sd_mm_yr_min"] = np.nan
                    row[f"{prefix}_rate_sd_mm_yr_max"] = np.nan
            else:
                for prefix, rate_col, sd_col in fields:
                    vals = sub.loc[pix_mask, rate_col].to_numpy(dtype=float)
                    sds = sub.loc[pix_mask, sd_col].to_numpy(dtype=float)
                    row[f"{prefix}_rate_mm_yr_mean"] = float(np.mean(vals))
                    row[f"{prefix}_rate_mm_yr_min"] = float(np.min(vals))
                    row[f"{prefix}_rate_mm_yr_max"] = float(np.max(vals))
                    row[f"{prefix}_rate_sd_mm_yr_min"] = float(np.min(sds))
                    row[f"{prefix}_rate_sd_mm_yr_max"] = float(np.max(sds))
            rows.append(row)

    return pd.DataFrame(rows)
