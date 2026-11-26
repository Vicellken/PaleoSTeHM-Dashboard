"""Lightweight CSV preview without full GP training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "Level",
    "Region",
    "Latitude",
    "Longitude",
    "Age",
    "RSL",
}


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace(".", "")


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {_normalize_col(c): c for c in columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def _column_is_numeric_id(df: pd.DataFrame, col: str) -> bool:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return False
    return bool(np.all(np.isfinite(values)))


def _resolve_region_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Return (id_column, name_column) matching PSTHM/load.py conventions."""
    id_candidates = ["Region.1", "region.1", "Region_ID", "region_id", "RegionID"]
    name_candidates = ["Region", "Region.2", "region.2", "RegionName", "region_name", "Region name"]

    id_col = _find_column(df.columns, id_candidates)
    if id_col is None:
        region_col = _find_column(df.columns, ["Region", "region"])
        if region_col is not None and _column_is_numeric_id(df, region_col):
            id_col = region_col

    name_col = _find_column(df.columns, name_candidates)
    if name_col == id_col:
        name_col = _find_column(
            df.columns,
            [c for c in name_candidates if _normalize_col(c) != _normalize_col(id_col or "")],
        )

    return id_col, name_col


def preview_rsl_csv(csv_path: str | Path) -> Dict[str, Any]:
    path = Path(csv_path).expanduser().resolve()
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "warnings": [],
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append(f"File not found: {path}")
        return result

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Could not read CSV: {exc}")
        return result

    result["n_rows"] = int(len(df))
    result["columns"] = list(df.columns)

    level_col = _find_column(df.columns, ["Level", "level"])
    lat_col = _find_column(df.columns, ["Latitude", "lat"])
    lon_col = _find_column(df.columns, ["Longitude", "lon", "long"])
    age_col = _find_column(df.columns, ["Age", "age", "Age (cal yr BP)"])
    region_id_col, region_name_col = _resolve_region_columns(df)

    missing = []
    for label, col in [
        ("Latitude", lat_col),
        ("Longitude", lon_col),
        ("Age", age_col),
    ]:
        if col is None:
            missing.append(label)
    if missing:
        result["errors"].append(f"Missing required columns: {', '.join(missing)}")
        return result

    ages = pd.to_numeric(df[age_col], errors="coerce")
    lats = pd.to_numeric(df[lat_col], errors="coerce")
    lons = pd.to_numeric(df[lon_col], errors="coerce")

    if level_col is not None:
        levels = pd.to_numeric(df[level_col], errors="coerce")
        level_counts = levels.value_counts(dropna=True).sort_index()
        result["level_counts"] = {str(int(k)): int(v) for k, v in level_counts.items()}
    else:
        result["level_counts"] = {}
        result["warnings"].append(
            "No 'Level' column found — all SLIP rows will be included without level filtering."
        )

    finite_mask = np.isfinite(ages) & np.isfinite(lats) & np.isfinite(lons)
    result["n_finite_coords"] = int(finite_mask.sum())
    if finite_mask.any():
        result["age_min_bp"] = float(np.nanmin(ages[finite_mask]))
        result["age_max_bp"] = float(np.nanmax(ages[finite_mask]))
        result["lat_min"] = float(np.nanmin(lats[finite_mask]))
        result["lat_max"] = float(np.nanmax(lats[finite_mask]))
        result["lon_min"] = float(np.nanmin(lons[finite_mask]))
        result["lon_max"] = float(np.nanmax(lons[finite_mask]))
    else:
        result["warnings"].append("No rows with finite age/lat/lon.")

    if region_id_col is not None:
        region_ids = pd.to_numeric(df[region_id_col], errors="coerce").dropna()
        unique_ids = sorted(region_ids.unique())
        result["n_regions"] = int(len(unique_ids))

        region_options = []
        seen_ids: set[int] = set()
        for raw_id in unique_ids:
            try:
                rid = int(float(raw_id))
            except (TypeError, ValueError):
                continue
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            label = f"R{rid}"
            name = label
            if region_name_col is not None:
                names = (
                    df.loc[pd.to_numeric(df[region_id_col], errors="coerce") == raw_id, region_name_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if len(names):
                    name = str(names[0])
                    label = f"R{rid}: {name}"

            region_options.append({"id": rid, "label": label, "name": name})

        result["region_options"] = region_options
        result["regions"] = [opt["label"] for opt in region_options[:50]]
    else:
        result["warnings"].append(
            "Region ID column not found (expected Region.1 or numeric Region); "
            "region merge list unavailable."
        )

    if result["n_finite_coords"] < 5:
        result["warnings"].append("Very few usable rows for GP training.")

    return result
