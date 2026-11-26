"""Boundary proximity reporting for dashboard results."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd

from stgp_dashboard.paths import ensure_import_paths

ensure_import_paths()

from stgp_pipeline.sensitivity.workflow_common import (  # noqa: E402
    boundary_proximity_dataframe,
    format_boundary_proximity_report,
)


def load_boundary_report(scenario_dir: str) -> pd.DataFrame:
    from pathlib import Path

    path = Path(scenario_dir) / "boundary_proximity.csv"
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def boundary_summary_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        hit = bool(row.get("boundary_hit", False))
        pct = float(row.get("proximity_pct", float("nan")))
        status = "ok"
        if hit:
            status = "hit"
        elif pct <= 15 or pct >= 85:
            status = "warn"
        rows.append(
            {
                "parameter": row.get("parameter", ""),
                "value": row.get("value", ""),
                "proximity_pct": pct,
                "boundary": row.get("boundary_side", ""),
                "status": status,
                "suggestion": _suggestion_for_row(row),
            }
        )
    return rows


def _suggestion_for_row(row: pd.Series) -> str:
    if not bool(row.get("boundary_hit", False)):
        return ""
    side = str(row.get("boundary_side", ""))
    param = str(row.get("parameter", ""))
    if side == "upper":
        return f"Consider widening the upper prior bound for {param}."
    if side == "lower":
        return f"Consider lowering the lower prior bound for {param}."
    return f"Review prior range for {param}."


def format_report_text(df_params: pd.DataFrame, cfg: Mapping[str, Any]) -> str:
    df_boundary = boundary_proximity_dataframe(df_params, cfg)
    return format_boundary_proximity_report(df_boundary)


def suggest_widen_factor(boundary_rows: List[Dict[str, Any]]) -> float:
    hits = [r for r in boundary_rows if r.get("status") == "hit"]
    if len(hits) >= 3:
        return 2.0
    if hits:
        return 1.5
    return 1.0
