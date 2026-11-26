"""Sensitivity analytical plots for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from stgp_dashboard.paths import ensure_import_paths

ensure_import_paths()

from stgp_pipeline.sensitivity.summary import (  # noqa: E402
    build_params_bounds_nill_long,
    build_workflow_summary,
)


def generate_sensitivity_analytics(output_root: str | Path) -> Dict[str, Any]:
    """Run summary aggregation and write heatmaps / markdown reports."""
    root = Path(output_root).resolve()
    return build_workflow_summary(root)


def list_display_images(summary_outputs: Mapping[str, Any]) -> List[Path]:
    """Return PNG paths suitable for inline Streamlit display."""
    images: List[Path] = []
    seen: set[str] = set()
    for key, value in summary_outputs.items():
        if "png" not in str(key).lower():
            continue
        p = Path(value)
        text = str(p)
        if p.is_file() and text not in seen:
            images.append(p)
            seen.add(text)
    return images


def build_bounds_table(output_root: str | Path) -> Any:
    """Load long bounds table for optional dataframe display."""
    import pandas as pd

    root = Path(output_root).resolve()
    wide_path = root / "summary_params_bounds_nill.csv"
    if wide_path.is_file():
        return pd.read_csv(wide_path)
    boundary_path = root / "summary_boundary_proximity.csv"
    metrics_path = root / "summary_metrics.csv"
    if not boundary_path.is_file():
        return None
    df_boundary = pd.read_csv(boundary_path)
    df_metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    scenario_order = list(df_boundary["scenario"].drop_duplicates())
    return build_params_bounds_nill_long(df_boundary, df_metrics, scenario_order)
