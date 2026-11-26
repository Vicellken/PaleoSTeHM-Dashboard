"""Execute STGP analysis from dashboard session state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stgp_dashboard.paths import ensure_import_paths, repo_root
from stgp_dashboard.services.config_builder import build_config, save_config
from stgp_dashboard.services.progress import RunProgress, make_progress_callbacks

ensure_import_paths()

from stgp_pipeline.run import run_stgp_analysis  # noqa: E402


def execute_run(
    state: Dict[str, Any],
    progress: Optional[RunProgress] = None,
) -> Dict[str, Any]:
    cfg = build_config(state)
    output_dir = Path(cfg["paths"]["output_dir"])
    config_path = save_config(cfg, output_dir / "dashboard_config.yaml")

    progress = progress or RunProgress()
    on_iteration, on_stage = make_progress_callbacks(progress)

    try:
        metrics = run_stgp_analysis(
            cfg,
            output_dir,
            base_dir=repo_root(),
            progress_callback=on_iteration,
            stage_callback=on_stage,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error": str(exc),
            "config_path": str(config_path),
            "progress": progress,
        }

    scenario_dir = Path(metrics.get("scenario_dir", output_dir / "run"))
    boundary_path = scenario_dir / "boundary_proximity.csv"
    track_path = scenario_dir / "optimization_track.csv"
    log_path = output_dir / "run.log"

    boundary_df = None
    track_df = None
    if boundary_path.is_file():
        boundary_df = pd.read_csv(boundary_path)
    if track_path.is_file():
        track_df = pd.read_csv(track_path)

    run_log = ""
    if log_path.is_file():
        run_log = log_path.read_text(encoding="utf-8", errors="replace")

    return {
        "status": metrics.get("status", "success"),
        "metrics": metrics,
        "config_path": str(config_path),
        "scenario_dir": str(scenario_dir),
        "boundary_df": boundary_df,
        "track_df": track_df,
        "run_log": run_log,
        "progress": progress,
        "cfg": cfg,
    }
