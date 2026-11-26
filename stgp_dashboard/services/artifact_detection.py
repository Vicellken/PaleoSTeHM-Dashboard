"""Detect reusable post-processing artifacts on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _newest_mtime(paths: Sequence[Path]) -> Optional[float]:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return None
    return max(_mtime(p) for p in existing)


def _oldest_mtime(paths: Sequence[Path]) -> Optional[float]:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return None
    return min(_mtime(p) for p in existing)


def _first_glob(scenario_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(scenario_dir.glob(pattern))
    return matches[0] if matches else None


def _resolve_checkpoint(
    scenario_dir: Path,
    checkpoint_path: Optional[str | Path],
) -> Optional[Path]:
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser()
        if ckpt.is_file():
            return ckpt.resolve()
    default = scenario_dir / "trained_param_store.pt"
    if default.is_file():
        return default.resolve()
    return None


def detect_postprocess_artifacts(
    scenario_dir: str | Path,
    *,
    checkpoint_path: Optional[str | Path] = None,
    workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check whether temporal/spatial prediction exports can be reused.

    GIA workflows require ``spatial_predictions.csv`` and ``spatial_rates.csv``.
    No-GIA (16ka) requires ``rsl_*.nc`` and ``rate_*.nc``.

    Artifacts are treated as stale when a checkpoint exists and is newer than
    the oldest required artifact.
    """
    scenario = Path(scenario_dir).expanduser()
    result: Dict[str, Any] = {
        "reusable": False,
        "kind": None,
        "paths": {},
        "stale_reason": None,
        "note": None,
        "missing": [],
    }
    if not scenario.is_dir():
        result["stale_reason"] = "Scenario directory not found."
        result["missing"] = ["scenario_dir"]
        return result

    workflow = (workflow_id or "").strip().lower()
    spatial_pred = scenario / "spatial_predictions.csv"
    spatial_rates = scenario / "spatial_rates.csv"
    temporal_csv = scenario / "temporal_decomposition.csv"
    temporal_rates_csv = scenario / "temporal_decomposition_rates.csv"
    rsl_nc = _first_glob(scenario, "rsl_*.nc")
    rate_nc = _first_glob(scenario, "rate_*.nc")

    prefer_gia = workflow in {"4ka", "3d-gia", "regional-gia"} or (
        not workflow and spatial_pred.is_file() and spatial_rates.is_file()
    )
    prefer_16ka = workflow == "16ka" or (
        not prefer_gia and rsl_nc is not None and rate_nc is not None
    )

    required: List[Path] = []
    paths: Dict[str, str] = {}

    if prefer_gia:
        result["kind"] = "gia"
        required = [spatial_pred, spatial_rates]
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            result["missing"] = missing
            result["stale_reason"] = (
                "Missing spatial prediction exports: " + ", ".join(missing)
            )
            return result
        paths["spatial_csv_path"] = str(spatial_pred.resolve())
        paths["spatial_rates_csv_path"] = str(spatial_rates.resolve())
        if temporal_csv.is_file():
            paths["temporal_csv_path"] = str(temporal_csv.resolve())
        if temporal_rates_csv.is_file():
            paths["temporal_rates_csv_path"] = str(temporal_rates_csv.resolve())
        if rsl_nc is not None:
            paths["rsl_nc_path"] = str(rsl_nc.resolve())
        if rate_nc is not None:
            paths["rate_nc_path"] = str(rate_nc.resolve())
        # Spatial CSVs are enough for the spatial gallery; note if temporal
        # exports are absent so the user can recompute if needed.
        if not temporal_csv.is_file() and rsl_nc is None:
            result["note"] = (
                "Spatial CSVs found, but temporal CSV/NetCDF are missing. "
                "Recompute if you need temporal figures."
            )
    elif prefer_16ka:
        result["kind"] = "16ka"
        if rsl_nc is None or rate_nc is None:
            missing = []
            if rsl_nc is None:
                missing.append("rsl_*.nc")
            if rate_nc is None:
                missing.append("rate_*.nc")
            result["missing"] = missing
            result["stale_reason"] = (
                "Missing temporal NetCDF exports: " + ", ".join(missing)
            )
            return result
        required = [rsl_nc, rate_nc]
        paths["rsl_nc_path"] = str(rsl_nc.resolve())
        paths["rate_nc_path"] = str(rate_nc.resolve())
    else:
        result["missing"] = [
            "spatial_predictions.csv / spatial_rates.csv",
            "or rsl_*.nc / rate_*.nc",
        ]
        result["stale_reason"] = "No post-processing prediction exports found."
        return result

    result["paths"] = paths

    ckpt = _resolve_checkpoint(scenario, checkpoint_path)
    if ckpt is not None:
        paths["param_store_path"] = str(ckpt)
        artifact_oldest = _oldest_mtime(required)
        if artifact_oldest is not None and _mtime(ckpt) > artifact_oldest:
            result["stale_reason"] = (
                "Checkpoint is newer than existing prediction exports; "
                "recompute to refresh outputs."
            )
            return result

    result["reusable"] = True
    result["stale_reason"] = None
    result["newest_mtime"] = _newest_mtime(required)
    return result


def artifacts_from_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper using dashboard run_metrics fields."""
    scenario_dir = metrics.get("scenario_dir") or ""
    if not scenario_dir and metrics.get("output_dir"):
        scenario_dir = str(Path(str(metrics["output_dir"])) / "run")
    return detect_postprocess_artifacts(
        scenario_dir,
        checkpoint_path=metrics.get("param_store_path"),
        workflow_id=metrics.get("workflow_id") or metrics.get("workflow"),
    )
