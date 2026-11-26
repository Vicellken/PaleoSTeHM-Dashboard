"""
Unified STGP analysis entry point for dashboard and CLI use.

Routes to the appropriate workflow adapter based on config flags rather than
legacy period labels (4ka / 16ka).
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .sensitivity.workflow_common import save_yaml, uses_combined_3d_gia


StageCallback = Callable[[str, str], None]
ProgressCallback = Callable[[int, int, float, list], None]


def resolve_workflow_id(cfg: Mapping[str, Any]) -> str:
    """
    Pick the internal workflow adapter from config capabilities.

    Returns one of: ``16ka``, ``4ka``, ``3d-gia``, ``regional-gia``.
    """
    dashboard = cfg.get("dashboard") or {}
    use_gia = dashboard.get("use_gia")
    if use_gia is None:
        gia_cfg = cfg.get("gia") or {}
        use_gia = bool(gia_cfg) and gia_cfg.get("enabled", True) is not False

    if not use_gia:
        return "16ka"

    hint = dashboard.get("workflow_hint")
    if hint in {"4ka", "3d-gia", "regional-gia", "16ka"}:
        return str(hint)

    paths = cfg.get("paths") or {}
    gia_cfg = cfg.get("gia") or {}

    if dashboard.get("region_profile") == "regional":
        return "regional-gia"

    if uses_combined_3d_gia(paths):
        return "3d-gia"

    mode = str(gia_cfg.get("mode") or gia_cfg.get("format") or "")
    if mode == "npy_5model":
        return "3d-gia"
    if mode == "mat" and not uses_combined_3d_gia(paths):
        return "4ka"
    if gia_cfg.get("ensemble") in {"all", "3d"}:
        return "3d-gia"

    gia_3d_dir = paths.get("gia_3d_dir")
    if isinstance(gia_3d_dir, str) and gia_3d_dir.strip():
        return "3d-gia"

    return "4ka"


def resolve_workflow_label(cfg: Mapping[str, Any], workflow_id: str) -> str:
    """User-facing capability label for a resolved workflow id."""
    dashboard = cfg.get("dashboard") or {}
    if dashboard.get("workflow_label"):
        return str(dashboard["workflow_label"])

    use_gia = dashboard.get("use_gia")
    if use_gia is None:
        gia_cfg = cfg.get("gia") or {}
        use_gia = bool(gia_cfg) and gia_cfg.get("enabled", True) is not False

    if not use_gia:
        return "No GIA prior"

    if workflow_id == "3d-gia":
        return "GIA prior (3D combined)"
    if workflow_id == "regional-gia":
        return "GIA prior (Regional)"

    paths = cfg.get("paths") or {}
    gia_cfg = cfg.get("gia") or {}
    if paths.get("gia_3d_dir") or gia_cfg.get("ensemble") in {"all", "3d"}:
        return "GIA prior (3D combined)"

    source = str(gia_cfg.get("1d_source", "both")).lower()
    if source == "both":
        return "GIA prior (1D ICE6G + ANU)"
    if source == "mat":
        return "GIA prior (1D ICE6G)"
    if source == "anu":
        return "GIA prior (1D ANU)"
    return f"GIA prior ({workflow_id})"


def run_stgp_analysis(
    cfg: Mapping[str, Any],
    output_dir: Path,
    *,
    base_dir: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stage_callback: Optional[StageCallback] = None,
    log_capture: bool = True,
    postprocess: bool = True,
    force_recompute: bool = True,
) -> Dict[str, Any]:
    """
    Run a single adopted STGP analysis and write artifacts to ``output_dir``.

    Parameters
    ----------
    cfg
        Full study configuration dict (from dashboard or YAML).
    output_dir
        Directory for outputs, config dump, and CSV/NetCDF artifacts.
    base_dir
        Base directory for resolving relative paths in ``cfg['paths']``.
        Defaults to ``output_dir``.
    progress_callback
        Optional ``(iteration, total, loss, track_row)`` callback during SVI.
    stage_callback
        Optional ``(stage_id, message)`` callback for pipeline stages.
    log_capture
        When True, capture stdout/stderr to ``run.log`` in ``output_dir``.
    postprocess
        When False, stop after training / NLL / prior-range (boundary) analysis
        and skip temporal/spatial prediction exports. Defaults to True so CLI
        and sensitivity suites keep full-run behaviour.
    force_recompute
        When True (default), overwrite temporal/spatial prediction exports.
        When False, reuse existing CSVs/NetCDF that are not older than the
        checkpoint.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir or output_dir).resolve()

    workflow_id = resolve_workflow_id(cfg)
    cfg_dict = copy.deepcopy(dict(cfg))
    paths_dict = cfg_dict.setdefault("paths", {})
    paths_dict["output_dir"] = str(output_dir)
    for key in ("paleo_rsl", "dataset"):
        raw = paths_dict.get(key)
        if isinstance(raw, str) and raw.strip():
            p = Path(raw).expanduser()
            if not p.is_absolute():
                candidate = (base_dir / p).resolve()
                if candidate.exists() or not p.exists():
                    paths_dict[key] = str(candidate)

    config_path = output_dir / "dashboard_config.yaml"
    save_yaml(config_path, cfg_dict)

    scenario_dir = output_dir / "run"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    def _stage(stage_id: str, message: str) -> None:
        if stage_callback is not None:
            stage_callback(stage_id, message)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    metrics: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            _stage("start", f"Starting workflow: {workflow_id}")
            if workflow_id == "16ka":
                from .sensitivity.workflow_16ka import run_workflow_16ka

                _stage("load_data", "Loading and filtering RSL data")
                metrics = run_workflow_16ka(
                    cfg_dict,
                    scenario_dir,
                    base_dir=base_dir,
                    progress_callback=progress_callback,
                    stage_callback=stage_callback,
                    postprocess=postprocess,
                    force_recompute=force_recompute,
                )
            elif workflow_id == "4ka":
                from .sensitivity.workflow_4ka import run_workflow_4ka

                _stage("load_data", "Loading RSL data and GIA ensemble")
                metrics = run_workflow_4ka(
                    cfg_dict,
                    scenario_dir,
                    base_dir=base_dir,
                    progress_callback=progress_callback,
                    stage_callback=stage_callback,
                    postprocess=postprocess,
                    force_recompute=force_recompute,
                )
            elif workflow_id == "3d-gia":
                from .sensitivity.workflow_3d_gia import run_workflow_3d_gia

                _stage("load_data", "Loading RSL data and combined GIA ensemble")
                metrics = run_workflow_3d_gia(
                    cfg_dict,
                    scenario_dir,
                    base_dir=base_dir,
                    progress_callback=progress_callback,
                    stage_callback=stage_callback,
                    postprocess=postprocess,
                    force_recompute=force_recompute,
                )
            elif workflow_id == "regional-gia":
                from .sensitivity.workflow_regional_gia import (
                    run_workflow_regional_gia,
                )

                _stage("load_data", "Loading regional GIA workflow")
                metrics = run_workflow_regional_gia(
                    cfg_dict,
                    scenario_dir,
                    base_dir=base_dir,
                    progress_callback=progress_callback,
                    stage_callback=stage_callback,
                    postprocess=postprocess,
                    force_recompute=force_recompute,
                )
            else:
                raise ValueError(f"Unsupported workflow: {workflow_id!r}")

            _stage(
                "complete",
                "Training and prior-range analysis complete"
                if not postprocess
                else "Analysis complete",
            )
            metrics["workflow_id"] = workflow_id
            metrics["workflow_label"] = resolve_workflow_label(cfg_dict, workflow_id)
            metrics["status"] = "success"
            metrics["output_dir"] = str(output_dir)
            metrics["scenario_dir"] = str(scenario_dir)
            metrics["config_path"] = str(config_path)
            metrics.setdefault("postprocess_complete", bool(postprocess))
    except Exception as exc:  # noqa: BLE001
        metrics = {
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(exc),
            "output_dir": str(output_dir),
            "scenario_dir": str(scenario_dir),
            "config_path": str(config_path),
            "postprocess_complete": False,
        }
        _stage("error", str(exc))
        raise
    finally:
        if log_capture:
            log_path = output_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as f:
                f.write("--- stdout ---\n")
                f.write(stdout_buffer.getvalue())
                f.write("\n--- stderr ---\n")
                f.write(stderr_buffer.getvalue())
                if metrics.get("status") == "failed":
                    f.write(f"\n--- error ---\n{metrics.get('error', '')}\n")

    return metrics


def run_stgp_postprocess(
    cfg: Mapping[str, Any],
    output_dir: Path,
    *,
    param_store_path: Optional[str | Path] = None,
    base_dir: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stage_callback: Optional[StageCallback] = None,
    log_capture: bool = True,
    force_recompute: bool = True,
) -> Dict[str, Any]:
    """
    Resume a trained run and execute temporal/spatial post-processing only.

    Loads the saved param store (from ``param_store_path`` or
    ``{output_dir}/run/trained_param_store.pt``), skips SVI, and runs full
    prediction exports.

    When ``force_recompute`` is False, existing prediction CSVs/NetCDF that are
    not older than the checkpoint are reused.
    """
    from .sensitivity.workflow_common import cfg_for_checkpoint_postprocess

    output_dir = Path(output_dir).resolve()
    scenario_dir = output_dir / "run"
    store = Path(
        param_store_path
        if param_store_path is not None
        else scenario_dir / "trained_param_store.pt"
    ).resolve()
    if not store.is_file():
        raise FileNotFoundError(f"Trained param store not found: {store}")

    cfg_pp = cfg_for_checkpoint_postprocess(cfg, store)
    return run_stgp_analysis(
        cfg_pp,
        output_dir,
        base_dir=base_dir,
        progress_callback=progress_callback,
        stage_callback=stage_callback,
        log_capture=log_capture,
        postprocess=True,
        force_recompute=force_recompute,
    )
