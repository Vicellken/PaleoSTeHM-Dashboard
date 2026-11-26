"""Load an existing YAML config into wizard session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from stgp_dashboard.services.artifact_detection import detect_postprocess_artifacts
from stgp_dashboard.services.workflow_labels import gia_pipeline_to_ui


def _resolve_input_path(raw: Any, base_dir: Path) -> str:
    """Resolve a config path against ``base_dir`` when relative.

    Empty values stay empty. Absolute paths are expanded/resolved as-is.
    Relative paths become ``(base_dir / raw).resolve()`` even if missing.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).expanduser()
    return str(p.resolve())


def summarize_resolved_inputs(state: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Return found/missing labels for resolved input paths in ``state``."""
    labels = {
        "paleo_rsl": "Paleo RSL",
        "gia_mat_dir": "GIA MAT dir",
        "gia_anu_dir": "GIA ANU dir",
        "gia_3d_dir": "GIA 3D dir",
    }
    found: List[str] = []
    missing: List[str] = []
    for key, label in labels.items():
        path = str(state.get(key) or "").strip()
        if not path:
            continue
        p = Path(path)
        ok = p.is_file() if key == "paleo_rsl" else p.is_dir()
        (found if ok else missing).append(label)
    return {"found": found, "missing": missing}


def config_dict_to_state(
    cfg: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = cfg.get("paths") or {}
    analysis = cfg.get("analysis") or {}
    gia = cfg.get("gia") or {}
    gp = cfg.get("gp_model") or {}
    temporal = cfg.get("temporal_test_age") or {}
    spatial_age = cfg.get("spatial_test_age") or {}
    rate_age = cfg.get("rate_test_age") or {}
    spatial = cfg.get("spatial_grid") or {}
    data_filter = cfg.get("data_filter") or {}
    adj = (cfg.get("data_adjustments") or {}).get("level2_downweighting") or {}
    centroid = cfg.get("temporal_centroid") or {}
    dashboard = cfg.get("dashboard") or {}
    spatial_mask = cfg.get("spatial_plot_mask") or {}
    budget = cfg.get("budget") or {}

    paleo_raw = paths.get("paleo_rsl") or paths.get("dataset") or ""
    gia_mat_raw = paths.get("gia_mat_dir", "")
    gia_anu_raw = paths.get("gia_ANU_dir", "")
    gia_3d_raw = paths.get("gia_3d_dir", "")

    if base_dir is not None:
        paleo = _resolve_input_path(paleo_raw, base_dir)
        gia_mat = _resolve_input_path(gia_mat_raw, base_dir)
        gia_anu = _resolve_input_path(gia_anu_raw, base_dir)
        gia_3d = _resolve_input_path(gia_3d_raw, base_dir)
    else:
        paleo = str(paleo_raw or "")
        gia_mat = str(gia_mat_raw or "")
        gia_anu = str(gia_anu_raw or "")
        gia_3d = str(gia_3d_raw or "")

    state: Dict[str, Any] = {
        "paleo_rsl": paleo,
        "output_dir": paths.get("output_dir") or "",
        "max_age_bp": int(analysis.get("max_age_bp", temporal.get("end", 16000))),
        "viz_start_bp": int(temporal.get("start", 0)),
        "viz_end_bp": int(temporal.get("end", analysis.get("max_age_bp", 16000))),
        "temporal_interval": int(temporal.get("interval", 200)),
        "spatial_interval": int(spatial_age.get("interval", 1000)),
        "rate_interval": int(rate_age.get("interval", 100)),
        "use_gia": dashboard.get("use_gia", bool(gia)),
        "gia_mat_dir": gia_mat,
        "gia_anu_dir": gia_anu,
        "gia_3d_dir": gia_3d,
        "gia_1d_source": gia_pipeline_to_ui(str(gia.get("1d_source", "both"))),
        "gia_mean_weighting": gia.get("mean_weighting", "wrss_all"),
        "gia_n_search": int(gia.get("n_search", 50)),
        "gia_use_3d": bool(gia_3d),
        "gia_3d_source_mode": "folder",
        "gia_3d_selected_files": [],
        "num_iterations": int(gp.get("num_iterations", 500)),
        "learning_rate": float(gp.get("learning_rate", 0.05)),
        "decay_r": float(gp.get("decay_r", 0.2)),
        "step_size": int(gp.get("step_size", 50)),
        "jitter": float(gp.get("jitter", 1e-5)),
        "fixed_seed": bool(gp.get("fixed_seed", True)),
        "seed": int(gp.get("seed", 0)),
        "gpu": bool(gp.get("gpu", False)),
        "filter_levels": list(data_filter.get("levels", [1, 2])),
        "l2_downweight_enabled": bool(adj.get("enabled", False)),
        "l2_downweight_multiplier": float(adj.get("multiplier", 2.0)),
        "l2_downweight_scope": adj.get("scope", "slip_only"),
        "region_merges": list(data_filter.get("region_merges") or []),
        "temporal_centroid_mode": centroid.get("mode", "data_mean"),
        "prior_preset": dashboard.get("prior_preset", "regional"),
        "kernel_priors": dict(gp.get("kernels") or {}),
        "lat_padding": float(spatial.get("lat_padding", 2.0)),
        "lon_padding": float(spatial.get("lon_padding", 2.0)),
        "grid_resolution": float(spatial.get("resolution", 0.3)),
        "spatial_mask_enabled": bool(spatial_mask.get("enabled", True)),
        "spatial_mask_calibration_factor": float(
            spatial_mask.get("calibration_factor", 2.0)
        ),
        "budget_age_windows": _budget_windows_from_cfg(budget),
        "budget_merge_gia_global": bool(budget.get("merge_gia_global", True)),
        "budget_gmsl_xlsx": _budget_gmsl_xlsx_from_cfg(budget, base_dir=base_dir),
        "training_mode": (
            "load_checkpoint"
            if gp.get("run_optimization") is False
            else "train"
        ),
        "param_store_path": str(paths.get("trained_param_store") or ""),
        "loaded_config_path": "",
    }
    return state


def _budget_windows_from_cfg(budget: Mapping[str, Any]) -> List[List[float]]:
    from stgp_dashboard.services.config_builder import normalize_budget_age_windows

    windows = normalize_budget_age_windows(budget.get("age_windows_bp"))
    return windows or [[0.0, 4000.0]]


def _budget_gmsl_xlsx_from_cfg(
    budget: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    from stgp_dashboard.services.config_builder import normalize_budget_gmsl_xlsx

    normalized = normalize_budget_gmsl_xlsx(budget.get("gmsl_xlsx"))
    if normalized is None:
        return None
    if base_dir is None:
        return normalized
    return {
        "anu": _resolve_input_path(normalized["anu"], base_dir),
        "ice6g": _resolve_input_path(normalized["ice6g"], base_dir),
    }


def load_config_file(path: str | Path) -> Dict[str, Any]:
    import yaml

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    state = config_dict_to_state(cfg, base_dir=config_path.parent)
    state["loaded_config_path"] = str(config_path)

    output_dir = Path(str(state.get("output_dir") or "")).expanduser()
    scenario_dir = output_dir / "run"
    default_checkpoint = scenario_dir / "trained_param_store.pt"
    # Detect an existing checkpoint for convenience, but do NOT force
    # load_checkpoint — only honor explicit run_optimization: false from YAML.
    if default_checkpoint.is_file():
        state["checkpoint_available"] = True
        if not state.get("param_store_path"):
            state["param_store_path"] = str(default_checkpoint.resolve())
    else:
        state["checkpoint_available"] = False

    # Restore post-process completion when prediction exports already exist.
    workflow_hint = (cfg.get("dashboard") or {}).get("workflow_hint")
    if not workflow_hint:
        use_gia = state.get("use_gia", True)
        workflow_hint = "16ka" if not use_gia else None
    detected = detect_postprocess_artifacts(
        scenario_dir,
        checkpoint_path=state.get("param_store_path") or None,
        workflow_id=workflow_hint,
    )
    if detected.get("reusable"):
        state["postprocess_complete"] = True
        try:
            from stgp_dashboard.services.visualize_runner import discover_saved_figures

            figs = discover_saved_figures(scenario_dir)
            if figs:
                state["visualization_outputs"] = figs
        except Exception:  # noqa: BLE001
            pass
    else:
        state["postprocess_complete"] = False

    return state
