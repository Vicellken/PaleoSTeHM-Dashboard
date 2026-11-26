"""Build full STGP YAML config from wizard session state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from stgp_dashboard.services.workflow_labels import (
    gia_pipeline_to_ui,
    gia_ui_to_pipeline,
    resolve_workflow_label,
)


def _presets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "presets"


def load_yaml_preset(name: str) -> Dict[str, Any]:
    path = _presets_dir() / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_kernel_priors(state: Mapping[str, Any]) -> Dict[str, Any]:
    priors_yaml = load_yaml_preset("priors_regional.yaml")
    preset_key = str(state.get("prior_preset", "regional"))
    preset = priors_yaml.get(preset_key, priors_yaml.get("regional", {}))
    kernels = deepcopy(priors_yaml.get("kernels_base", {}))

    for kernel_name in ("regional_linear", "regional_nonlinear", "local_nonlinear"):
        if kernel_name in preset and kernel_name in kernels:
            kernels[kernel_name].update(preset[kernel_name])

    overrides = state.get("kernel_priors") or {}
    for kernel_name, block in overrides.items():
        if kernel_name in kernels and isinstance(block, dict):
            kernels[kernel_name].update(block)

    return kernels


def validate_state(state: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    state = _ensure_gia_discovery(dict(state))
    paleo = str(state.get("paleo_rsl", "")).strip()
    output = str(state.get("output_dir", "")).strip()
    if not paleo:
        errors.append("Paleo RSL CSV path is required.")
    elif not Path(paleo).expanduser().is_file():
        errors.append(f"Paleo RSL file not found: {paleo}")
    if not output:
        errors.append("Output directory is required.")

    max_age = int(state.get("max_age_bp", 0))
    viz_end = int(state.get("viz_end_bp", 0))
    if max_age <= 0:
        errors.append("Analysis max age must be positive.")
    if viz_end > max_age:
        errors.append(
            f"Visualization end ({viz_end} BP) must be ≤ data cutoff ({max_age} BP)."
        )

    if state.get("use_gia"):
        discovery = state.get("gia_discovery") or {}
        source = gia_ui_to_pipeline(state.get("gia_1d_source", "both"))
        if source in {"mat", "both"} and not discovery.get("has_mat"):
            errors.append("ICE6G .mat directory is empty or missing.")
        if source in {"anu", "both"} and not discovery.get("has_anu"):
            errors.append("ANU .txt directory is empty or missing.")
        if state.get("gia_use_3d"):
            source_mode = str(state.get("gia_3d_source_mode", "folder") or "folder")
            selected = list(state.get("gia_3d_selected_files") or [])
            if source_mode == "files":
                if not selected:
                    errors.append("Select at least one 3D GIA .csv / .txt file.")
                else:
                    missing = [
                        p for p in selected if not Path(p).expanduser().is_file()
                    ]
                    if missing:
                        errors.append(
                            f"3D GIA file(s) not found: {', '.join(missing[:3])}"
                        )
            elif not discovery.get("has_3d"):
                errors.append(
                    "No 3D GIA .csv / .txt files found in gia_3d_dir. "
                    "Point at a folder containing 3D GIA grids (e.g., Data/3D_GIA)."
                )

    if state.get("training_mode") == "load_checkpoint":
        ckpt = str(state.get("param_store_path", "")).strip()
        if not ckpt:
            errors.append(
                "Trained model checkpoint (.pt) is required when loading a saved model."
            )
        elif not Path(ckpt).expanduser().is_file():
            errors.append(f"Checkpoint file not found: {ckpt}")

    windows = normalize_budget_age_windows(state.get("budget_age_windows"))
    if not windows:
        errors.append(
            "Sea-level budget needs at least one age window with start < end (BP)."
        )

    return errors


def normalize_budget_age_windows(raw: Any) -> List[List[float]]:
    """Return structurally valid ``[[lo, hi], ...]`` budget windows (BP)."""
    windows: List[List[float]] = []
    if not raw:
        return windows
    for item in raw:
        if item is None:
            continue
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lo, hi = float(item[0]), float(item[1])
            else:
                continue
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            continue
        windows.append([lo, hi])
    return windows


def normalize_budget_gmsl_xlsx(raw: Any) -> Optional[Dict[str, str]]:
    """Normalize optional ``budget.gmsl_xlsx`` to ``{anu, ice6g}`` or ``None``."""
    if raw is None or raw is False or raw == "":
        return None
    anu: Any = None
    ice6g: Any = None
    if isinstance(raw, Mapping):
        anu = raw.get("anu") or raw.get("ANU")
        ice6g = (
            raw.get("ice6g")
            or raw.get("6g")
            or raw.get("ice_6g")
            or raw.get("ICE6G")
        )
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        anu, ice6g = raw[0], raw[1]
    else:
        return None
    anu_text = str(anu or "").strip()
    ice6g_text = str(ice6g or "").strip()
    if not anu_text or not ice6g_text:
        return None
    return {"anu": anu_text, "ice6g": ice6g_text}


def _ensure_gia_discovery(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("use_gia"):
        return state
    from stgp_dashboard.paths import repo_root
    from stgp_dashboard.services.gia_discovery import discover_gia_paths

    discovery = discover_gia_paths(
        repo_root(),
        gia_mat_dir=state.get("gia_mat_dir") or None,
        gia_anu_dir=state.get("gia_anu_dir") or None,
        gia_3d_dir=state.get("gia_3d_dir") if state.get("gia_use_3d") else None,
    )
    state["gia_discovery"] = discovery
    return state


def _stage_3d_selected_files(
    selected_files: List[str], output_dir: Path
) -> Path:
    """Copy selected 3D grids into ``output_dir/gia_3d_selected`` for the pipeline."""
    import shutil

    stage_dir = output_dir / "gia_3d_selected"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous staged copies so removed selections do not linger.
    for existing in stage_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    for src in selected_files:
        src_path = Path(src).expanduser().resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"3D GIA file not found: {src_path}")
        shutil.copy2(src_path, stage_dir / src_path.name)
    return stage_dir


def _resolve_gia_3d_dir(state: Mapping[str, Any], output_path: str) -> str:
    """Return the directory path written to ``paths.gia_3d_dir``."""
    if not state.get("gia_use_3d"):
        return ""
    source_mode = str(state.get("gia_3d_source_mode", "folder") or "folder")
    selected = list(state.get("gia_3d_selected_files") or [])
    if source_mode == "files" and selected:
        return str(_stage_3d_selected_files(selected, Path(output_path)))
    discovery = state.get("gia_discovery") or {}
    return str(
        discovery.get("gia_3d_dir")
        or state.get("gia_3d_dir")
        or ""
    ).strip()


def _csv_has_level_column(
    paleo_path: str, preview: Optional[Mapping[str, Any]] = None
) -> bool:
    """Return True if the dataset has a ``Level`` column (or cannot be inspected)."""
    if isinstance(preview, Mapping) and preview.get("columns"):
        return "Level" in preview["columns"]
    try:
        p = Path(paleo_path)
        if p.is_file():
            import pandas as pd

            return "Level" in pd.read_csv(p, nrows=0).columns
    except Exception:  # noqa: BLE001
        pass
    return True


def build_config(state: Mapping[str, Any]) -> Dict[str, Any]:
    state = _ensure_gia_discovery(dict(state))
    defaults = load_yaml_preset("defaults.yaml")
    cfg: Dict[str, Any] = deepcopy(defaults)

    paleo_path = str(Path(state["paleo_rsl"]).expanduser().resolve())
    output_path = str(Path(state["output_dir"]).expanduser().resolve())

    cfg["paths"] = {
        "paleo_rsl": paleo_path,
        "output_dir": output_path,
    }

    if state.get("use_gia"):
        discovery = state.get("gia_discovery") or {}
        cfg["paths"]["gia_mat_dir"] = discovery.get("gia_mat_dir", "") or str(
            state.get("gia_mat_dir", "")
        )
        cfg["paths"]["gia_ANU_dir"] = discovery.get("gia_anu_dir", "") or str(
            state.get("gia_anu_dir", "")
        )

        gia: Dict[str, Any] = deepcopy(cfg.get("gia", {}))
        gia["enabled"] = True
        gia["1d_source"] = gia_ui_to_pipeline(state.get("gia_1d_source", "both"))
        gia["mean_weighting"] = state.get("gia_mean_weighting", "wrss_all")
        gia["n_search"] = int(state.get("gia_n_search", 50))
        gia["exclude_models"] = ["14055"]
        if state.get("gia_use_3d"):
            gia_3d = _resolve_gia_3d_dir(state, output_path)
            if gia_3d:
                cfg["paths"]["gia_3d_dir"] = gia_3d
            gia["ensemble"] = "all"
            gia.pop("mode", None)
        else:
            gia["mode"] = "mat"
            gia["ensemble"] = None
            cfg["paths"].pop("gia_3d_dir", None)
        cfg["gia"] = gia
    else:
        cfg.pop("gia", None)

    cfg["analysis"] = {"max_age_bp": int(state["max_age_bp"])}

    viz_start = int(state.get("viz_start_bp", 0))
    viz_end = int(state.get("viz_end_bp", state["max_age_bp"]))
    cfg["temporal_test_age"] = {
        "start": viz_start,
        "end": viz_end,
        "interval": int(state.get("temporal_interval", 200)),
    }
    cfg["spatial_test_age"] = {
        "start": viz_start,
        "end": viz_end,
        "interval": int(state.get("spatial_interval", 1000)),
    }
    cfg["rate_test_age"] = {
        "start": viz_start,
        "end": viz_end,
        "interval": int(state.get("rate_interval", 100)),
    }

    cfg["spatial_grid"] = {
        "lat_padding": float(state.get("lat_padding", 2.0)),
        "lon_padding": float(state.get("lon_padding", 2.0)),
        "resolution": float(state.get("grid_resolution", 0.3)),
    }

    levels = list(state.get("filter_levels", [1, 2]))
    if levels and _csv_has_level_column(paleo_path, state.get("data_preview")):
        cfg["data_filter"] = {"levels": levels}
    else:
        cfg["data_filter"] = {}
    merges = state.get("region_merges") or []
    if merges:
        cfg["data_filter"]["region_merges"] = merges

    cfg["data_adjustments"] = {
        "level2_downweighting": {
            "enabled": bool(state.get("l2_downweight_enabled", False)),
            "multiplier": float(state.get("l2_downweight_multiplier", 2.0)),
            "scope": state.get("l2_downweight_scope", "slip_only"),
            "scale_targets": ["rsl", "age"],
        }
    }

    cfg["temporal_centroid"] = {
        "mode": state.get("temporal_centroid_mode", "data_mean"),
        "grid_n": 3,
    }

    cfg["spatial_plot_mask"] = {
        "enabled": bool(state.get("spatial_mask_enabled", True)),
        "calibration_factor": float(
            state.get("spatial_mask_calibration_factor", 2.0)
        ),
    }

    budget_windows = normalize_budget_age_windows(
        state.get("budget_age_windows") or [[0, 4000]]
    )
    if not budget_windows:
        budget_windows = [[0.0, 4000.0]]
    cfg["budget"] = {
        "age_windows_bp": budget_windows,
        "merge_gia_global": bool(state.get("budget_merge_gia_global", True)),
    }
    gmsl_xlsx = normalize_budget_gmsl_xlsx(state.get("budget_gmsl_xlsx"))
    if gmsl_xlsx is not None:
        cfg["budget"]["gmsl_xlsx"] = gmsl_xlsx

    training_mode = str(state.get("training_mode", "train"))
    run_optimization = training_mode != "load_checkpoint"

    cfg["gp_model"] = {
        "run_optimization": run_optimization,
        "fixed_seed": bool(state.get("fixed_seed", True)),
        "seed": int(state.get("seed", 0)),
        "num_iterations": int(state.get("num_iterations", 500)),
        "learning_rate": float(state.get("learning_rate", 0.05)),
        "jitter": float(state.get("jitter", 1e-5)),
        "decay_r": float(state.get("decay_r", 0.2)),
        "step_size": int(state.get("step_size", 50)),
        "gpu": bool(state.get("gpu", False)),
        "kernels": build_kernel_priors(state),
    }

    if not run_optimization:
        ckpt = str(state.get("param_store_path", "")).strip()
        if ckpt:
            cfg["paths"]["trained_param_store"] = str(
                Path(ckpt).expanduser().resolve()
            )

    from stgp_dashboard.paths import ensure_import_paths

    ensure_import_paths()
    from stgp_pipeline.run import resolve_workflow_id  # noqa: E402

    workflow_id = resolve_workflow_id(cfg)
    cfg["dashboard"] = {
        "use_gia": bool(state.get("use_gia", False)),
        "prior_preset": state.get("prior_preset", "regional"),
        "source": "stgp_dashboard",
        "workflow_label": resolve_workflow_label(cfg, workflow_id),
    }

    return cfg


def save_config(cfg: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# STGP dashboard configuration\n"
        "# Generated by stgp_dashboard — edit for advanced debugging.\n\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    return path


def config_to_summary_rows(cfg: Mapping[str, Any]) -> List[Tuple[str, str]]:
    paths = cfg.get("paths") or {}
    analysis = cfg.get("analysis") or {}
    gia = cfg.get("gia") or {}
    gp = cfg.get("gp_model") or {}
    rows = [
        ("Paleo RSL", str(paths.get("paleo_rsl", ""))),
        ("Output dir", str(paths.get("output_dir", ""))),
        ("Max age (BP)", str(analysis.get("max_age_bp", ""))),
        ("Use GIA", "Yes" if gia else "No"),
    ]
    if gia:
        rows.extend(
            [
                (
                    "GIA 1D source",
                    gia_pipeline_to_ui(str(gia.get("1d_source", ""))),
                ),
                ("GIA mean weighting", str(gia.get("mean_weighting", ""))),
            ]
        )
        if paths.get("gia_3d_dir"):
            rows.append(("3D GIA dir", str(paths.get("gia_3d_dir", ""))))
            rows.append(("GIA ensemble", str(gia.get("ensemble", "all"))))
    dashboard = cfg.get("dashboard") or {}
    if dashboard.get("workflow_label"):
        rows.append(("Analysis mode", str(dashboard["workflow_label"])))
    rows.extend(
        [
            (
                "Training",
                "Load checkpoint"
                if gp.get("run_optimization") is False
                else "Train new model",
            ),
            ("SVI iterations", str(gp.get("num_iterations", ""))),
            ("Learning rate", str(gp.get("learning_rate", ""))),
            ("GPU", str(gp.get("gpu", False))),
        ]
    )
    if gp.get("run_optimization") is False:
        rows.append(
            ("Checkpoint", str(paths.get("trained_param_store", "")))
        )
    mask = cfg.get("spatial_plot_mask") or {}
    if mask:
        rows.append(
            (
                "Spatial mask factor",
                str(mask.get("calibration_factor", 2.0)),
            )
        )
    budget = cfg.get("budget") or {}
    windows = budget.get("age_windows_bp") or []
    if windows:
        win_txt = "; ".join(
            f"{int(float(w[0]))}–{int(float(w[1]))} BP" for w in windows if len(w) >= 2
        )
        rows.append(("Budget windows", win_txt))
        rows.append(
            (
                "Budget merge GIA+Gm",
                "Yes" if budget.get("merge_gia_global", True) else "No",
            )
        )
    return rows
