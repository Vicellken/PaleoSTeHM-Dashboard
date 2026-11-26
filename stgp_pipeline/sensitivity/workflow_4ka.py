"""
Adapter for 1D GIA STGP training and temporal/spatial exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .. import data, gia_setup, gia_setup_3d, training
from .workflow_common import (
    boundary_proximity_dataframe,
    emit_stage,
    export_gia_temporal_and_spatial,
    save_trained_param_store,
    training_run_options,
)


def _resolve_cfg_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def run_workflow_4ka(
    cfg: Mapping[str, Any],
    scenario_dir: Path,
    *,
    base_dir: Path,
    progress_callback=None,
    stage_callback=None,
    postprocess: bool = True,
    force_recompute: bool = True,
) -> Dict[str, Any]:
    """
    Run the 4 ka workflow: training, parameter extraction, and optional exports.

    When ``postprocess`` is False, stop after NLL / boundary proximity so the
    dashboard can gate temporal/spatial work on prior-range review.
    """
    paths = cfg.get("paths", {}) or {}
    paleo_csv = paths.get("paleo_rsl")
    if not paleo_csv:
        raise KeyError("Missing cfg['paths']['paleo_rsl'] for 4ka workflow")

    max_age_bp = int((cfg.get("analysis", {}) or {}).get("max_age_bp", 4000))
    rsl = data.load_and_filter_rsl(
        str(paleo_csv), max_age_bp=max_age_bp, config=dict(cfg)
    )

    gia_cfg = cfg.get("gia", {}) or {}
    gia_mode = str(gia_cfg.get("mode", "mat"))
    exclude_models = list(gia_cfg.get("exclude_models", ["14055"]))

    if gia_mode == "npy_5model":
        interp_funcs, model_names, indices_3d = (
            gia_setup_3d.load_npy_5model_ensemble(
                cfg,
                base_dir=base_dir,
            )
        )
        gia = gia_setup_3d.compute_weights_and_covariance_3d(
            interp_funcs,
            model_names,
            indices_3d,
            rsl,
            cfg,
            bp_min=0.0,
            bp_max=max_age_bp,
            base_dir=base_dir,
        )
    elif gia_mode == "mat":
        gia_mat_dir = _resolve_cfg_path(str(paths["gia_mat_dir"]), base_dir=base_dir)
        interp_funcs, model_names = gia_setup.load_gia_ensemble(
            str(gia_mat_dir), exclude_models=exclude_models
        )
        gia = gia_setup.compute_weights_and_covariance(
            interp_funcs,
            model_names,
            rsl,
            cfg,
            bp_min=0.0,
            bp_max=max_age_bp,
        )
    else:
        interp_funcs, model_names, indices_3d = gia_setup_3d.load_ensemble_from_cfg(
            dict(cfg),
            exclude_models=exclude_models,
            time_start=float((gia_cfg or {}).get("time_start_bp", 16000.0)),
            base_dir=base_dir,
        )
        gia = gia_setup_3d.compute_weights_and_covariance_3d(
            interp_funcs,
            model_names,
            indices_3d,
            rsl,
            cfg,
            bp_min=0.0,
            bp_max=max_age_bp,
            base_dir=base_dir,
        )

    residual = gia_setup.compute_residuals_and_noise(rsl, gia)
    run_opt, load_path, save_path = training_run_options(cfg, scenario_dir)
    if run_opt:
        emit_stage(stage_callback, "train", "Training GP model (SVI)")
    else:
        emit_stage(
            stage_callback,
            "load_model",
            "Loading trained parameters from checkpoint",
        )
    model = training.build_and_train(
        cfg,
        rsl.X,
        residual.y_residual,
        residual.noise_total,
        rsl.age_sigma,
        run_key="stgp_with_gia",
        train=run_opt,
        param_store_path=str(load_path) if load_path else None,
        progress_callback=progress_callback,
    )
    if run_opt:
        emit_stage(stage_callback, "save_model", "Saving trained parameters")
        param_store_path = save_trained_param_store(save_path)
    else:
        param_store_path = str(load_path)

    emit_stage(stage_callback, "nll", "Computing negative log-likelihood")
    nll = float(training.negative_log_likelihhod(model.gpr).item())
    df_params = training.extract_hyperparams(model)
    df_params.to_csv(scenario_dir / "optimized_params.csv", index=False)

    emit_stage(stage_callback, "boundary", "Running prior range / boundary check")
    df_boundary = boundary_proximity_dataframe(df_params, cfg)
    df_boundary.to_csv(scenario_dir / "boundary_proximity.csv", index=False)

    if model.track_list is not None and not model.track_list.empty:
        model.track_list.to_csv(
            scenario_dir / "optimization_track.csv", index_label="iteration"
        )

    base_metrics: Dict[str, Any] = {
        "workflow": "4ka",
        "negative_log_likelihood": nll,
        "n_training_points": int(rsl.n_slip),
        "n_hyperparameters": int(len(df_params)),
        "n_regions": int(len(rsl.unique_regions)),
        "gia_mode": gia_mode,
        "param_store_path": param_store_path,
        "run_optimization": run_opt,
        "postprocess_complete": False,
    }

    if not postprocess:
        return base_metrics

    post = export_gia_temporal_and_spatial(
        model,
        rsl,
        gia,
        cfg,
        scenario_dir,
        workflow="4ka",
        max_age_bp=max_age_bp,
        bp_min=0.0,
        bp_max=max_age_bp,
        nc_suffix="4ka",
        stage_callback=stage_callback,
        force_recompute=force_recompute,
    )
    return {**base_metrics, **post}
