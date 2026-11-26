"""Step 8 — Review configuration and run analysis."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stgp_dashboard.services.config_builder import (
    build_config,
    config_to_summary_rows,
    save_config,
    validate_state,
)
from stgp_dashboard.services.progress import RunProgress
from stgp_dashboard.services.viz_scope import step_before_review
from stgp_dashboard.state import get_state, go_to_step, update_state
from stgp_dashboard.steps.priors import flush_live_priors_to_state


def render() -> None:
    st.header("Review & run")

    # Safety net: commit any live prior widgets before building config.
    flush_live_priors_to_state()

    state = get_state()
    errors = validate_state(state)
    if errors:
        for err in errors:
            st.error(err)

    cfg = build_config(state)

    run_optimization = bool(cfg.get("gp_model", {}).get("run_optimization", True))
    if not run_optimization:
        ckpt = (cfg.get("paths") or {}).get("trained_param_store", "")
        st.warning(
            f"**Skipping SVI** — loading checkpoint "
            f"`{ckpt or '(path not set)'}`. Hyperparameters will be frozen from "
            "the saved model, not re-tuned."
        )
    else:
        output_dir = str((cfg.get("paths") or {}).get("output_dir") or "").strip()
        if output_dir:
            existing = Path(output_dir).expanduser() / "run" / "trained_param_store.pt"
            if existing.is_file():
                st.info(
                    f"Existing checkpoint will be **overwritten** by fresh SVI: "
                    f"`{existing.resolve()}`"
                )

    st.subheader("Configuration summary")
    for label, value in config_to_summary_rows(cfg):
        st.write(f"**{label}:** `{value}`")

    with st.expander("Full YAML preview"):
        import yaml

        st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

    if st.button("Run model", type="primary", disabled=bool(errors), key="run_model_btn"):
        import pandas as pd

        flush_live_priors_to_state()
        state = get_state()
        cfg = build_config(state)

        output_dir = cfg["paths"]["output_dir"]
        save_config(cfg, Path(output_dir) / "dashboard_config.yaml")

        progress = RunProgress()
        stage_box = st.empty()
        progress_bar = st.progress(0.0, text="Starting…")
        metrics_box = st.empty()
        log_box = st.empty()

        orig_on_iter, orig_on_stage = __import__(
            "stgp_dashboard.services.progress", fromlist=["make_progress_callbacks"]
        ).make_progress_callbacks(progress)

        def on_stage(stage_id: str, message: str) -> None:
            orig_on_stage(stage_id, message)
            stage_box.info(f"**Stage:** {message}")
            if stage_id != "train":
                progress_bar.progress(1.0, text=message)
            metrics_box.markdown(
                f"**{progress.status_label()}** — {progress.format_status()}"
            )

        # Throttle Streamlit widget updates — every-iteration UI is much slower
        # than notebooks/CLI (which use tqdm or no progress widgets).
        _ui_every = 20

        def on_iteration(iteration: int, total: int, loss: float, _row: list) -> None:
            orig_on_iter(iteration, total, loss, _row)
            is_last = bool(total) and iteration >= total
            if iteration % _ui_every != 0 and not is_last:
                return
            status = progress.format_status()
            if total:
                progress_bar.progress(
                    min(1.0, iteration / total),
                    text=status,
                )
            metrics_box.markdown(f"**{progress.status_label()}** — {status}")

        from stgp_dashboard.paths import repo_root
        from stgp_pipeline.run import run_stgp_analysis

        cfg_built = build_config(state)
        output_path = Path(cfg_built["paths"]["output_dir"])
        config_path = save_config(cfg_built, output_path / "dashboard_config.yaml")

        try:
            metrics = run_stgp_analysis(
                cfg_built,
                output_path,
                base_dir=repo_root(),
                progress_callback=on_iteration,
                stage_callback=on_stage,
                postprocess=False,
            )
            scenario_dir = Path(metrics.get("scenario_dir", output_path / "run"))
            boundary_df = None
            track_df = None
            bp = scenario_dir / "boundary_proximity.csv"
            tp = scenario_dir / "optimization_track.csv"
            if bp.is_file():
                boundary_df = pd.read_csv(bp)
            if tp.is_file():
                track_df = pd.read_csv(tp)
            log_path = output_path / "run.log"
            run_log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""

            st.success(
                "Training complete — review prior range analysis before post-processing."
            )
            update_state(
                run_metrics=metrics,
                run_error=None,
                boundary_df=boundary_df,
                optimization_track=track_df,
                saved_config_path=str(config_path),
                run_log=run_log,
                postprocess_complete=False,
                visualization_outputs=None,
                mode="results",
                step=9,
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Run failed: {exc}")
            log_path = output_path / "run.log"
            run_log = ""
            if log_path.is_file():
                run_log = log_path.read_text(encoding="utf-8", errors="replace")
            update_state(
                run_error=str(exc),
                run_log=run_log,
                postprocess_complete=False,
            )
            if run_log:
                log_box.code(run_log[-4000:], language="text")

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="review_back"):
            go_to_step(step_before_review(st.session_state.get("viz_scope", "full")))
            st.rerun()
