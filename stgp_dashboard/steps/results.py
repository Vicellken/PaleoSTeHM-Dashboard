"""Step 9 — Post-run results and boundary analysis."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stgp_dashboard.components.figure_display import show_figure_image
from stgp_dashboard.services.artifact_detection import artifacts_from_metrics
from stgp_dashboard.services.boundary_report import (
    boundary_summary_rows,
    suggest_widen_factor,
)
from stgp_dashboard.services.mask_preview import (
    _load_paleo_rsl,
    _spatial_slice_at_age,
    compute_mask_stats,
    mask_settings_from_ui,
    render_mask_preview_figure,
)
from stgp_dashboard.services.open_folder import open_folder
from stgp_dashboard.services.progress import RunProgress, make_progress_callbacks
from stgp_dashboard.services.config_builder import (
    normalize_budget_age_windows,
    save_config,
)
from stgp_dashboard.services.visualize_runner import (
    discover_saved_figures,
    generate_budget_figures,
    generate_post_run_figures,
)
from stgp_dashboard.services.viz_scope import (
    RESULTS_SCOPE_RADIO_KEY,
    SCOPE_LABEL_TO_VALUE,
    SCOPE_LABELS,
    SCOPE_VALUE_TO_LABEL,
    normalize_viz_scope,
)
from stgp_dashboard.services.workflow_labels import resolve_workflow_label
from stgp_dashboard.state import update_state

_VIEW_OPTIONS = [
    "Mask calibration",
    "Temporal decomposition",
    "Sea-level budget",
    "Spatial decomposition",
    "Spatial rates",
    "Optimization",
    "All",
]

_VIEW_TO_CATEGORY = {
    "Temporal decomposition": "temporal_decomposition",
    "Sea-level budget": "sea_level_budget",
    "Spatial decomposition": "spatial_decomposition",
    "Spatial rates": "spatial_rate",
    "Optimization": "optimization",
}

_CATEGORY_LABELS = [
    ("temporal_decomposition", "Temporal decomposition"),
    ("sea_level_budget", "Sea-level budget"),
    ("spatial_decomposition", "Spatial decomposition"),
    ("spatial_rate", "Spatial rates"),
    ("optimization", "Optimization"),
]


def _render_gallery_item(item: dict) -> bool:
    """Show one figure if its PNG exists. Returns True when displayed."""
    png = item.get("png")
    if not png or not Path(png).is_file():
        return False
    show_figure_image(
        png,
        category=item.get("category", "other"),
        caption=item.get("name", Path(png).stem),
        pdf_path=item.get("pdf"),
    )
    return True


def partition_budget_gallery_items(
    items: list,
) -> dict[str, list]:
    """Split budget gallery metadata into all_regions / field / per_region."""
    all_regions: list = []
    field: list = []
    per_region: list = []
    other: list = []
    for item in items:
        name = str(item.get("name") or "")
        stem = Path(str(item.get("png") or name)).stem
        if "all_regions" in stem or "all_regions" in name:
            all_regions.append(item)
        elif stem == "sea_level_budget_field" or name == "sea_level_budget_field":
            field.append(item)
        elif item.get("region") or "_region_" in stem:
            per_region.append(item)
        else:
            other.append(item)
    return {
        "all_regions": all_regions,
        "field": field,
        "per_region": per_region,
        "other": other,
    }


def _render_items_in_expander(
    label: str,
    items: list,
    *,
    expanded: bool,
) -> int:
    """Render gallery items inside an expander. Returns count shown."""
    if not items:
        return 0
    shown = 0
    with st.expander(label, expanded=expanded):
        for item in items:
            if _render_gallery_item(item):
                shown += 1
    return shown


def _filter_gallery_items(
    viz_outputs: list,
    *,
    selected_category: str | None,
    region_filter: str,
    age_choice: str,
) -> list:
    filtered = []
    for item in viz_outputs:
        category = item.get("category", "other")
        if selected_category is not None and category != selected_category:
            continue
        if (
            category in {"temporal_decomposition", "sea_level_budget"}
            and region_filter != "All regions"
            and item.get("region")
            and item.get("region") != region_filter
        ):
            continue
        if category == "spatial_decomposition" and age_choice != "All ages":
            chosen_age = int(age_choice.replace(" BP", ""))
            if item.get("age_bp") != chosen_age:
                continue
        filtered.append(item)
    return filtered


def _persist_budget_settings(
    *,
    config_path: str,
    windows: list[list[float]],
    merge_gia_global: bool,
) -> None:
    """Update session state and rewrite ``budget:`` in the saved run config."""
    update_state(
        budget_age_windows=windows,
        budget_merge_gia_global=merge_gia_global,
    )
    if not config_path:
        return
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        return
    try:
        import yaml

        from stgp_dashboard.services.config_builder import normalize_budget_gmsl_xlsx

        with cfg_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        existing_budget = dict(cfg.get("budget") or {})
        budget_block = {
            "age_windows_bp": windows,
            "merge_gia_global": bool(merge_gia_global),
        }
        # Prefer session-state GMSL paths; else preserve YAML so regen keeps split.
        gmsl = normalize_budget_gmsl_xlsx(st.session_state.get("budget_gmsl_xlsx"))
        if gmsl is None:
            gmsl = normalize_budget_gmsl_xlsx(existing_budget.get("gmsl_xlsx"))
        if gmsl is not None:
            budget_block["gmsl_xlsx"] = gmsl
        cfg["budget"] = budget_block
        save_config(cfg, cfg_path)
    except Exception:  # noqa: BLE001
        pass


def _render_budget_controls(scenario_dir: str, config_path: str) -> None:
    """Results UI: edit budget windows and regenerate without spatial redo."""
    rates_csv = Path(scenario_dir) / "temporal_decomposition_rates.csv"
    budget_csv = Path(scenario_dir) / "sea_level_budget.csv"
    budget_pdf = Path(scenario_dir) / "sea_level_budget.pdf"

    if not rates_csv.is_file():
        st.info(
            "No temporal rates CSV yet. Finish post-processing first "
            "(temporal export must complete before budget)."
        )
        return

    settings_expanded = not budget_csv.is_file()
    with st.expander("Budget settings", expanded=settings_expanded):
        st.caption(
            "Default is one window (0–4000 BP). Percents partition "
            r"$\sum |\mathrm{mean}(\mathrm{rate})| \times \Delta t$ "
            "within each region and window (not shares of total rate)."
        )

        stored = normalize_budget_age_windows(
            st.session_state.get("budget_age_windows") or [[0, 4000]]
        )
        if not stored:
            stored = [[0.0, 4000.0]]

        n_windows = st.number_input(
            "Number of budget windows",
            min_value=1,
            max_value=8,
            value=len(stored),
            key="results_budget_n_windows",
        )
        windows: list[list[float]] = []
        for i in range(int(n_windows)):
            default_lo = float(stored[i][0]) if i < len(stored) else 0.0
            default_hi = float(stored[i][1]) if i < len(stored) else 4000.0
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input(
                    f"Window {i + 1} start (BP)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=default_lo,
                    step=100.0,
                    key=f"results_budget_lo_{i}",
                )
            with c2:
                hi = st.number_input(
                    f"Window {i + 1} end (BP)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=default_hi,
                    step=100.0,
                    key=f"results_budget_hi_{i}",
                )
            windows.append([float(lo), float(hi)])
        windows = normalize_budget_age_windows(windows) or [[0.0, 4000.0]]

        merge = st.checkbox(
            "Merge GIA + Global (Gm) into one budget wedge",
            value=bool(st.session_state.get("budget_merge_gia_global", True)),
            key="results_budget_merge",
        )

        if st.button("Regenerate budget plots", type="primary", key="regen_budget"):
            _persist_budget_settings(
                config_path=config_path,
                windows=windows,
                merge_gia_global=merge,
            )
            with st.spinner("Computing sea-level budget and figures…"):
                try:
                    age_tuples = [(float(w[0]), float(w[1])) for w in windows]
                    generate_budget_figures(
                        scenario_dir,
                        age_windows=age_tuples,
                        merge_gia_global=merge,
                        recompute=True,
                    )
                    figs = discover_saved_figures(scenario_dir)
                    update_state(visualization_outputs=figs)
                    st.success(
                        f"Updated sea-level budget for "
                        f"{len(windows)} window(s) in `{scenario_dir}`."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Budget generation failed: {exc}")

        if budget_csv.is_file() or budget_pdf.is_file():
            st.caption(
                "Written: "
                + ", ".join(
                    f"`{p.name}`"
                    for p in (budget_csv, budget_pdf)
                    if p.is_file()
                )
            )


def _workflow_display_label(metrics: dict) -> str:
    if metrics.get("workflow_label"):
        return str(metrics["workflow_label"])
    workflow_id = metrics.get("workflow_id", metrics.get("workflow", ""))
    cfg = st.session_state.get("saved_config_path")
    if cfg:
        try:
            from stgp_dashboard.services.config_loader import load_config_file

            state = load_config_file(cfg)
            from stgp_dashboard.services.config_builder import build_config

            return resolve_workflow_label(build_config(state), str(workflow_id))
        except Exception:  # noqa: BLE001
            pass
    mapping = {
        "16ka": "No GIA prior",
        "4ka": "GIA prior (1D ICE6G)",
        "3d-gia": "GIA prior (3D combined)",
        "regional-gia": "GIA prior (Regional)",
    }
    return mapping.get(str(workflow_id), str(workflow_id) or "—")


def _has_boundary_hits(boundary_df) -> bool:
    if boundary_df is None or getattr(boundary_df, "empty", True):
        return False
    if "boundary_hit" not in boundary_df.columns:
        return False
    return bool(boundary_df["boundary_hit"].astype(bool).any())


def _try_reuse_existing_artifacts(metrics: dict) -> bool:
    """
    If reusable prediction exports exist, mark post-process complete and seed
    the figure gallery. Returns True when reuse succeeded.
    """
    detected = artifacts_from_metrics(metrics)
    if not detected.get("reusable"):
        return False

    scenario_dir = metrics.get("scenario_dir") or ""
    if not scenario_dir and metrics.get("output_dir"):
        scenario_dir = str(Path(str(metrics["output_dir"])) / "run")

    merged = dict(metrics)
    merged.update(detected.get("paths") or {})
    merged["postprocess_complete"] = True

    fig_paths = st.session_state.get("visualization_outputs") or []
    if scenario_dir and not fig_paths:
        fig_paths = discover_saved_figures(scenario_dir)

    note = detected.get("note")
    msg = f"Loaded existing prediction outputs from `{scenario_dir}`."
    if note:
        msg += f" {note}"

    update_state(
        run_metrics=merged,
        postprocess_complete=True,
        visualization_outputs=fig_paths or None,
        postprocess_reuse_message=msg,
    )
    return True


def _run_postprocess_and_visualize(
    *,
    metrics: dict,
    config_path: str,
    mask_enabled: bool,
    mask_factor: float,
    force_recompute: bool = True,
    scope: str = "full",
) -> None:
    from pathlib import Path as _Path

    import pandas as pd
    import yaml

    from stgp_dashboard.paths import repo_root
    from stgp_pipeline.run import run_stgp_postprocess

    output_dir = _Path(metrics.get("output_dir") or "")
    if not output_dir.is_dir():
        st.error("Output directory not found; re-run training from Review & run.")
        return

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    progress = RunProgress()
    stage_box = st.empty()
    progress_bar = st.progress(0.0, text="Starting post-processing…")
    metrics_box = st.empty()
    _, orig_on_stage = make_progress_callbacks(progress)

    def on_stage(stage_id: str, message: str) -> None:
        orig_on_stage(stage_id, message)
        stage_box.info(f"**Stage:** {message}")
        progress_bar.progress(1.0, text=message)
        metrics_box.markdown(
            f"**{progress.status_label()}** — {progress.format_status()}"
        )

    param_store = metrics.get("param_store_path")
    try:
        new_metrics = run_stgp_postprocess(
            cfg,
            output_dir,
            param_store_path=param_store,
            base_dir=repo_root(),
            stage_callback=on_stage,
            force_recompute=force_recompute,
        )
        # Preserve training-time fields that postprocess may overwrite
        # (checkpoint reload sets run_optimization=False).
        orig_run_opt = metrics.get("run_optimization")
        merged = dict(metrics)
        merged.update(new_metrics)
        if orig_run_opt is not None:
            merged["run_optimization"] = orig_run_opt
        merged["postprocess_complete"] = True

        scenario_dir = _Path(merged.get("scenario_dir", output_dir / "run"))
        boundary_df = st.session_state.get("boundary_df")
        bp = scenario_dir / "boundary_proximity.csv"
        if bp.is_file():
            boundary_df = pd.read_csv(bp)
        track_df = st.session_state.get("optimization_track")
        tp = scenario_dir / "optimization_track.csv"
        if tp.is_file():
            track_df = pd.read_csv(tp)
        log_path = output_dir / "run.log"
        run_log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""

        stage_box.info("**Stage:** Generating figures…")
        fig_paths = generate_post_run_figures(
            scenario_dir,
            metrics=merged,
            mask_enabled=mask_enabled,
            mask_calibration_factor=mask_factor,
            scope=scope,
        )
        update_state(
            run_metrics=merged,
            boundary_df=boundary_df,
            optimization_track=track_df,
            run_log=run_log,
            postprocess_complete=True,
            visualization_outputs=fig_paths,
            viz_scope=scope,
            viz_view_choice="All",
        )
        st.success(
            f"Post-processing complete. Saved {len(fig_paths)} figure(s) to "
            f"`{scenario_dir}/figures/`."
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Post-processing failed: {exc}")


def render() -> None:
    st.header("Results")

    metrics = st.session_state.get("run_metrics") or {}
    postprocess_complete = bool(
        st.session_state.get("postprocess_complete")
        or metrics.get("postprocess_complete")
    )
    reuse_msg = st.session_state.pop("postprocess_reuse_message", None)
    if reuse_msg:
        st.info(reuse_msg)
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Analysis mode", _workflow_display_label(metrics))
        c2.metric("Training points", metrics.get("n_training_points", "—"))
        c3.metric("NLL", f"{metrics.get('negative_log_likelihood', float('nan')):.2f}")
        if metrics.get("param_store_path"):
            st.caption(f"Model checkpoint: `{metrics['param_store_path']}`")
        if metrics.get("run_optimization") is False and not postprocess_complete:
            st.info("This run loaded a saved checkpoint and skipped SVI training.")

    config_path = st.session_state.get("saved_config_path", "")
    if config_path:
        st.write(f"**Config saved:** `{config_path}`")

    scenario_dir = metrics.get("scenario_dir", "")
    output_dir = metrics.get("output_dir", "")
    if scenario_dir or output_dir:
        folder = output_dir or scenario_dir
        if st.button("Open output folder", key="open_results_folder"):
            try:
                open_folder(folder)
                st.toast("Opened output folder.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not open folder: {exc}")

    # --- Prior range analysis (always first after training) ---
    boundary_df = st.session_state.get("boundary_df")
    if boundary_df is not None and not boundary_df.empty:
        st.subheader("Prior range analysis")
        rows = boundary_summary_rows(boundary_df)
        for row in rows:
            if row["status"] == "hit":
                st.error(
                    f"**{row['parameter']}** — {row['proximity_pct']:.1f}% within bounds "
                    f"({row['boundary']} hit). {row['suggestion']}"
                )
            elif row["status"] == "warn":
                st.warning(
                    f"**{row['parameter']}** — {row['proximity_pct']:.1f}% within bounds "
                    "(near edge)."
                )
            else:
                st.success(
                    f"**{row['parameter']}** — {row['proximity_pct']:.1f}% within bounds."
                )

        factor = suggest_widen_factor(rows)
        if factor > 1.0:
            if st.button(f"Pre-fill priors widened ×{factor} and edit", key="widen_priors"):
                update_state(step=6, mode="wizard")
                st.info(
                    "Go to Kernel priors step and increase spatial upper bounds, "
                    f"or run sensitivity with bounds ×{factor}."
                )
                st.rerun()
    else:
        st.info("No boundary proximity report available yet.")

    # --- Gate: proceed to post-processing ---
    if metrics and not postprocess_complete:
        # Auto-reuse existing prediction exports when fresh relative to checkpoint.
        if _try_reuse_existing_artifacts(metrics):
            metrics = st.session_state.get("run_metrics") or metrics
            postprocess_complete = True
            scenario_dir = metrics.get("scenario_dir", scenario_dir)
            reuse_msg = st.session_state.pop("postprocess_reuse_message", None)
            if reuse_msg:
                st.info(reuse_msg)
        else:
            detected = artifacts_from_metrics(metrics)
            st.subheader("Post-processing")
            st.markdown(
                "Training and prior-range checks are done. Proceed only after you are "
                "satisfied that kernel hyperparameters are statistically identifiable "
                "(interior to prior bounds)."
            )
            if detected.get("stale_reason"):
                st.caption(detected["stale_reason"])
            if _has_boundary_hits(boundary_df):
                st.warning(
                    "One or more hyperparameters hit prior bounds. Consider widening "
                    "priors (above) or running the sensitivity suite before proceeding. "
                    "You may still continue if you accept the current fit."
                )
            if config_path and st.button(
                "Proceed with post-processing & visualization",
                type="primary",
                key="proceed_postprocess",
            ):
                mask_enabled = bool(st.session_state.get("spatial_mask_enabled", True))
                mask_factor = float(
                    st.session_state.get("spatial_mask_calibration_factor", 2.0)
                )
                with st.spinner("Running temporal/spatial post-processing and figures…"):
                    _run_postprocess_and_visualize(
                        metrics=metrics,
                        config_path=config_path,
                        mask_enabled=mask_enabled,
                        mask_factor=mask_factor,
                        # Reuse fresh prediction artifacts when present; explicit
                        # "Recompute predictions" still forces a full recompute.
                        force_recompute=False,
                        scope=str(st.session_state.get("viz_scope", "full") or "full"),
                    )

            track = st.session_state.get("optimization_track")
            if track is not None and not track.empty:
                with st.expander("Optimization loss"):
                    if "loss" in track.columns:
                        st.line_chart(track["loss"], height=200)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("← Edit & re-run", key="results_edit_pre"):
                    update_state(mode="wizard", step=8)
                    st.rerun()
            with c2:
                if st.button("Open sensitivity suite →", key="results_sensitivity_pre"):
                    update_state(mode="sensitivity", step=10)
                    st.rerun()
            with c3:
                if st.button("New analysis", key="results_new_pre"):
                    update_state(mode="wizard", step=0)
                    st.rerun()
            return

    # --- Visualization (only after post-processing) ---
    st.subheader("Visualization")

    default_scope = normalize_viz_scope(st.session_state.get("viz_scope", "full"))
    scope_label = st.radio(
        "Visualization scope",
        options=SCOPE_LABELS,
        index=SCOPE_LABELS.index(SCOPE_VALUE_TO_LABEL[default_scope]),
        horizontal=True,
        key=RESULTS_SCOPE_RADIO_KEY,
        help=(
            "Synced with Analysis period. Full analysis builds the complete "
            "figure suite. RSL curve only generates Total RSL (one panel per "
            "region). Change scope, then click Regenerate figures."
        ),
    )
    viz_scope = SCOPE_LABEL_TO_VALUE[scope_label]
    update_state(viz_scope=viz_scope)
    rsl_only = viz_scope == "rsl_curve"
    if rsl_only:
        st.caption(
            "RSL curve only — Total RSL per region. Decomposition, budget, and "
            "spatial figures are skipped. Scope can also be set on **Analysis "
            "period**. Click **Regenerate figures** after changing scope."
        )
    else:
        st.caption(
            "Full analysis — temporal/spatial decomposition, sea-level budget, "
            "and rates. Scope can also be set on **Analysis period**. Click "
            "**Regenerate figures** after changing scope."
        )

    mask_enabled = bool(st.session_state.get("spatial_mask_enabled", True))
    mask_factor = float(st.session_state.get("spatial_mask_calibration_factor", 2.0))
    if not rsl_only:
        mask_enabled = st.toggle(
            "Apply SD mask on spatial mean/rate maps",
            value=mask_enabled,
            key="results_mask_enabled",
            help="Visualization only — exported CSV/NetCDF values are unchanged.",
        )
        mask_factor = st.slider(
            "Mask calibration factor",
            min_value=0.5,
            max_value=3.0,
            value=mask_factor,
            step=0.25,
            key="results_mask_factor",
            help=(
                "Threshold τ = factor × median(SD at SLIP nearest grid cells). "
                "Lower values mask more aggressively."
            ),
        )
        update_state(
            spatial_mask_enabled=bool(mask_enabled),
            spatial_mask_calibration_factor=float(mask_factor),
        )

        with st.expander("What does spatial plot masking mean?", expanded=False):
            st.markdown(
                """
                **Purpose:** Hide spatial map pixels where the posterior uncertainty is high
                relative to data-supported regions. This affects **figures only**, not the
                underlying predictions saved to CSV or NetCDF.

                **Rule:** For each map panel, compute
                **τ = calibration factor × median(SD at nearest grid cell to each SLIP)**.
                Pixels where **SD > τ** are masked (shown as empty).

                **Choosing a factor:**
                - **2.0** (default) — balanced masking for most regional studies.
                - **1.5** — stricter; useful when far-field extrapolation looks too smooth.
                - **1.25** — very strict; try when distant ocean areas still show structure.

                Component **std** panels are never masked; only mean and rate maps use this rule.
                """
            )

    mask_settings = mask_settings_from_ui(
        enabled=mask_enabled,
        calibration_factor=mask_factor,
    )

    if scenario_dir:
        scenario_path = Path(scenario_dir)

    # Reload gallery from disk when session state is empty (e.g. after refresh).
    viz_outputs = st.session_state.get("visualization_outputs") or []
    if scenario_dir and not viz_outputs:
        discovered = discover_saved_figures(scenario_dir)
        if discovered:
            update_state(visualization_outputs=discovered)
            viz_outputs = discovered

    if rsl_only:
        view_choice = "RSL curve"
    else:
        default_view = st.session_state.get("viz_view_choice") or "All"
        if default_view not in _VIEW_OPTIONS:
            default_view = "All"
        view_choice = st.selectbox(
            "View",
            options=_VIEW_OPTIONS,
            index=_VIEW_OPTIONS.index(default_view),
            key="viz_view_select",
            help="Filter which outputs appear below. Choose Temporal / Spatial / All "
            "to see decomposition plots; Mask calibration is for SD threshold tuning.",
        )
        update_state(viz_view_choice=view_choice)
        st.caption(
            "Use **View** to filter plot types. Expand a section below to view plots."
        )

    if not rsl_only and view_choice == "Sea-level budget" and scenario_dir:
        _render_budget_controls(str(scenario_dir), config_path)

    show_mask = (not rsl_only) and view_choice == "Mask calibration"
    if show_mask and scenario_dir:
        scenario_path = Path(scenario_dir)
        cfg_for_preview = {}
        if config_path:
            try:
                from stgp_dashboard.services.config_loader import load_config_file

                cfg_state = load_config_file(config_path)
                from stgp_dashboard.services.config_builder import build_config

                cfg_for_preview = build_config(cfg_state)
            except Exception:  # noqa: BLE001
                pass

        spatial_csv = scenario_path / "spatial_predictions.csv"
        if spatial_csv.is_file():
            import pandas as pd

            df_spatial = pd.read_csv(spatial_csv)
            preview_ages = sorted(
                {int(round(float(a))) for a in df_spatial["age_bp"].unique()}
            )
            if preview_ages:
                preview_age = st.selectbox(
                    "Mask preview age (BP)",
                    options=preview_ages,
                    format_func=lambda a: f"{a} BP",
                    key="mask_preview_age",
                )
                spatial_slice = _spatial_slice_at_age(scenario_path, float(preview_age))
                rsl_data = _load_paleo_rsl(cfg_for_preview) if cfg_for_preview else None
                if spatial_slice is not None:
                    stats = compute_mask_stats(spatial_slice, rsl_data, mask_settings)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Threshold τ (m)", f"{stats['threshold_m']:.3g}")
                    c2.metric(
                        "Median slip SD (m)",
                        f"{stats['median_slip_sd_m']:.3g}"
                        if stats["median_slip_sd_m"] == stats["median_slip_sd_m"]
                        else "—",
                    )
                    c3.metric("Grid cells masked", f"{stats['masked_fraction_pct']:.1f}%")
                    preview_fig = render_mask_preview_figure(
                        spatial_slice, rsl_data, mask_settings
                    )
                    st.pyplot(preview_fig, clear_figure=True)
        elif view_choice == "Mask calibration":
            st.info("No spatial_predictions.csv yet — mask preview unavailable.")

    if rsl_only:
        st.info(
            "RSL curve generation is usually quick. **Recompute predictions** still "
            "exports full temporal/spatial CSVs/NetCDF; only the figure suite is scoped."
        )
    else:
        st.warning(
            "Re-generating figures can take several minutes for long analysis windows "
            "and large spatial grids, and may require substantial RAM. "
            "**Recompute predictions** re-runs temporal/spatial exports (including "
            "spatial rates) and overwrites CSVs/NetCDF; use only when you need fresh "
            "prediction files."
        )
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        regenerate_clicked = scenario_dir and st.button(
            "Regenerate figures", key="visualize_results"
        )
    with btn_c2:
        recompute_clicked = bool(
            scenario_dir
            and config_path
            and st.button(
                "Recompute predictions",
                key="recompute_predictions",
                help=(
                    "Force re-run of temporal/spatial post-processing, then "
                    "regenerate figures. Existing CSVs/NetCDF will be overwritten."
                ),
            )
        )

    if recompute_clicked:
        with st.spinner("Recomputing predictions and figures…"):
            _run_postprocess_and_visualize(
                metrics=metrics,
                config_path=config_path,
                mask_enabled=mask_enabled,
                mask_factor=mask_factor,
                force_recompute=True,
                scope=viz_scope,
            )

    if regenerate_clicked:
        with st.spinner("Generating figures (PNG + PDF)…"):
            try:
                fig_paths = generate_post_run_figures(
                    scenario_dir,
                    metrics=metrics,
                    mask_enabled=mask_enabled,
                    mask_calibration_factor=mask_factor,
                    scope=viz_scope,
                )
                update_state(visualization_outputs=fig_paths, viz_scope=viz_scope)
                counts: dict[str, int] = {}
                for item in fig_paths:
                    cat = item.get("category", "other")
                    counts[cat] = counts.get(cat, 0) + 1
                summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                st.success(
                    f"Saved {len(fig_paths)} figure(s) to `{scenario_dir}/figures/`. "
                    f"({summary})"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Visualization failed: {exc}")

    viz_outputs = st.session_state.get("visualization_outputs") or []
    if rsl_only:
        viz_outputs = [
            item for item in viz_outputs if item.get("category") == "rsl_curve"
        ]
    # Gallery shows saved figures for non-mask views (and All).
    show_gallery = view_choice != "Mask calibration"
    if show_gallery and not viz_outputs:
        if rsl_only:
            st.info(
                "No RSL curve figure yet. Click **Regenerate figures** with "
                "**RSL curve only** selected. Requires temporal_decomposition.csv "
                "and a valid paleo RSL path in the saved config."
            )
        else:
            category_hint = _VIEW_TO_CATEGORY.get(view_choice)
            if view_choice == "All":
                st.info(
                    "No saved figures found yet. Click **Regenerate figures** to build "
                    "temporal/spatial decompositions and the optimization track."
                )
            elif category_hint:
                st.info(
                    f"No figures for **{view_choice}** yet. Click **Regenerate figures**, "
                    "or switch View to All / Mask calibration. Temporal plots also need a "
                    "valid paleo RSL path in the saved config."
                )
    if viz_outputs and show_gallery:
        if rsl_only:
            selected_category = "rsl_curve"
        else:
            selected_category = _VIEW_TO_CATEGORY.get(view_choice)
            if view_choice == "All":
                selected_category = None  # show all categories

        region_filter = "All regions"
        age_choice = "All ages"
        if not rsl_only:
            region_items = [
                item
                for item in viz_outputs
                if item.get("category")
                in {"temporal_decomposition", "sea_level_budget"}
                and item.get("region")
            ]
            if region_items and (
                selected_category is None
                or selected_category
                in {"temporal_decomposition", "sea_level_budget"}
            ):
                region_names = sorted(
                    {
                        str(item.get("region", ""))
                        for item in region_items
                        if item.get("region")
                    },
                    key=str.casefold,
                )
                if region_names:
                    # Default to first region when multiple exist (fewer wide figures).
                    default_region = (
                        region_names[0] if len(region_names) > 1 else "All regions"
                    )
                    region_options = ["All regions"] + region_names
                    region_label = (
                        "Region (budget)"
                        if selected_category == "sea_level_budget"
                        else "Region (temporal / budget)"
                    )
                    region_filter = st.selectbox(
                        region_label,
                        options=region_options,
                        index=region_options.index(default_region),
                        key="viz_region_filter",
                    )

            spatial_items = [
                item
                for item in viz_outputs
                if item.get("category") == "spatial_decomposition"
            ]
            if spatial_items and (
                selected_category is None
                or selected_category == "spatial_decomposition"
            ):
                age_options = sorted(
                    {
                        int(item["age_bp"])
                        for item in spatial_items
                        if item.get("age_bp") is not None
                    }
                )
                if age_options:
                    # Default to youngest age (min BP) so only one spatial figure loads.
                    youngest = age_options[0]
                    age_labels = ["All ages"] + [f"{age} BP" for age in age_options]
                    default_age_label = f"{youngest} BP"
                    age_choice = st.selectbox(
                        "Spatial snapshot age",
                        options=age_labels,
                        index=age_labels.index(default_age_label),
                        key="viz_age_filter",
                    )

        filtered = _filter_gallery_items(
            viz_outputs,
            selected_category=selected_category,
            region_filter=region_filter,
            age_choice=age_choice,
        )

        shown = 0
        if rsl_only:
            n = len(filtered)
            shown += _render_items_in_expander(
                f"RSL curve ({n})",
                filtered,
                expanded=True,
            )
        elif view_choice == "All":
            for cat_id, cat_label in _CATEGORY_LABELS:
                cat_items = [i for i in filtered if i.get("category") == cat_id]
                if not cat_items:
                    continue
                if cat_id == "sea_level_budget":
                    parts = partition_budget_gallery_items(cat_items)
                    shown += _render_items_in_expander(
                        f"Sea-level budget — all-regions grids ({len(parts['all_regions'])})",
                        parts["all_regions"],
                        expanded=False,
                    )
                    shown += _render_items_in_expander(
                        f"Sea-level budget — field-wide ({len(parts['field'])})",
                        parts["field"],
                        expanded=False,
                    )
                    shown += _render_items_in_expander(
                        f"Sea-level budget — per-region ({len(parts['per_region'])})",
                        parts["per_region"],
                        expanded=False,
                    )
                    shown += _render_items_in_expander(
                        f"Sea-level budget — other ({len(parts['other'])})",
                        parts["other"],
                        expanded=False,
                    )
                else:
                    shown += _render_items_in_expander(
                        f"{cat_label} ({len(cat_items)})",
                        cat_items,
                        expanded=False,
                    )
            other_items = [
                i
                for i in filtered
                if i.get("category") not in {c for c, _ in _CATEGORY_LABELS}
            ]
            shown += _render_items_in_expander(
                f"Other ({len(other_items)})",
                other_items,
                expanded=False,
            )
        elif view_choice == "Sea-level budget":
            parts = partition_budget_gallery_items(filtered)
            shown += _render_items_in_expander(
                f"All-regions grids ({len(parts['all_regions'])})",
                parts["all_regions"],
                expanded=len(parts["all_regions"]) <= 2 and bool(parts["all_regions"]),
            )
            shown += _render_items_in_expander(
                f"Field-wide ({len(parts['field'])})",
                parts["field"],
                expanded=False,
            )
            shown += _render_items_in_expander(
                f"Per-region donuts ({len(parts['per_region'])})",
                parts["per_region"],
                expanded=False,
            )
            shown += _render_items_in_expander(
                f"Other budget figures ({len(parts['other'])})",
                parts["other"],
                expanded=False,
            )
        else:
            n = len(filtered)
            shown += _render_items_in_expander(
                f"Figures ({n})",
                filtered,
                expanded=n <= 3,
            )

        if shown == 0 and selected_category is not None:
            st.info(
                f"No figures matched **{view_choice}** with the current filters. "
                "Try All regions / All ages, or Regenerate figures."
            )

    track = st.session_state.get("optimization_track")
    if track is not None and not track.empty:
        with st.expander("Optimization loss (post-run)"):
            if "loss" in track.columns:
                st.line_chart(track["loss"], height=200)

    if scenario_dir:
        with st.expander("Output files", expanded=False):
            out = Path(scenario_dir)
            found = False
            for pattern in ("*.csv", "*.nc", "figures/*.png", "figures/*.pdf"):
                for f in sorted(out.glob(pattern)):
                    st.write(f"`{f}`")
                    found = True
            if not found:
                st.caption("No output files found yet.")

    run_log = st.session_state.get("run_log", "")
    if run_log:
        with st.expander("Run log"):
            st.code(run_log[-8000:], language="text")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Edit & re-run", key="results_edit"):
            update_state(mode="wizard", step=8)
            st.rerun()
    with c2:
        if st.button("Open sensitivity suite →", key="results_sensitivity"):
            update_state(mode="sensitivity", step=10)
            st.rerun()
    with c3:
        if st.button("New analysis", key="results_new"):
            update_state(mode="wizard", step=0)
            st.rerun()
