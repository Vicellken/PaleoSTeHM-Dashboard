"""Step 2 — Analysis period (user-defined, not 4ka/16ka presets)."""

from __future__ import annotations

import streamlit as st

from stgp_dashboard.services.age_bounds import (
    age_quality_warnings,
    default_max_age_from_preview,
    resolve_age_widget_bounds,
)
from stgp_dashboard.services.config_builder import normalize_budget_age_windows
from stgp_dashboard.services.viz_scope import (
    PERIOD_SCOPE_RADIO_KEY,
    SCOPE_LABEL_TO_VALUE,
    SCOPE_LABELS,
    SCOPE_VALUE_TO_LABEL,
    is_rsl_curve_only,
    normalize_viz_scope,
    requires_budget_windows,
)
from stgp_dashboard.state import go_to_step, update_state


def _render_budget_windows_editor() -> tuple[list[list[float]], bool]:
    """Editable budget age windows (summary periods; not clamped to viz span)."""
    stored = normalize_budget_age_windows(
        st.session_state.get("budget_age_windows") or [[0, 4000]]
    )
    if not stored:
        stored = [[0.0, 4000.0]]

    with st.expander(
        "Sea-level budget windows (default 0–4 ka)",
        expanded=True,
    ):
        st.caption(
            "Default is one window (0–4000 BP). Add more only if needed. "
            "These are summary periods for percent-contribution donuts after "
            "temporal rates are exported — independent of data cutoff / "
            "visualization span. You can change them again on Results."
        )

        n_windows = st.number_input(
            "Number of budget windows",
            min_value=1,
            max_value=8,
            value=len(stored),
            key="budget_n_windows",
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
                    key=f"budget_win_lo_{i}",
                )
            with c2:
                hi = st.number_input(
                    f"Window {i + 1} end (BP)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=default_hi,
                    step=100.0,
                    key=f"budget_win_hi_{i}",
                )
            windows.append([float(lo), float(hi)])

        merge = st.checkbox(
            "Merge GIA + Global (Gm) into one budget wedge",
            value=bool(st.session_state.get("budget_merge_gia_global", True)),
            key="budget_merge_gia_global_widget",
            help=(
                "When on (default), GIA prior and the global temporal kernel share "
                "one wedge labeled GIA + Global (Gm)."
            ),
        )
        return normalize_budget_age_windows(windows) or [[0.0, 4000.0]], bool(merge)


def render() -> None:
    st.header("Analysis period")
    st.markdown(
        "Define the age window for **data filtering** and **visualization**. "
        "These settings replace legacy 4ka/16ka workflow labels."
    )

    default_scope = normalize_viz_scope(st.session_state.get("viz_scope", "full"))
    scope_label = st.radio(
        "Visualization scope",
        options=SCOPE_LABELS,
        index=SCOPE_LABELS.index(SCOPE_VALUE_TO_LABEL[default_scope]),
        horizontal=True,
        key=PERIOD_SCOPE_RADIO_KEY,
        help=(
            "Full analysis keeps the complete wizard and figure suite. "
            "RSL curve only focuses on Total RSL curves and skips Spatial grid "
            "(budget/spatial map settings are hidden here)."
        ),
    )
    viz_scope = SCOPE_LABEL_TO_VALUE[scope_label]
    update_state(viz_scope=viz_scope)
    rsl_only = is_rsl_curve_only(viz_scope)
    if rsl_only:
        st.caption(
            "RSL curve only — Spatial grid will be skipped later; budget and "
            "spatial/rate intervals are not needed. GIA, Training, Filtering, "
            "and Priors remain available."
        )
    else:
        st.caption(
            "Full analysis — configure budget windows and spatial/rate intervals "
            "below. You can still change scope on Results before regenerating figures."
        )

    preview = st.session_state.get("data_preview") or {}
    for warning in age_quality_warnings(preview):
        st.warning(warning)

    stored_max = int(st.session_state.get("max_age_bp", 16000))
    min_bp, max_bp, clamped_max = resolve_age_widget_bounds(preview, stored_max)
    stored_viz_end = int(st.session_state.get("viz_end_bp", clamped_max))
    if stored_max != clamped_max or stored_viz_end > clamped_max:
        update_state(
            max_age_bp=clamped_max,
            viz_end_bp=min(stored_viz_end, clamped_max),
        )
        stored_max = clamped_max

    if "age_min_bp" in preview:
        st.info(
            f"Dataset covers **{preview['age_min_bp']:.0f} – {preview['age_max_bp']:.0f}** cal yr BP."
        )
        if st.button("Use full dataset range", key="use_full_range"):
            full_max = default_max_age_from_preview(preview, int(preview["age_max_bp"]))
            update_state(
                max_age_bp=full_max,
                viz_start_bp=int(max(0, preview["age_min_bp"])),
                viz_end_bp=full_max,
            )
            st.rerun()

    max_age = st.number_input(
        "Data cutoff (max age BP)",
        min_value=min_bp,
        max_value=max_bp,
        value=stored_max,
        help="Rows with Age > this value are excluded (analysis.max_age_bp).",
    )
    c1, c2 = st.columns(2)
    with c1:
        viz_start = st.number_input(
            "Visualization start (BP)",
            min_value=0,
            max_value=int(max_age),
            value=min(int(st.session_state.get("viz_start_bp", 0)), int(max_age)),
        )
    with c2:
        viz_end = st.number_input(
            "Visualization end (BP)",
            min_value=int(viz_start),
            max_value=int(max_age),
            value=min(int(st.session_state.get("viz_end_bp", max_age)), int(max_age)),
        )

    st.subheader("Output grid intervals")
    # Preserve session defaults when RSL-only hides spatial/rate widgets.
    spatial_int = int(st.session_state.get("spatial_interval", 1000))
    rate_int = int(st.session_state.get("rate_interval", 100))
    if rsl_only:
        temporal_int = st.number_input(
            "Temporal RSL grid (yr)",
            min_value=50,
            max_value=2000,
            value=int(st.session_state.get("temporal_interval", 200)),
            help="Age spacing for Total RSL curves.",
        )
    else:
        c3, c4, c5 = st.columns(3)
        with c3:
            temporal_int = st.number_input(
                "Temporal decomposition (yr)",
                min_value=50,
                max_value=2000,
                value=int(st.session_state.get("temporal_interval", 200)),
            )
        with c4:
            spatial_int = st.number_input(
                "Spatial map snapshots (yr)",
                min_value=100,
                max_value=5000,
                value=spatial_int,
            )
        with c5:
            rate_int = st.number_input(
                "Rate estimates (yr)",
                min_value=50,
                max_value=1000,
                value=rate_int,
            )

    budget_windows = normalize_budget_age_windows(
        st.session_state.get("budget_age_windows") or [[0, 4000]]
    ) or [[0.0, 4000.0]]
    budget_merge = bool(st.session_state.get("budget_merge_gia_global", True))
    if not rsl_only:
        budget_windows, budget_merge = _render_budget_windows_editor()

    if viz_end > max_age:
        st.error(f"Visualization end ({viz_end}) must be ≤ data cutoff ({max_age}).")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="period_back"):
            go_to_step(1)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="period_next"):
            if viz_end > max_age:
                st.error("Fix visualization end before continuing.")
            elif requires_budget_windows(viz_scope) and not budget_windows:
                st.error("Add at least one budget window with start < end.")
            else:
                update_state(
                    viz_scope=viz_scope,
                    max_age_bp=int(max_age),
                    viz_start_bp=int(viz_start),
                    viz_end_bp=int(viz_end),
                    temporal_interval=int(temporal_int),
                    spatial_interval=int(spatial_int),
                    rate_interval=int(rate_int),
                    budget_age_windows=budget_windows,
                    budget_merge_gia_global=budget_merge,
                )
                go_to_step(3)
                st.rerun()
