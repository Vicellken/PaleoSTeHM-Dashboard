"""Step 5 — Advanced data filtering."""

from __future__ import annotations

import streamlit as st

from stgp_dashboard.state import go_to_step, update_state


def render() -> None:
    st.header("Data filtering")
    st.warning("Advanced options — defaults are fine for a first run.")

    levels = st.multiselect(
        "Include data levels",
        options=[1, 2, 3],
        default=st.session_state.get("filter_levels", [1, 2]),
        help=(
            "Optional HOLSEA quality-level filter (applied only if your CSV has a "
            "'Level' column; e.g., Level 1 = high quality, Level 2 = lower quality, "
            "Level 3 = supplementary). SLIP vs. limiting indicator type is controlled "
            "separately by the 'Limiting' column (0 = SLIP, -1 = marine, 1 = terrestrial)."
        ),
    )

    l2_enabled = bool(st.session_state.get("l2_downweight_enabled", False))
    l2_mult = float(st.session_state.get("l2_downweight_multiplier", 2.0))
    l2_scope = st.session_state.get("l2_downweight_scope", "slip_only")

    with st.expander("Level 2 downweighting"):
        l2_enabled = st.checkbox(
            "Enable L2 uncertainty inflation",
            value=l2_enabled,
        )
        l2_mult = st.number_input(
            "Multiplier",
            min_value=1.0,
            max_value=10.0,
            value=l2_mult,
        )
        l2_scope = st.selectbox(
            "Scope",
            options=["slip_only", "all"],
            index=0 if l2_scope != "all" else 1,
        )

    region_options = _region_options_from_preview()
    merges = list(st.session_state.get("region_merges") or [])
    st.markdown("### Region merges")
    st.caption("Combine small or duplicate regions by merging one into another.")

    if not region_options:
        st.info(
            "Preview your dataset on the **Data & paths** step to see region IDs here."
        )
    else:
        st.write("Detected regions:", _format_region_list(region_options))

    n_rows = max(len(merges), int(st.session_state.get("region_merge_rows", 0)))
    if n_rows == 0 and region_options:
        n_rows = 0

    new_merges: list[list[int]] = []
    if region_options:
        if st.button("Add merge row", key="add_merge_row"):
            update_state(region_merge_rows=max(n_rows, len(merges)) + 1)
            st.rerun()

        row_count = max(
            int(st.session_state.get("region_merge_rows", 0)),
            len(merges),
        )
        option_ids = [opt["id"] for opt in region_options]
        labels = {opt["id"]: opt["label"] for opt in region_options}

        for i in range(row_count):
            default_source, default_target = (option_ids[0], option_ids[0])
            if i < len(merges):
                default_source = int(merges[i][0])
                default_target = int(merges[i][1])

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                source = st.selectbox(
                    f"Source region #{i + 1}",
                    options=option_ids,
                    index=option_ids.index(default_source)
                    if default_source in option_ids
                    else 0,
                    format_func=lambda rid: labels.get(rid, str(rid)),
                    key=f"merge_source_{i}",
                )
            with c2:
                target = st.selectbox(
                    f"Merge into (target) #{i + 1}",
                    options=option_ids,
                    index=option_ids.index(default_target)
                    if default_target in option_ids
                    else 0,
                    format_func=lambda rid: labels.get(rid, str(rid)),
                    key=f"merge_target_{i}",
                )
            with c3:
                keep = st.checkbox("Keep", value=True, key=f"merge_keep_{i}")
            if keep and source != target:
                new_merges.append([int(source), int(target)])

    centroid = st.selectbox(
        "Temporal centroid mode",
        options=["data_mean", "bbox_center", "grid_mean"],
        index=["data_mean", "bbox_center", "grid_mean"].index(
            st.session_state.get("temporal_centroid_mode", "data_mean")
        ),
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="filter_back"):
            go_to_step(4)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="filter_next"):
            update_state(
                filter_levels=levels or [1, 2],
                l2_downweight_enabled=l2_enabled,
                l2_downweight_multiplier=float(l2_mult),
                l2_downweight_scope=l2_scope,
                region_merges=new_merges,
                region_merge_rows=len(new_merges),
                temporal_centroid_mode=centroid,
            )
            go_to_step(6)
            st.rerun()


def _region_options_from_preview() -> list[dict]:
    preview = st.session_state.get("data_preview") or {}
    options = preview.get("region_options")
    if options:
        return list(options)
    return []


def _format_region_list(options: list[dict]) -> str:
    return ", ".join(opt["label"] for opt in options[:20])
