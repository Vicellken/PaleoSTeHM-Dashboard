"""Step 7 — Spatial grid and visualization settings."""

from __future__ import annotations

import streamlit as st

from stgp_dashboard.services.viz_scope import (
    STEP_REVIEW,
    should_skip_grids_step,
)
from stgp_dashboard.state import go_to_step, update_state


def render() -> None:
    if should_skip_grids_step(st.session_state.get("viz_scope", "full")):
        # Light RSL-only path: Spatial grid is unused; keep YAML defaults.
        go_to_step(STEP_REVIEW)
        st.rerun()
        return

    st.header("Spatial grid & outputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        lat_pad = st.number_input(
            "Lat padding (°)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.get("lat_padding", 2.0)),
        )
    with c2:
        lon_pad = st.number_input(
            "Lon padding (°)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.get("lon_padding", 2.0)),
        )
    with c3:
        resolution = st.number_input(
            "Grid resolution (°)",
            min_value=0.05,
            max_value=2.0,
            value=float(st.session_state.get("grid_resolution", 0.3)),
            help="Smaller = denser grid, slower prediction.",
        )

    st.markdown(
        "**Outputs include:** `dashboard_config.yaml`, optimized parameters CSV, "
        "boundary proximity report, optimization track, NetCDF temporal exports, "
        "and `spatial_predictions.csv` / `spatial_rates.csv`."
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="grid_back"):
            go_to_step(6)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="grid_next"):
            update_state(
                lat_padding=float(lat_pad),
                lon_padding=float(lon_pad),
                grid_resolution=float(resolution),
            )
            go_to_step(8)
            st.rerun()
