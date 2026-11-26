"""STGP Interactive Dashboard — Streamlit application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows: PyTorch (libomp) vs NumPy/MKL (libiomp5md) — set before any torch/numpy load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Streamlit runs this file as a script, so the repo root is not on sys.path by
# default. Bootstrap before any stgp_dashboard imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from stgp_dashboard.paths import ensure_import_paths
from stgp_dashboard.services.viz_scope import (
    STEP_GRIDS,
    should_skip_grids_step,
    sync_viz_scope_from_widgets,
)
from stgp_dashboard.state import STEP_LABELS, init_session_state
from stgp_dashboard.steps import (
    analysis_period,
    data_paths,
    filtering,
    gia,
    grids,
    priors,
    results,
    review_run,
    sensitivity,
    training,
    welcome,
)

ensure_import_paths()

st.set_page_config(
    page_title="STGP Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEP_RENDERERS = {
    0: welcome.render,
    1: data_paths.render,
    2: analysis_period.render,
    3: gia.render,
    4: training.render,
    5: filtering.render,
    6: priors.render,
    7: grids.render,
    8: review_run.render,
    9: results.render,
    10: sensitivity.render,
}


def main() -> None:
    init_session_state()
    mode = st.session_state.get("mode", "wizard")
    step = int(st.session_state.get("step", 0))
    # Widget keys update before this run; sync before sidebar so "(skipped)"
    # matches the radio the user just clicked (not the previous viz_scope).
    viz_scope = sync_viz_scope_from_widgets(
        st.session_state, mode=str(mode), step=step
    )

    st.sidebar.title("STGP Dashboard")
    st.sidebar.caption("Guided spatiotemporal GP analysis")

    if mode == "wizard":
        skip_grids = should_skip_grids_step(viz_scope)
        for i, label in enumerate(STEP_LABELS[:9]):
            marker = "▸ " if i == step else "  "
            suffix = " (skipped)" if skip_grids and i == STEP_GRIDS else ""
            st.sidebar.write(f"{marker}{i}. {label}{suffix}")
    elif mode == "results":
        st.sidebar.success("Results")
        if st.sidebar.button("Sensitivity suite"):
            st.session_state["mode"] = "sensitivity"
            st.session_state["step"] = 10
            st.rerun()
    elif mode == "sensitivity":
        st.sidebar.info("Sensitivity suite")

    st.sidebar.divider()
    if st.sidebar.button("Jump to Welcome"):
        st.session_state["mode"] = "wizard"
        st.session_state["step"] = 0
        st.rerun()

    if mode == "results" and step != 10:
        step = 9
    elif mode == "sensitivity":
        step = 10
    elif mode == "wizard" and step > 8:
        step = 8

    renderer = STEP_RENDERERS.get(step, welcome.render)
    renderer()


if __name__ == "__main__":
    main()
