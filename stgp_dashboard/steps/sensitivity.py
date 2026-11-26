"""Step 10 — Optional sensitivity suite (post-first-run)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stgp_dashboard.components._streamlit_compat import width_kwargs
from stgp_dashboard.services.config_builder import build_config
from stgp_dashboard.services.open_folder import open_folder
from stgp_dashboard.services.sensitivity_plots import (
    build_bounds_table,
    generate_sensitivity_analytics,
    list_display_images,
)
from stgp_dashboard.services.sensitivity_runner import run_sensitivity_from_config
from stgp_dashboard.state import get_state, go_to_step, update_state


def render() -> None:
    st.header("Sensitivity suite")
    st.markdown(
        "Run optional sensitivity scenarios after your adopted configuration works. "
        "This is **not required** for a first run — use it once you understand "
        "your model and want to test prior widening or kernel toggles."
    )

    if not st.session_state.get("saved_config_path") and not st.session_state.get(
        "run_metrics"
    ):
        st.warning("Complete a first adopted run before using the sensitivity suite.")
        if st.button("← Back to wizard", key="sens_back_no_run"):
            go_to_step(0)
            st.rerun()
        return

    state = get_state()
    cfg = build_config(state)
    output_root = Path(str(cfg.get("paths", {}).get("output_dir", "."))) / "sensitivity"

    selector = st.selectbox(
        "Scenario group",
        options=[
            "adopted",
            "bounds",
            "kernel-toggles",
            "merged-nonlinear",
            "all",
        ],
        index=["adopted", "bounds", "kernel-toggles", "merged-nonlinear", "all"].index(
            st.session_state.get("sensitivity_selector", "bounds")
        ),
        help="bounds = widen prior ranges; kernel-toggles = disable individual kernels.",
    )

    st.caption(
        "Scenarios run sequentially via the existing stgp_pipeline sensitivity engine."
    )

    if st.button("Run sensitivity scenarios", type="primary", key="run_sensitivity"):
        with st.spinner("Running sensitivity scenarios…"):
            try:
                payload = run_sensitivity_from_config(cfg, scenario_selector=selector)
                update_state(
                    sensitivity_results=payload.get("results"),
                    sensitivity_selector=selector,
                    sensitivity_output_root=payload.get("output_root"),
                    sensitivity_summary=payload.get("summary"),
                )
                st.success(
                    f"Completed {len(payload.get('results', []))} scenario(s)."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Sensitivity run failed: {exc}")

    sens_root = st.session_state.get("sensitivity_output_root") or str(output_root)
    if sens_root and Path(sens_root).is_dir():
        if st.button("Open sensitivity output folder", key="open_sens_folder"):
            try:
                open_folder(sens_root)
                st.toast("Opened sensitivity output folder.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not open folder: {exc}")

    results = st.session_state.get("sensitivity_results")
    if results:
        st.subheader("Scenario results")
        for item in results:
            status = item.get("status", "unknown")
            name = item.get("scenario", item.get("name", "—"))
            out_dir = item.get("output_dir", "")
            if status == "success":
                st.success(f"**{name}** — {out_dir}")
            elif status == "dry_run":
                st.info(f"**{name}** (dry run) — {out_dir}")
            else:
                st.error(f"**{name}** — failed")

    summary = st.session_state.get("sensitivity_summary")
    if summary or (sens_root and Path(sens_root).is_dir()):
        st.subheader("Analytical plots")
        if summary is None and st.button("Generate summary plots", key="gen_sens_plots"):
            with st.spinner("Building sensitivity summary…"):
                try:
                    summary = generate_sensitivity_analytics(sens_root)
                    update_state(sensitivity_summary=summary)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Summary generation failed: {exc}")

        if summary:
            for png in list_display_images(summary):
                st.image(str(png), caption=png.stem.replace("_", " "))

            bounds_df = build_bounds_table(sens_root)
            if bounds_df is not None and not bounds_df.empty:
                with st.expander("Parameter bounds table"):
                    st.dataframe(bounds_df, **width_kwargs(stretch=True))

            md_path = summary.get("params_bounds_nill_md")
            if md_path and Path(md_path).is_file():
                with st.expander("Bounds markdown report"):
                    st.markdown(Path(md_path).read_text(encoding="utf-8"))

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("← Results", key="sens_back_results"):
            update_state(mode="results", step=9)
            st.rerun()
    with col2:
        if st.button("← Wizard", key="sens_back_wizard"):
            update_state(mode="wizard", step=8)
            st.rerun()
