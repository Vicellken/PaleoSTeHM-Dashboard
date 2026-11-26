"""Step 1 — Data paths and preview."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stgp_dashboard.components.path_inputs import dir_path_input, file_path_input
from stgp_dashboard.paths import repo_root
from stgp_dashboard.services.age_bounds import (
    age_quality_warnings,
    default_max_age_from_preview,
)
from stgp_dashboard.services.data_preview import preview_rsl_csv
from stgp_dashboard.state import go_to_step, update_state


def render() -> None:
    st.header("Data & output paths")

    flash = st.session_state.pop("config_load_message", None)
    if flash:
        st.success(flash)

    paleo = file_path_input(
        "Paleo RSL CSV path",
        state_key="paleo_rsl",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        help="HOLSEA-format CSV with Level, Region, Region.1 (numeric ID), lat/lon, Age, RSL columns.",
    )
    output = dir_path_input(
        "Analysis output directory",
        state_key="output_dir",
        help="All results, figures, and dashboard_config.yaml will be saved here.",
    )

    if st.button("Preview dataset", key="preview_data"):
        preview = preview_rsl_csv(paleo)
        update_state(data_preview=preview)
        if preview.get("errors"):
            for err in preview["errors"]:
                st.error(err)
        else:
            st.success(f"Loaded preview: {preview.get('n_rows', 0)} rows")
        st.rerun()

    preview = st.session_state.get("data_preview")
    if preview and preview.get("path") == str(Path(paleo).expanduser().resolve()):
        if preview.get("errors"):
            for err in preview["errors"]:
                st.error(err)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", preview.get("n_rows", 0))
            c2.metric("Finite coords", preview.get("n_finite_coords", 0))
            c3.metric("Regions", preview.get("n_regions", "—"))
            if "age_min_bp" in preview:
                st.write(
                    f"**Age range:** {preview['age_min_bp']:.0f} – "
                    f"{preview['age_max_bp']:.0f} cal yr BP"
                )
            if "lat_min" in preview:
                st.write(
                    f"**Spatial extent:** lat {preview['lat_min']:.2f}–{preview['lat_max']:.2f}, "
                    f"lon {preview['lon_min']:.2f}–{preview['lon_max']:.2f}"
                )
            if preview.get("level_counts"):
                st.write("**Levels:**", preview["level_counts"])
            for w in preview.get("warnings", []):
                st.warning(w)
            for w in age_quality_warnings(preview):
                st.warning(w)

            if "age_max_bp" in preview:
                age_default = int(st.session_state.get("max_age_bp", 16000))
                preview_max = default_max_age_from_preview(preview, int(preview["age_max_bp"]))
                if age_default == 16000 or preview_max < age_default:
                    update_state(
                        max_age_bp=preview_max,
                        viz_end_bp=preview_max,
                    )

    if not output and paleo:
        suggested = repo_root() / "Outputs" / "dashboard_run"
        st.caption(f"Suggested output: `{suggested}`")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="data_back"):
            go_to_step(0)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="data_next"):
            if not paleo or not output:
                st.error("Both input CSV and output directory are required.")
            else:
                preview = preview_rsl_csv(paleo)
                update_state(data_preview=preview)
                if preview.get("errors"):
                    for err in preview["errors"]:
                        st.error(err)
                else:
                    go_to_step(2)
                    st.rerun()
