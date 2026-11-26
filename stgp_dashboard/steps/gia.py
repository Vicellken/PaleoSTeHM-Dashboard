"""Step 3 — GIA prior configuration."""

from __future__ import annotations

import streamlit as st

from stgp_dashboard.components.path_inputs import dir_path_input, files_path_input
from stgp_dashboard.paths import repo_root
from stgp_dashboard.services.gia_discovery import discover_gia_paths
from stgp_dashboard.services.workflow_labels import GIA_UI_OPTIONS, gia_pipeline_to_ui
from stgp_dashboard.state import go_to_step, update_state


def _default_3d_dir() -> str:
    return str(repo_root() / "Data" / "3D_GIA")


def render() -> None:
    st.header("GIA prior (optional)")
    use_gia = st.toggle(
        "Include GIA as prior",
        value=bool(st.session_state.get("use_gia", False)),
        help="Subtract a GIA ensemble mean and add GIA covariance to observation noise.",
    )

    discovery = st.session_state.get("gia_discovery")
    use_3d = bool(st.session_state.get("gia_use_3d", False))
    gia_3d_dir = str(st.session_state.get("gia_3d_dir", "") or "")
    source_mode = str(st.session_state.get("gia_3d_source_mode", "folder") or "folder")
    selected_files = list(st.session_state.get("gia_3d_selected_files") or [])
    n_search = int(st.session_state.get("gia_n_search", 50))

    if use_gia:
        st.markdown("### GIA data directories")
        default_root = repo_root()
        if not st.session_state.get("gia_mat_dir"):
            st.session_state["gia_mat_dir"] = str(default_root / "Data" / "GIA")
        if not st.session_state.get("gia_anu_dir"):
            st.session_state["gia_anu_dir"] = str(default_root / "Data" / "ANU_1D")
        mat_dir = dir_path_input(
            "ICE6G .mat directory",
            state_key="gia_mat_dir",
        )
        anu_dir = dir_path_input(
            "ANU-ICE .txt directory",
            state_key="gia_anu_dir",
        )

        st.markdown("### 3D GIA grids (optional)")
        use_3d = st.toggle(
            "Include 3D GIA grids",
            value=use_3d,
            key="gia_use_3d_toggle",
            help=(
                "Load regional 3D GIA models (.csv / .txt) and combine them with the "
                "1D ensemble (gia.ensemble=all)."
            ),
        )

        if use_3d:
            source_mode = st.radio(
                "3D GIA source",
                options=["folder", "files"],
                format_func=lambda x: (
                    "Folder (all .csv / .txt in directory)"
                    if x == "folder"
                    else "Selected file(s)"
                ),
                index=0 if source_mode == "folder" else 1,
                horizontal=True,
                key="gia_3d_source_mode_radio",
            )
            if source_mode == "folder":
                if not st.session_state.get("gia_3d_dir"):
                    st.session_state["gia_3d_dir"] = _default_3d_dir()
                gia_3d_dir = dir_path_input(
                    "3D GIA directory",
                    state_key="gia_3d_dir",
                    help=(
                        "Directory containing regional 3D GIA grids (.csv / .txt). "
                        "Default location: Data/3D_GIA."
                    ),
                    placeholder="path/to/3D_GIA",
                )
                selected_files = []
            else:
                selected_files = files_path_input(
                    "3D GIA model files",
                    state_key="gia_3d_selected_files",
                    filetypes=[
                        ("3D GIA grids", "*.csv *.txt"),
                        ("CSV", "*.csv"),
                        ("TXT", "*.txt"),
                        ("All files", "*.*"),
                    ],
                    help=(
                        "Pick one or more .csv / .txt grids. They are staged into "
                        "output_dir/gia_3d_selected/ when you run the model."
                    ),
                )
                # Effective dir for discovery: parent of first file, or empty.
                if selected_files:
                    from pathlib import Path

                    gia_3d_dir = str(Path(selected_files[0]).expanduser().parent)
                else:
                    gia_3d_dir = ""
        else:
            gia_3d_dir = ""
            selected_files = []
            source_mode = "folder"

        scan_3d = gia_3d_dir if use_3d else None
        if st.button("Scan GIA directories", key="scan_gia"):
            discovery = discover_gia_paths(
                default_root,
                gia_mat_dir=mat_dir,
                gia_anu_dir=anu_dir,
                gia_3d_dir=scan_3d,
            )
            update_state(
                gia_mat_dir=mat_dir,
                gia_anu_dir=anu_dir,
                gia_3d_dir=gia_3d_dir if use_3d else "",
                gia_use_3d=use_3d,
                gia_3d_source_mode=source_mode,
                gia_3d_selected_files=selected_files,
                gia_discovery=discovery,
            )
            st.rerun()

        if discovery:
            c1, c2, c3 = st.columns(3)
            c1.metric("ICE6G .mat files", discovery.get("mat_count", 0))
            c2.metric("ANU files", discovery.get("anu_count", 0))
            c3.metric("3D GIA files", discovery.get("combined_3d_count", 0))
            if discovery.get("mat_samples"):
                st.caption(f"MAT samples: {', '.join(discovery['mat_samples'][:5])}")
            if discovery.get("combined_3d_samples"):
                st.caption(
                    "3D models: "
                    + ", ".join(discovery["combined_3d_samples"][:8])
                )
            if not discovery.get("has_mat") and not discovery.get("has_anu"):
                st.warning("No 1D GIA model files found in the specified directories.")
            if use_3d and not discovery.get("has_3d"):
                st.warning(
                    "No 3D GIA .csv / .txt files found. Place grids in Data/3D_GIA, "
                    "point at another folder, or select files explicitly."
                )

        st.markdown("### GIA weighting")
        current_source = gia_pipeline_to_ui(
            str(st.session_state.get("gia_1d_source", "both"))
        )
        if current_source not in GIA_UI_OPTIONS:
            current_source = "both"
        source = st.selectbox(
            "1D model source",
            options=GIA_UI_OPTIONS,
            index=GIA_UI_OPTIONS.index(current_source),
            help="ICE6G (.mat), ANU (.txt), or both ensembles.",
        )
        weighting = st.selectbox(
            "GIA mean weighting",
            options=["wrss_all", "wrss_3d", "equal"],
            index=["wrss_all", "wrss_3d", "equal"].index(
                st.session_state.get("gia_mean_weighting", "wrss_all")
            ),
            help="Covariance is always WRSS-weighted over the full ensemble.",
        )
        with st.expander("Advanced GIA options"):
            n_search = st.slider(
                "WRSS age search grid size (n_search)",
                min_value=10,
                max_value=100,
                value=int(st.session_state.get("gia_n_search", 50)),
            )
    else:
        source = st.session_state.get("gia_1d_source", "both")
        weighting = st.session_state.get("gia_mean_weighting", "wrss_all")
        n_search = st.session_state.get("gia_n_search", 50)
        use_3d = False
        gia_3d_dir = ""
        source_mode = "folder"
        selected_files = []
        mat_dir = st.session_state.get("gia_mat_dir", "")
        anu_dir = st.session_state.get("gia_anu_dir", "")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="gia_back"):
            go_to_step(2)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="gia_next"):
            if use_gia and use_3d and source_mode == "files" and not selected_files:
                st.error("Select at least one 3D GIA file, or switch to Folder mode.")
            else:
                payload = {
                    "use_gia": use_gia,
                    "gia_1d_source": source,
                    "gia_mean_weighting": weighting,
                    "gia_n_search": int(n_search),
                    "gia_mat_dir": mat_dir,
                    "gia_anu_dir": anu_dir,
                    "gia_3d_dir": gia_3d_dir if use_3d else "",
                    "gia_use_3d": bool(use_gia and use_3d),
                    "gia_3d_source_mode": source_mode if use_3d else "folder",
                    "gia_3d_selected_files": selected_files if use_3d else [],
                }
                if use_gia:
                    discovery = discover_gia_paths(
                        repo_root(),
                        gia_mat_dir=mat_dir,
                        gia_anu_dir=anu_dir,
                        gia_3d_dir=gia_3d_dir if use_3d else None,
                    )
                    payload["gia_discovery"] = discovery
                update_state(**payload)
                go_to_step(4)
                st.rerun()
