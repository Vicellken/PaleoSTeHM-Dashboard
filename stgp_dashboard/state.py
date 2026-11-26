"""Session state schema and helpers for the STGP dashboard wizard."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import streamlit as st

STEP_LABELS = [
    "Welcome",
    "Data & paths",
    "Analysis period",
    "GIA prior",
    "Training",
    "Filtering (advanced)",
    "Kernel priors",
    "Spatial grid",
    "Review & run",
    "Results",
    "Sensitivity",
]

DEFAULT_STATE: Dict[str, Any] = {
    "step": 0,
    "mode": "wizard",  # wizard | results | sensitivity
    # Step 0
    "loaded_config_path": "",
    # Step 1
    "paleo_rsl": "",
    "output_dir": "",
    "data_preview": None,
    # Step 2
    "max_age_bp": 16000,
    "viz_start_bp": 0,
    "viz_end_bp": 16000,
    "temporal_interval": 200,
    "spatial_interval": 1000,
    "rate_interval": 100,
    # Sea-level budget (post-temporal summary windows; independent of viz span)
    "budget_age_windows": [[0, 4000]],
    "budget_merge_gia_global": True,
    # Optional {anu, ice6g} Excel paths for GMSL/GRD split (empty = unset)
    "budget_gmsl_xlsx": None,
    # Step 3
    "use_gia": False,
    "gia_mat_dir": "",
    "gia_anu_dir": "",
    "gia_3d_dir": "",
    "gia_1d_source": "both",  # ice6g | anu | both (ice6g maps to pipeline mat)
    "gia_mean_weighting": "wrss_all",
    "gia_n_search": 50,
    "gia_use_3d": False,
    "gia_custom_mode": False,
    "gia_custom_path": "",
    "gia_discovery": None,
    # Step 4
    "num_iterations": 500,
    "learning_rate": 0.05,
    "decay_r": 0.2,
    "step_size": 50,
    "jitter": 1.0e-5,
    "fixed_seed": True,
    "seed": 0,
    "gpu": False,
    # Step 5
    "filter_levels": [1, 2],
    "l2_downweight_enabled": False,
    "l2_downweight_multiplier": 2.0,
    "l2_downweight_scope": "slip_only",
    "region_merges": [],
    "region_merge_rows": 0,
    "temporal_centroid_mode": "data_mean",
    # Step 6
    "prior_preset": "regional",
    "kernel_priors": {},
    # Step 7
    "lat_padding": 2.0,
    "lon_padding": 2.0,
    "grid_resolution": 0.3,
    # Visualization mask (spatial plot SD threshold)
    "spatial_mask_enabled": True,
    "spatial_mask_calibration_factor": 2.0,
    # Training mode
    "training_mode": "train",  # train | load_checkpoint
    "param_store_path": "",
    "checkpoint_available": False,
    # Run / results
    "run_metrics": None,
    "run_error": None,
    "boundary_df": None,
    "optimization_track": None,
    "saved_config_path": "",
    "run_log": "",
    "postprocess_complete": False,
    # Sensitivity
    "sensitivity_selector": "bounds",
    "sensitivity_results": None,
    "sensitivity_output_root": "",
    "sensitivity_summary": None,
    "visualization_outputs": None,
    "viz_scope": "full",
    "viz_view_choice": "All",
    # 3D GIA source: folder | files
    "gia_3d_source_mode": "folder",
    "gia_3d_selected_files": [],
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def get_state() -> Dict[str, Any]:
    init_session_state()
    return {k: st.session_state.get(k, deepcopy(v)) for k, v in DEFAULT_STATE.items()}


def update_state(**kwargs: Any) -> None:
    init_session_state()
    for key, value in kwargs.items():
        st.session_state[key] = value


def go_to_step(step: int) -> None:
    step = max(0, min(step, len(STEP_LABELS) - 1))
    update_state(step=step)


def merge_region_merges_text(text: str) -> List[List[int]]:
    merges: List[List[int]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) >= 2:
            merges.append([int(parts[0]), int(parts[1])])
    return merges
