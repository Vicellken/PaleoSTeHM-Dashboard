"""Step 0 — Welcome and environment check."""

from __future__ import annotations

import streamlit as st

from stgp_dashboard.components.path_inputs import file_path_input
from stgp_dashboard.services.config_loader import (
    load_config_file,
    summarize_resolved_inputs,
)
from stgp_dashboard.services.env_check import check_environment
from stgp_dashboard.state import go_to_step, update_state


def render() -> None:
    st.header("Welcome to STGP Analysis Dashboard")
    st.markdown(
        "This guided wizard helps you configure and run a spatiotemporal Gaussian "
        "Process (STGP) Holocene sea-level analysis without editing YAML by hand. "
        "**GPU acceleration is optional** — regular CPU training works fine for "
        "typical runs."
    )

    env = check_environment()
    if env["ok"]:
        st.success("All required packages are available.")
    else:
        st.error("Some required packages are missing. Activate the conda environment first.")

    gpu_info = env.get("gpu_info", "CPU only")
    gpu_detail = env.get("gpu_detail", "")
    if "CUDA" in gpu_info:
        st.success(f"Optional GPU available: {gpu_info}")
        if gpu_detail:
            st.caption(gpu_detail)
        st.caption(
            "CUDA can speed up training when you enable **Use GPU** on the Training "
            "step. CPU-only runs are still fully supported."
        )
    elif "MPS" in gpu_info:
        st.info(
            "Apple MPS/GPU is **not supported** by this dashboard. Training runs on "
            "**CPU**, which is enough for typical analyses — you do **not** need to "
            "install extra packages for MPS."
        )
        if gpu_detail:
            st.caption(gpu_detail)
    else:
        st.info(
            "Running in **CPU mode** — this is enough for typical analyses. "
            "CUDA acceleration is **not required**. On NVIDIA machines you may "
            "optionally install a CUDA-enabled PyTorch build later to enable "
            "**Use GPU** on the Training step:\n\n"
            "`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`\n\n"
            "Prefer `python -m stgp_dashboard` (sets the OpenMP workaround automatically)."
        )
        if gpu_detail:
            st.caption(gpu_detail)
    st.caption(f"Platform: {env['platform']} | Python {env['python']} | {gpu_info}")

    with st.expander("Package check details", expanded=not env["ok"]):
        for item in env["checks"]:
            icon = "✅" if item["status"] == "ok" else "❌"
            st.write(f"{icon} **{item['name']}** — {item['detail']}")

    st.subheader("Resume a prior run (optional)")
    config_path = file_path_input(
        "Load existing dashboard config YAML",
        state_key="loaded_config_path",
        filetypes=[
            ("YAML config", "*.yaml;*.yml"),
            ("All files", "*.*"),
        ],
        help=(
            "Select a dashboard or pipeline YAML to restore settings. "
            "Relative input paths (paleo CSV, GIA dirs) are resolved from the "
            "YAML file's folder. Output directory is left for you to set."
        ),
        placeholder="path/to/config.yaml",
    )
    if st.button("Load config", key="load_config_btn"):
        try:
            loaded = load_config_file(config_path)
            msg = f"Loaded config from {config_path}"
            inputs = summarize_resolved_inputs(loaded)
            if inputs["found"]:
                msg += f". Resolved inputs: {', '.join(inputs['found'])}"
            if inputs["missing"]:
                msg += (
                    f". Missing (check paths): {', '.join(inputs['missing'])}"
                )
            if loaded.get("training_mode") == "load_checkpoint":
                ckpt = loaded.get("param_store_path", "")
                msg += (
                    ". Config sets run_optimization=false — Training step will "
                    "skip SVI and load the saved checkpoint."
                )
                if ckpt:
                    msg += f" (`{ckpt}`)"
            elif loaded.get("checkpoint_available"):
                ckpt = loaded.get("param_store_path", "")
                msg += (
                    ". Found saved model checkpoint — default remains "
                    "**Train new model**; choose Load on the Training step to skip SVI."
                )
                if ckpt:
                    msg += f" (`{ckpt}`)"
            if loaded.get("postprocess_complete"):
                msg += (
                    " Existing prediction exports found — Results will skip "
                    "recomputing spatial rates unless you choose Recompute."
                )
            update_state(**loaded, step=1, config_load_message=msg)
            # Keep Streamlit widget keys in sync with loaded values.
            st.session_state["prior_preset_select"] = loaded.get(
                "prior_preset", "regional"
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load config: {exc}")

    st.info(
        "Click **Next** to begin. You will choose your dataset, analysis time window, "
        "optional GIA prior, training settings, and output paths."
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col2:
        if st.button("Next →", type="primary", key="welcome_next"):
            go_to_step(1)
            st.rerun()
