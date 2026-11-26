"""Step 4 — SVI training parameters."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stgp_dashboard.components.path_inputs import file_path_input
from stgp_dashboard.services.env_check import check_environment
from stgp_dashboard.state import go_to_step, update_state


def _default_checkpoint_path() -> str:
    output_dir = str(st.session_state.get("output_dir", "")).strip()
    if output_dir:
        candidate = Path(output_dir).expanduser() / "run" / "trained_param_store.pt"
        if candidate.is_file():
            return str(candidate.resolve())
    return str(st.session_state.get("param_store_path", ""))


def render() -> None:
    st.header("Training behavior")
    st.caption("Defaults work for most studies. Adjust only if loss behaves oddly.")

    env = check_environment()
    gpu_available = "CUDA" in env.get("gpu_info", "")
    gpu_info = env.get("gpu_info", "CPU only")

    # Bind radio to the same session key as persisted training_mode so load-config
    # and the widget cannot disagree.
    if st.session_state.get("training_mode") not in {"train", "load_checkpoint"}:
        st.session_state["training_mode"] = "train"

    training_mode = st.radio(
        "Training mode",
        options=["train", "load_checkpoint"],
        format_func=lambda x: (
            "Train new model (run SVI)"
            if x == "train"
            else "Load saved checkpoint (skip SVI)"
        ),
        horizontal=True,
        key="training_mode",
    )

    param_store_path = str(st.session_state.get("param_store_path", "") or "")
    if training_mode == "load_checkpoint":
        if not param_store_path.strip():
            st.session_state["param_store_path"] = _default_checkpoint_path()
        param_store_path = file_path_input(
            "Trained model checkpoint (.pt)",
            state_key="param_store_path",
            filetypes=[
                ("Pyro param store", "*.pt"),
                ("All files", "*.*"),
            ],
            help=(
                "Pyro ParamStore saved after a prior run "
                "(default: output_dir/run/trained_param_store.pt)."
            ),
            placeholder="path/to/run/trained_param_store.pt",
        )
        st.warning(
            "Loading a checkpoint skips SVI and freezes hyperparameters from the "
            ".pt file. Use the same kernel structure and priors as the saved model."
        )
    elif st.session_state.get("checkpoint_available") or _default_checkpoint_path():
        existing = _default_checkpoint_path()
        if existing:
            st.info(
                f"A checkpoint already exists at `{existing}`. "
                "**Train new model** will overwrite it after fresh SVI. "
                "Switch to **Load saved checkpoint** only if you want to skip training."
            )

    c1, c2 = st.columns(2)
    with c1:
        num_iter = st.number_input(
            "SVI iterations",
            min_value=100,
            max_value=5000,
            value=int(st.session_state.get("num_iterations", 500)),
            disabled=training_mode == "load_checkpoint",
        )
        lr = st.number_input(
            "Learning rate",
            min_value=0.001,
            max_value=1.0,
            value=float(st.session_state.get("learning_rate", 0.05)),
            format="%.4f",
            disabled=training_mode == "load_checkpoint",
        )
        decay = st.number_input(
            "Decay rate (gamma)",
            min_value=0.05,
            max_value=1.0,
            value=float(st.session_state.get("decay_r", 0.2)),
            format="%.3f",
            disabled=training_mode == "load_checkpoint",
        )
    with c2:
        step_size = st.number_input(
            "LR step size (iterations between decay)",
            min_value=10,
            max_value=500,
            value=int(st.session_state.get("step_size", 50)),
            disabled=training_mode == "load_checkpoint",
        )
        jitter = st.number_input(
            "Jitter",
            min_value=1e-8,
            max_value=1e-3,
            value=float(st.session_state.get("jitter", 1e-5)),
            format="%.1e",
        )
        gpu = st.checkbox(
            "Use GPU",
            value=bool(st.session_state.get("gpu", False)),
            disabled=not gpu_available or training_mode == "load_checkpoint",
            help=gpu_info,
        )

    if not gpu_available:
        gpu_detail = env.get("gpu_detail", "")
        if "MPS" in gpu_info:
            st.info(
                "Apple MPS/GPU is not supported here — training uses **CPU**, which "
                "is enough for typical runs. No extra MPS packages are needed. "
                "Launch with `python -m stgp_dashboard` to avoid OpenMP conflicts."
            )
        else:
            st.info(
                "Training will use **CPU** (fully supported). CUDA acceleration is "
                "optional — only needed if you want faster training on an NVIDIA GPU:\n\n"
                "`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`\n\n"
                "Launch with `python -m stgp_dashboard` to avoid OpenMP conflicts."
            )
        if gpu_detail:
            st.caption(gpu_detail)
    else:
        st.caption(f"Detected: {gpu_info}")
        if env.get("gpu_detail"):
            st.caption(env["gpu_detail"])

    fixed_seed = st.checkbox(
        "Fixed random seed (reproducible)",
        value=bool(st.session_state.get("fixed_seed", True)),
    )
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=99999,
        value=int(st.session_state.get("seed", 0)),
        disabled=not fixed_seed,
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="train_back"):
            go_to_step(3)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="train_next"):
            if training_mode == "load_checkpoint" and not param_store_path.strip():
                st.error("Select a checkpoint file or switch to Train new model.")
            elif (
                training_mode == "load_checkpoint"
                and not Path(param_store_path).expanduser().is_file()
            ):
                st.error(f"Checkpoint not found: {param_store_path}")
            else:
                # training_mode is owned by the radio widget key — do not reassign it.
                update_state(
                    param_store_path=param_store_path.strip(),
                    num_iterations=int(num_iter),
                    learning_rate=float(lr),
                    decay_r=float(decay),
                    step_size=int(step_size),
                    jitter=float(jitter),
                    gpu=bool(gpu),
                    fixed_seed=bool(fixed_seed),
                    seed=int(seed),
                )
                go_to_step(5)
                st.rerun()
