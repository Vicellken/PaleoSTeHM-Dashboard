"""Environment and dependency checks."""

from __future__ import annotations

import os

# Must precede torch import on Windows when multiple OpenMP runtimes are present.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import platform
from typing import Any, Dict, List

import streamlit as st

from stgp_dashboard.paths import ensure_import_paths


@st.cache_resource
def check_environment() -> Dict[str, Any]:
    """Probe packages and GPU once per process (cached across Streamlit reruns)."""
    ensure_import_paths()
    checks: List[Dict[str, str]] = []
    ok = True

    for name, import_path in [
        ("torch", "torch"),
        ("pyro", "pyro"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("cartopy", "cartopy"),
        ("netCDF4", "netCDF4"),
        ("pyyaml", "yaml"),
        ("openpyxl", "openpyxl"),
        ("PSTHM", "PSTHM"),
        ("stgp_pipeline", "stgp_pipeline"),
    ]:
        try:
            __import__(import_path)
            checks.append({"name": name, "status": "ok", "detail": "imported"})
        except ImportError as exc:
            ok = False
            checks.append({"name": name, "status": "missing", "detail": str(exc)})

    gpu_info = "CPU only"
    gpu_detail = ""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = f"CUDA: {torch.cuda.get_device_name(0)}"
            gpu_detail = f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple MPS detected (unused — CPU)"
            gpu_detail = (
                f"PyTorch {torch.__version__}; MPS is not used by this dashboard "
                "(CPU training is fine; no extra MPS packages needed)"
            )
        else:
            gpu_detail = (
                f"PyTorch {torch.__version__} in CPU mode "
                f"(built with CUDA: {torch.version.cuda or 'none'}; "
                "CUDA is optional)"
            )
    except Exception as exc:  # noqa: BLE001
        gpu_detail = str(exc)

    return {
        "ok": ok,
        "checks": checks,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_info": gpu_info,
        "gpu_detail": gpu_detail,
    }
