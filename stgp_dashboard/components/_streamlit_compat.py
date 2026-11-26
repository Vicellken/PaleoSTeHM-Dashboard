"""Small Streamlit version compatibility helpers."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def _streamlit_version_tuple() -> tuple[int, int]:
    parts = getattr(st, "__version__", "0.0").split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def width_kwargs(*, stretch: bool = True) -> Dict[str, Any]:
    """Return ``width='stretch'/'content'`` on Streamlit >= 1.50, else ``use_container_width``."""
    if _streamlit_version_tuple() >= (1, 50):
        return {"width": "stretch" if stretch else "content"}
    return {"use_container_width": bool(stretch)}
