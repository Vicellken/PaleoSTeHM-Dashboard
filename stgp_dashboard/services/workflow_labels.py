"""User-facing labels for internal workflow IDs and GIA options."""

from __future__ import annotations

from typing import Any, Mapping

GIA_UI_TO_PIPELINE = {"ice6g": "mat", "anu": "anu", "both": "both"}
GIA_PIPELINE_TO_UI = {"mat": "ice6g", "anu": "anu", "both": "both"}
GIA_UI_OPTIONS = ["ice6g", "anu", "both"]

EARTH_RADIUS_KM = 6371.0


def gia_ui_to_pipeline(source: str) -> str:
    return GIA_UI_TO_PIPELINE.get(str(source).lower(), str(source))


def gia_pipeline_to_ui(source: str) -> str:
    return GIA_PIPELINE_TO_UI.get(str(source).lower(), str(source))


def resolve_workflow_label(cfg: Mapping[str, Any], workflow_id: str) -> str:
    from stgp_dashboard.paths import ensure_import_paths

    ensure_import_paths()
    from stgp_pipeline.run import resolve_workflow_label as _pipeline_label

    return _pipeline_label(cfg, workflow_id)
