"""Run sensitivity scenarios from the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from stgp_dashboard.paths import ensure_import_paths, repo_root

ensure_import_paths()

from stgp_pipeline.run import resolve_workflow_id  # noqa: E402
from stgp_pipeline.sensitivity.runner import run_workflow_scenarios  # noqa: E402
from stgp_pipeline.sensitivity.scenarios import (  # noqa: E402
    build_workflow_scenarios,
    filter_scenarios_for_cfg,
)
from stgp_pipeline.sensitivity.summary import build_workflow_summary  # noqa: E402
from stgp_pipeline.sensitivity.workflow_common import save_yaml  # noqa: E402


def run_sensitivity_from_config(
    cfg: Mapping[str, Any],
    *,
    scenario_selector: str = "bounds",
    output_subdir: str = "sensitivity",
) -> Dict[str, Any]:
    workflow = resolve_workflow_id(cfg)
    paths = cfg.get("paths") or {}
    output_root = Path(str(paths.get("output_dir", "."))).resolve() / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = output_root / "adopted_config.yaml"
    save_yaml(config_path, dict(cfg))

    scenarios = build_workflow_scenarios(workflow, scenario_selector)
    scenarios = filter_scenarios_for_cfg(scenarios, cfg)

    results = run_workflow_scenarios(
        workflow,
        config_path=config_path,
        output_root=output_root,
        scenarios=scenarios,
    )

    summary: Dict[str, Any] = {}
    try:
        summary = build_workflow_summary(output_root)
    except Exception:  # noqa: BLE001
        summary = {}

    return {
        "results": results,
        "output_root": str(output_root),
        "summary": summary,
    }
