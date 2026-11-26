"""
Scenario execution engine for the configurable sensitivity suite.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for dry-run-only envs
    torch = None

from PSTHM.config.load import load_config

from .config_mutations import (
    apply_bound_multiplier,
    clone_config,
    configure_merged_nonlinear,
    set_kernel_enabled_map,
)
from .scenarios import ScenarioSpec
from .workflow_common import save_yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _apply_iteration_override(
    cfg: Mapping[str, Any],
    workflow: str,
    iterations: int | None,
) -> Dict[str, Any]:
    out = clone_config(cfg)
    if iterations is None:
        return out

    iters = int(iterations)
    out.setdefault("gp_model", {})["num_iterations"] = iters
    if workflow in {"4ka", "3d-gia", "regional-gia"}:
        out.setdefault("runs", {}).setdefault("stgp_with_gia", {})[
            "num_iterations"
        ] = iters
    return out


def _effective_kernel_enabled(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    kernels = ((cfg.get("gp_model") or {}).get("kernels")) or {}
    out: Dict[str, bool] = {}
    if not isinstance(kernels, Mapping):
        return out
    for name, block in kernels.items():
        if isinstance(block, Mapping):
            out[str(name)] = bool(block.get("enabled", True))
    return out


def _resolve_paths_against_base(
    cfg: Mapping[str, Any], base_dir: Path
) -> Dict[str, Any]:
    """
    Resolve all string entries under cfg['paths'] against the YAML directory.

    `PSTHM.config.load.load_config()` only resolves a subset of known keys; legacy
    no-GIA configs may use `paths.dataset`, so we normalize everything here.
    """
    out = clone_config(cfg)
    paths = out.setdefault("paths", {})
    for key, value in list(paths.items()):
        if not isinstance(value, str):
            continue
        path_value = Path(value).expanduser()
        if not path_value.is_absolute():
            candidates = [
                (base_dir / path_value).resolve(),
                (base_dir.parent / path_value).resolve(),
            ]
            # Some legacy configs were authored to be resolved from notebook CWD
            # (repo-root-relative), not from the YAML directory itself.
            if len(base_dir.parents) >= 3:
                candidates.append((base_dir.parents[2] / path_value).resolve())
            chosen = next((c for c in candidates if c.exists()), candidates[0])
            paths[key] = str(chosen)
    return out


def _apply_scenario_mutations(
    cfg: Mapping[str, Any],
    scenario: ScenarioSpec,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = clone_config(cfg)
    records: List[Dict[str, Any]] = []

    if scenario.kernel_enabled:
        out, toggle_records = set_kernel_enabled_map(out, scenario.kernel_enabled)
        records.extend(toggle_records)

    if scenario.merged_nonlinear_enabled is not None:
        out, merged_records = configure_merged_nonlinear(
            out,
            enabled=bool(scenario.merged_nonlinear_enabled),
            prior_ranges=scenario.merged_nonlinear_prior_ranges,
        )
        records.extend(merged_records)

    if scenario.bounds_multiplier is not None:
        out, bound_records = apply_bound_multiplier(
            out,
            multiplier=float(scenario.bounds_multiplier),
            include_disabled=False,
        )
        records.extend({"type": "bounds", **item} for item in bound_records)

    return out, records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=False)


def run_single_scenario(
    workflow: str,
    *,
    config_path: Path,
    output_root: Path,
    scenario: ScenarioSpec,
    iterations: int | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute one sensitivity scenario and persist artifacts.
    """
    loaded = load_config(config_path)
    base_cfg = _resolve_paths_against_base(loaded.config, loaded.base_dir)

    scenario_dir = output_root / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = scenario_dir / "manifest.yaml"
    config_dump_path = scenario_dir / "config.yaml"
    run_log_path = scenario_dir / "run.log"
    metrics_path = scenario_dir / "metrics.json"

    cfg = _apply_iteration_override(base_cfg, workflow, iterations)
    cfg, mutation_records = _apply_scenario_mutations(cfg, scenario)

    # Keep workflow-specific runtime outputs localized per scenario.
    cfg.setdefault("paths", {})["output_dir"] = str(
        (scenario_dir / "workflow_outputs").resolve()
    )

    save_yaml(config_dump_path, cfg)

    manifest: Dict[str, Any] = {
        "name": scenario.name,
        "workflow": workflow,
        "category": scenario.category,
        "description": scenario.description,
        "config_source": str(loaded.config_path),
        "config_resolved": str(config_dump_path),
        "output_dir": str(scenario_dir),
        "status": "running",
        "dry_run": bool(dry_run),
        "started_at_utc": _utc_now(),
        "iterations_override": iterations,
        "kernel_enabled": dict(scenario.kernel_enabled or {}),
        "kernel_enabled_effective": _effective_kernel_enabled(cfg),
        "merged_nonlinear_enabled": scenario.merged_nonlinear_enabled,
        "merged_nonlinear_prior_ranges": dict(
            scenario.merged_nonlinear_prior_ranges or {}
        ),
        "bounds_multiplier": scenario.bounds_multiplier,
        "mutations": mutation_records,
    }
    save_yaml(manifest_path, manifest)

    if dry_run:
        manifest["status"] = "dry_run"
        manifest["finished_at_utc"] = _utc_now()
        save_yaml(manifest_path, manifest)
        return {
            "workflow": workflow,
            "scenario": scenario.name,
            "status": "dry_run",
            "output_dir": str(scenario_dir),
        }

    try:
        with run_log_path.open("w", encoding="utf-8") as run_log:
            with contextlib.redirect_stdout(run_log), contextlib.redirect_stderr(
                run_log
            ):
                print(
                    f"[{_utc_now()}] workflow={workflow} scenario={scenario.name} start"
                )
                print(f"Config source: {loaded.config_path}")
                print(f"Mutations: {json.dumps(mutation_records, default=str)}")
                with _temporary_cwd(scenario_dir):
                    if workflow == "16ka":
                        from .workflow_16ka import run_workflow_16ka

                        metrics = run_workflow_16ka(cfg, scenario_dir)
                    elif workflow == "4ka":
                        from .workflow_4ka import run_workflow_4ka

                        metrics = run_workflow_4ka(
                            cfg,
                            scenario_dir,
                            base_dir=loaded.base_dir,
                        )
                    elif workflow == "3d-gia":
                        from .workflow_3d_gia import run_workflow_3d_gia

                        metrics = run_workflow_3d_gia(
                            cfg,
                            scenario_dir,
                            base_dir=loaded.base_dir,
                        )
                    elif workflow == "regional-gia":
                        from .workflow_regional_gia import run_workflow_regional_gia

                        metrics = run_workflow_regional_gia(
                            cfg,
                            scenario_dir,
                            base_dir=loaded.base_dir,
                        )
                    else:
                        raise ValueError(f"Unsupported workflow: {workflow!r}")
                print(
                    f"[{_utc_now()}] workflow={workflow} scenario={scenario.name} done"
                )

        _write_json(metrics_path, metrics)
        manifest["status"] = "success"
        manifest["metrics"] = metrics
    except Exception as exc:  # noqa: BLE001
        err = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with run_log_path.open("a", encoding="utf-8") as run_log:
            run_log.write("\n\n--- ERROR ---\n")
            run_log.write(err["traceback"])
        _write_json(metrics_path, err)
        manifest["status"] = "failed"
        manifest["error"] = err

    manifest["finished_at_utc"] = _utc_now()
    save_yaml(manifest_path, manifest)
    return {
        "workflow": workflow,
        "scenario": scenario.name,
        "status": manifest["status"],
        "output_dir": str(scenario_dir),
    }


def run_workflow_scenarios(
    workflow: str,
    *,
    config_path: Path,
    output_root: Path,
    scenarios: Sequence[ScenarioSpec],
    iterations: int | None = None,
    dry_run: bool = False,
    jobs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run all scenarios for one workflow.

    `jobs` is accepted for CLI compatibility; execution is currently sequential
    to keep Pyro parameter-store behavior deterministic across scenarios.
    """
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        result = run_single_scenario(
            workflow,
            config_path=config_path,
            output_root=output_root,
            scenario=scenario,
            iterations=iterations,
            dry_run=dry_run,
        )
        results.append(result)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
