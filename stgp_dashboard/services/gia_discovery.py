"""Scan GIA data directories for available model files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _scan_glob(directory: Path, pattern: str, limit: int = 20) -> List[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob(pattern))[:limit]


def _scan_3d_models(directory: Path, limit: int = 40) -> List[str]:
    """Match pipeline ``load_3d_gia_ensemble``: top-level ``*.csv`` then ``*.txt``."""
    if not directory.is_dir():
        return []
    csv_files = _scan_glob(directory, "*.csv", limit=limit)
    txt_files = _scan_glob(directory, "*.txt", limit=limit)
    # Prefer CSV first (same order as pipeline), then TXT; de-dupe by name.
    seen = set()
    ordered: List[str] = []
    for name in csv_files + txt_files:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered[:limit]


def discover_gia_paths(
    repo_root: Path,
    *,
    gia_mat_dir: Optional[str] = None,
    gia_anu_dir: Optional[str] = None,
    gia_3d_dir: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    default_mat = root / "Data" / "GIA"
    default_anu = root / "Data" / "ANU_1D"

    mat_dir = Path(gia_mat_dir).expanduser() if gia_mat_dir else default_mat
    if not mat_dir.is_absolute():
        mat_dir = (root / mat_dir).resolve()

    anu_dir = Path(gia_anu_dir).expanduser() if gia_anu_dir else default_anu
    if not anu_dir.is_absolute():
        anu_dir = (root / anu_dir).resolve()

    gia_3d_path: Optional[Path] = None
    if gia_3d_dir:
        gia_3d_path = Path(gia_3d_dir).expanduser()
        if not gia_3d_path.is_absolute():
            gia_3d_path = (root / gia_3d_path).resolve()

    mat_files = _scan_glob(mat_dir, "*.mat", limit=30)
    anu_files = _scan_glob(anu_dir, "*.txt", limit=30) + _scan_glob(
        anu_dir, "*.csv", limit=30
    )
    models_3d: List[str] = []
    if gia_3d_path and gia_3d_path.is_dir():
        models_3d = _scan_3d_models(gia_3d_path, limit=40)

    return {
        "gia_mat_dir": str(mat_dir),
        "gia_anu_dir": str(anu_dir),
        "gia_3d_dir": str(gia_3d_path) if gia_3d_path else "",
        "mat_count": len(mat_files),
        "mat_samples": mat_files[:8],
        "anu_count": len(anu_files),
        "anu_samples": anu_files[:8],
        "combined_3d_count": len(models_3d),
        "combined_3d_samples": models_3d[:8],
        "has_mat": len(mat_files) > 0,
        "has_anu": len(anu_files) > 0,
        "has_3d": len(models_3d) > 0,
    }
