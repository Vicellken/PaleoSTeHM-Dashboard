"""Ensure repo root is on sys.path for pipeline and PSTHM imports."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_import_paths() -> None:
    text = str(repo_root())
    if text not in sys.path:
        sys.path.insert(0, text)
