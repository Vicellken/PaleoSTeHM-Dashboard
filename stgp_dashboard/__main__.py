"""Launch the STGP dashboard: python -m stgp_dashboard"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    app_path = Path(__file__).resolve().parent / "app.py"
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sep = os.pathsep
    prefix = str(repo_root)
    env["PYTHONPATH"] = (
        f"{prefix}{sep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else prefix
    )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(repo_root), env=env))


if __name__ == "__main__":
    main()
