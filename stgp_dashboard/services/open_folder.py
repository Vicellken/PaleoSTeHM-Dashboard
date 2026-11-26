"""Open a directory in the OS file explorer."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def open_folder(path: str | Path) -> None:
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    system = platform.system()
    if system == "Windows":
        os.startfile(str(folder))  # noqa: S606
    elif system == "Darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)
