"""File and directory path inputs with optional native browse dialogs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import streamlit as st

from stgp_dashboard.components._streamlit_compat import width_kwargs

FileTypeSpec = Tuple[str, str]

# Tk must run on the process main thread. Streamlit's script runner is not that
# thread on macOS, so dialogs run in a short-lived subprocess instead.
_DIALOG_TIMEOUT_S = 600

_TK_DIALOG_SCRIPT = r"""
import json
import sys
import tkinter as tk
from tkinter import filedialog

mode = sys.argv[1]
initialdir = sys.argv[2] or None
raw_filetypes = json.loads(sys.argv[3]) if sys.argv[3] else None

def _coerce_filetype(ft):
    label, patterns = ft[0], ft[1]
    if isinstance(patterns, list):
        patterns = patterns[0] if len(patterns) == 1 else tuple(patterns)
    return (label, patterns)

root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except tk.TclError:
    pass

kwargs = {}
if initialdir:
    kwargs["initialdir"] = initialdir
if raw_filetypes:
    kwargs["filetypes"] = [_coerce_filetype(ft) for ft in raw_filetypes]

try:
    if mode == "file":
        selected = filedialog.askopenfilename(**kwargs)
        print(selected or "", end="")
    elif mode == "dir":
        selected = filedialog.askdirectory(**kwargs)
        print(selected or "", end="")
    elif mode == "files":
        selected = filedialog.askopenfilenames(**kwargs)
        if selected:
            print("\n".join(selected), end="")
    else:
        sys.exit(2)
except Exception:
    try:
        root.destroy()
    except Exception:
        pass
    sys.exit(1)

root.destroy()
"""


def _widget_key(state_key: str) -> str:
    return f"_{state_key}_input"


def _sync_key(state_key: str) -> str:
    return f"_{state_key}_last_synced"


def _sync_widget_from_state(state_key: str, current: str) -> str:
    """Sync widget key from canonical state only on init or external change.

    Avoids clobbering freshly typed text on rerun when ``st.session_state[wkey]``
    holds the user's new edit while ``st.session_state[state_key]`` still holds
    the value from the prior run.
    """
    wkey = _widget_key(state_key)
    skey = _sync_key(state_key)
    if wkey not in st.session_state or st.session_state.get(skey) != current:
        st.session_state[wkey] = current
        st.session_state[skey] = current
    return wkey


def _resolve_initialdir(initial_dir: Optional[str]) -> str:
    if not initial_dir:
        return ""
    init = Path(initial_dir).expanduser()
    if init.is_dir():
        return str(init)
    if init.parent.is_dir():
        return str(init.parent)
    return ""


def _split_filetype_patterns(pattern: str) -> List[str]:
    """Split Windows/space-style patterns into individual Tk globs."""
    parts: List[str] = []
    for chunk in pattern.replace(";", " ").split():
        p = chunk.strip()
        if not p:
            continue
        if p in {"*.*", ".*", "*"}:
            parts.append("*")
        else:
            parts.append(p)
    return parts or ["*"]


def _normalize_filetypes(
    filetypes: Optional[Sequence[FileTypeSpec]],
) -> List[List[Union[str, List[str]]]]:
    """Convert call-site filetypes to JSON-serializable Tk-safe entries.

    Each entry is ``[label, pattern]`` or ``[label, [pattern, ...]]``.
    Semicolon- and space-separated patterns (Windows-style) are split.
    ``*.*`` is mapped to ``*`` for Aqua Tk compatibility.
    """
    if not filetypes:
        return []
    out: List[List[Union[str, List[str]]]] = []
    for label, pattern in filetypes:
        parts = _split_filetype_patterns(str(pattern))
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        if len(uniq) == 1:
            out.append([label, uniq[0]])
        else:
            out.append([label, uniq])
    return out


def _run_tk_dialog(
    mode: str,
    *,
    initial_dir: Optional[str] = None,
    filetypes: Optional[Sequence[FileTypeSpec]] = None,
) -> Union[str, List[str]]:
    """Run a Tk file/folder dialog in a subprocess; return path(s) or empty."""
    empty: Union[str, List[str]] = [] if mode == "files" else ""
    if not _tkinter_available():
        return empty

    initialdir = _resolve_initialdir(initial_dir)
    normalized = _normalize_filetypes(filetypes)
    ft_json = json.dumps(normalized) if normalized else ""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TK_DIALOG_SCRIPT, mode, initialdir, ft_json],
            capture_output=True,
            text=True,
            timeout=_DIALOG_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return empty

    if result.returncode != 0:
        return empty

    out = (result.stdout or "").strip("\n")
    if mode == "files":
        return [line for line in out.split("\n") if line] if out else []
    return out


def _browse_file(
    *,
    initial_dir: Optional[str] = None,
    filetypes: Optional[Sequence[FileTypeSpec]] = None,
) -> str:
    selected = _run_tk_dialog("file", initial_dir=initial_dir, filetypes=filetypes)
    return selected if isinstance(selected, str) else ""


def _browse_directory(*, initial_dir: Optional[str] = None) -> str:
    selected = _run_tk_dialog("dir", initial_dir=initial_dir)
    return selected if isinstance(selected, str) else ""


def _path_caption(path: str, *, expect: str) -> None:
    if not path.strip():
        return
    resolved = Path(path).expanduser()
    if expect == "file" and resolved.is_file():
        st.caption(f"Resolved file: `{resolved.resolve()}`")
    elif expect == "dir":
        if resolved.is_dir():
            st.caption(f"Resolved directory: `{resolved.resolve()}`")
        elif not resolved.exists():
            st.caption(f"Directory will be created: `{resolved.resolve()}`")


def _browse_files(
    *,
    initial_dir: Optional[str] = None,
    filetypes: Optional[Sequence[FileTypeSpec]] = None,
) -> List[str]:
    selected = _run_tk_dialog("files", initial_dir=initial_dir, filetypes=filetypes)
    return selected if isinstance(selected, list) else []


def file_path_input(
    label: str,
    *,
    state_key: str,
    filetypes: Optional[List[FileTypeSpec]] = None,
    help: Optional[str] = None,
    placeholder: str = "",
) -> str:
    """Text field plus Browse button for selecting a file path."""
    current = str(st.session_state.get(state_key) or "")
    wkey = _sync_widget_from_state(state_key, current)
    skey = _sync_key(state_key)

    col_text, col_btn = st.columns([6, 1])
    with col_btn:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        if st.button("Browse…", key=f"{state_key}_browse", **width_kwargs(stretch=True)):
            initial = str(st.session_state.get(wkey) or current) or str(Path.cwd())
            selected = _browse_file(initial_dir=initial, filetypes=filetypes)
            if selected:
                st.session_state[state_key] = selected
                st.session_state[wkey] = selected
                st.session_state[skey] = selected
                st.rerun()
            elif not _tkinter_available():
                st.warning("Native file browser unavailable in this environment.")

    with col_text:
        path = st.text_input(
            label,
            key=wkey,
            help=help,
            placeholder=placeholder,
        )

    st.session_state[state_key] = path
    st.session_state[skey] = path
    _path_caption(path, expect="file")
    return path


def dir_path_input(
    label: str,
    *,
    state_key: str,
    help: Optional[str] = None,
    placeholder: str = "",
) -> str:
    """Text field plus Browse button for selecting a directory path."""
    current = str(st.session_state.get(state_key) or "")
    wkey = _sync_widget_from_state(state_key, current)
    skey = _sync_key(state_key)

    col_text, col_btn = st.columns([6, 1])
    with col_btn:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        if st.button("Browse…", key=f"{state_key}_browse", **width_kwargs(stretch=True)):
            initial = str(st.session_state.get(wkey) or current) or str(Path.cwd())
            selected = _browse_directory(initial_dir=initial)
            if selected:
                st.session_state[state_key] = selected
                st.session_state[wkey] = selected
                st.session_state[skey] = selected
                st.rerun()
            elif not _tkinter_available():
                st.warning("Native folder browser unavailable in this environment.")

    with col_text:
        path = st.text_input(
            label,
            key=wkey,
            help=help,
            placeholder=placeholder,
        )

    st.session_state[state_key] = path
    st.session_state[skey] = path
    _path_caption(path, expect="dir")
    return path


def files_path_input(
    label: str,
    *,
    state_key: str,
    filetypes: Optional[List[FileTypeSpec]] = None,
    help: Optional[str] = None,
) -> List[str]:
    """Browse button for selecting one or more files; paths stored as a list."""
    current = list(st.session_state.get(state_key) or [])
    col_label, col_btn, col_clear = st.columns([5, 1, 1])
    with col_label:
        st.markdown(f"**{label}**")
        if help:
            st.caption(help)
    with col_btn:
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        if st.button("Browse…", key=f"{state_key}_browse", **width_kwargs(stretch=True)):
            initial = str(Path(current[0]).parent) if current else str(Path.cwd())
            selected = _browse_files(initial_dir=initial, filetypes=filetypes)
            if selected:
                st.session_state[state_key] = selected
                st.rerun()
            elif not _tkinter_available():
                st.warning("Native file browser unavailable in this environment.")
    with col_clear:
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        if current and st.button(
            "Clear", key=f"{state_key}_clear", **width_kwargs(stretch=True)
        ):
            st.session_state[state_key] = []
            st.rerun()

    paths = list(st.session_state.get(state_key) or [])
    if paths:
        for p in paths:
            st.caption(f"• `{p}`")
    else:
        st.caption("No files selected yet.")
    return paths


def _tkinter_available() -> bool:
    """Check tkinter availability without importing it on Streamlit's script thread."""
    try:
        return importlib.util.find_spec("tkinter") is not None
    except (ImportError, ValueError):
        return False
