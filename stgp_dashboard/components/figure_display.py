"""Constrained figure display for the Results gallery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from stgp_dashboard.components._streamlit_compat import width_kwargs


def show_figure_image(
    png_path: str | Path,
    *,
    category: str = "other",
    caption: Optional[str] = None,
    pdf_path: str | Path | None = None,
) -> None:
    """Display a saved figure scaled to the page width.

    ``category`` is retained for call-site compatibility. When ``pdf_path``
    exists, offer a PDF download for full-resolution viewing.
    """
    _ = category
    path = Path(png_path)
    if not path.is_file():
        return

    st.image(str(path), caption=caption, **width_kwargs(stretch=True))

    pdf = Path(pdf_path) if pdf_path else None
    if pdf is not None and pdf.is_file():
        # Stable, path-safe widget key (Windows paths break Streamlit keys).
        key = f"dl_pdf_{pdf.stem}_{abs(hash(str(pdf.resolve()))) % 10_000_000}"
        st.download_button(
            label="Download PDF",
            data=pdf.read_bytes(),
            file_name=pdf.name,
            mime="application/pdf",
            key=key,
            **width_kwargs(stretch=False),
        )
