"""Publication-quality figure saving.

Standalone figures are publication artifacts, so they save at 300 DPI and, by
default, also emit a vector sibling (SVG) next to the raster — journals expect
>=300 DPI raster or vector line art.
"""

import os
from typing import Optional

from castle.core.config import FIGURE_VECTOR_FORMAT, PUBLICATION_DPI


def save_publication_figure(
    fig,
    path: str,
    *,
    dpi: int = PUBLICATION_DPI,
    also_vector: bool = True,
    bbox_inches: Optional[str] = "tight",
) -> list:
    """Save *fig* at publication quality and return the written paths.

    Saves the raster at *path* (format inferred from its extension) at *dpi*. If
    *path* is itself a raster (png/jpg/tif) and *also_vector* is set, also writes
    a vector sibling (``<stem>.svg``) for figure-ready line art. A vector *path*
    (svg/pdf/eps) is written once and never duplicated.
    """
    written = []
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    written.append(path)

    stem, ext = os.path.splitext(path)
    is_vector = ext.lower() in (".svg", ".pdf", ".eps")
    if also_vector and not is_vector:
        vpath = f"{stem}.{FIGURE_VECTOR_FORMAT}"
        try:
            fig.savefig(vpath, bbox_inches=bbox_inches)
            written.append(vpath)
        except Exception:  # noqa: BLE001 — vector sibling is a bonus, never fail the save
            pass
    return written
