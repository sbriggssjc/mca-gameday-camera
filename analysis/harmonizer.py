from __future__ import annotations

"""Simple label harmonizer.

This module provides a minimal ``harmonize`` helper which normalises a label
into a canonical form.  The logic is intentionally lightweight – it simply
strips leading/trailing whitespace, collapses internal whitespace and converts
text to lower-case.  This mirrors the behaviour of the production harmoniser
used in the full pipeline.
"""

import re


def harmonize(label: str) -> str:
    """Return a canonical representation of ``label``.

    Parameters
    ----------
    label:
        Raw label string which may contain inconsistent spacing or casing.

    Returns
    -------
    str
        Normalised label suitable for *_canon fields.  Empty input results in
        an empty string.
    """

    if not label:
        return ""
    # Collapse whitespace and lower-case the label
    return re.sub(r"\s+", " ", label).strip().lower()
