"""
sitecustomize.py

Auto-save matplotlib figures when they are shown.
Enabled when ICESEE_SAVEFIG_DIR is set in the environment.

This is wrapper-side only (does not touch ICESEE submodule).
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def _enabled() -> bool:
    return bool(os.environ.get("ICESEE_SAVEFIG_DIR", "").strip())


def _get_out_dir() -> Path:
    d = os.environ.get("ICESEE_SAVEFIG_DIR", "").strip()
    if not d:
        # disabled
        return Path(".")
    out = Path(d).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_save_all_figures():
    # Import lazily to avoid forcing matplotlib in notebooks that don't use it
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not _enabled():
        return

    out_dir = _get_out_dir()

    # Save all open figures
    fignums = list(plt.get_fignums())
    if not fignums:
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    for n in fignums:
        try:
            fig = plt.figure(n)
            # Prefer a stable name; include fig number + timestamp
            fname = out_dir / f"fig_{n:02d}_{ts}.png"
            # Avoid overwriting in same second
            k = 1
            while fname.exists():
                fname = out_dir / f"fig_{n:02d}_{ts}_{k}.png"
                k += 1
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        except Exception:
            # Never break notebook execution just because saving failed
            pass


def _patch_matplotlib_show():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if getattr(plt.show, "__icesee_patched__", False):
        return

    _orig_show = plt.show

    def _show_patched(*args, **kwargs):
        # Save *before* show to capture state even if show clears in some backends
        _safe_save_all_figures()
        return _orig_show(*args, **kwargs)

    _show_patched.__icesee_patched__ = True  # type: ignore[attr-defined]
    plt.show = _show_patched  # type: ignore[assignment]


# Patch at import time (only matters if matplotlib is imported later)
try:
    _patch_matplotlib_show()
except Exception:
    pass