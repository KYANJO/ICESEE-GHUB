"""
@author: Brian Kyanjo
@created: February 2026

---
ICESEE-GHUB Bootstrap

This module ensures that the ICESEE submodule located in:

    external/ICESEE

is importable in every notebook, without requiring pip installation.

Usage (first cell in notebooks):

    from _bootstrap._bootstrap import enable_icesee
    enable_icesee()
"""

import sys
import os
from pathlib import Path


def enable_icesee(verbose: bool = True) -> Path:
    """
    Add the GHUB external/ directory to sys.path so that:

        import ICESEE

    works reliably in all notebooks.

    Returns
    -------
    Path
        The resolved external/ directory path.
    """

    # ------------------------------------------------------------
    # 1. Locate repo root by walking upward until external/ICESEE exists
    # ------------------------------------------------------------
    root = Path.cwd()

    while root != root.parent and not (root / "external" / "ICESEE").exists():
        root = root.parent

    external_dir = (root / "external").resolve()
    icesee_dir = external_dir / "ICESEE"

    if not (icesee_dir / "__init__.py").exists():
        raise FileNotFoundError(
            f"[ICESEE bootstrap] Could not find ICESEE package at: {icesee_dir}"
        )

    # ------------------------------------------------------------
    # 2. Add external/ to sys.path (this enables `import ICESEE`)
    # ------------------------------------------------------------
    if str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))

    # ------------------------------------------------------------
    # 3. (Optional) Also set PYTHONPATH for subprocesses
    # ------------------------------------------------------------
    os.environ["PYTHONPATH"] = f"{external_dir}:{os.environ.get('PYTHONPATH','')}"

    if verbose:
        import ICESEE

        print("[ICESEE bootstrap] external path enabled:", external_dir)
        print("[ICESEE bootstrap] ICESEE loaded from:", ICESEE.__file__)

    return external_dir