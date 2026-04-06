from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """
    Locate the ICESEE-GHUB repo root.

    Search order:
      1) ICESEE_GHUB_ROOT environment variable, if valid
      2) start path, if provided
      3) location of this file
      4) current working directory (last resort)

    A valid repo root must contain:
      - external/ICESEE
      - icesee_jupyter_book
    """
    env_root = os.environ.get("ICESEE_GHUB_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "external" / "ICESEE").exists() and (p / "icesee_jupyter_book").exists():
            return p

    candidates = []

    if start is not None:
        candidates.append(Path(start).resolve())

    # Start from this installed/source file location
    candidates.append(Path(__file__).resolve())

    # Last resort
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        p = candidate
        if p.is_file():
            p = p.parent

        while p != p.parent:
            if (p / "external" / "ICESEE").exists() and (p / "icesee_jupyter_book").exists():
                return p
            p = p.parent

    raise FileNotFoundError(
        "Could not locate repo root containing external/ICESEE and icesee_jupyter_book. "
        "Set ICESEE_GHUB_ROOT to the ICESEE-GHUB repo root if needed."
    )


REPO = find_repo_root()
BOOK = REPO / "icesee_jupyter_book"
EXT = REPO / "external" / "ICESEE"