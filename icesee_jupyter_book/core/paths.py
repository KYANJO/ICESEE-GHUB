
# ============================================================
# Repo discovery
# ============================================================
from __future__ import annotations

from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / "external" / "ICESEE").exists() and (p / "icesee_jupyter_book").exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Could not locate repo root containing external/ICESEE and icesee_jupyter_book."
    )

REPO = find_repo_root()
BOOK = REPO / "icesee_jupyter_book"
EXT = REPO / "external" / "ICESEE"