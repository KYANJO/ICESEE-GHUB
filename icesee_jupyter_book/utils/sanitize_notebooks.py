#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]  # icesee_jupyter_book/
NB_DIR = BOOK / "icesee_jupyter_notebooks"
EXT_NB = BOOK.parent / "external" / "ICESEE"  # optional: sanitize submodule notebooks too

def sanitize_ipynb(nb_path: Path) -> tuple[int, bool]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Force old-compatible minor version if present
    changed_minor = False
    if nb.get("nbformat", 4) == 4 and nb.get("nbformat_minor", 0) > 4:
        nb["nbformat_minor"] = 4
        changed_minor = True

    removed = 0
    for cell in nb.get("cells", []):
        if "id" in cell:
            del cell["id"]
            removed += 1

    if removed or changed_minor:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    return removed, changed_minor

def walk_and_sanitize(root: Path) -> int:
    n_changed = 0
    for nb_path in root.rglob("*.ipynb"):
        removed, changed_minor = sanitize_ipynb(nb_path)
        if removed or changed_minor:
            print(f"[ok] {nb_path.relative_to(BOOK.parent)}  removed_ids={removed}  minor->4={changed_minor}")
            n_changed += 1
    return n_changed

if __name__ == "__main__":
    total = 0
    if NB_DIR.exists():
        total += walk_and_sanitize(NB_DIR)

    # Optional: sanitize notebooks inside the submodule if any get executed or copied
    # Comment this out if you never want to touch files under external/ICESEE
    if EXT_NB.exists():
        total += walk_and_sanitize(EXT_NB)

    print(f"\nDone. Notebooks modified: {total}")