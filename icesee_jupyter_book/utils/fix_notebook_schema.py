from __future__ import annotations
import json
from pathlib import Path

def _fix_one(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    # If notebook is nbformat 4 and minor < 5, bump to 5 (allows cell.id)
    if nb.get("nbformat", 4) == 4:
        minor = int(nb.get("nbformat_minor", 0))
        if minor < 5:
            nb["nbformat_minor"] = 5
            changed = True

    # Also strip any existing cell ids (extra safety)
    for cell in nb.get("cells", []):
        if "id" in cell:
            del cell["id"]
            changed = True

    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    return changed

def fix_all_notebooks(book_root: Path) -> None:
    nb_dir = book_root / "icesee_jupyter_notebooks"
    if nb_dir.exists():
        for nb in nb_dir.rglob("*.ipynb"):
            _fix_one(nb)

    # Optional: if you execute notebooks from external/ICESEE (papermill), fix them too
    ext = book_root.parent / "external" / "ICESEE"
    if ext.exists():
        for nb in ext.rglob("*.ipynb"):
            _fix_one(nb)