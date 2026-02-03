from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def _load_nb(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _save_nb(path: Path, nb: Dict[str, Any]) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

def _remove_cell_ids(nb: Dict[str, Any]) -> int:
    n = 0
    for cell in nb.get("cells", []):
        if "id" in cell:
            del cell["id"]
            n += 1
    return n

def _tag_remove_input_on_code_cells(nb: Dict[str, Any]) -> int:
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        md = cell.setdefault("metadata", {})
        tags = md.setdefault("tags", [])
        if "remove-input" not in tags:
            tags.append("remove-input")
            n += 1
    return n

def _ensure_h1_title(nb: Dict[str, Any], title: str) -> bool:
    cells: List[Dict[str, Any]] = nb.get("cells", [])
    if not cells:
        nb["cells"] = [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"],
        }]
        return True

    first = cells[0]
    if first.get("cell_type") == "markdown":
        src = first.get("source", "")
        src_text = "".join(src) if isinstance(src, list) else str(src)
        if src_text.lstrip().startswith("# "):
            return False

    nb["cells"] = [{
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n"],
    }] + cells
    return True

def sanitize_notebook(
    nb_path: Path,
    *,
    hide_input: bool = False,
    title: Optional[str] = None,
) -> Dict[str, int]:
    """
    Mutates notebook in place:
      - remove cell 'id'
      - optionally tag code cells with 'remove-input'
      - optionally ensure first cell is H1 title
    """
    nb = _load_nb(nb_path)
    removed = _remove_cell_ids(nb)
    tagged = _tag_remove_input_on_code_cells(nb) if hide_input else 0
    titled = 1 if (title and _ensure_h1_title(nb, title)) else 0
    _save_nb(nb_path, nb)
    return {"removed_ids": removed, "tagged_remove_input": tagged, "inserted_title": titled}

def sanitize_tree(book_root: Path) -> None:
    """
    Called from Sphinx conf.py before notebooks are executed.

    - removes cell ids from ALL notebooks (compat)
    - applies remove-input + H1 title ONLY to GUI notebooks
    """
    nb_dir = book_root / "icesee_jupyter_notebooks"
    if not nb_dir.exists():
        return

    # GUI notebooks: hide code, ensure title
    gui_targets = {
        "run_center.ipynb": "ICESEE-OnLINE",
        "latest_run.ipynb": "Latest Run",
    }

    # 1) remove ids everywhere
    for nb in nb_dir.glob("*.ipynb"):
        try:
            sanitize_notebook(nb, hide_input=False, title=None)
        except Exception:
            # don't hard-fail build on sanitize; Sphinx will show the real error
            pass

    # 2) apply GUI rules
    for name, title in gui_targets.items():
        nb = nb_dir / name
        if nb.exists():
            sanitize_notebook(nb, hide_input=True, title=title)