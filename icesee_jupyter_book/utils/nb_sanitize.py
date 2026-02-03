#!/usr/bin/env python3
"""
Sanitize notebooks for Jupyter Book / older nbformat validators:

- Remove 'id' field from cells (fixes: "Notebook JSON is invalid: 'id' unexpected")
- Optionally tag all code cells with 'remove-input' so UI pages don't show code
- Ensure the notebook has an H1 title as the first cell

Usage:
  python utils/nb_sanitize.py icesee_jupyter_notebooks/run_center.ipynb --hide-input --title "ICESEE-OnLINE"
  python utils/nb_sanitize.py icesee_jupyter_notebooks/latest_run.ipynb --hide-input --title "Latest Run"

Or sanitize all notebooks (IDs only):
  python utils/nb_sanitize.py icesee_jupyter_notebooks/*.ipynb
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

def load_nb(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_nb(path: Path, nb: Dict[str, Any]) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

def remove_cell_ids(nb: Dict[str, Any]) -> int:
    n = 0
    for cell in nb.get("cells", []):
        if "id" in cell:
            del cell["id"]
            n += 1
    return n

def tag_remove_input_on_code_cells(nb: Dict[str, Any]) -> int:
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

def ensure_h1_title(nb: Dict[str, Any], title: str) -> bool:
    """
    Make sure the first cell is a markdown H1 title. If not, insert it.
    """
    cells: List[Dict[str, Any]] = nb.get("cells", [])
    if not cells:
        nb["cells"] = [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"]
        }]
        return True

    first = cells[0]
    if first.get("cell_type") == "markdown":
        # source can be list[str] or str
        src = first.get("source", "")
        src_text = "".join(src) if isinstance(src, list) else str(src)
        if src_text.lstrip().startswith("# "):
            return False

    nb["cells"] = [{
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n"]
    }] + cells
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", help="Notebook paths (.ipynb)")
    ap.add_argument("--hide-input", action="store_true", help="Add remove-input tag to all code cells")
    ap.add_argument("--title", type=str, default="", help="Ensure an H1 title cell is present")
    args = ap.parse_args()

    for nb_path_str in args.notebooks:
        nb_path = Path(nb_path_str)
        if not nb_path.exists():
            print(f"[skip] not found: {nb_path}")
            continue
        if nb_path.suffix.lower() != ".ipynb":
            print(f"[skip] not a notebook: {nb_path}")
            continue

        nb = load_nb(nb_path)
        removed = remove_cell_ids(nb)
        tagged = tag_remove_input_on_code_cells(nb) if args.hide_input else 0
        titled = ensure_h1_title(nb, args.title) if args.title else False

        save_nb(nb_path, nb)
        print(f"[ok] {nb_path.name}: removed_ids={removed}, tagged_remove_input={tagged}, inserted_title={titled}")

if __name__ == "__main__":
    main()