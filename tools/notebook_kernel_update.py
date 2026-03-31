import json
from pathlib import Path

ROOT = Path(".")  # change if needed

new_kernelspec = {
    "display_name": "Python3 (icesee1)",
    "language": "python",
    "name": "icesee1"
}

for nb_path in ROOT.rglob("*.ipynb"):
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)

        nb.setdefault("metadata", {})
        nb["metadata"]["kernelspec"] = new_kernelspec

        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"Updated: {nb_path}")
    except Exception as e:
        print(f"Failed: {nb_path} -> {e}")