from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward until we find a directory that contains `external/ICESEE`.
    Works from any notebook CWD inside the GHUB repo.
    """
    root = (start or Path.cwd()).resolve()
    while root != root.parent and not (root / "external" / "ICESEE").exists():
        root = root.parent
    if not (root / "external" / "ICESEE").exists():
        raise FileNotFoundError("Could not locate repo root containing external/ICESEE")
    return root

def find_run_base_dir(
    start: Path | None = None,
    *,
    expected_rel_results: str = "results",
    h5_glob: str = "*.h5",
    max_depth: int = 6,
) -> Path:
    """
    Find the directory that *contains* a `results/` folder with at least one `.h5`.
    Search upward from `start`, then fall back to searching under the repo root.

    This is general across all ICESEE examples because they write to `results/`.
    """
    start = (start or Path.cwd()).resolve()

    # 1) Search upward: current dir, parent, grandparent, ...
    cur = start
    while True:
        results_dir = cur / expected_rel_results
        if results_dir.exists() and any(results_dir.glob(h5_glob)):
            return cur
        if cur == cur.parent:
            break
        cur = cur.parent

    # 2) Fallback: search under repo root
    root = find_repo_root(start)
    # look for any results/ directory containing .h5
    candidates = []
    for p in root.rglob(expected_rel_results):
        if p.is_dir() and any(p.glob(h5_glob)):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find any '{expected_rel_results}/' containing '{h5_glob}' "
            f"starting from {start} or under repo root {root}"
        )

    # Choose the most recently modified .h5 among all candidates
    best_results_dir = max(
        candidates,
        key=lambda d: max((f.stat().st_mtime for f in d.glob(h5_glob)), default=0),
    )
    return best_results_dir.parent

def run_notebook(
    input_nb: str | Path,
    output_nb: str | Path,
    parameters: Optional[Dict[str, Any]] = None,
    cwd: Optional[str | Path] = None,
    log_output: bool = True,
) -> Path:
    """
    Execute a notebook and write an executed copy to output_nb.

    - Does NOT require modifying the input notebook.
    - Parameters are optional and only used if the notebook references them.
    - `cwd` controls the execution working directory (critical for notebooks that use relative paths).

    Returns the resolved output path.
    """
    import papermill as pm

    input_nb = Path(input_nb).expanduser().resolve()
    output_nb = Path(output_nb).expanduser()

    if cwd is not None:
        cwd = Path(cwd).expanduser().resolve()
    else:
        # Default: execute where output lives (helps with relative `results/` paths)
        cwd = output_nb.parent.resolve()

    output_nb = (cwd / output_nb.name).resolve()
    output_nb.parent.mkdir(parents=True, exist_ok=True)

    if not input_nb.exists():
        raise FileNotFoundError(f"Input notebook not found: {input_nb}")

    pm.execute_notebook(
        input_path=str(input_nb),
        output_path=str(output_nb),
        parameters=parameters or {},
        cwd=str(cwd),
        log_output=log_output,
    )
    return output_nb


def run_postprocess_notebook(
    postprocess_nb: str | Path,
    *,
    base_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_name: str = "results_report.ipynb",
    parameters: Optional[Dict[str, Any]] = None,
) -> Path:
    base_dir = Path(base_dir).expanduser().resolve() if base_dir else find_run_base_dir()
    if out_dir is None:
        out_dir = base_dir / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nb = out_dir / out_name

    return run_notebook(
        input_nb=postprocess_nb,
        output_nb=out_nb,
        parameters=parameters,
        cwd=base_dir,   # will now be the *true* run directory
        log_output=True,
    )