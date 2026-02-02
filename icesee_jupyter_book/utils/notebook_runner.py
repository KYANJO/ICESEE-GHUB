from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Repo discovery utilities
# -------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward until we find a directory that contains external/ICESEE.
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
    results_rel: str = "results",
    h5_glob: str = "*.h5",
) -> Path:
    """
    Find the directory that contains a `results/` folder with at least one `.h5`.

    Strategy:
    1) Search upward from start (cwd, parent, ...)
    2) Fallback: search under repo root for any results/ containing .h5 and pick newest
    """
    start = (start or Path.cwd()).resolve()

    # 1) Search upward
    cur = start
    while True:
        results_dir = cur / results_rel
        if results_dir.exists() and any(results_dir.glob(h5_glob)):
            return cur
        if cur == cur.parent:
            break
        cur = cur.parent

    # 2) Fallback: search under repo root
    root = find_repo_root(start)
    candidates: List[Path] = []
    for p in root.rglob(results_rel):
        if p.is_dir() and any(p.glob(h5_glob)):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find any '{results_rel}/' containing '{h5_glob}' "
            f"starting from {start} or under repo root {root}"
        )

    # Choose the results dir with the most recently modified .h5
    def newest_mtime(d: Path) -> float:
        mtimes = [f.stat().st_mtime for f in d.glob(h5_glob)]
        return max(mtimes) if mtimes else 0.0

    best_results_dir = max(candidates, key=newest_mtime)
    return best_results_dir.parent


# -------------------------
# Snapshot / mirroring
# -------------------------

def _fingerprint_file(p: Path) -> Tuple[int, float]:
    st = p.stat()
    return (st.st_size, st.st_mtime)


def snapshot_dir(d: Path, patterns: List[str] | None = None) -> Dict[str, Tuple[int, float]]:
    """
    Snapshot files under directory d. Returns {relative_path: (size, mtime)}.
    """
    d = d.resolve()
    patterns = patterns or ["*"]
    out: Dict[str, Tuple[int, float]] = {}

    for pat in patterns:
        for p in d.rglob(pat):
            if p.is_file():
                rel = str(p.relative_to(d))
                out[rel] = _fingerprint_file(p)
    return out


def diff_snapshots(before: Dict[str, Tuple[int, float]], after: Dict[str, Tuple[int, float]]) -> List[str]:
    """Return list of relative paths that are new or changed."""
    changed: List[str] = []
    for rel, fp in after.items():
        if rel not in before or before[rel] != fp:
            changed.append(rel)
    return sorted(changed)


def copy_changed_files(src_dir: Path, dst_dir: Path, rel_paths: List[str]) -> List[Path]:
    """
    Copy selected rel_paths from src_dir to dst_dir preserving structure.
    Returns list of destination paths created.
    """
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for rel in rel_paths:
        sp = src_dir / rel
        dp = dst_dir / rel
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        written.append(dp)
    return written


# -------------------------
# Notebook execution
# -------------------------

def run_notebook(
    input_nb: str | Path,
    output_nb: str | Path,
    parameters: Optional[Dict[str, Any]] = None,
    cwd: Optional[str | Path] = None,
    log_output: bool = True,
    inject_autosave: bool = False,
    autosave_dir: Optional[str | Path] = None,
) -> Path:
    """
    Execute a notebook and write an executed copy to output_nb.

    - Does NOT modify the original notebook.
    - Optionally injects a setup cell into a TEMP copy to autosave matplotlib figs.
    - `cwd` is critical for notebooks that use relative paths (e.g., results/...).
    """
    import papermill as pm
    import nbformat
    from pathlib import Path

    input_nb = Path(input_nb).expanduser().resolve()
    if not input_nb.exists():
        raise FileNotFoundError(f"Input notebook not found: {input_nb}")

    output_nb = Path(output_nb).expanduser()
    if cwd is not None:
        cwd = Path(cwd).expanduser().resolve()
    else:
        cwd = output_nb.parent.resolve()

    # Write output notebook inside cwd
    output_nb = (cwd / output_nb.name).resolve()
    output_nb.parent.mkdir(parents=True, exist_ok=True)

    # ---- Create a temporary copy (so we never touch the original) ----
    tmp_nb = output_nb.with_name(output_nb.stem + "._tmp_execute.ipynb")

    nb = nbformat.read(str(input_nb), as_version=4)

    if inject_autosave:
        if autosave_dir is None:
            raise ValueError("inject_autosave=True requires autosave_dir")

        autosave_dir = Path(autosave_dir).expanduser().resolve()
        autosave_dir.mkdir(parents=True, exist_ok=True)

        setup_code = f"""
# --- ICESEE-GHUB wrapper: autosave matplotlib figures (injected) ---
import os, time
from pathlib import Path

_SAVE_DIR = Path(r"{str(autosave_dir)}").resolve()
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def _save_all_figs(prefix="fig"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    nums = list(plt.get_fignums())
    if not nums:
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    for n in nums:
        try:
            fig = plt.figure(n)
            fname = _SAVE_DIR / f"{{prefix}}_{{n:02d}}_{{ts}}.png"
            k = 1
            while fname.exists():
                fname = _SAVE_DIR / f"{{prefix}}_{{n:02d}}_{{ts}}_{{k}}.png"
                k += 1
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        except Exception:
            pass

# Patch plt.show() so every display saves figures
try:
    import matplotlib.pyplot as plt
    _orig_show = plt.show
    def _show(*a, **kw):
        _save_all_figs("fig")
        return _orig_show(*a, **kw)
    plt.show = _show
except Exception:
    pass

# Also save at kernel exit (covers notebooks that never call plt.show)
import atexit
@atexit.register
def _final_save():
    _save_all_figs("fig_exit")

print(f"[ICESEE wrapper] autosave enabled -> {{_SAVE_DIR}}")
"""

        nb.cells.insert(
            0,
            nbformat.v4.new_code_cell(setup_code.strip())
        )

    nbformat.write(nb, str(tmp_nb))

    try:
        pm.execute_notebook(
            input_path=str(tmp_nb),
            output_path=str(output_nb),
            parameters=parameters or {},
            cwd=str(cwd),
            log_output=log_output,
        )
    finally:
        # Clean up temp notebook copy
        if tmp_nb.exists():
            try:
                tmp_nb.unlink()
            except Exception:
                pass

    return output_nb


def run_postprocess_notebook(
    postprocess_nb: str | Path,
    *,
    base_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    out_name: str = "results_report.ipynb",
    parameters: Optional[Dict[str, Any]] = None,
    results_rel: str = "results",
) -> Path:
    """
    Execute a post-processing notebook in `base_dir` so relative paths like
    results/... resolve correctly. Save the executed notebook to out_dir.

    If base_dir is None, it will be auto-detected by searching for results/*.h5.
    """
    base_dir = Path(base_dir).expanduser().resolve() if base_dir else find_run_base_dir()
    if out_dir is None:
        out_dir = base_dir / results_rel
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nb = out_dir / out_name

    return run_notebook(
        input_nb=postprocess_nb,
        output_nb=out_nb,
        parameters=parameters,
        cwd=base_dir,
        log_output=True,
    )


def run_postprocess_to_wrapper(
    postprocess_nb: str | Path,
    *,
    base_dir: str | Path | None = None,
    results_rel: str = "results",
    wrapper_out_dir: str | Path = "icesee_jupyter_book/_static/generated",
    out_name: str = "results_report.ipynb",
    patterns: List[str] | None = None,
    enable_autosave_figs: bool = True,
) -> Path:
    """
    Execute a postprocess notebook and mirror new/changed outputs from base_dir/results/
    into a wrapper-owned directory (published by Jupyter Book).

    Also enables figure autosaving (wrapper-side) via ICESEE_SAVEFIG_DIR,
    so plots become visible as PNGs without modifying the submodule notebooks.
    """
    base_dir = Path(base_dir).expanduser().resolve() if base_dir else find_run_base_dir()
    results_dir = (base_dir / results_rel).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Expected results directory at {results_dir}")

    # What artifacts to mirror
    if patterns is None:
        patterns = ["*.ipynb", "*.png", "*.svg", "*.pdf", "*.jpg", "*.jpeg", "*.txt", "*.csv", "*.h5"]

    before = snapshot_dir(results_dir, patterns=patterns)

    # Resolve wrapper output dir
    root = find_repo_root(base_dir)
    wrapper_out_dir = Path(wrapper_out_dir)
    if not wrapper_out_dir.is_absolute():
        wrapper_out_dir = (root / wrapper_out_dir).resolve()
    wrapper_out_dir.mkdir(parents=True, exist_ok=True)

    # Use base_dir name as a generic tag (e.g., lorenz96, flowline, etc.)
    tag = base_dir.name
    dst_dir = wrapper_out_dir / tag
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Enable autosave figs for the executed notebook
    if enable_autosave_figs:
        fig_dir = dst_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ICESEE_SAVEFIG_DIR"] = str(fig_dir)

        # Ensure sitecustomize can be found by the kernel:
        # Put icesee_jupyter_book on PYTHONPATH (wrapper-side)
        book_dir = (root / "icesee_jupyter_book").resolve()
        os.environ["PYTHONPATH"] = f"{book_dir}:{os.environ.get('PYTHONPATH','')}"

    # Execute notebook; write executed copy into results/
    # out_nb = run_postprocess_notebook(
    #     postprocess_nb,
    #     base_dir=base_dir,
    #     out_dir=results_dir,
    #     out_name=out_name,
    # )

    # Enable autosave figures into wrapper-owned folder
    fig_dir = dst_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_nb = run_notebook(
        input_nb=postprocess_nb,
        output_nb=results_dir / out_name,
        cwd=base_dir,                 # important for results/... paths
        log_output=True,
        inject_autosave=True,         #  force autosave inside executed kernel
        autosave_dir=fig_dir,         #  write images to wrapper
    )

    # Always export the executed report to HTML so plots are visible
    html_report = dst_dir / "read_results_report.html"
    export_notebook_to_html(out_nb, html_report)

    # Extract any rendered images from notebook outputs (works for plotly/mpl/etc.)
    img_dir = dst_dir / "figures"
    n = extract_output_images(out_nb, img_dir)
    print(f"[ICESEE wrapper] extracted {n} images from executed notebook outputs -> {img_dir}")

    after = snapshot_dir(results_dir, patterns=patterns)
    changed = diff_snapshots(before, after)

    # Mirror changed artifacts to wrapper dir
    copy_changed_files(results_dir, dst_dir, changed)

    return out_nb

def export_notebook_to_html(executed_nb: Path, html_out: Path) -> Path:
    """
    Export an executed notebook to a standalone HTML file.
    Works for matplotlib, plotly, rich output, images, etc.
    """
    import nbformat
    from nbconvert import HTMLExporter

    executed_nb = Path(executed_nb).resolve()
    html_out = Path(html_out).resolve()
    html_out.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(str(executed_nb), as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input = True  # optional: hide code, keep outputs
    body, _ = exporter.from_notebook_node(nb)

    html_out.write_text(body, encoding="utf-8")
    return html_out

def extract_output_images(executed_nb: Path, out_dir: Path) -> int:
    """
    Extract image outputs (png/jpeg/svg) from executed notebook cells and write to files.
    Returns number of images written.
    """
    import base64
    from pathlib import Path
    import nbformat

    executed_nb = Path(executed_nb).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(str(executed_nb), as_version=4)
    count = 0

    for ci, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for oi, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            # PNG
            if "image/png" in data:
                raw = data["image/png"]
                b = base64.b64decode(raw) if isinstance(raw, str) else raw
                fname = out_dir / f"cell{ci:03d}_out{oi:02d}.png"
                fname.write_bytes(b)
                count += 1
            # JPEG
            if "image/jpeg" in data:
                raw = data["image/jpeg"]
                b = base64.b64decode(raw) if isinstance(raw, str) else raw
                fname = out_dir / f"cell{ci:03d}_out{oi:02d}.jpg"
                fname.write_bytes(b)
                count += 1
            # SVG
            if "image/svg+xml" in data:
                svg = data["image/svg+xml"]
                fname = out_dir / f"cell{ci:03d}_out{oi:02d}.svg"
                fname.write_text(svg if isinstance(svg, str) else "".join(svg), encoding="utf-8")
                count += 1

    return count