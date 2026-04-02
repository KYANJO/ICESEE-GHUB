# ============================================================
#  Local run helpers
# ============================================================
from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime

from .paths import BOOK, REPO

def run_dir() -> Path:
    rd = BOOK / "icesee_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results").mkdir(exist_ok=True)
    (rd / "figures").mkdir(exist_ok=True)
    return rd


def force_external_icesee_env():
    external_dir = (REPO / "external").resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{external_dir}{os.pathsep}{env.get('PYTHONPATH','')}"
    env["PYTHONNOUSERSITE"] = "1"
    return env, external_dir


def mirror_assets_for_report(example_cfg: dict, rd: Path):
    base = example_cfg["base"]
    for a in example_cfg.get("assets", []):
        src = base / a
        if src.exists():
            dst = rd / a
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

def ensure_report_h5(rd: Path, example_cfg: dict, expected_prefix: str)->Path:
    """
    Ensure results/<expected_prefix>-<model>.h5 exists in rd for report notebooks.
    If missing, search for any *-<model>.h5 under rd/results or the example base, and copy.
    """
    model_name = example_cfg.get("model_name", "lorenz")
    exp = rd / "results" / f"{expected_prefix}-{model_name}.h5"
    if exp.exists():
        return exp

    # 1) search inside this run dir first
    candidates = sorted((rd / "results").glob(f"*-{model_name}.h5"), key=lambda p: p.stat().st_mtime, reverse=True)

    # 2) also search in the example base (some scripts write there)
    if not candidates:
        base = example_cfg["base"]
        candidates = sorted(base.glob(f"**/results/*-{model_name}.h5"), key=lambda p: p.stat().st_mtime, reverse=True)

    if candidates:
        exp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], exp)
        return exp

    return exp

def run_report_notebook(report_nb: Path | None, example_cfg: dict, rd: Path):

    try:
        import papermill as pm
    except Exception as e:
        raise RuntimeError(
            "papermill is required to run read_results.ipynb automatically. "
            "Install it (pip install papermill) or replace with nbclient."
        ) from e

    nb_out = rd / "report.ipynb"
    mirror_assets_for_report(example_cfg, rd)

    pm.execute_notebook(
        input_path=str(report_nb),
        output_path=str(nb_out),
        cwd=str(rd),
        log_output=True,
    )
    return nb_out
