# ============================================================
#  Find runner/template/report
# ============================================================
from __future__ import annotations

from pathlib import Path
from .paths import BOOK


def find_run_script(cfg: dict) -> Path:
    base = cfg["base"]
    rs = cfg.get("run_script")
    if rs and (base / rs).exists():
        return base / rs
    cands = list(base.rglob("run_da_*.py")) + list(base.rglob("run_*.py"))
    cands = [c for c in cands if c.is_file()]
    if not cands:
        raise FileNotFoundError(f"No run script found under {base}")
    cands.sort(key=lambda x: len(str(x)))
    return cands[0]


def find_params_template(cfg: dict) -> Path:
    wrapper_template = BOOK / "params.yaml"
    if wrapper_template.exists():
        return wrapper_template
    base = cfg["base"]
    p = base / (cfg.get("params") or "params.yaml")
    if p.exists():
        return p
    cands = list(base.rglob("params.yaml"))
    if not cands:
        raise FileNotFoundError(f"No params.yaml found under {base}")
    cands.sort(key=lambda x: len(str(x)))
    return cands[0]


def find_report_notebook(cfg: dict) -> Path | None:
    nb = cfg.get("report_nb")
    if not nb:
        return None
    p = cfg["base"] / nb
    return p if p.exists() else None