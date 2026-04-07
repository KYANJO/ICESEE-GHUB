# ============================================================
# YAML helpers
# ============================================================
from __future__ import annotations

import yaml
from pathlib import Path

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def dump_yaml(d: dict, p: Path) -> None:
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")