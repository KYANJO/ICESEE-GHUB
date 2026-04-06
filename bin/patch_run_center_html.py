#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

REDIRECT_BLOCK = """
<div class="redirect-wrap">
  <div class="redirect-title">ICESEE GUI</div>
  <div class="redirect-sub">
    Launching interactive GUI...
  </div>

  <a class="redirect-btn" href="http://127.0.0.1:8866/">
    Open ICESEE GUI
  </a>
</div>

<style>
.redirect-wrap {
  text-align: center;
  margin-top: 80px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
}

.redirect-title {
  font-size: 32px;
  font-weight: 800;
  margin-bottom: 10px;
}

.redirect-sub {
  color: rgba(0,0,0,.7);
  margin-bottom: 20px;
}

.redirect-btn {
  display: inline-block;
  padding: 14px 22px;
  border-radius: 10px;
  background: #0d6efd;
  color: white !important;
  text-decoration: none;
  font-weight: 700;
}

.redirect-btn:hover {
  background: #0b5ed7;
}
</style>

<script>
  setTimeout(function() {
    window.location.href = "http://127.0.0.1:8866/";
  }, 800);
</script>
""".strip()


def patch_html(path: Path) -> None:
    html = path.read_text(encoding="utf-8")

    pattern = re.compile(
        r'(<article class="bd-article">)(.*?)(</article>)',
        re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{REDIRECT_BLOCK}\n{match.group(3)}"

    new_html, n = pattern.subn(repl, html, count=1)

    if n != 1:
        raise RuntimeError(
            f"Could not uniquely locate <article class=\"bd-article\"> in {path}"
        )

    path.write_text(new_html, encoding="utf-8")
    print(f"[patch_run_center_html] patched: {path}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_run_center_html.py /path/to/run_center.html")

    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    patch_html(path)


if __name__ == "__main__":
    main()