#!/bin/bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
repoRoot="$(cd "${scriptDir}/.." && pwd)"

BOOK_DIR="${repoRoot}/icesee_jupyter_book/_build/html"
VOILA_NOTEBOOK="${repoRoot}/icesee_jupyter_book/icesee_jupyter_notebooks/run_center_voila.ipynb"

BOOK_PORT=8081
VOILA_PORT=8866
ROUTER_PORT=8080

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[ICESEE] repoRoot=${repoRoot}"
echo "[ICESEE] book=${BOOK_DIR}"
echo "[ICESEE] voila notebook=${VOILA_NOTEBOOK}"
echo "[ICESEE] python=${PYTHON_BIN}"

if [ ! -d "${BOOK_DIR}" ]; then
  echo "[ICESEE] Book build directory not found. Building book first..."
  (
    cd "${repoRoot}"
    "${PYTHON_BIN}" -m jupyter_book build icesee_jupyter_book
  )
fi

if [ ! -d "${BOOK_DIR}" ]; then
  echo "[ICESEE][ERROR] Book build directory still not found: ${BOOK_DIR}"
  exit 2
fi

if [ ! -f "${VOILA_NOTEBOOK}" ]; then
  echo "[ICESEE][ERROR] Voilà notebook not found: ${VOILA_NOTEBOOK}"
  exit 2
fi

echo "[ICESEE] Starting static book on 127.0.0.1:${BOOK_PORT}"
(
  cd "${BOOK_DIR}"
  exec "${PYTHON_BIN}" -m http.server --bind 127.0.0.1 "${BOOK_PORT}"
) &
BOOK_PID=$!

echo "[ICESEE] Starting Voilà on 127.0.0.1:${VOILA_PORT}"
(
  cd "${repoRoot}"
  exec "${PYTHON_BIN}" -m voila "${VOILA_NOTEBOOK}" \
    --no-browser \
    --Voila.ip=127.0.0.1 \
    --port="${VOILA_PORT}" \
    # --Voila.base_url=/icesee-gui/
) &
VOILA_PID=$!

echo "[ICESEE] Starting router on 127.0.0.1:${ROUTER_PORT}"
(
  cd "${repoRoot}"
  exec "${PYTHON_BIN}" "${repoRoot}/bin/icesee_router.py" \
    --listen-port "${ROUTER_PORT}" \
    --book-port "${BOOK_PORT}" \
    --voila-port "${VOILA_PORT}"
) &
ROUTER_PID=$!

wait "${BOOK_PID}" "${VOILA_PID}" "${ROUTER_PID}"