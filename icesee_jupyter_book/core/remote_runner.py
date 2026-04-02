# ============================================================
# Remote backend (system ssh + Slurm)
# ============================================================

from __future__ import annotations

import os
import re
import time
import json
import getpass
import subprocess
from pathlib import Path
import urllib.request
import urllib.error

def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _ssh_base(host: str, user: str, port: int):
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-p", str(int(port)),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2",
        target,
    ]


def ssh_run(host: str, user: str, port: int, remote_cmd: str, timeout: int = 20):
    cmd = _ssh_base(host, user, port) + [remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def http_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None, timeout: int = 20):
    headers = headers or {}
    data = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        data = body
        headers = {**headers, "Content-Type": "application/json"}

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            # try JSON; fall back to text
            try:
                return resp.status, json.loads(txt), txt
            except Exception:
                return resp.status, None, txt
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return e.code, None, txt

def make_remote_run_dir(base_dir="~/r-arobel3-0", tag="icesee") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_dir.rstrip('/')}/{tag}-{ts}"


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -t {{TIME}}
#SBATCH -J {{JOB_NAME}}
#SBATCH -N {{NODES}}
#SBATCH -n {{NTASKS}}
#SBATCH --ntasks-per-node={{TPN}}
#SBATCH -p {{PARTITION}}
#SBATCH --mem={{MEM}}
{{ACCOUNT_LINE}}
#SBATCH -o {{OUTFILE}}
{{MAIL_LINES}}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# --- Modules (optional; provided by UI) ---
module purge || true
{{MODULE_LINES}}

# --- Activate ICESEE-Spack env ---
SPACK_PATH="{{SPACK_PATH}}"
if [ ! -d "${SPACK_PATH}" ]; then
  echo "[ERROR] SPACK_PATH does not exist: ${SPACK_PATH}"
  exit 2
fi
source "${SPACK_PATH}/scripts/activate.sh"

# --- Optional: add MPI from spack env to PATH/LD_LIBRARY_PATH ---
# Many spack envs already handle this; keep it robust:
if command -v spack >/dev/null 2>&1; then
  # Use the env inside the repo (if it exists)
  ENV_DIR="${SPACK_PATH}/.spack-env/icesee"
  if [ -d "${ENV_DIR}" ]; then
    MPI_DIR="$(spack -e "${ENV_DIR}" location -i openmpi 2>/dev/null || true)"
    if [ -n "${MPI_DIR}" ] && [ -d "${MPI_DIR}" ]; then
      export PATH="${MPI_DIR}/bin:${PATH}"
      export LD_LIBRARY_PATH="${MPI_DIR}/lib:${LD_LIBRARY_PATH:-}"
      echo "[info] MPI_DIR=${MPI_DIR}"
    else
      echo "[info] openmpi not resolved via spack -e ${ENV_DIR} (ok if system MPI is used)"
    fi
  fi
fi

# --- Exports (optional; provided by UI) ---
{{EXPORT_LINES}}

# --- Run config from UI ---
NP="{{NP}}"
NENS="{{NENS}}"
MODEL_NPROCS="{{MODEL_NPROCS}}"
RUN_SCRIPT="{{RUN_SCRIPT}}"
PARAMS_PATH="{{PARAMS_PATH}}"
EXAMPLE_DIR="{{EXAMPLE_DIR}}"

if [ ! -d "${EXAMPLE_DIR}" ]; then
  echo "[ERROR] EXAMPLE_DIR missing: ${EXAMPLE_DIR}"
  exit 3
fi

cd "${EXAMPLE_DIR}"

echo "[run] hostname=$(hostname)"
echo "[run] pwd=$(pwd)"
echo "[run] NP=${NP} NENS=${NENS} MODEL_NPROCS=${MODEL_NPROCS}"
echo "[run] RUN_SCRIPT=${RUN_SCRIPT}"
echo "[run] PARAMS_PATH=${PARAMS_PATH}"

# --- Launcher: prefer srun on Slurm clusters; fallback to mpirun ---
if command -v srun >/dev/null 2>&1; then
  /usr/bin/time -v \
    srun {{SRUN_MPI_FLAG}} -n "${NP}" \
      python "${RUN_SCRIPT}" \
        -F "${PARAMS_PATH}" \
        --Nens="${NENS}" \
        --model_nprocs="${MODEL_NPROCS}" \
        --verbose
else
  /usr/bin/time -v \
    mpirun -np "${NP}" \
      python "${RUN_SCRIPT}" \
        -F "${PARAMS_PATH}" \
        --Nens="${NENS}" \
        --model_nprocs="${MODEL_NPROCS}" \
        --verbose
fi

echo "=== Finished ==="
"""

def slurm_optional_lines(account: str, mail: str) -> tuple[str, str]:
    account_line = f"#SBATCH -A {account.strip()}" if account.strip() else ""
    if mail.strip():
        mail_lines = "\n".join([
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            f"#SBATCH --mail-user={mail.strip()}",
        ])
    else:
        mail_lines = ""
    return account_line, mail_lines

def sanitize_multiline(s: str) -> str:
    # keep user text but strip trailing spaces
    return "\n".join([ln.rstrip() for ln in (s or "").splitlines() if ln.strip() != ""])

def render_slurm_script(d: dict) -> str:
    txt = SLURM_TEMPLATE
    for k, v in d.items():
        txt = txt.replace("{{" + k + "}}", str(v))
    return txt


def remote_write_text(host: str, user: str, port: int, remote_path: str, text: str, timeout: int = 30):
    cmd = f"""
set -e
mkdir -p $(dirname {sh_quote(remote_path)})
cat > {sh_quote(remote_path)} <<'EOF_ICESEE'
{text.rstrip()}
EOF_ICESEE
"""
    r = ssh_run(host, user, port, cmd, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)

def remote_stage_and_submit(
    *,
    host: str,
    user: str,
    port: int,
    remote_dir: str,
    params_text: str,
    slurm_text: str,
) -> str:
    # stage files in remote_dir
    remote_write_text(host, user, port, f"{remote_dir}/params.yaml", params_text, timeout=30)
    remote_write_text(host, user, port, f"{remote_dir}/slurm_run.sh", slurm_text, timeout=30)

    # make executable and submit
    r = ssh_run(host, user, port, f"chmod +x {sh_quote(remote_dir+'/slurm_run.sh')} && cd {sh_quote(remote_dir)} && sbatch slurm_run.sh", timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)

    m = re.search(r"Submitted batch job\s+(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Could not parse JobID from:\n{r.stdout}\n{r.stderr}")
    return m.group(1)

def ensure_local_ssh_key(key_type: str = "ed25519") -> tuple[Path, Path]:
    """
    Ensure ~/.ssh/id_<type> and .pub exist. If not, generate them non-interactively.
    Returns (private_key_path, public_key_path).
    """
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(ssh_dir, 0o700)

    priv = ssh_dir / f"id_{key_type}"
    pub  = ssh_dir / f"id_{key_type}.pub"

    if pub.exists() and priv.exists():
        return priv, pub

    # Generate keypair
    cmd = [
        "ssh-keygen",
        "-t", key_type,
        "-f", str(priv),
        "-N", "",                 # empty passphrase (for non-interactive GHUB use)
        "-C", f"icesee-{getpass.getuser()}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout or "ssh-keygen failed")

    os.chmod(priv, 0o600)
    os.chmod(pub, 0o644)
    return priv, pub


def _paramiko_connect_password(host: str, user: str, port: int, password: str, timeout: int = 20):
    try:
        import paramiko
    except Exception as e:
        raise RuntimeError(
            "Paramiko is required for password bootstrap. Install with: pip install paramiko"
        ) from e

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=int(port),
        username=user,
        password=password,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def remote_install_pubkey_with_password(
    *,
    host: str,
    user: str,
    port: int,
    password: str,
    pubkey_text: str
):
    """
    Connect using Paramiko password auth and add pubkey to ~/.ssh/authorized_keys safely.
    Idempotent: won't duplicate the key if already installed.
    """
    client = _paramiko_connect_password(host, user, port, password, timeout=25)

    # Normalize key line
    key_line = pubkey_text.strip()
    if not key_line or "ssh-" not in key_line:
        client.close()
        raise ValueError("Public key text looks invalid.")

    # Remote commands: create ~/.ssh, set perms, append key if missing
    # Using sh to keep it portable.
    cmd = f"""
set -e
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
# Append key if not already present
grep -Fqx {sh_quote(key_line)} ~/.ssh/authorized_keys || echo {sh_quote(key_line)} >> ~/.ssh/authorized_keys
echo OK
"""
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    client.close()

    if rc != 0:
        raise RuntimeError(err or out or "Failed to update authorized_keys")


def explain_ssh_failure_hint(stderr: str) -> str:
    s = (stderr or "").lower()
    if "permission denied" in s:
        return "Auth failed (Permission denied). If you normally type a password in a terminal, use Bootstrap with password once."
    if "could not resolve hostname" in s:
        return "Host not reachable / DNS issue."
    if "operation timed out" in s or "connection timed out" in s:
        return "Connection timed out (VPN/firewall/host unreachable)."
    if "keyboard-interactive" in s or "authentication" in s:
        return "Interactive auth is being requested. BatchMode blocks it; bootstrap keys or use OnDemand."
    return "SSH failed. Check host/user/VPN and whether passwordless key auth is enabled."