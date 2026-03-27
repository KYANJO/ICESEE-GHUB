# icesee_jupyter_book/ui/icesee_gateway_ui.py
# ------------------------------------------------------------
# ICESEE GHUB Gateway UI (Local / Remote(SSH+Slurm) / Cloud(AWS Batch))
#
# Usage (in a notebook / Jupyter Book page):
#   from icesee_jupyter_book.ui.icesee_gateway_ui import build_icesee_ui
#   build_icesee_ui()
#
# Notes:
# - Remote uses *system ssh* (recommended for GHUB). Requires non-interactive auth.
# - Cloud uses AWS CLI + AWS Batch. Requires AWS credentials available in the container.
# - Local runs directly in the GHUB notebook kernel.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import time
import json
import yaml
import shutil
import getpass
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error
from urllib.parse import urlencode

import ipywidgets as W
from IPython.display import display, Image


# ============================================================
# 0) Repo discovery
# ============================================================
def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    while p != p.parent:
        if (p / "external" / "ICESEE").exists() and (p / "icesee_jupyter_book").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate repo root containing external/ICESEE and icesee_jupyter_book.")


REPO = find_repo_root()
BOOK = REPO / "icesee_jupyter_book"
EXT = REPO / "external" / "ICESEE"


# ============================================================
# 1) Example registry (edit here)
# ============================================================
EXAMPLES = {
   "Lorenz-96 (fully runnable in GHUB)": dict(
        enabled=True,
        # local
        base=EXT / "applications" / "lorenz_model" / "examples" / "lorenz96",
        run_script="run_da_lorenz96.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="lorenz",
        figures_dir="figures",
        # remote (relative to ICESEE-Spack repo root)
        remote_rel="ICESEE/applications/lorenz_model/examples/lorenz96",
        remote_sbatch=None,  # likely none
    ),
    "ISSM (fully runable in Remote)": dict(
        enabled=True,
        base=EXT / "applications" / "issm_model" / "examples" / "ISMIP_Choi",
        run_script="run_da_issm.py",
        params="params.yaml",
        report_nb="read_results.m",
        assets=["_modelrun_datasets"],
        model_name="issm",
        figures_dir="figures",
        remote_rel="ICESEE/applications/issm_model/examples/ISMIP_Choi",
        remote_sbatch="run_job_spack.sbatch",  # you said this exists
    ),
    "Flowline (under development)": dict(
        enabled=True,
        base=EXT / "applications" / "flowline_model" / "examples" / "flowline_1d",
        run_script="run_da_flowline.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="flowline",
        figures_dir="figures",
        remote_rel="ICESEE/applications/flowline_model/examples/flowline_1d",
        remote_sbatch=None,
    ),
    "Icepack (under development)": dict(
        enabled=True,
        base=EXT / "applications" / "icepack_model" / "examples" / "synthetic_ice_stream",
        run_script="run_da_icepack.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="icepack",
        figures_dir="figures",
        remote_rel="ICESEE/applications/icepack_model/examples/synthetic_ice_stream",
        remote_sbatch=None,
    ),
}


def enabled_names():
    return [k for k, v in EXAMPLES.items() if v.get("enabled", False)]


# ============================================================
# 2) YAML helpers
# ============================================================
def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def dump_yaml(d: dict, p: Path) -> None:
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


# ============================================================
# 3) Find runner/template/report
# ============================================================
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


# ============================================================
# 4) Params widgets factory
# ============================================================
def widget_for(key: str, val):
    if isinstance(val, str):
        if key.lower() == "filter_type":
            opts = ["EnKF", "DEnKF", "EnTKF", "EnRSKF"]
            return W.Dropdown(options=opts, value=val if val in opts else opts[0], layout=W.Layout(width="100%"))
        if key.lower() in {"parallel_flag", "parallel"}:
            opts = ["serial", "MPI", "MPI_model"]
            return W.Dropdown(options=opts, value=val if val in opts else opts[0], layout=W.Layout(width="100%"))
        return W.Text(value=val, layout=W.Layout(width="100%"))

    if isinstance(val, bool):
        return W.Checkbox(value=val)

    if isinstance(val, int) and not isinstance(val, bool):
        return W.IntText(value=val, layout=W.Layout(width="100%"))
    if isinstance(val, float):
        return W.FloatText(value=val, layout=W.Layout(width="100%"))

    if isinstance(val, (list, dict)):
        return W.Textarea(
            value=yaml.safe_dump(val, sort_keys=False).strip(),
            layout=W.Layout(width="100%", height="110px"),
        )

    return W.Text(value=str(val), layout=W.Layout(width="100%"))


def read_widget(w):
    if isinstance(w, W.Textarea):
        try:
            return yaml.safe_load(w.value)
        except Exception:
            return w.value
    if hasattr(w, "value"):
        return w.value
    return None


# ============================================================
# 5) Local run helpers
# ============================================================
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

def ensure_report_h5(rd: Path, example_cfg: dict, expected_prefix: str, log_out: W.Output):
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
        with log_out:
            print(f"[wrapper] Report shim: copied {candidates[0].name} -> {exp.name}")
        return exp

    with log_out:
        print(f"[wrapper] WARNING: expected report H5 not found: {exp}")
    return exp

def run_report_notebook(report_nb: Path | None, example_cfg: dict, rd: Path, log_out: W.Output):
    if not report_nb or not report_nb.exists():
        with log_out:
            print("[wrapper] No report notebook configured/found; skipping read_results.")
        return None

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


def refresh_results_preview(rd: Path, results_out: W.Output):
    results_out.clear_output()
    with results_out:
        fig_dir = rd / "figures"
        pngs = sorted(fig_dir.glob("*.png"))
        if not pngs:
            pngs = sorted((rd / "results").glob("*.png"))
        h5s = sorted((rd / "results").glob("*.h5"))

        print("Run folder:", rd)
        print(f"Results: {len(h5s)} H5, {len(pngs)} PNG\n")
        for p in h5s[:10]:
            print(" -", p.name)

        if pngs:
            print("\nFigures:")
            for p in pngs[:6]:
                display(Image(filename=str(p)))
        else:
            print("\nNo figures found yet.")


# ============================================================
# 6) Remote backend (system ssh + Slurm)
# ============================================================
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

def ensure_local_ssh_key(log_out: W.Output, key_type: str = "ed25519") -> tuple[Path, Path]:
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

    with log_out:
        print(f"[auth] Generating SSH keypair: {priv.name}")

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
    pubkey_text: str,
    log_out: W.Output,
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

    with log_out:
        print("[auth] remote authorized_keys update rc:", rc)
        if out.strip():
            print("[auth] stdout:", out.strip())
        if err.strip():
            print("[auth] stderr:", err.strip())

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

# ============================================================
# 7) Cloud backend (AWS CLI + AWS Batch)
# ============================================================
@dataclass
class AWSBatchConfig:
    region: str = "us-east-1"
    profile: str | None = None
    s3_prefix: str = ""  # s3://bucket/prefix
    job_queue: str = ""
    job_definition: str = ""  # name[:revision]
    job_name: str = "icesee"


def _aws_cmd(cfg: AWSBatchConfig) -> list[str]:
    cmd = ["aws"]
    if cfg.profile:
        cmd += ["--profile", cfg.profile]
    if cfg.region:
        cmd += ["--region", cfg.region]
    return cmd


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _parse_s3(s3_uri: str) -> tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/(.*)$", s3_uri.rstrip("/"))
    if not m:
        raise ValueError("S3 path must look like: s3://bucket/prefix")
    return m.group(1), m.group(2)


def aws_test(cfg: AWSBatchConfig) -> None:
    code, out, err = _run(_aws_cmd(cfg) + ["sts", "get-caller-identity"])
    if code != 0:
        raise RuntimeError(err or out)


def aws_batch_submit(
    cfg: AWSBatchConfig,
    local_run_dir: Path,
    example_name: str,
    run_script_name: str,
) -> dict:
    if not cfg.s3_prefix or not cfg.job_queue or not cfg.job_definition:
        raise ValueError("Cloud config missing: s3_prefix/job_queue/job_definition")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    bucket, prefix = _parse_s3(cfg.s3_prefix)
    s3_run = f"s3://{bucket}/{prefix}/{run_id}"

    params_path = local_run_dir / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    # upload params
    code, out, err = _run(_aws_cmd(cfg) + ["s3", "cp", str(params_path), f"{s3_run}/params.yaml"])
    if code != 0:
        raise RuntimeError(err or out)

    manifest = {"run_id": run_id, "example": example_name, "run_script": run_script_name}
    (local_run_dir / "cloud_manifest.json").write_text(json.dumps(manifest, indent=2))
    _run(_aws_cmd(cfg) + ["s3", "cp", str(local_run_dir / "cloud_manifest.json"), f"{s3_run}/cloud_manifest.json"])

    env = [
        {"name": "ICESEE_S3_RUN", "value": s3_run},
        {"name": "ICESEE_EXAMPLE", "value": example_name},
        {"name": "ICESEE_RUN_SCRIPT", "value": run_script_name},
    ]

    submit_cmd = _aws_cmd(cfg) + [
        "batch",
        "submit-job",
        "--job-name",
        f"{cfg.job_name}-{run_id}",
        "--job-queue",
        cfg.job_queue,
        "--job-definition",
        cfg.job_definition,
        "--container-overrides",
        json.dumps({"environment": env}),
    ]
    code, out, err = _run(submit_cmd)
    if code != 0:
        raise RuntimeError(err or out)

    job_id = json.loads(out)["jobId"]
    return {"run_id": run_id, "batch_job_id": job_id, "s3_run": s3_run}


def aws_batch_status(cfg: AWSBatchConfig, job_id: str) -> dict:
    code, out, err = _run(_aws_cmd(cfg) + ["batch", "describe-jobs", "--jobs", job_id])
    if code != 0:
        raise RuntimeError(err or out)
    job = json.loads(out)["jobs"][0]
    return {"status": job.get("status", "?"), "reason": job.get("statusReason", "")}


# ============================================================
# 8) UI builder (single entry point)
# ============================================================
def build_icesee_ui():
    # -----------------------------
    # UI state containers
    # -----------------------------
    STATUS = {"mode": "idle", "remote_dir": None, "jobid": None, "batch_job_id": None, "s3_run": None}

    def set_status(state: str):
        cls = {"idle": "icesee-idle", "running": "icesee-running", "done": "icesee-done", "fail": "icesee-fail"}[state]
        label = {"idle": "Idle", "running": "Running…", "done": "Done", "fail": "Failed"}[state]
        status_chip.value = f"<span class='icesee-status {cls}'>{label}</span>"

    # -----------------------------
    # Top controls
    # -----------------------------
    example_dd = W.Dropdown(options=enabled_names(), value=enabled_names()[0], layout=W.Layout(width="320px"))
    preset_dd = W.Dropdown(options=["Default"], value="Default", layout=W.Layout(width="320px"))

    filter_alg_dd = W.Dropdown(
        options=[("EnKF", "EnKF"), ("DEnKF", "DEnKF"), ("EnTKF", "EnTKF"), ("EnRSKF", "EnRSKF")],
        value="EnKF",
        layout=W.Layout(width="320px"),
    )

    output_label_dd = W.Dropdown(
        options=[("true-wrong (demo output)", "true-wrong"), ("EnKF (output name)", "enkf")],
        value="true-wrong",
        layout=W.Layout(width="320px"),
    )

    ens_sl = W.IntSlider(min=1, max=200, value=30, layout=W.Layout(width="320px"), continuous_update=False)
    seed_in = W.IntText(value=1, layout=W.Layout(width="320px"))

    gen_report = W.Checkbox(value=True, description="Generate report (read_results.ipynb)")
    open_latest = W.Checkbox(value=False, description="After run: open latest run folder")

    # run_btn = W.Button(description="Run", button_style="success", icon="play")
    action_btn = W.Button(description="Run", button_style="success", icon="play")
    clear_btn = W.Button(description="Clear", button_style="", icon="trash")
    

    status_chip = W.HTML("<span class='icesee-status icesee-idle'>Idle</span>")
    log_out = W.Output(layout=W.Layout(border="1px solid rgba(0,0,0,.12)", padding="10px", height="220px", overflow="auto"))
    results_out = W.Output(layout=W.Layout(border="1px solid rgba(0,0,0,.12)", padding="10px", height="260px", overflow="auto"))

    # -----------------------------
    # Mode Tabs
    # -----------------------------
    MODE_LOCAL, MODE_REMOTE, MODE_CLOUD = "local", "cluster", "cloud"
    mode_tabs = W.Tab()
    mode_tabs.layout = W.Layout(width="100%")
    mode_tabs.layout.flex = "1 1 auto"
    mode_tabs.layout.min_width = "0"

    def get_mode():
        return {0: MODE_LOCAL, 1: MODE_REMOTE, 2: MODE_CLOUD}.get(mode_tabs.selected_index, MODE_LOCAL)
    
    def update_action_button():
        mode = get_mode()
        if mode == MODE_LOCAL:
            action_btn.description = "Run"
            action_btn.icon = "play"
            action_btn.button_style = "success"
        elif mode == MODE_REMOTE:
            action_btn.description = "Submit (Remote)"
            action_btn.icon = "server"
            action_btn.button_style = "warning"
        else:
            action_btn.description = "Submit (Cloud)"
            action_btn.icon = "cloud-upload"
            action_btn.button_style = "warning"

    def on_action_click(_=None):
        # simple anti-double-submit (optional but recommended)
        if STATUS.get("_busy"):
            with log_out:
                print("[ui] Busy — ignoring extra click.")
            return

        STATUS["_busy"] = True
        action_btn.disabled = True
        try:
            mode = get_mode()
            if mode == MODE_LOCAL:
                return run_example_local()
            elif mode == MODE_REMOTE:
                return run_example_remote_submit()
            else:
                return run_example_cloud_submit()
        finally:
            action_btn.disabled = False
            STATUS["_busy"] = False

    # =========================================================
    # Params UI (auto from template)
    # =========================================================
    params_holder = W.VBox([])
    params_accordion = None
    PARAMS0 = {}
    WIDGETS = {}
    EXTRA_YAML = {}
    RUN_SCRIPT = None
    TEMPLATE = None
    REPORT_NB = None

    def build_params_ui(template_path: Path):
        nonlocal params_accordion, PARAMS0, WIDGETS, EXTRA_YAML
        PARAMS0 = load_yaml(template_path)
        WIDGETS = {}
        EXTRA_YAML = {}

        children, titles = [], []

        for sec, sec_dict in (PARAMS0 or {}).items():
            titles.append(sec)
            sec_widgets = {}
            rows = []

            if isinstance(sec_dict, dict):
                for k, v in sec_dict.items():
                    w = widget_for(k, v)
                    sec_widgets[k] = w
                    rows.append(
                        W.HBox(
                            [W.HTML(f"<div class='icesee-k'>{k}</div>"), w],
                            layout=W.Layout(gap="12px"),
                        )
                    )

                extra = W.Textarea(
                    value="# Add future keys here (YAML)\n",
                    layout=W.Layout(width="100%", height="90px"),
                )
                EXTRA_YAML[sec] = extra
                rows.append(W.HTML("<div class='icesee-subtle' style='margin-top:6px'>Extra keys (optional)</div>"))
                rows.append(extra)
            else:
                w = W.Textarea(
                    value=yaml.safe_dump(sec_dict, sort_keys=False).strip(),
                    layout=W.Layout(width="100%", height="140px"),
                )
                sec_widgets["__raw__"] = w
                rows.append(w)

            WIDGETS[sec] = sec_widgets
            children.append(W.VBox(rows, layout=W.Layout(gap="8px")))

        params_accordion = W.Accordion(children=children)
        for i, t in enumerate(titles):
            params_accordion.set_title(i, t)

    def sync_quick_into_widgets():
        sec = None
        for candidate in ["enkf-parameters", "enkf_parameters", "enkf"]:
            if candidate in WIDGETS:
                sec = candidate
                break
        if not sec:
            return

        if "Nens" in WIDGETS[sec]:
            WIDGETS[sec]["Nens"].value = int(ens_sl.value)
        if "seed" in WIDGETS[sec]:
            WIDGETS[sec]["seed"].value = int(seed_in.value)
        if "filter_type" in WIDGETS[sec]:
            WIDGETS[sec]["filter_type"].value = str(filter_alg_dd.value)

    def build_config_from_widgets() -> dict:
        cfg = {}
        for sec, sw in WIDGETS.items():
            if "__raw__" in sw:
                cfg[sec] = yaml.safe_load(sw["__raw__"].value)
                continue

            cfg[sec] = {}
            for k, w in sw.items():
                if k == "__raw__":
                    continue
                cfg[sec][k] = read_widget(w)

            extra = EXTRA_YAML.get(sec)
            if extra:
                txt = extra.value.strip()
                if txt and not txt.startswith("#"):
                    extra_obj = yaml.safe_load(txt) or {}
                    if isinstance(extra_obj, dict):
                        cfg[sec].update(extra_obj)
                    else:
                        cfg[sec]["__extra__"] = extra_obj
        return cfg

    # -----------------------------
    # Rebuild on example change
    # -----------------------------
    def rebuild_for_example(_=None):
        nonlocal RUN_SCRIPT, TEMPLATE, REPORT_NB
        cfg = EXAMPLES[example_dd.value]
        RUN_SCRIPT = find_run_script(cfg)
        TEMPLATE = find_params_template(cfg)
        REPORT_NB = find_report_notebook(cfg)

        build_params_ui(TEMPLATE)
        params_holder.children = (params_accordion,)

        with log_out:
            print("[Loaded]")
            print("Template:", TEMPLATE)
            print("Runner  :", RUN_SCRIPT)
            print("Report  :", REPORT_NB if REPORT_NB else "(none)")

    example_dd.observe(rebuild_for_example, names="value")

    # =========================================================
    # Remote panel widgets
    # =========================================================
    cluster_host = W.Text(value="login-phoenix-rh9.pace.gatech.edu", layout=W.Layout(width="320px"))
    cluster_user = W.Text(value=os.environ.get("USER", ""), placeholder="username", layout=W.Layout(width="320px"))
    cluster_port = W.IntText(value=22, layout=W.Layout(width="120px"))
    
    auth_mode = W.ToggleButtons(
    options=[("Key-only", "key"), ("Bootstrap with password (one-time)", "bootstrap")],
    value="key",
    layout=W.Layout(width="420px")
    )

    cluster_password = W.Password(
        value="",
        placeholder="One-time password (not stored)",
        layout=W.Layout(width="320px")
    )

    bootstrap_btn = W.Button(
        description="Enable passwordless SSH",
        icon="key",
        button_style="warning"
    )

    remote_base_dir = W.Text(value="~/r-arobel3-0", layout=W.Layout(width="320px"))
    remote_tag = W.Text(value="icesee", layout=W.Layout(width="220px"))

    connect_btn = W.Button(description="Test SSH", icon="terminal", button_style="info")
    submit_btn = W.Button(description="Submit job", icon="server", button_style="warning")
    status_btn = W.Button(description="Check status", icon="tasks", button_style="")
    tail_btn = W.Button(description="Tail log", icon="file-text", button_style="")
    terminate_btn = W.Button(description="Terminate job",icon="stop",button_style="danger")

    slurm_job_name = W.Text(value="ICESEE", layout=W.Layout(width="220px"))
    slurm_time = W.Text(value="50:00:00", layout=W.Layout(width="220px"))
    slurm_nodes = W.IntText(value=2, layout=W.Layout(width="120px"))
    slurm_ntasks = W.IntText(value=24, layout=W.Layout(width="120px"))
    slurm_tpn = W.IntText(value=24, layout=W.Layout(width="120px"))
    slurm_part = W.Text(value="cpu-large", layout=W.Layout(width="220px"))
    slurm_mem = W.Text(value="256G", layout=W.Layout(width="220px"))
    slurm_account = W.Text(value="gts-arobel3-atlas", layout=W.Layout(width="220px"))
    slurm_mail = W.Text(value="bankyanjo@gmail.com", layout=W.Layout(width="220px"))

    cluster_mpi_np = W.IntText(value=40, layout=W.Layout(width="120px"))
    cluster_model_nprocs = W.IntText(value=4, layout=W.Layout(width="120px"))

    # minimal module/export lines (you can expand later)
    remote_module_lines = W.Textarea(
        value="# module load ...\n",
        layout=W.Layout(width="100%", height="80px"),
    )
    remote_export_lines = W.Textarea(
        value="# export ISSM_DIR=...\n",
        layout=W.Layout(width="100%", height="80px"),
    )

    remote_backend = W.ToggleButtons(
        options=[("SSH (Slurm)", "ssh"), ("HTTPS (Webhook)", "https")],
        value="ssh",
        layout=W.Layout(width="320px")
    )

    https_base = W.Text(value="", placeholder="https://your-service.example.com", layout=W.Layout(width="520px"))
    https_submit_path = W.Text(value="/submit", layout=W.Layout(width="260px"))
    https_status_path = W.Text(value="/status", layout=W.Layout(width="260px"))  # will call /status/<run_id>
    https_tail_path   = W.Text(value="/tail", layout=W.Layout(width="260px"))    # will call /tail/<run_id>?n=120
    https_health_path = W.Text(value="/health", layout=W.Layout(width="260px"))

    https_token = W.Password(value="", placeholder="optional bearer token", layout=W.Layout(width="320px"))
    https_headers = W.Textarea(
        value="# optional extra headers (YAML dict)\n# X-API-Key: abc\n",
        layout=W.Layout(width="100%", height="80px")
    )

    https_webhook_box = W.VBox([
    W.HTML("<div class='icesee-subtle'>HTTPS backend (user-provided webhook/service)</div>"),
    W.HBox([W.HTML("<div class='icesee-lbl'>Base URL:</div>"), https_base], layout=W.Layout(gap="12px")),
    W.HBox([W.HTML("<div class='icesee-lbl'>Paths:</div>"),
            https_submit_path, https_status_path, https_tail_path, https_health_path],
           layout=W.Layout(gap="8px")),
    W.HBox([W.HTML("<div class='icesee-lbl'>Token:</div>"), https_token], layout=W.Layout(gap="12px")),
    W.HTML("<div class='icesee-subtle'>Extra headers (YAML)</div>"),
    https_headers,
    ], layout=W.Layout(gap="8px"))

    ood_cluster = W.Dropdown(
        options=[
            ("Phoenix OnDemand", "https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard/"),
            ("Hive OnDemand",    "https://ondemand-hive.pace.gatech.edu/pun/sys/dashboard/"),
            ("ICE OnDemand",     "https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard/"),
        ],
        value="https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard/",
        layout=W.Layout(width="520px")
    )

    open_ood_btn = W.Button(description="Open OnDemand", icon="external-link", button_style="info")

    # --- ICESEE-Spack bootstrap ---
    spack_enable = W.Checkbox(value=True, description="Use ICESEE-Spack on Remote")
    spack_repo_url = W.Text(
        value="https://github.com/ICESEE-project/ICESEE-Spack.git",
        layout=W.Layout(width="520px"),
    )
    spack_dirname = W.Text(value="ICESEE-Spack", layout=W.Layout(width="220px"))

    spack_install_if_needed = W.Checkbox(value=False, description="Run install.sh if not installed")
    spack_install_mode = W.Dropdown(
        options=[
            ("Default", ""),
            ("With ISSM", "--with-issm"),
            ("With Firedrake", "--with-firedrake"),
            ("With Icepack", "--with-icepack"),
        ],
        value="--with-issm",
        layout=W.Layout(width="220px"),
    )

    # README mentions SLURM_DIR + PMIX_DIR for install.sh
    spack_slurm_dir = W.Text(value="", placeholder="e.g. /opt/slurm/current", layout=W.Layout(width="320px"))
    spack_pmix_dir  = W.Text(value="", placeholder="e.g. /opt/pmix/5.0.1", layout=W.Layout(width="320px"))

    # Optional: use an existing sbatch from the repo if present
    spack_use_existing_sbatch = W.Checkbox(
        value=True,
        description="If run_job_spack.sbatch exists for this example, submit it",
    )

    ssh_box = W.VBox([
    # existing SSH fields: host/user/port/auth/... and buttons
    ])

    ondemand_box = W.VBox([
        W.HTML("<div class='icesee-subtle'>OnDemand (web portal)</div>"),
        W.HBox([W.HTML("<div class='icesee-lbl'>Portal:</div>"), ood_cluster], layout=W.Layout(gap="12px")),
        W.HBox([open_ood_btn], layout=W.Layout(gap="10px")),
        W.HTML("<div class='icesee-subtle'>Tip: You may need GT VPN to access OnDemand.</div>"),
    ])

    def _toggle_remote_backend(_=None):
        is_ssh = (remote_backend.value == "ssh")
        ssh_box.layout.display = "block" if is_ssh else "none"
        ondemand_box.layout.display = "none" if is_ssh else "block"

    remote_backend.observe(_toggle_remote_backend, names="value")
    _toggle_remote_backend()
    W.HBox([W.HTML("<div class='icesee-lbl'>Backend:</div>"), remote_backend], layout=W.Layout(gap="12px")),
    ssh_box,
    ondemand_box,

    def on_test_remote(_=None):
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "https":
            with log_out:
                print("[remote:https] OnDemand portal:", ood_cluster.value)
                print("Open it in a browser tab (VPN may be required).")
            set_status("done")
            return

        # else: your SSH test (with timeout) as you already fixed
        return run_example_remote_test()
    connect_btn.on_click(on_test_remote)

    def submit_remote(_=None):
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "ssh":
            run_example_remote_submit()
            return

        # HTTPS assisted mode
        example_cfg = EXAMPLES[example_dd.value]
        sync_quick_into_widgets()
        cfg_yaml = build_config_from_widgets()

        rd = run_dir()
        dump_yaml(cfg_yaml, rd / "params.yaml")

        # write slurm script locally so user can upload via OnDemand Files
        slurm_text = render_slurm_script({...})  # same as SSH branch
        (rd / "slurm_run.sh").write_text(slurm_text)
        if "{{" in slurm_text or "}}" in slurm_text:
            raise RuntimeError("SLURM_TEMPLATE render left unresolved placeholders. Check keys passed to render_slurm_script().")

        with log_out:
            print("[remote:https] Prepared files in:", rd)
            print(" - params.yaml")
            print(" - slurm_run.sh")
            print("\nNext (OnDemand):")
            print(" 1) Open OnDemand portal:", ood_cluster.value)
            print(" 2) Go to Files -> Home (or project dir) and upload these files")
            print(" 3) Open a Shell and run:")
            print("     sbatch slurm_run.sh")
            print("\nTip: OnDemand access may require GT VPN.")

        set_status("done")
    # submit_btn.on_click(submit_remote)

    def _toggle_auth_widgets(_=None):
        show = (auth_mode.value == "bootstrap")
        cluster_password.layout.display = "block" if show else "none"
        bootstrap_btn.layout.display = "block" if show else "none"

    auth_mode.observe(_toggle_auth_widgets, names="value")
    _toggle_auth_widgets()

    connect_btn.icon = "terminal"
    submit_btn.icon  = "server"
    status_btn.icon  = "tasks"
    tail_btn.icon    = "file-text"

    def _https_url(path_widget: W.Text, run_id: str | None = None, query: dict | None = None) -> str:
        base = https_base.value.strip().rstrip("/")
        path = path_widget.value.strip()
        if not path.startswith("/"):
            path = "/" + path
        url = base + path
        if run_id is not None:
            url = url.rstrip("/") + "/" + run_id
        if query:
            url = url + "?" + urlencode(query)
        return url

    def _extra_headers() -> dict:
        h = {}
        # bearer token
        if https_token.value.strip():
            h["Authorization"] = "Bearer " + https_token.value.strip()
        # yaml headers
        txt = https_headers.value.strip()
        if txt and not txt.startswith("#"):
            try:
                y = yaml.safe_load(txt) or {}
                if isinstance(y, dict):
                    h.update({str(k): str(v) for k, v in y.items()})
            except Exception:
                pass
        return h
    
    def remote_test():
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "ssh":
            return run_example_remote_test()

        with log_out:
            print("[remote:https] Testing health endpoint…")
            print("base:", https_base.value.strip())
        try:
            url = _https_url(https_health_path)
            code, j, txt = http_json("GET", url, headers=_extra_headers(), timeout=15)
            with log_out:
                print("GET", url)
                print("status:", code)
                print("json:", j)
                if txt and not j:
                    print("text:", txt[:4000])
            set_status("done" if 200 <= code < 300 else "fail")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote:https][ERROR]", type(e).__name__, e)

    def remote_submit():
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "ssh":
            return run_example_remote_submit()

        # Build job request
        example_cfg = EXAMPLES[example_dd.value]
        sync_quick_into_widgets()
        cfg_yaml = build_config_from_widgets()

        rd = run_dir()
        params_path = rd / "params.yaml"
        dump_yaml(cfg_yaml, params_path)

        # Optional: generate slurm script (still useful for many services)
        slurm_text = render_slurm_script(
            dict(
                TIME=slurm_time.value.strip(),
                JOB_NAME=slurm_job_name.value.strip() or "ICESEE",
                NODES=int(slurm_nodes.value),
                NTASKS=int(slurm_ntasks.value),
                TPN=int(slurm_tpn.value),
                PARTITION=slurm_part.value.strip(),
                MEM=slurm_mem.value.strip(),
                ACCOUNT=(slurm_account.value.strip() or ""),
                OUTFILE="icesee-enkf-%j.out",
                MAIL_USER=(slurm_mail.value.strip() or ""),
                MODULE_LINES=remote_module_lines.value.rstrip(),
                EXPORT_LINES=remote_export_lines.value.rstrip(),
                RUN_LAUNCH_LINE=(
                    f"mpirun -np {int(cluster_mpi_np.value)} "
                    f"python3 {find_run_script(example_cfg).name} "
                    f"-F params.yaml --Nens={int(ens_sl.value)} --model_nprocs={int(cluster_model_nprocs.value)}"
                ),
            )
        )
        if "{{" in slurm_text or "}}" in slurm_text:
            raise RuntimeError("SLURM_TEMPLATE render left unresolved placeholders. Check keys passed to render_slurm_script().")
        

        payload = {
            "kind": "icesee-run",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "example": example_dd.value,
            "run_script": find_run_script(example_cfg).name,
            "params_yaml": params_path.read_text(encoding="utf-8"),
            "slurm_script": slurm_text,  # optional; service may ignore
            "metadata": {
                "repo": str(REPO),
                "tag": remote_tag.value.strip(),
            },
        }

        with log_out:
            print("[remote:https] Submitting to webhook…")
            print("example:", payload["example"])
            print("endpoint:", _https_url(https_submit_path))
            print("-" * 70)

        try:
            url = _https_url(https_submit_path)
            code, j, txt = http_json("POST", url, payload=payload, headers=_extra_headers(), timeout=30)
            if not (200 <= code < 300):
                raise RuntimeError(f"HTTP {code}: {txt[:2000]}")
            run_id = (j or {}).get("run_id") or (j or {}).get("id")
            if not run_id:
                raise RuntimeError(f"No run_id in response. Response json={j}, text={txt[:2000]}")

            STATUS["run_id"] = run_id
            STATUS["remote_mode"] = "https"

            set_status("done")
            with log_out:
                print("[remote:https] ✅ Submitted")
                print("run_id:", run_id)
                if (j or {}).get("url"):
                    print("url  :", (j or {}).get("url"))
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote:https][ERROR]", type(e).__name__, e)

    def remote_status():
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "ssh":
            return run_example_remote_status()

        run_id = STATUS.get("run_id")
        if not run_id:
            set_status("fail")
            with log_out:
                print("[remote:https] No run_id yet. Submit first.")
            return

        try:
            url = _https_url(https_status_path, run_id=run_id)
            code, j, txt = http_json("GET", url, headers=_extra_headers(), timeout=15)
            with log_out:
                print("[remote:https] GET", url)
                print("status:", code)
                if j:
                    print(json.dumps(j, indent=2)[:4000])
                else:
                    print(txt[:4000])
            set_status("done" if 200 <= code < 300 else "fail")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote:https][ERROR]", type(e).__name__, e)

    
    def remote_tail():
        log_out.clear_output()
        set_status("running")

        if remote_backend.value == "ssh":
            return run_example_remote_tail()

        run_id = STATUS.get("run_id")
        if not run_id:
            set_status("fail")
            with log_out:
                print("[remote:https] No run_id yet. Submit first.")
            return

        try:
            url = _https_url(https_tail_path, run_id=run_id, query={"n": 120})
            code, j, txt = http_json("GET", url, headers=_extra_headers(), timeout=15)
            with log_out:
                print("[remote:https] GET", url)
                print("status:", code)
                # tail is usually text; show text first
                if txt:
                    print(txt.rstrip()[:8000])
                elif j:
                    print(json.dumps(j, indent=2)[:8000])
            set_status("done" if 200 <= code < 300 else "fail")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote:https][ERROR]", type(e).__name__, e)

    # connect_btn.on_click(lambda b: remote_test())
    # submit_btn.on_click(lambda b: remote_submit())
    # status_btn.on_click(lambda b: remote_status())
    # tail_btn.on_click(lambda b: remote_tail())

    def on_bootstrap_keys(_=None):
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)
        pw   = cluster_password.value

        if not host or not user:
            set_status("fail")
            with log_out:
                print("[auth][ERROR] Provide Host + User first.")
            return
        if not pw:
            set_status("fail")
            with log_out:
                print("[auth][ERROR] Enter your password (used once; not stored).")
            return

        try:
            # 1) ensure local keypair exists
            priv, pub = ensure_local_ssh_key(log_out, key_type="ed25519")
            pubkey_text = pub.read_text(encoding="utf-8").strip()

            with log_out:
                print("[auth] Installing public key to remote authorized_keys…")

            # 2) install pubkey using password auth
            remote_install_pubkey_with_password(
                host=host, user=user, port=port,
                password=pw, pubkey_text=pubkey_text,
                log_out=log_out
            )

            # 3) verify system ssh works (BatchMode)
            with log_out:
                print("[auth] Verifying non-interactive SSH…")
            r = ssh_run(host, user, port, "hostname && whoami && date", timeout=15)

            if r.returncode == 0:
                set_status("done")
                with log_out:
                    print("[auth] ✅ Passwordless SSH is working. You can switch back to Key-only.")
                # Optional: flip auth mode back
                auth_mode.value = "key"
                cluster_password.value = ""
            else:
                set_status("fail")
                with log_out:
                    print("[auth][ERROR] Key install ran, but BatchMode SSH still failed.")
                    print("stdout:", (r.stdout or "").strip())
                    print("stderr:", (r.stderr or "").strip())
                    print("hint :", explain_ssh_failure_hint(r.stderr or ""))

        except Exception as e:
            set_status("fail")
            with log_out:
                print("[auth][ERROR]", type(e).__name__, e)

    def rsh(host, user, port, cmd, timeout=60):
        """SSH run with a timeout and returned stdout/stderr."""
        r = ssh_run(host, user, port, cmd, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def remote_ensure_spack(host, user, port, remote_parent_dir, spack_name, repo_url):
        """
        Ensure ICESEE-Spack is cloned inside remote_parent_dir/spack_name.
        """
        spack_path = f"{remote_parent_dir.rstrip('/')}/{spack_name}"
        cmd = f"""
    set -e
    mkdir -p {sh_quote(remote_parent_dir)}
    if [ ! -d {sh_quote(spack_path)} ]; then
    echo "[spack] cloning {repo_url} -> {spack_path}"
    cd {sh_quote(remote_parent_dir)}
    git clone --recurse-submodules {sh_quote(repo_url)} {sh_quote(spack_name)}
    else
    echo "[spack] exists: {spack_path}"
    fi
    echo "[spack] done"
    """
        return spack_path, rsh(host, user, port, cmd, timeout=180)

    def remote_maybe_install_spack(host, user, port, spack_path, install_flag, slurm_dir, pmix_dir):
        """
        Run scripts/install.sh (idempotent-ish) if user requested.
        Uses a marker file to avoid re-running.
        """
        marker = f"{spack_path}/.icesee_spack_installed"
        slurm = slurm_dir.strip()
        pmix  = pmix_dir.strip()

        # Build env prefix only if provided
        env_prefix = ""
        if slurm:
            env_prefix += f"SLURM_DIR={sh_quote(slurm)} "
        if pmix:
            env_prefix += f"PMIX_DIR={sh_quote(pmix)} "

        cmd = f"""
    set -e
    cd {sh_quote(spack_path)}
    if [ -f {sh_quote(marker)} ]; then
    echo "[spack] install marker present: {marker}"
    exit 0
    fi
    echo "[spack] running install.sh {install_flag}"
    {env_prefix} ./scripts/install.sh {install_flag}
    touch {sh_quote(marker)}
    echo "[spack] install completed"
    """
        return rsh(host, user, port, cmd, timeout=60*60)  # installs can be long

    # =========================================================
    # Cloud panel widgets (AWS Batch)
    # =========================================================
    aws_region = W.Text(value="us-east-1", layout=W.Layout(width="220px"))
    aws_profile = W.Text(value="", placeholder="(optional) AWS profile", layout=W.Layout(width="220px"))
    cloud_bucket = W.Text(value="", placeholder="s3://bucket/prefix", layout=W.Layout(width="320px"))

    batch_job_queue = W.Text(value="", placeholder="AWS Batch job queue", layout=W.Layout(width="320px"))
    batch_job_def = W.Text(value="", placeholder="job definition (name[:rev])", layout=W.Layout(width="320px"))
    batch_job_name = W.Text(value="icesee", layout=W.Layout(width="220px"))

    cloud_submit_btn = W.Button(description="Submit", icon="cloud-upload", button_style="warning")
    cloud_status_btn = W.Button(description="Check status", icon="search", button_style="")
    cloud_logs_btn = W.Button(description="Logs hint", icon="file-text", button_style="")

    # =========================================================
    # Actions: Local / Remote / Cloud
    # =========================================================
    def run_example_local():
        example_cfg = EXAMPLES[example_dd.value]

        sync_quick_into_widgets()
        cfg = build_config_from_widgets()

        rd = run_dir()
        dump_yaml(cfg, rd / "params.yaml")

        env, external_dir = force_external_icesee_env()
        cmd = [sys.executable, str(RUN_SCRIPT), "-F", str(rd / "params.yaml")]

        set_status("running")
        log_out.clear_output()
        with log_out:
            print("[local] Example :", example_dd.value)
            print("[local] Runner  :", RUN_SCRIPT)
            print("[local] Report  :", REPORT_NB if REPORT_NB else "(none)")
            print("[local] CWD     :", rd)
            print("[local] Command :", " ".join(cmd))
            print("[local] PYTHONPATH(prepended):", external_dir)
            print("-" * 70)

        proc = subprocess.Popen(
            cmd,
            cwd=str(rd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        full_log = []
        for line in proc.stdout:
            full_log.append(line)
            with log_out:
                print(line.rstrip())
        rc = proc.wait()

        log_text = "".join(full_log)
        looks_like_failure = ("Traceback (most recent call last)" in log_text) or ("Error in serial run mode" in log_text)

        with log_out:
            print("-" * 70)
            print("Return code:", rc)

        if rc != 0 or looks_like_failure:
            set_status("fail")
            refresh_results_preview(rd, results_out)
            return

        if gen_report.value:
            try:
                with log_out:
                    print("[local] Running report notebook…")
                # Make sure report sees the expected file name
                ensure_report_h5(rd, example_cfg, output_label_dd.value, log_out)
                run_report_notebook(REPORT_NB, example_cfg, rd, log_out)
                with log_out:
                    print("[local] Report done.")
            except Exception as e:
                with log_out:
                    print("[local] Report failed:", type(e).__name__, e)
                set_status("fail")
                refresh_results_preview(rd, results_out)
                return

        set_status("done")
        refresh_results_preview(rd, results_out)

        if open_latest.value:
            with log_out:
                print("\nRun folder:", rd)

    def run_example_remote_submit():
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)

        with log_out:
            print("[remote] Submit job")
            print("  host:", host)
            print("  user:", user)
            print("  port:", port)
            print("  example:", example_dd.value)
            print("-" * 70)

        if not host or not user:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] Provide Host + User first.")
            return

        example_cfg = EXAMPLES[example_dd.value]

        # Build params locally (text only; we write it remotely via heredoc)
        sync_quick_into_widgets()
        cfg_yaml = build_config_from_widgets()
        params_text = yaml.safe_dump(cfg_yaml, sort_keys=False)

        # Remote run dir
        rdir = make_remote_run_dir(
            remote_base_dir.value.strip() or "~/r-arobel3-0",
            remote_tag.value.strip() or "icesee",
        )

        # --- ensure ICESEE-Spack ---
        spack_path = None
        if spack_enable.value:
            spack_parent = remote_base_dir.value.strip() or "~/r-arobel3-0"
            spack_name = spack_dirname.value.strip() or "ICESEE-Spack"
            repo = spack_repo_url.value.strip()

            with log_out:
                print("[remote] Spack enabled")
                print("  parent:", spack_parent)
                print("  repo  :", repo)
                print("  name  :", spack_name)

            spack_path, (rc, out, err) = remote_ensure_spack(host, user, port, spack_parent, spack_name, repo)
            # after spack_path is set (like "~/r-arobel3-0/ICESEE-Spack")
            rcp, outp, errp = rsh(host, user, port, f"python3 - <<'PY'\nimport os\nprint(os.path.abspath(os.path.expanduser({spack_path!r})))\nPY", timeout=20)
            spack_path_abs = (outp or "").strip()
            if not spack_path_abs:
                raise RuntimeError("Could not resolve remote absolute spack path.")
            spack_path = spack_path_abs
            with log_out:
                if out.strip(): print(out.strip())
                if err.strip(): print(err.strip())
            if rc != 0:
                set_status("fail")
                return

            if spack_install_if_needed.value:
                install_flag = spack_install_mode.value or ""
                with log_out:
                    print("[remote] Spack install requested:", install_flag or "(default)")
                rc, out, err = remote_maybe_install_spack(host, user, port, spack_path, install_flag, spack_slurm_dir.value, spack_pmix_dir.value)
                with log_out:
                    if out.strip(): print(out.strip())
                    if err.strip(): print(err.strip())
                if rc != 0:
                    set_status("fail")
                    return
        else:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] Remote currently requires ICESEE-Spack enabled.")
            return

        # Remote example dir inside ICESEE-Spack
        remote_rel = example_cfg.get("remote_rel")
        if not remote_rel:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] This example has no remote_rel configured in EXAMPLES.")
            return

        remote_example_dir = f"{spack_path.rstrip('/')}/{remote_rel.lstrip('/')}"
        with log_out:
            print("[remote] Remote ICESEE-Spack:", spack_path)
            print("[remote] Remote example dir :", remote_example_dir)

        # Verify example dir exists remotely
        rc, out, err = rsh(host, user, port, f"test -d {sh_quote(remote_example_dir)} && echo OK || echo MISSING", timeout=20)
        if "OK" not in (out or ""):
            set_status("fail")
            with log_out:
                print("[remote][ERROR] Remote example directory not found.")
                print("stdout:", (out or "").strip())
                print("stderr:", (err or "").strip())
            return

        # If repo provides a job script for this example (ISSM), use it
        remote_sbatch = example_cfg.get("remote_sbatch")
        if spack_use_existing_sbatch.value and remote_sbatch:
            # confirm it exists
            chk = f"test -f {sh_quote(remote_example_dir + '/' + remote_sbatch)} && echo OK || echo MISSING"
            rc2, out2, err2 = rsh(host, user, port, chk, timeout=20)
            if "OK" in (out2 or ""):
                try:
                    # stage params.yaml in rdir
                    remote_write_text(host, user, port, f"{rdir}/params.yaml", params_text, timeout=30)
                    # copy into example dir so the sbatch finds it (simple + robust)
                    rsh(host, user, port, f"cp {sh_quote(rdir+'/params.yaml')} {sh_quote(remote_example_dir+'/params.yaml')}", timeout=20)

                    submit_cmd = f"""
    set -e
    cd {sh_quote(remote_example_dir)}
    source {sh_quote(spack_path)}/scripts/activate.sh
    sbatch {sh_quote(remote_sbatch)}
    """
                    r = ssh_run(host, user, port, submit_cmd, timeout=30)
                    if r.returncode != 0:
                        raise RuntimeError(r.stderr or r.stdout)

                    m = re.search(r"Submitted batch job\s+(\d+)", r.stdout)
                    if not m:
                        raise RuntimeError(f"Could not parse JobID from:\n{r.stdout}\n{r.stderr}")
                    jobid = m.group(1)

                    STATUS["remote_dir"] = remote_example_dir
                    STATUS["jobid"] = jobid

                    set_status("done")
                    with log_out:
                        print("[remote] ✅ Submitted existing sbatch from example dir")
                        print("  sbatch:", remote_sbatch)
                        print("  jobid :", jobid)
                    return

                except Exception as e:
                    set_status("fail")
                    with log_out:
                        print("[remote][ERROR]", type(e).__name__, e)
                    return
            else:
                with log_out:
                    print("[remote] NOTE: remote_sbatch configured but not found; falling back to generated slurm_run.sh.")

        # Otherwise generate sbatch wrapper that runs python from remote example dir
        outfile = "icesee-enkf-%j.out"

        run_script_name = example_cfg.get("run_script") or find_run_script(example_cfg).name

        # run_launch = (
        #     f"cd {sh_quote(remote_example_dir)}\n"
        #     f"python3 {sh_quote(run_script_name)} "
        #     f"-F {sh_quote(rdir + '/params.yaml')} "
        #     f"--Nens={int(ens_sl.value)} --model_nprocs={int(cluster_model_nprocs.value)}"
        # )

        # slurm_text = render_slurm_script(
        #     dict(
        #         TIME=slurm_time.value.strip(),
        #         JOB_NAME=slurm_job_name.value.strip() or "ICESEE",
        #         NODES=int(slurm_nodes.value),
        #         NTASKS=int(slurm_ntasks.value),
        #         TPN=int(slurm_tpn.value),
        #         PARTITION=slurm_part.value.strip(),
        #         MEM=slurm_mem.value.strip(),
        #         ACCOUNT=(slurm_account.value.strip() or "REPLACE_ME"),
        #         OUTFILE=outfile,
        #         MODULE_LINES=remote_module_lines.value.rstrip(),
        #         EXPORT_LINES=remote_export_lines.value.rstrip(),
        #         SPACK_PATH=spack_path,
        #         RUN_DIR=rdir,
        #         RUN_LAUNCH_LINE=run_launch,
        #     )
        # )

        # build sbatch optional lines correctly
        account_line, mail_lines = slurm_optional_lines(
            slurm_account.value.strip(),
            slurm_mail.value.strip(),
        )

        slurm_text = render_slurm_script(
            dict(
                TIME=slurm_time.value.strip(),
                JOB_NAME=slurm_job_name.value.strip() or "ICESEE",
                NODES=int(slurm_nodes.value),
                NTASKS=int(slurm_ntasks.value),
                TPN=int(slurm_tpn.value),
                PARTITION=slurm_part.value.strip(),
                MEM=slurm_mem.value.strip(),

                # ✅ these match the template
                ACCOUNT_LINE=account_line,
                MAIL_LINES=mail_lines,

                OUTFILE=outfile,
                MODULE_LINES=sanitize_multiline(remote_module_lines.value),
                EXPORT_LINES=sanitize_multiline(remote_export_lines.value),

                SPACK_PATH=spack_path,
                NP=int(cluster_mpi_np.value),
                NENS=int(ens_sl.value),
                MODEL_NPROCS=int(cluster_model_nprocs.value),
                RUN_SCRIPT=run_script_name,
                PARAMS_PATH=f"{rdir}/params.yaml",
                EXAMPLE_DIR=remote_example_dir,

                # add this so the template doesn’t leave {{SRUN_MPI_FLAG}} behind
                SRUN_MPI_FLAG="--mpi=pmix",
            )
        )
        if "{{" in slurm_text or "}}" in slurm_text:
            raise RuntimeError("SLURM_TEMPLATE render left unresolved placeholders. Check keys passed to render_slurm_script().")

        with log_out:
            print("[remote] Remote run dir:", rdir)
            print("[remote] Writing params.yaml + slurm_run.sh, then sbatch…")
            print("-" * 70)

        try:
            jobid = remote_stage_and_submit(
                host=host,
                user=user,
                port=port,
                remote_dir=rdir,
                params_text=params_text,
                slurm_text=slurm_text,
            )

            STATUS["remote_dir"] = rdir
            STATUS["jobid"] = jobid

            set_status("done")
            with log_out:
                print("[remote] ✅ Submitted generated slurm_run.sh")
                print("  jobid :", jobid)
                print("  rdir  :", rdir)
                print("  example dir:", remote_example_dir)

        except subprocess.TimeoutExpired:
            set_status("fail")
            with log_out:
                print("[remote][TIMEOUT] SSH/Sbatch step timed out.")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_remote_test():
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)

        with log_out:
            print("[remote] Test SSH")
            print("  host:", host)
            print("  user:", user)
            print("  port:", port)
            print("  cmd : hostname && whoami && date")
            print("-" * 70)

        if not host or not user:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] Provide Host + User first.")
            return

        try:
            r = ssh_run(host, user, port, "hostname && whoami && date", timeout=15)
            with log_out:
                print("returncode:", r.returncode)

                if r.stdout.strip():
                    print("--- stdout ---")
                    print(r.stdout.strip())

                if r.stderr.strip():
                    print("--- stderr ---")
                    print(r.stderr.strip())

                if r.returncode != 0:
                    err = (r.stderr or "").lower()

                    if "permission denied" in err:
                        print()
                        print("⚠ SSH authentication failed.")
                        print("Looks like passwordless SSH is not configured.")
                        print()
                        print("➡ Fix:")
                        print("   1) Switch Auth → 'Bootstrap with password'")
                        print("   2) Enter your cluster password")
                        print("   3) Click 'Enable passwordless SSH'")
                        print()
                        print("After that the UI will connect automatically.")

                    elif "timed out" in err or "connection timed out" in err:
                        print()
                        print("⚠ Connection timed out.")
                        print("Check VPN, firewall, or hostname.")

                    elif "could not resolve hostname" in err:
                        print()
                        print("⚠ Hostname not reachable.")
                        print("Check the cluster hostname.")

            set_status("done" if r.returncode == 0 else "fail")

        except subprocess.TimeoutExpired:
            set_status("fail")
            with log_out:
                print("[remote][TIMEOUT] SSH did not respond within 15s.")
                print("Likely: network/DNS issue, firewall/VPN, or auth prompt prevented non-interactive login.")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_remote_status():
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)

        jobid = STATUS.get("jobid")

        with log_out:
            print("[remote] Check status")
            print("  host:", host)
            print("  user:", user)
            print("  jobid:", jobid)
            print("-" * 70)

        if not jobid:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] No JobID yet. Submit first.")
            return

        try:
            r = ssh_run(host, user, port, f"squeue -j {jobid} -o '%i %T %M %D %R'", timeout=15)

            with log_out:
                if r.stdout.strip():
                    print("--- squeue ---")
                    print(r.stdout.strip())
                    set_status("done" if r.returncode == 0 else "fail")
                    return

            # If squeue is empty, ask sacct (history)
            r2 = ssh_run(
                host, user, port,
                f"sacct -j {jobid} --format=JobID,JobName%20,Partition,Account,State,ExitCode,Elapsed -X",
                timeout=15
            )
            with log_out:
                print("(squeue empty; job likely finished or left the queue)")
                print("--- sacct ---")
                print((r2.stdout or "").strip() or "(no sacct output)")
                if r2.stderr.strip():
                    print("--- stderr ---")
                    print(r2.stderr.strip())

            set_status("done" if r2.returncode == 0 else "fail")

        except subprocess.TimeoutExpired:
            set_status("fail")
            with log_out:
                print("[remote][TIMEOUT] Status check timed out.")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_remote_cancel():
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)

        jobid = STATUS.get("jobid")

        with log_out:
            print("[remote] Cancel job")
            print("  host:", host)
            print("  user:", user)
            print("  jobid:", jobid)
            print("-" * 70)

        if not jobid:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] No JobID found.")
            return

        try:
            r = ssh_run(host, user, port, f"scancel {jobid}", timeout=15)

            with log_out:
                print("returncode:", r.returncode)

                if r.stdout.strip():
                    print("--- stdout ---")
                    print(r.stdout.strip())

                if r.stderr.strip():
                    print("--- stderr ---")
                    print(r.stderr.strip())

            if r.returncode == 0:
                with log_out:
                    print(f"✅ Job {jobid} cancelled.")
                set_status("done")
            else:
                set_status("fail")

        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_remote_tail():
        log_out.clear_output()
        set_status("running")

        host = cluster_host.value.strip()
        user = cluster_user.value.strip()
        port = int(cluster_port.value)

        rdir = STATUS.get("remote_dir")
        jobid = STATUS.get("jobid")

        with log_out:
            print("[remote] Tail log")
            print("  host:", host)
            print("  user:", user)
            print("  rdir:", rdir)
            print("  jobid:", jobid)
            print("-" * 70)

        if not rdir or not jobid:
            set_status("fail")
            with log_out:
                print("[remote][ERROR] No remote dir / JobID. Submit first.")
            return

        out_file = f"{rdir}/icesee-enkf-{jobid}.out"
        cmd = f"test -f {sh_quote(out_file)} && tail -n 120 {sh_quote(out_file)} || echo 'log not yet created'"

        try:
            r = ssh_run(host, user, port, cmd, timeout=15)
            with log_out:
                print("[remote] file:", out_file)
                print("--- tail ---")
                print((r.stdout or "").rstrip())
                if r.stderr.strip():
                    print("--- stderr ---")
                    print(r.stderr.strip())

            set_status("done" if r.returncode == 0 else "fail")

        except subprocess.TimeoutExpired:
            set_status("fail")
            with log_out:
                print("[remote][TIMEOUT] Tail timed out.")
        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_cloud_submit():
        example_cfg = EXAMPLES[example_dd.value]

        sync_quick_into_widgets()
        cfg_yaml = build_config_from_widgets()

        rd = run_dir()
        dump_yaml(cfg_yaml, rd / "params.yaml")

        cfg = AWSBatchConfig(
            region=aws_region.value.strip() or "us-east-1",
            profile=(aws_profile.value.strip() or None),
            s3_prefix=cloud_bucket.value.strip(),
            job_queue=batch_job_queue.value.strip(),
            job_definition=batch_job_def.value.strip(),
            job_name=(batch_job_name.value.strip() or "icesee"),
        )

        set_status("running")
        log_out.clear_output()
        with log_out:
            print("[cloud] AWS Batch submit")
            print("region :", cfg.region)
            print("profile:", cfg.profile or "(default)")
            print("s3     :", cfg.s3_prefix)

        try:
            aws_test(cfg)
            resp = aws_batch_submit(cfg, rd, example_dd.value, find_run_script(example_cfg).name)

            STATUS["batch_job_id"] = resp["batch_job_id"]
            STATUS["s3_run"] = resp["s3_run"]

            set_status("done")
            with log_out:
                print("[cloud] Submitted.")
                print("batch_job_id:", resp["batch_job_id"])
                print("s3_run      :", resp["s3_run"])
                print("\n[batch image requirement]")
                print("Your AWS Batch container must read ICESEE_S3_RUN and ICESEE_RUN_SCRIPT,")
                print("download params.yaml from S3, run, then sync results back to S3.")

        except Exception as e:
            set_status("fail")
            with log_out:
                print("[cloud][ERROR]", type(e).__name__, e)

    def run_example_cloud_status():
        if not STATUS.get("batch_job_id"):
            with log_out:
                print("[cloud] No Batch job id yet. Submit first.")
            return
        cfg = AWSBatchConfig(
            region=aws_region.value.strip() or "us-east-1",
            profile=(aws_profile.value.strip() or None),
        )
        try:
            st = aws_batch_status(cfg, STATUS["batch_job_id"])
            with log_out:
                print("[cloud] status:", st["status"])
                if st["reason"]:
                    print("[cloud] reason:", st["reason"])
        except Exception as e:
            with log_out:
                print("[cloud][ERROR]", type(e).__name__, e)

    def run_example_cloud_logs_hint():
        if not STATUS.get("batch_job_id"):
            with log_out:
                print("[cloud] No Batch job id yet.")
            return
        with log_out:
            print("[cloud] Logs depend on your job definition (awslogs driver).")
            print("Open the AWS Console -> Batch -> Job -> Logs")
            print("JobID:", STATUS["batch_job_id"])
            if STATUS.get("s3_run"):
                print("S3 run prefix:", STATUS["s3_run"])

    # master run
    def run_example():
        mode = get_mode()
        if mode == MODE_REMOTE:
            return run_example_remote_submit()
        if mode == MODE_CLOUD:
            return run_example_cloud_submit()
        return run_example_local()

    # =========================================================
    # Wire buttons
    # =========================================================
    # run_btn.on_click(lambda b: run_example())
    action_btn.on_click(on_action_click)
    clear_btn.on_click(lambda b: (log_out.clear_output(), results_out.clear_output(), set_status("idle")))

    connect_btn.on_click(lambda b: run_example_remote_test())
    submit_btn.on_click(lambda b: run_example_remote_submit())
    status_btn.on_click(lambda b: run_example_remote_status())
    tail_btn.on_click(lambda b: run_example_remote_tail())
    terminate_btn.on_click(lambda b: run_example_remote_cancel())

    cloud_submit_btn.on_click(lambda b: run_example_cloud_submit())
    cloud_status_btn.on_click(lambda b: run_example_cloud_status())
    cloud_logs_btn.on_click(lambda b: run_example_cloud_logs_hint())

    # keep template in sync with quick knobs
    def _sync_knobs(_=None):
        sync_quick_into_widgets()

    filter_alg_dd.observe(_sync_knobs, names="value")
    ens_sl.observe(_sync_knobs, names="value")
    seed_in.observe(_sync_knobs, names="value")

    bootstrap_btn.on_click(on_bootstrap_keys)

   # =========================================================
    # UX CSS
    # =========================================================
    css = """
    <style>
    /* --- your existing styles --- */
    .icesee-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    .icesee-title { font-size: 18px; font-weight: 700; margin: 6px 0 4px; }
    .icesee-subtitle { color: rgba(0,0,0,.65); margin-bottom: 14px; }
    .icesee-card { border: 1px solid rgba(0,0,0,.10); border-radius: 12px; padding: 14px; background: #fff; }
    .icesee-h { font-size: 18px; font-weight: 800; margin: 2px 0 10px; }
    .icesee-lbl { width: 120px; font-weight: 650; }
    .icesee-k { width: 220px; font-weight: 650; color: rgba(0,0,0,.78); }
    .icesee-subtle { color: rgba(0,0,0,.60); font-size: 12px; }
    .icesee-status { display:inline-block; padding: 8px 14px; border-radius: 999px; font-weight: 700; border: 1px solid rgba(0,0,0,.10); }
    .icesee-idle { background: rgba(0,0,0,.04); }
    .icesee-running { background: rgba(16, 122, 255, .12); }
    .icesee-done { background: rgba(30, 170, 80, .14); }
    .icesee-fail { background: rgba(220, 60, 60, .14); }

    /* --- make notebook/page use full width (JLab/classic) --- */
    .jp-NotebookPanel, .jp-Notebook, .jp-Cell, .jp-OutputArea { max-width: 100% !important; }
    .icesee-page { width: 100% !important; }

    /* --- stretch left/right columns properly --- */
    .icesee-row { display: flex; gap: 26px; width: 100%; align-items: stretch; }
    .icesee-col { flex: 1 1 0; min-width: 0; }  /* min-width:0 is KEY */

    /* --- outputs: full width + readable long lines --- */
    .icesee-out { width: 100% !important; }
    .icesee-out .output_area pre {
    white-space: pre;      /* keep formatting */
    overflow-x: auto;      /* horizontal scroll for long lines */
    }

    /* Optional: if you're in Jupyter Book and it's still constrained, uncomment:
    .bd-main .bd-content, .bd-container, .container-xl, .container-lg { max-width: 100% !important; }
    */
    </style>
    """
    display(W.HTML(css))

    # =========================================================
    # Layout
    # =========================================================
    header = W.HTML(
        "<div class='icesee-wrap'>"
        "<div class='icesee-title'>Run ICESEE examples with one click.</div>"
        "<div class='icesee-subtitle'>Outputs and reports are saved and previewed on the right.</div>"
        "</div>"
    )

    # Local tab content
    local_tab_card = W.VBox([W.HTML("<div class='icesee-subtle'>Local mode runs directly in this notebook.</div>")])
    local_tab_card.add_class("icesee-card")

    # Remote panel
    cluster_panel = W.VBox(
        [
            W.HTML("<div class='icesee-h'>Remote</div>"),
            W.HTML("<div class='icesee-subtle'>SSH connection</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Host:</div>"), cluster_host], layout=W.Layout(gap="12px")),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>User:</div>"), cluster_user, W.HTML("<div class='icesee-lbl'>Port:</div>"), cluster_port],
                layout=W.Layout(gap="12px"),
            ),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>Remote dir:</div>"), remote_base_dir, W.HTML("<div class='icesee-lbl'>Tag:</div>"), remote_tag],
                layout=W.Layout(gap="12px"),
            ),
            W.HTML("<div class='icesee-subtle' style='margin-top:10px'>ICESEE-Spack</div>"),
            W.Box([spack_enable], layout=W.Layout(margin="0 0 0 120px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Repo:</div>"), spack_repo_url], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Dir name:</div>"), spack_dirname], layout=W.Layout(gap="12px")),
            W.Box([spack_install_if_needed], layout=W.Layout(margin="0 0 0 120px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Install:</div>"), spack_install_mode], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>SLURM_DIR:</div>"), spack_slurm_dir], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>PMIX_DIR:</div>"),  spack_pmix_dir],  layout=W.Layout(gap="12px")),
            W.Box([spack_use_existing_sbatch], layout=W.Layout(margin="0 0 0 120px")),
            # W.HBox([connect_btn, submit_btn, status_btn, tail_btn], layout=W.Layout(gap="10px")),
            W.HBox([connect_btn, status_btn, tail_btn, terminate_btn], layout=W.Layout(gap="10px")),
            W.HTML("<div class='icesee-subtle' style='margin-top:8px'>Slurm resources</div>"),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>Job:</div>"), slurm_job_name, W.HTML("<div class='icesee-lbl'>Time:</div>"), slurm_time],
                layout=W.Layout(gap="12px"),
            ),
            W.HBox(
                [
                    W.HTML("<div class='icesee-lbl'>Nodes:</div>"),
                    slurm_nodes,
                    W.HTML("<div class='icesee-lbl'>Tasks:</div>"),
                    slurm_ntasks,
                    W.HTML("<div class='icesee-lbl'>TPN:</div>"),
                    slurm_tpn,
                ],
                layout=W.Layout(gap="12px"),
            ),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>Part:</div>"), slurm_part, W.HTML("<div class='icesee-lbl'>Mem:</div>"), slurm_mem],
                layout=W.Layout(gap="12px"),
            ),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>Acct:</div>"), slurm_account, W.HTML("<div class='icesee-lbl'>Mail:</div>"), slurm_mail],
                layout=W.Layout(gap="12px"),
            ),
            W.HBox(
                [W.HTML("<div class='icesee-lbl'>MPI np:</div>"), cluster_mpi_np, W.HTML("<div class='icesee-lbl'>Model nprocs:</div>"), cluster_model_nprocs],
                layout=W.Layout(gap="12px"),
            ),
            W.HTML("<div class='icesee-subtle' style='margin-top:10px'>Modules</div>"),
            remote_module_lines,
            W.HTML("<div class='icesee-subtle' style='margin-top:10px'>Exports</div>"),
            remote_export_lines,
            W.HTML("<div class='icesee-subtle' style='margin-top:10px'>Auth</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Method:</div>"), auth_mode], layout=W.Layout(gap="12px")),
            W.Box([cluster_password], layout=W.Layout(margin="0 0 0 120px")),
            W.Box([bootstrap_btn], layout=W.Layout(margin="0 0 0 120px")),
        ],
        layout=W.Layout(gap="8px"),
    )
    cluster_panel.add_class("icesee-card")

    # Cloud panel
    cloud_panel = W.VBox(
        [
            W.HTML("<div class='icesee-h'>Cloud</div>"),
            W.HTML("<div class='icesee-subtle'>AWS Batch backend via AWS CLI.</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Region:</div>"), aws_region, W.HTML("<div class='icesee-lbl'>Profile:</div>"), aws_profile],
                   layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>S3 prefix:</div>"), cloud_bucket], layout=W.Layout(gap="12px")),
            W.HTML("<div class='icesee-subtle' style='margin-top:10px'>AWS Batch</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Queue:</div>"), batch_job_queue], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Job def:</div>"), batch_job_def], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Job name:</div>"), batch_job_name], layout=W.Layout(gap="12px")),
            W.HBox([cloud_submit_btn, cloud_status_btn, cloud_logs_btn], layout=W.Layout(gap="10px")),
        ],
        layout=W.Layout(gap="8px"),
    )
    cloud_panel.add_class("icesee-card")

    mode_tabs.children = [local_tab_card, cluster_panel, cloud_panel]
    mode_tabs.set_title(0, "Local (GHUB)")
    mode_tabs.set_title(1, "Remote")
    mode_tabs.set_title(2, "Cloud")

    local_tab_card.layout = W.Layout(width="100%")
    cluster_panel.layout   = W.Layout(width="100%")
    cloud_panel.layout     = W.Layout(width="100%")

    def _toggle_panels_from_tabs(_=None):
        mode = get_mode()
        cluster_panel.layout.display = "block" if mode == MODE_REMOTE else "none"
        cloud_panel.layout.display = "block" if mode == MODE_CLOUD else "none"

        is_remote = (mode == MODE_REMOTE)
        connect_btn.disabled = not is_remote
        submit_btn.disabled = not is_remote
        status_btn.disabled = not is_remote
        tail_btn.disabled = not is_remote
        terminate_btn.disabled = not is_remote

        is_cloud = (mode == MODE_CLOUD)
        cloud_submit_btn.disabled = not is_cloud
        cloud_status_btn.disabled = not is_cloud
        cloud_logs_btn.disabled = not is_cloud

    # mode_tabs.observe(_toggle_panels_from_tabs, names="selected_index")
    # _toggle_panels_from_tabs()
    def _toggle_panels_from_tabs(_=None):
        mode = get_mode()
        # (keep your existing enable/disable logic here)

        update_action_button()

    mode_tabs.observe(_toggle_panels_from_tabs, names="selected_index")
    _toggle_panels_from_tabs()
    update_action_button()

    left = W.VBox(
        [
            W.HTML("<div class='icesee-h'>Run settings</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Mode:</div>"), mode_tabs], layout=W.Layout(gap="12px", width="100%")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Example:</div>"), example_dd], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Preset:</div>"), preset_dd], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Filter:</div>"), filter_alg_dd], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Output:</div>"), output_label_dd], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Ens:</div>"), ens_sl], layout=W.Layout(gap="12px")),
            W.HBox([W.HTML("<div class='icesee-lbl'>Seed:</div>"), seed_in], layout=W.Layout(gap="12px")),
            W.Box([gen_report], layout=W.Layout(margin="6px 0 0 120px")),
            W.Box([open_latest], layout=W.Layout(margin="0 0 8px 120px")),
            W.HTML("<div class='icesee-subtle' style='margin:8px 0 8px'>Full configuration (from <code>params.yaml</code>)</div>"),
            params_holder,
        ],
        layout=W.Layout(gap="8px"),
    )
    left_card = W.VBox([left])
    left_card.add_class("icesee-card")
    left_card.layout = W.Layout(width="100%")

    right = W.VBox(
        [
            W.HTML("<div class='icesee-h'>Run log</div>"),
            log_out,
            W.HTML("<div class='icesee-h' style='margin-top:14px'>Results preview</div>"),
            results_out,
        ]
    )
    right_card = W.VBox([right])
    right_card.add_class("icesee-card")

    log_out.add_class("icesee-out")
    results_out.add_class("icesee-out")

    # actions = W.HBox([run_btn, clear_btn, status_chip], layout=W.Layout(gap="12px"))
    actions = W.HBox([action_btn, clear_btn, status_chip], layout=W.Layout(gap="12px"))
    actions_card = W.VBox([W.HTML("<div class='icesee-h'>Status</div>"), actions])
    actions_card.add_class("icesee-card")

    left_card.add_class("icesee-col")
    right_card.add_class("icesee-col")

    row = W.HBox([left_card, right_card], layout=W.Layout(width="100%", display="flex", gap="26px"))
    row.add_class("icesee-row")

    page = W.VBox([header, row, actions_card], layout=W.Layout(width="100%"))
    page.add_class("icesee-page")

    cloud_submit_btn.layout.display = "none"

    row = W.HBox([left_card, right_card], layout=W.Layout(width="100%", display="flex", gap="26px"))
    page = W.VBox([header, row, actions_card], layout=W.Layout(width="100%"))
    # page = W.VBox([header, W.HBox([left_card, right_card], layout=W.Layout(gap="26px")), actions_card])
    # display(page)

    set_status("idle")
    rebuild_for_example()
    return page