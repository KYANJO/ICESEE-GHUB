# icesee_jupyter_book/ui/icesee_runner.py
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
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

from .example_registry import EXAMPLES


# ============================================================
# Repo / paths
# ============================================================
def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / "external" / "ICESEE").exists() and (p / "icesee_jupyter_book").exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Could not locate repo root containing external/ICESEE and icesee_jupyter_book."
    )


REPO = find_repo_root()
BOOK = REPO / "icesee_jupyter_book"
EXT = REPO / "external" / "ICESEE"


def enabled_examples() -> list[str]:
    return [k for k, v in EXAMPLES.items() if v.get("enabled", False)]


def get_example(example_name: str) -> dict:
    if example_name not in EXAMPLES:
        raise KeyError(f"Unknown example: {example_name}")
    return EXAMPLES[example_name]


# ============================================================
# YAML helpers
# ============================================================
def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ============================================================
# Discovery helpers
# ============================================================
def find_run_script(example_cfg: dict) -> Path:
    base = Path(example_cfg["base"])
    rs = example_cfg.get("run_script")
    if rs and (base / rs).exists():
        return base / rs

    candidates = list(base.rglob("run_da_*.py")) + list(base.rglob("run_*.py"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No run script found under {base}")

    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]


def find_params_template(example_cfg: dict) -> Path:
    wrapper_template = BOOK / "params.yaml"
    if wrapper_template.exists():
        return wrapper_template

    base = Path(example_cfg["base"])
    p = base / (example_cfg.get("params") or "params.yaml")
    if p.exists():
        return p

    candidates = list(base.rglob("params.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No params.yaml found under {base}")

    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]


def find_report_notebook(example_cfg: dict) -> Path | None:
    nb = example_cfg.get("report_nb")
    if not nb:
        return None
    p = Path(example_cfg["base"]) / nb
    return p if p.exists() else None


# ============================================================
# Run directory / result helpers
# ============================================================
def make_run_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or (BOOK / "icesee_runs")
    rd = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results").mkdir(exist_ok=True)
    (rd / "figures").mkdir(exist_ok=True)
    return rd


def force_external_icesee_env() -> tuple[dict, Path]:
    external_dir = (REPO / "external").resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{external_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONNOUSERSITE"] = "1"
    return env, external_dir


def mirror_assets_for_report(example_cfg: dict, run_dir: Path) -> None:
    base = Path(example_cfg["base"])
    for asset in example_cfg.get("assets", []):
        src = base / asset
        if src.exists():
            dst = run_dir / asset
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def ensure_report_h5(
    run_dir: Path,
    example_cfg: dict,
    expected_prefix: str,
) -> Path:
    model_name = example_cfg.get("model_name", "lorenz")
    expected = run_dir / "results" / f"{expected_prefix}-{model_name}.h5"
    if expected.exists():
        return expected

    candidates = sorted(
        (run_dir / "results").glob(f"*-{model_name}.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        base = Path(example_cfg["base"])
        candidates = sorted(
            base.glob(f"**/results/*-{model_name}.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if candidates:
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], expected)

    return expected


def list_run_outputs(run_dir: Path) -> dict:
    fig_dir = run_dir / "figures"
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        pngs = sorted((run_dir / "results").glob("*.png"))
    h5s = sorted((run_dir / "results").glob("*.h5"))

    return {
        "run_dir": run_dir,
        "h5_files": h5s,
        "png_files": pngs,
    }


# ============================================================
# Local execution
# ============================================================
@dataclass
class LocalRunResult:
    success: bool
    returncode: int
    run_dir: Path
    command: list[str]
    log_text: str
    report_notebook: Path | None = None
    outputs: dict | None = None


def run_local_example(
    example_cfg: dict,
    config: dict,
    output_label: str = "true-wrong",
    generate_report: bool = False,
    run_dir: Path | None = None,
) -> LocalRunResult:
    run_script = find_run_script(example_cfg)
    report_nb = find_report_notebook(example_cfg)
    rd = run_dir or make_run_dir()

    dump_yaml(config, rd / "params.yaml")

    env, _ = force_external_icesee_env()
    cmd = [sys.executable, str(run_script), "-F", str(rd / "params.yaml")]

    proc = subprocess.Popen(
        cmd,
        cwd=str(rd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)

    rc = proc.wait()
    log_text = "".join(lines)

    looks_like_failure = (
        "Traceback (most recent call last)" in log_text
        or "Error in serial run mode" in log_text
    )

    success = (rc == 0) and (not looks_like_failure)

    executed_report = None
    if success and generate_report and report_nb and report_nb.exists():
        ensure_report_h5(rd, example_cfg, output_label)
        executed_report = execute_report_notebook(report_nb, example_cfg, rd)

    outputs = list_run_outputs(rd)

    return LocalRunResult(
        success=success,
        returncode=rc,
        run_dir=rd,
        command=cmd,
        log_text=log_text,
        report_notebook=executed_report,
        outputs=outputs,
    )


def execute_report_notebook(report_nb: Path, example_cfg: dict, run_dir: Path) -> Path:
    try:
        import papermill as pm
    except Exception as e:
        raise RuntimeError(
            "papermill is required to execute report notebooks automatically."
        ) from e

    mirror_assets_for_report(example_cfg, run_dir)
    nb_out = run_dir / "report.ipynb"

    pm.execute_notebook(
        input_path=str(report_nb),
        output_path=str(nb_out),
        cwd=str(run_dir),
        log_output=True,
    )
    return nb_out


# ============================================================
# SSH / remote helpers
# ============================================================
def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def ssh_base(host: str, user: str, port: int) -> list[str]:
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
    cmd = ssh_base(host, user, port) + [remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def rsh(host: str, user: str, port: int, cmd: str, timeout: int = 60):
    r = ssh_run(host, user, port, cmd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


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


def make_remote_run_dir(base_dir: str = "~/r-arobel3-0", tag: str = "icesee") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_dir.rstrip('/')}/{tag}-{ts}"


def sanitize_multiline(text: str) -> str:
    return "\n".join([ln.rstrip() for ln in (text or "").splitlines() if ln.strip() != ""])


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

module purge || true
{{MODULE_LINES}}

SPACK_PATH="{{SPACK_PATH}}"
if [ ! -d "${SPACK_PATH}" ]; then
  echo "[ERROR] SPACK_PATH does not exist: ${SPACK_PATH}"
  exit 2
fi
source "${SPACK_PATH}/scripts/activate.sh"

{{EXPORT_LINES}}

NP="{{NP}}"
NENS="{{NENS}}"
MODEL_NPROCS="{{MODEL_NPROCS}}"
RUN_SCRIPT="{{RUN_SCRIPT}}"
PARAMS_PATH="{{PARAMS_PATH}}"
EXAMPLE_DIR="{{EXAMPLE_DIR}}"

cd "${EXAMPLE_DIR}"

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


def render_slurm_script(values: dict) -> str:
    txt = SLURM_TEMPLATE
    for k, v in values.items():
        txt = txt.replace("{{" + k + "}}", str(v))
    return txt


@dataclass
class RemoteJobResult:
    jobid: str
    remote_dir: str
    remote_example_dir: str
    slurm_script: str


def remote_stage_and_submit(
    *,
    host: str,
    user: str,
    port: int,
    remote_dir: str,
    params_text: str,
    slurm_text: str,
) -> str:
    remote_write_text(host, user, port, f"{remote_dir}/params.yaml", params_text, timeout=30)
    remote_write_text(host, user, port, f"{remote_dir}/slurm_run.sh", slurm_text, timeout=30)

    r = ssh_run(
        host,
        user,
        port,
        f"chmod +x {sh_quote(remote_dir+'/slurm_run.sh')} && cd {sh_quote(remote_dir)} && sbatch slurm_run.sh",
        timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)

    m = re.search(r"Submitted batch job\s+(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Could not parse JobID from:\n{r.stdout}\n{r.stderr}")
    return m.group(1)


def remote_test_connection(host: str, user: str, port: int) -> dict:
    r = ssh_run(host, user, port, "hostname && whoami && date", timeout=15)
    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "ok": (r.returncode == 0),
    }


def remote_job_status(host: str, user: str, port: int, jobid: str) -> dict:
    r = ssh_run(host, user, port, f"squeue -j {jobid} -o '%i %T %M %D %R'", timeout=15)
    if r.returncode == 0 and r.stdout.strip():
        return {
            "source": "squeue",
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }

    r2 = ssh_run(
        host,
        user,
        port,
        f"sacct -j {jobid} --format=JobID,JobName%20,Partition,Account,State,ExitCode,Elapsed -X",
        timeout=15,
    )
    return {
        "source": "sacct",
        "returncode": r2.returncode,
        "stdout": r2.stdout,
        "stderr": r2.stderr,
    }


def remote_tail_log(host: str, user: str, port: int, remote_dir: str, jobid: str, n: int = 120) -> dict:
    out_file = f"{remote_dir}/icesee-enkf-{jobid}.out"
    cmd = f"test -f {sh_quote(out_file)} && tail -n {int(n)} {sh_quote(out_file)} || echo 'log not yet created'"
    r = ssh_run(host, user, port, cmd, timeout=15)
    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "log_file": out_file,
    }


def remote_cancel_job(host: str, user: str, port: int, jobid: str) -> dict:
    r = ssh_run(host, user, port, f"scancel {jobid}", timeout=15)
    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "ok": (r.returncode == 0),
    }


# ============================================================
# SSH key bootstrap
# ============================================================
def ensure_local_ssh_key(key_type: str = "ed25519") -> tuple[Path, Path]:
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(ssh_dir, 0o700)

    priv = ssh_dir / f"id_{key_type}"
    pub = ssh_dir / f"id_{key_type}.pub"

    if pub.exists() and priv.exists():
        return priv, pub

    cmd = [
        "ssh-keygen",
        "-t", key_type,
        "-f", str(priv),
        "-N", "",
        "-C", f"icesee-{getpass.getuser()}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout or "ssh-keygen failed")

    os.chmod(priv, 0o600)
    os.chmod(pub, 0o644)
    return priv, pub


def paramiko_connect_password(host: str, user: str, port: int, password: str, timeout: int = 20):
    try:
        import paramiko
    except Exception as e:
        raise RuntimeError("Paramiko is required for password bootstrap.") from e

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
):
    client = paramiko_connect_password(host, user, port, password, timeout=25)

    key_line = pubkey_text.strip()
    if not key_line or "ssh-" not in key_line:
        client.close()
        raise ValueError("Public key text looks invalid.")

    cmd = f"""
set -e
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
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

    return {"returncode": rc, "stdout": out, "stderr": err}


# ============================================================
# Cloud / AWS Batch
# ============================================================
@dataclass
class AWSBatchConfig:
    region: str = "us-east-1"
    profile: str | None = None
    s3_prefix: str = ""
    job_queue: str = ""
    job_definition: str = ""
    job_name: str = "icesee"


def aws_cmd(cfg: AWSBatchConfig) -> list[str]:
    cmd = ["aws"]
    if cfg.profile:
        cmd += ["--profile", cfg.profile]
    if cfg.region:
        cmd += ["--region", cfg.region]
    return cmd


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_s3(s3_uri: str) -> tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/(.*)$", s3_uri.rstrip("/"))
    if not m:
        raise ValueError("S3 path must look like: s3://bucket/prefix")
    return m.group(1), m.group(2)


def aws_test(cfg: AWSBatchConfig) -> None:
    code, out, err = run_cmd(aws_cmd(cfg) + ["sts", "get-caller-identity"])
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
    bucket, prefix = parse_s3(cfg.s3_prefix)
    s3_run = f"s3://{bucket}/{prefix}/{run_id}"

    params_path = local_run_dir / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    code, out, err = run_cmd(aws_cmd(cfg) + ["s3", "cp", str(params_path), f"{s3_run}/params.yaml"])
    if code != 0:
        raise RuntimeError(err or out)

    manifest = {"run_id": run_id, "example": example_name, "run_script": run_script_name}
    (local_run_dir / "cloud_manifest.json").write_text(json.dumps(manifest, indent=2))
    run_cmd(aws_cmd(cfg) + ["s3", "cp", str(local_run_dir / "cloud_manifest.json"), f"{s3_run}/cloud_manifest.json"])

    env = [
        {"name": "ICESEE_S3_RUN", "value": s3_run},
        {"name": "ICESEE_EXAMPLE", "value": example_name},
        {"name": "ICESEE_RUN_SCRIPT", "value": run_script_name},
    ]

    submit_cmd = aws_cmd(cfg) + [
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
    code, out, err = run_cmd(submit_cmd)
    if code != 0:
        raise RuntimeError(err or out)

    job_id = json.loads(out)["jobId"]
    return {"run_id": run_id, "batch_job_id": job_id, "s3_run": s3_run}


def aws_batch_status(cfg: AWSBatchConfig, job_id: str) -> dict:
    code, out, err = run_cmd(aws_cmd(cfg) + ["batch", "describe-jobs", "--jobs", job_id])
    if code != 0:
        raise RuntimeError(err or out)
    job = json.loads(out)["jobs"][0]
    return {"status": job.get("status", "?"), "reason": job.get("statusReason", "")}


# ============================================================
# HTTP helper for future webhook mode
# ============================================================
def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
):
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
            try:
                return resp.status, json.loads(txt), txt
            except Exception:
                return resp.status, None, txt
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return e.code, None, txt