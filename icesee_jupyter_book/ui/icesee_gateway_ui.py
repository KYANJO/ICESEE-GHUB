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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

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
        base=EXT / "applications" / "lorenz_model" / "examples" / "lorenz96",
        run_script="run_da_lorenz96.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="lorenz",
        figures_dir="figures",
    ),
    "ISSM (under development)": dict(
        enabled=True,
        base=EXT / "applications" / "issm_model" / "examples" / "ISMIP_Choi",
        run_script="run_da_issm.py",
        params="params.yaml",
        report_nb="read_results.m",
        assets=["_modelrun_datasets"],
        model_name="issm",
        figures_dir="figures",
    ),
    "Flowline (under development)": dict(
        enabled=False,
        base=EXT / "applications" / "flowline_model" / "examples" / "flowline_1d",
        run_script="run_da_flowline.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="flowline",
        figures_dir="figures",
    ),
    "Icepack (under development)": dict(
        enabled=False,
        base=EXT / "applications" / "icepack_model" / "examples" / "synthetic_ice_stream",
        run_script="run_da_icepack.py",
        params="params.yaml",
        report_nb="read_results.ipynb",
        assets=["_modelrun_datasets"],
        model_name="icepack",
        figures_dir="figures",
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
def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _ssh_base(host: str, user: str, port: int):
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-p",
        str(int(port)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        target,
    ]


def ssh_run(host: str, user: str, port: int, remote_cmd: str) -> subprocess.CompletedProcess[str]:
    cmd = _ssh_base(host, user, port) + [remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


def make_remote_run_dir(base_dir="~/icesee-runs", tag="icesee") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_dir.rstrip('/')}/{tag}-{ts}"


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -t {{TIME}}
#SBATCH -J {{JOB_NAME}}
#SBATCH -N {{NODES}}
#SBATCH -n {{NTASKS}}
#SBATCH --ntasks-per-node={{TPN}}
#SBATCH --partition={{PARTITION}}
#SBATCH --mem={{MEM}}
#SBATCH -A {{ACCOUNT}}
#SBATCH -o {{OUTFILE}}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={{MAIL_USER}}

set -euo pipefail

cd $SLURM_SUBMIT_DIR

module purge
{{MODULE_LINES}}

# Optional user paths
{{EXPORT_LINES}}

# Run
{{RUN_LAUNCH_LINE}}
"""


def render_slurm_script(d: dict) -> str:
    txt = SLURM_TEMPLATE
    for k, v in d.items():
        txt = txt.replace("{{" + k + "}}", str(v))
    return txt


def remote_stage_and_submit(
    *,
    host: str,
    user: str,
    port: int,
    remote_dir: str,
    local_params: Path,
    local_run_script: Path,
    slurm_text: str,
) -> str:
    # Create remote dir + upload files using ssh + heredoc + cat + base64 to avoid scp dependency
    # (scp usually exists, but this method is reliable on locked-down systems).
    # If you prefer scp, you can swap this out.

    def put_file(local_path: Path, remote_path: str):
        data = local_path.read_bytes()
        b64 = json.dumps(data.decode("latin1"))  # latin1 round-trip safe for bytes
        remote_cmd = f"""
set -e
mkdir -p {remote_dir}
python3 - <<'PY'
import json,sys
s=json.loads({b64})
b=s.encode('latin1')
open({remote_path!r},'wb').write(b)
print("WROTE", {remote_path!r}, "bytes", len(b))
PY
"""
        r = ssh_run(host, user, port, remote_cmd)
        if r.returncode != 0:
            raise RuntimeError(f"Upload failed for {local_path.name}:\n{r.stderr or r.stdout}")

    def put_text(text: str, remote_path: str):
        # safe heredoc
        remote_cmd = f"""
set -e
mkdir -p {remote_dir}
cat > {remote_path} <<'EOF'
{text}
EOF
chmod +x {remote_path}
"""
        r = ssh_run(host, user, port, remote_cmd)
        if r.returncode != 0:
            raise RuntimeError(f"Upload failed for slurm script:\n{r.stderr or r.stdout}")

    put_file(local_params, f"{remote_dir}/params.yaml")
    put_file(local_run_script, f"{remote_dir}/{local_run_script.name}")
    put_text(slurm_text, f"{remote_dir}/slurm_run.sh")

    # Submit
    r = ssh_run(host, user, port, f"cd {remote_dir} && sbatch slurm_run.sh")
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{r.stderr or r.stdout}")

    m = re.search(r"Submitted batch job\s+(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Could not parse JobID from:\n{r.stdout}\n{r.stderr}")
    return m.group(1)


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

    run_btn = W.Button(description="Run", button_style="success", icon="play")
    clear_btn = W.Button(description="Clear", button_style="", icon="trash")

    status_chip = W.HTML("<span class='icesee-status icesee-idle'>Idle</span>")
    log_out = W.Output(layout=W.Layout(border="1px solid rgba(0,0,0,.12)", padding="10px", height="220px", overflow="auto"))
    results_out = W.Output(layout=W.Layout(border="1px solid rgba(0,0,0,.12)", padding="10px", height="260px", overflow="auto"))

    # -----------------------------
    # Mode Tabs
    # -----------------------------
    MODE_LOCAL, MODE_REMOTE, MODE_CLOUD = "local", "cluster", "cloud"
    mode_tabs = W.Tab()
    mode_tabs.layout = W.Layout(width="420px")

    def get_mode():
        return {0: MODE_LOCAL, 1: MODE_REMOTE, 2: MODE_CLOUD}.get(mode_tabs.selected_index, MODE_LOCAL)

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
    cluster_host = W.Text(value="", placeholder="login.cluster.edu", layout=W.Layout(width="320px"))
    cluster_user = W.Text(value=os.environ.get("USER", ""), placeholder="username", layout=W.Layout(width="320px"))
    cluster_port = W.IntText(value=22, layout=W.Layout(width="120px"))

    remote_base_dir = W.Text(value="~/icesee-runs", layout=W.Layout(width="320px"))
    remote_tag = W.Text(value="icesee", layout=W.Layout(width="220px"))

    connect_btn = W.Button(description="Test SSH", icon="link", button_style="")
    submit_btn = W.Button(description="Submit job", icon="cloud-upload", button_style="warning")
    status_btn = W.Button(description="Check status", icon="search", button_style="")
    tail_btn = W.Button(description="Tail log", icon="file-text", button_style="")

    slurm_job_name = W.Text(value="ICESEE", layout=W.Layout(width="220px"))
    slurm_time = W.Text(value="50:00:00", layout=W.Layout(width="220px"))
    slurm_nodes = W.IntText(value=1, layout=W.Layout(width="120px"))
    slurm_ntasks = W.IntText(value=40, layout=W.Layout(width="120px"))
    slurm_tpn = W.IntText(value=24, layout=W.Layout(width="120px"))
    slurm_part = W.Text(value="cpu-large", layout=W.Layout(width="220px"))
    slurm_mem = W.Text(value="256G", layout=W.Layout(width="220px"))
    slurm_account = W.Text(value="", placeholder="account", layout=W.Layout(width="220px"))
    slurm_mail = W.Text(value="", placeholder="email", layout=W.Layout(width="220px"))

    cluster_mpi_np = W.IntText(value=40, layout=W.Layout(width="120px"))
    cluster_model_nprocs = W.IntText(value=4, layout=W.Layout(width="120px"))

    # minimal module/export lines (you can expand later)
    remote_module_lines = W.Textarea(
        value="module load gcc/13\n",
        layout=W.Layout(width="100%", height="80px"),
    )
    remote_export_lines = W.Textarea(
        value="# export ISSM_DIR=...\n",
        layout=W.Layout(width="100%", height="80px"),
    )

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
        example_cfg = EXAMPLES[example_dd.value]

        if not cluster_host.value.strip() or not cluster_user.value.strip():
            set_status("fail")
            with log_out:
                print("[remote][ERROR] Provide Host + User.")
            return

        sync_quick_into_widgets()
        cfg_yaml = build_config_from_widgets()

        rd = run_dir()
        dump_yaml(cfg_yaml, rd / "params.yaml")

        rdir = make_remote_run_dir(remote_base_dir.value.strip() or "~/icesee-runs", remote_tag.value.strip() or "icesee")

        outfile = f"steadystate-%j.out"

        slurm_text = render_slurm_script(
            dict(
                TIME=slurm_time.value.strip(),
                JOB_NAME=slurm_job_name.value.strip() or "ICESEE",
                NODES=int(slurm_nodes.value),
                NTASKS=int(slurm_ntasks.value),
                TPN=int(slurm_tpn.value),
                PARTITION=slurm_part.value.strip(),
                MEM=slurm_mem.value.strip(),
                ACCOUNT=(slurm_account.value.strip() or "REPLACE_ME"),
                OUTFILE=outfile,
                MAIL_USER=(slurm_mail.value.strip() or "REPLACE_ME"),
                MODULE_LINES=remote_module_lines.value.rstrip(),
                EXPORT_LINES=remote_export_lines.value.rstrip(),
                RUN_LAUNCH_LINE=f"mpirun -np {int(cluster_mpi_np.value)} python3 {find_run_script(example_cfg).name} "
                                f"-F params.yaml --Nens={int(ens_sl.value)} --model_nprocs={int(cluster_model_nprocs.value)}",
            )
        )

        set_status("running")
        log_out.clear_output()
        with log_out:
            print("[remote] Example:", example_dd.value)
            print("[remote] Host   :", cluster_host.value.strip())
            print("[remote] Dir    :", rdir)

        try:
            jobid = remote_stage_and_submit(
                host=cluster_host.value.strip(),
                user=cluster_user.value.strip(),
                port=int(cluster_port.value),
                remote_dir=rdir,
                local_params=rd / "params.yaml",
                local_run_script=find_run_script(example_cfg),
                slurm_text=slurm_text,
            )
            STATUS["remote_dir"] = rdir
            STATUS["jobid"] = jobid
            set_status("done")
            with log_out:
                print("[remote] Submitted.")
                print("[remote] JobID :", jobid)
                print("[remote] Next : Check status / Tail log")

        except Exception as e:
            set_status("fail")
            with log_out:
                print("[remote][ERROR]", type(e).__name__, e)

    def run_example_remote_test():
        log_out.clear_output()
        if not cluster_host.value.strip() or not cluster_user.value.strip():
            with log_out:
                print("[remote] Provide host + user first.")
            return
        r = ssh_run(cluster_host.value.strip(), cluster_user.value.strip(), int(cluster_port.value), "hostname && whoami && date")
        with log_out:
            print("[remote] returncode:", r.returncode)
            if r.stdout:
                print("--- stdout ---")
                print(r.stdout.strip())
            if r.stderr:
                print("--- stderr ---")
                print(r.stderr.strip())
            if r.returncode != 0:
                print("\n[hint] Needs non-interactive SSH auth (keys/agent).")

    def run_example_remote_status():
        if not STATUS.get("jobid"):
            with log_out:
                print("[remote] No JobID yet. Submit first.")
            return
        jobid = STATUS["jobid"]
        r = ssh_run(cluster_host.value.strip(), cluster_user.value.strip(), int(cluster_port.value),
                    f"squeue -j {jobid} -o '%i %T %M %D %R'")
        with log_out:
            print("[remote] squeue:")
            print((r.stdout.strip() or "(not in queue)"))
            if r.stderr.strip():
                print("STDERR:", r.stderr.strip())

    def run_example_remote_tail():
        if not STATUS.get("remote_dir") or not STATUS.get("jobid"):
            with log_out:
                print("[remote] No remote dir / JobID. Submit first.")
            return
        rdir, jobid = STATUS["remote_dir"], STATUS["jobid"]
        out_file = f"{rdir}/steadystate-{jobid}.out"
        r = ssh_run(cluster_host.value.strip(), cluster_user.value.strip(), int(cluster_port.value),
                    f"test -f {_sh_quote(out_file)} && tail -n 80 {_sh_quote(out_file)} || echo 'log not yet created'")
        with log_out:
            print("[remote] tail:", out_file)
            print(r.stdout.rstrip())
            if r.stderr.strip():
                print("STDERR:", r.stderr.strip())

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
    run_btn.on_click(lambda b: run_example())
    clear_btn.on_click(lambda b: (log_out.clear_output(), results_out.clear_output(), set_status("idle")))

    connect_btn.on_click(lambda b: run_example_remote_test())
    submit_btn.on_click(lambda b: run_example_remote_submit())
    status_btn.on_click(lambda b: run_example_remote_status())
    tail_btn.on_click(lambda b: run_example_remote_tail())

    cloud_submit_btn.on_click(lambda b: run_example_cloud_submit())
    cloud_status_btn.on_click(lambda b: run_example_cloud_status())
    cloud_logs_btn.on_click(lambda b: run_example_cloud_logs_hint())

    # keep template in sync with quick knobs
    def _sync_knobs(_=None):
        sync_quick_into_widgets()

    filter_alg_dd.observe(_sync_knobs, names="value")
    ens_sl.observe(_sync_knobs, names="value")
    seed_in.observe(_sync_knobs, names="value")

    # =========================================================
    # UX CSS
    # =========================================================
    css = """
    <style>
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
            W.HBox([connect_btn, submit_btn, status_btn, tail_btn], layout=W.Layout(gap="10px")),
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

    def _toggle_panels_from_tabs(_=None):
        mode = get_mode()
        cluster_panel.layout.display = "block" if mode == MODE_REMOTE else "none"
        cloud_panel.layout.display = "block" if mode == MODE_CLOUD else "none"

        is_remote = (mode == MODE_REMOTE)
        connect_btn.disabled = not is_remote
        submit_btn.disabled = not is_remote
        status_btn.disabled = not is_remote
        tail_btn.disabled = not is_remote

        is_cloud = (mode == MODE_CLOUD)
        cloud_submit_btn.disabled = not is_cloud
        cloud_status_btn.disabled = not is_cloud
        cloud_logs_btn.disabled = not is_cloud

    mode_tabs.observe(_toggle_panels_from_tabs, names="selected_index")
    _toggle_panels_from_tabs()

    left = W.VBox(
        [
            W.HTML("<div class='icesee-h'>Run settings</div>"),
            W.HBox([W.HTML("<div class='icesee-lbl'>Mode:</div>"), mode_tabs], layout=W.Layout(gap="12px")),
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

    actions = W.HBox([run_btn, clear_btn, status_chip], layout=W.Layout(gap="12px"))
    actions_card = W.VBox([W.HTML("<div class='icesee-h'>Status</div>"), actions])
    actions_card.add_class("icesee-card")

    page = W.VBox([header, W.HBox([left_card, right_card], layout=W.Layout(gap="26px")), actions_card])
    # display(page)

    set_status("idle")
    rebuild_for_example()
    return page