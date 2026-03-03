# @author: Brian Kyanjo
# AWS Batch backend for ICESEE. Uses AWS CLI under the hood, so it should "just work" if you have AWS CLI configured locally.
# Requires:
# - s3_prefix: S3 bucket/prefix for storing run params and (optionally) code bundle. Example: "s3://my-bucket/my-prefix"
# - job_queue and job_definition: AWS Batch job queue and job definition to use for runs. Job definition must be configured to read params from S3 and execute the run script.

from __future__ import annotations
import os, json, time, re, subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AWSBatchConfig:
    region: str = "us-east-1"
    profile: str | None = None          # optional
    s3_prefix: str = ""                 # "s3://bucket/prefix"
    job_queue: str = ""
    job_definition: str = ""            # "name:revision" or "name"
    job_name: str = "icesee"

def _run(cmd: list[str]) -> tuple[int,str,str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def _parse_s3(s3_uri: str) -> tuple[str,str]:
    m = re.match(r"^s3://([^/]+)/(.*)$", s3_uri.rstrip("/"))
    if not m:
        raise ValueError("s3_prefix must look like: s3://bucket/prefix")
    return m.group(1), m.group(2)

class AWSBatchBackend:
    def __init__(self, cfg: AWSBatchConfig):
        self.cfg = cfg

    def _aws(self) -> list[str]:
        base = ["aws"]
        if self.cfg.profile:
            base += ["--profile", self.cfg.profile]
        if self.cfg.region:
            base += ["--region", self.cfg.region]
        return base

    def test(self) -> None:
        # quick auth sanity check
        code, out, err = _run(self._aws() + ["sts", "get-caller-identity"])
        if code != 0:
            raise RuntimeError(f"AWS auth failed:\n{err or out}")

    def submit(self, local_run_dir: Path, example_name: str, run_script_name: str) -> dict:
        """
        Upload params + (optional) code bundle, then submit AWS Batch job.
        Returns dict with run_id and batch_job_id.
        """
        if not self.cfg.s3_prefix or not self.cfg.job_queue or not self.cfg.job_definition:
            raise ValueError("Missing s3_prefix/job_queue/job_definition")

        # run id
        run_id = time.strftime("%Y%m%d-%H%M%S")
        bucket, prefix = _parse_s3(self.cfg.s3_prefix)
        s3_run = f"s3://{bucket}/{prefix}/{run_id}"

        # upload params.yaml
        params_path = local_run_dir / "params.yaml"
        if not params_path.exists():
            raise FileNotFoundError(f"params.yaml not found in {local_run_dir}")

        code, out, err = _run(self._aws() + ["s3", "cp", str(params_path), f"{s3_run}/params.yaml"])
        if code != 0:
            raise RuntimeError(f"Failed to upload params.yaml:\n{err or out}")

        # OPTIONAL: upload a lightweight manifest (helps debugging)
        manifest = {
            "run_id": run_id,
            "example": example_name,
            "run_script": run_script_name,
        }
        manifest_path = local_run_dir / "cloud_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _run(self._aws() + ["s3", "cp", str(manifest_path), f"{s3_run}/cloud_manifest.json"])

        # submit batch job
        env = [
            {"name": "ICESEE_S3_RUN", "value": s3_run},
            {"name": "ICESEE_EXAMPLE", "value": example_name},
            {"name": "ICESEE_RUN_SCRIPT", "value": run_script_name},
        ]

        submit_cmd = self._aws() + [
            "batch", "submit-job",
            "--job-name", f"{self.cfg.job_name}-{run_id}",
            "--job-queue", self.cfg.job_queue,
            "--job-definition", self.cfg.job_definition,
            "--container-overrides", json.dumps({"environment": env}),
        ]

        code, out, err = _run(submit_cmd)
        if code != 0:
            raise RuntimeError(f"batch submit-job failed:\n{err or out}")

        job_id = json.loads(out)["jobId"]
        return {"run_id": run_id, "batch_job_id": job_id, "s3_run": s3_run}

    def status(self, job_id: str) -> dict:
        code, out, err = _run(self._aws() + ["batch", "describe-jobs", "--jobs", job_id])
        if code != 0:
            raise RuntimeError(f"describe-jobs failed:\n{err or out}")
        data = json.loads(out)["jobs"][0]
        return {"status": data.get("status"), "statusReason": data.get("statusReason", "")}

    def logs_hint(self, job_id: str) -> str:
        # CloudWatch log stream is only known if job definition uses awslogs driver.
        return (
            "If your job definition uses CloudWatch logs (awslogs), open the AWS Console:\n"
            f"Batch job: {job_id}\n"
            "Or implement `aws logs tail` once you know logGroupName/logStreamName."
        )