# ICESEE + ISSM — Multi-Cluster Deployment Guide

Deployment guide for running the ICESEE + ISSM coupled workflow on:
- **AWS** (EC2 / ParallelCluster)
- **Other HPC clusters** (generic SLURM + PMIx, Apptainer)

The container includes an inbuilt MATLAB runtime — no host MATLAB bind-mount required.

---

## Table of Contents

- [ICESEE + ISSM — Multi-Cluster Deployment Guide](#icesee--issm--multi-cluster-deployment-guide)
  - [Table of Contents](#table-of-contents)
  - [Repository Layout](#repository-layout)
  - [Container Strategy](#container-strategy)
  - [Building the Container](#building-the-container)
    - [Docker (cloud)](#docker-cloud)
    - [Apptainer (HPC)](#apptainer-hpc)
  - [AWS Deployment](#aws-deployment)
    - [AWS Prerequisites](#aws-prerequisites)
    - [Cluster Setup](#cluster-setup)
    - [Uploading the Container](#uploading-the-container)
    - [Running a Job on AWS](#running-a-job-on-aws)
    - [MATLAB Licensing on AWS](#matlab-licensing-on-aws)
  - [Generic HPC Deployment](#generic-hpc-deployment)
    - [HPC Prerequisites](#hpc-prerequisites)
    - [Running a Job on HPC](#running-a-job-on-hpc)
    - [MPI Compatibility Notes](#mpi-compatibility-notes)
  - [Bind Mounts Reference](#bind-mounts-reference)
  - [Environment Variables](#environment-variables)
  - [Troubleshooting](#troubleshooting)

---

## Repository Layout

After cloning, the relevant deployment files are:

```
.
├── Dockerfile                        # Docker build (cloud)
├── combined-env-inbuilt-matlab.def   # Apptainer build (HPC)
├── cluster-config.yaml               # AWS ParallelCluster 3.x config
├── run_icesee_issm.slurm             # Generic SLURM job script
├── run_da_issm.py                    # Main DA script
└── params.yaml                       # Model/filter configuration
```

---

## Container Strategy

| Environment | Format | Image | MATLAB | MPI |
|-------------|--------|-------|--------|-----|
| AWS EC2 / cloud | Docker → Apptainer `.sif` | `bkyanjo/icesee-applications:v1.0` | Inbuilt | OpenMPI 5.0.10 + PMIx, inside container |
| HPC (non-PACE) | Apptainer `.sif` | `bkyanjo/icesee-applications:v1.0` | Inbuilt | OpenMPI 5.0.10 + PMIx, inside container |

**Published image:** `docker.io/bkyanjo/icesee-applications:v1.0`
- Index digest: `sha256:59799aa610636dcec4fd974b2f042178a0ca2648e951f62798ea28493f20eff8`
- Manifest: `sha256:27d54a876bbc0455d80cb711d8ed05adda0acfaa31d0436d428fbc3fcfd0214a`

Since MATLAB is embedded, there is no host bind-mount requirement for MATLAB on any target cluster.

---

## Building the Container

The image is already built and published to Docker Hub. No local build step is required.

### Docker (cloud)

```bash
# Pull the published image
docker pull bkyanjo/icesee-applications:v1.0

# Verify (optional — image has already been tested)
docker run --rm bkyanjo/icesee-applications:v1.0 with-firedrake python -c "import firedrake; print('firedrake ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-icepack  python -c "import icepack;    print('icepack ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-icesee   python -c "import ICESEE;     print('ICESEE ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-issm bash -c 'test -f /opt/ISSM/etc/environment.sh && echo "issm wrapper ok"'
```

### Apptainer (HPC)

On any cluster with Apptainer, convert the Docker image to a `.sif` directly from Docker Hub — no intermediate Docker install needed:

```bash
# Pull and convert in one step
apptainer pull icesee-applications-v1.0.sif docker://bkyanjo/icesee-applications:v1.0

# Pin to the exact manifest digest for reproducibility
apptainer pull icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications@sha256:27d54a876bbc0455d80cb711d8ed05adda0acfaa31d0436d428fbc3fcfd0214a
```

> **Tip:** Run the pull once on a login node, store the `.sif` on `/scratch` or a shared filesystem, and reference that path in all job scripts. Compute nodes never need internet access.

---

## AWS Deployment

### AWS Prerequisites

- AWS CLI v2 installed and configured (`aws configure`)
- `pcluster` CLI installed: `pip install aws-parallelcluster`
- An existing VPC, public subnet, and EC2 key pair
- IAM permissions for EC2, S3, FSx, and CloudFormation

### Cluster Setup

1. Edit `cluster-config.yaml` and fill in the placeholders:

   | Placeholder | Replace with |
   |-------------|-------------|
   | `subnet-XXXXXXXXXXXXXXXXX` | Your subnet ID |
   | `your-keypair-name` | Your EC2 key pair name |
   | `YOUR-BUCKET` | Your S3 bucket name |
   | `us-east-1` | Your preferred region |

2. Create the cluster:

   ```bash
   pcluster create-cluster \
     --cluster-name icesee-issm \
     --cluster-configuration cluster-config.yaml
   ```

3. Monitor creation (takes ~10–15 min):

   ```bash
   pcluster describe-cluster --cluster-name icesee-issm
   ```

4. SSH into the head node:

   ```bash
   pcluster ssh --cluster-name icesee-issm -i ~/.ssh/your-key.pem
   ```

### Uploading the Container

Pull the image directly from Docker Hub onto the shared FSx filesystem — no local transfer needed:

```bash
# SSH into the head node, then:
mkdir -p /scratch/containers
apptainer pull /scratch/containers/icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications:v1.0

# Or pin to the exact digest for strict reproducibility
apptainer pull /scratch/containers/icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications@sha256:27d54a876bbc0455d80cb711d8ed05adda0acfaa31d0436d428fbc3fcfd0214a
```

> The head node needs outbound internet access (TCP 443) to reach Docker Hub. Compute nodes only need access to FSx — they never pull from the internet directly.

### Running a Job on AWS

Edit `run_icesee_issm.slurm` — only two lines need changing:

```bash
# Point to the pulled .sif on FSx
SIF=/scratch/containers/icesee-applications-v1.0.sif

# Bind mounts to FSx paths
BINDS="/scratch/ICESEE:/opt/ICESEE,/scratch/runs/${SLURM_JOB_ID}/examples:/opt/ISSM/examples,/scratch/runs/${SLURM_JOB_ID}/execution:/opt/ISSM/execution"
```

Submit:

```bash
# Use the 'compute' queue (on-demand EFA nodes)
sbatch --partition=compute run_icesee_issm.slurm

# Or the spot queue for dev runs
sbatch --partition=spot run_icesee_issm.slurm
```

### MATLAB Licensing on AWS

The container image has an inbuilt MATLAB installation but still requires a valid license at runtime. The Dockerfile bakes in a default of `MLM_LICENSE_FILE=1711@matlablic.ecs.gatech.edu` — **that license server is on the Georgia Tech network and will not be reachable from an AWS VPC.** You must override it before running on AWS.

Override at runtime without rebuilding:
```bash
# In run_icesee_issm.slurm, add:
export MLM_LICENSE_FILE=27000@your-license-server

# Or pass directly via apptainer --env:
apptainer exec --env MLM_LICENSE_FILE=27000@your-license-server ...

# Or via Docker:
docker run --rm -e MLM_LICENSE_FILE=27000@your-license-server icesee-issm:latest with-issm matlab -batch "issmversion"
```

**Option A — MathWorks Network License Manager (recommended)**
Deploy a license server EC2 instance inside the same VPC. Set in `run_icesee_issm.slurm`:
```bash
export MLM_LICENSE_FILE=27000@license-server.internal
```

**Option B — License file**
Bind-mount a `license.lic` file into the container:
```bash
# Apptainer
--bind /scratch/matlab/license.lic:/opt/matlab/licenses/license.lic

# Docker
-v /path/to/license.lic:/opt/matlab/licenses/license.lic
```

**Option C — MathWorks hosted licensing**
Set the `MHLM_CONTEXT` variable; requires outbound internet access from compute nodes.

---

## Generic HPC Deployment

### HPC Prerequisites

- Apptainer (≥ 1.0) installed on all compute nodes — verify with `apptainer --version`
- SLURM with PMIx support — verify with `srun --mpi=list | grep pmix`
- A shared filesystem accessible from all nodes (Lustre, GPFS, NFS, etc.)

### Running a Job on HPC

1. Copy `run_icesee_issm.slurm` to your working directory on the cluster.

2. Edit the top section of the script:

   ```bash
   SIF=/shared/containers/combined-env-inbuilt-matlab.sif  # path on shared FS
   BINDS="/shared/ICESEE:/opt/ICESEE,$(pwd)/examples:/opt/ISSM/examples,$(pwd)/execution:/opt/ISSM/execution"
   NENS=4
   MODEL_NPROCS=1
   ```

3. Adjust `#SBATCH` directives for your cluster's partition name, time limits, and memory policy.

4. Submit:

   ```bash
   sbatch run_icesee_issm.slurm
   ```

5. Monitor:

   ```bash
   squeue -u $USER
   tail -f logs/icesee-issm_<JOBID>.out
   ```

### MPI Compatibility Notes

The container's OpenMPI is built with `--with-pmix`, matching SLURM's `--mpi=pmix` launcher. If your cluster uses a different MPI interface:

| Cluster MPI interface | Change in SLURM script |
|-----------------------|------------------------|
| PMIx (default) | `srun --mpi=pmix` ✓ |
| PMI2 | `srun --mpi=pmi2` |
| UCX/OMPI | May need `--env OMPI_MCA_btl=^openib` to suppress IB warnings |

If the cluster's system MPI must be used instead of the container's, bind-mount it:

```bash
apptainer exec \
  --bind /usr/lib/x86_64-linux-gnu/openmpi:/opt/openmpi-host \
  --env MPI_DIR=/opt/openmpi-host \
  ...
```

---

## Bind Mounts Reference

| Host Path | Container Path | Required | Purpose |
|-----------|---------------|----------|---------|
| `/path/to/ICESEE` | `/opt/ICESEE` | Dev only | Live ICESEE source (skip in production) |
| `./examples` | `/opt/ISSM/examples` | Yes | ISSM example input files |
| `./execution` | `/opt/ISSM/execution` | Yes | ISSM runtime working dir and outputs |
| `./results` | `/workspace/results` | Optional | Output `.h5` files |

> In production, `ICESEE` is baked into the container. The bind-mount is only needed during active development.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATLABROOT` | `/opt/matlab/R2025b` | Set inside container via `with-issm` |
| `ISSM_DIR` | `/opt/ISSM` | ISSM installation path |
| `ICESEE_DIR` | `/opt/ICESEE` | ICESEE source path |
| `MATLAB_PREFDIR` | `/tmp/matlab_prefs` | MATLAB preferences dir (must be writable) |
| `TMPDIR` | `/tmp` | Temp dir for MATLAB scratch files |
| `MLM_LICENSE_FILE` | _(unset)_ | Point to license server: `27000@host` |

---

## Troubleshooting

**`Unable to launch MVM server` / `Unexpected Server Shutdown`**
MATLAB's service host crashed, usually due to a missing writable temp dir or missing libraries.
```bash
# Ensure these are set
--env MATLAB_PREFDIR=/tmp/matlab_prefs
--env TMPDIR=/tmp

# Check for missing shared libraries inside the container
ldd /opt/matlab/R2025b/bin/glnxa64/matlab | grep "not found"
```

**`srun: error: PMIx is not available`**
Your cluster's SLURM was not compiled with PMIx. Try `--mpi=pmi2` instead.

**Container not found on compute nodes**
The `.sif` must be on a shared filesystem accessible from all nodes — not on local `/home` if that is not shared. Use `/scratch` or equivalent.

**`with-issm: command not found`**
The entrypoint wrapper is only present inside the container. Make sure you are running via `apptainer exec ... with-issm ...` and not calling `with-issm` directly on the host.

**MATLAB license errors**
Verify `MLM_LICENSE_FILE` is set and the license server is reachable from compute nodes:
```bash
ping license-server.internal
telnet license-server.internal 27000
```

**Out of memory on compute nodes**
Increase `--mem-per-cpu` in the SLURM script, or reduce `--model_nprocs` to lower per-task memory usage.