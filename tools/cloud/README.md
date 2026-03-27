# ICESEE Applications Container

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-bkyanjo%2Ficesee--applications-blue?logo=docker)](https://hub.docker.com/repository/docker/bkyanjo/icesee-applications)
[![Image Version](https://img.shields.io/badge/version-v1.0-green)](https://hub.docker.com/repository/docker/bkyanjo/icesee-applications/tags/v1.0)

Containerized runtime for the [ICESEE](https://github.com/ICESEE-project/ICESEE) data assimilation library coupled with [ISSM](https://issm.jpl.nasa.gov/), [Firedrake](https://www.firedrakeproject.org/), and [Icepack](https://icepack.github.io/). Designed for reproducible execution across HPC clusters and cloud environments.

---

## What's in the Container

| Component | Description |
|-----------|-------------|
| **ICESEE** | Ensemble Kalman filter (EnKF) data assimilation library for ice sheet models |
| **ISSM** | Ice Sheet System Model — forward model driven by MATLAB |
| **Firedrake** | Automated finite element system (FEniCS-based) |
| **Icepack** | Ice sheet and glacier flow modeling library built on Firedrake |
| **MATLAB** | Inbuilt runtime — no host MATLAB installation or bind-mount required |
| **OpenMPI 5.0.10** | MPI implementation built with PMIx, used for ensemble parallelism |
| **HDF5 + NetCDF** | I/O libraries for model input/output |

---

## Quick Start

### On any machine with Docker

```bash
# Pull the image
docker pull bkyanjo/icesee-applications:v1.0

# Launch an interactive shell with the ISSM + ICESEE environment
docker run --rm -it bkyanjo/icesee-applications:v1.0 with-issm

# Run a parallel ICESEE + ISSM data assimilation experiment
docker run --rm \
  -v $(pwd)/examples:/opt/ISSM/examples \
  -v $(pwd)/execution:/opt/ISSM/execution \
  -e MLM_LICENSE_FILE=27000@your-license-server \
  bkyanjo/icesee-applications:v1.0 \
  with-issm mpirun -np 4 python run_da_issm.py --Nens=4 --model_nprocs=1 --verbose
```

### On an HPC cluster with Apptainer

```bash
# Convert the Docker image to a .sif (run once on the login node)
apptainer pull icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications:v1.0

# Interactive shell
apptainer exec icesee-applications-v1.0.sif with-issm

# Submit a SLURM job
sbatch run_icesee_issm.slurm
```

---

## Published Image

| Property | Value |
|----------|-------|
| Registry | [Docker Hub](https://hub.docker.com/repository/docker/bkyanjo/icesee-applications) |
| Image | `docker.io/bkyanjo/icesee-applications:v1.0` |
| Index digest | `sha256:59799aa610636dcec4fd974b2f042178a0ca2648e951f62798ea28493f20eff8` |
| Manifest | `sha256:27d54a876bbc0455d80cb711d8ed05adda0acfaa31d0436d428fbc3fcfd0214a` |
| Base image | `docker.io/bkyanjo/combined-lean:v1.0` |

To pin to an exact digest for strict reproducibility:

```bash
# Docker
docker pull bkyanjo/icesee-applications@sha256:59799aa610636dcec4fd974b2f042178a0ca2648e951f62798ea28493f20eff8

# Apptainer
apptainer pull icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications@sha256:27d54a876bbc0455d80cb711d8ed05adda0acfaa31d0436d428fbc3fcfd0214a
```

---

## Runtime Wrappers

The container exposes four environment wrappers. Each activates the correct Python virtual environment, sets library paths, and passes through any command you give it.

| Wrapper | Activates | Use for |
|---------|-----------|---------|
| `with-issm` | ICESEE venv + ISSM + OpenMPI/HDF5 | ICESEE–ISSM data assimilation runs |
| `with-firedrake` | ICESEE venv + Firedrake venv | Firedrake-based modeling |
| `with-icepack` | ICESEE venv + Icepack venv | Icepack glacier flow runs |
| `with-icesee` | ICESEE venv only | ICESEE standalone usage |

```bash
# Run a Python script under each wrapper
apptainer exec icesee-applications-v1.0.sif with-issm      python run_da_issm.py ...
apptainer exec icesee-applications-v1.0.sif with-firedrake python my_firedrake_script.py
apptainer exec icesee-applications-v1.0.sif with-icepack   python my_icepack_script.py
apptainer exec icesee-applications-v1.0.sif with-icesee    python my_icesee_script.py
```

---

## Repository Files

```
.
├── README.md                         # This file
├── Dockerfile                        # Docker image definition (builds on combined-lean:v1.0)
├── combined-env-inbuilt-matlab.def   # Apptainer definition file (equivalent, for HPC builds)
├── cluster-config.yaml               # AWS ParallelCluster 3.x configuration
├── run_icesee_issm.slurm             # Generic SLURM job script (PMIx, Apptainer)
├── DEPLOYMENT.md                     # Full deployment guide (AWS + generic HPC)
├── run_da_issm.py                    # Main ICESEE–ISSM data assimilation script
└── params.yaml                       # Model and filter configuration
```

---

## Deployment Targets

### AWS (EC2 / ParallelCluster)

Uses the Docker image directly. See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full walkthrough. The short version:

```bash
# 1. Create the cluster
pcluster create-cluster \
  --cluster-name icesee-issm \
  --cluster-configuration cluster-config.yaml

# 2. SSH in and pull the .sif onto shared storage
pcluster ssh --cluster-name icesee-issm -i ~/.ssh/your-key.pem
apptainer pull /scratch/containers/icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications:v1.0

# 3. Set your MATLAB license and submit
export MLM_LICENSE_FILE=27000@your-license-server
sbatch --partition=compute run_icesee_issm.slurm
```

### Generic HPC (SLURM + PMIx)

```bash
# Pull the image to shared storage (once)
apptainer pull /shared/containers/icesee-applications-v1.0.sif \
  docker://bkyanjo/icesee-applications:v1.0

# Edit the two required fields in run_icesee_issm.slurm:
#   SIF=/shared/containers/icesee-applications-v1.0.sif
#   #SBATCH --partition=<your-partition>

sbatch run_icesee_issm.slurm
```

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for MPI compatibility notes, bind mount reference, and troubleshooting.

---

## MATLAB Licensing

MATLAB is embedded in the container but still requires a license at runtime. The image defaults to `1711@matlablic.ecs.gatech.edu` (Georgia Tech network only). **Override this before running outside of PACE/Georgia Tech:**

```bash
# Apptainer
apptainer exec --env MLM_LICENSE_FILE=27000@your-server ...

# Docker
docker run -e MLM_LICENSE_FILE=27000@your-server ...

# SLURM script
export MLM_LICENSE_FILE=27000@your-server
```

Options: network license manager (recommended), license file bind-mount, or MathWorks hosted licensing. See the [MATLAB Licensing section in DEPLOYMENT.md](./DEPLOYMENT.md#matlab-licensing-on-aws) for details.

---

## Environment Variables

| Variable | Default in container | Description |
|----------|---------------------|-------------|
| `ISSM_DIR` | `/opt/ISSM` | ISSM installation path |
| `MATLABPATH` | `/opt/matlab` | MATLAB startup path (loads ISSM toolbox) |
| `MLM_LICENSE_FILE` | `1711@matlablic.ecs.gatech.edu` | MATLAB license — **override outside GT** |
| `OMP_NUM_THREADS` | `1` | Disable OpenMP threading (MPI handles parallelism) |
| `OPENBLAS_NUM_THREADS` | `1` | Disable BLAS threading |
| `HDF5_MPI` | `ON` | Enable parallel HDF5 I/O |
| `PMIX_MCA_psec` | `native` | PMIx security — required for SLURM compatibility |
| `ICESEE_CONTAINER` | `1` | Flag indicating containerized execution |
| `PYTHONPATH` | `/opt` | Makes ICESEE importable |

---

## Verifying the Image

```bash
docker run --rm bkyanjo/icesee-applications:v1.0 with-firedrake python -c "import firedrake; print('firedrake ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-icepack   python -c "import icepack;    print('icepack ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-icesee    python -c "import ICESEE;     print('ICESEE ok')"
docker run --rm bkyanjo/icesee-applications:v1.0 with-issm bash -c \
  'test -f /opt/ISSM/etc/environment.sh && echo "issm wrapper ok"'
```

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [ICESEE-project/ICESEE](https://github.com/ICESEE-project/ICESEE) | Core ICESEE data assimilation library |
| [ICESEE-project/ICESEE-Containers](https://github.com/ICESEE-project/ICESEE-Containers) | Container definitions and deployment configs |
| [ISSMteam/ISSM](https://github.com/ISSMteam/ISSM) | Ice Sheet System Model |