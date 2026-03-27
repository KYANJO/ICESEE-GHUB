#!/bin/bash
# ============================================================
# SLURM Job Script — ICESEE + ISSM Data Assimilation
# Target: Generic SLURM cluster (PMIx, Apptainer)
# ============================================================
#
# Usage:
#   sbatch run_icesee_issm.slurm
#
# Adjust the #SBATCH directives below for your cluster's
# partition names, memory limits, and time policies.
# ============================================================

# ── Job metadata ─────────────────────────────────────────────
#SBATCH --job-name=icesee-issm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@institution.edu

# ── Resources ─────────────────────────────────────────────────
# Nens ensemble members, each on its own task.
# Set --ntasks == NENS below.
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # == NENS
#SBATCH --cpus-per-task=4           # == model_nprocs
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=compute         # ← change to your cluster's partition

# ── Apptainer image and bind paths ───────────────────────────
SIF=/scratch/containers/icesee-applications-v1.0.sif   # ← path on shared filesystem

# Bind mounts: host:container (comma-separated)
BINDS="\
/path/to/ICESEE:/opt/ICESEE,\
$(pwd)/examples:/opt/ISSM/examples,\
$(pwd)/execution:/opt/ISSM/execution"

# ── Ensemble / model parameters ──────────────────────────────
NENS=4          # number of ensemble members  (== --ntasks)
MODEL_NPROCS=1  # MPI processes per ISSM instance
FILTER=EnKF     # filter type (EnKF, etc.)

# ── Environment ───────────────────────────────────────────────
module purge
module load apptainer        # adjust module name for your cluster
# module load openmpi/4.1.6  # load if MPI is not inside the container

# Ensure log and output dirs exist
mkdir -p logs execution results _modelrun_datasets

echo "========================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node(s):    $SLURM_NODELIST"
echo "Tasks:      $SLURM_NTASKS"
echo "CPUs/task:  $SLURM_CPUS_PER_TASK"
echo "NENS:       $NENS"
echo "Started:    $(date)"
echo "========================================"

# ── Launch ────────────────────────────────────────────────────
srun --mpi=pmix \
     -n ${NENS} \
     apptainer exec \
       --bind ${BINDS} \
       --env MATLAB_PREFDIR=/tmp/matlab_prefs \
       --env TMPDIR=/tmp \
       ${SIF} \
       with-issm python run_da_issm.py \
         --Nens=${NENS} \
         --model_nprocs=${MODEL_NPROCS} \
         --filter_type=${FILTER} \
         --verbose

EXIT_CODE=$?

echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
