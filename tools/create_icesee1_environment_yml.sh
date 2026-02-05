#!/bin/bash -l

# Create the icesee1_environment.yml file used to create the icesee conda environments.
# Also see go.icesee1.
# Need to make sure the associated kernel is not selected in JupyterLab or JupyterNotebooks (202210) first to prevent nfs device busy errors.

# Usage:
# source ./create_icesee1_environment_yml.sh
# or
# source ./create_icesee1_environment_yml.sh 2>&1 | tee create_icesee1_environment_yml_log.txt | less --RAW-CONTROL-CHARS
# - will need to explicitly enter a carriage return or Shift+g when prompted with :

# Version number.
# Once the icesee tool is published, do not overwrite the published Python environment.
version_number="1"

# Choose MPI implementation for mpi4py + parallel HDF5 builds.
# mpich tends to be the safest default on shared systems.
# Allowed: mpich | openmpi
mpi_impl="${MPI_IMPL:-mpich}"

# conda uses the libmamba solver.
# See conda info user-agent: solver/libmamba

# -e: Environment only.  Act on the current shell.  Do not preserve.
# -r: Replace an already selected version without prompting.
# --- Select conda (GHUB vs non-GHUB) ---
if [[ -f /etc/environ.sh ]]; then
  . /etc/environ.sh
  conda_choice=anaconda-7
  echo "conda_choice: ${conda_choice}"
  use -e -r ${conda_choice}
else
  echo "/etc/environ.sh not found (non-GHUB). Using current conda from PATH."
fi

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found in PATH"; exit 2; }
echo "which conda: $(which conda)"

# Modern activation support
eval "$(conda shell.bash hook)"


echo "which jupyter: "$(which jupyter)
echo "which python: "$(which python)

environment=icesee${version_number}-dev
echo "environment: "${environment}

# conda_root=/data/groups/ghub/tools/icesee/
if [[ -f /etc/environ.sh ]]; then
  conda_root=/data/groups/ghub/tools/icesee
else
  # Non-GHUB machine: must be writable
  conda_root="${ICESEE_CONDA_ROOT:-${HOME}/.icesee_conda}"
fi
echo "conda_root: ${conda_root}"
mkdir -p "${conda_root}/envs" "${conda_root}/pkgs"

echo "conda_root: "${conda_root}

# Update ~/.condarc for the build.
# When the conda environment is not stored in the default location /envs,
# will also need envs_dirs to see the name of a named conda environment with conda env list or conda info --envs.
conda config --add envs_dirs ${conda_root}/envs
if [[ ! -w "${conda_root}/pkgs" ]]; then
  echo "ERROR: pkgs dir not writable: ${conda_root}/pkgs"
  exit 2
fi

conda config --add pkgs_dirs ${conda_root}/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

conda_list=./icesee${version_number}_conda_list.txt
conda_env_yml=./icesee${version_number}_environment.yml

start1=$(date +%s)
echo "Removing env "${environment}"..."
# -n,--name: Name of the environment, use if environment was created with --name
# -p,--prefix: Full path to environment location (i.e. prefix), use if environment was created with --prefix
conda remove --name ${environment} --all -y
# echo "removing directory "${conda_root}/envs/${environment}"..."
# rm -rf ${conda_root}/envs/${environment}
if [[ -f /etc/environ.sh ]]; then
  echo "removing directory ${conda_root}/envs/${environment}..."
  rm -rf "${conda_root}/envs/${environment}"
fi

end=$(date +%s)
echo "Env ${environment} removed. Elapsed time: $((($end-$start1)/60)) minutes"

start2=$(date +%s)
echo "Creating env "${environment}"..."
# -n,--name: Name of the environment
conda create --name ${environment} python=3.11 -y
end=$(date +%s)
echo "Env ${environment} created. Elapsed time: $((($end-$start2)/60)) minutes"

echo "Activating env "${environment}"..."
# Observation, conda activate returns Your shell has not been properly configured to use conda activate and
# conda init bash returns No action taken.
# source activate ${environment}
eval "$(conda shell.bash hook)"
conda activate ${environment}

echo "Env ${environment} activated"
echo "which conda: "$(which conda)
echo "which jupyter: "$(which jupyter)
echo "which python: "$(which python)

echo "Installing required python packages..."

# NOTE:
# - notebook<7 + ipywidgets<8 tends to be safest on older GHUB notebook stacks.
# - Critical: parallel HDF5 + mpi4py requires MPI + HDF5 built with MPI.
#   We enforce conda-forge MPI selection via mpi metapackage.
start3=$(date +%s)
echo "conda install..."
conda install \
  pip \
  numpy scipy \
  pyyaml psutil tqdm dask zarr "numcodecs<0.13" \
  pandas matplotlib \
  ipykernel "ipywidgets>=7.6.0,<8" "notebook<7" jupyterlab nbformat nbclient jupyter_client \
  -y
# MPI + HDF5 + mpi4py
conda install -y -c conda-forge \
  "mpi=1.0=${mpi_impl}" ${mpi_impl} mpi4py \
  "hdf5=*=mpi*" "h5py=*=mpi*" \
  -y

# Jupyter Book for building documentation
conda install -c conda-forge "jupyter-book=1.*" -y
end=$(date +%s)
echo "conda install elapsed time: $((($end-$start3)/60)) minutes"

echo "Sanity: check libmpi is present in the conda env..."
python - <<'PY'
import os, glob, sys
prefix = os.environ.get("CONDA_PREFIX", "")
print("CONDA_PREFIX:", prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libmpi.so*"))
print("libmpi candidates:", cands[:10])
if not cands:
    raise SystemExit("ERROR: libmpi.so not found in env. MPI runtime missing.")
PY

# pip-only packages (keep this section small; GHUB prefers most deps via conda)
start4=$(date +%s)
echo "pip install..."
#python -m pip install --upgrade pip
python -m pip install -U pip setuptools wheel

# Optional tools commonly used in ICESEE workflows:
python -m pip install \
  bigmpi4py \
  mpi-pytest \
  papermill \
  gstools \
  pqdm \
  rich \
  progress
python -m pip install "jax[cpu]"
end=$(date +%s)
echo "pip install elapsed time: $((($end-$start4)/60)) minutes"
echo "Required software packages installed. Elapsed time: $((($end-$start3)/60)) minutes"

# Quick sanity checks so the exported YAML represents a working MPI/HDF5 stack.
python - <<PY
import mpi4py
from mpi4py import MPI
import h5py
cfg = getattr(h5py, "get_config", lambda: None)()
print("MPI size:", MPI.COMM_WORLD.Get_size())
print("h5py mpi:", getattr(cfg, "mpi", None))
PY

rm -rf ${conda_list}
conda list > ${conda_list}

rm -rf ${conda_env_yml}
conda env export | grep -v "^name: " | grep -v "^prefix: " > ${conda_env_yml}

echo "Deactivating env "${environment}"..."
conda deactivate
echo "Env ${environment} deactivated"
echo "which jupyter: "$(which jupyter)
echo "jupyter --version: "$(jupyter --version)
echo "which python: "$(which python)

# Cleanup ~/.condarc
conda config --remove envs_dirs ${conda_root}/envs
conda config --remove pkgs_dirs ${conda_root}/pkgs

end=$(date +%s)
echo "Done. Total elasped time: $((($end-$start1)/60)) minutes"