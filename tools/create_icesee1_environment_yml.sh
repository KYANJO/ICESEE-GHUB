#!/bin/bash -l
# Create the icesee1_environment.yml file used to create the icesee environments.
# Also see go.icesee1.
# Make sure the kernel is NOT selected in JupyterLab/Notebook first (nfs device busy risk).

# Usage:
# source ./create_icesee1_environment_yml.sh
# or:
# source ./create_icesee1_environment_yml.sh 2>&1 | tee create_icesee1_environment_yml_log.txt | less --RAW-CONTROL-CHARS

version_number="1"

. /etc/environ.sh
conda_choice=anaconda-7
echo "conda_choice: ${conda_choice}"
use -e -r ${conda_choice}
echo "which conda: $(which conda)"
echo "which jupyter: $(which jupyter)"
echo "which python: $(which python)"

environment="icesee${version_number}-dev"
echo "environment: ${environment}"

conda_root=/data/groups/ghub/tools/icesee
echo "conda_root: ${conda_root}"

# Update ~/.condarc for the build (non-default envs/pkgs locations)
conda config --add envs_dirs ${conda_root}/envs
conda config --add pkgs_dirs ${conda_root}/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

conda_list=./icesee${version_number}_conda_list.txt
conda_env_yml=./icesee${version_number}_environment.yml

start1=$(date +%s)
echo "Removing env ${environment}..."
conda remove --name ${environment} --all -y || true
echo "removing directory ${conda_root}/envs/${environment}..."
rm -rf ${conda_root}/envs/${environment}
end=$(date +%s)
echo "Env ${environment} removed. Elapsed time: $((($end-$start1)/60)) minutes"

start2=$(date +%s)
echo "Creating base env ${environment}..."
conda create --name ${environment} python=3.11 pip -y
end=$(date +%s)
echo "Env ${environment} created. Elapsed time: $((($end-$start2)/60)) minutes"

echo "Activating env ${environment}..."
source activate ${environment}
echo "Env ${environment} activated"
echo "which python: $(which python)"

echo "Installing required packages..."

start3=$(date +%s)
echo "conda install..."
# NOTE:
# - notebook<7 + ipywidgets<8 tends to be safest on older GHUB notebook stacks.
# - If GHUB is already Notebook 7/JLab 4 everywhere, we can modernize to ipywidgets>=8.
conda install -y \
  numpy scipy h5py pyyaml psutil tqdm dask zarr "numcodecs<0.13" \
  matplotlib pandas \
  ipykernel "ipywidgets>=7.6,<8" "notebook<7" jupyterlab \
  nbformat nbclient jupyter_client \
  && true
end=$(date +%s)
echo "conda install elapsed time: $((($end-$start3)/60)) minutes"

start4=$(date +%s)
echo "pip install..."
python -m pip install -U pip setuptools wheel

# ICESEE extras tooling for the wrapper:
python -m pip install papermill

# Your pyproject has gstools, and jax/jaxlib; install safely here.
python -m pip install gstools

# JAX: CPU-safe default; comment out if GHUB doesn’t need it
python -m pip install "jax[cpu]"

end=$(date +%s)
echo "pip install elapsed time: $((($end-$start4)/60)) minutes"

rm -rf ${conda_list}
conda list > ${conda_list}

rm -rf ${conda_env_yml}
conda env export | grep -v "^name: " | grep -v "^prefix: " > ${conda_env_yml}
echo "Wrote ${conda_env_yml}"

echo "Deactivating env ${environment}..."
conda deactivate
echo "Env ${environment} deactivated"

# Cleanup ~/.condarc
conda config --remove envs_dirs ${conda_root}/envs
conda config --remove pkgs_dirs ${conda_root}/pkgs

end=$(date +%s)
echo "Done. Total elapsed time: $((($end-$start1)/60)) minutes"