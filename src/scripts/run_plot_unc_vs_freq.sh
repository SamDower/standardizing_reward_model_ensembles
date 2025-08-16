#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="act-pm"
#SBATCH --output=/users/ug21sdd/logs/slurm-%j.out
#SBATCH --error=/users/ug21sdd/logs/slurm-%j.err
#SBATCH --partition="msc"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/

export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/StarcGridWorld/grid_world/environment_cloud.yml
source /scratch-ssd/oatml/miniconda3/bin/activate act-pm

echo $TMPDIR

nvidia-smi

# huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

cd ~/StarcGridWorld

python3 grid_world/plot_unc_vs_freq.py

echo "Finished!"
