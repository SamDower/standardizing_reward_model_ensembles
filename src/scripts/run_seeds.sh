#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --job-name="starc"
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

echo $1

config_file=$1
exp_name=$2
# Set the number of runs (N)
N=$3

# Function to generate a random seed
generate_seed() {
  echo $((RANDOM % 10000))
}

cd ~/StarcGridWorld

# Run script.py N times in parallel
for ((i = 1; i <= N; i++)); do
  seed=$(generate_seed)
  echo "Running script.py with seed $seed (Run $i)"
  python3 grid_world/run_preference_simulator.py --seed $seed --config_file $config_file &  # Execute in the background
done

# Wait for all background processes to finish
wait

echo "All runs completed!"

#python3 grid_world/plot_aggregated_seeds.py --exp_name $exp_name
