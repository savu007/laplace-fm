#!/bin/bash
#SBATCH --job-name=lapflow-repair
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=19:00:00
#SBATCH --output=lapflow-repair-%j.out
#SBATCH --error=lapflow-repair-%j.err
set -euo pipefail
cd /global/u1/s/savu/projects/laplace-fm
module load pytorch
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
run_dir="${RUN_DIR:-/pscratch/sd/s/savu/laplace-fm/full-stack-v2}"
config="${CONFIG:-configs/dns384.json}"
export MPLCONFIGDIR="$run_dir/.matplotlib"
# Reuse only verified preparation; preserve the failed run and its evidence.
prepared_from="${PREPARED_FROM:-/pscratch/sd/s/savu/laplace-fm/full-stack-v1}"
# This executes every stage sequentially. Rerunning this unchanged job resumes it.
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python -u -B -m lapflow.pipeline --config "$config" --run "$run_dir" \
  --prepared-from "$prepared_from" --device cuda:0 --threads 8 --samples "${SAMPLES:-8}" --seed-base "${SEED_BASE:-50000}"
