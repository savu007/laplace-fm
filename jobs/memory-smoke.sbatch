#!/bin/bash
#SBATCH --job-name=lapflow-memory
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=lapflow-memory-%j.out
#SBATCH --error=lapflow-memory-%j.err
set -euo pipefail
cd /global/u1/s/savu/projects/laplace-fm
module load pytorch
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python -u -B -m lapflow.train smoke --device cuda:0 --threads 4
