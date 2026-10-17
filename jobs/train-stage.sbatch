#!/bin/bash
#SBATCH --job-name=lapflow-stage
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=lapflow-stage-%j.out
#SBATCH --error=lapflow-stage-%j.err
set -euo pipefail
cd /global/u1/s/savu/projects/laplace-fm
module load pytorch
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
: "${STAGE:?Set STAGE to prepare, autoencoder, reconstruct, cache, flow, or sample}"
case "$STAGE" in prepare|autoencoder|reconstruct|cache|flow|sample) ;; *) exit 2 ;; esac
run_dir="${RUN_DIR:-/pscratch/sd/s/savu/laplace-fm/dns384-v1}"
config="${CONFIG:-configs/dns384.json}"
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python -u -B -m lapflow.train "$STAGE" --config "$config" \
  --run "$run_dir" --device cuda:0 --threads 8 "${extra[@]}"
