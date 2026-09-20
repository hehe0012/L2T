#!/bin/sh
#SBATCH --job-name=guian-l2t-arith-pre
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu

cd /home/guian/L2T
mkdir -p logs/slurm logs checkpoints

echo "job=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
date
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/home/guian/L2T/.venv/bin/python scripts/pretrain_arithmetic_state_embedding.py \
  --epochs 500 \
  --batch-size 512 \
  --lr 3e-3 \
  --weight-decay 1e-2 \
  --min-lr-ratio 0.005 \
  --mixed-precision bf16 \
  --state-dim 8 \
  --max-exponent 12 \
  --device cuda \
  --checkpoint-out checkpoints/arithmetic_state_embedding.pt \
  --out logs/arithmetic_state_embedding.json
