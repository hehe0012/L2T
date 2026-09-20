#!/bin/sh
#SBATCH --job-name=guian-l2t-oracle-exec
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu

cd /home/guian/L2T
mkdir -p logs/gridworld_oracle_executor checkpoints logs/slurm

echo "job=${SLURM_JOB_ID:-unknown}"
hostname
date
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

.venv/bin/python scripts/test_gridworld_oracle_executor.py \
  --seed 0 \
  --state-path cnn \
  --state-dropout 0.1 \
  --hidden-dim 32 \
  --ff-dim 128 \
  --action-dim 16 \
  --steps 20000 \
  --batch-size 400 \
  --lr 1e-3 \
  --state-autoencoder-checkpoint checkpoints/gridworld_cnn_vae_state_autoencoder.pt \
  --freeze-state-autoencoder \
  --eval-states 100 \
  --max-programs-per-length 512 \
  --device cuda \
  --out logs/gridworld_oracle_executor/cnn_vae_frozen_seed0.json \
  --checkpoint-out checkpoints/gridworld_oracle_executor_cnn_vae_frozen_seed0.pt
