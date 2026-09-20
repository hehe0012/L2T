#!/bin/sh
#SBATCH --job-name=guian-l2t-state-vd
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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/home/guian/L2T/.venv/bin/python scripts/pretrain_gridworld_state_autoencoder.py \
  --seed 0 \
  --pretrain-mode vae \
  --epochs 500 \
  --batch-size 512 \
  --lr 5e-3 \
  --weight-decay 1e-2 \
  --min-lr-ratio 0.005 \
  --grad-clip 1.0 \
  --mixed-precision bf16 \
  --hidden-dim 32 \
  --ff-dim 128 \
  --action-dim 16 \
  --codebook-size 6 \
  --state-path cnn \
  --state-dropout 0.1 \
  --vae-beta 1e-5 \
  --displacement-loss-weight 1.0 \
  --device cuda \
  --out checkpoints/gridworld_cnn_vae_displacement_state_autoencoder.pt \
  --log-out logs/gridworld_cnn_vae_displacement_state_autoencoder.json
