#!/bin/sh
#SBATCH --job-name=guian-l2t-grid-eval
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu
cd /home/guian/L2T

BASE="gridworld_neo_s_alpha0.33_seed0_samples64_alpha033_seed0_100ep"
CHECKPOINT="checkpoints/gridworld_neo_s_full_pretrained/${BASE}.pt"
OUT_DIR="logs/gridworld_eval_alpha033_seed0_100ep"
mkdir -p "${OUT_DIR}" logs/slurm

test -f "${CHECKPOINT}"

/home/guian/L2T/.venv/bin/python scripts/evaluate_gridworld_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --neo-s-samples 64 \
  --device cuda \
  --out "${OUT_DIR}/neo_vs_neo_s_b64.json"

/home/guian/L2T/.venv/bin/python scripts/evaluate_gridworld_alignment_lengths.py \
  --checkpoint "${CHECKPOINT}" \
  --max-length 3 \
  --device cuda \
  --out "${OUT_DIR}/alignment_lengths_1_3.json"
