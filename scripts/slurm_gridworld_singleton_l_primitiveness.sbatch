#!/bin/sh
#SBATCH --job-name=guian-l2t-prime-swap
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu
cd /home/guian/L2T
mkdir -p logs/gridworld_primitiveness_alignment

BASE="gridworld_neo_alpha0.33_seed0_evalneos64_singleton_l"
/home/guian/L2T/.venv/bin/python scripts/evaluate_gridworld_primitiveness_alignment.py \
  --checkpoint "checkpoints/gridworld_neo_singleton_l_alpha033/${BASE}.pt" \
  --device cpu \
  --out "logs/gridworld_primitiveness_alignment/${BASE}.json"
