#!/bin/sh
#SBATCH --job-name=guian-l2t-gt-exec
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu
cd /home/guian/L2T

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WEIGHT_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
case "${WEIGHT_INDEX}" in
  0) WEIGHT="0.1" ;;
  1) WEIGHT="0.3" ;;
  2) WEIGHT="1.0" ;;
  *) echo "invalid weight index=${WEIGHT_INDEX}" >&2; exit 2 ;;
esac
case "${SEED_INDEX}" in
  0) SEED="0" ;;
  1) SEED="1" ;;
  2) SEED="2" ;;
  *) echo "invalid seed index=${SEED_INDEX}" >&2; exit 2 ;;
esac

BASE="gridworld_neo_s_alpha0.33_seed${SEED}_samples64_gw${WEIGHT}_grounding_sum_weight_sweep"
CHECKPOINT="checkpoints/gridworld_neo_s_grounding_sum_weight_sweep/${BASE}.pt"
OUT="logs/gridworld_gt_program_executor/${BASE}.json"
mkdir -p logs/gridworld_gt_program_executor

echo "job=${SLURM_JOB_ID:-unknown} task=${TASK_ID} checkpoint=${CHECKPOINT}"
date

/home/guian/L2T/.venv/bin/python scripts/evaluate_gridworld_gt_program_executor.py \
  --checkpoint "${CHECKPOINT}" \
  --alpha 0.33 \
  --id-size 10000 \
  --comp-size 10000 \
  --length-size 20000 \
  --batch-size 512 \
  --device cuda \
  --out "${OUT}"
