#!/bin/sh
#SBATCH --job-name=guian-l2t-gridworld
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-26%8
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu

cd /home/guian/L2T
mkdir -p logs/gridworld_full logs/slurm checkpoints/gridworld_full

echo "job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "host=$(hostname)"
date
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_INDEX=$((TASK_ID % 3))
ALPHA_INDEX=$(((TASK_ID / 3) % 3))
MODEL_INDEX=$((TASK_ID / 9))

case "$MODEL_INDEX" in
  0) MODEL="neo" ;;
  1) MODEL="disc_mono" ;;
  2) MODEL="neo_s" ;;
  *) echo "invalid MODEL_INDEX=${MODEL_INDEX}" >&2; exit 2 ;;
esac

case "$ALPHA_INDEX" in
  0) ALPHA="0.33" ;;
  1) ALPHA="0.66" ;;
  2) ALPHA="1.00" ;;
  *) echo "invalid ALPHA_INDEX=${ALPHA_INDEX}" >&2; exit 2 ;;
esac

case "$SEED_INDEX" in
  0) SEED="0" ;;
  1) SEED="1" ;;
  2) SEED="2" ;;
  *) echo "invalid SEED_INDEX=${SEED_INDEX}" >&2; exit 2 ;;
esac

RUN_NAME="gridworld_${MODEL}_alpha${ALPHA}_seed${SEED}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  --train-size 100000 \
  --id-eval-size 10000 \
  --comp-eval-size 10000 \
  --length-eval-size 20000 \
  --steps 20000 \
  --batch-size 128 \
  --lr 5e-4 \
  --neo-s-samples 16 \
  --device cuda \
  --out "logs/gridworld_full/${RUN_NAME}.json" \
  --checkpoint-out "checkpoints/gridworld_full/${RUN_NAME}.pt"
