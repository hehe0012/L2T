#!/bin/sh
# Grounding-loss weight sweep after switching grounding aggregation from mean_k to sum_k.
# Array layout: weight index = task / 3, seed index = task % 3.
#SBATCH --job-name=guian-l2t-grid-ground
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu

cd /home/guian/L2T

RUN_TAG="grounding_sum_weight_sweep"
OUT_DIR="logs/gridworld_neo_s_${RUN_TAG}"
CKPT_DIR="checkpoints/gridworld_neo_s_${RUN_TAG}"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WEIGHT_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))

case "${WEIGHT_INDEX}" in
  0) GROUNDING_WEIGHT="0.1" ;;
  1) GROUNDING_WEIGHT="0.3" ;;
  2) GROUNDING_WEIGHT="1.0" ;;
  *) echo "invalid weight index=${WEIGHT_INDEX}" >&2; exit 2 ;;
esac

case "${SEED_INDEX}" in
  0) SEED="0" ;;
  1) SEED="1" ;;
  2) SEED="2" ;;
  *) echo "invalid seed index=${SEED_INDEX}" >&2; exit 2 ;;
esac

STATE_AE="checkpoints/gridworld_cnn_vae_state_autoencoder.pt"
if [ ! -f "${STATE_AE}" ]; then
  echo "missing state autoencoder checkpoint: ${STATE_AE}" >&2
  exit 2
fi

ALPHA="0.33"
MODEL="neo_s"
SAMPLES="64"
MDL_WEIGHT="1.0526315789473684"
RUN_NAME="gridworld_${MODEL}_alpha${ALPHA}_seed${SEED}_samples${SAMPLES}_gw${GROUNDING_WEIGHT}_${RUN_TAG}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job=${SLURM_JOB_ID:-unknown} task=${TASK_ID} seed=${SEED} grounding_weight=${GROUNDING_WEIGHT}"
echo "host=$(hostname)"
date
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  --train-size 100000 \
  --id-eval-size 10000 \
  --comp-eval-size 10000 \
  --length-eval-size 20000 \
  --epochs 100 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-2 \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1 \
  --grad-clip 1.0 \
  --policy-lr-scale 0.25 \
  --transition-lr-scale 1.0 \
  --gumbel-tau-start 0.3 \
  --gumbel-tau-end 0.1 \
  --hidden-dim 32 \
  --ff-dim 128 \
  --action-dim 16 \
  --state-path cnn \
  --state-dropout 0.1 \
  --max-steps 10 \
  --train-rollout-steps 4 \
  --eval-rollout-steps 4 \
  --length-eval-rollout-steps 10 \
  --commitment-cost 0.25 \
  --split-mode paper \
  --mdl-score-mode multiplicative \
  --mdl-weight "${MDL_WEIGHT}" \
  --mdl-lambda 0.0 \
  --train-query-loss-weight 0.0 \
  --neo-s-selection majority_exact \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight "${GROUNDING_WEIGHT}" \
  --grounding-transition-only \
  --state-autoencoder-checkpoint "${STATE_AE}" \
  --freeze-state-autoencoder \
  --neo-s-samples "${SAMPLES}" \
  --validation-size 200 \
  --validation-interval 1 \
  --device cuda \
  --out "${OUT_DIR}/${RUN_NAME}.json" \
  --checkpoint-out "${CKPT_DIR}/${RUN_NAME}.pt"
