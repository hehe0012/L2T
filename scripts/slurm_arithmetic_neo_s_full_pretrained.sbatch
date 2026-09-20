#!/bin/sh
#SBATCH --job-name=guian-l2t-arith-neo-s
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-8%1
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu

cd /home/guian/L2T
RUN_TAG="${RUN_TAG:-paper_corrected}"
OUT_DIR="logs/arithmetic_neo_s_${RUN_TAG}"
CKPT_DIR="checkpoints/arithmetic_neo_s_${RUN_TAG}"
mkdir -p "${OUT_DIR}" logs/slurm "${CKPT_DIR}"

STATE_EMBEDDING="checkpoints/arithmetic_state_embedding.pt"
if [ ! -f "${STATE_EMBEDDING}" ]; then
  echo "missing arithmetic state embedding checkpoint: ${STATE_EMBEDDING}" >&2
  exit 2
fi

echo "job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "host=$(hostname)"
date
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_INDEX=$((TASK_ID % 3))
ALPHA_INDEX=$((TASK_ID / 3))

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

case "$ALPHA" in
  "1.00") COMP_EVAL_SIZE="0" ;;
  *) COMP_EVAL_SIZE="" ;;
esac

MODEL="neo_s"
SAMPLES="1024"
RUN_NAME="arithmetic_${MODEL}_alpha${ALPHA}_seed${SEED}_samples${SAMPLES}_${RUN_TAG}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

COMP_ARGS=""
if [ -n "${COMP_EVAL_SIZE}" ]; then
  COMP_ARGS="--comp-eval-size ${COMP_EVAL_SIZE}"
fi

# shellcheck disable=SC2086
/home/guian/L2T/.venv/bin/python scripts/run_arithmetic.py \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  ${COMP_ARGS} \
  --epochs 200 \
  --batch-size 512 \
  --eval-batch-size 64 \
  --lr 1.5e-3 \
  --weight-decay 1e-2 \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1 \
  --grad-clip 1.0 \
  --gumbel-tau-start 0.3 \
  --gumbel-tau-end 0.05 \
  --mixed-precision bf16 \
  --state-dim 8 \
  --max-exponent 12 \
  --action-dim 4 \
  --policy-d-model 64 \
  --policy-ff-dim 256 \
  --policy-heads 4 \
  --policy-layers 6 \
  --transition-d-model 32 \
  --transition-ff-dim 128 \
  --transition-heads 2 \
  --transition-layers 4 \
  --max-steps 6 \
  --train-rollout-steps 3 \
  --eval-rollout-steps 3 \
  --length-eval-rollout-steps 6 \
  --neo-s-samples "${SAMPLES}" \
  --mdl-start 1.01 \
  --mdl-end 0.99 \
  --train-query-loss-weight 0.0 \
  --neo-s-selection majority_exact \
  --commitment-cost 0.25 \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight 0.5 \
  --state-checkpoint "${STATE_EMBEDDING}" \
  --freeze-state \
  --device cuda \
  --out "${OUT_DIR}/${RUN_NAME}.json" \
  --checkpoint-out "${CKPT_DIR}/${RUN_NAME}.pt"
