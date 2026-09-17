#!/bin/sh
#SBATCH --job-name=guian-l2t-neo-s-pre
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-8%1
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu

cd /home/guian/L2T
mkdir -p logs/gridworld_neo_s_full_pretrained logs/slurm checkpoints/gridworld_neo_s_full_pretrained

STATE_AE="checkpoints/gridworld_state_autoencoder.pt"
if [ ! -f "${STATE_AE}" ]; then
  echo "missing state autoencoder checkpoint: ${STATE_AE}" >&2
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

case "$ALPHA" in
  "1.00") COMP_EVAL_SIZE="0" ;;
  *) COMP_EVAL_SIZE="10000" ;;
esac

case "$SEED_INDEX" in
  0) SEED="0" ;;
  1) SEED="1" ;;
  2) SEED="2" ;;
  *) echo "invalid SEED_INDEX=${SEED_INDEX}" >&2; exit 2 ;;
esac

MODEL="neo_s"
SAMPLES="64"
RUN_NAME="gridworld_${MODEL}_alpha${ALPHA}_seed${SEED}_samples${SAMPLES}_pretrained"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --alpha "${ALPHA}" \
  --train-size 100000 \
  --id-eval-size 10000 \
  --comp-eval-size "${COMP_EVAL_SIZE}" \
  --length-eval-size 20000 \
  --steps 20000 \
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
  --max-steps 10 \
  --train-rollout-steps 4 \
  --eval-rollout-steps 4 \
  --length-eval-rollout-steps 10 \
  --commitment-cost 0.25 \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight 0.1 \
  --state-autoencoder-checkpoint "${STATE_AE}" \
  --freeze-state-autoencoder \
  --neo-s-samples "${SAMPLES}" \
  --device cuda \
  --out "logs/gridworld_neo_s_full_pretrained/${RUN_NAME}.json" \
  --checkpoint-out "checkpoints/gridworld_neo_s_full_pretrained/${RUN_NAME}.pt"
