#!/bin/sh
# Full paper-aligned GridWorld run with all five training-monitoring groups.
# Array index selects alpha; all runs use seed=42.
#SBATCH --job-name=guian-grid-test42
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu
cd /home/guian/L2T

OUT_DIR="logs/gridworld_test_plan_seed42"
CKPT_DIR="checkpoints/gridworld_test_plan_seed42"
STATE_AE="checkpoints/gridworld_state_autoencoder.pt"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" logs/slurm

if [ ! -f "${STATE_AE}" ]; then
  echo "missing state autoencoder checkpoint: ${STATE_AE}" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ALPHA="0.33"; EPOCHS="100"; WARMUP="0.10"; COMP_EVAL_SIZE="10000" ;;
  1) ALPHA="0.66"; EPOCHS="50"; WARMUP="0.05"; COMP_EVAL_SIZE="10000" ;;
  2) ALPHA="1.00"; EPOCHS="50"; WARMUP="0.05"; COMP_EVAL_SIZE="0" ;;
  *) echo "invalid array index=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

SEED="42"
MODEL="neo_s"
SAMPLES="64"
RUN_NAME="gridworld_${MODEL}_alpha${ALPHA}_seed${SEED}_test_plan"

echo "job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-0} alpha=${ALPHA} seed=${SEED}"
echo "host=$(hostname)"
date
nvidia-smi || true

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
  --epochs "${EPOCHS}" \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-2 \
  --warmup-ratio "${WARMUP}" \
  --min-lr-ratio 0.1 \
  --grad-clip 1.0 \
  --policy-lr-scale 0.25 \
  --transition-lr-scale 1.0 \
  --gumbel-tau-start 0.3 \
  --gumbel-tau-end 0.1 \
  --mixed-precision bf16 \
  --hidden-dim 32 \
  --ff-dim 128 \
  --action-dim 16 \
  --codebook-size 6 \
  --state-path cnn \
  --state-dropout 0.1 \
  --max-steps 10 \
  --train-rollout-steps 4 \
  --eval-rollout-steps 4 \
  --length-eval-rollout-steps 10 \
  --commitment-cost 0.25 \
  --split-mode paper \
  --paper-singleton U \
  --train-query-loss-weight 0.0 \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight 0.1 \
  --state-autoencoder-checkpoint "${STATE_AE}" \
  --freeze-state-autoencoder \
  --neo-s-samples "${SAMPLES}" \
  --validation-size 200 \
  --validation-interval 1 \
  --training-metrics-interval 1 \
  --loss-log-interval 0 \
  --device cuda \
  --out "${OUT_DIR}/${RUN_NAME}.json" \
  --checkpoint-out "${CKPT_DIR}/${RUN_NAME}.pt"
