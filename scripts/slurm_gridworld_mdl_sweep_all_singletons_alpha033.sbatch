#!/bin/sh
#SBATCH --job-name=guian-l2t-grid-mdl
#SBATCH --comment=guian
#SBATCH --chdir=/home/guian/L2T
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-14%3
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%A_%a.err

set -eu
cd /home/guian/L2T

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MDL_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))

case "${MDL_INDEX}" in
  0) MDL_WEIGHT="0.5" ;;
  1) MDL_WEIGHT="0.8" ;;
  2) MDL_WEIGHT="1.0" ;;
  3) MDL_WEIGHT="1.2" ;;
  4) MDL_WEIGHT="1.5" ;;
  *) echo "invalid MDL index=${MDL_INDEX}" >&2; exit 2 ;;
esac
case "${SEED_INDEX}" in
  0) SEED="0" ;;
  1) SEED="1" ;;
  2) SEED="2" ;;
  *) echo "invalid seed index=${SEED_INDEX}" >&2; exit 2 ;;
esac

STATE_AE="checkpoints/gridworld_cnn_vae_state_autoencoder.pt"
RUN_TAG="mdl_sweep_all_singletons_alpha033"
OUT_DIR="logs/gridworld_${RUN_TAG}"
CKPT_DIR="checkpoints/gridworld_${RUN_TAG}"
RUN_NAME="gridworld_neo_alpha0.33_seed${SEED}_mdl${MDL_WEIGHT}_all_singletons"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" logs/slurm

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job=${SLURM_JOB_ID:-unknown} task=${TASK_ID} seed=${SEED} mdl_weight=${MDL_WEIGHT}"
date
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model neo \
  --seed "${SEED}" \
  --alpha 0.33 \
  --paper-singleton ALL \
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
  --neo-s-samples 64 \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight 0.1 \
  --grounding-transition-only \
  --state-autoencoder-checkpoint "${STATE_AE}" \
  --freeze-state-autoencoder \
  --validation-size 200 \
  --validation-interval 1 \
  --device cuda \
  --out "${OUT_DIR}/${RUN_NAME}.json" \
  --checkpoint-out "${CKPT_DIR}/${RUN_NAME}.pt"
