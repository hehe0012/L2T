#!/bin/sh
#SBATCH --job-name=guian-l2t-grid-all1
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu
cd /home/guian/L2T

STATE_AE="checkpoints/gridworld_cnn_vae_state_autoencoder.pt"
OUT_DIR="logs/gridworld_neo_all_singletons_alpha033"
CKPT_DIR="checkpoints/gridworld_neo_all_singletons_alpha033"
RUN_NAME="gridworld_neo_alpha0.33_seed0_evalneos64_all_singletons"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" logs/slurm

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model neo \
  --seed 0 \
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
  --mdl-weight 0.95 \
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
