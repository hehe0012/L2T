#!/bin/sh
#SBATCH --job-name=guian-l2t-grid-loss
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu

cd /home/guian/L2T
mkdir -p logs/slurm logs/gridworld_loss_probe

STATE_AE="checkpoints/gridworld_cnn_vae_state_autoencoder.pt"
if [ ! -f "${STATE_AE}" ]; then
  echo "missing state autoencoder checkpoint: ${STATE_AE}" >&2
  exit 2
fi

echo "job=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
date
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/home/guian/L2T/.venv/bin/python scripts/run_gridworld.py \
  --model neo_s \
  --seed 0 \
  --alpha 0.33 \
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
  --mdl-lambda 0.0 \
  --train-query-loss-weight 0.0 \
  --neo-s-selection majority_exact \
  --vq-loss-weight 1.0 \
  --grounding-loss-weight 0.1 \
  --state-autoencoder-checkpoint "${STATE_AE}" \
  --freeze-state-autoencoder \
  --neo-s-samples 64 \
  --validation-size 200 \
  --validation-interval 1 \
  --loss-log-interval 0 \
  --device cuda \
  --out logs/gridworld_loss_probe/gridworld_neo_s_alpha0.33_seed0_loss_history.json
