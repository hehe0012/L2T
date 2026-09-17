#!/usr/bin/env bash
set -euo pipefail

cd /home/guian/L2T
mkdir -p logs/slurm logs checkpoints

echo "== Slurm nodes / GPU GRES =="
sinfo -N -o '%N %P %t %G %C %m'

echo
echo "== Current queue =="
squeue -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R %.20b %.20k'

echo
echo "== Submit GridWorld state autoencoder pretraining =="
PRETRAIN_OUTPUT=$(sbatch scripts/slurm_gridworld_state_autoencoder.sbatch)
echo "${PRETRAIN_OUTPUT}"
PRETRAIN_JOB_ID=$(printf '%s\n' "${PRETRAIN_OUTPUT}" | awk '{print $4}')

echo
echo "== Submit NEO-S full sweep after pretraining succeeds =="
sbatch --dependency="afterok:${PRETRAIN_JOB_ID}" scripts/slurm_gridworld_neo_s_full_pretrained.sbatch
