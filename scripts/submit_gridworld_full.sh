#!/usr/bin/env bash
set -euo pipefail

cd /home/guian/L2T
mkdir -p logs/gridworld_full logs/slurm checkpoints/gridworld_full

echo "== Slurm partitions =="
sinfo -o '%P %a %l %D %t %G'

echo
echo "== Current queue =="
squeue -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R'

echo
echo "== Submit GridWorld full sweep with tag guian =="
sbatch scripts/slurm_gridworld_full.sbatch
