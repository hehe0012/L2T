#!/bin/sh
#SBATCH --job-name=guian-l2t-diag
#SBATCH --comment=guian
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/guian/L2T/logs/slurm/%x_%j.out
#SBATCH --error=/home/guian/L2T/logs/slurm/%x_%j.err

set -eu

cd /home/guian/L2T

hostname
pwd
date
ls -ld /home/guian /home/guian/L2T
ls -l /bin/bash /usr/bin/env /home/guian/L2T/.venv/bin/python
command -v nvidia-smi || true
nvidia-smi || true
/home/guian/L2T/.venv/bin/python --version
/home/guian/L2T/.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
