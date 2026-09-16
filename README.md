# L2T Reproduction

This repository is a local reproduction scaffold for **Learning to Theorize the World from Observation**.

The first milestone targets a minimal OTIB GridWorld loop:

1. Generate support/query pairs from the same hidden program.
2. Infer a latent program from the support pair.
3. Execute that program on the query input.
4. Measure transfer accuracy.

## Current Status

- GridWorld data generation: implemented
- GridWorld metrics: implemented
- NEO-style compositional model: initial PyTorch scaffold
- NEO-S sampled inference: implemented
- Disc-Mono baseline: initial PyTorch scaffold
- Smoke training script: implemented
- Full paper reproduction: not yet complete

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Smoke Test

```bash
python -m unittest discover tests
python scripts/run_gridworld.py --model neo --train-size 1000 --eval-size 200 --steps 200 --batch-size 64 --device cpu --out logs/gridworld_smoke.json
python scripts/run_gridworld.py --model neo_s --neo-s-samples 8 --train-size 1000 --eval-size 200 --steps 200 --batch-size 64 --device cpu --out logs/gridworld_neo_s_smoke.json
```

## Slurm GridWorld Full Sweep

The Slurm-visible mirror lives at `/home/guian/L2T` because compute nodes use a different `/raid` view.

```bash
cd /home/guian/L2T
scripts/submit_gridworld_full.sh
python scripts/summarize_gridworld.py --log-dir logs/gridworld_full
```

## Length Direction Sweep

Compare standard short-to-long evaluation against reverse long-to-short evaluation:

```bash
python scripts/run_gridworld_length_direction_sweep.py \
  --models neo neo_s disc_mono \
  --alphas 0.66 1.0 \
  --seeds 0 1 \
  --train-size 3000 \
  --eval-size 500 \
  --steps 600 \
  --max-steps 8 \
  --device cpu
```
