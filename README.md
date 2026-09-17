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

## Config Gap vs Paper

This repository is still a scaffold, not a faithful implementation of the paper. The table below records the main differences that explain why the current GridWorld NEO-S numbers can trail the paper's reported `0.9+` transfer accuracy.

Paper reference: [Learning to Theorize the World from Observation](https://arxiv.org/pdf/2605.03413)

### GridWorld

| Item | Paper GridWorld NEO / NEO-S | Current scaffold |
| --- | --- | --- |
| Train data | `100000` examples | `100000` in full Slurm runs |
| ID eval | `10000` examples | `10000` in full Slurm runs |
| Comp OOD eval | `10000` for `alpha=0.33/0.66`; none for `alpha=1.00` | scripts support `--comp-eval-size 0`; full Slurm scripts skip comp OOD for `alpha=1.00` |
| Length OOD eval | `20000` examples | `20000` in full Slurm runs |
| Train program lengths | `1-3` | `1-3` |
| Length OOD program lengths | `4-8` | `4-8` |
| Split construction | fixed anchors `UUU`, `DDD`, `LLL`, `RRR`, then alpha-sampled remaining short programs | implemented with deterministic alpha sampling |
| Train / ID / Comp max transition length | `K=4` | implemented via `--train-rollout-steps 4` and `--eval-rollout-steps 4` |
| Length OOD max transition length | `K=10` | implemented via `--length-eval-rollout-steps 10` with model `--max-steps 10` |
| NEO-S sampling budget | `B=64` for GridWorld | `--neo-s-samples 64` in latest NEO-S full |
| Learning rate | `5e-4` for NEO | `5e-4` |
| Weight decay | `1e-2` | implemented via `--weight-decay 1e-2` |
| LR schedule | warmup plus cosine decay, min LR ratio `0.1` | implemented via `--warmup-ratio` and `--min-lr-ratio` |
| Gradient clipping | `1.0` | implemented via `--grad-clip 1.0` |
| Two-timescale LR | policy scale `0.25`, transition scale `1.0` | implemented via optimizer param groups |
| Hidden / feedforward dims | `d_model=32`, `d_ff=128` | embedding/MLP hidden dim `32`; no separate `d_ff` |
| Policy / transition nets | FiLM-MLP | simple MLP |
| State representation | pretrained CNN VAE, state dim `32` | direct discrete state embedding |
| Latent action | discrete VQ, action dim `16`, codebook size `6` | categorical code embedding, codebook size `6`; no true VQ straight-through |
| Commitment / VQ loss | commitment cost `0.25`, action VQ loss `1.0` | not implemented |
| Gumbel-Softmax | tau `0.3 -> 0.1` | approximated with straight-through Gumbel-Softmax during training |
| State grounding loss | `0.1` | not implemented |
| MDL weight | `0.95` for `alpha=0.33/0.66`, `1.00` for `alpha=1.00` | implemented as alpha-dependent default; note the current NLL loss scale differs from the paper's reconstruction loss |

Priority fixes for GridWorld fidelity:

1. Replace categorical action embeddings with a true VQ / straight-through latent action path and commitment loss.
2. Add state grounding loss.
3. Replace the simple MLPs with the paper's FiLM-MLP policy / transition architecture.
4. Add the pretrained CNN VAE state encoder/decoder path, if reproducing the image-like GridWorld observation setting exactly.
5. Recalibrate MDL against the current loss scale, or move to the paper's reconstruction loss scale.

### Arithmetic

The Arithmetic task is not implemented yet in this scaffold. Important paper settings to match later:

| Item | Paper Arithmetic NEO / NEO-S |
| --- | --- |
| Train sizes | `147968` / `206520` / `279611` for `alpha=0.33/0.66/1.00` |
| ID eval sizes | `14796` / `20651` / `27961` |
| Comp OOD eval sizes | `146306` / `80401` / none |
| Length OOD eval size | `15317` |
| Batch size | `512` |
| Max epochs | `200` |
| Learning rate | `1.5e-3` |
| State embedding pretrain | embedding dim `8`, `500` epochs |
| Policy network | Transformer, `d_model=64`, `d_ff=256`, `4` heads, `6` layers |
| Transition network | Cross-attention, `d_model=32`, `d_ff=128`, `2` heads, `4` layers |
| Max transition length | `K=3` for train setting; length OOD uses longer inference budget |
| Codebook size | `16` for NEO, `40` for Disc-Mono |
| NEO-S sampling budget | `B=1024` for Arithmetic length OOD visualizations / scaling |
| Grounding loss | `0.5` |
| MDL | scheduled `1.01 -> 0.99` |

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
