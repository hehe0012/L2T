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
- Arithmetic Factorization Reasoning: implemented with prime-exponent data, Transformer policy, cross-attention transition, VQ actions, grounding loss, NEO-S sampling, and state embedding pretraining

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
| Validation | held-out ID examples, `200` samples every epoch | `--validation-size 200 --validation-interval 1`; records `validation_history` and `best_validation` |
| Learning rate | `5e-4` for NEO | `5e-4` |
| Weight decay | `1e-2` | implemented via `--weight-decay 1e-2` |
| LR schedule | warmup plus cosine decay, min LR ratio `0.1` | implemented via `--warmup-ratio` and `--min-lr-ratio` |
| Gradient clipping | `1.0` | implemented via `--grad-clip 1.0` |
| Two-timescale LR | policy scale `0.25`, transition scale `1.0` | implemented via optimizer param groups |
| Hidden / feedforward dims | `d_model=32`, `d_ff=128` | implemented as state dim `32`, feedforward dim `128` |
| Policy / transition nets | FiLM-MLP | implemented as FiLM-conditioned MLP policy and transition |
| State representation | pretrained CNN VAE, state dim `32` | CNN state encoder/decoder path; paper pretraining settings are restored |
| Latent action | discrete VQ, action dim `16`, codebook size `6` | implemented as nearest-code VQ path with action dim `16` and codebook size `6` |
| Commitment / VQ loss | commitment cost `0.25`, action VQ loss `1.0` | implemented via `--commitment-cost 0.25` and `--vq-loss-weight 1.0` |
| Gumbel-Softmax | tau `0.3 -> 0.1` | approximated with straight-through Gumbel-Softmax during training |
| State grounding loss | `0.1` | implemented as decode-encode state grounding with stop-gradient target |
| MDL weight | `0.95` for `alpha=0.33/0.66`, `1.00` for `alpha=1.00` | implemented as alpha-dependent default; note the current NLL loss scale differs from the paper's reconstruction loss |

Priority fixes for GridWorld fidelity:

1. Re-run the paper-standard CNN VAE pretraining checkpoint before launching NEO-S.
2. Recalibrate MDL only if the reconstruction loss implementation is changed from the paper's scale.
3. Run a fresh full NEO-S Slurm sweep from the paper-standard entry point.

### Arithmetic

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

Current implementation status:

| Item | Current scaffold |
| --- | --- |
| Data generation | implemented as multiplication programs over `x2`, `x3`, `x5`, `x7`; observations are integer values with prime-exponent tensors for stable learning |
| Split construction | fixed anchors `x2 x2 x2`, `x3 x3 x3`, `x5 x5 x5`, `x7 x7 x7`, then deterministic alpha sampling |
| Train / ID / Comp lengths | `1-3` |
| Length OOD lengths | `4-6` |
| State path | embedding dim `8`; separate `500` epoch state encoder/decoder pretraining script |
| Policy network | Transformer defaults match paper: `d_model=64`, `d_ff=256`, `4` heads, `6` layers |
| Transition network | cross-attention defaults match paper: `d_model=32`, `d_ff=128`, `2` heads, `4` layers |
| Latent action | NEO codebook size `16`; Disc-Mono codebook size `40`; action dim `4` |
| NEO-S sampling | `--neo-s-samples 1024` default |
| Losses | VQ commitment loss and state grounding loss enabled; grounding weight `0.5` |
| MDL | linear schedule from `1.01` to `0.99` across epochs |

Arithmetic smoke run:

```bash
python scripts/pretrain_arithmetic_state_embedding.py --epochs 2 --batch-size 256 --device cpu --checkpoint-out checkpoints/arithmetic_state_embedding_smoke.pt
python scripts/run_arithmetic.py \
  --model neo_s \
  --train-size 512 \
  --id-eval-size 128 \
  --comp-eval-size 128 \
  --length-eval-size 128 \
  --epochs 1 \
  --batch-size 64 \
  --neo-s-samples 8 \
  --policy-layers 1 \
  --transition-layers 1 \
  --device cpu \
  --state-checkpoint checkpoints/arithmetic_state_embedding_smoke.pt \
  --out logs/arithmetic_smoke.json
```

Paper-scale Arithmetic run uses the script defaults for size, batch, epochs, architecture, and NEO-S sampling budget:

```bash
python scripts/pretrain_arithmetic_state_embedding.py --device cuda --checkpoint-out checkpoints/arithmetic_state_embedding.pt
python scripts/run_arithmetic.py \
  --model neo_s \
  --alpha 0.66 \
  --device cuda \
  --state-checkpoint checkpoints/arithmetic_state_embedding.pt \
  --freeze-state \
  --out logs/arithmetic_neo_s_alpha_0.66.json
```

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

Pretrain the GridWorld state encoder/decoder first, then run the full NEO-S sweep:

```bash
cd /home/guian/L2T
scripts/submit_gridworld_pretrain_then_neo_s_full.sh
```

This submits the full sweep with a Slurm `afterok` dependency on the paper-standard CNN-VAE pretraining job. The NEO-S run loads `checkpoints/gridworld_state_autoencoder.pt` with `--freeze-state-autoencoder` and writes to:

```text
logs/gridworld_neo_s_full_pretrained
checkpoints/gridworld_neo_s_full_pretrained
```

Older full sweep entry point:

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
