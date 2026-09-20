# Latent Transition Geometry Diagnosis

## Experimental Setup

- Checkpoint: `gridworld_neo_s_alpha0.33_seed0_samples64_epochs150_paper_vae.pt`
- Training architecture: pre-fix residual executor
- Grid states evaluated: all `100` states in the 10x10 GridWorld
- Latent dimension: `32`
- Learned action codes evaluated: `6`
- For each code `z`, evaluate all states and measure:

```text
delta(s, z) = s_next(s, z) - s
```

The actual pre-fix executor used:

```text
s_next = s + Transition(s, z)
delta = Transition(s, z)
```

Therefore the `residual_transition_delta` column is the relevant result for the
checkpoint as it was trained. The `direct_transition_delta` column is included
to show what the same weights would imply under the paper-style direct executor.

## Results

Metrics are computed across all pairs of states for each learned code.

| Code | Actual mean norm | Actual norm std | Actual mean pairwise cosine | Positive direction fraction | Actual coordinate variance | Actual mean pairwise distance |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17.0711 | 5.2200 | 0.1190 | 0.6620 | 8.667895 | 22.9223 |
| 1 | 15.4664 | 4.7761 | 0.1911 | 0.7459 | 6.486353 | 19.8283 |
| 2 | 14.6223 | 4.7675 | 0.1458 | 0.6927 | 6.243788 | 19.3915 |
| 3 | 16.0150 | 5.1302 | 0.0855 | 0.6077 | 7.918607 | 21.9064 |
| 4 | 13.9295 | 4.3655 | 0.1735 | 0.7053 | 5.323613 | 17.9833 |
| 5 | 14.2894 | 4.3071 | 0.0904 | 0.6228 | 6.333137 | 19.5431 |

For comparison, under the direct interpretation of the same transition weights,
the mean pairwise cosine values are `0.0835-0.1375`, also low.

## Interpretation

The same learned code does not produce a stable direction across different
latent states:

- Mean pairwise direction cosine is only `0.0855-0.1911`.
- The positive-direction fraction is only `0.6077-0.7459`.
- Norm variation is substantial, with norm standard deviation `4.31-5.22`.
- Pairwise delta distances are large relative to the mean direction agreement.

This is evidence that the learned code semantics are state-dependent and are
not yet behaving like reusable primitive translations. It is consistent with
poor compositional transfer, but it does not by itself identify the cause.
Possible causes still include:

1. VAE latent geometry is not translation-friendly.
2. The transition network learned a state-dependent shortcut.
3. The codebook has collapsed or codes do not correspond to primitive actions.
4. The checkpoint was trained with the residual executor, while the paper uses
   direct deterministic transition.

The raw diagnostic JSON was generated at:

```text
/tmp/latent_transition_geometry_alpha033_seed0.json
```

