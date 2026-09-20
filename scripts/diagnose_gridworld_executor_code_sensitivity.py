#!/usr/bin/env python3
"""Measure whether the transition executor responds to different action codes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l2t.models import build_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(checkpoint["args"]["model"], **checkpoint["model_kwargs"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    size = int(checkpoint["args"].get("grid_size", 10))
    states = torch.arange(size * size, dtype=torch.long, device=args.device)
    state_latent = model._encode_indices(model.module, states)
    code_count = model.codebook_size
    codes = torch.arange(code_count, dtype=torch.long, device=args.device)

    with torch.no_grad():
        source = state_latent[:, None, :].expand(-1, code_count, -1).reshape(-1, state_latent.shape[-1])
        code_ids = codes[None, :].expand(states.shape[0], -1).reshape(-1)
        actions = model.module.action_codebook(code_ids)
        next_latent = model._execute_step(model.module, source, actions).view(states.shape[0], code_count, -1)
        next_logits = model._decode_logits(model.module, next_latent.reshape(-1, next_latent.shape[-1]))
        next_states = next_logits.argmax(dim=-1).view(states.shape[0], code_count)

        pairwise_latent = []
        unique_decoded = []
        decoded_changed = []
        for row in range(states.shape[0]):
            latents = next_latent[row]
            distances = torch.pdist(latents, p=2)
            pairwise_latent.append(float(distances.mean()))
            unique = int(torch.unique(next_states[row]).numel())
            unique_decoded.append(unique)
            decoded_changed.append(float(unique > 1))

    payload = {
        "checkpoint": str(args.checkpoint),
        "codebook_size": code_count,
        "state_count": int(states.numel()),
        "mean_pairwise_next_latent_distance": sum(pairwise_latent) / len(pairwise_latent),
        "median_pairwise_next_latent_distance": float(torch.tensor(pairwise_latent).median()),
        "mean_unique_decoded_next_states": sum(unique_decoded) / len(unique_decoded),
        "fraction_states_with_decoded_change": sum(decoded_changed) / len(decoded_changed),
        "all_codes_same_decoded_next_state_fraction": sum(x == 1 for x in unique_decoded) / len(unique_decoded),
        "per_state": [
            {
                "state": int(states[row]),
                "unique_decoded_next_states": unique_decoded[row],
                "pairwise_next_latent_distance": pairwise_latent[row],
                "decoded_next_states_by_code": next_states[row].cpu().tolist(),
            }
            for row in range(states.shape[0])
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_state"}, indent=2))


if __name__ == "__main__":
    main()
