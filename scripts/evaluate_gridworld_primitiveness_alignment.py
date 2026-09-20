#!/usr/bin/env python3
"""Evaluate GridWorld Action Primitiveness and Code-Primitive Alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action
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
    source_latent = model._encode_indices(model.module, states)
    codebook_size = model.codebook_size

    with torch.no_grad():
        source = source_latent[:, None, :].expand(-1, codebook_size, -1).reshape(-1, source_latent.shape[-1])
        codes = torch.arange(codebook_size, device=args.device)[None, :].expand(states.numel(), -1).reshape(-1)
        actions = model.module.action_codebook(codes)
        next_latent = model._execute_step(model.module, source, actions)
        predicted = model._decode_logits(model.module, next_latent).argmax(dim=-1).view(states.numel(), codebook_size)

    targets = torch.empty((states.numel(), len(ACTIONS)), dtype=torch.long, device=args.device)
    for action in range(len(ACTIONS)):
        targets[:, action] = torch.tensor(
            [
                apply_action(GridState.from_index(int(state), size=size), action, size=size, boundary="wrap").to_index(size)
                for state in states.cpu()
            ],
            dtype=torch.long,
            device=args.device,
        )

    # C[i, j] from the paper: exact matches of learned code i to GT primitive j.
    alignment = torch.zeros((codebook_size, len(ACTIONS)), dtype=torch.long)
    exact = torch.zeros((states.numel(), codebook_size, len(ACTIONS)), dtype=torch.bool)
    for action in range(len(ACTIONS)):
        matches = predicted == targets[:, action, None]
        exact[:, :, action] = matches
        alignment[:, action] = matches.sum(dim=0).cpu()

    primitiveness = exact.any(dim=1).float().mean().item()
    per_primitive = exact.any(dim=1).float().mean(dim=0).cpu().tolist()
    per_code = exact.float().mean(dim=0).cpu().tolist()
    best_code = exact.float().mean(dim=0).argmax(dim=0).cpu().tolist()

    payload = {
        "checkpoint": str(args.checkpoint),
        "grid_size": size,
        "codebook_size": codebook_size,
        "primitive_count": len(ACTIONS),
        "num_primitive_examples": int(states.numel() * len(ACTIONS)),
        "action_primitiveness": primitiveness,
        "primitiveness_by_primitive": {
            ACTIONS[action]: per_primitive[action] for action in range(len(ACTIONS))
        },
        "alignment_counts_code_by_primitive": {
            f"code_{code}": {
                ACTIONS[action]: int(alignment[code, action])
                for action in range(len(ACTIONS))
            }
            for code in range(codebook_size)
        },
        "alignment_rate_code_by_primitive": {
            f"code_{code}": {
                ACTIONS[action]: per_code[code][action]
                for action in range(len(ACTIONS))
            }
            for code in range(codebook_size)
        },
        "best_code_for_primitive": {
            ACTIONS[action]: int(best_code[action]) for action in range(len(ACTIONS))
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
