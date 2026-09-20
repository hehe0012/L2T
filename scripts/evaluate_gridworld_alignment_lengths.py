#!/usr/bin/env python3
"""Measure exact code/primitive alignment for sequence lengths 1, 2, and 3."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from l2t.gridworld import ACTIONS, GridState, apply_action
from l2t.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(ckpt["args"]["model"], **ckpt["model_kwargs"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = int(ckpt["args"].get("grid_size", 10))
    codebook_size = int(model.codebook_size)
    states = torch.arange(size * size, dtype=torch.long, device=args.device)
    with torch.no_grad():
        source = model._encode_indices(model.module, states)

    by_length = {}
    for length in range(1, args.max_length + 1):
        gt_programs = list(itertools.product(range(len(ACTIONS)), repeat=length))
        learned_programs = list(itertools.product(range(codebook_size), repeat=length))
        with torch.no_grad():
            predictions = []
            for learned in learned_programs:
                latent = source
                for code in learned:
                    action = model.module.action_codebook(
                        torch.full((states.shape[0],), code, dtype=torch.long, device=args.device)
                    )
                    latent = model._execute_step(model.module, latent, action)
                predictions.append(model._decode_logits(model.module, latent).argmax(dim=-1))
            predictions = torch.stack(predictions, dim=1)

        targets = []
        for program in gt_programs:
            values = []
            for state in states.cpu().tolist():
                current = GridState.from_index(state, size=size)
                for action in program:
                    current = apply_action(current, action, size=size, boundary="wrap")
                values.append(current.to_index(size))
            targets.append(torch.tensor(values, dtype=torch.long, device=args.device))
        targets = torch.stack(targets, dim=1)
        exact = predictions[:, :, None] == targets[:, None, :]
        alignment = exact.sum(dim=0)
        covered = exact.any(dim=1).any(dim=0)
        gt_to_best = alignment.max(dim=0).values
        by_length[str(length)] = {
            "num_ground_truth_programs": len(gt_programs),
            "num_learned_code_sequences": len(learned_programs),
            "action_primitiveness": float(covered.float().mean().cpu()),
            "covered_ground_truth_programs": int(covered.sum().cpu()),
            "alignment_counts": {
                "gt_programs": ["".join(ACTIONS[a] for a in p) for p in gt_programs],
                "learned_code_sequences": ["".join(str(c) for c in p) for p in learned_programs],
                "counts_code_sequence_by_gt_program": alignment.cpu().tolist(),
                "max_exact_matches_per_gt": [int(x) for x in gt_to_best.cpu().tolist()],
            },
        }

    payload = {"checkpoint": str(args.checkpoint), "grid_size": size, "codebook_size": codebook_size, "by_length": by_length}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
