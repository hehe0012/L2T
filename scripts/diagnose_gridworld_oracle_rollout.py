#!/usr/bin/env python3
"""Separate one-step transition error from latent rollout drift."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action
from l2t.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-checkpoint", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_oracle_rollout_diagnosis.json"))
    return parser.parse_args()


def summarize(pred, target, predicted_latent, target_latent) -> dict[str, float]:
    import torch.nn.functional as F

    return {
        "accuracy": float((pred == target).float().mean().cpu()),
        "latent_mse": float((predicted_latent - target_latent).pow(2).mean().cpu()),
        "latent_cosine": float(
            F.cosine_similarity(predicted_latent, target_latent, dim=-1).mean().cpu()
        ),
        "predicted_latent_norm": float(predicted_latent.norm(dim=-1).mean().cpu()),
        "target_latent_norm": float(target_latent.norm(dim=-1).mean().cpu()),
    }


def main() -> None:
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    oracle = torch.load(args.oracle_checkpoint, map_location=args.device)
    oracle_args = oracle.get("args", {})
    state_checkpoint = torch.load(args.state_checkpoint, map_location=args.device)
    state_args = state_checkpoint.get("args", {})
    oracle_model_state = oracle["model"]
    inferred_codebook_size = int(oracle_model_state["action_codebook.weight"].shape[0])
    model = build_model(
        "neo",
        grid_size=args.grid_size,
        codebook_size=int(oracle_args.get("codebook_size", inferred_codebook_size)),
        hidden_dim=int(oracle_args.get("hidden_dim", state_args.get("hidden_dim", 32))),
        ff_dim=int(oracle_args.get("ff_dim", state_args.get("ff_dim", 128))),
        action_dim=int(oracle_args.get("action_dim", state_args.get("action_dim", 16))),
        max_steps=max(args.max_len, 1),
        state_path=oracle_args.get("state_path", state_args.get("state_path", "cnn")),
        state_dropout=float(oracle_args.get("state_dropout", state_args.get("state_dropout", 0.0))),
    ).to(args.device)
    model.load_state_dict(oracle_model_state)
    model.eval()

    starts = [random.randrange(args.grid_size * args.grid_size) for _ in range(args.num_sequences)]
    actions = [
        [random.randrange(len(ACTIONS)) for _ in range(args.max_len)]
        for _ in range(args.num_sequences)
    ]
    states_by_step = [starts]
    for step in range(args.max_len):
        states_by_step.append(
            [
                apply_action(
                    GridState.from_index(states_by_step[-1][row], size=args.grid_size),
                    actions[row][step],
                    size=args.grid_size,
                    boundary="wrap",
                ).to_index(args.grid_size)
                for row in range(args.num_sequences)
            ]
        )

    rows: dict[str, dict[str, dict[str, float]]] = {}
    with torch.no_grad():
        initial = torch.tensor(starts, dtype=torch.long, device=args.device)
        free_latent = model._encode_indices(model.module, initial)
        projected_latent = free_latent.clone()
        for step in range(1, args.max_len + 1):
            action_codes = torch.tensor(
                [actions[row][step - 1] for row in range(args.num_sequences)],
                dtype=torch.long,
                device=args.device,
            )
            action_embedding = model.module.action_codebook(action_codes)
            target_indices = torch.tensor(states_by_step[step], dtype=torch.long, device=args.device)
            target_latent = model._encode_indices(model.module, target_indices)

            # Teacher forcing: each step starts from the true encoded state.
            teacher_source = model._encode_indices(
                model.module,
                torch.tensor(states_by_step[step - 1], dtype=torch.long, device=args.device),
            )
            teacher_next = model._execute_step(model.module, teacher_source, action_embedding)
            teacher_logits = model._decode_logits(model.module, teacher_next)

            free_latent = model._execute_step(model.module, free_latent, action_embedding)
            free_logits = model._decode_logits(model.module, free_latent)

            projected_latent = model._execute_step(model.module, projected_latent, action_embedding)
            projected_logits = model._decode_logits(model.module, projected_latent)
            projected_next_latent = model._encode_decoded_state(model.module, projected_latent)

            rows[str(step)] = {
                "teacher_forced": summarize(
                    teacher_logits.argmax(dim=-1), target_indices, teacher_next, target_latent
                ),
                "free_latent": summarize(
                    free_logits.argmax(dim=-1), target_indices, free_latent, target_latent
                ),
                "decode_reencode": summarize(
                    projected_logits.argmax(dim=-1), target_indices, projected_latent, target_latent
                ),
            }
            projected_latent = projected_next_latent

    payload = {
        "args": vars(args)
        | {
            "oracle_checkpoint": str(args.oracle_checkpoint),
            "state_checkpoint": str(args.state_checkpoint),
            "out": str(args.out),
        },
        "oracle_args": oracle_args,
        "num_sequences": args.num_sequences,
        "actions": list(ACTIONS),
        "per_step": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
