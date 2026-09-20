#!/usr/bin/env python3
"""Diagnose whether GridWorld state latents have stable action geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l2t.gridworld import ACTIONS, apply_action, GridState
from l2t.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=6)
    parser.add_argument("--state-path", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--state-dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_state_geometry.json"))
    return parser.parse_args()


def next_indices_for_action(action: int, *, size: int):
    indices = []
    for index in range(size * size):
        state = GridState.from_index(index, size=size)
        indices.append(apply_action(state, action, size=size, boundary="wrap").to_index(size))
    return indices


def summarize_action_geometry(model, states, latents, action: int, *, size: int):
    import torch
    import torch.nn.functional as F

    next_indices = torch.tensor(next_indices_for_action(action, size=size), dtype=torch.long, device=states.device)
    next_latents = model._encode_indices(model.module, next_indices)
    deltas = next_latents - latents
    mean_delta = deltas.mean(dim=0)
    centered = deltas - mean_delta
    delta_norms = deltas.norm(dim=-1)
    mean_delta_norm = mean_delta.norm()
    centered_norms = centered.norm(dim=-1)
    cosine_to_mean = F.cosine_similarity(deltas, mean_delta[None, :], dim=-1)

    translated_logits = model._decode_logits(model.module, latents + mean_delta[None, :])
    translated_predictions = translated_logits.argmax(dim=-1)
    translated_accuracy = (translated_predictions == next_indices).float().mean()

    true_next_logits = model._decode_logits(model.module, next_latents)
    true_next_accuracy = (true_next_logits.argmax(dim=-1) == next_indices).float().mean()

    return {
        "action": ACTIONS[action],
        "mean_delta_norm": float(mean_delta_norm.detach().cpu()),
        "mean_delta_l2_std": float(centered_norms.pow(2).mean().sqrt().detach().cpu()),
        "relative_l2_std": float((centered_norms.pow(2).mean().sqrt() / mean_delta_norm.clamp_min(1e-12)).detach().cpu()),
        "mean_delta_norm_over_states": float(delta_norms.mean().detach().cpu()),
        "std_delta_norm_over_states": float(delta_norms.std(unbiased=False).detach().cpu()),
        "mean_cosine_to_mean_delta": float(cosine_to_mean.mean().detach().cpu()),
        "min_cosine_to_mean_delta": float(cosine_to_mean.min().detach().cpu()),
        "mean_delta_decode_accuracy": float(translated_accuracy.detach().cpu()),
        "true_next_latent_decode_accuracy": float(true_next_accuracy.detach().cpu()),
        "mean_delta": [float(x) for x in mean_delta.detach().cpu().tolist()],
    }, deltas, next_indices


def main() -> None:
    import torch
    import torch.nn.functional as F

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = torch.load(args.state_checkpoint, map_location=args.device)
    checkpoint_args = checkpoint.get("args", {})
    hidden_dim = int(checkpoint_args.get("hidden_dim", args.hidden_dim))
    ff_dim = int(checkpoint_args.get("ff_dim", args.ff_dim))
    action_dim = int(checkpoint_args.get("action_dim", args.action_dim))
    codebook_size = int(checkpoint_args.get("codebook_size", args.codebook_size))
    state_path = checkpoint_args.get("state_path", args.state_path)
    state_dropout = float(checkpoint_args.get("state_dropout", args.state_dropout))

    model = build_model(
        "neo",
        grid_size=args.grid_size,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        action_dim=action_dim,
        codebook_size=codebook_size,
        state_path=state_path,
        state_dropout=state_dropout,
        max_steps=1,
    ).to(args.device)
    model.module.state_encoder.load_state_dict(checkpoint["state_encoder"])
    model.module.state_decoder.load_state_dict(checkpoint["state_decoder"])
    model.eval()

    states = torch.arange(args.grid_size * args.grid_size, dtype=torch.long, device=args.device)
    with torch.no_grad():
        latents = model._encode_indices(model.module, states)
        recon_logits = model._decode_logits(model.module, latents)
        recon_predictions = recon_logits.argmax(dim=-1)
        state_reconstruction_accuracy = (recon_predictions == states).float().mean()

        action_rows = []
        deltas_by_action = []
        next_indices_by_action = []
        for action in range(len(ACTIONS)):
            row, deltas, next_indices = summarize_action_geometry(
                model,
                states,
                latents,
                action,
                size=args.grid_size,
            )
            action_rows.append(row)
            deltas_by_action.append(deltas)
            next_indices_by_action.append(next_indices)

        mean_deltas = torch.stack([d.mean(dim=0) for d in deltas_by_action], dim=0)
        action_classification_correct = 0
        total = 0
        for action, deltas in enumerate(deltas_by_action):
            similarities = F.cosine_similarity(deltas[:, None, :], mean_deltas[None, :, :], dim=-1)
            predictions = similarities.argmax(dim=-1)
            action_classification_correct += int((predictions == action).sum().detach().cpu())
            total += int(predictions.numel())

    payload = {
        "state_checkpoint": str(args.state_checkpoint),
        "checkpoint_args": checkpoint_args,
        "state_reconstruction_accuracy": float(state_reconstruction_accuracy.detach().cpu()),
        "nearest_mean_delta_action_accuracy": action_classification_correct / max(total, 1),
        "actions": action_rows,
        "summary": {
            "mean_relative_l2_std": sum(row["relative_l2_std"] for row in action_rows) / len(action_rows),
            "mean_cosine_to_mean_delta": sum(row["mean_cosine_to_mean_delta"] for row in action_rows) / len(action_rows),
            "mean_delta_decode_accuracy": sum(row["mean_delta_decode_accuracy"] for row in action_rows) / len(action_rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
