#!/usr/bin/env python3
"""Pretrain the GridWorld CNN state encoder/decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from l2t.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gridworld_state_autoencoder.pt"))
    parser.add_argument("--log-out", type=Path, default=Path("logs/gridworld_state_autoencoder.json"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)

    model = build_model(
        "neo",
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        action_dim=args.action_dim,
        codebook_size=args.codebook_size,
        max_steps=1,
    ).to(args.device)
    state_indices = torch.arange(args.grid_size * args.grid_size, dtype=torch.long)
    loader = DataLoader(TensorDataset(state_indices), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(model.module.state_encoder.parameters()) + list(model.module.state_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_time = time.time()
    last_loss = None
    last_accuracy = None
    for _ in range(args.epochs):
        correct = 0
        total = 0
        for (batch_indices,) in loader:
            batch_indices = batch_indices.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            images = model._state_image(batch_indices)
            latent = model.module.state_encoder(images)
            logits = model.module.state_decoder(latent).flatten(1)
            loss = torch.nn.functional.cross_entropy(logits, batch_indices)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch_indices).sum().detach().cpu())
            total += int(batch_indices.numel())
        last_accuracy = correct / total if total else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_encoder": model.module.state_encoder.state_dict(),
            "state_decoder": model.module.state_decoder.state_dict(),
            "args": vars(args) | {
                "out": str(args.out),
                "log_out": str(args.log_out),
            },
            "last_loss": last_loss,
            "last_accuracy": last_accuracy,
            "runtime_sec": time.time() - start_time,
        },
        args.out,
    )
    payload = {
        "args": vars(args) | {
            "out": str(args.out),
            "log_out": str(args.log_out),
        },
        "last_loss": last_loss,
        "last_accuracy": last_accuracy,
        "runtime_sec": time.time() - start_time,
    }
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
