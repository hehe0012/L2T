#!/usr/bin/env python3
"""Pretrain the Arithmetic state encoder/decoder on prime-exponent states."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from l2t.arithmetic_models import build_arithmetic_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--max-exponent", type=int, default=12)
    parser.add_argument("--min-lr-ratio", type=float, default=0.005)
    parser.add_argument("--mixed-precision", choices=["none", "bf16"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/arithmetic_state_embedding.pt"))
    parser.add_argument("--out", type=Path, default=Path("logs/arithmetic_state_embedding.json"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_states(max_exponent: int):
    import torch

    values = []
    for e2 in range(max_exponent + 1):
        for e3 in range(max_exponent + 1):
            for e5 in range(max_exponent + 1):
                for e7 in range(max_exponent + 1):
                    values.append((e2, e3, e5, e7))
    return torch.tensor(values, dtype=torch.long)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)

    model = build_arithmetic_model(
        "neo",
        state_dim=args.state_dim,
        max_exponent=args.max_exponent,
        policy_layers=1,
        transition_layers=1,
    ).to(args.device)
    states = all_states(args.max_exponent)
    loader = DataLoader(TensorDataset(states), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(model.module.state_encoder.parameters()) + list(model.module.state_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    last_loss = 0.0
    last_accuracy = 0.0
    total_steps = max(1, args.epochs * len(loader))
    step = 0
    for _ in range(args.epochs):
        correct = 0
        total = 0
        for (batch,) in loader:
            batch = batch.to(args.device)
            progress = min(step / max(total_steps - 1, 1), 1.0)
            lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = args.mixed_precision == "bf16" and args.device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                state = model.module.state_encoder(model._normalize_exponents(batch))
                logits = model._decode_logits(model.module, state)
                loss = model._state_loss(logits, batch)
            loss.backward()
            optimizer.step()
            step += 1
            last_loss = float(loss.detach().cpu())
            correct += int((logits.argmax(dim=-1) == batch).all(dim=1).sum().detach().cpu())
            total += batch.shape[0]
        last_accuracy = correct / total

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_encoder": model.module.state_encoder.state_dict(),
            "state_decoder": model.module.state_decoder.state_dict(),
            "args": vars(args) | {"checkpoint_out": str(args.checkpoint_out), "out": str(args.out)},
        },
        args.checkpoint_out,
    )
    payload_args = vars(args).copy()
    payload_args["checkpoint_out"] = str(args.checkpoint_out)
    payload_args["out"] = str(args.out)
    payload = {"loss": last_loss, "accuracy": last_accuracy, "num_states": len(states), "args": payload_args}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
