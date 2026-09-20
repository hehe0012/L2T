#!/usr/bin/env python3
"""Pretrain the GridWorld state encoder/decoder."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action
from l2t.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=6)
    parser.add_argument("--state-path", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--state-dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-mode", choices=["deterministic", "vae"], default="deterministic")
    parser.add_argument("--vae-beta", type=float, default=1e-5)
    parser.add_argument("--displacement-loss-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["none", "bf16"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-out", type=Path, default=None)
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
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.out is None:
        suffix = "vae_state_autoencoder" if args.pretrain_mode == "vae" else "state_autoencoder"
        args.out = Path(f"checkpoints/gridworld_{args.state_path}_{suffix}.pt")
    if args.log_out is None:
        suffix = "vae_state_autoencoder" if args.pretrain_mode == "vae" else "state_autoencoder"
        args.log_out = Path(f"logs/gridworld_{args.state_path}_{suffix}.json")
    set_seed(args.seed)

    model = build_model(
        "neo",
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        action_dim=args.action_dim,
        codebook_size=args.codebook_size,
        state_path=args.state_path,
        state_dropout=args.state_dropout,
        max_steps=1,
    ).to(args.device)
    state_logvar_head = nn.Linear(args.hidden_dim, args.hidden_dim).to(args.device)
    state_indices = torch.arange(args.grid_size * args.grid_size, dtype=torch.long)
    loader = DataLoader(TensorDataset(state_indices), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(model.module.state_encoder.parameters())
        + list(model.module.state_decoder.parameters())
        + list(state_logvar_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_time = time.time()
    last_loss = None
    last_reconstruction_loss = None
    last_kl_loss = None
    last_accuracy = None
    last_displacement_loss = None
    transition_table = torch.tensor(
        [
            [
                apply_action(GridState.from_index(index, size=args.grid_size), action, size=args.grid_size).to_index(args.grid_size)
                for action in range(len(ACTIONS))
            ]
            for index in range(args.grid_size * args.grid_size)
        ],
        dtype=torch.long,
        device=args.device,
    )
    total_steps = max(1, args.epochs * len(loader))
    step = 0
    use_bf16 = args.mixed_precision == "bf16" and args.device == "cuda"
    for _ in range(args.epochs):
        correct = 0
        total = 0
        for (batch_indices,) in loader:
            batch_indices = batch_indices.to(args.device)
            lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * min(step / max(total_steps - 1, 1), 1.0))
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mean = model._encode_indices(model.module, batch_indices)
                if args.pretrain_mode == "vae":
                    logvar = state_logvar_head(mean).clamp(min=-20.0, max=20.0)
                    std = torch.exp(0.5 * logvar)
                    latent = mean + torch.randn_like(std) * std
                    kl_loss = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()
                else:
                    latent = mean
                    kl_loss = mean.new_zeros(())
                logits = model._decode_logits(model.module, latent)
                reconstruction_loss = torch.nn.functional.cross_entropy(logits, batch_indices)

                if args.displacement_loss_weight > 0:
                    # Pair every source state with all four one-step neighbors.
                    # Consistency is applied to the deterministic means so the
                    # saved encoder has stable geometry at inference time.
                    pair_sources = batch_indices[:, None].expand(-1, len(ACTIONS)).reshape(-1)
                    pair_targets = transition_table[batch_indices].reshape(-1)
                    pair_source_mean = model._encode_indices(model.module, pair_sources)
                    pair_target_mean = model._encode_indices(model.module, pair_targets)
                    pair_deltas = pair_target_mean - pair_source_mean
                    pair_deltas = pair_deltas.view(batch_indices.shape[0], len(ACTIONS), -1)
                    action_means = pair_deltas.mean(dim=0, keepdim=True)
                    displacement_loss = (pair_deltas - action_means).pow(2).mean()
                    target_logits = model._decode_logits(model.module, pair_target_mean)
                    reconstruction_loss = 0.5 * (
                        reconstruction_loss
                        + torch.nn.functional.cross_entropy(target_logits, pair_targets)
                    )
                else:
                    displacement_loss = mean.new_zeros(())
                loss = (
                    reconstruction_loss
                    + args.vae_beta * kl_loss
                    + args.displacement_loss_weight * displacement_loss
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.module.state_encoder.parameters())
                    + list(model.module.state_decoder.parameters())
                    + list(state_logvar_head.parameters()),
                    args.grad_clip,
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            last_reconstruction_loss = float(reconstruction_loss.detach().cpu())
            last_kl_loss = float(kl_loss.detach().cpu())
            last_displacement_loss = float(displacement_loss.detach().cpu())
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch_indices).sum().detach().cpu())
            total += int(batch_indices.numel())
            step += 1
        last_accuracy = correct / total if total else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_encoder": model.module.state_encoder.state_dict(),
            "state_decoder": model.module.state_decoder.state_dict(),
            "state_logvar_head": state_logvar_head.state_dict(),
            "args": vars(args) | {
                "out": str(args.out),
                "log_out": str(args.log_out),
            },
            "last_loss": last_loss,
            "last_reconstruction_loss": last_reconstruction_loss,
            "last_kl_loss": last_kl_loss,
            "last_displacement_loss": last_displacement_loss,
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
        "last_reconstruction_loss": last_reconstruction_loss,
        "last_kl_loss": last_kl_loss,
        "last_displacement_loss": last_displacement_loss,
        "last_accuracy": last_accuracy,
        "runtime_sec": time.time() - start_time,
    }
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
