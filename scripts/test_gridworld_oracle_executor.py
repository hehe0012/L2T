#!/usr/bin/env python3
"""Oracle-action sanity test for GridWorld executor.

This bypasses the programmer entirely. The model receives the true primitive
action id at every step, so success means the state encoder/decoder plus
executor can represent reusable primitive actions. Failure points to the
executor/state representation rather than the program inference module.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action, apply_program, enumerate_programs
from l2t.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-path", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--state-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-min-len", type=int, default=1)
    parser.add_argument("--train-max-len", type=int, default=1)
    parser.add_argument("--eval-states", type=int, default=100)
    parser.add_argument("--max-programs-per-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-autoencoder-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-state-autoencoder", action="store_true")
    parser.add_argument("--train-decoder", action="store_true")
    parser.add_argument("--boundary", choices=["wrap", "clamp"], default="wrap")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_oracle_executor.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_one_step_pairs(*, size: int, boundary: str) -> tuple[list[int], list[int], list[int]]:
    states: list[int] = []
    actions: list[int] = []
    targets: list[int] = []
    for index in range(size * size):
        state = GridState.from_index(index, size=size)
        for action in range(len(ACTIONS)):
            states.append(index)
            actions.append(action)
            targets.append(apply_action(state, action, size=size, boundary=boundary).to_index(size))
    return states, actions, targets


def sample_oracle_batch(
    *,
    rng: random.Random,
    batch_size: int,
    train_min_len: int,
    train_max_len: int,
    size: int,
    boundary: str,
    device: str,
):
    import torch

    starts: list[int] = []
    programs: list[list[int]] = []
    targets: list[int] = []
    lengths: list[int] = []
    for _ in range(batch_size):
        start = rng.randrange(size * size)
        length = rng.randint(train_min_len, train_max_len)
        program = [rng.randrange(len(ACTIONS)) for _ in range(length)]
        target = apply_program(GridState.from_index(start, size=size), program, size=size, boundary=boundary).to_index(size)
        starts.append(start)
        targets.append(target)
        lengths.append(length)
        programs.append(program + [0] * (train_max_len - length))
    return (
        torch.tensor(starts, dtype=torch.long, device=device),
        torch.tensor(programs, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def load_state_autoencoder(model, path: Path, *, device: str, freeze: bool) -> None:
    import torch

    checkpoint = torch.load(path, map_location=device)
    model.module.state_encoder.load_state_dict(checkpoint["state_encoder"])
    model.module.state_decoder.load_state_dict(checkpoint["state_decoder"])
    if freeze:
        for parameter in model.module.state_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.module.state_decoder.parameters():
            parameter.requires_grad = False


def decode_after_program(model, start_indices, programs, *, device: str):
    import torch

    state = model._encode_indices(model.module, start_indices)
    for step_codes in programs:
        action = model.module.action_codebook(step_codes.to(device))
        state = model._execute_step(model.module, state, action)
    return model._decode_logits(model.module, state)


def decode_after_padded_program(model, start_indices, programs, lengths, *, max_len: int, device: str):
    import torch

    state = model._encode_indices(model.module, start_indices)
    logits_by_step = []
    for step in range(max_len):
        action = model.module.action_codebook(programs[:, step].to(device))
        state = model._execute_step(model.module, state, action)
        logits_by_step.append(model._decode_logits(model.module, state))
    row = torch.arange(start_indices.shape[0], device=device)
    return torch.stack(logits_by_step, dim=1)[row, lengths - 1]


def evaluate_one_step(model, *, size: int, boundary: str, device: str) -> dict[str, float]:
    import torch

    states, actions, targets = all_one_step_pairs(size=size, boundary=boundary)
    states_t = torch.tensor(states, dtype=torch.long, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = decode_after_program(model, states_t, [actions_t], device=device)
        pred = logits.argmax(dim=-1)
    return {
        "accuracy": float((pred == targets_t).float().mean().detach().cpu()),
    }


def evaluate_programs(
    model,
    *,
    min_len: int,
    max_len: int,
    size: int,
    boundary: str,
    device: str,
    seed: int,
    eval_states: int,
    max_programs_per_length: int,
) -> dict[str, float]:
    import torch

    total = 0
    correct = 0
    per_len: dict[int, list[float]] = {}
    rng = random.Random(seed)
    all_states = list(range(size * size))
    if eval_states < len(all_states):
        state_list = sorted(rng.sample(all_states, eval_states))
    else:
        state_list = all_states
    states = torch.tensor(state_list, dtype=torch.long, device=device)
    programs: list[tuple[int, ...]] = []
    for length in range(min_len, max_len + 1):
        length_programs = enumerate_programs(min_len=length, max_len=length)
        if max_programs_per_length > 0 and len(length_programs) > max_programs_per_length:
            length_programs = sorted(rng.sample(length_programs, max_programs_per_length))
        programs.extend(length_programs)
    for program in programs:
        step_codes = [
            torch.full((len(state_list),), action, dtype=torch.long, device=device)
            for action in program
        ]
        targets = torch.tensor(
            [
                apply_program(GridState.from_index(index, size=size), program, size=size, boundary=boundary).to_index(size)
                for index in state_list
            ],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            pred = decode_after_program(model, states, step_codes, device=device).argmax(dim=-1)
        acc = float((pred == targets).float().mean().detach().cpu())
        per_len.setdefault(len(program), []).append(acc)
        correct += int((pred == targets).sum().detach().cpu())
        total += targets.numel()
    metrics = {"accuracy": correct / total if total else 0.0}
    metrics["num_programs"] = float(len(programs))
    metrics["num_states"] = float(len(state_list))
    for length, values in sorted(per_len.items()):
        metrics[f"length_{length}_accuracy"] = sum(values) / len(values)
    return metrics


def code_transition_consistency(model, *, size: int, device: str) -> list[dict[str, object]]:
    import torch
    from collections import Counter

    states = torch.arange(size * size, dtype=torch.long, device=device)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for code in range(len(ACTIONS)):
            action = torch.full((size * size,), code, dtype=torch.long, device=device)
            pred = decode_after_program(model, states, [action], device=device).argmax(dim=-1).cpu().tolist()
            counts = Counter()
            for start, end in zip(states.cpu().tolist(), pred):
                s = GridState.from_index(start, size=size)
                e = GridState.from_index(end, size=size)
                counts[((e.row - s.row) % size, (e.col - s.col) % size)] += 1
            dominant, count = counts.most_common(1)[0]
            rows.append(
                {
                    "code": code,
                    "expected_action": ACTIONS[code],
                    "dominant_displacement": list(dominant),
                    "consistency": count / (size * size),
                    "unique_displacements": len(counts),
                }
            )
    return rows


def main() -> None:
    import torch
    from torch import nn

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)

    model = build_model(
        "neo",
        grid_size=args.grid_size,
        codebook_size=len(ACTIONS),
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        action_dim=args.action_dim,
        max_steps=8,
        state_path=args.state_path,
        state_dropout=args.state_dropout,
        grounding_loss_weight=0.0,
        vq_loss_weight=0.0,
    ).to(args.device)
    if args.state_autoencoder_checkpoint is not None:
        load_state_autoencoder(
            model,
            args.state_autoencoder_checkpoint,
            device=args.device,
            freeze=args.freeze_state_autoencoder,
        )

    trainable = []
    for name, parameter in model.module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("transition.") or name.startswith("action_codebook."):
            trainable.append(parameter)
        elif args.train_decoder and name.startswith("state_decoder."):
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_rng = random.Random(args.seed)
    history: list[dict[str, float | int]] = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        batch_states, batch_programs, batch_lengths, batch_targets = sample_oracle_batch(
            rng=train_rng,
            batch_size=args.batch_size,
            train_min_len=args.train_min_len,
            train_max_len=args.train_max_len,
            size=args.grid_size,
            boundary=args.boundary,
            device=args.device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = decode_after_padded_program(
            model,
            batch_states,
            batch_programs,
            batch_lengths,
            max_len=args.train_max_len,
            device=args.device,
        )
        loss = nn.functional.cross_entropy(logits, batch_targets)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = float((pred == batch_targets).float().mean().detach().cpu())
            history.append({"step": step, "loss": float(loss.detach().cpu()), "batch_accuracy": acc})

    metrics = {
        "one_step": evaluate_one_step(model, size=args.grid_size, boundary=args.boundary, device=args.device),
        "train_lengths_1_3": evaluate_programs(
            model,
            min_len=1,
            max_len=3,
            size=args.grid_size,
            boundary=args.boundary,
            device=args.device,
            seed=args.seed + 11,
            eval_states=args.eval_states,
            max_programs_per_length=args.max_programs_per_length,
        ),
        "length_ood_4_8": evaluate_programs(
            model,
            min_len=4,
            max_len=8,
            size=args.grid_size,
            boundary=args.boundary,
            device=args.device,
            seed=args.seed + 22,
            eval_states=args.eval_states,
            max_programs_per_length=args.max_programs_per_length,
        ),
        "code_transition_consistency": code_transition_consistency(
            model,
            size=args.grid_size,
            device=args.device,
        ),
    }
    payload = {
        "args": vars(args) | {
            "out": str(args.out),
            "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
            "state_autoencoder_checkpoint": str(args.state_autoencoder_checkpoint)
            if args.state_autoencoder_checkpoint
            else None,
        },
        "history": history,
        "runtime_sec": time.time() - start_time,
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": payload["args"],
                "metrics": metrics,
            },
            args.checkpoint_out,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
