#!/usr/bin/env python3
"""Train and evaluate the initial GridWorld reproduction models."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from l2t.gridworld import generate_examples
from l2t.metrics import exact_match_accuracy
from l2t.models import build_model
from l2t.torch_data import GridWorldTorchDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["neo", "neo_s", "disc_mono"], default="neo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--id-eval-size", type=int, default=None)
    parser.add_argument("--comp-eval-size", type=int, default=None)
    parser.add_argument("--length-eval-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--policy-lr-scale", type=float, default=0.25)
    parser.add_argument("--transition-lr-scale", type=float, default=1.0)
    parser.add_argument("--gumbel-tau-start", type=float, default=0.3)
    parser.add_argument("--gumbel-tau-end", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--train-min-len", type=int, default=1)
    parser.add_argument("--train-max-len", type=int, default=3)
    parser.add_argument("--length-eval-min-len", type=int, default=4)
    parser.add_argument("--length-eval-max-len", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-rollout-steps", type=int, default=4)
    parser.add_argument("--eval-rollout-steps", type=int, default=4)
    parser.add_argument("--length-eval-rollout-steps", type=int, default=None)
    parser.add_argument("--neo-s-samples", type=int, default=16)
    parser.add_argument("--mdl-weight", type=float, default=None)
    parser.add_argument("--boundary", choices=["wrap", "clamp"], default="wrap")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_smoke.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(examples, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(GridWorldTorchDataset(examples), batch_size=batch_size, shuffle=shuffle)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def default_mdl_weight(alpha: float) -> float:
    return 1.0 if math.isclose(alpha, 1.0) else 0.95


def cosine_with_warmup(step: int, *, total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> float:
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def evaluate(
    model,
    examples,
    *,
    batch_size: int,
    device: str,
    sample_count: int = 0,
    rollout_steps: int | None = None,
) -> dict[str, float]:
    import torch

    loader = make_loader(examples, batch_size=batch_size, shuffle=False)
    predictions: list[int] = []
    support_predictions: list[int] = []
    targets: list[int] = []
    support_targets: list[int] = []
    lengths: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch["support_x"],
                batch["support_y"],
                batch["query_x"],
                batch["query_y"],
                hard=True,
                sample_count=sample_count,
                rollout_steps=rollout_steps,
            )
            support_pred = output.support_logits.argmax(dim=-1)
            pred = output.query_logits.argmax(dim=-1)
            support_predictions.extend(support_pred.cpu().tolist())
            predictions.extend(pred.cpu().tolist())
            support_targets.extend(batch["support_y"].cpu().tolist())
            targets.extend(batch["query_y"].cpu().tolist())
            lengths.extend(output.chosen_lengths.cpu().tolist())

    return {
        "self_explainability": exact_match_accuracy(support_predictions, support_targets),
        "transfer_accuracy": exact_match_accuracy(predictions, targets),
        "mean_chosen_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def main() -> None:
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.length_eval_rollout_steps is None:
        args.length_eval_rollout_steps = args.max_steps

    set_seed(args.seed)

    train_examples = generate_examples(
        n=args.train_size,
        split="train",
        seed=args.seed,
        alpha=args.alpha,
        size=args.grid_size,
        train_min_len=args.train_min_len,
        train_max_len=args.train_max_len,
        split_seed=args.split_seed,
        use_anchors=not args.no_anchors,
        boundary=args.boundary,
    )
    id_eval_size = args.id_eval_size if args.id_eval_size is not None else args.eval_size
    comp_eval_size = args.comp_eval_size if args.comp_eval_size is not None else args.eval_size
    length_eval_size = args.length_eval_size if args.length_eval_size is not None else args.eval_size
    eval_sets = {
        "id": generate_examples(
            n=id_eval_size,
            split="id",
            seed=args.seed + 101,
            alpha=args.alpha,
            size=args.grid_size,
            train_min_len=args.train_min_len,
            train_max_len=args.train_max_len,
            split_seed=args.split_seed,
            use_anchors=not args.no_anchors,
            boundary=args.boundary,
        ),
        "comp_ood": generate_examples(
            n=comp_eval_size,
            split="comp_ood",
            seed=args.seed + 202,
            alpha=args.alpha,
            size=args.grid_size,
            train_min_len=args.train_min_len,
            train_max_len=args.train_max_len,
            split_seed=args.split_seed,
            use_anchors=not args.no_anchors,
            boundary=args.boundary,
        ),
        "length_ood": generate_examples(
            n=length_eval_size,
            split="length_ood",
            seed=args.seed + 303,
            alpha=args.alpha,
            size=args.grid_size,
            train_min_len=args.train_min_len,
            train_max_len=args.train_max_len,
            length_ood_min_len=args.length_eval_min_len,
            length_ood_max_len=args.length_eval_max_len,
            split_seed=args.split_seed,
            use_anchors=not args.no_anchors,
            boundary=args.boundary,
        ),
    }
    if comp_eval_size == 0:
        eval_sets.pop("comp_ood")

    mdl_weight = args.mdl_weight if args.mdl_weight is not None else default_mdl_weight(args.alpha)

    model_kwargs = {
        "grid_size": args.grid_size,
        "hidden_dim": args.hidden_dim,
        "mdl_weight": mdl_weight,
    }
    if args.model in {"neo", "neo_s"}:
        model_kwargs.update({
            "codebook_size": args.codebook_size or 6,
            "max_steps": args.max_steps,
        })
    else:
        model_kwargs.update({
            "codebook_size": args.codebook_size or 32,
            "max_steps": 1,
        })

    model = build_model(args.model, **model_kwargs).to(args.device)
    policy_params = []
    transition_params = []
    for name, parameter in model.module.named_parameters():
        if name.startswith("programmer."):
            policy_params.append(parameter)
        else:
            transition_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_params, "lr": args.lr * args.policy_lr_scale},
            {"params": transition_params, "lr": args.lr * args.transition_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(train_examples, args.batch_size, shuffle=True)
    train_rollout_steps = 1 if args.model == "disc_mono" else args.train_rollout_steps
    eval_rollout_steps = 1 if args.model == "disc_mono" else args.eval_rollout_steps
    length_eval_rollout_steps = 1 if args.model == "disc_mono" else args.length_eval_rollout_steps

    start_time = time.time()
    step = 0
    last_loss = None
    while step < args.steps:
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["support_x"],
                batch["support_y"],
                batch["query_x"],
                batch["query_y"],
                hard=False,
                rollout_steps=train_rollout_steps,
                gumbel_tau=args.gumbel_tau_start
                + (args.gumbel_tau_end - args.gumbel_tau_start) * min(step / max(args.steps - 1, 1), 1.0),
            )
            output.loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
            optimizer.step()
            lr_scale = cosine_with_warmup(
                step,
                total_steps=args.steps,
                warmup_ratio=args.warmup_ratio,
                min_lr_ratio=args.min_lr_ratio,
            )
            optimizer.param_groups[0]["lr"] = args.lr * args.policy_lr_scale * lr_scale
            optimizer.param_groups[1]["lr"] = args.lr * args.transition_lr_scale * lr_scale
            last_loss = float(output.loss.detach().cpu())
            step += 1
            if step >= args.steps:
                break

    eval_sample_count = args.neo_s_samples if args.model == "neo_s" else 0
    metrics = {
        split: evaluate(
            model,
            examples,
            batch_size=args.batch_size,
            device=args.device,
            sample_count=eval_sample_count,
            rollout_steps=length_eval_rollout_steps if split == "length_ood" else eval_rollout_steps,
        )
        for split, examples in eval_sets.items()
    }
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args) | {
                    "out": str(args.out),
                    "checkpoint_out": str(args.checkpoint_out),
                },
                "model_kwargs": model_kwargs,
                "metrics": metrics,
            },
            args.checkpoint_out,
        )
    payload = {
        "args": vars(args) | {
            "out": str(args.out),
            "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
        },
        "model_kwargs": model_kwargs,
        "effective_mdl_weight": mdl_weight,
        "steps": step,
        "last_train_loss": last_loss,
        "runtime_sec": time.time() - start_time,
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
