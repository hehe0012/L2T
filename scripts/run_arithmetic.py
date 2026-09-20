#!/usr/bin/env python3
"""Train and evaluate Arithmetic Factorization Reasoning models."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from l2t.arithmetic import generate_examples, split_programs
from l2t.arithmetic_models import build_arithmetic_model
from l2t.metrics import exact_match_accuracy
from l2t.torch_data import ArithmeticTorchDataset


PAPER_TRAIN_SIZES = {0.33: 147968, 0.66: 206520, 1.0: 279611}
PAPER_ID_EVAL_SIZES = {0.33: 14796, 0.66: 20651, 1.0: 27961}
PAPER_COMP_EVAL_SIZES = {0.33: 146306, 0.66: 80401, 1.0: 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["neo", "neo_s", "disc_mono"], default="neo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--id-eval-size", type=int, default=None)
    parser.add_argument("--comp-eval-size", type=int, default=None)
    parser.add_argument("--length-eval-size", type=int, default=15317)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gumbel-tau-start", type=float, default=0.3)
    parser.add_argument("--gumbel-tau-end", type=float, default=0.05)
    parser.add_argument("--train-min-len", type=int, default=1)
    parser.add_argument("--train-max-len", type=int, default=3)
    parser.add_argument("--length-eval-min-len", type=int, default=4)
    parser.add_argument("--length-eval-max-len", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--max-start-exp", type=int, default=3)
    parser.add_argument("--max-exponent", type=int, default=12)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--policy-d-model", type=int, default=64)
    parser.add_argument("--policy-ff-dim", type=int, default=256)
    parser.add_argument("--policy-heads", type=int, default=4)
    parser.add_argument("--policy-layers", type=int, default=6)
    parser.add_argument("--transition-d-model", type=int, default=32)
    parser.add_argument("--transition-ff-dim", type=int, default=128)
    parser.add_argument("--transition-heads", type=int, default=2)
    parser.add_argument("--transition-layers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--train-rollout-steps", type=int, default=3)
    parser.add_argument("--eval-rollout-steps", type=int, default=3)
    parser.add_argument("--length-eval-rollout-steps", type=int, default=6)
    parser.add_argument("--neo-s-samples", type=int, default=1024)
    parser.add_argument("--mdl-start", type=float, default=1.01)
    parser.add_argument("--mdl-end", type=float, default=0.99)
    parser.add_argument("--train-query-loss-weight", type=float, default=0.0)
    parser.add_argument("--neo-s-selection", choices=["min_loss", "majority_exact"], default="majority_exact")
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--vq-loss-weight", type=float, default=1.0)
    parser.add_argument("--grounding-loss-weight", type=float, default=0.5)
    parser.add_argument("--mixed-precision", choices=["none", "bf16"], default="none")
    parser.add_argument("--state-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-state", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("logs/arithmetic.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paper_lookup(table: dict[float, int], alpha: float) -> int:
    for key, value in table.items():
        if math.isclose(alpha, key):
            return value
    raise ValueError("paper sizes are defined for alpha 0.33, 0.66, and 1.0")


def make_loader(examples, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(ArithmeticTorchDataset(examples), batch_size=batch_size, shuffle=shuffle)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def cosine_with_warmup(step: int, *, total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> float:
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def mdl_at(epoch: int, total_epochs: int, *, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    progress = epoch / (total_epochs - 1)
    return start + (end - start) * progress


def exponent_predictions(logits) -> list[tuple[int, ...]]:
    return [tuple(row) for row in logits.argmax(dim=-1).cpu().tolist()]


def evaluate(model, examples, *, batch_size: int, device: str, sample_count: int, rollout_steps: int) -> dict[str, float]:
    import torch

    loader = make_loader(examples, batch_size=batch_size, shuffle=False)
    support_predictions: list[tuple[int, ...]] = []
    query_predictions: list[tuple[int, ...]] = []
    support_targets: list[tuple[int, ...]] = []
    query_targets: list[tuple[int, ...]] = []
    lengths: list[int] = []
    exact_candidate_rates: list[float] = []
    fallback_rates: list[float] = []
    majority_counts: list[float] = []

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
            support_predictions.extend(exponent_predictions(output.support_logits))
            query_predictions.extend(exponent_predictions(output.query_logits))
            support_targets.extend(tuple(row) for row in batch["support_y"].cpu().tolist())
            query_targets.extend(tuple(row) for row in batch["query_y"].cpu().tolist())
            lengths.extend(output.chosen_lengths.cpu().tolist())
            if output.exact_candidate_rate is not None:
                exact_candidate_rates.append(float(output.exact_candidate_rate.detach().cpu()))
            if output.fallback_rate is not None:
                fallback_rates.append(float(output.fallback_rate.detach().cpu()))
            if output.majority_count is not None:
                majority_counts.append(float(output.majority_count.detach().cpu()))

    metrics = {
        "self_explainability": exact_match_accuracy(support_predictions, support_targets),
        "transfer_accuracy": exact_match_accuracy(query_predictions, query_targets),
        "mean_chosen_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }
    if exact_candidate_rates:
        metrics["exact_candidate_rate"] = sum(exact_candidate_rates) / len(exact_candidate_rates)
    if fallback_rates:
        metrics["fallback_rate"] = sum(fallback_rates) / len(fallback_rates)
    if majority_counts:
        metrics["mean_majority_count"] = sum(majority_counts) / len(majority_counts)
    return metrics


def load_state_checkpoint(model, path: Path, *, device: str, freeze: bool) -> None:
    import torch

    checkpoint = torch.load(path, map_location=device)
    model.module.state_encoder.load_state_dict(checkpoint["state_encoder"])
    model.module.state_decoder.load_state_dict(checkpoint["state_decoder"])
    if freeze:
        for parameter in model.module.state_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.module.state_decoder.parameters():
            parameter.requires_grad = False


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    for key in ["out", "checkpoint_out", "state_checkpoint"]:
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload


def main() -> None:
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.model == "disc_mono":
        args.train_rollout_steps = 1
        args.eval_rollout_steps = 1
        args.length_eval_rollout_steps = 1

    set_seed(args.seed)
    train_size = args.train_size if args.train_size is not None else paper_lookup(PAPER_TRAIN_SIZES, args.alpha)
    id_eval_size = args.id_eval_size if args.id_eval_size is not None else paper_lookup(PAPER_ID_EVAL_SIZES, args.alpha)
    comp_eval_size = args.comp_eval_size if args.comp_eval_size is not None else paper_lookup(PAPER_COMP_EVAL_SIZES, args.alpha)

    common_data = {
        "alpha": args.alpha,
        "train_min_len": args.train_min_len,
        "train_max_len": args.train_max_len,
        "split_seed": args.split_seed,
        "use_anchors": not args.no_anchors,
        "max_start_exp": args.max_start_exp,
    }
    train_examples = generate_examples(n=train_size, split="train", seed=args.seed, **common_data)
    _, comp_programs = split_programs(
        alpha=args.alpha,
        min_len=args.train_min_len,
        max_len=args.train_max_len,
        seed=args.split_seed,
        use_anchors=not args.no_anchors,
    )
    if not comp_programs:
        comp_eval_size = 0
        args.comp_eval_size = 0

    eval_sets = {
        "id": generate_examples(n=id_eval_size, split="id", seed=args.seed + 101, **common_data),
        "length_ood": generate_examples(
            n=args.length_eval_size,
            split="length_ood",
            seed=args.seed + 303,
            length_ood_min_len=args.length_eval_min_len,
            length_ood_max_len=args.length_eval_max_len,
            **common_data,
        ),
    }
    if comp_eval_size > 0:
        eval_sets["comp_ood"] = generate_examples(n=comp_eval_size, split="comp_ood", seed=args.seed + 202, **common_data)

    model_kwargs = {
        "state_dim": args.state_dim,
        "max_exponent": args.max_exponent,
        "action_dim": args.action_dim,
        "policy_d_model": args.policy_d_model,
        "policy_ff_dim": args.policy_ff_dim,
        "policy_heads": args.policy_heads,
        "policy_layers": args.policy_layers,
        "transition_d_model": args.transition_d_model,
        "transition_ff_dim": args.transition_ff_dim,
        "transition_heads": args.transition_heads,
        "transition_layers": args.transition_layers,
        "max_steps": args.max_steps,
        "mdl_weight": args.mdl_start,
        "query_loss_weight": args.train_query_loss_weight,
        "neo_s_selection": args.neo_s_selection,
        "commitment_cost": args.commitment_cost,
        "vq_loss_weight": args.vq_loss_weight,
        "grounding_loss_weight": args.grounding_loss_weight,
    }
    if args.model in {"neo", "neo_s"}:
        model_kwargs["codebook_size"] = args.codebook_size or 16
    else:
        model_kwargs["codebook_size"] = args.codebook_size or 40
        model_kwargs["max_steps"] = 1
    model = build_arithmetic_model(args.model, **model_kwargs).to(args.device)
    if args.state_checkpoint is not None:
        load_state_checkpoint(model, args.state_checkpoint, device=args.device, freeze=args.freeze_state)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_examples, args.batch_size, shuffle=True)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(
            step,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )

    start_time = time.time()
    last_loss = 0.0
    step = 0
    for epoch in range(args.epochs):
        model.module.outer.mdl_weight = mdl_at(epoch, args.epochs, start=args.mdl_start, end=args.mdl_end)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            tau = args.gumbel_tau_start + (args.gumbel_tau_end - args.gumbel_tau_start) * min(
                step / max(total_steps - 1, 1),
                1.0,
            )
            use_bf16 = args.mixed_precision == "bf16" and args.device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(
                    batch["support_x"],
                    batch["support_y"],
                    batch["query_x"],
                    batch["query_y"],
                    rollout_steps=args.train_rollout_steps,
                    gumbel_tau=tau,
                )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.grad_clip)
            optimizer.step()
            scheduler.step()
            last_loss = float(output.loss.detach().cpu())
            step += 1

    sample_count = args.neo_s_samples if args.model == "neo_s" else 0
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    results = {
        name: evaluate(
            model,
            examples,
            batch_size=eval_batch_size,
            device=args.device,
            sample_count=sample_count,
            rollout_steps=args.length_eval_rollout_steps if name == "length_ood" else args.eval_rollout_steps,
        )
        for name, examples in eval_sets.items()
    }
    payload = {
        "args": serializable_args(args),
        "train_size": train_size,
        "id_eval_size": id_eval_size,
        "comp_eval_size": comp_eval_size,
        "last_train_loss": last_loss,
        "runtime_seconds": time.time() - start_time,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": serializable_args(args)}, args.checkpoint_out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
