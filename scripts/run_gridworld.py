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
    parser.add_argument("--model", choices=["neo", "neo_s", "disc_mono", "cont_mono"], default="neo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--id-eval-size", type=int, default=None)
    parser.add_argument("--comp-eval-size", type=int, default=None)
    parser.add_argument("--length-eval-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None)
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
    parser.add_argument("--split-mode", choices=["paper", "random"], default="paper")
    parser.add_argument("--paper-singleton", choices=["U", "L", "ALL"], default="U")
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--state-path", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--state-dropout", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-rollout-steps", type=int, default=4)
    parser.add_argument("--eval-rollout-steps", type=int, default=4)
    parser.add_argument("--length-eval-rollout-steps", type=int, default=None)
    parser.add_argument("--neo-s-samples", type=int, default=64)
    parser.add_argument("--mdl-weight", type=float, default=None)
    parser.add_argument("--mdl-score-mode", choices=["multiplicative", "additive"], default="multiplicative")
    parser.add_argument("--mdl-lambda", type=float, default=0.0)
    parser.add_argument("--train-query-loss-weight", type=float, default=0.0)
    parser.add_argument("--neo-s-selection", choices=["min_loss", "majority_exact"], default="majority_exact")
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--vq-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-kl-weight", type=float, default=0.01)
    parser.add_argument("--grounding-loss-weight", type=float, default=0.1)
    parser.add_argument("--grounding-transition-only", action="store_true")
    parser.add_argument("--state-autoencoder-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-state-autoencoder", action="store_true")
    parser.add_argument("--boundary", choices=["wrap", "clamp"], default="wrap")
    parser.add_argument("--mixed-precision", choices=["none", "bf16"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_smoke.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--loss-log-interval", type=int, default=1000)
    parser.add_argument("--training-metrics-interval", type=int, default=1)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=1)
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
    grid_size: int,
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
    displacement_matches: list[int] = []
    program_length_errors: list[int] = []
    code_predictions: list[int] = []
    exact_candidate_rates: list[float] = []
    fallback_rates: list[float] = []
    majority_counts: list[float] = []

    def displacement(start, end):
        start_row = start // grid_size
        start_col = start % grid_size
        end_row = end // grid_size
        end_col = end % grid_size
        return torch.stack(((end_row - start_row) % grid_size, (end_col - start_col) % grid_size), dim=-1)

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
            true_disp = displacement(batch["query_x"], batch["query_y"])
            pred_disp = displacement(batch["query_x"], pred)
            displacement_matches.extend((true_disp == pred_disp).all(dim=-1).long().cpu().tolist())
            program_length_errors.extend((output.chosen_lengths - batch["program_len"]).abs().cpu().tolist())
            if output.code_logits is not None:
                code_predictions.extend(output.code_logits.argmax(dim=-1).reshape(-1).cpu().tolist())
            if output.exact_candidate_rate is not None:
                exact_candidate_rates.append(float(output.exact_candidate_rate.detach().cpu()))
            if output.fallback_rate is not None:
                fallback_rates.append(float(output.fallback_rate.detach().cpu()))
            if output.majority_count is not None:
                majority_counts.append(float(output.majority_count.detach().cpu()))

    code_usage_entropy = 0.0
    if code_predictions:
        code_counts: dict[int, int] = {}
        for code in code_predictions:
            code_counts[code] = code_counts.get(code, 0) + 1
        total_codes = sum(code_counts.values())
        code_usage_entropy = -sum(
            (count / total_codes) * math.log(max(count / total_codes, 1e-12))
            for count in code_counts.values()
        )

    metrics = {
        "self_explainability": exact_match_accuracy(support_predictions, support_targets),
        "transfer_accuracy": exact_match_accuracy(predictions, targets),
        "mean_chosen_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "net_displacement_accuracy": sum(displacement_matches) / len(displacement_matches)
        if displacement_matches
        else 0.0,
        "mean_program_length_error": sum(program_length_errors) / len(program_length_errors)
        if program_length_errors
        else 0.0,
        "code_usage_entropy": code_usage_entropy,
    }
    if exact_candidate_rates:
        metrics["exact_candidate_rate"] = sum(exact_candidate_rates) / len(exact_candidate_rates)
    if fallback_rates:
        metrics["fallback_rate"] = sum(fallback_rates) / len(fallback_rates)
    if majority_counts:
        metrics["mean_majority_count"] = sum(majority_counts) / len(majority_counts)
    return metrics


def evaluate_preserving_rng(
    model,
    examples,
    *,
    batch_size: int,
    device: str,
    grid_size: int,
    sample_count: int = 0,
    rollout_steps: int | None = None,
) -> dict[str, float]:
    import torch

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return evaluate(
            model,
            examples,
            batch_size=batch_size,
            device=device,
            grid_size=grid_size,
            sample_count=sample_count,
            rollout_steps=rollout_steps,
        )
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def collect_training_diagnostics(
    model,
    batch,
    output,
    *,
    args,
    step: int,
    epoch: int,
    previous_codebook,
    dead_code_streaks,
) -> tuple[dict[str, object], object, list[int]]:
    """Collect the five training-monitoring groups from test_plan.md.

    This is a no-grad probe on the current training batch. It intentionally
    does not reuse sampled NEO-S candidates: the diagnostics describe the
    learned executor and policy directly.
    """
    import torch
    import torch.nn.functional as F

    module = model.module
    was_training = module.training
    module.eval()
    with torch.no_grad():
        state = model._encode_indices(module, batch["support_x"])
        target = model._encode_indices(module, batch["support_y"])
        rollout_steps = 1 if args.model in {"disc_mono", "cont_mono"} else args.train_rollout_steps
        rollout_steps = min(rollout_steps, args.max_steps)
        codebook_size = int(getattr(model, "codebook_size", 0))
        code_hist = torch.zeros(codebook_size, device=state.device, dtype=torch.long)
        policy_entropies = []
        policy_y_sensitivities = []
        policy_s_sensitivities = []
        rollout_norms = [state.norm(dim=-1).mean()]
        grounding_by_step = []
        reconstruction_by_step = []
        score_losses = []
        executor_z_sensitivities = []
        all_intermediate_states = []
        current = state

        random_target = target[torch.randperm(target.shape[0], device=target.device)]
        random_state = state[torch.randperm(state.shape[0], device=state.device)]
        normal_pre = module.policy(state, target)
        random_y_pre = module.policy(state, random_target)
        random_s_pre = module.policy(random_state, target)
        policy_y_sensitivities.append((normal_pre - random_y_pre).pow(2).mean(dim=-1).sqrt().mean())
        policy_s_sensitivities.append((normal_pre - random_s_pre).pow(2).mean(dim=-1).sqrt().mean())

        for _ in range(rollout_steps):
            action, code_logits, _ = model._action_from_policy(
                module, current, target, hard=True, gumbel_tau=None
            )
            probs = code_logits.softmax(dim=-1)
            policy_entropies.append(-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean())
            codes = code_logits.argmax(dim=-1)
            code_hist += torch.bincount(codes, minlength=codebook_size)

            next_state = model._execute_step(module, current, action)
            all_intermediate_states.append(next_state)
            rollout_norms.append(next_state.norm(dim=-1).mean())
            grounding_by_step.append(
                (next_state - model._encode_decoded_state(module, next_state).detach())
                .pow(2).mean()
            )
            decoded = model._decode_logits(module, next_state)
            reconstruction_by_step.append(
                F.nll_loss(decoded.log_softmax(dim=-1), batch["support_y"])
            )
            score_losses.append(
                F.nll_loss(decoded.log_softmax(dim=-1), batch["support_y"], reduction="none")
            )

            if codebook_size:
                code_actions = module.action_codebook.weight
                expanded_state = current[:, None, :].expand(-1, codebook_size, -1).reshape(
                    -1, current.shape[-1]
                )
                expanded_actions = code_actions[None, :, :].expand(current.shape[0], -1, -1).reshape(
                    -1, code_actions.shape[-1]
                )
                code_outputs = model._execute_step(module, expanded_state, expanded_actions)
                code_outputs = code_outputs.view(current.shape[0], codebook_size, -1)
                # The full pairwise set mixes states; use within-state pairs.
                within = code_outputs[:, :, None, :] - code_outputs[:, None, :, :]
                mask = ~torch.eye(codebook_size, device=state.device, dtype=torch.bool)
                executor_z_sensitivities.append(within.norm(dim=-1)[:, mask].mean())
            current = next_state

        losses = torch.stack(score_losses, dim=1)
        score_curve = model._length_scores(losses, rollout_steps).mean(dim=0)
        chosen_lengths = output.chosen_lengths.float().mean()
        code_probs = code_hist.float() / code_hist.sum().clamp_min(1)
        code_entropy = -(code_probs * code_probs.clamp_min(1e-12).log()).sum()
        if codebook_size:
            normalized = F.normalize(module.action_codebook.weight.detach(), dim=-1)
            cosine = normalized @ normalized.T
            off_diag = cosine[~torch.eye(codebook_size, device=state.device, dtype=torch.bool)]
            codebook_cosine_mean = float(off_diag.mean().cpu())
            codebook_cosine_max = float(off_diag.max().cpu())
            current_codebook = module.action_codebook.weight.detach().clone()
        else:
            codebook_cosine_mean = None
            codebook_cosine_max = None
            current_codebook = previous_codebook
        if codebook_size:
            for index in range(codebook_size):
                dead_code_streaks[index] = dead_code_streaks[index] + 1 if code_hist[index].item() == 0 else 0
        codebook_update = None
        if previous_codebook is not None and current_codebook is not None:
            codebook_update = float((current_codebook - previous_codebook).norm().cpu())
        if was_training:
            module.train()

    record = {
        "step": step,
        "epoch": epoch,
        "codebook": {
            "usage_histogram": code_hist.cpu().tolist(),
            "usage_fraction": code_probs.cpu().tolist(),
            "entropy": float(code_entropy.cpu()),
            "dead_code_count": int((code_hist == 0).sum().cpu()),
            "dead_code_streaks": list(dead_code_streaks),
            "max_dead_code_streak": max(dead_code_streaks, default=0),
            "pairwise_cosine_mean": codebook_cosine_mean,
            "pairwise_cosine_max": codebook_cosine_max,
            "update_norm": codebook_update,
        },
        "policy": {
            "output_entropy": float(torch.stack(policy_entropies).mean().cpu()),
            "y_sensitivity": float(torch.stack(policy_y_sensitivities).mean().cpu()),
            "state_sensitivity": float(torch.stack(policy_s_sensitivities).mean().cpu()),
        },
        "executor": {
            "rollout_latent_norm_by_step": [float(x.cpu()) for x in rollout_norms],
            "grounding_loss_by_step": [float(x.cpu()) for x in grounding_by_step],
            "reconstruction_loss_by_step": [float(x.cpu()) for x in reconstruction_by_step],
            "z_sensitivity_by_step": [float(x.cpu()) for x in executor_z_sensitivities],
        },
        "mdl": {
            "mean_chosen_length": float(chosen_lengths.cpu()),
            "chosen_length_histogram": torch.bincount(
                output.chosen_lengths.detach().long(), minlength=rollout_steps + 1
            ).cpu().tolist(),
            "score_by_length": [float(x.cpu()) for x in score_curve],
        },
        "loss": {
            "total": float(output.loss.detach().cpu()),
            "support": float(output.support_loss.detach().cpu()),
            "vq": float(output.vq_loss.detach().cpu()) if output.vq_loss is not None else 0.0,
            "grounding": float(output.grounding_loss.detach().cpu()) if output.grounding_loss is not None else 0.0,
            "query": float(output.query_loss.detach().cpu()),
            "weighted_support": float(output.support_loss.detach().cpu()),
            "weighted_vq": float(args.vq_loss_weight * output.vq_loss.detach().cpu()) if output.vq_loss is not None else 0.0,
            "weighted_grounding": float(args.grounding_loss_weight * output.grounding_loss.detach().cpu()) if output.grounding_loss is not None else 0.0,
            "weighted_query": float(args.train_query_loss_weight * output.query_loss.detach().cpu()),
            "support_query_ratio": float(
                (output.support_loss.detach() / output.query_loss.detach().clamp_min(1e-12)).cpu()
            ),
        },
    }
    return record, current_codebook, dead_code_streaks


def load_state_autoencoder(model, path: Path, *, device: str, freeze: bool) -> None:
    import torch

    checkpoint = torch.load(path, map_location=device)
    state_encoder = checkpoint.get("state_encoder")
    state_decoder = checkpoint.get("state_decoder")
    if state_encoder is None or state_decoder is None:
        raise ValueError(f"{path} does not contain state_encoder/state_decoder weights")
    model.module.state_encoder.load_state_dict(state_encoder)
    model.module.state_decoder.load_state_dict(state_decoder)
    if freeze:
        for parameter in model.module.state_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.module.state_decoder.parameters():
            parameter.requires_grad = False


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    for key in ["out", "checkpoint_out", "state_autoencoder_checkpoint"]:
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload


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
        split_mode=args.split_mode,
        paper_singleton=args.paper_singleton,
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
            split_mode=args.split_mode,
            paper_singleton=args.paper_singleton,
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
            split_mode=args.split_mode,
            paper_singleton=args.paper_singleton,
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
            split_mode=args.split_mode,
            paper_singleton=args.paper_singleton,
            use_anchors=not args.no_anchors,
            boundary=args.boundary,
        ),
    }
    validation_examples = generate_examples(
        n=args.validation_size,
        split="id",
        seed=args.seed + 404,
        alpha=args.alpha,
        size=args.grid_size,
        train_min_len=args.train_min_len,
        train_max_len=args.train_max_len,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        paper_singleton=args.paper_singleton,
        use_anchors=not args.no_anchors,
        boundary=args.boundary,
    ) if args.validation_size > 0 else []
    if comp_eval_size == 0:
        eval_sets.pop("comp_ood")

    mdl_weight = args.mdl_weight if args.mdl_weight is not None else default_mdl_weight(args.alpha)

    model_kwargs = {
        "grid_size": args.grid_size,
        "hidden_dim": args.hidden_dim,
        "ff_dim": args.ff_dim,
        "action_dim": args.action_dim,
        "state_path": args.state_path,
        "state_dropout": args.state_dropout,
        "mdl_weight": mdl_weight,
        "mdl_score_mode": args.mdl_score_mode,
        "mdl_lambda": args.mdl_lambda,
        "query_loss_weight": args.train_query_loss_weight,
        "neo_s_selection": args.neo_s_selection,
        "grounding_loss_weight": args.grounding_loss_weight,
        "grounding_transition_only": args.grounding_transition_only,
    }
    if args.model in {"neo", "neo_s"}:
        model_kwargs.update({
            "codebook_size": args.codebook_size or 6,
            "max_steps": args.max_steps,
            "commitment_cost": args.commitment_cost,
            "vq_loss_weight": args.vq_loss_weight,
        })
    elif args.model == "disc_mono":
        model_kwargs.update({
            "codebook_size": args.codebook_size or 36,
            "max_steps": 1,
            "grounding_loss_weight": 0.0,
            "commitment_cost": args.commitment_cost,
            "vq_loss_weight": args.vq_loss_weight,
        })
    else:
        model_kwargs.update({
            "max_steps": 1,
            "grounding_loss_weight": 0.0,
            "action_kl_weight": args.action_kl_weight,
        })

    model = build_model(args.model, **model_kwargs).to(args.device)
    if args.state_autoencoder_checkpoint is not None:
        load_state_autoencoder(
            model,
            args.state_autoencoder_checkpoint,
            device=args.device,
            freeze=args.freeze_state_autoencoder,
        )
    policy_params = []
    transition_params = []
    for name, parameter in model.module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("policy.") or name.startswith("action_codebook."):
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
    if args.epochs is not None:
        args.steps = args.epochs * len(train_loader)
    train_rollout_steps = 1 if args.model in {"disc_mono", "cont_mono"} else args.train_rollout_steps
    eval_rollout_steps = 1 if args.model in {"disc_mono", "cont_mono"} else args.eval_rollout_steps
    length_eval_rollout_steps = 1 if args.model in {"disc_mono", "cont_mono"} else args.length_eval_rollout_steps

    start_time = time.time()
    step = 0
    epoch = 0
    last_loss = None
    last_support_loss = None
    last_query_loss = None
    last_vq_loss = None
    last_grounding_loss = None
    loss_history: list[dict[str, float | int | None]] = []
    training_diagnostics_history: list[dict[str, object]] = []
    validation_history: list[dict[str, object]] = []
    best_validation: dict[str, object] | None = None
    previous_codebook = None
    dead_code_streaks = [0] * int(getattr(model, "codebook_size", 0))
    last_training_batch = None
    last_training_output = None
    # NEO-S sampling is an inference-time procedure; ordinary NEO uses the
    # single greedy explanation described in the paper.
    eval_sample_count = args.neo_s_samples if args.model == "neo_s" else 0

    def record_loss_history() -> None:
        if last_loss is None:
            return
        vq_contribution = (
            args.vq_loss_weight * last_vq_loss
            if last_vq_loss is not None
            else None
        )
        grounding_contribution = (
            args.grounding_loss_weight * last_grounding_loss
            if last_grounding_loss is not None
            else None
        )
        query_contribution = args.train_query_loss_weight * last_query_loss if last_query_loss is not None else None
        loss_history.append(
            {
                "step": step,
                "epoch": epoch,
                "total_loss": last_loss,
                "support_loss": last_support_loss,
                "query_loss": last_query_loss,
                "query_contribution": query_contribution,
                "vq_loss": last_vq_loss,
                "vq_contribution": vq_contribution,
                "grounding_loss": last_grounding_loss,
                "grounding_contribution": grounding_contribution,
            }
        )

    while step < args.steps:
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = args.mixed_precision == "bf16" and args.device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
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
            last_training_batch = {key: value.detach() for key, value in batch.items()}
            last_training_output = output
            last_support_loss = float(output.support_loss.detach().cpu())
            last_query_loss = float(output.query_loss.detach().cpu())
            if output.vq_loss is not None:
                last_vq_loss = float(output.vq_loss.detach().cpu())
            if output.grounding_loss is not None:
                last_grounding_loss = float(output.grounding_loss.detach().cpu())
            step += 1
            if args.loss_log_interval > 0 and step % args.loss_log_interval == 0:
                record_loss_history()
            if step >= args.steps:
                break
        epoch += 1
        record_loss_history()
        if (
            args.training_metrics_interval > 0
            and epoch % args.training_metrics_interval == 0
            and last_training_batch is not None
            and last_training_output is not None
        ):
            diagnostic, previous_codebook, dead_code_streaks = collect_training_diagnostics(
                model,
                last_training_batch,
                last_training_output,
                args=args,
                step=step,
                epoch=epoch,
                previous_codebook=previous_codebook,
                dead_code_streaks=dead_code_streaks,
            )
            training_diagnostics_history.append(diagnostic)
        if (
            validation_examples
            and args.validation_interval > 0
            and epoch % args.validation_interval == 0
        ):
            validation_metrics = evaluate_preserving_rng(
                model,
                validation_examples,
                batch_size=args.batch_size,
                device=args.device,
                grid_size=args.grid_size,
                sample_count=eval_sample_count,
                rollout_steps=eval_rollout_steps,
            )
            validation_record = {
                "epoch": epoch,
                "step": step,
                "metrics": validation_metrics,
            }
            validation_history.append(validation_record)
            if best_validation is None or validation_metrics["transfer_accuracy"] > best_validation["metrics"]["transfer_accuracy"]:
                best_validation = validation_record

    if validation_examples and (not validation_history or validation_history[-1]["step"] != step):
        validation_metrics = evaluate_preserving_rng(
            model,
            validation_examples,
            batch_size=args.batch_size,
            device=args.device,
            grid_size=args.grid_size,
            sample_count=eval_sample_count,
            rollout_steps=eval_rollout_steps,
        )
        validation_record = {
            "epoch": epoch,
            "step": step,
            "metrics": validation_metrics,
        }
        validation_history.append(validation_record)
        if best_validation is None or validation_metrics["transfer_accuracy"] > best_validation["metrics"]["transfer_accuracy"]:
            best_validation = validation_record

    metrics = {
        split: evaluate(
            model,
            examples,
            batch_size=args.batch_size,
            device=args.device,
            grid_size=args.grid_size,
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
                "args": serializable_args(args),
                "model_kwargs": model_kwargs,
                "metrics": metrics,
                "loss_history": loss_history,
                "training_diagnostics_history": training_diagnostics_history,
                "validation_history": validation_history,
                "best_validation": best_validation,
            },
            args.checkpoint_out,
        )
    payload = {
        "args": serializable_args(args),
        "model_kwargs": model_kwargs,
        "effective_mdl_weight": mdl_weight,
        "steps": step,
        "last_train_loss": last_loss,
        "last_support_loss": last_support_loss,
        "last_query_loss": last_query_loss,
        "last_vq_loss": last_vq_loss,
        "last_grounding_loss": last_grounding_loss,
        "loss_history": loss_history,
        "training_diagnostics_history": training_diagnostics_history,
        "validation_history": validation_history,
        "best_validation": best_validation,
        "runtime_sec": time.time() - start_time,
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
