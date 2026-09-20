#!/usr/bin/env python3
"""Diagnose why GridWorld support explanations do not transfer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from l2t.gridworld import GridState, generate_examples
from l2t.models import build_model
from l2t.torch_data import GridWorldTorchDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "comp_ood", "length_ood"], default="id")
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_diagnosis.json"))
    return parser.parse_args()


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def displacement(start: int, end: int, *, size: int) -> tuple[int, int]:
    start_state = GridState.from_index(int(start), size=size)
    end_state = GridState.from_index(int(end), size=size)
    return ((end_state.row - start_state.row) % size, (end_state.col - start_state.col) % size)


def add_disp(left: tuple[int, int], right: tuple[int, int], *, size: int) -> tuple[int, int]:
    return ((left[0] + right[0]) % size, (left[1] + right[1]) % size)


def primitive_name(disp: tuple[int, int], *, size: int) -> str:
    mapping = {
        ((-1) % size, 0): "U",
        (1, 0): "D",
        (0, (-1) % size): "L",
        (0, 1): "R",
    }
    return mapping.get(disp, str(disp))


def load_model(path: Path, *, device: str):
    import torch

    checkpoint = torch.load(path, map_location=device)
    model_kwargs = dict(checkpoint["model_kwargs"])
    model = build_model(checkpoint["args"]["model"], **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def code_transition_consistency(model, *, size: int, device: str) -> tuple[list[dict[str, object]], dict[int, tuple[int, int]]]:
    import torch

    states = torch.arange(size * size, dtype=torch.long, device=device)
    state_latent = model._encode_indices(model.module, states)
    rows: list[dict[str, object]] = []
    dominant: dict[int, tuple[int, int]] = {}
    with torch.no_grad():
        for code in range(model.codebook_size):
            action = model.module.action_codebook(torch.full((states.shape[0],), code, dtype=torch.long, device=device))
            next_state = model._execute_step(model.module, state_latent, action)
            pred = model._decode_logits(model.module, next_state).argmax(dim=-1).cpu().tolist()
            counts = Counter(displacement(start, end, size=size) for start, end in zip(states.cpu().tolist(), pred))
            best_disp, best_count = counts.most_common(1)[0]
            dominant[code] = best_disp
            rows.append(
                {
                    "code": code,
                    "dominant_displacement": list(best_disp),
                    "dominant_name": primitive_name(best_disp, size=size),
                    "consistency": best_count / (size * size),
                    "unique_displacements": len(counts),
                    "top_displacements": [
                        {
                            "displacement": list(disp),
                            "name": primitive_name(disp, size=size),
                            "count": count,
                        }
                        for disp, count in counts.most_common(5)
                    ],
                }
            )
    return rows, dominant


def select_sampled_programs(model, batch, *, sample_count: int, rollout_steps: int):
    """Mirror NEO-S sampled selection while returning selected code programs."""

    import torch
    from torch import nn

    support_x = batch["support_x"]
    support_y = batch["support_y"]
    query_x = batch["query_x"]
    batch_size = support_x.shape[0]
    candidate_count = sample_count + 1

    flat_support_x = support_x[:, None].expand(batch_size, candidate_count).reshape(-1)
    flat_query_x = query_x[:, None].expand(batch_size, candidate_count).reshape(-1)
    flat_support_y = support_y[:, None].expand(batch_size, candidate_count).reshape(-1)

    support_state = model._encode_indices(model.module, flat_support_x)
    query_state = model._encode_indices(model.module, flat_query_x)
    target_state = model._encode_indices(model.module, flat_support_y)
    support_logits_by_step = []
    query_logits_by_step = []
    step_codes_by_step = []

    for _ in range(rollout_steps):
        action_pre = model.module.policy(support_state, target_state)
        distances = torch.cdist(action_pre[:, None, :], model.module.action_codebook.weight[None, :, :]).squeeze(1)
        step_logits = -distances
        step_logits_view = step_logits.view(batch_size, candidate_count, model.codebook_size)
        greedy_codes = step_logits_view[:, :1].argmax(dim=-1)
        sampled_codes = torch.multinomial(
            step_logits_view[:, 1:].reshape(batch_size * sample_count, model.codebook_size).softmax(dim=-1),
            num_samples=1,
            replacement=True,
        ).view(batch_size, sample_count)
        step_codes = torch.cat([greedy_codes, sampled_codes], dim=1).reshape(-1)
        action = model.module.action_codebook(step_codes)
        support_state = model._execute_step(model.module, support_state, action)
        query_state = model._execute_step(model.module, query_state, action)
        support_logits_by_step.append(model._decode_logits(model.module, support_state).log_softmax(dim=-1))
        query_logits_by_step.append(model._decode_logits(model.module, query_state).log_softmax(dim=-1))
        step_codes_by_step.append(step_codes.view(batch_size, candidate_count))

    support_losses = torch.stack(
        [nn.functional.nll_loss(logits, flat_support_y, reduction="none") for logits in support_logits_by_step],
        dim=1,
    ).view(batch_size, candidate_count, rollout_steps)
    scores = model._length_scores(support_losses, rollout_steps)
    best_flat = scores.reshape(batch_size, candidate_count * rollout_steps).argmin(dim=1)
    fallback_candidate = best_flat // rollout_steps
    fallback_length = best_flat % rollout_steps

    support_logits_stacked = torch.stack(support_logits_by_step, dim=1).view(
        batch_size, candidate_count, rollout_steps, model.num_states
    )
    query_logits_stacked = torch.stack(query_logits_by_step, dim=1).view(
        batch_size, candidate_count, rollout_steps, model.num_states
    )
    exact_mask = support_logits_stacked.argmax(dim=-1) == support_y[:, None, None]
    step_codes_stacked = torch.stack(step_codes_by_step, dim=2)

    selected_candidate = []
    selected_length = []
    selected_programs = []
    fallback_flags = []
    majority_counts = []
    for batch_index in range(batch_size):
        exact_positions = exact_mask[batch_index].nonzero(as_tuple=False)
        if exact_positions.numel() == 0:
            candidate = int(fallback_candidate[batch_index].item())
            length = int(fallback_length[batch_index].item())
            selected_candidate.append(candidate)
            selected_length.append(length)
            selected_programs.append(tuple(int(x) for x in step_codes_stacked[batch_index, candidate, : length + 1].cpu().tolist()))
            fallback_flags.append(1)
            majority_counts.append(0)
            continue

        counts: dict[tuple[int, ...], int] = {}
        first_position: dict[tuple[int, ...], tuple[int, int]] = {}
        for candidate_tensor, length_tensor in exact_positions.tolist():
            program = tuple(
                int(code)
                for code in step_codes_stacked[batch_index, candidate_tensor, : length_tensor + 1].cpu().tolist()
            )
            counts[program] = counts.get(program, 0) + 1
            first_position.setdefault(program, (candidate_tensor, length_tensor))

        best_program = max(counts, key=lambda program: (counts[program], -len(program)))
        candidate, length = first_position[best_program]
        selected_candidate.append(candidate)
        selected_length.append(length)
        selected_programs.append(best_program)
        fallback_flags.append(0)
        majority_counts.append(counts[best_program])

    row_index = torch.arange(batch_size, device=support_x.device)
    selected_candidate_t = torch.tensor(selected_candidate, dtype=torch.long, device=support_x.device)
    selected_length_t = torch.tensor(selected_length, dtype=torch.long, device=support_x.device)
    support_pred = support_logits_stacked[row_index, selected_candidate_t, selected_length_t].argmax(dim=-1)
    query_pred = query_logits_stacked[row_index, selected_candidate_t, selected_length_t].argmax(dim=-1)
    return selected_programs, support_pred, query_pred, fallback_flags, majority_counts


def diagnose_selection(model, checkpoint, dominant_code_disp, args) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    run_args = checkpoint["args"]
    eval_seed_offset = {"id": 101, "comp_ood": 202, "length_ood": 303}[args.split]
    examples = generate_examples(
        n=args.eval_size,
        split=args.split,
        seed=run_args["seed"] + eval_seed_offset,
        alpha=run_args["alpha"],
        size=run_args["grid_size"],
        train_min_len=run_args["train_min_len"],
        train_max_len=run_args["train_max_len"],
        length_ood_min_len=run_args["length_eval_min_len"],
        length_ood_max_len=run_args["length_eval_max_len"],
        split_seed=run_args["split_seed"],
        use_anchors=not run_args["no_anchors"],
        boundary=run_args["boundary"],
        split_mode=run_args["split_mode"],
    )
    loader = DataLoader(GridWorldTorchDataset(examples), batch_size=args.batch_size, shuffle=False)
    rollout_steps = args.rollout_steps
    if rollout_steps is None:
        rollout_steps = run_args["length_eval_rollout_steps"] if args.split == "length_ood" else run_args["eval_rollout_steps"]
    sample_count = args.sample_count if args.sample_count is not None else run_args["neo_s_samples"]
    size = run_args["grid_size"]

    totals = defaultdict(int)
    program_counter: Counter[tuple[int, ...]] = Counter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, args.device)
            programs, support_pred, query_pred, fallback_flags, majority_counts = select_sampled_programs(
                model,
                batch,
                sample_count=sample_count,
                rollout_steps=rollout_steps,
            )
            for index, program in enumerate(programs):
                support_x = int(batch["support_x"][index].item())
                support_y = int(batch["support_y"][index].item())
                query_x = int(batch["query_x"][index].item())
                query_y = int(batch["query_y"][index].item())
                true_disp = displacement(support_x, support_y, size=size)
                query_true_disp = displacement(query_x, query_y, size=size)
                pred_disp = displacement(query_x, int(query_pred[index].item()), size=size)
                support_pred_disp = displacement(support_x, int(support_pred[index].item()), size=size)
                code_disp = (0, 0)
                for code in program:
                    code_disp = add_disp(code_disp, dominant_code_disp[code], size=size)

                totals["n"] += 1
                totals["support_exact"] += int(int(support_pred[index].item()) == support_y)
                totals["query_exact"] += int(int(query_pred[index].item()) == query_y)
                totals["support_disp_exact"] += int(support_pred_disp == true_disp)
                totals["query_disp_exact"] += int(pred_disp == query_true_disp)
                totals["code_semantic_disp_exact"] += int(code_disp == true_disp)
                totals["fallback"] += int(fallback_flags[index])
                totals["majority_count_sum"] += int(majority_counts[index])
                totals["length_abs_error_sum"] += abs(len(program) - int(batch["program_len"][index].item()))
                program_counter[program] += 1

    n = max(totals["n"], 1)
    return {
        "split": args.split,
        "eval_size": totals["n"],
        "sample_count": sample_count,
        "rollout_steps": rollout_steps,
        "support_exact": totals["support_exact"] / n,
        "query_exact": totals["query_exact"] / n,
        "support_displacement_exact": totals["support_disp_exact"] / n,
        "query_displacement_exact": totals["query_disp_exact"] / n,
        "code_semantic_displacement_exact": totals["code_semantic_disp_exact"] / n,
        "fallback_rate": totals["fallback"] / n,
        "mean_majority_count": totals["majority_count_sum"] / n,
        "mean_length_abs_error": totals["length_abs_error_sum"] / n,
        "unique_selected_programs": len(program_counter),
        "top_selected_programs": [
            {"program": list(program), "count": count}
            for program, count in program_counter.most_common(10)
        ],
    }


def main() -> None:
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(0)

    model, checkpoint = load_model(args.checkpoint, device=args.device)
    code_rows, dominant = code_transition_consistency(
        model,
        size=checkpoint["args"]["grid_size"],
        device=args.device,
    )
    selection = diagnose_selection(model, checkpoint, dominant, args)
    payload = {
        "checkpoint": str(args.checkpoint),
        "run_args": checkpoint["args"],
        "code_transition_consistency": code_rows,
        "selection_diagnosis": selection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
