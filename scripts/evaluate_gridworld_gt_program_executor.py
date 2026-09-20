#!/usr/bin/env python3
"""Evaluate a trained executor with ground-truth primitive programs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action, generate_examples
from l2t.models import build_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.33)
    parser.add_argument("--id-size", type=int, default=10000)
    parser.add_argument("--comp-size", type=int, default=10000)
    parser.add_argument("--length-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(checkpoint["args"]["model"], **checkpoint["model_kwargs"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size = int(checkpoint["args"].get("grid_size", 10))
    codebook_size = model.codebook_size

    states = torch.arange(size * size, dtype=torch.long, device=args.device)
    source = model._encode_indices(model.module, states)
    with torch.no_grad():
        code_ids = torch.arange(codebook_size, device=args.device)
        source_expanded = source[:, None, :].expand(-1, codebook_size, -1).reshape(-1, source.shape[-1])
        code_expanded = code_ids[None, :].expand(states.numel(), -1).reshape(-1)
        next_latent = model._execute_step(model.module, source_expanded, model.module.action_codebook(code_expanded))
        predicted = model._decode_logits(model.module, next_latent).argmax(dim=-1).view(size * size, codebook_size)

    primitive_targets = []
    accuracy = []
    for action in range(len(ACTIONS)):
        target = torch.tensor(
            [apply_action(GridState.from_index(int(s), size=size), action, size=size).to_index(size) for s in states.cpu()],
            dtype=torch.long,
            device=args.device,
        )
        primitive_targets.append(target.cpu().tolist())
        accuracy.append([(predicted[:, code] == target).float().mean().item() for code in range(codebook_size)])

    best_total = -1.0
    best_mapping = None
    for selected_codes in itertools.permutations(range(codebook_size), len(ACTIONS)):
        total = sum(accuracy[action][code] for action, code in enumerate(selected_codes))
        if total > best_total:
            best_total = total
            best_mapping = selected_codes
    assert best_mapping is not None
    mapping = {ACTIONS[action]: int(best_mapping[action]) for action in range(len(ACTIONS))}

    def evaluate_split(split: str, count: int, seed: int) -> dict[str, float]:
        examples = generate_examples(
            n=count,
            split=split,
            seed=seed,
            alpha=args.alpha,
            size=size,
            split_mode="paper",
        )
        correct_query = 0
        correct_support = 0
        total = 0
        length_correct: dict[int, list[int]] = {}
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            support_x = torch.tensor([e.support_x for e in batch], dtype=torch.long, device=args.device)
            query_x = torch.tensor([e.query_x for e in batch], dtype=torch.long, device=args.device)
            support_y = torch.tensor([e.support_y for e in batch], dtype=torch.long, device=args.device)
            query_y = torch.tensor([e.query_y for e in batch], dtype=torch.long, device=args.device)
            max_len = max(len(e.program) for e in batch)
            with torch.no_grad():
                support_state = model._encode_indices(model.module, support_x)
                query_state = model._encode_indices(model.module, query_x)
                for step in range(max_len):
                    active = torch.tensor(
                        [mapping[ACTIONS[e.program[step]]] if step < len(e.program) else mapping[ACTIONS[0]] for e in batch],
                        dtype=torch.long,
                        device=args.device,
                    )
                    action = model.module.action_codebook(active)
                    active_mask = torch.tensor([step < len(e.program) for e in batch], device=args.device).view(-1, 1)
                    support_next = model._execute_step(model.module, support_state, action)
                    query_next = model._execute_step(model.module, query_state, action)
                    support_state = torch.where(active_mask, support_next, support_state)
                    query_state = torch.where(active_mask, query_next, query_state)
                support_pred = model._decode_logits(model.module, support_state).argmax(dim=-1)
                query_pred = model._decode_logits(model.module, query_state).argmax(dim=-1)
            support_ok = support_pred == support_y
            query_ok = query_pred == query_y
            correct_support += int(support_ok.sum())
            correct_query += int(query_ok.sum())
            total += len(batch)
            for e, ok in zip(batch, query_ok.cpu().tolist()):
                length_correct.setdefault(len(e.program), []).append(int(ok))
        return {
            "support_accuracy": correct_support / total if total else 0.0,
            "query_transfer_accuracy": correct_query / total if total else 0.0,
            "examples": total,
            **{f"length_{length}_accuracy": sum(values) / len(values) for length, values in sorted(length_correct.items())},
        }

    metrics = {
        "id": evaluate_split("id", args.id_size, 1001),
        "comp_ood": evaluate_split("comp_ood", args.comp_size, 1002),
        "length_ood": evaluate_split("length_ood", args.length_size, 1003),
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "code_to_primitive_accuracy": {ACTIONS[a]: accuracy[a] for a in range(len(ACTIONS))},
        "primitive_to_code_mapping": mapping,
        "mapping_mean_one_step_accuracy": best_total / len(ACTIONS),
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
