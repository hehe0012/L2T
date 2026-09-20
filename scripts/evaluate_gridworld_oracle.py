#!/usr/bin/env python3
"""Evaluate GridWorld oracle displacement transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l2t.gridworld import GridState, generate_examples
from l2t.metrics import exact_match_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--id-eval-size", type=int, default=10000)
    parser.add_argument("--comp-eval-size", type=int, default=None)
    parser.add_argument("--length-eval-size", type=int, default=20000)
    parser.add_argument("--train-min-len", type=int, default=1)
    parser.add_argument("--train-max-len", type=int, default=3)
    parser.add_argument("--length-eval-min-len", type=int, default=4)
    parser.add_argument("--length-eval-max-len", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--boundary", choices=["wrap", "clamp"], default="wrap")
    parser.add_argument("--out", type=Path, default=Path("logs/gridworld_oracle.json"))
    return parser.parse_args()


def predict_from_support(example, *, size: int, boundary: str) -> int:
    support_x = GridState.from_index(example.support_x, size=size)
    support_y = GridState.from_index(example.support_y, size=size)
    query_x = GridState.from_index(example.query_x, size=size)

    if boundary == "wrap":
        d_row = (support_y.row - support_x.row) % size
        d_col = (support_y.col - support_x.col) % size
        return GridState((query_x.row + d_row) % size, (query_x.col + d_col) % size).to_index(size)

    d_row = support_y.row - support_x.row
    d_col = support_y.col - support_x.col
    return GridState(
        min(max(query_x.row + d_row, 0), size - 1),
        min(max(query_x.col + d_col, 0), size - 1),
    ).to_index(size)


def evaluate_split(examples, *, size: int, boundary: str) -> dict[str, float]:
    predictions = [predict_from_support(example, size=size, boundary=boundary) for example in examples]
    targets = [example.query_y for example in examples]
    return {"transfer_accuracy": exact_match_accuracy(predictions, targets)}


def main() -> None:
    args = parse_args()
    comp_eval_size = 0 if args.comp_eval_size is None and args.alpha == 1.0 else args.comp_eval_size
    if comp_eval_size is None:
        comp_eval_size = args.id_eval_size

    common = {
        "alpha": args.alpha,
        "size": args.grid_size,
        "train_min_len": args.train_min_len,
        "train_max_len": args.train_max_len,
        "split_seed": args.split_seed,
        "use_anchors": not args.no_anchors,
        "boundary": args.boundary,
    }
    eval_sets = {
        "id": generate_examples(n=args.id_eval_size, split="id", seed=args.seed + 101, **common),
        "length_ood": generate_examples(
            n=args.length_eval_size,
            split="length_ood",
            seed=args.seed + 303,
            length_ood_min_len=args.length_eval_min_len,
            length_ood_max_len=args.length_eval_max_len,
            **common,
        ),
    }
    if comp_eval_size > 0:
        eval_sets["comp_ood"] = generate_examples(
            n=comp_eval_size,
            split="comp_ood",
            seed=args.seed + 202,
            **common,
        )

    payload = {
        "args": vars(args) | {"out": str(args.out)},
        "metrics": {
            name: evaluate_split(examples, size=args.grid_size, boundary=args.boundary)
            for name, examples in eval_sets.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
