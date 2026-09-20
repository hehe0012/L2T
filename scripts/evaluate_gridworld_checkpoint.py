#!/usr/bin/env python3
"""Evaluate one GridWorld checkpoint as NEO and NEO-S."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from l2t.gridworld import generate_examples
from l2t.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gridworld import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--neo-s-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--id-size", type=int, default=10000)
    parser.add_argument("--comp-size", type=int, default=10000)
    parser.add_argument("--length-size", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    saved_args = checkpoint["args"]
    model = build_model(saved_args["model"], **checkpoint["model_kwargs"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size = int(saved_args.get("grid_size", 10))
    common = {
        "alpha": float(saved_args["alpha"]),
        "size": size,
        "train_min_len": int(saved_args.get("train_min_len", 1)),
        "train_max_len": int(saved_args.get("train_max_len", 3)),
        "split_seed": int(saved_args.get("split_seed", 0)),
        "split_mode": saved_args.get("split_mode", "paper"),
        "paper_singleton": saved_args.get("paper_singleton", "U"),
        "use_anchors": not bool(saved_args.get("no_anchors", False)),
        "boundary": saved_args.get("boundary", "wrap"),
    }
    eval_sets = {
        "id": generate_examples(n=args.id_size, split="id", seed=101, **common),
        "comp_ood": generate_examples(n=args.comp_size, split="comp_ood", seed=202, **common),
        "length_ood": generate_examples(
            n=args.length_size,
            split="length_ood",
            seed=303,
            length_ood_min_len=int(saved_args.get("length_eval_min_len", 4)),
            length_ood_max_len=int(saved_args.get("length_eval_max_len", 8)),
            **common,
        ),
    }
    if args.comp_size == 0:
        eval_sets.pop("comp_ood")

    results = {}
    for name, examples in eval_sets.items():
        rollout = int(saved_args.get("length_eval_rollout_steps", 10)) if name == "length_ood" else int(saved_args.get("eval_rollout_steps", 4))
        results[name] = {
            "neo": evaluate(
                model, examples, batch_size=128, device=args.device, grid_size=size,
                sample_count=0, rollout_steps=rollout,
            ),
            "neo_s_b64": evaluate(
                model, examples, batch_size=128, device=args.device, grid_size=size,
                sample_count=args.neo_s_samples, rollout_steps=rollout,
            ),
        }

    payload = {"checkpoint": str(args.checkpoint), "neo_s_samples": args.neo_s_samples, "metrics": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
