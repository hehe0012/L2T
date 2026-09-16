#!/usr/bin/env python3
"""Summarize GridWorld experiment JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/gridworld_full"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.log_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_args = payload["args"]
        row = {
            "file": path.name,
            "model": run_args["model"],
            "alpha": run_args["alpha"],
            "seed": run_args["seed"],
            "train_lengths": f"{run_args.get('train_min_len', 1)}-{run_args.get('train_max_len', 3)}",
            "eval_lengths": f"{run_args.get('length_eval_min_len', 4)}-{run_args.get('length_eval_max_len', 8)}",
            "steps": payload["steps"],
            "runtime_sec": round(payload["runtime_sec"], 2),
        }
        for split, metrics in payload["metrics"].items():
            row[f"{split}_self"] = metrics["self_explainability"]
            row[f"{split}_transfer"] = metrics["transfer_accuracy"]
            row[f"{split}_len"] = metrics["mean_chosen_length"]
        rows.append(row)

    if not rows:
        print(f"no JSON files found in {args.log_dir}")
        return

    columns = [
        "model",
        "alpha",
        "seed",
        "train_lengths",
        "eval_lengths",
        "steps",
        "id_self",
        "id_transfer",
        "comp_ood_self",
        "comp_ood_transfer",
        "length_ood_self",
        "length_ood_transfer",
        "runtime_sec",
        "file",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row.get(column, "")) for column in columns))


if __name__ == "__main__":
    main()
