#!/usr/bin/env python3
"""Summarize GridWorld experiment JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/gridworld_full"))
    parser.add_argument("--detail-out", type=Path, default=None)
    parser.add_argument("--mean-out", type=Path, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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

    if args.detail_out is not None:
        write_csv(args.detail_out, rows, columns)

    groups: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["alpha"]), []).append(row)

    mean_rows = []
    metric_columns = [
        "id_self",
        "id_transfer",
        "comp_ood_self",
        "comp_ood_transfer",
        "length_ood_self",
        "length_ood_transfer",
        "runtime_sec",
    ]
    for (model, alpha), group_rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), float(item[0][1]))):
        mean_row: dict[str, object] = {"model": model, "alpha": alpha, "runs": len(group_rows)}
        for column in metric_columns:
            values = [float(row[column]) for row in group_rows if row.get(column, "") != ""]
            mean_row[column] = round(mean(values), 6) if values else ""
        mean_rows.append(mean_row)

    mean_columns = ["model", "alpha", "runs", *metric_columns]
    print()
    print(",".join(mean_columns))
    for row in mean_rows:
        print(",".join(str(row.get(column, "")) for column in mean_columns))

    if args.mean_out is not None:
        write_csv(args.mean_out, mean_rows, mean_columns)


if __name__ == "__main__":
    main()
