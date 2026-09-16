#!/usr/bin/env python3
"""Run short-to-long and long-to-short GridWorld length-transfer sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REGIMES = {
    "standard_short_to_long": {
        "train_min_len": 1,
        "train_max_len": 3,
        "length_eval_min_len": 4,
        "length_eval_max_len": 8,
    },
    "reverse_long_to_short": {
        "train_min_len": 4,
        "train_max_len": 8,
        "length_eval_min_len": 1,
        "length_eval_max_len": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("logs/gridworld_length_direction"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["neo", "neo_s", "disc_mono"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.66, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--neo-s-samples", nargs="+", type=int, default=[16])
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--mdl-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sample_counts_for_model(model: str, values: list[int]) -> list[int]:
    return values if model == "neo_s" else [0]


def run_name(regime: str, model: str, alpha: float, seed: int, sample_count: int) -> str:
    suffix = f"_samples{sample_count}" if model == "neo_s" else ""
    return f"{regime}_{model}_alpha{alpha:.2f}_seed{seed}{suffix}"


def build_command(args: argparse.Namespace, regime: str, model: str, alpha: float, seed: int, sample_count: int, out: Path) -> list[str]:
    lengths = REGIMES[regime]
    cmd = [
        sys.executable,
        "scripts/run_gridworld.py",
        "--model",
        model,
        "--seed",
        str(seed),
        "--alpha",
        str(alpha),
        "--train-size",
        str(args.train_size),
        "--eval-size",
        str(args.eval_size),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--grid-size",
        str(args.grid_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--max-steps",
        str(args.max_steps),
        "--mdl-weight",
        str(args.mdl_weight),
        "--train-min-len",
        str(lengths["train_min_len"]),
        "--train-max-len",
        str(lengths["train_max_len"]),
        "--length-eval-min-len",
        str(lengths["length_eval_min_len"]),
        "--length-eval-max-len",
        str(lengths["length_eval_max_len"]),
        "--neo-s-samples",
        str(sample_count or args.neo_s_samples[0]),
        "--device",
        args.device,
        "--out",
        str(out),
    ]
    return cmd


def collect_rows(out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_args = payload["args"]
        row: dict[str, object] = {
            "file": path.name,
            "regime": path.stem.split("_neo", 1)[0]
            if "_neo" in path.stem
            else path.stem.split("_disc_mono", 1)[0],
            "model": run_args["model"],
            "alpha": run_args["alpha"],
            "seed": run_args["seed"],
            "neo_s_samples": run_args["neo_s_samples"] if run_args["model"] == "neo_s" else "",
            "train_lengths": f"{run_args['train_min_len']}-{run_args['train_max_len']}",
            "eval_lengths": f"{run_args['length_eval_min_len']}-{run_args['length_eval_max_len']}",
            "steps": payload["steps"],
            "runtime_sec": round(payload["runtime_sec"], 2),
        }
        for split, metrics in payload["metrics"].items():
            row[f"{split}_self"] = metrics["self_explainability"]
            row[f"{split}_transfer"] = metrics["transfer_accuracy"]
            row[f"{split}_len"] = metrics["mean_chosen_length"]
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "regime",
        "model",
        "alpha",
        "seed",
        "neo_s_samples",
        "train_lengths",
        "eval_lengths",
        "steps",
        "id_self",
        "id_transfer",
        "comp_ood_self",
        "comp_ood_transfer",
        "length_ood_self",
        "length_ood_transfer",
        "length_ood_len",
        "runtime_sec",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = args.summary_out or args.out_dir.with_name(f"{args.out_dir.name}_summary.csv")

    for regime in REGIMES:
        for model in args.models:
            for alpha in args.alphas:
                for seed in args.seeds:
                    for sample_count in sample_counts_for_model(model, args.neo_s_samples):
                        name = run_name(regime, model, alpha, seed, sample_count)
                        out = args.out_dir / f"{name}.json"
                        cmd = build_command(args, regime, model, alpha, seed, sample_count, out)
                        if out.exists() and not args.force:
                            print(f"skip existing {out}")
                            continue
                        print(" ".join(cmd), flush=True)
                        if not args.dry_run:
                            subprocess.run(cmd, check=True)

    if not args.dry_run:
        rows = collect_rows(args.out_dir)
        write_summary(rows, summary_out)
        print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
