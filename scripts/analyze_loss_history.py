#!/usr/bin/env python3
"""Aggregate and plot weighted loss terms from a run_gridworld JSON log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "support_loss",
    "query_contribution",
    "vq_contribution",
    "grounding_contribution",
    "total_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--svg-out", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(history: list[dict[str, object]], bins: int) -> list[dict[str, float | int | str]]:
    if not history:
        return []
    max_step = max(int(row["step"]) for row in history)
    rows: list[dict[str, float | int | str]] = []
    for bin_index in range(bins):
        start = int(max_step * bin_index / bins)
        end = int(max_step * (bin_index + 1) / bins)
        if bin_index == 0:
            selected = [row for row in history if 0 <= int(row["step"]) <= end]
        else:
            selected = [row for row in history if start < int(row["step"]) <= end]
        out: dict[str, float | int | str] = {
            "period": f"{bin_index * 10:02d}-{(bin_index + 1) * 10:02d}%",
            "step_start": start + (0 if bin_index == 0 else 1),
            "step_end": end,
            "n": len(selected),
        }
        for field in FIELDS:
            out[field] = mean([
                float(row[field])
                for row in selected
                if row.get(field) is not None
            ])
        out["weighted_sum"] = (
            float(out["support_loss"])
            + float(out["query_contribution"])
            + float(out["vq_contribution"])
            + float(out["grounding_contribution"])
        )
        rows.append(out)
    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "period",
        "step_start",
        "step_end",
        "n",
        "support_loss",
        "query_contribution",
        "vq_contribution",
        "grounding_contribution",
        "weighted_sum",
        "total_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_bar_chart(rows: list[dict[str, float | int | str]], path: Path) -> None:
    width = 1100
    height = 620
    margin_left = 78
    margin_right = 36
    margin_top = 64
    margin_bottom = 94
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    components = [
        ("support_loss", "#3b82f6", "support"),
        ("query_contribution", "#94a3b8", "query weighted"),
        ("vq_contribution", "#f59e0b", "vq weighted"),
        ("grounding_contribution", "#10b981", "grounding weighted"),
    ]
    max_y = max(
        max(float(row["weighted_sum"]), float(row["total_loss"]))
        for row in rows
    )
    max_y = max_y * 1.12 if max_y > 0 else 1.0
    bar_gap = 12
    bar_w = (plot_w - bar_gap * (len(rows) + 1)) / len(rows)

    def x_at(index: int) -> float:
        return margin_left + bar_gap + index * (bar_w + bar_gap)

    def y_at(value: float) -> float:
        return margin_top + plot_h - value / max_y * plot_h

    def h_of(value: float) -> float:
        return value / max_y * plot_h

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append('<text x="48" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Weighted loss by training period</text>')
    parts.append('<text x="48" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">GridWorld NEO-S alpha=0.33 seed=0; 10 bins over 78,200 steps</text>')

    for tick in range(6):
        value = max_y * tick / 5
        y = y_at(value)
        parts.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{value:.2f}</text>')

    parts.append(f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{margin_top + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{margin_top + plot_h}" y2="{margin_top + plot_h}" stroke="#111827"/>')

    for i, row in enumerate(rows):
        x = x_at(i)
        bottom = margin_top + plot_h
        for field, color, _label in components:
            value = float(row[field])
            h = h_of(value)
            y = bottom - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
            bottom = y
        cx = x + bar_w / 2
        total_y = y_at(float(row["total_loss"]))
        if i > 0:
            prev = rows[i - 1]
            prev_cx = x_at(i - 1) + bar_w / 2
            prev_y = y_at(float(prev["total_loss"]))
            parts.append(f'<line x1="{prev_cx:.2f}" y1="{prev_y:.2f}" x2="{cx:.2f}" y2="{total_y:.2f}" stroke="#ef4444" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{cx:.2f}" cy="{total_y:.2f}" r="4" fill="#ef4444"/>')
        parts.append(f'<text x="{cx:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{row["period"]}</text>')

    legend_x = margin_left
    legend_y = height - 40
    for j, (_field, color, label) in enumerate(components):
        x = legend_x + j * 170
        parts.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{label}</text>')
    x = legend_x + len(components) * 170
    parts.append(f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 22}" y2="{legend_y - 4}" stroke="#ef4444" stroke-width="2.4"/>')
    parts.append(f'<circle cx="{x + 11}" cy="{legend_y - 4}" r="4" fill="#ef4444"/>')
    parts.append(f'<text x="{x + 30}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#374151">recorded total</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = aggregate(payload.get("loss_history", []), args.bins)
    write_csv(rows, args.csv_out)
    svg_bar_chart(rows, args.svg_out)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.svg_out}")


if __name__ == "__main__":
    main()
