#!/usr/bin/env python3
"""Generate dependency-free SVG charts for test_plan.md diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COLORS = {0.33: "#2563eb", 0.66: "#059669", 1.0: "#dc2626"}
LABELS = {0.33: "alpha=0.33", 0.66: "alpha=0.66", 1.0: "alpha=1.00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart_svg(title, panels, width=1800, panel_w=570, panel_h=300):
    rows = math.ceil(len(panels) / 3)
    height = 70 + rows * panel_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{esc(title)}</text>',
    ]
    for index, panel in enumerate(panels):
        row, col = divmod(index, 3)
        x0, y0 = col * panel_w + 18, 70 + row * panel_h
        left, top, pw, ph = x0 + 62, y0 + 40, panel_w - 92, panel_h - 82
        title_text, series, x_label = panel
        all_values = [value for _, values in series for value in values if value is not None and math.isfinite(value)]
        if not all_values:
            continue
        ymin, ymax = min(all_values), max(all_values)
        if abs(ymax - ymin) < 1e-9:
            ymin -= 1.0
            ymax += 1.0
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad
        parts.append(f'<text x="{x0+10}" y="{y0+22}" font-family="Arial" font-size="16" font-weight="bold">{esc(title_text)}</text>')
        for grid in range(5):
            gy = top + ph * grid / 4
            value = ymax - (ymax - ymin) * grid / 4
            parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left+pw}" y2="{gy:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{left-8}" y="{gy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#4b5563">{value:.3g}</text>')
        max_len = max(len(values) for _, values in series)
        for label, values in series:
            points = []
            for i, value in enumerate(values):
                if value is None:
                    continue
                px = left + pw * (i / max(1, max_len - 1))
                py = top + ph * (ymax - value) / (ymax - ymin)
                points.append(f"{px:.1f},{py:.1f}")
            color = next((COLORS[a] for a in COLORS if LABELS[a] == label), "#111827")
            if len(points) > 1:
                parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#374151"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#374151"/>')
        parts.append(f'<text x="{left+pw/2:.1f}" y="{top+ph+28}" text-anchor="middle" font-family="Arial" font-size="11">{esc(x_label)}</text>')
        for legend_index, label in enumerate(sorted({label for label, _ in series})):
            color = next((COLORS[a] for a in COLORS if LABELS[a] == label), "#111827")
            lx = left + legend_index * 115
            parts.append(f'<line x1="{lx}" y1="{y0+panel_h-16}" x2="{lx+18}" y2="{y0+panel_h-16}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{lx+23}" y="{y0+panel_h-12}" font-family="Arial" font-size="10">{esc(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.json"))
    if len(files) != 3:
        raise SystemExit(f"expected 3 result JSON files, found {len(files)}")
    records = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append((float(payload["args"]["alpha"]), payload))
    records.sort()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def epoch_series(group, key):
        return [(LABELS[a], [item[group][key] for item in p["training_diagnostics_history"]]) for a, p in records]

    def epoch_index_series(group, key, index):
        return [(LABELS[a], [item[group][key][index] for item in p["training_diagnostics_history"]]) for a, p in records]

    def final_step_series(group, key):
        return [(LABELS[a], p["training_diagnostics_history"][-1][group][key]) for a, p in records]

    # One detailed SVG is generated for each section of test_plan.md. These
    # include fields that are too specific for the compact overview chart.
    detailed_sections = {
        "codebook": [
            ("Code entropy", epoch_series("codebook", "entropy")),
            *[(f"Code {i} usage fraction", epoch_index_series("codebook", "usage_fraction", i)) for i in range(6)],
            ("Dead code count", epoch_series("codebook", "dead_code_count")),
            ("Max dead-code streak", epoch_series("codebook", "max_dead_code_streak")),
            ("Codebook cosine mean", epoch_series("codebook", "pairwise_cosine_mean")),
            ("Codebook cosine max", epoch_series("codebook", "pairwise_cosine_max")),
            ("Codebook update norm", epoch_series("codebook", "update_norm")),
        ],
        "policy": [
            ("Policy output entropy", epoch_series("policy", "output_entropy")),
            ("Policy y sensitivity", epoch_series("policy", "y_sensitivity")),
            ("Policy state sensitivity", epoch_series("policy", "state_sensitivity")),
        ],
        "executor": [
            ("Latent norm by rollout step", final_step_series("executor", "rollout_latent_norm_by_step")),
            ("Grounding loss by step", final_step_series("executor", "grounding_loss_by_step")),
            ("Reconstruction loss by step", final_step_series("executor", "reconstruction_loss_by_step")),
            ("Executor z sensitivity by step", final_step_series("executor", "z_sensitivity_by_step")),
        ],
        "mdl": [
            ("Mean chosen length", epoch_series("mdl", "mean_chosen_length")),
            *[(f"Chosen length {i} count", epoch_index_series("mdl", "chosen_length_histogram", i)) for i in range(1, 5)],
            *[(f"MDL score length {i}", epoch_index_series("mdl", "score_by_length", i - 1)) for i in range(1, 5)],
        ],
        "loss": [
            ("Total loss", epoch_series("loss", "total")),
            ("Raw support loss", epoch_series("loss", "support")),
            ("Raw VQ loss", epoch_series("loss", "vq")),
            ("Raw grounding loss", epoch_series("loss", "grounding")),
            ("Raw query loss", epoch_series("loss", "query")),
            ("Weighted support loss", epoch_series("loss", "weighted_support")),
            ("Weighted VQ loss", epoch_series("loss", "weighted_vq")),
            ("Weighted grounding loss", epoch_series("loss", "weighted_grounding")),
            ("Weighted query loss", epoch_series("loss", "weighted_query")),
            ("Support/query loss ratio", epoch_series("loss", "support_query_ratio")),
        ],
    }
    for section, section_panels in detailed_sections.items():
        (args.output_dir / f"gridworld_{section}_all_metrics.svg").write_text(
            chart_svg(f"GridWorld {section} diagnostics, seed=42", [(title, series, "epoch / step") for title, series in section_panels]),
            encoding="utf-8",
        )

    panels = [
        ("Code entropy", epoch_series("codebook", "entropy"), "epoch"),
        ("Dead code count", epoch_series("codebook", "dead_code_count"), "epoch"),
        ("Codebook cosine max", epoch_series("codebook", "pairwise_cosine_max"), "epoch"),
        ("Policy output entropy", epoch_series("policy", "output_entropy"), "epoch"),
        ("Policy y sensitivity", epoch_series("policy", "y_sensitivity"), "epoch"),
        ("Policy state sensitivity", epoch_series("policy", "state_sensitivity"), "epoch"),
        ("Mean chosen length", epoch_series("mdl", "mean_chosen_length"), "epoch"),
        ("Weighted support loss", epoch_series("loss", "weighted_support"), "epoch"),
        ("Weighted VQ loss", epoch_series("loss", "weighted_vq"), "epoch"),
        ("Weighted grounding loss", epoch_series("loss", "weighted_grounding"), "epoch"),
        ("Total loss", epoch_series("loss", "total"), "epoch"),
        ("Support/query loss ratio", epoch_series("loss", "support_query_ratio"), "epoch"),
    ]
    chart = chart_svg("GridWorld test_plan.md training diagnostics, seed=42", panels)
    (args.output_dir / "gridworld_training_diagnostics.svg").write_text(chart, encoding="utf-8")

    step_panels = []
    for title, group, key in [
        ("Latent norm", "executor", "rollout_latent_norm_by_step"),
        ("Grounding loss", "executor", "grounding_loss_by_step"),
        ("Reconstruction loss", "executor", "reconstruction_loss_by_step"),
        ("Executor z sensitivity", "executor", "z_sensitivity_by_step"),
        ("MDL score by length", "mdl", "score_by_length"),
    ]:
        series = []
        for alpha, payload in records:
            history = payload["training_diagnostics_history"]
            series.append((LABELS[alpha], history[-1][group][key]))
        step_panels.append((title, series, "step / candidate length"))
    step_chart = chart_svg("Per-step executor and MDL diagnostics, seed=42", step_panels)
    (args.output_dir / "gridworld_per_step_diagnostics.svg").write_text(step_chart, encoding="utf-8")

    lines = [
        "# GridWorld test_plan.md 图表结果", "",
        "三组实验均为 seed=42。SVG 图表可直接用浏览器打开：", "",
        "- `gridworld_training_diagnostics.svg`：五类训练指标按 epoch 的曲线",
        "- `gridworld_per_step_diagnostics.svg`：执行器逐步指标和 MDL score 曲线", "",
        "详细五部分指标：",
        "- `gridworld_codebook_all_metrics.svg`：使用率、熵、死码连续时长、码字相似度、更新幅度",
        "- `gridworld_policy_all_metrics.svg`：策略熵、对 y 敏感度、对状态敏感度",
        "- `gridworld_executor_all_metrics.svg`：潜范数、逐步 grounding、重构、z 敏感度",
        "- `gridworld_mdl_all_metrics.svg`：k* 均值、k* 分布、各长度 score",
        "- `gridworld_loss_all_metrics.svg`：raw/weighted 的 support、VQ、grounding、query 及比值", "",
        "## 最终评测", "",
        "| alpha | split | self-explainability | transfer accuracy | mean chosen length | fallback rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for alpha, payload in records:
        for split, metrics in payload["metrics"].items():
            fallback = metrics.get("fallback_rate", float("nan"))
            lines.append(f"| {alpha:.2f} | {split} | {metrics['self_explainability']:.4f} | {metrics['transfer_accuracy']:.4f} | {metrics['mean_chosen_length']:.3f} | {fallback:.4f} |")
    (args.output_dir / "gridworld_test_plan_charts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
