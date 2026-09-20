#!/usr/bin/env python3
"""Create concise charts and a summary Markdown from A-E diagnostic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLORS = ["#2563eb", "#059669", "#dc2626"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--chart-dir", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    return p.parse_args()


def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(title, labels, series, ylabel="", width=1200, height=560):
    # series: [(name, values, color)]
    left, top, pw, ph = 90, 70, width - 140, height - 170
    values = [v for _, xs, _ in series for v in xs]
    ymax = max(values) * 1.18 if values else 1
    if ymax <= 0:
        ymax = 1
    group_w = pw / max(1, len(labels))
    bar_w = group_w / max(1, len(series) + 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{esc(title)}</text>']
    for i in range(5):
        y = top + ph * i / 4
        val = ymax * (4 - i) / 4
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.3g}</text>')
    for j, label in enumerate(labels):
        gx = left + group_w * j
        out.append(f'<text x="{gx+group_w/2:.1f}" y="{top+ph+28}" text-anchor="middle" font-family="Arial" font-size="12">{esc(label)}</text>')
        for k, (_, xs, color) in enumerate(series):
            value = xs[j]
            x = gx + bar_w * (k + 0.5)
            y = top + ph * (1 - value / ymax)
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*.8:.1f}" height="{top+ph-y:.1f}" fill="{color}"/>')
    out.append(f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20 {top+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">{esc(ylabel)}</text>')
    for i, (name, _, color) in enumerate(series):
        x = left + i * 180
        out += [f'<rect x="{x}" y="{height-42}" width="14" height="14" fill="{color}"/>', f'<text x="{x+20}" y="{height-30}" font-family="Arial" font-size="12">{esc(name)}</text>']
    out.append('</svg>')
    return "\n".join(out)


def lines(title, xlabels, series, ylabel="", width=1200, height=560):
    left, top, pw, ph = 90, 70, width - 140, height - 170
    values = [v for _, xs, _ in series for v in xs]
    ymin, ymax = min(values), max(values)
    if ymax == ymin:
        ymin -= 1
        ymax += 1
    pad = (ymax - ymin) * .1
    ymin -= pad
    ymax += pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{esc(title)}</text>']
    for i in range(5):
        y = top + ph * i / 4
        val = ymax - (ymax-ymin) * i / 4
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>', f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.3g}</text>']
    for name, xs, color in series:
        pts = []
        for i, value in enumerate(xs):
            x = left + pw * i / max(1, len(xs)-1)
            y = top + ph * (ymax-value)/(ymax-ymin)
            pts.append(f'{x:.1f},{y:.1f}')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for i, label in enumerate(xlabels):
        x = left + pw * i / max(1, len(xlabels)-1)
        out.append(f'<text x="{x:.1f}" y="{top+ph+28}" text-anchor="middle" font-family="Arial" font-size="12">{esc(label)}</text>')
    out.append(f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20 {top+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">{esc(ylabel)}</text>')
    for i, (name, _, color) in enumerate(series):
        x = left + i * 180
        out += [f'<line x1="{x}" y1="{height-36}" x2="{x+18}" y2="{height-36}" stroke="{color}" stroke-width="3"/>', f'<text x="{x+24}" y="{height-30}" font-family="Arial" font-size="12">{esc(name)}</text>']
    out.append('</svg>')
    return "\n".join(out)


def main():
    a = args()
    records = []
    for p in sorted(a.input_dir.glob("*.json")):
        d = json.loads(p.read_text())
        records.append((float(d["alpha"]), d))
    records.sort()
    labels = [f"alpha={alpha:.2f}" for alpha, _ in records]
    a.chart_dir.mkdir(parents=True, exist_ok=True)

    a.chart_dir.joinpath("post_A_alignment.svg").write_text(bars("A: Action primitiveness", labels, [("primitiveness", [d["A_codebook"]["action_primitiveness"] for _, d in records], COLORS[0])], "rate"))
    a.chart_dir.joinpath("post_A_displacement.svg").write_text(bars("A: Code displacement diversity", labels, [("mean unique displacement", [sum(x["unique_displacements"] for x in d["A_codebook"]["code_displacement"])/6 for _, d in records], COLORS[1]), ("best consistency", [max(x["consistency"] for x in d["A_codebook"]["code_displacement"]) for _, d in records], COLORS[2])], "count / consistency"))
    lengths = ["1", "2", "3", "4"]
    a.chart_dir.joinpath("post_B_oracle_rollout.svg").write_text(lines("B: Oracle rollout accuracy", lengths, [(f"alpha={alpha:.2f} free", [d["B_executor"]["oracle"]["rollout_by_length"][x]["free_accuracy"] for x in lengths], COLORS[i]) for i, (alpha, d) in enumerate(records)] + [(f"alpha={alpha:.2f} reencode", [d["B_executor"]["oracle"]["rollout_by_length"][x]["decode_reencode_accuracy"] for x in lengths], COLORS[i]) for i, (alpha, d) in enumerate(records)], "accuracy"))
    a.chart_dir.joinpath("post_C_policy.svg").write_text(bars("C: Policy diagnostics", labels, [("entropy", [d["C_policy"]["output_entropy"] for _, d in records], COLORS[0]), ("y sensitivity", [d["C_policy"]["y_sensitivity"] for _, d in records], COLORS[1]), ("state sensitivity", [d["C_policy"]["state_sensitivity"] for _, d in records], COLORS[2])], "value"))
    a.chart_dir.joinpath("post_D_vae.svg").write_text(bars("D: VAE diagnostics", labels, [("reconstruction", [d["D_vae"]["reconstruction_accuracy"] for _, d in records], COLORS[0]), ("linear probe", [d["D_vae"]["linear_probe_delta_accuracy"] for _, d in records], COLORS[1]), ("latent norm std", [d["D_vae"]["state_latent_norm"]["std"] for _, d in records], COLORS[2])], "value"))
    e_labels = [f"{alpha:.2f} ID" for alpha, _ in records]
    e_values = [d["E_end_to_end"]["id"] for _, d in records]
    a.chart_dir.joinpath("post_E_end_to_end.svg").write_text(bars("E: ID oracle vs policy program", e_labels, [("GT code query", [x["gt_program_gt_code_query"] for x in e_values], COLORS[0]), ("policy code query", [x["gt_program_policy_code_query"] for x in e_values], COLORS[1])], "query accuracy"))

    lines_out = ["# GridWorld test_plan.md A-E 训练后诊断", "", "本文件只保留关键指标、图表链接和结论；完整原始诊断 JSON 位于：", "", "`/home/guian/L2T/logs/gridworld_test_plan_post_seed42/`", "", "## 图表", "", "| 部分 | 图表 | 重点指标 |", "|---|---|---|", "| A | [alignment](logs/gridworld_test_plan_post_seed42/charts/post_A_alignment.svg)、[displacement](logs/gridworld_test_plan_post_seed42/charts/post_A_displacement.svg) | 原语性、位移多样性、code 一致性 |", "| B | [oracle rollout](logs/gridworld_test_plan_post_seed42/charts/post_B_oracle_rollout.svg) | 单步/多步、free vs decode-reencode |", "| C | [policy](logs/gridworld_test_plan_post_seed42/charts/post_C_policy.svg) | 熵、目标敏感度、状态敏感度 |", "| D | [VAE](logs/gridworld_test_plan_post_seed42/charts/post_D_vae.svg) | 重建、位移线性探测、latent 范数 |", "| E | [end-to-end](logs/gridworld_test_plan_post_seed42/charts/post_E_end_to_end.svg) | GT code 与 policy code 的 query transfer |", "", "## 关键数值", "", "| alpha | A primitiveness | B one-step oracle | B length-4 free | B length-4 reencode | C policy entropy | C y/state sensitivity | D recon | D linear probe | E1 ID query | E2 ID query |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha, d in records:
        o = d["B_executor"]["oracle"]; c = d["C_policy"]; v = d["D_vae"]; e = d["E_end_to_end"]["id"]
        one = sum(o["one_step_accuracy_by_primitive"].values()) / 4
        lines_out.append(f"| {alpha:.2f} | {d['A_codebook']['action_primitiveness']:.4f} | {one:.4f} | {o['rollout_by_length']['4']['free_accuracy']:.4f} | {o['rollout_by_length']['4']['decode_reencode_accuracy']:.4f} | {c['output_entropy']:.4f} | {c['y_sensitivity']:.3f}/{c['state_sensitivity']:.3f} | {v['reconstruction_accuracy']:.4f} | {v['linear_probe_delta_accuracy']:.4f} | {e['gt_program_gt_code_query']:.4f} | {e['gt_program_policy_code_query']:.4f} |")
    lines_out += ["", "## 总结", "", "- A：原语性仅 `0.0275--0.1250`，code 在 100 个状态上产生大量不同位移，未形成稳定 primitive。", "- B：GT code 的单步和多步 oracle 都很低，decode-reencode 没有恢复准确率，说明主要瓶颈在 VAE latent geometry / executor，而不是单纯的 policy 选码。", "- C：policy 对 `y` 敏感，但对 `s_k` 也敏感；因此存在位置相关的 code 选择，不过这不是主要失败点。", "- D：VAE 重建为 `1.0`，但位移线性探测为 `0.2525`，接近四分类随机水平；VAE 能记住位置，却没有提供可组合的 primitive 几何。", "- E：GT program + GT code 的 query accuracy 只有 `0.020--0.036`，换成 policy code 后几乎不变，进一步排除 policy 是首要瓶颈。", "", "**最终定位：** 先修复 state latent 的位移几何和 executor 的 primitive 对齐，再调 policy、MDL 或采样数。"]

    # Detailed A1-E2 tables: show every post-training test item without
    # embedding the raw diagnostic JSON in the report.
    lines_out += ["", "# 逐项诊断结果"]
    lines_out += ["", "## A1. Code-Primitive Alignment", "", "每个单元格为该 code 在 100 个起始状态上与 primitive 完全匹配的次数 / 100。"]
    lines_out += ["", "| alpha | code | U | D | L | R |", "|---:|---:|---:|---:|---:|---:|"]
    for alpha, d in records:
        for code, row in enumerate(d["A_codebook"]["alignment_rates"]):
            lines_out.append(f"| {alpha:.2f} | {code} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    lines_out += ["", "**A1 结果：** 四个 primitive 的最佳 code 覆盖率仍很低；这不是近似的近对角 alignment。详见 [A alignment 图](logs/gridworld_test_plan_post_seed42/charts/post_A_alignment.svg)。"]

    lines_out += ["", "## A2. 每个 code 的位移多样性", "", "`unique_displacements` 越接近 1 越好；`consistency` 是该 code 产生其主导位移的比例。"]
    lines_out += ["", "| alpha | code | unique displacements | consistency | dominant displacement |", "|---:|---:|---:|---:|---|"]
    for alpha, d in records:
        for row in d["A_codebook"]["code_displacement"]:
            lines_out.append(f"| {alpha:.2f} | {row['code']} | {row['unique_displacements']} | {row['consistency']:.3f} | {row['dominant_displacement']} |")
    lines_out += ["", "**A2 结果：** code 通常产生 `34--74` 种位移，主导位移比例最高也只有约 `0.11`，说明 latent action 不是稳定 primitive。详见 [A displacement 图](logs/gridworld_test_plan_post_seed42/charts/post_A_displacement.svg)。"]

    lines_out += ["", "## A3. Primitive 位移一致性", "", "`pairwise_cosine_to_mean` 越高越好；`relative_norm_std` 越低越好。"]
    lines_out += ["", "| alpha | primitive | cosine to mean | norm mean | norm std | relative norm std |", "|---:|---|---:|---:|---:|---:|"]
    for alpha, d in records:
        for row in d["D_vae"]["primitive_geometry"]:
            lines_out.append(f"| {alpha:.2f} | {row['primitive']} | {row['pairwise_cosine_to_mean']:.4f} | {row['norm_mean']:.3f} | {row['norm_std']:.3f} | {row['relative_norm_std']:.3f} |")
    lines_out += ["", "**A3 结果：** 四个 primitive 的 cosine 都约为 `0`，而不是计划中的 `>0.5`；位移方向完全不一致。长度方差相对较小，但方向问题更严重。"]

    lines_out += ["", "## A4. 码字间关系", "", "| alpha | codebook cosine mean | codebook cosine max |", "|---:|---:|---:|"]
    for alpha, d in records:
        lines_out.append(f"| {alpha:.2f} | {d['A_codebook']['codebook_cosine_mean']:.4f} | {d['A_codebook']['codebook_cosine_max']:.4f} |")
    lines_out += ["", "**A4 结果：** alpha=0.33 的最大余弦相似度达到 `0.869`，存在高度相似码字；alpha=0.66/1.00 也未形成清晰的正交 primitive 结构。"]

    lines_out += ["", "## B1. Oracle 单步准确率", "", "| alpha | U | D | L | R | mean |", "|---:|---:|---:|---:|---:|---:|"]
    for alpha, d in records:
        vals = d["B_executor"]["oracle"]["one_step_accuracy_by_primitive"]
        xs = [vals[name] for name in ["up", "down", "left", "right"]]
        lines_out.append(f"| {alpha:.2f} | " + " | ".join(f"{x:.4f}" for x in xs) + f" | {sum(xs)/4:.4f} |")
    lines_out += ["", "**B1 结果：** oracle code 的单步准确率只有 `0.020--0.068`，远低于正常执行器应有的高准确率。"]

    lines_out += ["", "## B2/B3. Oracle 多步 rollout", "", "下表为 `free latent rollout / 每步 decode-reencode` 的最终准确率。"]
    lines_out += ["", "| alpha | len=1 | len=2 | len=3 | len=4 |", "|---:|---:|---:|---:|---:|"]
    for alpha, d in records:
        vals = d["B_executor"]["oracle"]["rollout_by_length"]
        lines_out.append(f"| {alpha:.2f} | " + " | ".join(f"{vals[str(k)]['free_accuracy']:.4f} / {vals[str(k)]['decode_reencode_accuracy']:.4f}" for k in range(1, 5)) + " |")
    lines_out += ["", "**B2/B3 结果：** decode-reencode 没有把准确率恢复到高水平，说明问题不是单纯的 free rollout 漂移，而是 code/transition 与 VAE 状态几何本身不匹配。", "", "## B4. Executor 对 z 的敏感度", "", "| alpha | mean pairwise latent distance | mean unique decoded states | all-codes-same fraction |", "|---:|---:|---:|---:|"]
    for alpha, d in records:
        x = d["B_executor"]["z_sensitivity"]
        lines_out.append(f"| {alpha:.2f} | {x['mean_pairwise_next_latent_distance']:.3f} | {x['mean_unique_decoded_next_states']:.3f} | {x['all_codes_same_decoded_next_state_fraction']:.3f} |")
    lines_out += ["", "**B4 结果：** executor 对 z 有明显响应，但不同 code 的输出不能稳定对应四个 primitive；因此不是简单的 z 被忽略。", "", "## B5. 潜空间漂移", "", "| alpha | length-4 step1 norm mean | step2 | step3 | step4 |", "|---:|---:|---:|---:|---:|"]
    for alpha, d in records:
        xs = d["B_executor"]["oracle"]["rollout_by_length"]["4"]["free_latent_norm_by_step"]
        lines_out.append(f"| {alpha:.2f} | " + " | ".join(f"{x['mean']:.2f}" for x in xs) + " |")
    lines_out += ["", "**B5 结果：** 本次没有出现范数指数爆炸；但范数稳定不代表状态在正确 manifold 上，B1-B3 的低准确率已经证明 dynamics 仍不可用。"]

    lines_out += ["", "## C1. 策略对 s_k 的敏感度 / primitive 条件方差", "", "| alpha | primitive | a_pre conditional variance |", "|---:|---|---:|"]
    for alpha, d in records:
        for primitive, value in d["C_policy"]["primitive_conditional_pre_variance"].items():
            lines_out.append(f"| {alpha:.2f} | {primitive} | {value:.4f} |")
    lines_out += ["", "**C1 结果：** 同一 primitive 的 `a_pre` 仍随状态变化，说明策略输出不是位置无关的纯变换。"]
    lines_out += ["", "## C2. 策略对 y 的敏感度", "", "| alpha | y sensitivity | state sensitivity | y/state ratio |", "|---:|---:|---:|---:|"]
    for alpha, d in records:
        c = d["C_policy"]
        lines_out.append(f"| {alpha:.2f} | {c['y_sensitivity']:.3f} | {c['state_sensitivity']:.3f} | {c['y_sensitivity']/max(c['state_sensitivity'],1e-12):.2f} |")
    lines_out += ["", "**C2 结果：** policy 没有忽略 y，但 state sensitivity 也不可忽略；alpha=0.33 的位置依赖最强。"]
    lines_out += ["", "## C3. 策略输出熵", "", "| alpha | output entropy |", "|---:|---:|"]
    for alpha, d in records:
        lines_out.append(f"| {alpha:.2f} | {d['C_policy']['output_entropy']:.4f} |")
    lines_out += ["", "**C3 结果：** 熵仅 `0.289--0.477`，策略已经相当确定，探索不足。"]

    lines_out += ["", "## D1. VAE 重建准确率", "", "| alpha | reconstruction accuracy |", "|---:|---:|"]
    for alpha, d in records:
        lines_out.append(f"| {alpha:.2f} | {d['D_vae']['reconstruction_accuracy']:.4f} |")
    lines_out += ["", "**D1 结果：** 三个 checkpoint 都能完美重建 100 个状态；问题不是 VAE 记不住状态，而是 latent 几何不可组合。", "", "## D2. VAE 位移一致性", "", "D2 与 A3 使用同一组四 primitive 的 `E(y)-E(x)` 统计，详见 [VAE 图](logs/gridworld_test_plan_post_seed42/charts/post_D_vae.svg) 和 A3 表。结果是方向 cosine 约为 `0`。", "", "## D3. 位移线性探测", "", "| alpha | linear probe accuracy | 四分类随机基线 |", "|---:|---:|---:|"]
    for alpha, d in records:
        lines_out.append(f"| {alpha:.2f} | {d['D_vae']['linear_probe_delta_accuracy']:.4f} | 0.2500 |")
    lines_out += ["", "**D3 结果：** 几乎等于随机基线，latent displacement 不包含可线性分离的 primitive 方向。", "", "## D4. 潜空间范数分布", "", "| alpha | mean | std | max |", "|---:|---:|---:|---:|"]
    for alpha, d in records:
        x = d["D_vae"]["state_latent_norm"]
        lines_out.append(f"| {alpha:.2f} | {x['mean']:.3f} | {x['std']:.3f} | {x['max']:.3f} |")
    lines_out += ["", "**D4 结果：** state latent 范数分布本身不算发散，但这不能弥补 D2/D3 的方向不可分问题。"]

    lines_out += ["", "## E1. GT program + GT code", "", "| alpha | split | support accuracy | query transfer accuracy |", "|---:|---|---:|---:|"]
    for alpha, d in records:
        for split, x in d["E_end_to_end"].items():
            lines_out.append(f"| {alpha:.2f} | {split} | {x['gt_program_gt_code_support']:.4f} | {x['gt_program_gt_code_query']:.4f} |")
    lines_out += ["", "**E1 结果：** 即使直接提供真实程序和最佳 GT code mapping，query transfer 仍只有约 `0.003--0.036`，执行器和 latent code 本身已经失败。", "", "## E2. GT program length + policy inferred code", "", "| alpha | split | support accuracy | query transfer accuracy |", "|---:|---|---:|---:|"]
    for alpha, d in records:
        for split, x in d["E_end_to_end"].items():
            lines_out.append(f"| {alpha:.2f} | {split} | {x['gt_program_policy_code_support']:.4f} | {x['gt_program_policy_code_query']:.4f} |")
    lines_out += ["", "**E2 结果：** E2 与 E1 的 query accuracy 几乎相同，说明额外替换为 policy code 并没有造成主要损失；主因在 executor/VAE，而不是 policy 选错程序。"]
    a.markdown.parent.mkdir(parents=True, exist_ok=True)
    a.markdown.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
