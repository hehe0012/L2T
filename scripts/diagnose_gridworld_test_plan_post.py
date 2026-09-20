#!/usr/bin/env python3
"""Run all post-training A-E diagnostics from test_plan.md."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from l2t.gridworld import ACTIONS, GridState, apply_action, generate_examples
from l2t.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(path, device):
    import torch

    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["args"]["model"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def targets_for_actions(size, device):
    import torch

    states = list(range(size * size))
    targets = []
    for action in range(len(ACTIONS)):
        targets.append(torch.tensor([
            apply_action(GridState.from_index(s, size=size), action, size=size, boundary="wrap").to_index(size)
            for s in states
        ], dtype=torch.long, device=device))
    return torch.stack(targets, dim=1)


def code_outputs(model, states, code_count):
    import torch

    source = model._encode_indices(model.module, states)
    source = source[:, None, :].expand(-1, code_count, -1).reshape(-1, source.shape[-1])
    ids = torch.arange(code_count, device=states.device)[None, :].expand(states.numel(), -1).reshape(-1)
    actions = model.module.action_codebook(ids)
    next_latent = model._execute_step(model.module, source, actions).view(states.numel(), code_count, -1)
    predicted = model._decode_logits(model.module, next_latent.reshape(-1, next_latent.shape[-1])).argmax(dim=-1)
    return next_latent, predicted.view(states.numel(), code_count)


def displacement(size, start, end):
    a, b = GridState.from_index(int(start), size=size), GridState.from_index(int(end), size=size)
    return ((b.row - a.row) % size, (b.col - a.col) % size)


def norm_stats(values):
    import statistics

    return {"mean": float(statistics.mean(values)), "std": float(statistics.pstdev(values)), "max": float(max(values))}


def analyze_checkpoint(path, device):
    import torch
    import torch.nn.functional as F

    model, ckpt = load_model(path, device)
    size = int(ckpt["args"].get("grid_size", 10))
    code_count = int(model.codebook_size)
    states = torch.arange(size * size, dtype=torch.long, device=device)
    targets = targets_for_actions(size, device)
    state_latent = model._encode_indices(model.module, states)
    next_latent, predicted = code_outputs(model, states, code_count)

    # A1/A2: exact code-primitive alignment and displacement diversity.
    exact = torch.stack([(predicted == targets[:, action, None]) for action in range(len(ACTIONS))], dim=-1)
    alignment = exact.sum(dim=0)
    per_primitive = exact.any(dim=1).float().mean(dim=0)
    per_code = exact.float().mean(dim=0)
    rows = []
    dominant = {}
    for code in range(code_count):
        counts = Counter(displacement(size, s, int(predicted[s, code])) for s in range(size * size))
        best, count = counts.most_common(1)[0]
        dominant[code] = best
        rows.append({"code": code, "consistency": count / (size * size), "unique_displacements": len(counts), "dominant_displacement": list(best), "top_displacements": [[list(k), v] for k, v in counts.most_common(8)]})
    best_mapping = None
    best_score = -1.0
    for perm in itertools.permutations(range(code_count), len(ACTIONS)):
        score = sum(float(per_code[perm[a], a]) for a in range(len(ACTIONS)))
        if score > best_score:
            best_score, best_mapping = score, perm
    mapping = {ACTIONS[a]: int(best_mapping[a]) for a in range(len(ACTIONS))}

    # A3/D2: VAE displacement geometry.
    geometry = []
    for action in range(len(ACTIONS)):
        next_state_latent = model._encode_indices(model.module, targets[:, action])
        delta = next_state_latent - state_latent
        mean_delta = delta.mean(dim=0)
        cosine = F.cosine_similarity(delta, mean_delta[None, :], dim=-1)
        norms = delta.norm(dim=-1)
        geometry.append({"primitive": ACTIONS[action], "pairwise_cosine_to_mean": float(cosine.mean()), "norm_mean": float(norms.mean()), "norm_std": float(norms.std(unbiased=False)), "relative_norm_std": float(norms.std(unbiased=False) / norms.mean().clamp_min(1e-12))})

    # A4/D4: codebook relations and state-latent norm distribution.
    code_norm = F.normalize(model.module.action_codebook.weight.detach(), dim=-1)
    code_cosine = code_norm @ code_norm.T
    off = code_cosine[~torch.eye(code_count, device=device, dtype=torch.bool)]
    state_norms = state_latent.norm(dim=-1).cpu().tolist()

    # B1/B2/B3/B5: oracle program rollouts with and without decode-reencode.
    oracle = {"one_step_accuracy_by_primitive": {ACTIONS[a]: float((predicted[:, best_mapping[a]] == targets[:, a]).float().mean()) for a in range(len(ACTIONS))}, "mapping": mapping, "rollout_by_length": {}}
    for length in range(1, 5):
        programs = list(itertools.product(range(len(ACTIONS)), repeat=length))
        free_correct = project_correct = 0
        total = 0
        free_norms = [[] for _ in range(length + 1)]
        free_step_correct = [0] * length
        project_step_correct = [0] * length
        for program in programs:
            current_free = state_latent.clone()
            current_project = state_latent.clone()
            for step, action in enumerate(program):
                code = torch.full((states.numel(),), int(best_mapping[action]), dtype=torch.long, device=device)
                code_vec = model.module.action_codebook(code)
                current_free = model._execute_step(model.module, current_free, code_vec)
                current_project = model._execute_step(model.module, current_project, code_vec)
                projected_logits = model._decode_logits(model.module, current_project)
                current_project = model._encode_decoded_state(model.module, current_project)
                gt = targets_for_program(size, program, device)
                free_pred = model._decode_logits(model.module, current_free).argmax(dim=-1)
                project_pred = projected_logits.argmax(dim=-1)
                free_step_correct[step] += int((free_pred == gt).sum())
                project_step_correct[step] += int((project_pred == gt).sum())
                free_norms[step + 1].extend(current_free.norm(dim=-1).cpu().tolist())
            free_final = model._decode_logits(model.module, current_free).argmax(dim=-1)
            project_final = model._decode_logits(model.module, current_project).argmax(dim=-1)
            final_gt = targets_for_program(size, program, device)
            free_correct += int((free_final == final_gt).sum())
            project_correct += int((project_final == final_gt).sum())
            total += states.numel()
        oracle["rollout_by_length"][str(length)] = {"programs": len(programs), "free_accuracy": free_correct / total, "decode_reencode_accuracy": project_correct / total, "free_step_accuracy": [x / total for x in free_step_correct], "decode_reencode_step_accuracy": [x / total for x in project_step_correct], "free_latent_norm_by_step": [norm_stats(x) if x else None for x in free_norms[1:]]}

    # B4: action-code sensitivity.
    unique_decoded = torch.unique(predicted, dim=1).shape[1] if False else [int(torch.unique(predicted[s]).numel()) for s in range(states.numel())]
    b4 = {"mean_pairwise_next_latent_distance": float(torch.stack([torch.pdist(next_latent[s]).mean() for s in range(states.numel())]).mean()), "mean_unique_decoded_next_states": float(sum(unique_decoded) / len(unique_decoded)), "all_codes_same_decoded_next_state_fraction": sum(x == 1 for x in unique_decoded) / len(unique_decoded)}

    # C1/C2/C3: policy entropy and sensitivity, plus strict primitive variance.
    normal = state_latent[:, None, :].expand(-1, states.numel(), -1)
    target_grid = state_latent[None, :, :].expand(states.numel(), -1, -1)
    flat_pre = model.module.policy(normal.reshape(-1, normal.shape[-1]), target_grid.reshape(-1, target_grid.shape[-1]))
    distances = torch.cdist(flat_pre[:, None, :], model.module.action_codebook.weight[None, :, :]).squeeze(1)
    probs = (-distances).softmax(dim=-1)
    c_entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean())
    perm = torch.randperm(states.numel(), device=device)
    normal_pre = model.module.policy(state_latent, state_latent)
    y_pre = model.module.policy(state_latent, state_latent[perm])
    s_pre = model.module.policy(state_latent[perm], state_latent)
    c = {"output_entropy": c_entropy, "y_sensitivity": float((normal_pre - y_pre).pow(2).mean(dim=-1).sqrt().mean()), "state_sensitivity": float((normal_pre - s_pre).pow(2).mean(dim=-1).sqrt().mean()), "primitive_conditional_pre_variance": {}}
    for action in range(len(ACTIONS)):
        pre = model.module.policy(state_latent, model._encode_indices(model.module, targets[:, action]))
        c["primitive_conditional_pre_variance"][ACTIONS[action]] = float((pre - pre.mean(dim=0)).pow(2).mean())

    # D1/D3/D4: VAE reconstruction, linear probe, norm statistics.
    recon = model._decode_logits(model.module, state_latent).argmax(dim=-1)
    delta_features = torch.stack([model._encode_indices(model.module, targets[:, a]) - state_latent for a in range(4)]).reshape(-1, state_latent.shape[-1])
    x = torch.cat([delta_features, torch.ones(4 * size * size, 1, device=device)], dim=-1)
    y = torch.eye(4, device=device).repeat_interleave(size * size, dim=0)
    weights = torch.linalg.pinv(x) @ y
    probe_pred = (x @ weights).argmax(dim=-1)
    d = {"reconstruction_accuracy": float((recon == states).float().mean()), "linear_probe_delta_accuracy": float((probe_pred == y.argmax(dim=-1)).float().mean()), "state_latent_norm": norm_stats(state_norms), "primitive_geometry": geometry}

    # E1/E2: GT program with oracle code versus policy-selected code.
    alpha = float(ckpt["args"]["alpha"])
    eval_sizes = {"id": 1000, "comp_ood": 1000 if alpha < 1.0 else 0, "length_ood": 2000}
    e = {}
    for split, count in eval_sizes.items():
        if count == 0:
            continue
        kwargs = dict(n=count, split=split, seed=42 + {"id": 101, "comp_ood": 202, "length_ood": 303}[split], alpha=alpha, size=size, split_mode="paper", train_min_len=1, train_max_len=3, length_ood_min_len=4, length_ood_max_len=8)
        examples = generate_examples(**kwargs)
        oracle_support = oracle_query = policy_support = policy_query = 0
        for start in range(0, len(examples), 512):
            batch = examples[start:start + 512]
            sx = torch.tensor([z.support_x for z in batch], dtype=torch.long, device=device)
            sy = torch.tensor([z.support_y for z in batch], dtype=torch.long, device=device)
            qx = torch.tensor([z.query_x for z in batch], dtype=torch.long, device=device)
            qy = torch.tensor([z.query_y for z in batch], dtype=torch.long, device=device)
            free_s, free_q = model._encode_indices(model.module, sx), model._encode_indices(model.module, qx)
            pol_s, pol_q = free_s.clone(), free_q.clone()
            target = model._encode_indices(model.module, sy)
            for step in range(max(len(z.program) for z in batch)):
                active = torch.tensor([step < len(z.program) for z in batch], device=device)
                oracle_codes = torch.tensor([best_mapping[z.program[step]] if step < len(z.program) else best_mapping[0] for z in batch], dtype=torch.long, device=device)
                oracle_action = model.module.action_codebook(oracle_codes)
                free_s = torch.where(active[:, None], model._execute_step(model.module, free_s, oracle_action), free_s)
                free_q = torch.where(active[:, None], model._execute_step(model.module, free_q, oracle_action), free_q)
                pre = model.module.policy(pol_s, target)
                policy_codes = (-torch.cdist(pre[:, None, :], model.module.action_codebook.weight[None, :, :]).squeeze(1)).argmax(dim=-1)
                policy_action = model.module.action_codebook(policy_codes)
                pol_s = torch.where(active[:, None], model._execute_step(model.module, pol_s, policy_action), pol_s)
                pol_q = torch.where(active[:, None], model._execute_step(model.module, pol_q, policy_action), pol_q)
            oracle_support += int((model._decode_logits(model.module, free_s).argmax(dim=-1) == sy).sum())
            oracle_query += int((model._decode_logits(model.module, free_q).argmax(dim=-1) == qy).sum())
            policy_support += int((model._decode_logits(model.module, pol_s).argmax(dim=-1) == sy).sum())
            policy_query += int((model._decode_logits(model.module, pol_q).argmax(dim=-1) == qy).sum())
        e[split] = {"examples": len(examples), "gt_program_gt_code_support": oracle_support / len(examples), "gt_program_gt_code_query": oracle_query / len(examples), "gt_program_policy_code_support": policy_support / len(examples), "gt_program_policy_code_query": policy_query / len(examples)}

    return {"alpha": alpha, "checkpoint": str(path), "A_codebook": {"alignment_counts": alignment.cpu().tolist(), "alignment_rates": per_code.cpu().tolist(), "primitiveness_by_primitive": {ACTIONS[a]: float(per_primitive[a]) for a in range(4)}, "action_primitiveness": float(per_primitive.mean()), "code_displacement": rows, "codebook_cosine_mean": float(off.mean()), "codebook_cosine_max": float(off.max())}, "B_executor": {"oracle": oracle, "z_sensitivity": b4}, "C_policy": c, "D_vae": d, "E_end_to_end": e}


def targets_for_program(size, program, device):
    import torch

    result = []
    for s in range(size * size):
        current = GridState.from_index(s, size=size)
        for action in program:
            current = apply_action(current, action, size=size, boundary="wrap")
        result.append(current.to_index(size))
    return torch.tensor(result, dtype=torch.long, device=device)


def write_markdown(results, path):
    lines = ["# GridWorld test_plan.md A-E 训练后诊断", "", "三组最终 checkpoint 均为 seed=42，诊断直接加载最终模型参数。", ""]
    lines += ["## 总览", "", "| alpha | A primitiveness | B one-step oracle | B length-4 free | B length-4 reencode | C policy entropy | D recon | D linear probe | E1 ID query | E2 ID query |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a = r["alpha"]; o = r["B_executor"]["oracle"]; d = r["D_vae"]; e = r["E_end_to_end"]["id"]
        lines.append(f"| {a:.2f} | {r['A_codebook']['action_primitiveness']:.4f} | {sum(o['one_step_accuracy_by_primitive'].values())/4:.4f} | {o['rollout_by_length']['4']['free_accuracy']:.4f} | {o['rollout_by_length']['4']['decode_reencode_accuracy']:.4f} | {r['C_policy']['output_entropy']:.4f} | {d['reconstruction_accuracy']:.4f} | {d['linear_probe_delta_accuracy']:.4f} | {e['gt_program_gt_code_query']:.4f} | {e['gt_program_policy_code_query']:.4f} |")
    for r in results:
        a = r["alpha"]; lines += ["", f"## alpha={a:.2f}", "", "### A. Codebook / primitive / latent geometry", "", "```json", json.dumps(r["A_codebook"], indent=2, ensure_ascii=False), "```", "", "### B. Executor oracle", "", "```json", json.dumps(r["B_executor"], indent=2, ensure_ascii=False), "```", "", "### C. Policy", "", "```json", json.dumps(r["C_policy"], indent=2, ensure_ascii=False), "```", "", "### D. VAE", "", "```json", json.dumps(r["D_vae"], indent=2, ensure_ascii=False), "```", "", "### E. End-to-end", "", "```json", json.dumps(r["E_end_to_end"], indent=2, ensure_ascii=False), "```"]
    lines += ["", "## 判读", "", "- A 区分 code 是否形成稳定 primitive；B 区分 executor/VAE 是否能执行已知程序；C 检查 policy 是否使用目标并错误依赖当前位置；D 检查 VAE 几何；E 对比 GT code 与 policy code，定位策略选择还是执行器。", "- 本次 E 使用每个 split 的 1000/2000 个样本，A-D 使用全部 100 个 grid state 和长度 1--4 的程序枚举。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    import torch

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    files = sorted(args.checkpoint_dir.glob("*.pt"))
    with torch.inference_mode():
        results = [analyze_checkpoint(path, device) for path in files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        out = args.output_dir / f"gridworld_test_plan_post_alpha{result['alpha']:.2f}_seed42.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, args.markdown)
    print(json.dumps({"alphas": [r["alpha"] for r in results], "markdown": str(args.markdown)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
