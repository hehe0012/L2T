# 复现实验记录

目标论文：Learning to Theorize the World from Observation  
论文链接：https://arxiv.org/pdf/2605.03413

## 当前状态

- 已建立本地复现工程骨架。
- 已实现 GridWorld 数据生成与规则测试。
- 已实现 NEO 风格组合 latent program 模型的初始 PyTorch 版本。
- 已实现 NEO-S sampled inference：评估时从 programmer 分布采样候选 latent program，用 support likelihood + MDL 长度惩罚选择解释，再迁移到 query。
- 已实现 Disc-Mono baseline 的初始 PyTorch 版本。
- 已安装本地虚拟环境依赖，并完成 CPU smoke run。
- 发现 Slurm 计算节点上的 `/raid` 是节点本地视图；GPU 作业需要使用共享路径 `/home/guian/L2T`。

## 环境

- Python: `3.12.14`，由 `uv venv` 创建
- CUDA: PyTorch wheel 为 `cu130`，当前运行环境未检测到 CUDA 设备
- PyTorch: `2.14.0+cu130`
- NumPy: `2.5.3`
- GPU: 当前 smoke run 使用 CPU
- Commit: 当前目录尚未初始化为 git 仓库

## 已知假设

- GridWorld 边界默认使用 `wrap`，即越过边界时从另一侧出现。论文没有在主计划中明确边界语义；该设置保证所有 primitive 在每个位置都可执行，减少边界导致的程序歧义。
- 当前模型是第一版最小实现，用于打通 support-to-query transfer 闭环；完整复现仍需继续对齐论文中的 VQ、MDL、state grounding 和图像 VAE 细节。

## 命令记录

安装依赖：

```bash
UV_CACHE_DIR=/raid/guian/repos/L2T/.uv-cache uv venv
UV_CACHE_DIR=/raid/guian/repos/L2T/.uv-cache uv pip install -e '.[dev]'
```

规则与语法测试：

```bash
.venv/bin/python -m unittest discover tests
.venv/bin/python -m compileall src scripts tests
```

GridWorld NEO reduced smoke：

```bash
.venv/bin/python scripts/run_gridworld.py \
  --model neo \
  --train-size 1000 \
  --eval-size 200 \
  --steps 200 \
  --batch-size 64 \
  --device cpu \
  --out logs/gridworld_smoke.json
```

结果：

```json
{
  "id": {"transfer_accuracy": 0.085, "mean_chosen_length": 2.085},
  "comp_ood": {"transfer_accuracy": 0.035, "mean_chosen_length": 2.11},
  "length_ood": {"transfer_accuracy": 0.03, "mean_chosen_length": 2.445}
}
```

GridWorld Disc-Mono reduced smoke：

```bash
.venv/bin/python scripts/run_gridworld.py \
  --model disc_mono \
  --train-size 1000 \
  --eval-size 200 \
  --steps 200 \
  --batch-size 64 \
  --device cpu \
  --out logs/gridworld_disc_mono_smoke.json
```

结果：

```json
{
  "id": {"transfer_accuracy": 0.055, "mean_chosen_length": 1.0},
  "comp_ood": {"transfer_accuracy": 0.055, "mean_chosen_length": 1.0},
  "length_ood": {"transfer_accuracy": 0.025, "mean_chosen_length": 1.0}
}
```

GridWorld NEO-S reduced smoke：

```bash
.venv/bin/python scripts/run_gridworld.py \
  --model neo_s \
  --train-size 1000 \
  --eval-size 200 \
  --steps 200 \
  --batch-size 64 \
  --device cpu \
  --neo-s-samples 8 \
  --out logs/gridworld_neo_s_smoke.json
```

结果：

```json
{
  "id": {"transfer_accuracy": 0.085, "mean_chosen_length": 2.04},
  "comp_ood": {"transfer_accuracy": 0.035, "mean_chosen_length": 1.985},
  "length_ood": {"transfer_accuracy": 0.035, "mean_chosen_length": 2.415}
}
```

## 下一步

- 将 NEO 初始实现进一步改成更贴近论文的 VQ/straight-through latent codebook。
- 增加 checkpoint 保存与 config 驱动运行。
- 跑完整 NEO-S GridWorld sweep，并比较不同 `--neo-s-samples` 下的收益。

## Slurm 运行记录

- 资源检查：`all` 分区可用，8 个 mixed 节点，GRES 为 `gpu:nvidia_h200:8(S:0-1)`。
- 已确认计算节点可用 CUDA：`torch.cuda.is_available() == True`，设备为 `NVIDIA H200`。
- 早期失败记录：
  - `1223`、`1241` 使用 `/raid/guian/repos/L2T` 路径提交，因计算节点 `/raid` 视图不同而失败。
  - 改为共享路径 `/home/guian/L2T` 后，前台 CUDA 检查通过。
- 当前完整 GridWorld sweep：
  - Slurm array job: `1275`
  - Job name: `guian-l2t-gridworld`
  - Comment/tag: `guian`
  - Array: `0-17%8`
  - 每个 task 请求：`1 x NVIDIA H200`，`8 CPU`，`64G` 内存
  - 实验矩阵：`neo`、`disc_mono` x `alpha={0.33,0.66,1.00}` x `seed={0,1,2}`
  - 数据规模：train `100000`，ID eval `10000`，Comp OOD eval `10000`，Length OOD eval `20000`
  - 训练步数：`20000`
  - 输出目录：`/home/guian/L2T/logs/gridworld_full`
  - 状态：全部 `COMPLETED 0:0`
- 当前 Slurm 脚本已更新为后续完整 sweep：
  - Array: `0-26%8`
  - 实验矩阵：`neo`、`disc_mono`、`neo_s` x `alpha={0.33,0.66,1.00}` x `seed={0,1,2}`
  - NEO-S 采样数：`--neo-s-samples 16`

按 3 个 seed 聚合的 transfer accuracy：

| model | alpha | ID | Comp OOD | Length OOD |
| --- | ---: | ---: | ---: | ---: |
| `disc_mono` | `0.33` | `0.1306` | `0.0507` | `0.0643` |
| `disc_mono` | `0.66` | `0.1365` | `0.1139` | `0.0426` |
| `disc_mono` | `1.00` | `0.1732` | `0.1734` | `0.0792` |
| `neo` | `0.33` | `0.7035` | `0.3135` | `0.4493` |
| `neo` | `0.66` | `0.7679` | `0.7345` | `0.4608` |
| `neo` | `1.00` | `0.6750` | `0.6818` | `0.4113` |

完整逐 seed 结果：

```bash
/home/guian/L2T/.venv/bin/python /home/guian/L2T/scripts/summarize_gridworld.py \
  --log-dir /home/guian/L2T/logs/gridworld_full
```

汇总 CSV：

```text
/home/guian/L2T/logs/gridworld_full_summary.csv
```

## 反向长度泛化实验

目的：比较标准短训长测与反向长训短测。

- `standard_short_to_long`: train program length `1-3`，`length_ood` eval length `4-8`
- `reverse_long_to_short`: train program length `4-8`，`length_ood` eval length `1-3`
- 模型：`neo`、`neo_s`、`disc_mono`
- 参数：`alpha={0.66,1.00}`，`seed={0,1}`，`neo_s_samples=16`
- reduced CPU 设置：train `3000`，eval `500`，steps `600`，batch `128`，NEO/NEO-S `max_steps=8`

运行命令：

```bash
.venv/bin/python scripts/run_gridworld_length_direction_sweep.py \
  --models neo neo_s disc_mono \
  --alphas 0.66 1.0 \
  --seeds 0 1 \
  --neo-s-samples 16 \
  --train-size 3000 \
  --eval-size 500 \
  --steps 600 \
  --batch-size 128 \
  --max-steps 8 \
  --device cpu \
  --force
```

结果文件：

```text
logs/gridworld_length_direction_summary.csv
```

按 2 个 seed 聚合的 `length_ood` transfer accuracy：

| regime | model | alpha | train len | eval len | ID transfer | eval transfer | eval self |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard_short_to_long` | `disc_mono` | `0.66` | `1-3` | `4-8` | `0.0820` | `0.0260` | `0.2190` |
| `standard_short_to_long` | `disc_mono` | `1.00` | `1-3` | `4-8` | `0.1040` | `0.0260` | `0.1980` |
| `standard_short_to_long` | `neo` | `0.66` | `1-3` | `4-8` | `0.0990` | `0.0270` | `0.3780` |
| `standard_short_to_long` | `neo` | `1.00` | `1-3` | `4-8` | `0.0960` | `0.0370` | `0.3930` |
| `standard_short_to_long` | `neo_s` | `0.66` | `1-3` | `4-8` | `0.0850` | `0.0320` | `0.6070` |
| `standard_short_to_long` | `neo_s` | `1.00` | `1-3` | `4-8` | `0.0960` | `0.0400` | `0.6390` |
| `reverse_long_to_short` | `disc_mono` | `0.66` | `4-8` | `1-3` | `0.0240` | `0.0320` | `0.3390` |
| `reverse_long_to_short` | `disc_mono` | `1.00` | `4-8` | `1-3` | `0.0270` | `0.0160` | `0.2250` |
| `reverse_long_to_short` | `neo` | `0.66` | `4-8` | `1-3` | `0.0250` | `0.0250` | `0.4580` |
| `reverse_long_to_short` | `neo` | `1.00` | `4-8` | `1-3` | `0.0230` | `0.0210` | `0.2840` |
| `reverse_long_to_short` | `neo_s` | `0.66` | `4-8` | `1-3` | `0.0360` | `0.0230` | `0.6210` |
| `reverse_long_to_short` | `neo_s` | `1.00` | `4-8` | `1-3` | `0.0240` | `0.0230` | `0.4880` |

初步结论：在这个 reduced 配置下，长训短测没有改善 query transfer。长程序训练更难优化，`ID transfer` 明显下降；`NEO-S` 能提高 support self-explainability，但没有同步提高迁移准确率。这说明当前简化 NEO/NEO-S 仍存在 support 过拟合或 latent program 对齐不足的问题，后续应优先补更贴近论文的 VQ/straight-through、state grounding 和更长 GPU 训练。
