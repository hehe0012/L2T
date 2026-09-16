# 复现计划：Learning to Theorize the World from Observation

目标论文：Learning to Theorize the World from Observation  
论文链接：https://arxiv.org/pdf/2605.03413

## 0. 仓库与环境检查

- 确认目标代码路径为 `/home/guian/data/repos/L2T`，该路径实际解析到 `/raid/guian/repos/L2T`。
- 如果已有官方或第三方开源代码，将代码克隆或复制到该目录。
- 如果没有可用开源代码，则在该目录下从零搭建复现工程。
- 在仓库中记录以下信息：
  - Python 版本
  - CUDA 版本
  - PyTorch 版本
  - 依赖锁定文件或安装命令
  - GPU 型号与显存
  - 每次实验对应的 commit hash

建议项目结构：

```text
repos/L2T/
  configs/
  data/
  checkpoints/
  logs/
  scripts/
  src/
  tests/
  plan.md
  REPRODUCTION.md
```

## 1. 最小复现目标

先从最小完整闭环开始：OTIB GridWorld。

核心任务形式：

- 给定一个 support 输入/输出样例，推断其背后的 latent program。
- 将推断出的 program 应用到 query 输入上。
- 判断生成的 query 输出是否正确。

不要一开始同时做所有领域。先复现 GridWorld 的主要趋势，再扩展到 Arithmetic，最后做 Image Editing。

## 2. GridWorld 阶段

实现或核对 GridWorld 版本的 OTIB 数据集。

论文设置：

- 网格大小：`10 x 10`
- 原子动作：`up`、`down`、`left`、`right`
- 训练 program 长度：`1-3`
- Length OOD program 长度：`4-8`
- 训练集规模：`100k`
- ID 评测集规模：`10k`
- Compositional OOD 评测集规模：`10k`
- Length OOD 评测集规模：`20k`

先跑 reduced smoke test：

- 训练样本：`1k`
- 评测样本：`200`
- 随机种子：`1`
- 目标：确认数据生成、模型 forward、训练循环和指标计算全部打通。

再跑完整 GridWorld 设置：

- 随机种子：`3`
- `alpha`：`0.33`、`0.66`、`1.00`
- 报告 ID、Compositional OOD、Length OOD 三类结果。

## 3. 模型实现

实现 NEO，包含以下组件：

- 状态编码器 `E_theta`
- 状态解码器 `D_theta`
- Programmer `q_phi(z | s, y)`
- Executor `f_theta(s, z)`
- 向量量化 latent action codebook
- 最多 `K` 步的 latent action rollout
- 使用最小描述长度原则选择最短有效解释

训练目标：

- 重建损失
- 向量量化损失
- 状态 grounding 损失
- 带 MDL 权重的 program 选择目标

GridWorld 优先对齐的超参数：

- 学习率：`5e-4`
- Batch size：`128`
- 训练 rollout 长度：`K = 4`
- Codebook size：`6`
- Grounding loss 权重：`0.1`
- VQ loss 权重：`1.0`
- MDL 权重：约 `0.95-1.0`
- Length OOD 评测 rollout 长度：`K = 10`

## 4. Baseline 实现

实现用于验证主结论的 baseline：

- `Disc-Mono`：用一个离散 VQ latent action 表示完整变换。
- `Cont-Mono`：用一个连续 VAE 风格 latent action 表示完整变换。
- `Cont-Mono-Opt`：测试时优化连续 latent。
- `NEO-S`：带测试时 program sampling 的 NEO。

对比指标：

- Self-explainability
- Transferability
- ID split
- Compositional OOD split
- Length OOD split

## 5. Arithmetic 阶段

GridWorld 跑通后，再复现 Arithmetic 领域。

论文设置：

- 原子操作：`x2`、`x3`、`x5`、`x7`
- 训练 program 长度：`1-3`
- Length OOD program 长度：`4-6`

重点关注：

- 学到的 latent code 是否能和 arithmetic primitive 对齐。
- `NEO-S` 是否提升长 program 泛化能力。
- Length OOD 的趋势是否与论文一致。

## 6. Image Editing 阶段

该阶段建议在 GridWorld 和 Arithmetic 稳定之后再做。

预期领域设置：

- 基于 CIFAR-10 图像
- 原子编辑操作：
  - 亮度调整
  - 色相调整
  - 水平翻转
  - 垂直翻转
  - 旋转
  - mask 相关编辑

该阶段训练成本预计最高。先使用较小图像规模或较小数据集做 smoke test，再扩展到完整实验。

## 7. 评测与日志

每次实验都需要记录：

- 配置文件路径
- 随机种子
- 数据集 split
- 训练和评测命令
- Checkpoint 路径
- Metrics JSON 或 CSV
- Wall-clock 运行时间
- 可获得时记录 GPU 显存占用

最终整理 `REPRODUCTION.md`，包含：

- 环境信息
- 完整运行命令
- 复现结果与论文结果对比表
- 已知偏差
- 失败案例
- 计算资源与耗时说明

## 8. 验收标准

最低验收：

- GridWorld 数据管线已实现。
- NEO 可以端到端训练。
- 在 reduced setting 中，NEO 在 Compositional OOD 或 Length OOD transfer 上优于 monolithic baseline。

较强验收：

- 完整 GridWorld 三个随机种子的趋势与论文定性一致。
- Arithmetic 复现完成。
- `NEO-S` 能提升长时域泛化能力。

完整验收：

- GridWorld、Arithmetic、Image Editing 三个领域全部复现。
- 主要表格和 qualitative alignment 分析都能重新生成。
- 所有命令和配置都记录在 `REPRODUCTION.md` 中。

## 9. 立即下一步

1. 已完成：确认当前目录没有官方代码，先按论文从零实现本地复现 scaffold。
2. 已完成：添加依赖文件和最小 Python 包结构。
3. 已完成：实现 GridWorld 数据生成。
4. 部分完成：已实现 transferability exact-match 指标；self-explainability 需要继续单独补齐。
5. 已完成：添加 smoke-test 训练脚本。
6. 部分完成：已实现 NEO 初始版本和 `Disc-Mono` baseline；仍需进一步对齐论文中的 VQ/MDL/state grounding 细节。
7. 已完成：已跑 CPU reduced smoke；下一步是在更长训练和 CUDA 环境中扩大实验。
