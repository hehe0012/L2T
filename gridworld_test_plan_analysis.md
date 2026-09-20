# GridWorld test_plan.md A-E 训练后诊断

本文件只保留关键指标、图表链接和结论；完整原始诊断 JSON 位于：

`/home/guian/L2T/logs/gridworld_test_plan_post_seed42/`

## 图表

| 部分 | 图表 | 重点指标 |
|---|---|---|
| A | [alignment](logs/gridworld_test_plan_post_seed42/charts/post_A_alignment.svg)、[displacement](logs/gridworld_test_plan_post_seed42/charts/post_A_displacement.svg) | 原语性、位移多样性、code 一致性 |
| B | [oracle rollout](logs/gridworld_test_plan_post_seed42/charts/post_B_oracle_rollout.svg) | 单步/多步、free vs decode-reencode |
| C | [policy](logs/gridworld_test_plan_post_seed42/charts/post_C_policy.svg) | 熵、目标敏感度、状态敏感度 |
| D | [VAE](logs/gridworld_test_plan_post_seed42/charts/post_D_vae.svg) | 重建、位移线性探测、latent 范数 |
| E | [end-to-end](logs/gridworld_test_plan_post_seed42/charts/post_E_end_to_end.svg) | GT code 与 policy code 的 query transfer |

## 关键数值

| alpha | A primitiveness | B one-step oracle | B length-4 free | B length-4 reencode | C policy entropy | C y/state sensitivity | D recon | D linear probe | E1 ID query | E2 ID query |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.33 | 0.0375 | 0.0300 | 0.0141 | 0.0144 | 0.2887 | 2.271/1.004 | 1.0000 | 0.2525 | 0.0360 | 0.0360 |
| 0.66 | 0.0275 | 0.0200 | 0.0167 | 0.0106 | 0.4767 | 1.644/0.624 | 1.0000 | 0.2525 | 0.0200 | 0.0190 |
| 1.00 | 0.1250 | 0.0675 | 0.0206 | 0.0194 | 0.3381 | 1.334/0.640 | 1.0000 | 0.2525 | 0.0300 | 0.0290 |

## 总结

- A：原语性仅 `0.0275--0.1250`，code 在 100 个状态上产生大量不同位移，未形成稳定 primitive。
- B：GT code 的单步和多步 oracle 都很低，decode-reencode 没有恢复准确率，说明主要瓶颈在 VAE latent geometry / executor，而不是单纯的 policy 选码。
- C：policy 对 `y` 敏感，但对 `s_k` 也敏感；因此存在位置相关的 code 选择，不过这不是主要失败点。
- D：VAE 重建为 `1.0`，但位移线性探测为 `0.2525`，接近四分类随机水平；VAE 能记住位置，却没有提供可组合的 primitive 几何。
- E：GT program + GT code 的 query accuracy 只有 `0.020--0.036`，换成 policy code 后几乎不变，进一步排除 policy 是首要瓶颈。

**最终定位：** 先修复 state latent 的位移几何和 executor 的 primitive 对齐，再调 policy、MDL 或采样数。

# 逐项诊断结果

## A1. Code-Primitive Alignment

每个单元格为该 code 在 100 个起始状态上与 primitive 完全匹配的次数 / 100。

| alpha | code | U | D | L | R |
|---:|---:|---:|---:|---:|---:|
| 0.33 | 0 | 0.06 | 0.00 | 0.00 | 0.03 |
| 0.33 | 1 | 0.07 | 0.00 | 0.00 | 0.02 |
| 0.33 | 2 | 0.07 | 0.00 | 0.00 | 0.01 |
| 0.33 | 3 | 0.07 | 0.00 | 0.00 | 0.01 |
| 0.33 | 4 | 0.08 | 0.00 | 0.00 | 0.01 |
| 0.33 | 5 | 0.09 | 0.00 | 0.00 | 0.03 |
| 0.66 | 0 | 0.02 | 0.01 | 0.01 | 0.03 |
| 0.66 | 1 | 0.02 | 0.01 | 0.02 | 0.02 |
| 0.66 | 2 | 0.02 | 0.01 | 0.02 | 0.02 |
| 0.66 | 3 | 0.02 | 0.01 | 0.02 | 0.02 |
| 0.66 | 4 | 0.02 | 0.01 | 0.02 | 0.02 |
| 0.66 | 5 | 0.02 | 0.01 | 0.02 | 0.02 |
| 1.00 | 0 | 0.04 | 0.06 | 0.06 | 0.06 |
| 1.00 | 1 | 0.06 | 0.07 | 0.08 | 0.07 |
| 1.00 | 2 | 0.05 | 0.08 | 0.05 | 0.06 |
| 1.00 | 3 | 0.04 | 0.05 | 0.07 | 0.03 |
| 1.00 | 4 | 0.04 | 0.04 | 0.04 | 0.04 |
| 1.00 | 5 | 0.02 | 0.06 | 0.05 | 0.05 |

**A1 结果：** 四个 primitive 的最佳 code 覆盖率仍很低；这不是近似的近对角 alignment。详见 [A alignment 图](logs/gridworld_test_plan_post_seed42/charts/post_A_alignment.svg)。

## A2. 每个 code 的位移多样性

`unique_displacements` 越接近 1 越好；`consistency` 是该 code 产生其主导位移的比例。

| alpha | code | unique displacements | consistency | dominant displacement |
|---:|---:|---:|---:|---|
| 0.33 | 0 | 46 | 0.070 | [2, 9] |
| 0.33 | 1 | 42 | 0.100 | [0, 7] |
| 0.33 | 2 | 39 | 0.110 | [0, 7] |
| 0.33 | 3 | 37 | 0.100 | [0, 7] |
| 0.33 | 4 | 34 | 0.110 | [1, 1] |
| 0.33 | 5 | 37 | 0.090 | [9, 0] |
| 0.66 | 0 | 62 | 0.040 | [2, 0] |
| 0.66 | 1 | 73 | 0.020 | [8, 2] |
| 0.66 | 2 | 74 | 0.030 | [9, 8] |
| 0.66 | 3 | 73 | 0.020 | [8, 2] |
| 0.66 | 4 | 74 | 0.020 | [8, 2] |
| 0.66 | 5 | 74 | 0.020 | [8, 2] |
| 1.00 | 0 | 42 | 0.060 | [1, 1] |
| 1.00 | 1 | 36 | 0.080 | [0, 9] |
| 1.00 | 2 | 43 | 0.080 | [1, 0] |
| 1.00 | 3 | 48 | 0.070 | [0, 9] |
| 1.00 | 4 | 47 | 0.050 | [2, 1] |
| 1.00 | 5 | 44 | 0.060 | [1, 0] |

**A2 结果：** code 通常产生 `34--74` 种位移，主导位移比例最高也只有约 `0.11`，说明 latent action 不是稳定 primitive。详见 [A displacement 图](logs/gridworld_test_plan_post_seed42/charts/post_A_displacement.svg)。

## A3. Primitive 位移一致性

`pairwise_cosine_to_mean` 越高越好；`relative_norm_std` 越低越好。

| alpha | primitive | cosine to mean | norm mean | norm std | relative norm std |
|---:|---|---:|---:|---:|---:|
| 0.33 | up | 0.0010 | 19.337 | 5.262 | 0.272 |
| 0.33 | down | -0.0022 | 19.337 | 5.262 | 0.272 |
| 0.33 | left | -0.0011 | 21.604 | 5.180 | 0.240 |
| 0.33 | right | 0.0008 | 21.604 | 5.180 | 0.240 |
| 0.66 | up | 0.0010 | 19.337 | 5.262 | 0.272 |
| 0.66 | down | -0.0022 | 19.337 | 5.262 | 0.272 |
| 0.66 | left | -0.0011 | 21.604 | 5.180 | 0.240 |
| 0.66 | right | 0.0008 | 21.604 | 5.180 | 0.240 |
| 1.00 | up | 0.0010 | 19.337 | 5.262 | 0.272 |
| 1.00 | down | -0.0022 | 19.337 | 5.262 | 0.272 |
| 1.00 | left | -0.0011 | 21.604 | 5.180 | 0.240 |
| 1.00 | right | 0.0008 | 21.604 | 5.180 | 0.240 |

**A3 结果：** 四个 primitive 的 cosine 都约为 `0`，而不是计划中的 `>0.5`；位移方向完全不一致。长度方差相对较小，但方向问题更严重。

## A4. 码字间关系

| alpha | codebook cosine mean | codebook cosine max |
|---:|---:|---:|
| 0.33 | 0.0953 | 0.8689 |
| 0.66 | 0.1980 | 0.6174 |
| 1.00 | 0.1884 | 0.5250 |

**A4 结果：** alpha=0.33 的最大余弦相似度达到 `0.869`，存在高度相似码字；alpha=0.66/1.00 也未形成清晰的正交 primitive 结构。

## B1. Oracle 单步准确率

| alpha | U | D | L | R | mean |
|---:|---:|---:|---:|---:|---:|
| 0.33 | 0.0900 | 0.0000 | 0.0000 | 0.0300 | 0.0300 |
| 0.66 | 0.0200 | 0.0100 | 0.0200 | 0.0300 | 0.0200 |
| 1.00 | 0.0600 | 0.0800 | 0.0700 | 0.0600 | 0.0675 |

**B1 结果：** oracle code 的单步准确率只有 `0.020--0.068`，远低于正常执行器应有的高准确率。

## B2/B3. Oracle 多步 rollout

下表为 `free latent rollout / 每步 decode-reencode` 的最终准确率。

| alpha | len=1 | len=2 | len=3 | len=4 |
|---:|---:|---:|---:|---:|
| 0.33 | 0.0300 / 0.0300 | 0.0213 / 0.0250 | 0.0181 / 0.0145 | 0.0141 / 0.0144 |
| 0.66 | 0.0200 / 0.0200 | 0.0131 / 0.0131 | 0.0181 / 0.0122 | 0.0167 / 0.0106 |
| 1.00 | 0.0675 / 0.0675 | 0.0312 / 0.0288 | 0.0266 / 0.0242 | 0.0206 / 0.0194 |

**B2/B3 结果：** decode-reencode 没有把准确率恢复到高水平，说明问题不是单纯的 free rollout 漂移，而是 code/transition 与 VAE 状态几何本身不匹配。

## B4. Executor 对 z 的敏感度

| alpha | mean pairwise latent distance | mean unique decoded states | all-codes-same fraction |
|---:|---:|---:|---:|
| 0.33 | 7.692 | 2.210 | 0.250 |
| 0.66 | 6.685 | 1.640 | 0.390 |
| 1.00 | 10.525 | 2.870 | 0.040 |

**B4 结果：** executor 对 z 有明显响应，但不同 code 的输出不能稳定对应四个 primitive；因此不是简单的 z 被忽略。

## B5. 潜空间漂移

| alpha | length-4 step1 norm mean | step2 | step3 | step4 |
|---:|---:|---:|---:|---:|
| 0.33 | 17.98 | 15.15 | 12.19 | 12.04 |
| 0.66 | 21.67 | 20.04 | 10.46 | 10.51 |
| 1.00 | 16.19 | 12.14 | 9.18 | 9.51 |

**B5 结果：** 本次没有出现范数指数爆炸；但范数稳定不代表状态在正确 manifold 上，B1-B3 的低准确率已经证明 dynamics 仍不可用。

## C1. 策略对 s_k 的敏感度 / primitive 条件方差

| alpha | primitive | a_pre conditional variance |
|---:|---|---:|
| 0.33 | up | 3.0752 |
| 0.33 | down | 2.8678 |
| 0.33 | left | 2.8760 |
| 0.33 | right | 2.8119 |
| 0.66 | up | 2.1689 |
| 0.66 | down | 2.0798 |
| 0.66 | left | 1.9076 |
| 0.66 | right | 1.9247 |
| 1.00 | up | 1.4872 |
| 1.00 | down | 1.4934 |
| 1.00 | left | 1.7718 |
| 1.00 | right | 1.6522 |

**C1 结果：** 同一 primitive 的 `a_pre` 仍随状态变化，说明策略输出不是位置无关的纯变换。

## C2. 策略对 y 的敏感度

| alpha | y sensitivity | state sensitivity | y/state ratio |
|---:|---:|---:|---:|
| 0.33 | 2.271 | 1.004 | 2.26 |
| 0.66 | 1.644 | 0.624 | 2.63 |
| 1.00 | 1.334 | 0.640 | 2.08 |

**C2 结果：** policy 没有忽略 y，但 state sensitivity 也不可忽略；alpha=0.33 的位置依赖最强。

## C3. 策略输出熵

| alpha | output entropy |
|---:|---:|
| 0.33 | 0.2887 |
| 0.66 | 0.4767 |
| 1.00 | 0.3381 |

**C3 结果：** 熵仅 `0.289--0.477`，策略已经相当确定，探索不足。

## D1. VAE 重建准确率

| alpha | reconstruction accuracy |
|---:|---:|
| 0.33 | 1.0000 |
| 0.66 | 1.0000 |
| 1.00 | 1.0000 |

**D1 结果：** 三个 checkpoint 都能完美重建 100 个状态；问题不是 VAE 记不住状态，而是 latent 几何不可组合。

## D2. VAE 位移一致性

D2 与 A3 使用同一组四 primitive 的 `E(y)-E(x)` 统计，详见 [VAE 图](logs/gridworld_test_plan_post_seed42/charts/post_D_vae.svg) 和 A3 表。结果是方向 cosine 约为 `0`。

## D3. 位移线性探测

| alpha | linear probe accuracy | 四分类随机基线 |
|---:|---:|---:|
| 0.33 | 0.2525 | 0.2500 |
| 0.66 | 0.2525 | 0.2500 |
| 1.00 | 0.2525 | 0.2500 |

**D3 结果：** 几乎等于随机基线，latent displacement 不包含可线性分离的 primitive 方向。

## D4. 潜空间范数分布

| alpha | mean | std | max |
|---:|---:|---:|---:|
| 0.33 | 17.649 | 3.844 | 26.856 |
| 0.66 | 17.649 | 3.844 | 26.856 |
| 1.00 | 17.649 | 3.844 | 26.856 |

**D4 结果：** state latent 范数分布本身不算发散，但这不能弥补 D2/D3 的方向不可分问题。

## E1. GT program + GT code

| alpha | split | support accuracy | query transfer accuracy |
|---:|---|---:|---:|
| 0.33 | id | 0.0310 | 0.0360 |
| 0.33 | comp_ood | 0.0090 | 0.0030 |
| 0.33 | length_ood | 0.0140 | 0.0120 |
| 0.66 | id | 0.0180 | 0.0200 |
| 0.66 | comp_ood | 0.0190 | 0.0210 |
| 0.66 | length_ood | 0.0100 | 0.0120 |
| 1.00 | id | 0.0340 | 0.0300 |
| 1.00 | length_ood | 0.0075 | 0.0170 |

**E1 结果：** 即使直接提供真实程序和最佳 GT code mapping，query transfer 仍只有约 `0.003--0.036`，执行器和 latent code 本身已经失败。

## E2. GT program length + policy inferred code

| alpha | split | support accuracy | query transfer accuracy |
|---:|---|---:|---:|
| 0.33 | id | 0.2300 | 0.0360 |
| 0.33 | comp_ood | 0.0660 | 0.0070 |
| 0.33 | length_ood | 0.1740 | 0.0110 |
| 0.66 | id | 0.2140 | 0.0190 |
| 0.66 | comp_ood | 0.1810 | 0.0190 |
| 0.66 | length_ood | 0.3840 | 0.0125 |
| 1.00 | id | 0.2870 | 0.0290 |
| 1.00 | length_ood | 0.2475 | 0.0145 |

**E2 结果：** E2 与 E1 的 query accuracy 几乎相同，说明额外替换为 policy code 并没有造成主要损失；主因在 executor/VAE，而不是 policy 选错程序。
