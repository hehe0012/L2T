# GridWorld test_plan.md 图表结果

三组实验均为 seed=42。SVG 图表可直接用浏览器打开：

- `gridworld_training_diagnostics.svg`：五类训练指标按 epoch 的曲线
- `gridworld_per_step_diagnostics.svg`：执行器逐步指标和 MDL score 曲线

详细五部分指标：
- `gridworld_codebook_all_metrics.svg`：使用率、熵、死码连续时长、码字相似度、更新幅度
- `gridworld_policy_all_metrics.svg`：策略熵、对 y 敏感度、对状态敏感度
- `gridworld_executor_all_metrics.svg`：潜范数、逐步 grounding、重构、z 敏感度
- `gridworld_mdl_all_metrics.svg`：k* 均值、k* 分布、各长度 score
- `gridworld_loss_all_metrics.svg`：raw/weighted 的 support、VQ、grounding、query 及比值

## 最终评测

| alpha | split | self-explainability | transfer accuracy | mean chosen length | fallback rate |
|---:|---|---:|---:|---:|---:|
| 0.33 | id | 0.9940 | 0.0390 | 2.972 | 0.0059 |
| 0.33 | comp_ood | 0.5239 | 0.0154 | 3.325 | 0.4764 |
| 0.33 | length_ood | 0.8723 | 0.0227 | 5.238 | 0.1276 |
| 0.66 | id | 0.9875 | 0.0179 | 3.397 | 0.0131 |
| 0.66 | comp_ood | 0.9067 | 0.0154 | 3.435 | 0.0937 |
| 0.66 | length_ood | 0.9800 | 0.0165 | 3.958 | 0.0199 |
| 1.00 | id | 0.9588 | 0.0309 | 3.018 | 0.0414 |
| 1.00 | length_ood | 0.9178 | 0.0223 | 4.185 | 0.0823 |
