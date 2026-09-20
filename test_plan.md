现在的情况已经积累了大量线索，需要一次系统性的诊断来收敛。下面按**训练时监控**、**训练后诊断**、**端到端验证**三个层次组织，每一节都说明“测什么、怎么测、怎么判读”。

---

## 一、训练过程中需要监控的量

这些量应该在训练循环中每 N 步记录一次，画成曲线。它们能告诉你训练是否健康、坍缩从什么时候开始。

### 1. Codebook 相关

| 指标 | 计算方式 | 判读 |
| :--- | :--- | :--- |
| **Code 使用直方图** | `bincount(z_indices, minlength=6)` | 是否均匀？如果某个 code 占比 > 50%，坍缩已经发生 |
| **Code 熵** | \(-\sum p_i \log p_i\) | 熵接近 log(6)≈1.79 为均匀；熵 < 0.5 说明坍缩 |
| **死码字数量** | 连续 N 步未被使用的 code 数 | 如果 > 0，说明码本退化 |
| **码字间余弦相似度矩阵** | 归一化后码字两两内积 | 如果所有码字与 U 的相似度 > 0.5，方向坍缩 |
| **码字更新幅度** | `||codebook_new - codebook_old||` | 如果长期接近 0，码字已经冻结 |

### 2. 策略相关

| 指标 | 计算方式 | 判读 |
| :--- | :--- | :--- |
| **策略输出熵** | 每步 `-(p log p).sum()` 的平均 | 熵高说明探索；熵低说明策略已确定 |
| **策略输出在 primitive 上的条件方差** | 对同一 primitive 的样本，a_pre 的方差 | 方差大 = 策略把位置信息编码进了 a_pre |
| **策略对 y 的敏感度** | 把 y 替换成随机 y'，看 a_pre 变化 | 变化大 = 策略依赖 y；变化小 = 策略忽略 y |
| **策略对 s_k 的敏感度** | 固定 y，改变 s_k，看 a_pre 变化 | **这是关键**。如果变化大，说明策略输出不是纯变换 |

### 3. 执行器相关

| 指标 | 计算方式 | 判读 |
| :--- | :--- | :--- |
| **Rollout 潜在范数** | 每步 `||s_k||` 的均值 | 如果随 k 爆炸（如 47 → 31145），流形漂移 |
| **每步 grounding loss** | `||s_k - sg[E(D(s_k))]||^2` | 如果某步突然变大，说明该步漂移 |
| **每步重构 loss** | `CE(D(s_k), y)` | 如果第 2 步就变大，说明多步 rollout 失败 |
| **执行器对 z 的敏感度** | 固定 s_k，替换 z，看输出差异 | 差异小 = 执行器忽略 z |

### 4. MDL 相关

| 指标 | 计算方式 | 判读 |
| :--- | :--- | :--- |
| **平均 k\*** | 每个 batch 的 `k*` 均值 | 应接近 GT 长度（2.46）。如果偏离，MDL 有问题 |
| **k\* 分布** | k* 的直方图 | 如果全部集中在 1 或 K，说明 MDL 失效 |
| **Score(k) 曲线** | 对单个样本画出 Score(k) vs k | 看 MDL 是否真的在惩罚长程序 |

### 5. 损失相关

| 指标 | 计算方式 | 判读 |
| :--- | :--- | :--- |
| **L_support** | 重构损失 | 应该稳定下降 |
| **L_vq (codebook + commitment)** | VQ 损失 | 如果持续高，码字不稳定 |
| **L_ground** | 接地损失 | 如果突然变大，说明漂移 |
| **L_query (如果启用)** | query 损失 | 如果远大于 L_support，说明过拟合 support |
| **L_support / L_query 的比值** | 训练/验证对比 | 比值大 = 支持集过拟合 |

---

## 二、训练完成后的诊断测试

这些测试在训练结束后运行，用来精确定位问题。

### A. Codebook 诊断

#### A1. Code-Primitive 对齐矩阵

```python
# 对每个 code，统计它在 GT primitive 上的使用分布
C[i, j] = #(code i 被用于 GT primitive j)
```

**判读**：
- 近对角结构 → 码本语义清晰（论文 Figure 9）。
- 全集中在一列 → 方向坍缩。
- 分散 → 语义不一致。

#### A2. 每个 code 的位移多样性

```python
for code in range(6):
    displacements = [decode(transition(s_k, code)) - decode(s_k)
                     for s_k in 100_states]
    unique_displacements = len(set(displacements))
```

**判读**：
- 1–2 种 → code 是纯变换。
- 35–41 种 → code 是 state-dependent（你当前的情况）。

#### A3. 位移一致性

```python
for primitive in [U, D, L, R]:
    d_list = [E(y) - E(x) for (x, y) in dataset[primitive]]
    pairwise_cos = mean([cosine(d_i, d_j)])
    norm_std = std([||d||])
```

**判读**：
- pairwise_cos > 0.5 → VAE 位移一致。
- norm_std / norm_mean < 0.5 → 位移长度稳定。

#### A4. 码字间关系

```python
code_norm = normalize(codebook)
gram = code_norm @ code_norm.T
```

**判读**：如果所有码字与某个码字的相似度 > 0.5，说明码字没有分化。

### B. 执行器诊断

#### B1. Oracle 单步准确率

```python
for (x, y, primitive) in dataset:
    s_next = transition(E(x), code_for(primitive))
    if decode(s_next) == y: correct += 1
```

**判读**：
- > 0.9 → 执行器单步没问题。
- 你之前测过是 1.0 → 单步正常。

#### B2. Oracle 多步 rollout

```python
s = E(x)
for k, primitive in enumerate(gt_program):
    s = transition(s, code_for(primitive))
    # 记录每步准确率
```

**判读**：
- 第 1 步 1.0，第 2 步 0.313 → exposure bias。
- 你之前测过 → 确认多步失败。

#### B3. Oracle 多步 + 每步 grounding

```python
s = E(x)
for k, primitive in enumerate(gt_program):
    s = transition(s, code_for(primitive))
    s = E(D(s))  # 强制回流形
    # 记录每步准确率
```

**判读**：
- 如果每步 grounding 后准确率恢复到 1.0 → 执行器 + VAE 本身没问题，问题是流形漂移。
- 你之前测过 → 确认。

#### B4. 执行器对 z 的敏感度

```python
for s_k in sample_states:
    outputs = [transition(s_k, code) for code in range(6)]
    pairwise_dist = mean([||outputs[i] - outputs[j]|| for i,j])
```

**判读**：
- 距离大 → 执行器使用 z。
- 距离小 → 执行器忽略 z。

#### B5. 潜空间漂移曲线

```python
s = E(x)
norms = [||s||]
for k in range(8):
    s = transition(s, gt_code[k])
    norms.append(||s||)
```

**判读**：范数应该在合理范围内波动，不应指数增长。

### C. 策略诊断

#### C1. 策略对 s_k 的敏感度

```python
# 固定 primitive，改变 s_k，看策略输出
for primitive in [U, D, L, R]:
    a_pre_list = [policy(E(x), E(y)) for (x, y) in dataset[primitive]]
    variance = mean([||a_pre - mean(a_pre)||^2])
```

**判读**：
- 方差小 → 策略学到“变换无关位置”。
- 方差大 → 策略把位置信息编码进 a_pre（**这可能是你的核心问题**）。

#### C2. 策略对 y 的敏感度

```python
a_pre_normal = policy(E(x), E(y))
a_pre_random = policy(E(x), E(y'))  # y' 是随机目标
diff = ||a_pre_normal - a_pre_random||
```

**判读**：
- diff 大 → 策略依赖 y（正常）。
- diff 小 → 策略忽略 y（异常）。

#### C3. 策略输出熵

```python
probs = softmax(policy_logits(s_k, s_y))
entropy = -(probs * log(probs)).sum()
```

**判读**：
- 熵接近 log(6) → 策略在探索。
- 熵接近 0 → 策略已坍缩。

### D. VAE 诊断

#### D1. 重建准确率

```python
for x in dataset:
    if decode(E(x)) == x: correct += 1
```

**判读**：应 > 0.99。

#### D2. 位移一致性（VAE 级别）

见 A3。

#### D3. 线性探测

```python
# 用 E(y) - E(x) 线性分类 primitive
from sklearn.linear_model import LogisticRegression
X = [E(y) - E(x) for (x, y, a) in dataset]
y = [a for (x, y, a) in dataset]
clf = LogisticRegression().fit(X, y)
accuracy = clf.score(X, y)
```

**判读**：
- 准确率 > 0.9 → latent 位移线性可分，VAE 几何好。
- 准确率接近 0.25 → latent 位移不可分，VAE 几何差。

#### D4. 潜空间范数分布

```python
norms = [||E(x)|| for x in dataset]
print(mean(norms), std(norms), max(norms))
```

**判读**：如果范数分布很宽，流形可能不平坦。

### E. 端到端诊断

#### E1. 用 GT 程序 + GT code 的迁移

```python
# 用 GT primitive 对应的 code，看 query 迁移
for (x_s, y_s, x_q, y_q, gt_program) in dataset:
    s = E(x_s)
    for primitive in gt_program:
        s = transition(s, code_for(primitive))
    # 用推断出的 code 序列应用到 query
    s_q = E(x_q)
    for primitive in gt_program:
        s_q = transition(s_q, code_for(primitive))
    if decode(s_q) == y_q: correct += 1
```

**判读**：
- 如果这个准确率高，说明执行器 + code 语义是对的，问题在策略。
- 你之前测过 → 1–4%，说明执行器 + code 语义本身就有问题。

#### E2. 用 GT 程序 + 策略推断的 code 的迁移

```python
# 用策略推断的 code 序列，但用 GT primitive 的 program 长度
```

**判读**：隔离“策略选错 code”和“执行器无法执行 code”。
