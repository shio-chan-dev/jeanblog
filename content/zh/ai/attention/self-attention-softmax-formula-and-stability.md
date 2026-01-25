---
title: "Self-Attention 计算公式与 Softmax 数值稳定：从推导到工程实现"
date: 2026-01-25T12:50:33+08:00
draft: false
categories: ["AI", "Multimodal"]
tags: ["attention", "transformer", "softmax", "numerical-stability", "pytorch"]
description: "用公式与可运行示例讲清 Self-Attention 的计算流程、softmax 的数值问题与工程实现要点。"
keywords: ["Self-Attention", "Softmax", "Scaled Dot-Product", "数值稳定", "Transformer"]
---

> **副标题 / 摘要**  
> Self-Attention 的公式很短，但工程细节很长：从 Q/K/V 计算到 softmax 数值稳定、mask 与缩放，每一步都影响效果与性能。本文用 ACERS 结构给出推导、实践步骤与可运行示例。

- **预计阅读时长**：12~16 分钟
- **标签**：`attention`、`transformer`、`softmax`
- **SEO 关键词**：Self-Attention, Softmax, Scaled Dot-Product, 数值稳定
- **元描述**：Self-Attention 的计算公式与 softmax 稳定实现方法，含工程实践与示例代码。

---

## 目标读者

- 想真正理解 Self-Attention 公式含义的学习者
- 需要处理训练不稳定/溢出的工程实践者
- 关注注意力数值稳定与实现细节的开发者

## 背景 / 动机

在 Transformer 中，Self-Attention 是计算量最大、数值最敏感的模块之一。  
很多训练不稳定、输出 NaN 的问题，都来自 softmax 的溢出/下溢或 mask 的错误处理。  
理解公式与稳定实现，可以显著减少工程“踩坑”。

## 核心概念

- **Q/K/V**：查询、键和值，来自输入线性投影
- **缩放点积注意力**：$\text{softmax}(QK^\top/\sqrt{d_k})V$
- **数值稳定**：通过减去行最大值避免 softmax 溢出

---

## 思路推导（从朴素到稳定实现）

### 朴素做法

先算所有相似度 $S = QK^\top$，再做 softmax 得到权重 $P$，最后 $O = PV$。  
这个实现最直观，但当 $S$ 很大时会出现 `exp` 溢出。

### 关键观察

softmax 对每行同时加上或减去一个常数不改变输出：  
$\text{softmax}(x) = \text{softmax}(x - \max(x))$。

### 稳定实现

对每行减去最大值，再计算指数和归一化，可以在不改变结果的情况下避免溢出。  
这就是工程里常见的“减 max”策略。

---

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

Self-Attention 的核心是：  
1) 计算 token 之间的相似度；  
2) 用 softmax 转成概率；  
3) 用概率加权汇总 V。

### 关键公式

给定输入 $X \in \mathbb{R}^{T\times d}$：

- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- $S = QK^\top / \sqrt{d_k}$
- $P = \text{softmax}(S)$
- $O = PV$

### 基础示例

假设 $T=3$，可以手算 3x3 的注意力分布，并观察 softmax 的归一化效果。

---

## C — Concepts（核心思想）

### 方法归类

- 矩阵乘法
- 归一化（softmax）
- 加权求和

### 关键公式与模型

- **缩放因子**：$1/\sqrt{d_k}$ 控制数值尺度
- **稳定 softmax**：$\exp(x - \max(x)) / \sum\exp(x - \max(x))$

### 直观解释

注意力权重是“相似度排序后的概率分布”。  
缩放与稳定 softmax 是为了让这个分布既可训练又可计算。

---

## 实践指南 / 步骤

1. 线性投影得到 Q/K/V
2. 计算缩放点积 $S = QK^\top / \sqrt{d_k}$
3. 对 $S$ 做“减 max”的稳定 softmax
4. 权重 $P$ 乘以 $V$ 得到输出
5. 处理 mask（padding 或 causal）

## 可运行示例（稳定 softmax 的 Self-Attention）

```python
import numpy as np


def stable_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(x, wq, wk, wv):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    dk = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(dk)
    probs = stable_softmax(scores, axis=-1)
    return probs @ v


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(3, 4)
    wq = np.random.randn(4, 4)
    wk = np.random.randn(4, 4)
    wv = np.random.randn(4, 4)
    out = self_attention(x, wq, wk, wv)
    print(out)
```

---

## E — Engineering（工程应用）

### 场景 1：混合精度训练的溢出控制（Python）

**背景**：FP16/bfloat16 下 softmax 更容易溢出。  
**为什么适用**：减 max 能显著缓解溢出。  

```python
scores = scores - scores.max(axis=-1, keepdims=True)
probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
```

### 场景 2：大序列的 mask 处理（Python）

**背景**：padding 与 causal mask 常导致负无穷输入。  
**为什么适用**：先加 mask，再做稳定 softmax。  

```python
scores = scores + mask  # mask 中 padding 位置为 -1e9
probs = stable_softmax(scores, axis=-1)
```

### 场景 3：工程排查与诊断（Python）

**背景**：出现 NaN 时定位 softmax 数值溢出。  
**为什么适用**：检查 softmax 输入范围。  

```python
print(scores.max(), scores.min())
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：$O(T^2 d)$
- **空间复杂度**：$O(T^2)$（注意力矩阵）

### 替代方案对比

| 方案 | 优点 | 风险 |
| --- | --- | --- |
| 朴素 softmax | 实现简单 | 容易溢出 | 
| 减 max 稳定 softmax | 稳定性高 | 需多一步计算 |
| 近似注意力 | 降低复杂度 | 可能影响精度 |

### 为什么当前方法最工程可行

稳定 softmax 在计算成本很小的情况下解决了最常见的数值问题，  
是工程实践中的默认选择。

---

## 解释与原理（为什么这么做）

softmax 的指数运算非常敏感，减去最大值可以把最大输入移动到 0，  
避免指数爆炸，同时保持概率分布不变。

---

## 常见问题与注意事项

1. **为什么要除以 $\sqrt{d_k}$？**  
   防止点积过大导致 softmax 过于尖锐。

2. **mask 应该在 softmax 前还是后？**  
   必须在 softmax 前加上负无穷，否则概率仍会分配到无效位置。

3. **softmax 仍然可能溢出吗？**  
   如果没有减 max 或者分布极端，仍可能溢出。

---

## 最佳实践与建议

- softmax 前 반드시减去行最大值
- 大序列与混合精度下要监控数值范围
- mask 的数值用 -1e9 或 -inf 并在 softmax 前加

---

## S — Summary（总结）

### 核心收获

- Self-Attention 的核心公式是 $\text{softmax}(QK^\top/\sqrt{d_k})V$
- softmax 数值稳定需要“减 max”
- mask 必须在 softmax 前处理
- 这些细节决定了训练稳定性与工程可靠性

### 小结 / 结论

理解公式是起点，掌握稳定实现才是工程落地关键。  
如果你在训练中遇到 NaN，优先检查 softmax 输入范围。

### 参考与延伸阅读

- https://arxiv.org/abs/1706.03762
- https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
- https://en.wikipedia.org/wiki/Softmax_function

---

## 元信息

- **阅读时长**：12~16 分钟
- **标签**：attention、transformer、softmax
- **SEO 关键词**：Self-Attention, Softmax, 数值稳定
- **元描述**：Self-Attention 公式与 softmax 数值稳定实现要点。

---

## 行动号召（CTA）

建议你用本文代码写一个最小注意力模块，  
把稳定 softmax 与 mask 处理封装成可复用函数。
