---
title: "FlashAttention 为什么能 one-pass：在线 softmax 与 Tiling 的核心思想"
date: 2026-01-25T12:51:14+08:00
draft: false
categories: ["AI", "Multimodal"]
tags: ["flash-attention", "attention", "tiling", "gpu", "memory"]
description: "解释 FlashAttention 的 one-pass 计算原理与 tiling 策略，并给出可运行的块级注意力示例。"
keywords: ["FlashAttention", "Tiling", "Online Softmax", "One-pass", "Attention"]
---

> **副标题 / 摘要**  
> softmax 本身看似需要多遍遍历，但 FlashAttention 通过在线 softmax 与分块（tiling）把注意力计算变成“边读边算”，不再显式存储 $QK^\top$。本文拆解其核心思想与工程实现。

- **预计阅读时长**：14~20 分钟
- **标签**：`flash-attention`、`tiling`、`gpu`
- **SEO 关键词**：FlashAttention, Tiling, One-pass, Online Softmax
- **元描述**：FlashAttention 的 one-pass 原理与 tiling 策略解析。

---

## 目标读者

- 想理解 FlashAttention 原理的工程读者
- 关注 GPU 内存带宽与注意力瓶颈的优化者
- 需要实现块级注意力的开发者

## 背景 / 动机

标准注意力需要显式构造 $QK^\top$，  
这在长序列上造成 $O(T^2)$ 内存占用与访存压力。  
FlashAttention 的目标是：**不存注意力矩阵，但得到同样的结果**。

## 核心概念

- **在线 softmax**：边遍历边更新 max 与 sum
- **Tiling**：把 Q/K/V 切成小块，放入共享内存计算
- **不落地中间矩阵**：避免 $QK^\top$ 写回显存

---

## 思路推导（从全量矩阵到分块流式）

### 朴素做法

1) 计算全量 $S = QK^\top$  
2) softmax 得到 $P$  
3) $O = PV$  

这需要存储 $S$ 和 $P$，开销为 $O(T^2)$。

### 关键观察

softmax 可以在线更新，且 $O$ 可以在扫描 K/V 块时逐步累积，  
无需完整保存 $S$。

### 方案选择

对 Q 的每个块，按 K/V 块顺序扫描：  
- 计算局部 $S$  
- 更新在线 softmax 统计量  
- 累积输出 $O$  

最终得到与全量注意力相同的结果。

---

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

FlashAttention 把“先算完再 softmax”的步骤改成“边算边 softmax”，  
把内存瓶颈变成寄存器/共享内存中的局部计算。

### 核心公式（在线 softmax）

维护 $m$ 与 $l$：

- $m' = \max(m, s_{ij})$
- $l' = l \cdot \exp(m - m') + \sum\exp(s_{ij} - m')$
- $o' = o \cdot \exp(m - m') + \sum\exp(s_{ij} - m') v_j$

---

## C — Concepts（核心思想）

### 方法归类

- IO-aware 算法
- 在线 softmax
- 块级矩阵乘法

### 直观解释

把 $Q$ 的一个块固定住，K/V 分块流过，  
输出在流过过程中逐步累积，  
这样就不需要把完整 $QK^\top$ 存到显存。

---

## 实践指南 / 步骤

1. 选择 tile 大小（适配共享内存）
2. 对 Q 的块进行循环
3. 依次加载 K/V 块，计算局部 scores
4. 用在线 softmax 更新输出
5. 输出最终 O

## 可运行示例（块级注意力，等价于全量）

```python
import numpy as np


def flash_attention_block(q, k, v, block=2):
    n, d = q.shape
    out = np.zeros((n, d))
    for i in range(n):
        m = -np.inf
        l = 0.0
        o = np.zeros(d)
        for start in range(0, n, block):
            kb = k[start:start + block]
            vb = v[start:start + block]
            scores = q[i] @ kb.T / np.sqrt(d)
            m_new = max(m, scores.max(initial=-np.inf))
            l = l * np.exp(m - m_new) + np.exp(scores - m_new).sum()
            o = o * np.exp(m - m_new) + (np.exp(scores - m_new)[:, None] * vb).sum(axis=0)
            m = m_new
        out[i] = o / l
    return out


def naive_attention(q, k, v):
    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return probs @ v


if __name__ == "__main__":
    np.random.seed(0)
    q = np.random.randn(4, 4)
    k = np.random.randn(4, 4)
    v = np.random.randn(4, 4)
    print(flash_attention_block(q, k, v, block=2))
    print(naive_attention(q, k, v))
```

---

## E — Engineering（工程应用）

### 场景 1：长序列 Transformer（GPU）

**背景**：序列长度上千时，注意力矩阵无法承受。  
**为什么适用**：FlashAttention 不存 $QK^\top$，显存占用显著下降。

### 场景 2：训练吞吐优化（GPU）

**背景**：注意力计算受带宽限制。  
**为什么适用**：块级访问更好地利用缓存与共享内存。

### 场景 3：推理延迟优化（GPU）

**背景**：推理需要低延迟与低显存。  
**为什么适用**：流式计算减少内存访问。

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：$O(T^2 d)$（计算量不变）
- **空间复杂度**：从 $O(T^2)$ 降到 $O(Td)$

### 为什么能 one-pass

因为在线 softmax 可以把“max 与 sum”的统计量流式更新，  
输出也能同步累积，避免多次读取同一行。

---

## 解释与原理（为什么这么做）

FlashAttention 不是减少计算量，而是减少显存 IO。  
通过 tiling，把“外存瓶颈”转化为“片上计算”。

---

## 常见问题与注意事项

1. **FlashAttention 是否改变结果？**  
   不改变，数学上等价于标准注意力。

2. **tile 大小如何选？**  
   受共享内存容量与硬件架构影响。

3. **是否适合所有模型？**  
   长序列收益最大，短序列收益有限。

---

## 最佳实践与建议

- 关注 tile 与 block 大小的硬件适配
- 与混合精度结合可进一步提升吞吐
- 先验证数值一致性再做性能优化

---

## S — Summary（总结）

### 核心收获

- FlashAttention 的核心是在线 softmax + tiling
- 通过块级扫描避免显式存储 $QK^\top$
- 计算量不变，但显存 IO 显著下降
- 这使得注意力在长序列上可扩展

### 小结 / 结论

FlashAttention 能 one-pass 的关键不在于 softmax 公式改变，  
而在于在线统计与块级流式计算。

### 参考与延伸阅读

- https://arxiv.org/abs/2205.14135
- https://github.com/Dao-AILab/flash-attention

---

## 元信息

- **阅读时长**：14~20 分钟
- **标签**：flash-attention、tiling、gpu
- **SEO 关键词**：FlashAttention, Tiling, One-pass
- **元描述**：FlashAttention 的 one-pass 与 tiling 原理。

---

## 行动号召（CTA）

尝试用本文的 block 版注意力做一次数值验证，  
感受“同结果、少内存”的工程价值。
