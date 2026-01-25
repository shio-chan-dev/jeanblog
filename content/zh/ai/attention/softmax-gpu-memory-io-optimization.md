---
title: "Softmax 工程实现与 GPU 访存优化：从两次遍历到在线计算"
date: 2026-01-25T12:51:13+08:00
draft: false
categories: ["AI", "Multimodal"]
tags: ["softmax", "gpu", "memory", "performance", "attention"]
description: "拆解 softmax 标准计算的访存问题，并给出在线 softmax 与融合实现的工程优化思路。"
keywords: ["Softmax", "GPU", "访存优化", "Online Softmax", "Attention"]
---

> **副标题 / 摘要**  
> softmax 的数学公式很简单，但在 GPU 上实现时会带来高昂的访存成本。本文从“标准两遍计算”出发，推导在线 softmax 的工程做法，并解释如何降低访存复杂度。

- **预计阅读时长**：12~18 分钟
- **标签**：`softmax`、`gpu`、`memory`
- **SEO 关键词**：Softmax, GPU, 访存优化, Online Softmax
- **元描述**：softmax 的工程实现与 GPU 访存优化策略，含在线算法与示例代码。

---

## 目标读者

- 想理解 softmax 在 GPU 上“慢在哪里”的工程读者
- 关注训练吞吐与内存带宽瓶颈的优化者
- 需要实现注意力融合算子的开发者

## 背景 / 动机

softmax 在注意力中出现频率极高。  
在 GPU 上，性能通常受限于内存带宽而不是算力。  
理解 softmax 的访存模式，是做 FlashAttention 等优化的基础。

## 核心概念

- **标准 softmax**：先求行最大值，再求指数和，再归一化
- **两遍遍历**：至少读两次输入（max 与 sum）
- **在线 softmax**：一遍扫描更新 max 与 sum

---

## 思路推导（从两遍遍历到在线算法）

### 朴素做法

对每一行 $x$：  
1) 读一遍找 $m = \max(x)$  
2) 再读一遍算 $\sum \exp(x - m)$  
3) 第三遍输出结果

这带来 2~3 次访存，尤其在长序列上成为瓶颈。

### 关键观察

可以在扫描时维护“当前最大值 $m$”和“归一化分母 $l$”，  
当遇到更大的元素时，通过重标定修正 $l$。

### 在线 softmax（核心公式）

初始化 $m=-\infty, l=0$，遍历元素 $x_i$：

- $m' = \max(m, x_i)$
- $l' = l \cdot \exp(m - m') + \exp(x_i - m')$

最终 $\text{softmax}(x_i) = \exp(x_i - m) / l$。

---

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

softmax 不只是数学函数，它是一段“访问内存、统计最大值、再归一化”的流程。  
优化 softmax 本质上就是优化这个流程的访存次数。

### 基础示例

- 传统 softmax：两次遍历（max + sum）
- 在线 softmax：一次遍历（max 与 sum 同时更新）

---

## C — Concepts（核心思想）

### 方法归类

- 数值稳定
- 在线统计（streaming）
- 融合内核（kernel fusion）

### 关键公式

- **稳定 softmax**：$\exp(x - \max(x)) / \sum \exp(x - \max(x))$
- **在线更新**：$m' = \max(m, x_i)$，$l' = l\exp(m-m') + \exp(x_i-m')$

---

## 实践指南 / 步骤

1. 选择在线 softmax 算法替代两遍遍历
2. 在 GPU 上将“max + sum + normalize”合并为一个 kernel
3. 尽量避免把中间结果写回显存

## 可运行示例（在线 softmax）

```python
import numpy as np


def online_softmax(x):
    m = -np.inf
    l = 0.0
    for xi in x:
        m_new = max(m, xi)
        l = l * np.exp(m - m_new) + np.exp(xi - m_new)
        m = m_new
    return np.exp(x - m) / l


def stable_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


if __name__ == "__main__":
    x = np.array([3.0, 1.0, -2.0, 5.0])
    print(online_softmax(x))
    print(stable_softmax(x))
```

---

## E — Engineering（工程应用）

### 场景 1：注意力融合算子（GPU）

**背景**：注意力中 softmax 与 $QK^\top$、$PV$ 相邻。  
**为什么适用**：将 softmax 与前后矩阵乘法融合，减少中间写回。  

实践：通过 kernel fusion 直接在寄存器/共享内存中完成归一化。

### 场景 2：大 batch 推理（GPU）

**背景**：推理中 softmax 常成为带宽瓶颈。  
**为什么适用**：在线 softmax 减少一次读内存。  

### 场景 3：混合精度训练（GPU）

**背景**：fp16/bf16 更敏感。  
**为什么适用**：在线更新可保持数值稳定性并减少内存开销。

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：$O(n)$
- **访存复杂度**：
  - 标准 softmax：2~3 次读写
  - 在线 softmax：1 次读 + 1 次写

### 替代方案

| 方案 | 访存 | 风险 |
| --- | --- | --- |
| 两遍 softmax | 高 | 带宽瓶颈 |
| 在线 softmax | 低 | 需重标定 |
| 近似 softmax | 低 | 可能损失精度 |

### 为什么在线方法更工程可行

它在不改变数学结果的前提下减少访存次数，  
在 GPU 上能显著提升吞吐。

---

## 解释与原理（为什么这么做）

GPU 性能大多受内存带宽限制。  
降低 softmax 的访存次数，就是直接提升吞吐和能效。

---

## 常见问题与注意事项

1. **在线 softmax 会带来误差吗？**  
   只要使用稳定更新公式，结果与标准 softmax 等价。

2. **能否做到完全 one-pass 输出？**  
   理论上仍需一次最终输出，但中间不需要写回显存。

3. **是否必须用 CUDA？**  
   在线 softmax 思想也适用于 CPU 与向量化实现。

---

## 最佳实践与建议

- 优先用在线 softmax 替代两遍遍历
- 将 softmax 与前后算子融合，减少中间写回
- 在长序列模型中重点优化 softmax 的带宽开销

---

## S — Summary（总结）

### 核心收获

- softmax 的瓶颈通常在访存而非计算
- 在线 softmax 可以把两遍遍历降到一遍
- kernel fusion 是工程落地的关键路径
- 这些优化是 FlashAttention 的基础之一

### 小结 / 结论

softmax 优化的核心不是“更快的 exp”，而是“更少的内存访问”。  
在线 softmax 是最直接有效的工程策略。

### 参考与延伸阅读

- https://arxiv.org/abs/2205.14135
- https://developer.nvidia.com/blog/optimizing-softmax

---

## 元信息

- **阅读时长**：12~18 分钟
- **标签**：softmax、gpu、memory
- **SEO 关键词**：Softmax, GPU, 访存优化
- **元描述**：softmax 的 GPU 访存优化与在线算法。

---

## 行动号召（CTA）

尝试用在线 softmax 实现你自己的注意力模块，  
对比传统实现的速度与显存占用。
