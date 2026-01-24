---
title: "为什么注意力要除以 \u221a(d_k)：从数值稳定到工程收益"
date: 2026-01-24T16:22:25+08:00
draft: false
categories: ["AI", "Multimodal"]
tags: ["attention", "transformer", "scaled-dot-product", "numerical-stability", "pytorch"]
description: "解释注意力中 QK^T 为何需要除以 \u221a(d_k)，并给出最小 PyTorch 示例与工程场景。"
keywords: ["Attention", "Scaled Dot-Product", "\u221a(d_k)", "Transformer", "数值稳定"]
---

> **副标题 / 摘要**  
> 注意力中的缩放项 \u221a(d_k) 不是装饰，而是数值稳定的关键：它控制 QK^T 的方差，避免 softmax 饱和和梯度消失。本文用公式与实验解释其必要性，并给出工程场景建议。

- **预计阅读时长**：12~16 分钟
- **标签**：`attention`、`transformer`、`scaled-dot-product`
- **SEO 关键词**：Attention, Scaled Dot-Product, \u221a(d_k)
- **元描述**：解释注意力缩放项的数学动机与工程收益。

---

## 目标读者

- 想理解 Transformer 注意力细节的入门读者
- 需要排查训练不稳定问题的工程实践者
- 关注数值稳定性与性能优化的开发者

## 背景 / 动机

在点积注意力中，维度越大，QK^T 的数值越大，softmax 越容易饱和。  
一旦饱和，梯度接近 0，训练会变慢甚至不稳定。  
\u221a(d_k) 的缩放项就是为了解决这个问题。

## 核心概念

- **点积注意力**：$QK^\top$ 衡量相似度。
- **缩放项 \u221a(d_k)**：控制相似度的尺度。
- **softmax 饱和**：输入过大导致概率趋近 0/1，梯度变小。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

- 维度大时，QK^T 变大，softmax 过于“自信”。
- 缩放 \u221a(d_k) 后，数值回到合理范围，梯度更健康。

### 基础示例（1）

- d_k=64 时，如果不缩放，softmax 输出会非常尖锐。

### 基础示例（2）

- d_k=512 时，缩放与否会直接影响训练是否稳定。

## 实践指南 / 步骤

1. 使用标准缩放：$QK^\top / \sqrt{d_k}$。
2. 如果做自定义注意力，先验证 softmax 分布是否过尖锐。
3. 在混合精度训练下，缩放更重要。

## 可运行示例（缩放与不缩放的对比）

```python
import torch
import torch.nn.functional as F


def attn_scores(d, scale=True):
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 8, d)
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / (d ** 0.5)
    probs = F.softmax(scores, dim=-1)
    return probs.max().item(), probs.min().item()

for d in [32, 128, 512]:
    mx_s, mn_s = attn_scores(d, scale=True)
    mx_u, mn_u = attn_scores(d, scale=False)
    print(f"d={d} scaled max={mx_s:.3f} min={mn_s:.3f} | unscaled max={mx_u:.3f} min={mn_u:.3f}")
```

## 解释与原理

如果 $q_i, k_i \sim \mathcal{N}(0, 1)$，

$ q \cdot k = \sum_i q_i k_i $ 的方差约为 $d_k$。  
缩放 $1/\sqrt{d_k}$ 后，方差回到 $1$，softmax 输入稳定。

## C — Concepts（核心思想）

### 方法类型

缩放点积注意力属于**数值稳定性改进**范式。

### 关键公式

$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V $

### 解释与原理

- 不缩放：softmax 输入过大，梯度接近 0。
- 缩放后：梯度更稳定，训练更可靠。

## E — Engineering（工程应用）

### 场景 1：大模型训练稳定性

- 背景：d_k 很大时 softmax 饱和严重。
- 为什么适用：缩放能降低梯度消失风险。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 4, 512)
k = torch.randn(2, 4, 512)
logits = q @ k.transpose(-2, -1) / (512 ** 0.5)
probs = F.softmax(logits, dim=-1)
print(probs.mean().item())
```

### 场景 2：混合精度训练

- 背景：FP16 易溢出，softmax 更敏感。
- 为什么适用：缩放降低数值幅度，减少溢出。
- 代码示例（Python）：

```python
import torch

q = torch.randn(1, 2, 256, dtype=torch.float16)
k = torch.randn(1, 2, 256, dtype=torch.float16)
logits = q @ k.transpose(-2, -1) / (256 ** 0.5)
print(logits.dtype)
```

### 场景 3：跨模态 cross-attention

- 背景：图文特征维度大且分布不同。
- 为什么适用：缩放让对齐更稳定。
- 代码示例（Python）：

```python
import torch

text = torch.randn(2, 10, 768)
image = torch.randn(2, 49, 768)
logits = text @ image.transpose(-2, -1) / (768 ** 0.5)
print(logits.shape)
```

## R — Reflection（反思与深入）

- **时间复杂度**：缩放是常数开销，复杂度不变。
- **空间复杂度**：不增加额外存储。
- **替代方案**：
  - 使用温度参数调节 softmax。
  - 使用归一化后的 Q/K（如 cosine attention）。
- **工程可行性**：缩放几乎无代价，但收益显著，是默认选择。

## 常见问题与注意事项

- 仅缩放 V 不会解决 softmax 饱和。
- 温度参数过低会导致过尖锐分布。
- 多头注意力里使用每个 head 的 $d_k$ 做缩放。

## 最佳实践与建议

- 默认使用 $1/\sqrt{d_k}$。
- 训练不稳定时先检查是否遗漏缩放。
- 如果自定义注意力，记录注意力权重分布。

## S — Summary（总结）

### 核心收获

- \u221a(d_k) 缩放是为控制点积方差。
- 缩放避免 softmax 饱和与梯度消失。
- 对大模型与混合精度训练尤为重要。
- 缩放几乎无成本，是默认最佳实践。

### 推荐延伸阅读

- Attention Is All You Need
- The Annotated Transformer
- 数值稳定性相关实践文档

## 参考与延伸阅读

- https://arxiv.org/abs/1706.03762
- https://nlp.seas.harvard.edu/annotated-transformer/

## 小结 / 结论

注意力的缩放项是“最小改动、最大收益”的典型工程技巧。  
理解它的统计意义，就能更稳地训练和扩展模型。

## 行动号召（CTA）

用本文示例替换你的维度配置，观察缩放前后的注意力分布差异。
