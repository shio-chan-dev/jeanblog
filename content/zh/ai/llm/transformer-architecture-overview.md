---
title: "Transformer 结构描述：从编码器到解码器"
date: 2026-01-24T16:18:19+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["transformer", "attention", "encoder-decoder", "self-attention", "architecture"]
description: "用 ACERS 框架讲清 Transformer 结构、模块职责与工程场景，并给出最小可运行示例。"
keywords: ["Transformer", "编码器", "解码器", "注意力机制", "多头注意力"]
---

> **副标题 / 摘要**  
> Transformer 由编码器与解码器堆叠而成，核心是自注意力与前馈网络。本文从结构出发解释各模块职责，并提供最小可运行示例与工程场景。

- **预计阅读时长**：16~20 分钟
- **标签**：`transformer`、`attention`、`encoder-decoder`
- **SEO 关键词**：Transformer, 编码器, 解码器, 注意力机制
- **元描述**：系统描述 Transformer 结构与工程应用，含最小示例。

---

## 目标读者

- 想理解 Transformer 结构的入门读者
- 需要搭建 NLP/多模态模型的工程实践者
- 关注模型架构取舍的开发者

## 背景 / 动机

在 Transformer 出现之前，序列建模主要依赖 RNN。  
Transformer 用注意力替代循环，大幅提升并行性与可扩展性。  
理解其结构，是学习大模型的起点。

## 核心概念

- **Encoder/Decoder**：编码器负责理解输入，解码器负责生成输出。
- **Self-Attention**：同一序列内部建模依赖。
- **Cross-Attention**：解码器对编码器输出做对齐。
- **FFN**：逐位置前馈网络。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

Transformer 的流程可以理解为：

1. 编码器把输入序列变成上下文表示。
2. 解码器在生成时，通过 cross-attention 读取编码器信息。
3. 多层堆叠形成深层表达。

### 基础示例（1）

- 机器翻译：编码器读英文，解码器生成中文。

### 基础示例（2）

- 文本生成：只保留解码器，逐词预测下一个 token。

## 实践指南 / 步骤

1. 选择结构：encoder-decoder（翻译）或 decoder-only（生成）。
2. 设置模型参数：层数、隐藏维度、注意力头数。
3. 训练：使用适当的损失（MLM/CLM）。
4. 推理：启用因果 mask 或 cross-attention。

## 可运行示例（最小 Transformer 模块）

```python
import torch
import torch.nn as nn


torch.manual_seed(42)

model = nn.Transformer(
    d_model=32,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=64,
    batch_first=True,
)

src = torch.randn(2, 5, 32)
tgt = torch.randn(2, 4, 32)

out = model(src, tgt)
print(out.shape)
```

## 解释与原理

- 编码器输出为“上下文记忆”。
- 解码器 self-attn 保证自回归顺序。
- cross-attn 让解码器读取编码器信息。

## C — Concepts（核心思想）

### 方法类型

Transformer 属于**注意力驱动的序列建模架构**。

### 关键公式

注意力：

$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^\top}{\sqrt{d}})V $

编码器层 = Self-Attention + FFN + 残差归一化。  
解码器层 = Masked Self-Attention + Cross-Attention + FFN。

### 解释与原理

- 多头注意力提升表示能力。
- 残差与归一化保证深层训练稳定。
- FFN 提供非线性变换。

## E — Engineering（工程应用）

### 场景 1：机器翻译（Encoder-Decoder）

- 背景：源语言到目标语言的映射。
- 为什么适用：cross-attention 直接建模对齐关系。
- 代码示例（Python）：

```python
import torch
import torch.nn as nn

model = nn.Transformer(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, batch_first=True)
src = torch.randn(1, 6, 16)
tgt = torch.randn(1, 5, 16)
print(model(src, tgt).shape)
```

### 场景 2：文本生成（Decoder-Only）

- 背景：对话、续写、代码生成。
- 为什么适用：自回归结构与生成目标一致。
- 代码示例（Python）：

```python
import torch

prompt = torch.tensor([[1, 2, 3]])
next_token = torch.randint(0, 100, (1, 1))
print(next_token.item())
```

### 场景 3：多模态对齐

- 背景：图像与文本对齐。
- 为什么适用：cross-attn 可直接关联图文特征。
- 代码示例（Python）：

```python
import torch

text = torch.randn(2, 10, 32)
image = torch.randn(2, 49, 32)
attn = text @ image.transpose(-2, -1)
print(attn.shape)
```

## R — Reflection（反思与深入）

- **时间复杂度**：注意力为 `O(n^2)`。
- **空间复杂度**：注意力矩阵占用大。
- **替代方案**：
  - 线性注意力、稀疏注意力降低复杂度。
  - Longformer/Performer 等结构。
- **工程可行性**：Transformer 在大规模任务中表现稳定，但需优化长序列成本。

## 常见问题与注意事项

- 长序列会导致显存爆炸。
- mask 设置错误会引发信息泄露。
- 训练需良好的初始化与归一化策略。

## 最佳实践与建议

- 先用小模型验证结构，再扩展规模。
- 注意力与 FFN 的维度配比要合理。
- 推理时启用缓存（KV cache）提升速度。

## S — Summary（总结）

### 核心收获

- Transformer 由编码器与解码器组成。
- Self-attention 建模序列内部依赖，cross-attention 做对齐。
- Decoder-only 是生成任务的简化形态。
- 工程落地需关注长序列成本。

### 推荐延伸阅读

- Attention Is All You Need
- The Annotated Transformer
- Transformer 结构优化研究

## 参考与延伸阅读

- https://arxiv.org/abs/1706.03762
- https://nlp.seas.harvard.edu/annotated-transformer/

## 小结 / 结论

Transformer 的核心是注意力与并行化。  
理解结构与模块职责，才能正确搭建和扩展模型。

## 行动号召（CTA）

从小规模模型开始搭建 Transformer，逐步增加层数与维度观察性能变化。
