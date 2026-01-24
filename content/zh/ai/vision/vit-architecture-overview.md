---
title: "ViT 结构描述：从 Patch Embedding 到 Transformer 编码器"
date: 2026-01-24T16:25:35+08:00
draft: false
categories: ["AI", "Vision"]
tags: ["vit", "transformer", "vision", "patch-embedding", "pytorch"]
description: "系统讲清 ViT 的结构组件、工作流程与工程实践，并给出最小 PyTorch 示例。"
keywords: ["ViT", "Vision Transformer", "Patch Embedding", "Transformer", "图像分类"]
---

> **副标题 / 摘要**  
> ViT 把图像切成 patch 序列，再交给 Transformer 编码器处理。本文用 ACERS 框架拆解 ViT 的核心结构与工程选择，并提供最小可运行的 PyTorch 示例。

- **预计阅读时长**：16~20 分钟
- **标签**：`vit`、`transformer`、`vision`
- **SEO 关键词**：ViT, Vision Transformer, Patch Embedding, 图像分类
- **元描述**：系统描述 ViT 架构与工程实践，含最小 PyTorch 示例。

---

## 目标读者

- 想理解 ViT 架构的入门读者
- 需要做视觉模型选型的工程实践者
- 想从 CNN 迁移到 Transformer 的开发者

## 背景 / 动机

CNN 通过局部卷积捕获特征，但长程依赖与全局建模能力有限。  
ViT 把图像当成序列，直接用自注意力做全局建模，  
在大规模数据预训练下表现非常强。

## 核心概念

- **Patch Embedding**：将图像切成 patch 并线性投影。
- **Position Embedding**：补充位置信息。
- **[CLS] Token**：聚合全局特征用于分类。
- **Transformer Encoder**：多头自注意力 + FFN 堆叠。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

ViT 的核心流程：

1. 把图像切成固定大小 patch。
2. 每个 patch 拉平成向量并投影成 token。
3. 加上位置编码和 [CLS] token。
4. 送入 Transformer Encoder 得到全局表征。
5. 用 [CLS] token 输出做分类。

### 基础示例（1）

- 图像 224x224，patch 16x16 → 196 个 patch + 1 个 [CLS]。

### 基础示例（2）

- 只保留编码器，就能做图像分类与检索。

## 实践指南 / 步骤

1. 选择 patch 大小（8/16/32）。
2. 设置隐藏维度与层数（如 12 层，768 维）。
3. 添加位置编码与 [CLS] token。
4. 训练：优先用预训练权重再微调。

## 可运行示例（最小 ViT 前向）

```python
import torch
import torch.nn as nn


torch.manual_seed(42)

class MiniViT(nn.Module):
    def __init__(self, img_size=32, patch=8, dim=64, depth=2, heads=4):
        super().__init__()
        self.patch = patch
        self.unfold = nn.Unfold(kernel_size=patch, stride=patch)
        num_patches = (img_size // patch) ** 2
        self.proj = nn.Linear(3 * patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        patches = self.unfold(x).transpose(1, 2)  # B, N, patch_dim
        tokens = self.proj(patches)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        z = self.encoder(tokens)
        return self.head(z[:, 0])

x = torch.randn(2, 3, 32, 32)
model = MiniViT()
print(model(x).shape)
```

## 解释与原理

- patch embedding 把图像变成序列。
- self-attention 能在全局范围建模依赖。
- [CLS] token 作为全局聚合向量用于分类。

## C — Concepts（核心思想）

### 方法类型

ViT 属于**基于注意力的视觉表征模型**，用 Transformer Encoder 替代卷积堆叠。

### 关键公式

Patch embedding：

$ x \in \mathbb{R}^{H\times W\times C} \rightarrow x_p \in \mathbb{R}^{N\times (P^2C)} $

自注意力：

$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^\top}{\sqrt{d}})V $

### 解释与原理

- patch 大小决定 token 数量，从而决定注意力复杂度。
- 全局注意力使 ViT 对长程依赖更敏感。

## E — Engineering（工程应用）

### 场景 1：图像分类

- 背景：ImageNet 级别分类任务。
- 为什么适用：ViT 在大规模预训练下精度高。
- 代码示例（Python）：

```python
import torch

logits = torch.randn(1, 1000)
print(logits.argmax(dim=1).item())
```

### 场景 2：小数据迁移学习

- 背景：小样本任务直接训练易过拟合。
- 为什么适用：预训练 ViT 微调更稳定。
- 代码示例（Python）：

```python
import torch

features = torch.randn(1, 768)
head = torch.randn(768, 5)
print((features @ head).shape)
```

### 场景 3：多模态图文对齐

- 背景：CLIP 等模型需要视觉编码器。
- 为什么适用：ViT 输出可直接对齐文本特征。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

img = F.normalize(torch.randn(1, 512), dim=-1)
text = F.normalize(torch.randn(1, 512), dim=-1)
print((img @ text.T).item())
```

## R — Reflection（反思与深入）

- **时间复杂度**：注意力为 `O(N^2)`，`N` 为 patch 数。
- **空间复杂度**：注意力矩阵也为 `O(N^2)`。
- **替代方案**：
  - CNN：更高效但全局建模弱。
  - Swin Transformer：窗口注意力降低复杂度。
  - Hybrid 模型：卷积 + Transformer。
- **工程可行性**：ViT 对数据量依赖更强，预训练是关键。

## 常见问题与注意事项

- patch 太小会导致显存爆炸。
- 小数据集训练易过拟合。
- 位置编码选择（绝对/相对）会影响性能。

## 最佳实践与建议

- 先用预训练权重，再做任务微调。
- 调整 patch 大小平衡精度与成本。
- 结合强数据增强提升泛化。

## S — Summary（总结）

### 核心收获

- ViT 将图像转成 token 序列并用 Transformer 编码。
- Patch 大小决定复杂度与表现。
- 预训练 + 微调是 ViT 的主流工程路径。
- 与 CNN 相比，ViT 更擅长全局建模。

### 推荐延伸阅读

- An Image is Worth 16x16 Words
- DeiT：Data-efficient Image Transformers
- Swin Transformer

## 参考与延伸阅读

- https://arxiv.org/abs/2010.11929
- https://arxiv.org/abs/2012.12877
- https://arxiv.org/abs/2103.14030

## 小结 / 结论

ViT 用最简洁的方式把视觉任务带入 Transformer 世界。  
理解 patch embedding 与编码器结构，就能快速上手 ViT。

## 行动号召（CTA）

用本文的最小 ViT 结构替换你的视觉模型，观察精度与成本变化。
