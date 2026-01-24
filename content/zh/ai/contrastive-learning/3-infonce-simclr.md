---
title: "对比学习损失函数系列（3/4）：InfoNCE 与 SimCLR"
date: 2026-01-24T13:22:02+08:00
draft: false
categories: ["AI", "Representation Learning"]
tags: ["infonce", "simclr", "self-supervised", "contrastive-learning", "augmentation"]
description: "从 InfoNCE 公式到 SimCLR 训练流程，理解自监督对比学习的关键设计。"
keywords: ["InfoNCE", "SimCLR", "对比学习", "自监督", "投影头"]
---

> **副标题 / 摘要**  
> InfoNCE 是现代对比学习的核心损失，SimCLR 则把它推向实用化。本文用公式、步骤与最小实验，带你理解“批内负样本 + 增强视图”的训练逻辑。

- **预计阅读时长**：18~22 分钟
- **标签**：`infonce`、`simclr`、`self-supervised`
- **SEO 关键词**：InfoNCE, SimCLR, 对比学习, 自监督
- **元描述**：讲清 InfoNCE 的数学目标与 SimCLR 的训练结构，含可运行代码示例。

---

## 系列导航

- （1/4）对比损失 Contrastive Loss
- （2/4）三元组损失 Triplet Loss
- （3/4）InfoNCE + SimCLR（本文）
- （4/4）CLIP 对比学习目标

## 目标读者

- 希望入门自监督对比学习的读者
- 需要理解 SimCLR 训练流程的工程实践者
- 想把对比学习迁移到业务数据的开发者

## 背景 / 动机

有标注数据昂贵，而无标注数据充足。  
InfoNCE 让我们用“正负样本对齐”替代人工标签，  
SimCLR 则证明：只要数据增强和 batch 够大，效果可以接近监督学习。

## 核心概念

- **正样本视图**：同一图像的两种增强视图。
- **批内负样本**：同一 batch 中其他样本视为负样本。
- **投影头**：把表示映射到对比空间，提高对比学习效果。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

InfoNCE 的核心是“在一堆负样本里找到正确配对”。  
SimCLR 则把“正确配对”定义为同一张图像的两个增强视图。

### 基础示例（1）

- 图像 A 经过两种增强得到 A1 与 A2
- 目标：A1 与 A2 相似度最大化

### 基础示例（2）

- A1 在 batch 中看到 B1、C1 等视为负样本
- 目标：A1 与 A2 的相似度高于 A1 与其他样本

## 实践指南 / 步骤

1. 设计增强策略（裁剪、翻转、颜色扰动）。
2. 构造两份增强视图作为正样本对。
3. 编码器 + 投影头输出对比向量。
4. 使用 InfoNCE 计算对比损失并训练。

## 可运行示例（最小 SimCLR 实验）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42)


class TwoCrops:
    def __init__(self, base_transform):
        self.base = base_transform

    def __call__(self, x):
        return self.base(x), self.base(x)


def info_nce(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    return (loss1 + loss2) / 2


class Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)


base_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

dataset = datasets.FakeData(
    size=512,
    image_size=(3, 32, 32),
    num_classes=10,
    transform=TwoCrops(base_tf),
)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = Encoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 6):
    total = 0.0
    for (x1, x2), _ in loader:
        z1 = model(x1)
        z2 = model(x2)
        loss = info_nce(z1, z2, temp=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch={epoch} loss={total/len(loader):.4f}")
```

## C — Concepts（核心思想）

### 方法类型

InfoNCE 与 SimCLR 属于**自监督对比学习**，通过增强视图构造正样本对。

### 关键公式（InfoNCE）

设正样本对 `(i, j)`，相似度 `s_{ij}`，则：

$ L_i = -\log \frac{\exp(s_{ij}/\tau)}{\sum_{k=1}^{N} \exp(s_{ik}/\tau)} $

通过 batch 内其他样本形成负样本集合。

### 解释与原理

- 更多负样本 → 更强的判别信号。
- 投影头只用于对比学习，最终特征取 backbone 输出。
- 温度参数控制相似度分布的尖锐度。

## E — Engineering（工程应用）

### 场景 1：医学影像预训练

- 背景：标注成本高，数据却很多。
- 为什么适用：自监督可先学到通用表征。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

images = torch.randn(16, 3, 224, 224)
features = F.normalize(images.mean(dim=[2, 3]), dim=-1)
print(features.shape)
```

### 场景 2：冷启动检索

- 背景：新领域缺少标签。
- 为什么适用：SimCLR 表征可用于检索初始化。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

vecs = F.normalize(torch.randn(100, 64), dim=-1)
query = F.normalize(torch.randn(1, 64), dim=-1)
idx = (query @ vecs.T).topk(k=3).indices
print(idx)
```

### 场景 3：小样本分类迁移

- 背景：下游标注少，直接训练易过拟合。
- 为什么适用：先自监督预训练，再微调少量标签。
- 代码示例（Python）：

```python
import torch

backbone = torch.nn.Linear(128, 64)
head = torch.nn.Linear(64, 10)
params = list(backbone.parameters()) + list(head.parameters())
print(len(params))
```

## R — Reflection（反思与深入）

- **时间复杂度**：每个 batch 需计算 `N x N` 相似度矩阵。
- **空间复杂度**：相似度矩阵 `O(N^2)`，大 batch 会显著占用显存。
- **替代方案**：
  - MoCo：用队列扩展负样本而不增大 batch。
  - BYOL/SimSiam：去除显式负样本。
- **工程可行性**：SimCLR 实现简单，但对 batch 大小敏感。

## 常见问题与注意事项

- 增强策略过弱会导致学习不到不变性。
- 温度参数不合适会造成训练不稳定。
- 只看 loss 可能误判效果，需要下游验证。

## 最佳实践与建议

- 优先验证“增强是否合理”。
- 如果显存不足，考虑梯度累积或 MoCo 类方案。
- 训练后用少量标签验证表征质量。

## S — Summary（总结）

### 核心收获

- InfoNCE 是现代对比学习的核心公式。
- SimCLR 用增强视图构造正样本对，简洁且有效。
- batch 内负样本数量决定了训练信号强度。
- 投影头可以提升对比学习效果而不影响下游特征。

### 推荐延伸阅读

- SimCLR 论文：A Simple Framework for Contrastive Learning of Visual Representations
- MoCo 论文：Momentum Contrast for Unsupervised Visual Representation Learning
- BYOL/SimSiam 相关工作

## 参考与延伸阅读

- https://arxiv.org/abs/2002.05709
- https://arxiv.org/abs/1911.05722
- https://arxiv.org/abs/2006.07733

## 小结 / 结论

InfoNCE 是对比学习的“通用目标函数”，SimCLR 则提供了最直接的训练模板。  
理解两者，就能把自监督对比学习迁移到自己的数据上。

## 行动号召（CTA）

把示例中的 FakeData 换成你的真实数据，观察增强策略对效果的影响。
