---
title: "对比学习损失函数系列（1/4）：对比损失 Contrastive Loss"
date: 2026-01-24T13:22:02+08:00
draft: false
categories: ["AI", "Representation Learning"]
tags: ["contrastive-loss", "metric-learning", "pairwise", "embedding", "margin"]
description: "从公式到实验，系统理解对比损失（Contrastive Loss）如何拉近正样本、推远负样本。"
keywords: ["对比损失", "Contrastive Loss", "度量学习", "嵌入空间", "margin"]
---

> **副标题 / 摘要**  
> 对比损失是度量学习最经典的成对目标：拉近同类、推远异类。本文用公式、几何直觉与最小可运行实验，帮你建立对比学习的第一块基石。

- **预计阅读时长**：15~18 分钟
- **标签**：`contrastive-loss`、`metric-learning`、`pairwise`
- **SEO 关键词**：对比损失, Contrastive Loss, 度量学习, 嵌入空间
- **元描述**：讲清对比损失的数学形式、训练细节与工程应用场景。

---

## 系列导航

- （1/4）对比损失 Contrastive Loss（本文）
- （2/4）三元组损失 Triplet Loss
- （3/4）InfoNCE + SimCLR
- （4/4）CLIP 对比学习目标

## 目标读者

- 想入门对比学习/度量学习的初学者
- 需要在工程中构建相似度模型的开发者
- 希望通过小实验理解公式含义的实践派

## 背景 / 动机

在推荐、检索、验证类任务里，我们往往不关心“分类标签”，而关心“相似度”。  
对比损失用成对样本表达“相似/不相似”，是把语义关系映射到向量空间的基础方法。

## 核心概念

- **嵌入空间**：把样本映射为向量，距离代表语义相近程度。
- **正负样本对**：正样本对应“相似”，负样本对对应“不相似”。
- **Margin**：负样本需要被推远的最小距离阈值。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

对比损失做的事很简单：

- 同类样本对要靠得更近。
- 异类样本对要至少分开一个 margin。

### 基础示例（1）

- 两张同一人的人脸：距离应该变小。
- 两个不同人的人脸：距离至少大于 `margin`。

### 基础示例（2）

- 同类商品图片：嵌入距离小。
- 异类商品图片：嵌入距离大。

## 实践指南 / 步骤

1. 选择特征编码器（如 MLP/CNN）。
2. 构造正负样本对，并标记 `y=1/0`。
3. 计算成对距离并应用对比损失。
4. 观察正负样本平均距离是否分离。

## 可运行示例（最小对比损失实验）

```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)


def make_data(n=200):
    c1 = torch.randn(n, 2) * 0.4 + torch.tensor([0.0, 0.0])
    c2 = torch.randn(n, 2) * 0.4 + torch.tensor([3.0, 3.0])
    x = torch.cat([c1, c2], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x, y


def make_pairs(x, y, num_pairs=1000):
    pairs = []
    labels = []
    for _ in range(num_pairs):
        if random.random() < 0.5:
            cls = random.randint(0, 1)
            idx = (y == cls).nonzero().flatten()
            i, j = idx[torch.randint(len(idx), (2,))]
            labels.append(1)
        else:
            i = (y == 0).nonzero().flatten()[torch.randint((y == 0).sum(), (1,))]
            j = (y == 1).nonzero().flatten()[torch.randint((y == 1).sum(), (1,))]
            labels.append(0)
        pairs.append((x[i], x[j]))
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs]), torch.tensor(labels)


def contrastive_loss(z1, z2, y, margin=1.0):
    d = F.pairwise_distance(z1, z2)
    pos = y * d.pow(2)
    neg = (1 - y) * F.relu(margin - d).pow(2)
    return (pos + neg).mean()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


x, y = make_data()
x1, x2, pair_y = make_pairs(x, y, num_pairs=2000)

model = Encoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(1, 201):
    z1 = model(x1)
    z2 = model(x2)
    loss = contrastive_loss(z1, z2, pair_y.float(), margin=1.0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            d = F.pairwise_distance(z1, z2)
            pos_d = d[pair_y == 1].mean().item()
            neg_d = d[pair_y == 0].mean().item()
        print(f"epoch={epoch} loss={loss.item():.4f} pos_d={pos_d:.3f} neg_d={neg_d:.3f}")
```

## C — Concepts（核心思想）

### 方法类型

对比损失属于**度量学习 / 表示学习**范式，使用成对样本将语义关系映射到向量距离。

### 关键公式

设成对样本 `(x_i, x_j)` 的标签 `y`：相似为 1，不相似为 0。  
嵌入 `z_i = f(x_i)`，距离 `d = \|z_i - z_j\|_2`，对比损失为：

$ L = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2 $

其中 `m` 为 margin，控制负样本要被推开的最小距离。

### 解释与原理

- 正样本：最小化距离，聚合同类。
- 负样本：距离低于 margin 才会产生惩罚，避免过度拉远。
- margin 过大：可能导致训练不稳定；过小：区分度不足。

## E — Engineering（工程应用）

### 场景 1：人脸验证

- 背景：判断两张脸是否为同一人。
- 为什么适用：成对标签天然可得（同人/不同人）。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

emb1 = torch.randn(1, 128)
emb2 = torch.randn(1, 128)
score = F.pairwise_distance(emb1, emb2).item()
match = score < 0.8
print(score, match)
```

### 场景 2：商品相似检索

- 背景：从商品库里找“相似商品”。
- 为什么适用：嵌入距离可以直接做检索排序。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

query = F.normalize(torch.randn(1, 64), dim=-1)
corpus = F.normalize(torch.randn(100, 64), dim=-1)
score = (query @ corpus.T).squeeze(0)
idx = score.topk(k=5).indices
print(idx)
```

### 场景 3：重复内容检测

- 背景：检测文本或图片是否重复或高度相似。
- 为什么适用：同一语义内容在嵌入空间距离更小。
- 代码示例（Python）：

```python
import torch
import torch.nn.functional as F

items = F.normalize(torch.randn(10, 64), dim=-1)
score = items @ items.T
near = (score > 0.9).nonzero(as_tuple=False)
print(near[:5])
```

## R — Reflection（反思与深入）

- **时间复杂度**：成对训练需要 `O(N^2)` 的潜在组合，通常需采样。
- **空间复杂度**：取决于 batch 与成对数量，一般为 `O(N)` 到 `O(N^2)`。
- **替代方案**：
  - 分类损失：无需成对标签，但相似度语义不直观。
  - Triplet Loss：提供更强的相对排序约束。
  - InfoNCE：批内负样本更丰富，训练更稳定。
- **工程可行性**：当成对标签可得时，对比损失简单有效，易于落地。

## 常见问题与注意事项

- 负样本过少会导致“全部都靠近”的塌缩。
- margin 选择依赖特征尺度，需与归一化策略协同。
- 样本对分布不均会导致模型偏向多数类。

## 最佳实践与建议

- 使用 L2 归一化或温度缩放稳定距离尺度。
- 对负样本进行难例挖掘（hard negative）。
- 用可视化或统计验证正负距离分离。

## S — Summary（总结）

### 核心收获

- 对比损失通过成对距离表达“相似与不相似”。
- margin 控制负样本的最小分离度，是关键超参数。
- 成对标签可得时，对比损失是最直接的度量学习方案。
- 正负距离统计能快速判断训练是否有效。

### 推荐延伸阅读

- Hadsell et al. (2006), Dimensionality Reduction by Learning an Invariant Mapping
- Metric Learning 综述文章
- PyTorch Metric Learning 库

## 参考与延伸阅读

- https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
- https://kevinmusgrave.github.io/pytorch-metric-learning/

## 小结 / 结论

对比损失的价值在于“把相似关系变成几何关系”。  
理解了成对距离与 margin，你就掌握了对比学习的基本语法。

## 行动号召（CTA）

试着把这段最小实验替换为你自己的数据，观察正负距离随训练如何变化。
