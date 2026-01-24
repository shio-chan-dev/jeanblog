---
title: "优化器的了解：从 SGD 到 Adam 的工程取舍"
date: 2026-01-24T16:27:20+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["optimizer", "sgd", "adam", "adamw", "training"]
description: "系统讲清常见优化器原理与工程取舍，含最小 PyTorch 示例与实践建议。"
keywords: ["优化器", "SGD", "Adam", "AdamW", "训练"]
---

> **副标题 / 摘要**  
> 优化器决定训练速度、稳定性与最终泛化。本文按 ACERS 框架对比 SGD、Momentum、Adam、AdamW 等主流优化器，并给出最小可运行示例与工程实践建议。

- **预计阅读时长**：15~18 分钟
- **标签**：`optimizer`、`sgd`、`adam`、`adamw`
- **SEO 关键词**：优化器, SGD, Adam, AdamW
- **元描述**：对比主流优化器原理与工程场景，给出可运行示例。

---

## 目标读者

- 刚入门深度学习训练的读者
- 需要在速度与泛化之间权衡的工程实践者
- 想系统理解优化器选择的开发者

## 背景 / 动机

在训练大模型时，损失函数不是唯一关键，优化器同样决定成败。  
同一模型下，不同优化器会带来完全不同的收敛曲线与最终效果。  
理解优化器差异，是做出稳定工程方案的前提。

## 核心概念

- **梯度下降**：沿损失函数梯度方向更新参数。
- **动量（Momentum）**：引入历史梯度方向，减少震荡。
- **自适应学习率**：为不同参数分配不同步长。
- **权重衰减（Weight Decay）**：控制参数规模，提升泛化。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

- SGD：每次更新都沿着当前梯度方向。
- Momentum：带“惯性”的 SGD，加速收敛。
- Adam：对每个参数自适应调整学习率。
- AdamW：把权重衰减从 Adam 的梯度中解耦。

### 基础示例（1）

- SGD 在陡峭峡谷会来回震荡。
- Adam 会自动缩小震荡方向的步长。

### 基础示例（2）

- Adam 收敛快但可能泛化弱。
- SGD 收敛慢但往往更稳。

## 实践指南 / 步骤

1. 快速验证模型可行性 → Adam/AdamW。
2. 追求最终泛化性能 → SGD + 动量。
3. 训练大模型时优先 AdamW。
4. 用验证集曲线而非训练 loss 评估。

## 可运行示例（最小 PyTorch 对比）

```python
import torch
import torch.nn as nn


torch.manual_seed(42)

x = torch.randn(256, 10)
y = torch.randn(256, 1)

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()

# SGD
sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for _ in range(5):
    pred = model(x)
    loss = loss_fn(pred, y)
    sgd.zero_grad()
    loss.backward()
    sgd.step()

# AdamW
adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
for _ in range(5):
    pred = model(x)
    loss = loss_fn(pred, y)
    adamw.zero_grad()
    loss.backward()
    adamw.step()

print("done")
```

## 解释与原理

- Adam 引入一阶与二阶动量，提升收敛速度。
- AdamW 通过“解耦权重衰减”更稳定。
- SGD 的优势在于更好的泛化表现。

## C — Concepts（核心思想）

### 方法类型

优化器属于**数值优化方法**，核心目标是稳定、快速、可泛化地找到最优解。

### 关键公式

**SGD：**

$ \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) $

**Momentum：**

$ v_t = \beta v_{t-1} + (1-\beta) \nabla L(\theta_t) $

$ \theta_{t+1} = \theta_t - \eta v_t $

**Adam：**

$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $

$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $

$ \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} $

### 解释与原理

- Momentum 减少震荡、加速收敛。
- Adam 为每个参数自适应步长，适合稀疏梯度。
- AdamW 避免权重衰减影响动量估计。

## E — Engineering（工程应用）

### 场景 1：大模型预训练

- 背景：训练成本高，追求快速收敛。
- 为什么适用：AdamW 兼顾速度与稳定性。
- 代码示例（Python）：

```python
import torch

opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=1e-4, weight_decay=0.01)
print(opt.defaults["lr"])
```

### 场景 2：图像分类训练

- 背景：ResNet/ViT 训练常用 SGD。
- 为什么适用：SGD 泛化稳定，配合学习率调度效果好。
- 代码示例（Python）：

```python
import torch

opt = torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.1, momentum=0.9)
print(opt.defaults["momentum"])
```

### 场景 3：稀疏梯度任务

- 背景：NLP/推荐场景梯度稀疏。
- 为什么适用：Adam 自适应学习率更友好。
- 代码示例（Python）：

```python
import torch

g = torch.tensor([0.0, 0.0, 1.0])
print(g.nonzero().numel())
```

## R — Reflection（反思与深入）

- **时间复杂度**：SGD 为 `O(n)`，Adam/AdamW 需额外动量状态。
- **空间复杂度**：Adam/AdamW 需要存两份动量缓存，内存约 2 倍。
- **替代方案**：
  - Adafactor：更省内存的自适应优化器。
  - Lion：更低成本的动量优化。
- **工程可行性**：小模型可用 SGD，规模化训练多用 AdamW。

## 常见问题与注意事项

- Adam 收敛快但可能泛化弱。
- SGD 需要更细致的学习率调度。
- 权重衰减要与优化器匹配（建议 AdamW）。

## 最佳实践与建议

- 先用 AdamW 快速验证，再用 SGD 精调。
- 对比训练与验证曲线，不只看 loss。
- 记录学习率与优化器配置以便复现。

## S — Summary（总结）

### 核心收获

- SGD 简洁稳定，Adam/AdamW 收敛更快。
- AdamW 是大模型训练的工程默认。
- 优化器选择应结合任务、数据与资源。
- 不同优化器需配合不同学习率策略。

### 推荐延伸阅读

- Adam: A Method for Stochastic Optimization
- Decoupled Weight Decay Regularization (AdamW)
- 优化器综述文章

## 参考与延伸阅读

- https://arxiv.org/abs/1412.6980
- https://arxiv.org/abs/1711.05101

## 小结 / 结论

理解优化器的差异，才能在稳定性与效率之间做出更好的取舍。  
工程上先快后稳，是最可靠的实践路线。

## 行动号召（CTA）

用同一模型比较 SGD 与 AdamW 的曲线，找到最适合你的优化器组合。
