---
title: "SGD vs Adam：优化器原理与工程取舍"
date: 2026-01-24T16:12:12+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["sgd", "adam", "optimizer", "training", "pytorch"]
description: "对比 SGD 与 Adam 的原理、收敛特性与应用场景，并提供最小 PyTorch 示例。"
keywords: ["SGD", "Adam", "优化器", "动量", "RMSProp"]
---

> **副标题 / 摘要**  
> SGD 简洁稳定，Adam 自适应学习率收敛更快。本文用 ACERS 框架对比两者原理与工程取舍，并给出最小 PyTorch 示例。

- **预计阅读时长**：14~18 分钟
- **标签**：`sgd`、`adam`、`optimizer`
- **SEO 关键词**：SGD, Adam, 优化器, 动量
- **元描述**：对比 SGD 与 Adam 的训练特性与使用场景。

---

## 目标读者

- 想理解优化器差异的入门读者
- 需要做训练稳定性与收敛速度取舍的工程实践者
- 想掌握常见调参策略的开发者

## 背景 / 动机

优化器决定训练速度与最终性能。  
SGD 以稳定著称，Adam 以快速收敛著称。  
理解两者差异有助于在不同任务中做更合理的选择。

## 核心概念

- **SGD**：基于当前梯度更新参数。
- **Momentum**：引入历史梯度方向，加速收敛。
- **Adam**：结合动量与 RMSProp，自适应学习率。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

- SGD：每步朝“当前梯度方向”走。
- Adam：用历史梯度估计方向，同时对每个参数自适应调节步长。

### 基础示例（1）

- SGD 在噪声大时会“抖动”，收敛慢但稳定。

### 基础示例（2）

- Adam 在稀疏梯度场景（NLP）通常收敛更快。

## 实践指南 / 步骤

1. 快速验证效果 → Adam。
2. 追求最终泛化 → SGD + 动量。
3. 对比验证集曲线，而非只看训练 loss。

## 可运行示例（最小 PyTorch 对比）

```python
import torch
import torch.nn as nn


torch.manual_seed(42)

x = torch.randn(128, 10)
y = torch.randn(128, 1)

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()

sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for _ in range(5):
    pred = model(x)
    loss = loss_fn(pred, y)
    sgd.zero_grad()
    loss.backward()
    sgd.step()

adam = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(5):
    pred = model(x)
    loss = loss_fn(pred, y)
    adam.zero_grad()
    loss.backward()
    adam.step()

print("done")
```

## 解释与原理

- SGD 只依赖当前梯度，步长固定。
- Adam 用一阶/二阶动量估计，使得学习率对每个参数自适应。

## C — Concepts（核心思想）

### 方法类型

SGD 是**一阶优化**基线，Adam 是**自适应学习率优化**。

### 关键公式

**SGD：**

$ \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) $

**Adam：**

$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $

$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $

$ \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $

### 解释与原理

- Adam 通过动量减少震荡，通过 RMS 缩放学习率。
- SGD 收敛慢但往往更利于泛化。

## E — Engineering（工程应用）

### 场景 1：大规模预训练

- 背景：训练成本高，需要稳定收敛。
- 为什么适用：Adam 更快收敛，省训练时间。
- 代码示例（Python）：

```python
import torch

opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-4)
print(opt.defaults["lr"])
```

### 场景 2：计算机视觉训练

- 背景：ResNet/ViT 训练常用 SGD。
- 为什么适用：SGD 泛化更稳定。
- 代码示例（Python）：

```python
import torch

opt = torch.optim.SGD([torch.randn(2, requires_grad=True)], lr=0.1, momentum=0.9)
print(opt.defaults["momentum"])
```

### 场景 3：稀疏梯度任务

- 背景：NLP/推荐系统中梯度稀疏。
- 为什么适用：Adam 的自适应学习率对稀疏梯度更友好。
- 代码示例（Python）：

```python
import torch

grad = torch.tensor([0.0, 0.0, 1.0])
print(grad.nonzero().numel())
```

## R — Reflection（反思与深入）

- **时间复杂度**：Adam 每步维护更多状态，略高于 SGD。
- **空间复杂度**：Adam 需保存一阶/二阶动量，内存翻倍。
- **替代方案**：
  - AdamW：更合理的权重衰减方式。
  - Lion：更低成本的自适应优化器。
- **工程可行性**：快速原型用 Adam，追求泛化再切回 SGD。

## 常见问题与注意事项

- Adam 学习率过大会导致不稳定。
- SGD 需要更长训练时间与更细致的学习率调度。
- AdamW 通常比 Adam 更稳定。

## 最佳实践与建议

- 用 Adam 快速验证，再用 SGD 精调。
- 记录学习率曲线，避免过早停止。
- 做小规模对比试验再确定长期方案。

## S — Summary（总结）

### 核心收获

- SGD 简洁稳定，Adam 收敛快。
- Adam 适合稀疏梯度与快速实验。
- SGD 往往有更好泛化表现。
- 实际工程常采用“Adam 验证 + SGD 精调”。

### 推荐延伸阅读

- Adam 论文：Adam: A Method for Stochastic Optimization
- SGD 与泛化性能相关研究
- AdamW 论文

## 参考与延伸阅读

- https://arxiv.org/abs/1412.6980
- https://arxiv.org/abs/1711.05101

## 小结 / 结论

SGD 与 Adam 的选择不是“谁更好”，而是“谁更合适”。  
理解梯度分布与训练目标，才能选到最适合的优化器。

## 行动号召（CTA）

用同一模型分别跑 SGD 与 Adam，比较收敛速度与验证集指标。
