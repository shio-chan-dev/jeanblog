---
title: "NMS 描述：非极大值抑制的原理与工程实践"
date: 2026-01-24T16:32:59+08:00
draft: false
categories: ["AI", "Vision"]
tags: ["nms", "object-detection", "iou", "post-processing", "pytorch"]
description: "系统讲清 NMS 的核心流程、IoU 计算与工程取舍，并给出最小 PyTorch 示例。"
keywords: ["NMS", "非极大值抑制", "IoU", "目标检测", "后处理"]
---

> **副标题 / 摘要**  
> NMS（Non-Maximum Suppression）是目标检测后处理的核心步骤。本文用 ACERS 框架拆解 NMS 的原理、流程与工程实践，并提供可运行的 PyTorch 示例。

- **预计阅读时长**：14~18 分钟
- **标签**：`nms`、`object-detection`、`iou`
- **SEO 关键词**：NMS, 非极大值抑制, IoU, 目标检测
- **元描述**：讲清 NMS 的核心算法、复杂度与工程取舍。

---

## 目标读者

- 想理解目标检测后处理的初学者
- 需要调参 IoU 阈值的工程实践者
- 关注推理速度与精度平衡的开发者

## 背景 / 动机

检测模型通常会输出多个重叠框。  
如果不做抑制，会出现“同一目标被重复检测”。  
NMS 用最简单的规则实现去重，是工业界的标准方案。

## 核心概念

- **IoU（Intersection over Union）**：衡量两个框重叠程度。
- **score**：置信度分数，决定优先保留的框。
- **阈值**：IoU 超过阈值则抑制。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

NMS 的逻辑很直观：

1. 选出最高分的框。
2. 删除与它重叠度过高的框。
3. 重复直到没有框。

### 基础示例（1）

- 两个高度重叠的人脸框，只保留分数更高的一个。

### 基础示例（2）

- 多个类别的检测结果，先按类别分开再做 NMS（class-wise）。

## 实践指南 / 步骤

1. 对检测框按 score 排序。
2. 取最高分框作为保留结果。
3. 计算 IoU，过滤高重叠框。
4. 重复直到框集合为空。

## 可运行示例（最小 PyTorch NMS）

```python
import torch


def iou(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def nms(boxes, scores, thresh=0.5):
    idx = scores.argsort(descending=True)
    keep = []
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i.item())
        if idx.numel() == 1:
            break
        rest = idx[1:]
        ious = iou(boxes[i], boxes[rest])
        idx = rest[ious <= thresh]
    return keep

boxes = torch.tensor([
    [0.0, 0.0, 1.0, 1.0],
    [0.1, 0.1, 1.1, 1.1],
    [2.0, 2.0, 3.0, 3.0],
])
scores = torch.tensor([0.9, 0.8, 0.7])
print(nms(boxes, scores, thresh=0.5))
```

## 解释与原理

- NMS 的核心是“先保留最可信框”。
- IoU 阈值越大，保留框越多；越小，抑制越强。

## C — Concepts（核心思想）

### 方法类型

NMS 属于**后处理过滤算法**，用局部贪心策略去重。

### 关键公式

IoU：

$ \text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} $

### 解释与原理

- IoU 衡量重叠程度。
- 通过阈值控制抑制强度。

## E — Engineering（工程应用）

### 场景 1：目标检测去重

- 背景：同一目标被多个框预测。
- 为什么适用：NMS 快速去重。
- 代码示例（Python）：

```python
import torch

boxes = torch.randn(5, 4).abs()
scores = torch.rand(5)
print(scores.argsort(descending=True))
```

### 场景 2：多类别检测（class-wise NMS）

- 背景：不同类别框不应互相抑制。
- 为什么适用：按类别分组 NMS。
- 代码示例（Python）：

```python
import torch

labels = torch.tensor([0, 0, 1, 1])
for c in labels.unique():
    idx = (labels == c).nonzero().flatten()
    print(c.item(), idx.tolist())
```

### 场景 3：实时检测加速

- 背景：NMS 成为推理瓶颈。
- 为什么适用：减少候选框数量。
- 代码示例（Python）：

```python
import torch

scores = torch.rand(1000)
keep = scores > 0.3
print(keep.sum().item())
```

## R — Reflection（反思与深入）

- **时间复杂度**：经典 NMS 为 `O(N^2)`。
- **空间复杂度**：主要存储候选框，`O(N)`。
- **替代方案**：
  - Soft-NMS：降低分数而非直接删除。
  - DIoU/CIoU-NMS：更精细的抑制策略。
  - NMS-Free：直接在模型中建模去重。
- **工程可行性**：NMS 简单稳定，是默认首选。

## 常见问题与注意事项

- 阈值过大 → 重复检测。
- 阈值过小 → 漏检。
- 跨类别是否抑制需明确策略。

## 最佳实践与建议

- 先用 class-wise NMS 再做全局筛选。
- 在高密度场景考虑 Soft-NMS。
- 用验证集调 IoU 阈值。

## S — Summary（总结）

### 核心收获

- NMS 是目标检测后处理的核心算法。
- IoU 阈值决定去重强度。
- 复杂度高时需要候选筛选或改进算法。
- Soft-NMS 与 NMS-Free 是重要替代方向。

### 推荐延伸阅读

- Faster R-CNN
- Soft-NMS 论文
- YOLO 系列检测模型

## 参考与延伸阅读

- https://arxiv.org/abs/1704.04503
- https://arxiv.org/abs/1704.04503
- https://arxiv.org/abs/1506.02640

## 小结 / 结论

NMS 是目标检测工程化的“最后一公里”。  
理解它的取舍，能让检测系统更稳、更快。

## 行动号召（CTA）

用本文的 NMS 实现替换你的后处理代码，观察精度与速度变化。
