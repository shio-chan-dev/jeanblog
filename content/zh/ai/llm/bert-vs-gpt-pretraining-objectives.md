---
title: "BERT vs GPT：预训练任务与应用差异"
date: 2026-01-24T16:12:12+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["bert", "gpt", "pretraining", "mlm", "clm"]
description: "对比 BERT 与 GPT 的预训练目标、架构假设与工程场景，并给出最小可运行示例。"
keywords: ["BERT", "GPT", "MLM", "CLM", "预训练"]
---

> **副标题 / 摘要**  
> BERT 通过 MLM/NSP 学习双向语义，GPT 通过 CLM 学习自回归生成。本文用 ACERS 框架对比两者预训练任务与应用场景，并提供最小 PyTorch 示例。

- **预计阅读时长**：14~18 分钟
- **标签**：`bert`、`gpt`、`pretraining`
- **SEO 关键词**：BERT, GPT, MLM, CLM
- **元描述**：对比 BERT 与 GPT 的预训练目标与工程应用差异。

---

## 目标读者

- 想入门理解 BERT 与 GPT 核心差异的读者
- 需要做模型选型的工程实践者
- 关注 NLP 任务适配策略的开发者

## 背景 / 动机

BERT 和 GPT 经常被混用，但它们的预训练目标决定了“擅长什么”。  
理解 MLM 与 CLM 的差异，能更快做出任务匹配与架构选型。

## 核心概念

- **MLM（Masked Language Modeling）**：随机遮蔽词，预测被遮蔽词。
- **NSP（Next Sentence Prediction）**：判断两句是否相邻（BERT 原版）。
- **CLM（Causal Language Modeling）**：预测下一个词（自回归）。

## A — Algorithm（题目与算法）

### 用通俗语言说明主题内容

- BERT 是“看全句补空词”的双向理解模型。
- GPT 是“从左到右续写”的生成模型。

### 基础示例（1）

- 输入："北京是[MASK]国首都" → BERT 预测“中”。

### 基础示例（2）

- 输入："北京是中国" → GPT 预测下一个词“首都”。

## 实践指南 / 步骤

1. 任务是理解/分类 → 首选 BERT 类模型。
2. 任务是生成/续写 → 首选 GPT 类模型。
3. 推理时注意：BERT 需要 [MASK]，GPT 需要 prompt。

## 可运行示例（最小 PyTorch 逻辑）

```python
import torch
import torch.nn.functional as F

# MLM: 预测被遮蔽位置
vocab = 100
seq = torch.tensor([[1, 2, 3, 4]])
mask_pos = 2
logits = torch.randn(1, 4, vocab)
mlm_loss = F.cross_entropy(logits[:, mask_pos], torch.tensor([3]))
print("MLM loss:", mlm_loss.item())

# CLM: 预测下一个 token
logits = torch.randn(1, 4, vocab)
labels = torch.tensor([[2, 3, 4, 5]])
clm_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1))
print("CLM loss:", clm_loss.item())
```

## 解释与原理

- MLM 学到双向上下文，因此更适合理解类任务。
- CLM 强调顺序生成，因此更适合生成类任务。
- GPT 不需要特殊 [MASK]，推理更自然。

## C — Concepts（核心思想）

### 方法类型

BERT 属于**双向编码器预训练**，GPT 属于**自回归生成预训练**。

### 关键公式

**MLM：**

$ L_{MLM} = -\sum_{i \in M} \log p(x_i | x_{\setminus i}) $

**CLM：**

$ L_{CLM} = -\sum_{t} \log p(x_t | x_{<t}) $

### 解释与原理

- BERT 通过遮蔽预测学习全局语义。
- GPT 通过因果遮罩确保生成因果性。

## E — Engineering（工程应用）

### 场景 1：文本分类（BERT）

- 背景：情感分析、意图识别。
- 为什么适用：双向语义表征更强。
- 代码示例（Python）：

```python
import torch

emb = torch.randn(1, 768)
logits = torch.randn(1, 2)
print(logits.argmax(dim=1).item())
```

### 场景 2：文本生成（GPT）

- 背景：对话、续写、摘要。
- 为什么适用：自回归生成天然适合输出序列。
- 代码示例（Python）：

```python
import torch

prompt = torch.tensor([[1, 2, 3]])
next_token = torch.randint(0, 100, (1, 1))
print(next_token.item())
```

### 场景 3：检索增强（BERT/GPT 组合）

- 背景：先检索，再生成答案。
- 为什么适用：BERT 做检索/排序，GPT 做生成。
- 代码示例（Python）：

```python
import torch

score = torch.randn(1, 10)
idx = score.argmax(dim=1).item()
print(idx)
```

## R — Reflection（反思与深入）

- **时间复杂度**：两者都为 `O(n^2)` 注意力计算。
- **空间复杂度**：依赖序列长度与隐藏维度。
- **替代方案**：
  - RoBERTa：去除 NSP，增强 MLM。
  - GPT-2/3：更大规模自回归预训练。
- **工程可行性**：理解类任务更适合 BERT，生成类任务更适合 GPT。

## 常见问题与注意事项

- BERT 推理需要 [MASK] 或分类头。
- GPT 生成需要控制温度与长度。
- 两者都可迁移，但任务适配方式不同。

## 最佳实践与建议

- 先明确任务类型，再选预训练目标。
- 生成任务优先用 GPT 类；理解任务优先用 BERT 类。
- 混合系统可组合使用两者优势。

## S — Summary（总结）

### 核心收获

- BERT 与 GPT 的差异核心在预训练目标。
- MLM 更适合理解任务，CLM 更适合生成任务。
- 任务选型决定模型效果上限。
- 组合使用可发挥各自优势。

### 推荐延伸阅读

- BERT 论文：Pre-training of Deep Bidirectional Transformers for Language Understanding
- GPT 论文：Improving Language Understanding by Generative Pre-Training
- RoBERTa 论文

## 参考与延伸阅读

- https://arxiv.org/abs/1810.04805
- https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- https://arxiv.org/abs/1907.11692

## 小结 / 结论

BERT 与 GPT 是“理解与生成”的两条路线。  
理解差异后，模型选型会更直接、工程路径更清晰。

## 行动号召（CTA）

把你的任务映射到“理解或生成”，再决定用哪类预训练模型。
