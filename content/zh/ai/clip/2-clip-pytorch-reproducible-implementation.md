---
title: "CLIP 系列（2/3）：PyTorch 完整可复现实战——从数据到训练闭环"
date: 2026-01-24T12:46:49+08:00
draft: false
categories: ["AI", "Multimodal"]
tags: ["clip", "pytorch", "reproducible", "cifar10", "training", "multimodal"]
description: "用 CIFAR-10 + 文本提示搭建最小 CLIP 训练闭环，提供完整可复现的 PyTorch 实战脚本。"
keywords: ["CLIP", "PyTorch", "可复现", "CIFAR10", "对比学习", "多模态"]
---

> **副标题 / 摘要**  
> 这篇文章给出一个“最小但完整”的 CLIP 训练闭环：CIFAR-10 图像 + 文本提示，配套可直接运行的 PyTorch 脚本，确保你可以本地复现训练与零样本分类。

- **预计阅读时长**：20~25 分钟
- **标签**：`clip`、`pytorch`、`reproducible`、`cifar10`
- **SEO 关键词**：CLIP, PyTorch, 可复现, CIFAR10, 对比学习
- **元描述**：从数据准备到训练与评估，给出完整可复现的 CLIP PyTorch 实战脚本。

---

## 系列导航

- （1/3）原理与对比学习公式
- （2/3）PyTorch 完整可复现实战（本文）
- （3/3）工程化与优化

## 目标读者

- 想跑通 CLIP 训练闭环的初学者
- 需要可复现实验模板的工程实践者
- 希望基于 PyTorch 做多模态原型验证的读者

## 背景 / 动机

CLIP 的训练流程看起来简单，但“可复现”很难：  
缺数据、缺脚本、缺评估，导致很多实验停在“理论上懂了”。  
本篇用一个小数据集闭环复现，优先保证你能**在本地跑起来**。

## 核心概念

- **可复现性**：固定随机种子、控制数据划分与预处理。
- **弱标注文本**：用类名构造文本提示，模拟图文对齐。
- **对比损失**：双向交叉熵 + 温度参数。
- **零样本评估**：用文本提示作为“类别描述”进行分类。

## A — Algorithm（题目与算法）

### 训练闭环的核心流程

1. 为每张图像生成文本提示（如 `a photo of a cat`）。
2. 图像与文本分别编码成向量并归一化。
3. 计算相似度矩阵并用对比损失训练。
4. 推理时用“文本提示集合”做零样本分类。

### 基础示例（1）

- 图像：一只猫
- 文本提示集合：`cat`, `dog`, `car`
- 目标：相似度最高的提示即为预测类别

### 基础示例（2）

- 同一 batch 内，对角线是“正确图文对”
- 训练目标：对角线最大化，非对角线最小化

## 实践指南 / 步骤

1. 创建环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision tqdm
```

2. 把下面脚本保存为 `clip_cifar10.py`。
3. 运行训练（推荐 GPU）：

```bash
python clip_cifar10.py --epochs 10 --batch-size 256 --device cuda
```

4. 观察输出：loss 逐步下降，零样本准确率逐步上升。

## 可运行示例（完整 PyTorch 脚本）

```python
import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleTokenizer:
    def __init__(self, texts):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        for text in texts:
            for token in text.lower().split():
                if token not in vocab:
                    vocab[token] = len(vocab)

        self.stoi = vocab
        self.itos = {i: t for t, i in vocab.items()}
        self.pad_id = self.stoi[self.pad_token]
        self.unk_id = self.stoi[self.unk_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

    def encode(self, text, max_len=16):
        tokens = text.lower().split()
        ids = [self.bos_id]
        ids.extend(self.stoi.get(t, self.unk_id) for t in tokens)
        ids.append(self.eos_id)

        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.eos_id
        return ids


def pad_tokens(token_lists, pad_id):
    max_len = max(len(t) for t in token_lists)
    tokens = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(token_lists), max_len), dtype=torch.bool)
    for i, ids in enumerate(token_lists):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = True
    return tokens, attn


class CIFAR10Text(Dataset):
    def __init__(self, root, train, transform, tokenizer, max_len=16):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        text = self.prompts[label]
        token_ids = self.tokenizer.encode(text, max_len=self.max_len)
        return image, token_ids, label


def collate_fn(batch, pad_id):
    images, token_lists, labels = zip(*batch)
    images = torch.stack(images)
    tokens, attn = pad_tokens(token_lists, pad_id)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, tokens, attn, labels


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, width=256, layers=2, heads=4, max_len=16):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.pos = nn.Embedding(max_len, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj = nn.Linear(width, embed_dim)

    def forward(self, token_ids, attn_mask):
        bsz, seq_len = token_ids.shape
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token(token_ids) + self.pos(pos_ids)
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        attn = attn_mask.unsqueeze(-1)
        x = (x * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class CLIPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_len=16):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, max_len=max_len)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, images, token_ids, attn_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(token_ids, attn_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * image_features @ text_features.T
        return logits


def clip_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


@torch.no_grad()
def zero_shot_accuracy(model, loader, tokenizer, device, max_len=16):
    model.eval()
    prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
    token_lists = [tokenizer.encode(p, max_len=max_len) for p in prompts]
    tokens, attn = pad_tokens(token_lists, tokenizer.pad_id)
    tokens = tokens.to(device)
    attn = attn.to(device)
    text_features = model.text_encoder(tokens, attn)

    correct = 0
    total = 0
    for images, _, _, labels in loader:
        images = images.to(device)
        image_features = model.image_encoder(images)
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="./data")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    set_seed(args.seed)

    prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
    tokenizer = SimpleTokenizer(prompts)

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = CIFAR10Text(args.data_root, True, train_tf, tokenizer, args.max_len)
    test_ds = CIFAR10Text(args.data_root, False, test_tf, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    model = CLIPModel(len(tokenizer.stoi), embed_dim=args.embed_dim, max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, tokens, attn, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            tokens = tokens.to(device)
            attn = attn.to(device)

            logits = model(images, tokens, attn)
            loss = clip_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        acc = zero_shot_accuracy(model, test_loader, tokenizer, device, args.max_len)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, zero-shot acc={acc:.4f}")


if __name__ == "__main__":
    main()
```

## C — Concepts（核心思想）

### 方法类型

CLIP 属于**对比学习 + 多模态表示学习**范式，采用图文双塔编码器对齐语义空间。

### 关键公式（最小闭环）

相似度矩阵：

$ S_{ij} = \frac{v_i^\top t_j}{\tau} $

损失函数：

$ L = \frac{\text{CE}(S, y) + \text{CE}(S^\top, y)}{2} $

其中 `y` 为对角线匹配标签。

### 解释与原理

- **弱标注文本**：用类名构造 prompt，形成图文对；虽然不如真实描述丰富，但足以验证训练闭环。
- **对称损失**：`clip_loss` 同时优化图像检索文本与文本检索图像。
- **零样本评估**：只需把类名变成文本提示即可完成分类。

## E — Engineering（工程应用）

### 场景 1：迁移到垂直领域数据

- 背景：业务图像与通用数据集差异较大。
- 为什么适用：CLIP 允许用“文本提示”快速构造弱标注。
- 代码示例（Python）：

```python
from torch.utils.data import Dataset
from PIL import Image

class LocalImageText(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, text = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), text
```

### 场景 2：冻结文本编码器提升训练稳定性

- 背景：小数据集容易过拟合文本侧。
- 为什么适用：冻结一侧可减少参数更新噪声。
- 代码示例（Python）：

```python
for p in model.text_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
)
```

### 场景 3：批量生成向量用于离线检索

- 背景：线上检索需要提前构建向量库。
- 为什么适用：CLIP 的编码器可直接输出归一化向量。
- 代码示例（Python）：

```python
import torch

model.eval()
embeddings = []

with torch.no_grad():
    for images, _, _, _ in loader:
        images = images.to(device)
        vec = model.image_encoder(images).cpu()
        embeddings.append(vec)

embeddings = torch.cat(embeddings, dim=0)
print(embeddings.shape)
```

## R — Reflection（反思与深入）

- **时间复杂度**：训练时需要 `O(N^2)` 的相似度计算。
- **空间复杂度**：相似度矩阵为 `N x N`，显存占用高。
- **替代方案**：
  - 使用预训练 CLIP（open_clip、transformers）直接微调。
  - 用分类模型做封闭集任务，训练更快但无法零样本泛化。
- **工程可行性**：在真实场景中，通常先用预训练模型，再做少量领域微调。

## 常见问题与注意事项

- CIFAR-10 的文本提示是弱监督，不代表真实图文描述。
- batch 太小会显著削弱对比学习的训练信号。
- 小数据集上准确率波动较大，需多次运行取平均。

## 最佳实践与建议

- 保留训练日志与随机种子，方便对比实验。
- 优先验证“能跑通”，再追求更高精度。
- 后续可替换为更真实的图文数据集。

## S — Summary（总结）

### 核心收获

- 一个最小 CLIP 训练闭环即可复现对比学习效果。
- 弱标注文本足以验证流程，但真实数据更关键。
- 零样本分类可以直接用文本提示完成。
- 复现实验的关键是固定随机性与数据预处理。

### 推荐延伸阅读

- OpenCLIP 文档与训练指南
- PyTorch 官方对比学习教程
- CLIP 论文实现细节

### 小结 / 结论

这份最小脚本强调“能跑通”的价值：先保证可复现，再逐步替换为更真实的数据与更强的模型。  
一旦闭环跑通，你就拥有了可持续迭代的实验基线。

## 参考与延伸阅读

- https://github.com/mlfoundations/open_clip
- https://pytorch.org/tutorials
- https://arxiv.org/abs/2103.00020

## 行动号召（CTA）

跑通这个脚本后，继续阅读系列第 3 篇，把 CLIP 引入检索系统与工程部署。
