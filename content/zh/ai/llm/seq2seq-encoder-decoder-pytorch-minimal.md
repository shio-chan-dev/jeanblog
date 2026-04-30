---
title: "Seq2Seq 与 Encoder-Decoder：从翻译任务到最小可运行 PyTorch 实现"
date: 2026-04-23T15:27:55+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["seq2seq", "encoder-decoder", "gru", "pytorch", "autoregressive"]
---

> **副标题 / 摘要**
> 这篇文章不把 seq2seq 和 encoder-decoder 当成术语表来讲，而是从一个最小翻译任务出发，解释为什么“输入一段序列、输出另一段序列”会自然逼出编码器和解码器的分工，最后收束成一份最小可运行的 PyTorch GRU 实现。

## 从一个最小翻译任务开始

假设源序列是：

```text
<bos> I love apples <eos>
```

目标序列是：

```text
<bos> 我 喜欢 苹果 <eos>
```

当模型要生成“苹果”时，它至少要解决三件事：

- 它必须知道整句英文大意，而不只是当前一个词
- 它必须记住自己前面已经生成了“我 喜欢”
- 它必须按顺序一个词一个词地产生输出，而不是一次吐出整个目标序列

如果你只用一个普通分类器把源句子映射成一个类别，这个任务做不成。
因为这里的输出不是一个固定标签，而是长度可变的目标序列。

所以这里天然会逼出一个更具体的数据流：

1. 先把源序列读完，压成可传递的状态
2. 再从 `<bos>` 开始，逐步生成目标序列
3. 每生成一步，都要同时依赖“源侧信息”和“目标侧历史”

这就是 `sequence-to-sequence` 的最小问题形态。
`seq2seq` 说的是任务：输入一段序列，输出另一段序列。
`encoder-decoder` 说的是实现：先编码输入，再逐步解码输出。

下面不先堆名词，直接按这个压力把代码一步一步长出来。

## 快速掌握地图

- 问题形态：`src -> encoder -> hidden/context -> decoder -> logits`
- 核心目标：学习条件概率 `p(y_t | y_{<t}, x)`
- 最小实现：`Embedding + GRU Encoder + GRU Decoder + output projection`
- 何时适用：翻译、摘要、改写、问答这类“输入序列 -> 输出序列”任务
- 明显局限：如果所有源信息都被压进一个固定长度向量，长句子会吃力

## 这篇文章重点深挖的两个概念

1. **隐藏状态交接**：encoder 到底把什么交给 decoder
2. **右移目标序列与 teacher forcing**：训练时 decoder 为什么不能直接喂完整真实答案

## 大师级心智模型

这类模型的核心抽象不是“两个 RNN 拼起来”，而是：

> 用一个条件状态机去建模目标序列的逐步生成。

更具体一点：

- encoder 负责把源序列读成条件状态
- decoder 负责在这个条件状态下做 next-token prediction
- 每一步预测的对象，都是“下一个目标 token”

所以它和语言模型的关系很近。
不同在于语言模型只条件于左侧历史，而 seq2seq 还额外条件于源序列。

写成公式就是：

$$
p(y_1, y_2, \dots, y_m \mid x) = \prod_{t=1}^{m} p(y_t \mid y_{<t}, x)
$$

其中：

- `x` 是源序列
- `y_t` 是第 `t` 个目标 token
- `y_{<t}` 是它之前已经生成的目标历史

如果你记住这条公式，后面的 encoder、decoder、teacher forcing 都只是为了把它实现出来。

## 可行性与下界直觉

先说一个固定长度状态的硬限制。

如果源序列长度是 `n=8`，把整句压到一个隐藏向量里通常还能工作。
但如果长度变成 `n=128` 或 `n=512`，模型就被迫把所有关键信息都塞进一个固定宽度 `hidden_dim` 的状态里。

例如：

- `hidden_dim = 64`
- 源序列长度 `n = 128`

这不表示“64 维不能表示 128 个 token”，而是说：

- 所有对翻译有用的信息都必须经过反复更新后挤进同一个状态
- decoder 后续只能依赖这个状态和自己的历史
- 如果前面信息在编码阶段已经被冲淡，后面就拿不回来了

这也是经典 seq2seq 在长句子上常见的瓶颈。
后来的 attention，本质上就是在修这个问题。

## Step 1：先把源序列读成一个可传递状态

第一步只解决一个问题：

> 如果输出要依赖整个输入序列，我们先把输入读到哪里？

最直接的办法是：

- 先把 token id 变成 embedding
- 再让一个循环单元按顺序读完整个源序列
- 读完后拿最后的隐藏状态作为“源句摘要”

这里选 `GRU`，不是因为它唯一正确，而是因为它比手写 vanilla RNN 更稳定，又比 LSTM 少一个 `cell state`，适合做最小教学实现。

最小 `Encoder` 是：

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(src_ids)              # [B, S, emb_dim]
        outputs, hidden = self.rnn(x)           # outputs: [B, S, H], hidden: [1, B, H]
        return outputs, hidden
```

这一版现在能做到：

- 把 `[B, S]` 的源序列读成一串隐藏状态 `outputs`
- 额外给出最后的隐藏状态 `hidden`

这里两个输出的区别要马上说清楚：

- `outputs[:, t, :]` 表示 encoder 在第 `t` 个源位置的状态
- `hidden` 表示“读完整句后”的最终状态

在这个最小版本里，我们先只用 `hidden`。
因为这篇文章的目标是先理解最基本的 encoder-decoder，不先引入 attention。

但它还缺：

- 目标侧怎么一步一步生成
- 生成时 decoder 到底吃什么输入

## Step 2：只生成一个目标 token，先长出最小 Decoder

第二步只解决一个问题：

> 如果 decoder 现在要预测下一个词，它需要什么？

最少需要两样东西：

1. 它上一步的隐藏状态
2. 当前输入 token

这里的“当前输入 token”通常不是要预测的那个词本身，而是它左边已经存在的那个词。
例如要预测“喜欢”，decoder 当前输入应该是“我”。

所以最小 `Decoder` 是：

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids)           # [B, 1, emb_dim]
        outputs, hidden = self.rnn(x, hidden)  # outputs: [B, 1, H]
        logits = self.output_proj(outputs)      # [B, 1, vocab_size]
        return logits, hidden
```

这一版现在能做到：

- 给定一个当前 token 和一个隐藏状态
- 输出下一个词的打分 `logits`
- 同时更新 decoder 自己的隐藏状态

这时最关键的因果关系是：

- `hidden` 提供“到当前为止记住了什么”
- `token_ids` 提供“这一步刚刚看到了哪个词”
- `logits` 表示“下一步最可能输出什么词”

但它还缺：

- encoder 的最终状态怎么交给 decoder
- 整个目标序列怎么按步展开

## Step 3：把 encoder 的最终状态交给 decoder

现在开始回答文章里第一个需要深挖的核心概念：

> encoder 到底交给 decoder 的是什么？

在这个最小 GRU 版本里，交接物就是 `hidden`。

更具体地说：

- encoder 读完整个源序列后，得到 `hidden_enc`
- decoder 的初始隐藏状态直接设为 `hidden_enc`
- 这表示 decoder 从一开始就带着“源句条件”开始生成

也就是说，最小 encoder-decoder 的桥梁不是额外的模块，而是**隐藏状态初始化**。

用一个极小例子看这个信息流：

- 源句：`<bos> I love apples <eos>`
- 编码完成后，得到 `hidden_enc`
- decoder 第一步输入 `<bos>`
- decoder 在 `hidden_enc` 的条件下预测第一个中文词“我”

这个版本的优势是简单，代价也很明确：

- 优势：接口干净，代码很短
- 代价：所有源信息都必须挤进一个固定长度 `hidden_enc`

这就是前面说的 fixed-length context bottleneck。

## Step 4：训练时不能直接喂答案，所以要右移目标序列

现在进入第二个需要深挖的核心概念：

> 训练时为什么 decoder 不能直接把完整真实目标序列当输入？

因为 decoder 的工作是“用左侧历史预测下一个 token”。
如果你把当前时刻要预测的词也直接喂进去，任务就被泄露了。

最标准的做法是把目标序列拆成两份：

```text
tgt_input_ids = <bos> 我 喜欢 苹果
tgt_label_ids = 我 喜欢 苹果 <eos>
```

这叫**右移一位**。

含义是：

- decoder 在第 1 步看到 `<bos>`，预测“我”
- decoder 在第 2 步看到“我”，预测“喜欢”
- decoder 在第 3 步看到“喜欢”，预测“苹果”
- decoder 在第 4 步看到“苹果”，预测 `<eos>`

这就是 teacher forcing 的最小形式：

- 训练时，把真实左侧历史喂给 decoder
- 不让它用自己前一步的错误预测继续滚雪球

如果你用上面的翻译例子，那么训练对齐关系是：

| step | decoder 输入 | 预测目标 |
| --- | --- | --- |
| 1 | `<bos>` | `我` |
| 2 | `我` | `喜欢` |
| 3 | `喜欢` | `苹果` |
| 4 | `苹果` | `<eos>` |

这一点如果不先吃透，后面看 Transformer 里的 `tgt_input_ids` 和 causal mask 会一直混。

## Step 5：把训练时的逐步解码接成一个 Seq2Seq 类

到这一步，我们已经有：

- 一个能把源序列读成 `hidden` 的 `Encoder`
- 一个能根据 `(token, hidden)` 预测下一词的 `Decoder`

接下来只差一个总装类，把二者接起来。

先看最小版本：

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids: torch.Tensor, tgt_input_ids: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(src_ids)

        logits_list = []
        dec_input = tgt_input_ids[:, :1]

        for t in range(tgt_input_ids.size(1)):
            logits, hidden = self.decoder(dec_input, hidden)
            logits_list.append(logits)

            if t + 1 < tgt_input_ids.size(1):
                dec_input = tgt_input_ids[:, t + 1:t + 2]

        return torch.cat(logits_list, dim=1)
```

这里最关键的执行链是：

1. `encoder(src_ids)` 把源序列读完
2. 拿 encoder 最后的 `hidden` 作为 decoder 初始状态
3. 让 decoder 从 `<bos>` 开始一步一步生成训练 logits
4. 把每一步的输出拼成 `[B, T_tgt, vocab_size]`

这一版现在能做到：

- 完整跑通 teacher forcing 训练前向
- 给每个目标位置输出一个词表打分向量

但它还缺：

- 损失怎么和 label 对齐
- 一个最小完整 demo，证明整个东西真的能跑

## 最后一段：把前面的模块收束成最小可运行实现

下面这段代码把前面长出来的 `Encoder`、`Decoder`、`Seq2Seq` 接成一份完整 demo。
它不是工业级翻译系统，但已经是一个真正能前向、能算 loss、能帮助你理解 encoder-decoder 数据流的最小实现。

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(src_ids)
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids)
        outputs, hidden = self.rnn(x, hidden)
        logits = self.output_proj(outputs)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids: torch.Tensor, tgt_input_ids: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(src_ids)

        logits_list = []
        dec_input = tgt_input_ids[:, :1]

        for t in range(tgt_input_ids.size(1)):
            logits, hidden = self.decoder(dec_input, hidden)
            logits_list.append(logits)

            if t + 1 < tgt_input_ids.size(1):
                dec_input = tgt_input_ids[:, t + 1:t + 2]

        return torch.cat(logits_list, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)

    src_vocab_size = 16
    tgt_vocab_size = 16
    pad_id = 0

    src_ids = torch.tensor([
        [1, 5, 7, 9, 2],
        [1, 4, 8, 2, 0],
    ])

    tgt_input_ids = torch.tensor([
        [1, 6, 3, 10],
        [1, 9, 2, 0],
    ])

    tgt_label_ids = torch.tensor([
        [6, 3, 10, 2],
        [9, 2, 0, 0],
    ])

    encoder = Encoder(vocab_size=src_vocab_size, emb_dim=32, hidden_dim=64)
    decoder = Decoder(vocab_size=tgt_vocab_size, emb_dim=32, hidden_dim=64)
    model = Seq2Seq(encoder, decoder)

    logits = model(src_ids, tgt_input_ids)
    print(logits.shape)  # torch.Size([2, 4, 16])

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(logits.reshape(-1, tgt_vocab_size), tgt_label_ids.reshape(-1))
    print(round(loss.item(), 4))
```

## 这份完整代码现在到底在做什么

如果用上面的 toy batch：

- `src_ids` 形状是 `[2, 5]`
- `tgt_input_ids` 形状是 `[2, 4]`
- `tgt_label_ids` 形状是 `[2, 4]`

那么关键形状变化是：

1. encoder embedding 后：
   - `[2, 5, 32]`
2. encoder 输出：
   - `outputs`: `[2, 5, 64]`
   - `hidden`: `[1, 2, 64]`
3. decoder 每一步输出：
   - `logits_t`: `[2, 1, 16]`
4. 全部拼接后：
   - `logits`: `[2, 4, 16]`

这意味着：

- 每个目标位置都会得到一个长度为 `16` 的词表打分向量
- loss 会把这些打分与右移后的 `tgt_label_ids` 对齐
- `ignore_index=0` 会让 padding 位置不参与损失

## Worked Example：一步一步看数据流

继续用第一条样本：

```text
src_ids       = <bos> I love apples <eos>
tgt_input_ids = <bos> 我 喜欢 苹果
tgt_label_ids = 我 喜欢 苹果 <eos>
```

训练前向时发生的事情是：

1. encoder 先读完整个源句，得到最终 `hidden_enc`
2. decoder 第一步输入 `<bos>`，结合 `hidden_enc`，输出对“我”的词表打分
3. decoder 第二步输入真实词“我”，输出对“喜欢”的打分
4. decoder 第三步输入真实词“喜欢”，输出对“苹果”的打分
5. decoder 第四步输入真实词“苹果”，输出对 `<eos>` 的打分

这就是为什么：

- `tgt_input_ids` 是“喂给 decoder 的历史”
- `tgt_label_ids` 是“希望它预测出来的下一个词”

如果你把这条链真正看顺了，后面 Transformer 里看到“右移 target 作为输入”就不会再觉得抽象。

## 正确性与不变量

这份最小实现有三个关键不变量。

### 1. 条件不变量

在 decoder 第 `t` 步，模型估计的是：

$$
p(y_t \mid y_{<t}, x)
$$

而不是 `p(y_t | x)`。
也就是说，每一步都同时条件于：

- 源序列 `x`
- 目标左侧历史 `y_{<t}`

### 2. 状态不变量

在不加 attention 的最小版本里，decoder 的隐藏状态始终携带两类信息：

- 从 encoder 初始状态继承来的源侧摘要
- 到当前步为止已经处理过的目标历史

所以可以把 decoder 的 `hidden_t` 理解成：

> “到第 `t` 步为止，基于源句和已生成历史形成的条件状态”

### 3. 对齐不变量

训练时必须保持：

- 输入给 decoder 的是左侧历史
- 作为监督信号的是右移后的真实目标

如果这两个张量没有右移对齐，loss 就不再对应“next-token prediction”。

## 复杂度

设：

- 源序列长度是 `n`
- 目标序列长度是 `m`
- embedding 维度是 `d_e`
- 隐状态维度是 `h`

那么最小 GRU seq2seq 的主要代价可以粗略写成：

- encoder：`O(n (d_e h + h^2))`
- decoder：`O(m (d_e h + h^2 + hV))`

其中 `V` 是目标词表大小。
如果只看循环状态更新，常写成近似：

- encoder：`O(n h^2)`
- decoder：`O(m h^2)`

它的一个现实特点是：

- 时间复杂度对长度是线性的
- 但 encoder 和 decoder 都带顺序依赖，难以像 Transformer 那样在时间维完全并行

这也是为什么：

- 短序列、小模型时，RNN/GRU 版本很直观
- 长序列、大规模训练时，Transformer 更占优

## 常数因子与工程现实

虽然上面写的是线性复杂度，但工程上不要忽略下面这些常数：

- decoder 是按时间步循环的，Python for-loop 本身就会带来调度开销
- 词表投影 `hidden_dim -> vocab_size` 在大词表下不便宜
- batch 内长度差异大时，padding 会浪费不少计算

例如：

- `hidden_dim = 512`
- `vocab_size = 50000`

那最后一层线性投影就有大约 `512 x 50000 = 2560 万` 个权重。
所以很多时候最贵的未必是 RNN 本身，而是输出层和训练数据管线。

## Alternatives and Tradeoffs

### 1. Vanilla RNN vs GRU

- vanilla RNN 更容易讲清递推形式
- 但在 `n >= 50`、`n >= 100` 这类长度上更容易遇到梯度消失
- GRU 多了门控，参数量仍比 LSTM 更小

所以在“最小教学实现”和“能稳定跑起来”之间，GRU 是一个更平衡的选择。

### 2. GRU Seq2Seq vs Attention-based Seq2Seq

- 纯 GRU encoder-decoder：实现短，接口干净
- 加 attention 后：decoder 不必只依赖一个固定长度状态，长句效果更稳
- 代价是每一步解码都要额外和源序列做对齐计算

如果你的目标是给 Transformer 做前置桥接，先学**不带 attention 的最小 seq2seq**最合适。
因为它能让你先看清“编码、解码、右移目标、逐步生成”这些最基本的数据流。

### 3. GRU Seq2Seq vs Transformer

- GRU Seq2Seq：时间上顺序推进，路径长，概念直观
- Transformer：通过 self-attention / cross-attention 缩短信息路径，并提高并行性
- 代价是注意力在长度上通常有更高的内存和计算压力

这也是这篇文章和下一篇 Transformer 文之间的真正桥梁。

## 迁移路径

如果你已经吃透这篇里的最小实现，下一步最值得学的是：

1. **Bahdanau / Luong attention**
   解决 fixed-length context bottleneck，可以接着读 [Attention-Based Seq2Seq：为什么会自然过渡到 Transformer](./attention-based-seq2seq-to-transformer.md)
2. **Transformer encoder-decoder**
   用 attention 取代循环递推
3. **Decoder-only GPT**
   当任务变成纯自回归生成时，可以进一步删掉 encoder

你也可以按这条顺序连起来读：

- [Attention-Based Seq2Seq：为什么会自然过渡到 Transformer](./attention-based-seq2seq-to-transformer.md)
- [Transformer 结构推导：一步一步搭出最小可运行 PyTorch 实现](./transformer-architecture-overview.md)
- [为什么 GPT 是 Decoder-Only：自回归生成的最佳形态](./why-gpt-decoder-only.md)

## 常见问题与注意事项

### 1. `seq2seq` 和 `encoder-decoder` 是一回事吗？

不完全一样。

- `seq2seq` 是任务形式
- `encoder-decoder` 是常见实现方式

很多 seq2seq 任务可以用 encoder-decoder 做，但它们不是同义词。

### 2. 为什么这篇不先讲 attention？

因为这篇的任务是搭桥。
如果一开始就把 attention 也混进来，读者反而会把“基本数据流”和“后续增强机制”搅在一起。

### 3. 为什么训练时 decoder 输入的是目标序列，而不是自己的预测？

因为这是 teacher forcing。
它能让训练更稳定，否则早期错误会一路传播，loss 很难收敛。

### 4. 推理时也这样喂吗？

不是。

训练时你有真实目标序列，可以右移后喂入。
推理时没有真实答案，所以必须把模型上一步预测的 token 再喂回去，这就是自回归解码。

### 5. 这份代码为什么还不算工业级？

因为它还没有加入：

- packed sequence
- attention
- beam search
- scheduled sampling
- tied embeddings
- label smoothing

但这些都属于后续增强，不是理解 seq2seq/encoder-decoder 最小工作机制的前提。

## 最佳实践与建议

- 第一次实现时，把 `hidden_dim` 设成 `32` 或 `64`，先把形状和右移对齐跑通
- 先打印 `src_ids`、`tgt_input_ids`、`tgt_label_ids` 三者的对应关系，很多 bug 出在这里
- 如果 loss 对不上，优先检查是不是把“当前输入”和“当前标签”错位了
- 在进入 Transformer 之前，先确认自己能清楚回答“decoder 每一步到底在条件于什么”

## 小结 / 结论

最小 seq2seq / encoder-decoder 值得先学的，不是“它用了 GRU”这件事，而是下面这条生成链：

1. 先把源序列读成条件状态
2. 再让 decoder 在这个条件状态下逐步做 next-token prediction
3. 训练时用右移目标序列提供左侧历史
4. loss 监督的是“下一个词”，不是“当前输入词”

更具体地说，读完这篇你应该带走四个结论：

- `seq2seq` 讲的是任务形态，`encoder-decoder` 讲的是实现分工
- 最小 encoder-decoder 的桥梁，就是 encoder 最终隐藏状态初始化 decoder
- teacher forcing 的本质，是用真实左侧历史来训练 next-token prediction
- Transformer 不是凭空出现的新名词堆，它是在这个基本框架上继续解决固定长度状态和并行性问题

## 参考与延伸阅读

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention-Based Seq2Seq：为什么会自然过渡到 Transformer](./attention-based-seq2seq-to-transformer.md)
- [Transformer 结构推导：一步一步搭出最小可运行 PyTorch 实现](./transformer-architecture-overview.md)
- [CNN、RNN、LSTM 与 Transformer 的区别与适用场景](../architecture/cnn-rnn-lstm-transformer-comparison.md)
