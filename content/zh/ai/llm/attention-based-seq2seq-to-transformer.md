---
title: "Attention-Based Seq2Seq：为什么会自然过渡到 Transformer"
date: 2026-04-23T15:51:40+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["attention", "seq2seq", "encoder-decoder", "transformer", "pytorch"]
---

> **副标题 / 摘要**
> 这篇文章专门解释一个关键过渡：为什么 fixed-length 的 seq2seq 很快会不够用，attention-based seq2seq 是怎么补上“按需读取源序列”这个能力的，以及这个思路为什么几乎会自然长成 Transformer。最后会收束到一份最小可运行的 PyTorch GRU + additive attention 实现。

## 从“苹果”为什么老翻不准开始

还是用这个最小翻译任务：

```text
src: <bos> I really love green apples <eos>
tgt: <bos> 我 真的 喜欢 青 苹果 <eos>
```

当 decoder 走到要生成“苹果”这一步时，最理想的行为其实很明确：

- 它应该重点回头看源序列里的 `apples`
- 它可能顺手也看一眼 `green`
- 它不能只依赖一个已经被反复压缩过很多轮的最终隐藏状态

如果你用上一篇里那个最小 seq2seq：

- encoder 把整句读完
- 只把最后一个 `hidden_enc` 交给 decoder
- decoder 后面每一步都只靠这个固定长度状态和自己的历史

那么句子一长，这里就会出现一个很具体的问题：

> decoder 明明需要“现在按需去看源序列的某几个位置”，但 fixed-length seq2seq 只给了它“一次性打包好的整句摘要”。

这就是 attention-based seq2seq 出现的真实压力。
它不是为了“概念更高级”，而是因为 decoder 在每个时间步都需要**重新决定自己该看源序列的哪里**。

## 快速掌握地图

- fixed-length seq2seq：`encoder outputs -> 丢弃大部分，只保留 final hidden`
- attention-based seq2seq：`decoder step t -> 对所有 encoder outputs 打分 -> 加权求和得到 context_t`
- 核心收益：不同目标位置可以读取不同源位置
- 仍然存在的限制：encoder 和 decoder 还是循环结构，时间上依然串行
- 通向 Transformer 的关键桥：`decoder state 作为 query，encoder outputs 作为 memory`

## 这篇文章重点深挖的两个概念

1. **对齐分数与上下文向量**：decoder 怎样在每一步决定“该看源序列哪里”
2. **从 attention-based seq2seq 到 Transformer 的结构映射**：哪些东西被保留了，哪些东西被替换了

## 大师级心智模型

fixed-length seq2seq 的核心假设是：

> 整个源序列可以先压成一个固定长度状态，再一次性交给 decoder。

attention-based seq2seq 则把这个假设改成：

> encoder 不只产出一个最终摘要，而是保留每个源位置的表示；decoder 在每一步都自己去检索最相关的位置。

如果把它写成条件概率，公式其实没变：

$$
p(y_1, \dots, y_m \mid x) = \prod_{t=1}^{m} p(y_t \mid y_{<t}, x)
$$

变的是“条件于 `x`”这件事的实现方式。

- fixed-length 版本：`x` 先被压成一个 `hidden_enc`
- attention 版本：`x` 被保留成一整串 encoder outputs，decoder 每一步再取一个 `context_t`

这就是从“单一全局摘要”到“按步检索 memory”的过渡。

## 可行性与下界直觉

先看 fixed-length 版本为什么会吃力。

假设：

- 源序列长度 `n = 40`
- 隐状态维度 `hidden_dim = 128`
- 目标序列长度 `m = 45`

在最小 seq2seq 里，decoder 的每一步都只能拿到同一个 `hidden_enc`。
这不表示 128 维一定装不下 40 个 token，而是说：

- 所有可能有用的对齐信息都已经被混合到一个向量里
- decoder 第 3 步和第 30 步面对的是同一份源侧摘要
- 模型缺少“第 `t` 步只去看第 `i` 个源位置”的显式路径

而一旦把 encoder outputs 全保留：

- encoder memory 形状变成 `[B, S, H]`
- decoder 第 `t` 步可以对 `S` 个源位置逐个打分
- 再通过 softmax 形成一组权重 `alpha_t`

也就是说，fixed-length 版本的信息路径是：

```text
src_i -> hidden_enc -> decoder_t
```

attention 版本的信息路径则变成：

```text
src_i -> encoder_output_i -> context_t -> decoder_t
```

路径虽然还是跨过 encoder 和 decoder，但至少不再被迫先压成一个全局单点。

## Step 1：先把“保留所有 encoder outputs”这件事说清楚

第一步先不写 attention，只先修一个更早的缺口：

> 如果 decoder 以后想回头看源序列，那 encoder 至少不能把中间状态全扔掉。

所以在上一篇最小 seq2seq 里，真正值得保留的不只是 `hidden_enc`，还有：

```python
outputs, hidden = self.rnn(x)
```

其中：

- `outputs`: `[B, S, H]`，每个源位置一个状态
- `hidden`: `[1, B, H]`，读完整句后的最终状态

这里先明确角色分工：

- `outputs` 会成为后面 attention 的源侧 memory
- `hidden` 仍然可以拿来初始化 decoder

这一版现在能做到：

- 既保存整句最终摘要
- 又保留源序列逐位置表示

但它还缺：

- decoder 每一步怎么对这些源位置做选择

## Step 2：decoder 当前这一步，先学会给所有源位置打分

现在开始回答第一个核心概念：

> decoder 怎样知道自己现在该看哪一个源位置？

最直接的办法是：
用 decoder 当前状态去和每个 encoder output 做一次匹配打分。

这里先引入最经典也最适合教学的 **additive attention**：

$$
e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i)
$$

其中：

- `s_{t-1}` 是 decoder 当前步之前的隐藏状态
- `h_i` 是第 `i` 个源位置的 encoder 输出
- `e_{t,i}` 是“第 `t` 个目标位置看第 `i` 个源位置”的原始分数

它的含义并不神秘：

- decoder 先问：“我这一步需要什么信息？”
- 然后逐个问每个源位置：“你是不是我现在最该看的那个位置？”

最小实现可以写成：

```python
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        query = decoder_hidden.transpose(0, 1)                    # [B, 1, H]
        scores = self.score_proj(
            torch.tanh(self.query_proj(query) + self.key_proj(encoder_outputs))
        ).squeeze(-1)                                             # [B, S]
        return scores
```

这一版现在能做到：

- 对每个目标步 `t`
- 给所有源位置 `1...S` 产出一组原始分数

但它还缺：

- 怎么把分数变成可用权重
- 怎么用这些权重真正取回源侧信息

## Step 3：把打分变成权重，再合成这一步的 context

光有分数还不够，因为它们只是“偏好强弱”，还不是一个能直接混合信息的权重分布。

所以接下来做两件事：

1. 在源序列长度维度做 softmax
2. 用得到的权重对 encoder outputs 加权求和

公式是：

$$
\alpha_{t,i} = \text{softmax}(e_{t,i})
$$

$$
c_t = \sum_{i=1}^{S} \alpha_{t,i} h_i
$$

这里：

- `alpha_{t,i}` 表示第 `t` 步对第 `i` 个源位置的注意力权重
- `c_t` 是第 `t` 步真正拿回来的上下文向量

继续把上面的类补完整：

```python
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = decoder_hidden.transpose(0, 1)                    # [B, 1, H]
        scores = self.score_proj(
            torch.tanh(self.query_proj(query) + self.key_proj(encoder_outputs))
        ).squeeze(-1)                                             # [B, S]

        attn_weights = torch.softmax(scores, dim=-1)             # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights
```

这里最关键的形状变化是：

- `attn_weights`: `[B, S]`
- `encoder_outputs`: `[B, S, H]`
- `context`: `[B, 1, H]`

这意味着：

- decoder 第 `t` 步不再只依赖同一个 `hidden_enc`
- 它现在能得到一个专属于第 `t` 步的 `context_t`

如果目标词是“苹果”，理想情况下这一步的 `attn_weights` 会在 `apples` 附近更大。
这就是“对齐”第一次被显式写进模型里。

## Step 4：decoder 这一步不能只吃 token embedding，还要吃 context

现在 attention 已经能产出 `context_t` 了。
下一步要解决的是：

> decoder 这一步怎么把“当前输入词”和“从源侧取回来的上下文”一起用起来？

最直接的办法是把二者拼接起来，再送进 GRU：

```python
rnn_input = torch.cat([token_emb, context], dim=-1)
```

对应的最小 attention decoder 是：

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = AdditiveAttention(hidden_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_emb = self.embedding(token_ids)                          # [B, 1, E]
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([token_emb, context], dim=-1)           # [B, 1, E + H]
        outputs, hidden = self.rnn(rnn_input, hidden)                 # [B, 1, H]
        logits = self.output_proj(torch.cat([outputs, context], dim=-1))
        return logits, hidden, attn_weights
```

这一版现在能做到：

- 当前输入 token 告诉 decoder“我刚看到了什么”
- `context_t` 告诉 decoder“源序列里现在最该参考什么”
- `hidden_t` 继续维护目标侧历史

这时 decoder 的条件关系已经从上一篇的：

$$
y_t \sim p(y_t \mid y_{<t}, hidden_{enc})
$$

自然升级成了：

$$
y_t \sim p(y_t \mid y_{<t}, c_t)
$$

其中 `c_t` 是按当前步动态计算出来的源侧上下文。

## Step 5：把完整 attention-based seq2seq 接起来

到这一步，我们手里已经有：

- 保留所有源位置状态的 `Encoder`
- 产出 `(context_t, attn_weights_t)` 的 `AdditiveAttention`
- 同时吃 `token_emb + context_t` 的 `AttentionDecoder`

接下来只差一个总装：

```python
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids: torch.Tensor, tgt_input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, hidden = self.encoder(src_ids)

        logits_list = []
        attn_list = []
        dec_input = tgt_input_ids[:, :1]

        for t in range(tgt_input_ids.size(1)):
            logits, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_outputs)
            logits_list.append(logits)
            attn_list.append(attn_weights.unsqueeze(1))

            if t + 1 < tgt_input_ids.size(1):
                dec_input = tgt_input_ids[:, t + 1:t + 2]

        return torch.cat(logits_list, dim=1), torch.cat(attn_list, dim=1)
```

这一版现在能做到：

- 训练时按 teacher forcing 逐步生成 logits
- 同时保留每一步对源序列的注意力权重

但它还缺：

- 一份最小完整 demo，证明整条链能跑通
- 以及更重要的：它为什么会自然过渡到 Transformer

## 最后一段：把前面的模块收束成最小可运行 PyTorch 实现

下面这份代码把前面的部件全部接起来。
它不是工业级 NMT 系统，但已经是一个真正能前向、能算 loss、能输出 attention map 的最小 attention-based seq2seq。

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


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = decoder_hidden.transpose(0, 1)                         # [B, 1, H]
        scores = self.score_proj(
            torch.tanh(self.query_proj(query) + self.key_proj(encoder_outputs))
        ).squeeze(-1)                                                  # [B, S]
        attn_weights = torch.softmax(scores, dim=-1)                  # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = AdditiveAttention(hidden_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_emb = self.embedding(token_ids)                           # [B, 1, E]
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([token_emb, context], dim=-1)            # [B, 1, E + H]
        outputs, hidden = self.rnn(rnn_input, hidden)                  # [B, 1, H]
        logits = self.output_proj(torch.cat([outputs, context], dim=-1))
        return logits, hidden, attn_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, hidden = self.encoder(src_ids)

        logits_list = []
        attn_list = []
        dec_input = tgt_input_ids[:, :1]

        for t in range(tgt_input_ids.size(1)):
            logits, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_outputs)
            logits_list.append(logits)
            attn_list.append(attn_weights.unsqueeze(1))

            if t + 1 < tgt_input_ids.size(1):
                dec_input = tgt_input_ids[:, t + 1:t + 2]

        return torch.cat(logits_list, dim=1), torch.cat(attn_list, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)

    src_vocab_size = 20
    tgt_vocab_size = 20
    pad_id = 0

    src_ids = torch.tensor([
        [1, 5, 11, 7, 9, 2],
        [1, 4, 8, 6, 2, 0],
    ])

    tgt_input_ids = torch.tensor([
        [1, 3, 10, 12, 13],
        [1, 9, 6, 2, 0],
    ])

    tgt_label_ids = torch.tensor([
        [3, 10, 12, 13, 2],
        [9, 6, 2, 0, 0],
    ])

    encoder = Encoder(vocab_size=src_vocab_size, emb_dim=32, hidden_dim=64)
    decoder = AttentionDecoder(vocab_size=tgt_vocab_size, emb_dim=32, hidden_dim=64)
    model = Seq2SeqWithAttention(encoder, decoder)

    logits, attn = model(src_ids, tgt_input_ids)
    print(logits.shape)  # torch.Size([2, 5, 20])
    print(attn.shape)    # torch.Size([2, 5, 6])

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(logits.reshape(-1, tgt_vocab_size), tgt_label_ids.reshape(-1))
    print(round(loss.item(), 4))
```

## 这份完整代码现在到底在做什么

如果用上面的 toy batch：

- `src_ids` 形状是 `[2, 6]`
- `tgt_input_ids` 形状是 `[2, 5]`

那么关键张量变化是：

1. encoder 输出：
   - `encoder_outputs`: `[2, 6, 64]`
   - `hidden`: `[1, 2, 64]`
2. decoder 第 `t` 步打分：
   - `scores_t`: `[2, 6]`
3. 第 `t` 步 attention 权重：
   - `attn_weights_t`: `[2, 6]`
4. 第 `t` 步上下文：
   - `context_t`: `[2, 1, 64]`
5. 所有步拼起来：
   - `logits`: `[2, 5, 20]`
   - `attn`: `[2, 5, 6]`

这意味着：

- 目标序列每个位置都有一张对源序列的“读取分布”
- decoder 不再在所有时间步共享同一个源侧摘要
- `attn[t, i]` 越大，表示第 `t` 步越依赖第 `i` 个源位置

## Worked Example：为什么这已经很像“简化版 cross-attention”

假设源句是：

```text
<bos> I really love green apples <eos>
```

目标侧生成到“苹果”这一步时，理想的 attention 可能像这样：

```text
<bos>   I   really   love   green   apples   <eos>
0.02   0.03  0.05    0.08   0.20    0.57     0.05
```

这张权重图的含义很直接：

- 当前步主要读取 `apples`
- 次要参考 `green`
- 几乎不看句首 `<bos>`

注意这里已经出现了一个和 Transformer 非常接近的模式：

- decoder 当前状态提出“查询需求”
- encoder outputs 作为可被检索的 memory
- 输出一个加权汇总的 context

如果你把这个过程抽象化，它其实已经在做：

```text
query(decoder state) -> score against source memory -> weighted sum
```

这就是为什么它会自然过渡到 Transformer 里的 cross-attention。

## 正确性与不变量

这份最小实现有四个关键不变量。

### 1. 对齐分布不变量

对每个目标步 `t`，attention 权重满足：

$$
\sum_{i=1}^{S} \alpha_{t,i} = 1
$$

因为它来自对源长度维度的 softmax。
所以每一步拿回来的 `context_t` 都是对 encoder outputs 的加权平均。

### 2. 信息流方向不变量

在 attention-based seq2seq 里：

- decoder state 决定“现在想找什么”
- encoder outputs 提供“源序列里分别有什么”

也就是说，信息流方向始终是：

```text
decoder query -> source memory -> context
```

这和 Transformer 的 cross-attention 信息流方向是一致的。

### 3. 条件生成不变量

decoder 第 `t` 步估计的仍然是：

$$
p(y_t \mid y_{<t}, x)
$$

attention 并没有改变任务，只是改变了“条件于源序列 `x`”的实现方式。

### 4. 串行解码不变量

即使加了 attention，这个模型依然要按 `t = 1, 2, ..., m` 逐步推进。
所以它虽然修了 fixed-length bottleneck，但没有修“时间上串行”这个问题。

这正是它继续过渡到 Transformer 的动机之一。

## 为什么它会自然过渡到 Transformer

到这里，真正关键的问题已经不是“attention 有什么公式”，而是：

> attention-based seq2seq 已经解决了什么，又还没解决什么？

它已经解决了：

1. decoder 不必只靠一个固定长度向量
2. 每个目标步都能按需读取不同源位置
3. 源序列和目标序列之间的对齐被显式建模

但它还没解决：

1. encoder 仍然是 RNN，源侧信息传播路径长
2. decoder 仍然是 RNN，目标侧历史建模仍然串行
3. attention 只出现在“目标读源”这一步，还没有统一成更通用的序列交互机制

所以再往前走一步，思路几乎是自然的：

- 既然 decoder 可以按需读取 encoder outputs
  那么 encoder 内部自己能不能也用这种“按需读取”来做全局交互？
- 既然源侧可以被当作 memory
  那目标侧历史自己能不能也被当作 memory？
- 既然“匹配打分 + 加权求和”已经有用了
  那能不能把它推广成统一的注意力计算，而不是只给 RNN decoder 当外挂？

这三问一旦成立，Transformer 的轮廓就已经出来了。

## 从 attention-based seq2seq 到 Transformer 的结构映射

| attention-based seq2seq | Transformer 中对应的东西 | 变化点 |
| --- | --- | --- |
| encoder outputs | encoder memory | 保留 |
| decoder 当前状态去读源序列 | cross-attention | 保留并标准化 |
| RNN encoder | encoder self-attention blocks | 被替换 |
| RNN decoder | masked self-attention + cross-attention | 被替换 |
| additive score | dot-product QK^T | 常见实现被替换 |
| 每步串行更新 hidden | 并行计算整段表示 | 训练并行性增强 |

最关键的不是名字，而是这两个对应关系：

1. **attention-based seq2seq 的“目标读源”就是 cross-attention 的前身**
2. **RNN 被拿掉之后，self-attention 接管了源侧和目标侧内部的信息传播**

所以你完全可以把 Transformer 理解成：

> 把 attention 从“RNN decoder 的一个外挂对齐模块”，升级成“整个模型统一的信息交互原语”。

## 复杂度与代价

设：

- 源长度 `n`
- 目标长度 `m`
- 隐维 `h`

那么 attention-based GRU seq2seq 的主要代价是：

- encoder GRU：`O(n h^2)`
- decoder GRU：`O(m h^2)`
- attention 对齐：`O(m n h)` 级别的打分与加权

和 fixed-length 版本相比，多出来的核心代价是：

- 每个目标步都要遍历所有源位置

但换来的能力也很明确：

- 不再用一个固定长度向量承包全部源侧信息

和 Transformer 比：

- 它避免了 encoder/decoder 内部的全量 `n^2`、`m^2` 自注意力
- 但仍然保留了 RNN 的顺序依赖，训练并行性更差

所以它在历史上正好是一个中间形态：

- 表达力和对齐能力比纯 fixed-length seq2seq 强
- 并行性和统一性又不如 Transformer

## 常数因子与工程现实

虽然加 attention 后理论上只多了一层对齐，但工程上要注意几件事：

- decoder 每一步都要对整个源序列打分，长源句时开销会明显抬升
- 训练时如果 batch 内句长差异大，padding 区域最好配合 mask 处理
- attention 权重容易被误解成“可解释性证明”，但它更多是一个读取分布，不等于完整因果解释

例如：

- `m = 64`
- `n = 128`

那么一次前向就要做大约 `64 x 128 = 8192` 个目标-源位置匹配。
这个量不算夸张，但如果再叠加大 hidden_dim、大 batch 和更长句子，成本会上升得很快。

## Alternatives and Tradeoffs

### 1. Fixed-Length Seq2Seq vs Attention-Based Seq2Seq

- fixed-length：实现最简单，但长句子容易丢信息
- attention-based：每步都能读源序列，翻译和对齐更稳
- 代价：多了逐步对齐计算和额外参数

### 2. Additive Attention vs Dot-Product Attention

- additive attention：更符合早期 RNN seq2seq 语境，教学直观
- dot-product attention：矩阵化更自然，也更接近 Transformer
- 代价：要开始引入 Q/K/V 视角

这篇先选 additive attention，是为了把“当前步对齐打分”讲清楚。
等你把它看懂，再切换到 dot-product attention 会顺很多。

### 3. Attention-Based Seq2Seq vs Transformer

- 前者：attention 是增强模块，RNN 仍是主干
- 后者：attention 是主干，RNN 被整体拿掉
- 前者更像“修补 fixed-length bottleneck”
- 后者更像“重写整套信息流”

## 迁移路径

把这篇和前后两篇连起来读，会是一条完整链路：

1. [Seq2Seq 与 Encoder-Decoder：从翻译任务到最小可运行 PyTorch 实现](./seq2seq-encoder-decoder-pytorch-minimal.md)
   先理解固定长度上下文和 teacher forcing
2. 当前这篇
   理解 attention 怎样把“目标读源”写进模型
3. [Transformer 结构推导：一步一步搭出最小可运行 PyTorch 实现](./transformer-architecture-overview.md)
   理解为什么 attention 最后会升级成统一主干

如果再往后读，下一篇最自然的是：

- [Self-Attention vs Cross-Attention：机制、差异与工程应用](../attention/self-attention-vs-cross-attention.md)

## 常见问题与注意事项

### 1. attention-based seq2seq 已经是 Transformer 吗？

不是。

它和 Transformer 最接近的部分是“目标读源”的 cross-attention 思想。
但它的 encoder 和 decoder 主干仍然是 RNN，而不是 self-attention block。

### 2. 为什么这篇还保留 GRU？

因为如果一上来把 RNN 全拿掉，你就看不清“attention 到底修了哪个具体问题”。
保留 GRU，反而更容易看到 attention 是如何作为“第一处结构突破”出现的。

### 3. 为什么这里不展开 mask？

因为这篇的重心是“目标读源”的对齐，不是目标侧自回归约束。
mask 会在 Transformer 里变成更中心的问题。

### 4. attention 权重是不是一定能解释模型为什么输出这个词？

不能绝对这么理解。
它能告诉你模型“主要从哪里读取信息”，但不等于完整解释了内部决策。

### 5. 这份代码为什么还不算工业级？

因为它还没有加入：

- source padding mask
- beam search
- scheduled sampling
- 多层 encoder / decoder
- 双向 encoder
- 更稳定的训练细节

但这些都不是理解“从 attention-based seq2seq 过渡到 Transformer”这条主线的前提。

## 最佳实践与建议

- 先看懂 `context_t` 是“每一步单独算出来的”，再去看 Transformer 的 cross-attention
- 调试时优先打印 `attn_weights` 的形状和逐步分布，而不是先盯 loss
- 如果你发现所有目标步都看同一两个源位置，先检查 attention score 是否写错
- 学 Transformer 前，最好先能用一句话说清“attention-based seq2seq 到底比 fixed-length seq2seq 多了什么”

## 小结 / 结论

attention-based seq2seq 真正重要的，不是“它在 RNN 外面又加了一层东西”，而是它把一个关键能力显式写进了模型：

1. decoder 每一步都可以重新决定自己要看源序列的哪里
2. 源序列不再只以一个固定长度向量存在
3. “目标读源”的对齐关系第一次成为模型里的可计算对象
4. 这条思路继续推广，就会自然长成 Transformer 的 cross-attention

更进一步说，Transformer 不是凭空跳出来替换 RNN 的。
它更像是沿着 attention-based seq2seq 已经打开的方向，把 attention 从“外挂模块”升级成了“整套架构的统一主干”。

## 参考与延伸阅读

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Seq2Seq 与 Encoder-Decoder：从翻译任务到最小可运行 PyTorch 实现](./seq2seq-encoder-decoder-pytorch-minimal.md)
- [Transformer 结构推导：一步一步搭出最小可运行 PyTorch 实现](./transformer-architecture-overview.md)
