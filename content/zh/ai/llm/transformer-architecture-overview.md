---
title: "Transformer 结构推导：一步一步搭出最小可运行 PyTorch 实现"
date: 2026-04-23T10:13:23+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["transformer", "attention", "encoder-decoder", "self-attention", "pytorch"]
---

> **副标题 / 摘要**  
> 这篇文章不把 Transformer 当成一个现成黑盒来介绍，而是直接从一个最小翻译任务开始，让需要的结构一层一层长出来，最后收束成一份最小可运行的 PyTorch encoder-decoder Transformer。

## 从一个最小翻译任务开始

假设源序列是：

```text
<bos> I love apples <eos>
```

目标序列是：

```text
<bos> 我 喜欢 苹果
```

当模型要生成“苹果”时：

- 它不能看目标序列里未来还没生成的位置
- 它需要重点读取源序列中的 `apples`
- 它可能还需要参考前面的“我 喜欢”来决定当前词

所以这里天然会逼出三件事：

1. 目标侧必须有**因果约束**
2. 源侧和目标侧都需要**全局读取**
3. 解码器不仅要读自己，还要读编码器输出

RNN 和 CNN 也能处理序列，但它们在长距离依赖和全并行训练上都有明显限制。  
所以这里真正要解决的，不只是“做一个更深的网络”，而是让任意位置能直接交互，并且显式控制信息流方向。

下面开始按这个压力一步一步长代码。

## Step 1：先有输入表示，但先不谈注意力

第一步只解决一个问题：

> token id 怎么变成模型能计算的向量？

这里第一次引入 **embedding** 这个词：

- 它的作用不是“理解上下文”
- 它只是把离散 id 映射成长度为 `d_model` 的连续向量

最小骨架如下：

```python
import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self.scale
```

这一版现在能做到：

- 把 `[B, T]` 的 token id 变成 `[B, T, d_model]` 的向量

但它还缺：

- 顺序信息
- 上下文交互

因为这时第 `i` 个位置和第 `j` 个位置还是彼此独立的。

## Step 2：输入有内容了，但还没有顺序，所以补位置编码

如果只有 embedding，那么 `[A, B, C]` 和 `[C, B, A]` 只是同一组向量的重排。  
模型没有天然机制知道谁在前、谁在后。

所以这里才需要 **positional encoding**：

- token embedding 负责“这个词是什么”
- position encoding 负责“它在第几个位置”

所以我们在上一版基础上新增一个位置编码模块：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
```

接回当前输入表示后：

```python
x = token_embedding(token_ids)
x = positional_encoding(x)
```

这一版现在能做到：

- 每个位置既有词内容，也有顺序信息

但它还缺：

- 一个位置读取其他位置的能力

也就是说，现在模型知道“我是第几个 token”，但还不会问“我应该看谁”。

## Step 3：让一个位置读取其他位置，先长出单头 self-attention

接下来只解决一个问题：

> 一个 token 怎样根据相关性，去读取整条序列里的其他 token？

这里第一次引入 **self-attention** 这个词。  
做法其实并不神秘：先把输入投影成三组向量，再用相似度决定“我应该看谁”。

- `Q` 是 query，表示“我现在想找什么信息”
- `K` 是 key，表示“我这里有什么信息可供匹配”
- `V` 是 value，表示“如果你决定看我，真正拿走的内容是什么”

先写最小单头注意力：

```python
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(x.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v
```

这里最关键的形状变化是：

- 输入 `x`：`[B, T, d_model]`
- `scores = QK^T`：`[B, T, T]`
- `weights @ V` 后输出仍然是：`[B, T, d_model]`

这意味着：

- 第 `i` 个位置可以读取整条序列的所有位置
- 每一行注意力权重和为 `1`

把上面这三步收束成标准写法，就是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里的 `d_k` 是 key 的维度。  
除以 `sqrt(d_k)` 的作用，先记住是“控制分数尺度”，后面在常见问题里再回头解释。

这一版现在能做到：

- 全局上下文聚合

但它还缺：

- 解码器的“不能看未来”约束
- 一个 head 不足以同时表示多种关系
- 注意力之后的非线性变换

## Step 4：解码器不能偷看未来，所以给 attention 加 mask

如果我们直接把上面的 self-attention 用在解码器里，训练时第 `t` 个位置就能看到 `t+1` 之后的 token。  
这和自回归生成的目标冲突。

所以这里第一次引入 **causal mask**：

- 它不是新模块
- 它只是对注意力分数加上的一个约束，告诉模型哪些位置不允许看

所以在上一版基础上，只新增一个 `mask` 参数：

```python
def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    scores = q @ k.transpose(-2, -1) / math.sqrt(x.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
```

因果 mask 的最小形式是一个下三角矩阵：

```python
causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
```

例如当 `T = 4` 时：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这一版现在能做到：

- 编码器里做全局 self-attention
- 解码器里做带方向约束的 self-attention

但它还缺：

- 一个 head 同时表达多种关系的能力

## Step 5：一个头不够，所以扩展成 multi-head attention

只用一个 head，模型只能在一个子空间里计算相关性。  
但真实序列里，模型可能同时想看：

- 语法依赖
- 语义对齐
- 位置模式

所以我们把 `d_model` 拆成多个 head，各自独立计算注意力，再拼回来。

这一步不要只看成“多做几次 attention”。  
真正的代码变化是：把 Step 3 的单头版本整体替换成一个多头版本。

先看最关键的变形：

```python
q = self.q_proj(query).view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
k = self.k_proj(key).view(batch_size, k_len, num_heads, head_dim).transpose(1, 2)
v = self.v_proj(value).view(batch_size, k_len, num_heads, head_dim).transpose(1, 2)
```

然后每个 head 独立算：

- `scores` 形状是 `[B, h, q_len, k_len]`
- 输出重新拼回 `[B, q_len, d_model]`

把 Step 3 的 `SingleHeadSelfAttention` 替换成下面这个类，当前版本就真正升级成了 multi-head attention：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_len, _ = query.shape
        k_len = key.size(1)

        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out), attn
```

这一版现在能做到：

- 在多个子空间并行建模相关性

但它还缺：

- 注意力之后更强的逐位置非线性变换
- 更稳定的深层堆叠结构

## Step 6：注意力只是加权混合，所以加上 FFN、残差和归一化

注意力本质上是在做“根据权重混合别人的值向量”。  
如果只有这一层，表达力还不够。

所以我们再补三样东西：

1. **FFN**：逐位置的非线性变换
2. **Residual**：保留原输入路径
3. **LayerNorm**：让深层训练更稳

这里的 **FFN, feed-forward network** 可以先理解成：

- 注意力负责“去哪里拿信息”
- FFN 负责“拿回来以后怎么再加工一遍”

最小前馈网络是：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

接着把它接进 encoder block。  
这一步不要只定义成员变量，而是把信息流真正接起来：

- 先做 self-attention
- 再做残差 + 归一化
- 再做 FFN
- 再做一次残差 + 归一化

在上一版代码基础上，新增下面这个完整的 `EncoderBlock`：

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```

这一版现在能做到：

- 得到一个完整的 encoder 层

但它还缺：

- 目标侧自己的 masked self-attention
- 目标侧读取编码器输出的 cross-attention

## Step 7：光有 encoder 不够，decoder 还要“边看自己边看源序列”

解码器块和编码器块最大的差别是多了一层 **cross-attention**。

这里第一次引入这个词：

- self-attention 是“读自己这条序列”
- cross-attention 是“用目标侧当前状态去读源侧 memory”

这里要非常明确三件事：

- decoder self-attention：`query = key = value = decoder 当前状态`
- cross-attention：`query = decoder 当前状态`
- cross-attention：`key = value = encoder memory`

所以 decoder block 会比 encoder block 多一个注意力子层。  
这一步同样不要只停在构造器，要把完整 `forward` 也长出来：

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
```

这一版现在能做到：

- 目标侧先看历史，再看源序列

到这里，所有关键机制都齐了。  
下一步不再继续贴碎片，而是把前面的模块真正收束成一份最小完整实现。

## 最后一段：把前面的模块接成一份最小可运行 Transformer

下面这份代码包含：

- token embedding
- sinusoidal positional encoding
- multi-head self-attention
- encoder block
- decoder block
- causal mask
- 最终词表投影

它不是工业级实现，但已经是一个真正可运行的最小 encoder-decoder Transformer。

```python
import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_len, _ = query.shape
        k_len = key.size(1)

        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        return (src_ids != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = tgt_ids.shape
        padding_mask = (tgt_ids != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt_ids.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        return padding_mask & causal_mask

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src_ids)
        tgt_mask = self.make_tgt_mask(tgt_ids)
        memory_mask = src_mask

        src = self.dropout(self.pos_encoding(self.src_embed(src_ids)))
        tgt = self.dropout(self.pos_encoding(self.tgt_embed(tgt_ids)))

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        out = tgt
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, memory_mask)

        return self.output_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)

    src_ids = torch.tensor([
        [1, 5, 7, 9, 2, 0],
        [1, 8, 4, 2, 0, 0],
    ])
    tgt_input_ids = torch.tensor([
        [1, 6, 3, 2],
        [1, 9, 2, 0],
    ])

    model = Transformer(
        src_vocab_size=32,
        tgt_vocab_size=32,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.1,
        max_len=32,
        pad_id=0,
    )

    logits = model(src_ids, tgt_input_ids)
    print(logits.shape)  # torch.Size([2, 4, 32])
```

## 这份完整代码现在到底在做什么

如果用上面的 demo 输入：

- `src_ids` 形状是 `[2, 6]`
- `tgt_input_ids` 形状是 `[2, 4]`

那么前向过程的关键形状是：

1. embedding + position 后：
   - 源序列：`[2, 6, 32]`
   - 目标序列：`[2, 4, 32]`
2. encoder 输出 `memory`：
   - `[2, 6, 32]`
3. decoder 输出：
   - `[2, 4, 32]`
4. 最终词表投影 `logits`：
   - `[2, 4, 32]`

这意味着：

- 每个目标位置都会产出一个长度为 `32` 的词表打分向量
- 训练时可以把 `tgt_input_ids` 右移作为输入，把真实目标作为 label

## 解释与原理

到这里，Transformer 的结构已经不是一堆名词，而是一条连续的构建链：

1. **Embedding** 解决“token id 不能直接算”的问题
2. **Positional Encoding** 解决“模型不知道顺序”的问题
3. **Self-Attention** 解决“一个位置不会读取其他位置”的问题
4. **Mask** 解决“解码器会偷看未来”的问题
5. **Multi-Head** 解决“一个相似度空间不够用”的问题
6. **FFN + Residual + LayerNorm** 解决“只有加权混合、深层不稳定”的问题
7. **Cross-Attention** 解决“目标侧还不会读取源序列”的问题

这就是为什么它最后会长成 encoder-decoder 的样子，而不是作者事先拍脑袋定出来的结构。

## 正确性与不变量

这份最小实现有几个重要不变量：

1. 注意力权重在最后一维做 softmax，所以每个 query 对所有 key 的权重和为 `1`
2. causal mask 保证第 `t` 个目标位置永远看不到 `t` 之后的位置
3. encoder 和 decoder 的隐藏维度始终保持 `d_model`，所以模块之间可以直接级联
4. cross-attention 中，decoder 提供 query，encoder memory 提供 key/value，这正对应“目标读源”的信息流方向

## 复杂度与代价

设：

- 源序列长度是 `n`
- 目标序列长度是 `m`
- 隐藏维度是 `d`

那么一层里的主要代价是：

- encoder self-attention：`O(n^2 d)`
- decoder self-attention：`O(m^2 d)`
- decoder cross-attention：`O(m n d)`
- FFN：`O((n + m) d d_ff)`

最容易爆掉的不是参数量，而是注意力矩阵。

例如：

- 当 `n = 2048` 时，一个 head 的注意力矩阵就有 `2048 x 2048 ≈ 420 万` 个元素
- 如果 `h = 8`，只存注意力权重就已经很可观

所以 Transformer 的核心代价是序列长度平方，而不是“层数看起来很多”。

## 常见问题与注意事项

### 1. 为什么要除以 `sqrt(d_k)`？

如果不缩放，`QK^T` 的数值会随着维度变大而变大。  
当 `d_k = 64` 时，点积方差会明显抬升，softmax 更容易过尖，梯度更不稳定。

### 2. 为什么位置编码要加在 embedding 上？

因为注意力本身对输入顺序不敏感。  
如果没有位置编码，模型看到的是一组 token 向量，而不是一个有先后关系的序列。

### 3. 为什么 decoder 需要两层 attention？

因为它要解决的是两个不同问题：

- 先看自己已经生成的历史
- 再看源序列里哪些位置最相关

这两种读取目标不同，所以拆成 masked self-attention 和 cross-attention 更自然。

### 4. 这份代码为什么还不算工业级？

因为它还没有加入：

- label smoothing
- weight tying
- KV cache
- mixed precision
- flash attention
- 更复杂的位置编码，如 RoPE

但这些都是在核心结构已经成立之后的增强，而不是理解 Transformer 的前提。

## 最佳实践与建议

- 第一次手写 Transformer 时，把 `d_model` 设小一点，比如 `32` 或 `64`，先把张量形状跑通
- 调试时优先打印 `src_mask`、`tgt_mask` 和注意力分数形状，很多 bug 都出在这里
- 如果 forward 能跑通，再去加训练循环；不要一开始就把训练、数据集、调参全堆进来
- 如果你只做语言建模，可以在这个最小实现基础上继续删成 decoder-only，而不是反过来先学大而全的工业实现

## 小结 / 结论

Transformer 最值得学的，不是“它由 encoder 和 decoder 组成”这句话，而是它为什么会一步一步长成这个样子：

- 先解决输入表示
- 再解决顺序
- 再解决全局交互
- 再解决信息流约束
- 再解决表达力和深层稳定性
- 最后才自然得到 encoder-decoder 结构

当你能从这些压力反推出代码，Transformer 就不再是一个需要死记硬背的架构图，而是一个你自己能重新搭出来的模型。

## 参考与延伸阅读

- Attention Is All You Need
- The Annotated Transformer
- [Attention Is All You Need：Transformer 的核心算法与工程落地](../attention/attention-is-all-you-need.md)
- [Self-Attention 计算公式与 Softmax 数值稳定：从推导到工程实现](../attention/self-attention-softmax-formula-and-stability.md)
