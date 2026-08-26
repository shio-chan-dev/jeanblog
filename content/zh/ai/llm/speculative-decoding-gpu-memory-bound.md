---
title: "Speculative Decoding 为什么能加速：从并行验证到 GPU Memory-Bound"
date: 2026-08-26T00:00:00+08:00
draft: false
categories: ["AI", "LLM"]
tags: ["speculative decoding", "LLM inference", "GPU", "memory bandwidth", "vLLM"]
description: "从自回归生成的串行依赖出发，解释 speculative decoding 如何用一次 Target forward 并行验证多个 Draft token，以及 acceptance rate、batch 和 KV Cache 如何决定实际收益。"
keywords: ["Speculative Decoding", "推测解码", "LLM 推理", "GPU Memory-Bound", "Causal Mask", "KV Cache", "vLLM"]
---

> **副标题 / 摘要**
> Speculative decoding 并不是让大模型先生成一遍，再与小模型逐个比较；它让小模型先提供未来 token 的候选路径，使大模型能在一次 forward 中并行验证多个位置。它的本质，是用更多、甚至部分无效的并行计算，换取更少的串行 Target forward。

- **预计阅读时长**：12~16 分钟
- **目标读者**：了解 Transformer 基础，希望理解 LLM 推理性能、vLLM 或 speculative decoding 的读者
- **核心问题**：为什么一次验证多个 token 会比逐个生成更快？如果中间猜错，后面的计算是否全浪费了？

## 先给结论

理解 speculative decoding，只需要抓住四件事：

1. 自回归生成慢在串行依赖：不知道前一个 token，就无法确定下一个 token 的条件分布。
2. Draft 模型先猜出一条候选路径后，Target 模型就能在一次 forward 中并行计算这条路径上多个位置的条件分布。
3. 第一个未被接受的 token 之后，验证结果确实全部作废；这是用并行冗余换串行轮次的代价。
4. 加速是否成立，取决于接受长度、Draft 成本、Target verify 成本、batch 大小和 KV Cache 流量，而不是只看一次 draft 了多少个 token。

一句话概括：

> Draft 模型最重要的作用，不只是“帮大模型猜答案”，而是暂时补上未知的未来输入，让 Target 能跨过原本必须串行执行的多个解码位置。

## 普通自回归生成为什么必须串行

假设当前上下文是：

```text
I love
```

Target 模型准备生成：

```text
New York City very much
```

普通 decode 只能这样执行：

```text
输入: I love
forward #1 -> New

输入: I love New
forward #2 -> York

输入: I love New York
forward #3 -> City

输入: I love New York City
forward #4 -> very

输入: I love New York City very
forward #5 -> much
```

之所以需要 5 次，不是因为 Transformer 一次只能计算一个位置，而是因为生成第一个 token 之前，后面的输入还不存在：

```text
New -> York -> City -> very -> much
       ^       ^       ^       ^
       每一步都依赖前一步的结果
```

训练或 prefill 时，整段输入已经给定，所以所有位置可以一起计算。推理时，未来 token 尚未产生，这才形成了自回归的串行瓶颈。

## Draft 如何让 Target 一次验证多个位置

现在加入一个便宜得多的 Draft 模型。它先猜出：

```text
New -> York -> City -> very -> much
```

Target 此时不需要自己从零逐个生成这 5 个候选，而是一次性评估下面这些条件分布：

| 待验证 token | Target 使用的条件 |
| --- | --- |
| `New` | `I love` |
| `York` | `I love New` |
| `City` | `I love New York` |
| `very` | `I love New York City` |
| `much` | `I love New York City very` |

这些条件中的候选 token 已由 Draft 给出，因此 Target 可以把它们组织成一次类似短 prefill 的 forward：

```text
Draft:   New   York   City   very   much
           |      |      |      |      |
           v      v      v      v      v
        +--------------------------------+
        |        one Target forward      |
        +--------------------------------+
           |      |      |      |      |
           v      v      v      v      v
Target:  p1     p2     p3     p4     p5
```

这里的 `p1...p5` 不是五个彼此无条件的答案，而分别表示：

$$
p(d_1\mid x),\quad
p(d_2\mid x,d_1),\quad
\ldots,\quad
p(d_K\mid x,d_{<K})
$$

其中 `x` 是已有上下文，`d_1...d_K` 是 Draft 候选。

causal language model 的 logits 与 token 之间存在一个标准的一位偏移。把完整序列展开后，实际对应关系是：

| logits 来源位置 | 该位置可见的内容 | 用来验证 |
| --- | --- | --- |
| `love` | `I love` | `New` |
| `New` | `I love New` | `York` |
| `York` | `I love New York` | `City` |
| `City` | `I love New York City` | `very` |
| `very` | `I love New York City very` | `much` |

使用 KV Cache 时，边界处的 logits 可能来自已缓存前缀或由运行时专门对齐。具体张量组织因实现而异，但不影响关键事实：验证这些候选所需的条件分布可以在一次 Target 验证操作中得到。

## Causal mask 不让看未来，为什么还能并行

这看起来像一个矛盾：causal mask 明明规定不能看未来，为什么 GPU 又能一次算出多个位置？

答案是：

> causal mask 限制的是每个位置可以读取哪些信息，不是要求 GPU 必须按位置顺序执行。

候选序列已经存在后，用来验证每个 token 的条件前缀都是确定的：

```text
New   <- I love
York  <- I love New
City  <- I love New York
very  <- I love New York City
much  <- I love New York City very
```

对完整输入做 forward 时，输入位置之间的因果注意力结构仍是下三角。例如下面每一行表示该输入位置能读取哪些输入 token：

```text
        love  New  York  City  very  much
love      1    0     0     0     0     0
New       1    1     0     0     0     0
York      1    1     1     0     0     0
City      1    1     1     1     0     0
very      1    1     1     1     1     0
much      1    1     1     1     1     1
```

每一层 Transformer 都可以对所有位置并行执行矩阵运算，只是在 attention 中把不允许读取的位置屏蔽掉。层与层之间仍然串行，但同一层内的 token 位置可以并行。

因此，并行验证没有破坏因果关系。它只是基于 Draft 给出的假设路径，同时求值多个合法的条件概率。

## 如果中间猜错，后面的计算确实白做了

假设 Draft 猜测：

```text
New -> York -> City -> is -> big
```

而 Target 验证得到：

```text
New    York    City    is    big
 |      |       |      |      |
 ok     ok      ok    reject  invalid
```

`is` 是第一个未被接受的位置，那么：

- `New York City` 可以接受。
- `is` 被拒绝。
- `big` 的计算作废，因为它基于错误前缀 `New York City is`。

即使 `big` 碰巧等于 Target 在正确路径上的某个预测，也不能使用。自回归概率依赖完整前缀：

$$
p(\text{big}\mid x,\text{New York City is})
$$

与

$$
p(\text{big}\mid x,\text{New York City very})
$$

不是同一个条件分布。

所以答案很直接：**首个错误之后的并行计算就是浪费了。**

但这不等于整轮毫无进展。以 greedy decoding 为例，Target 在验证时已经算出了错误位置应该选择的 token，因此通常会：

1. 接受错误之前的 Draft 前缀。
2. 在错误位置追加 Target 自己的纠正 token。
3. 从纠正后的新前缀开始下一轮 Draft。

也就是说，若第 4 个候选错误，这轮通常不只是前进 3 个 token，而是“3 个已接受 Draft token + 1 个 Target 纠正 token”。如果全部候选都通过，很多实现还能追加一个 Target 已顺带算出的 bonus token。

## “接受”不总是比较两个 token 是否相等

为了建立直觉，可以先把 greedy decoding 理解为：

```text
Draft token == Target argmax token -> accept
```

但启用随机采样后，简单比较 token 会改变 Target 原本的输出分布。经典 speculative sampling 使用 Draft 分布 `q` 和 Target 分布 `p` 做概率接受：

$$
P(\text{accept } d_i)=\min\left(1,\frac{p(d_i)}{q(d_i)}\right)
$$

若拒绝，则从经过校正的分布中采样。这样做的目标是：在数学上保持与直接从 Target 采样相同的输出分布。

因此更准确的说法不是“Target 和 Draft 对答案”，而是：

> Target 在 Draft 给出的候选路径上一次计算多个概率，并按解码策略决定最长可接受前缀。

## 为什么多算几个位置，反而可能更快

现在进入硬件核心。

Transformer 中大量计算可以简化成线性层：

$$
Y=XW
$$

假设权重矩阵：

```text
W: [4096, 4096]
```

FP16 下约占：

$$
4096\times4096\times2\text{ bytes}\approx32\text{ MiB}
$$

### 单 token decode

一次只处理一个 token 时：

```text
X: [1, 4096]
W: [4096, 4096]

[1, 4096] x [4096, 4096] -> [1, 4096]
```

为了完成这一小行计算，GPU 需要从显存层级读取大量权重。矩阵很“瘦”，计算单元未必能被充分利用，瓶颈往往更接近显存带宽而不是峰值 FLOPs。

这就是常说的 decode 容易 **memory-bound**：

```text
HBM 搬权重 -> GPU 很快算完 -> 等下一批数据
```

### 一次验证多个 token

若一次验证 5 个候选：

```text
X: [5, 4096]
W: [4096, 4096]

[5, 4096] x [4096, 4096] -> [5, 4096]
```

权重规模没有变，但同一批权重数据可以在矩阵乘中服务更多 token 行。于是权重读取成本被摊薄，GPU 完成了更多有用计算。

这可以用 **Arithmetic Intensity（算术强度）** 粗略表达：

$$
\text{Arithmetic Intensity}=\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

只看这个 FP16 线性层并忽略激活等流量：

- `m=1` 个 token 时，权重约 32 MiB，计算约 `2H^2` FLOPs。
- `m=5` 个 token 时，权重仍约 32 MiB，计算约 `10H^2` FLOPs。
- token 行数增加后，每搬运一份权重产生了更多计算，算术强度随之提高。

“一次读取权重”是便于理解的近似说法。真实 GPU 有分块、缓存、量化、并行策略和不同算子，不能据此认为 verify 成本恒定，也不能认为它一定已经 compute-bound。更严谨的结论是：**多 token verify 通常比单 token decode 更容易摊薄权重流量并提高硬件利用率。**

## Prefill、Decode 与 Verify 的关系

三者可以放在同一张表里理解：

| 阶段 | 一次处理的 token 位置 | 典型特征 |
| --- | ---: | --- |
| Prefill | 整段 prompt，可能上千个 | 大型矩阵乘，通常更容易利用 GPU 计算能力 |
| Decode | 每条请求通常 1 个新位置 | 瘦矩阵乘，低 batch 时容易受权重和 KV 读取限制 |
| Verify | 每条请求的多个候选位置 | 类似一个很短的 prefill，用额外计算换更少的 Target 串行轮次 |

所以 speculative decoding 并没有神奇地消除 Target 的数学运算。它改变的是运算的组织方式：

```text
多个串行、低利用率的 decode step
                |
                v
一个更宽、更容易并行的 verify step
```

## 为什么 acceptance rate 决定收益

假设每轮 Draft 提议 `K=5` 个 token。

| 最长接受前缀 | 后续作废位置 | 直观效果 |
| ---: | ---: | --- |
| 5 | 0 | 最理想，一次 Target verify 跨过多个位置 |
| 4 | 1 | 通常仍有明显收益 |
| 3 | 2 | 是否划算取决于 verify 和 Draft 成本 |
| 1 | 4 | 串行轮次减少很少，收益可能很小 |
| 0 | 5 | 几乎只靠 Target 纠正 token 前进一步 |

如果把每个候选在前缀正确的条件下被接受的概率粗略记作 `alpha`，并假设各位置近似一致，那么一轮产生的 token 数期望可简化为：

$$
E[\text{tokens per round}]=1+\alpha+\alpha^2+\cdots+\alpha^K
$$

这里的 `1` 对应首个拒绝位置的 Target 纠正 token，或全部接受后的 bonus token。这个公式只是帮助建立直觉，真实系统中的接受事件并不独立，成本也不会只由 `K` 决定。

下面这段代码可以快速观察 `K` 和接受率的关系：

```python
def expected_tokens_per_round(k: int, acceptance_rate: float) -> float:
    return sum(acceptance_rate**i for i in range(k + 1))


for rate in (0.3, 0.6, 0.8, 0.95):
    expected = expected_tokens_per_round(k=5, acceptance_rate=rate)
    print(f"acceptance={rate:.0%}, expected tokens/round={expected:.2f}")
```

输出：

```text
acceptance=30%, expected tokens/round=1.43
acceptance=60%, expected tokens/round=2.38
acceptance=80%, expected tokens/round=3.69
acceptance=95%, expected tokens/round=5.30
```

这也解释了为什么 `K` 不是越大越好。`K` 增大时，理论上一次能跨过更多位置，但 Draft 成本、verify 计算量和首错之后的无效计算也一起增加。

## 为什么低并发可能加速，高并发却可能降吞吐

到目前为止，我们默认只有一条请求。但 vLLM 一类服务框架通常会做 continuous batching。

假设有 100 条请求同时 decode：

```text
request 1   -> 1 token
request 2   -> 1 token
...
request 100 -> 1 token
```

对线性层来说，GPU 看到的已经近似是：

```text
X: [100, H]
```

也就是说，即使不开 speculative decoding，权重也能同时服务很多 token 行，单 token decode 的低利用率已经被 batch 部分缓解。

若每条请求再验证 5 个候选，工作量可能扩展到近似：

```text
100 requests x 5 positions = 500 token positions
```

这时 speculative decoding 带来的额外位置可能让系统从 memory-bound 逐渐转向 compute-bound，其中一部分还是首错后的无效计算。结果可能是：

- 单请求的 TPOT 降低，用户看到 token 更快。
- 单次请求需要的 Target 调度轮数减少。
- 但高并发下的总 tokens/s 不再提高，甚至下降。
- 单位有效 token 的 GPU 成本可能上升。

因此“speculative decoding 能加速两倍”不是一个完整结论。必须先问：

- 优化目标是单请求延迟，还是集群最大吞吐？
- 当前 batch 多大？
- 平均接受长度是多少？
- Draft 自身占用了多少算力和显存？
- 验证出来的 token 中，有多少最终被保留？

## Decode 不只搬权重，还要读 KV Cache

真实 LLM decode 的内存流量至少有两个主要来源：

```text
                 Decode memory traffic
                         |
              +----------+----------+
              |                     |
              v                     v
        Model weights           KV Cache
        各层线性变换        Attention 读取历史 K/V
```

KV Cache 保存每一层历史 token 的 Key 和 Value。生成新 token 时，只需要计算新位置的 Query、Key、Value，再让新 Query 与历史 Key/Value 做 attention；否则每一步都要重新计算整个历史前缀。

若忽略 batch，KV Cache 大小可粗略写成：

$$
2\times L\times T\times n_{kv}\times d_{head}\times\text{bytes per element}
$$

其中：

- `2` 表示 Key 和 Value。
- `L` 是 Transformer 层数。
- `T` 是上下文长度。
- `n_kv` 是 KV head 数量。
- `d_head` 是每个 head 的维度。

因此 KV Cache 容量随上下文长度线性增长，而每个新 query 读取历史 K/V 的流量也会随上下文变长。长上下文下，只用“模型权重扫了几遍”解释 decode 性能就不够了。

PagedAttention 主要解决 KV Cache 的分配、碎片和共享问题，使服务端能更灵活地管理不同长度请求；它不会自动消除 attention 对历史 K/V 的读取，也不会改变标准 attention 的全部计算复杂度。

## 应该用什么指标判断是否值得开启

实际测试时，至少同时记录这些指标：

| 指标 | 回答的问题 |
| --- | --- |
| TTFT | 用户多久看到第一个 token？ |
| TPOT / inter-token latency | 后续 token 出现得有多快？ |
| End-to-end latency | 一条请求总共多久完成？ |
| Output tokens/s | 整个服务的有效输出吞吐是多少？ |
| Acceptance rate | Draft 候选有多少被接受？ |
| Mean accepted length | 每次 verify 平均跨过多少 Draft 位置？ |
| Draft overhead | Draft 消耗了多少时间、显存和计算？ |
| Goodput | 满足延迟目标的有效请求吞吐是多少？ |

只看 Target forward 次数不够，只看 GPU 利用率也不够。GPU 利用率变高，可能是在计算更多最终会被丢弃的位置；真正有意义的是单位时间、单位成本产生了多少满足服务目标的有效 token。

## 常见误解

### 1. Target 也自回归生成 K 个 token，再与 Draft 比较

不是。这样仍然需要 K 次昂贵的 Target forward，失去了 speculative decoding 的核心收益。

### 2. 一次 forward 得到多个位置，说明模型看到了未来

不是。候选 token 虽然作为已知输入送入，但 causal mask 仍保证每个位置只能使用合法前缀。

### 3. 第一个候选错了，这轮完全没有输出

通常不是。Target 已经算出该位置的分布，可以在这里产生纠正 token；只是这一轮没有跨过多少串行步骤。

### 4. Verify 处理 5 个 token，成本就等于 5 次 decode

通常不等价。多位置矩阵运算可以摊薄权重读取并提高 GPU 利用率，但 verify 也绝不是免费操作，实际比例取决于硬件、模型、batch、上下文和算子实现。

### 5. Acceptance rate 越高，系统一定越快

不一定。还要计入 Draft 成本、verify 宽度、调度开销，以及高并发时 Target 是否已经接近 compute-bound。

## 把整条逻辑串起来

```text
自回归生成存在串行依赖
          |
          v
Draft 先给出一条候选未来
          |
          v
Target 用 causal mask 一次并行求值多个条件分布
          |
          v
接受最长合法前缀，在首错位置纠正
          |
          +----------------------+
          |                      |
          v                      v
接受得多                  首错后的计算作废
减少 Target 串行轮次       增加无效 FLOPs
          |                      |
          +----------+-----------+
                     v
       是否加速取决于硬件与服务负载
       acceptance / K / batch / KV / Draft cost
```

这就是 speculative decoding 的核心取舍：

> 用更多并行计算，换取更少串行等待；只要并行冗余的代价，小于省下来的 Target 解码轮次，它就值得。

## 小结

- 普通 decode 必须等前一个 token 产生，因此天然串行。
- Draft 候选补上了未知输入，让 Target 可以一次验证多个位置。
- causal mask 约束信息可见性，但不禁止同一层内的位置并行。
- 首个错误之后的验证计算确实无效，这是 speculative decoding 的主要代价之一。
- 低 batch decode 常受显存带宽限制，多位置 verify 能提高算术强度并摊薄权重流量。
- 高并发时普通 decode 已经形成较大 batch，speculative decoding 的吞吐收益可能缩小甚至转负。
- KV Cache 是另一项关键内存流量，长上下文下不能忽略。
- 最终应以 TPOT、有效 tokens/s、平均接受长度和 goodput 做真实负载测试，而不是只看理论 `K` 值。

## 参考与延伸阅读

- Yaniv Leviathan, Matan Kalman, Yossi Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Charlie Chen et al. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- Woosuk Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM Documentation. [Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
