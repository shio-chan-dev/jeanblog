---
title: "FlashAttention 的 MQA/GQA：共享 KV 的等价、收益与实现要点（含可运行验证）"
subtitle: "把 GQA/MQA 从“概念”落到“实现”：head→KV 映射、KV tile 复用与常见踩坑"
date: 2026-01-25T12:51:15+08:00
summary: "解释 FlashAttention 在 MQA/GQA 下如何利用共享 KV：从数学等价（复制 KV）到工程收益（KV cache 与带宽），并给出可运行代码验证。"
description: "解释 FlashAttention 如何处理 MQA/GQA：共享 KV、按组计算与内存复用策略，并附可运行示例验证等价性。"
categories: ["AI", "Multimodal"]
tags: ["flash-attention", "mqa", "gqa", "attention", "kv-cache", "gpu", "inference"]
keywords: ["FlashAttention", "MQA", "GQA", "KV cache", "Grouped Query Attention", "memory bandwidth", "tiling"]
readingTime: 15
draft: false
---

## 副标题 / 摘要

> MQA/GQA 通过减少 K/V 头数来降低 KV cache 与访存，但注意力实现也必须跟着改变：  
> **Q 头数（Hq）不变，K/V 头数（Hkv）减少，并通过 head→KV 的映射关系共享 K/V**。  
> 本文用“数学等价 + 访存模型 + FlashAttention 的分块复用”把这件事讲透，并附可运行示例验证输出等价。

- **预计阅读时长**：约 15 分钟  
- **标签**：`flash-attention`、`mqa`、`gqa`、`kv-cache`、`inference`  
- **SEO 关键词**：FlashAttention, MQA, GQA, KV cache, Grouped Query Attention  
- **元描述**：FlashAttention 在 MQA/GQA 下如何共享 KV：映射等价、带宽收益与实现要点，附可运行验证。  

---

## 目标读者

- 想把 MQA/GQA 从论文概念落到代码实现的工程读者
- 关注 KV cache、带宽瓶颈、推理吞吐的优化者
- 需要在自研 kernel / 推理引擎中正确处理 GQA/MQA 的开发者

---

## 背景 / 动机（为什么“共享 KV”值得你关心）

在大模型推理（尤其是 **decode：每步生成 1 个 token**）里，最常见的瓶颈不是算力，而是 **读 KV cache 的带宽**。
如果你有：

- 序列长度 `T = 8192`
- head dim `D = 128`
- Q 头数 `Hq = 32`
- 数据类型 fp16（2 bytes）

那么 KV cache 体积（只算 K+V，不算其他）约为：

$$
\text{KV bytes} \approx 2 \times H_{kv} \times T \times D \times 2
$$

- **MHA（Hkv=Hq=32）**：约 `2*32*8192*128*2 ≈ 128 MB`
- **GQA（Hkv=8）**：约 `32 MB`（4× 更小）
- **MQA（Hkv=1）**：约 `4 MB`（32× 更小）

这还只是“存储量”。更关键的是：decode 每一步都要把这些 K/V（或其中很大一部分）从显存读进来。
**减少 Hkv 会直接减少带宽压力**，而 FlashAttention 的 fused kernel 能进一步把“读一次 K/V，多头复用”的收益吃满。

---

## 快速掌握地图（60–120 秒）

- 问题形状：`Q: [B, Hq, Tq, D]`，`K/V: [B, Hkv, Tk, D]`，且 `Hkv < Hq`
- 核心一句话：**每个 Q head 选择一个 KV head（kv(h)），并在 kernel 中复用同一份 K/V tile**
- 什么时候用：KV cache/带宽成瓶颈（长上下文、decode、吞吐优先）
- 什么时候慎用：极端 MQA 可能影响质量；或 `Hq % Hkv != 0` 导致实现/对齐复杂
- 复杂度 headline：计算量仍 ~`O(B·Hq·Tq·Tk·D)`；但 K/V 读带宽 ~随 `Hkv` 线性缩小
- 常见坑（一个就能把你搞崩）：把 `kv(h)` 写错，或把 `Hq/Hkv` 当成 `Hkv/Hq`，结果输出直接错误但不一定报错

---

## Deepening Focus（PDKH Ladder：只深挖两件事）

本文只深挖两个核心概念，并贯穿 PDKH：

1) **GQA 的 head→KV 映射与“数学等价”**（你如何确保实现没改数学）
- P：把问题重述成“复制 KV 的等价变换”
- D：用最小例子 `Hq=4, Hkv=2` 走一遍映射
- K：给出不变式：每个 head 的 K/V 只取决于 `kv(h)`
- H：用代码验证：GQA == 把 K/V 复制到每个 head 的 MHA

2) **FlashAttention 为什么能在 GQA 下赚到更多：KV tile 复用的 IO 模型**
- P：把优化目标明确成“减少 global memory 读 K/V 次数”
- D：用 `T=4096, D=128, Hq=32, Hkv=8` 算一遍字节量
- K：给出一个可检查的工程断言：同一 KV tile 被同组 g 个 Q heads 使用
- H：解释 shared memory / register 压力与 tile size 的现实约束

---

## Master Mental Model（你真正利用的结构是什么）

把注意力看成“对每个 head 做一次 softmax 加权求和”，本质上：

- **Q 变**：每个 head 的 Q 不同 → softmax 权重不同
- **K/V 可共享**：在 MQA/GQA 中，一组 Q heads 共享同一份 K/V → 读取是可复用的

因此最关键的工程心智模型是：

> **K/V 是“只读公共素材”，Q heads 是“多个消费者”。**
> 
> 你无法共享 softmax 的结果（因为 Q 不同），但你可以共享 K/V 的“加载”和“缓存”。

FlashAttention 的 tiling/fusion 让你在一个 kernel 内做到：

- 把 K/V 的一个 tile（比如 `Bk × D`）读进 shared memory
- 在这个 tile 还在 shared memory 时，对同组的多个 Q heads 反复使用它
- 避免“每个 Q head 都从显存再读一遍同样的 K/V”

---

## 核心概念与术语（定义 + 形状 + 公式）

### 1) 形状约定（建议你先在代码里统一）

本文统一使用：

- `Q`: `[B, Hq, Tq, D]`
- `K`: `[B, Hkv, Tk, D]`
- `V`: `[B, Hkv, Tk, D]`
- 输出 `O`: `[B, Hq, Tq, D]`

其中：

- `B` batch size
- `Tq/Tk` query/key 的序列长度（自注意力时通常 `Tq=Tk=T`）
- `D` head dim（例如 64/128）
- `Hq` query heads 数
- `Hkv` key/value heads 数

### 2) MQA / GQA 的定义

- **MHA（标准多头）**：`Hkv = Hq`，每个 Q head 都有自己的 K/V
- **MQA（Multi-Query Attention）**：`Hkv = 1`，所有 Q heads 共享同一份 K/V
- **GQA（Grouped-Query Attention）**：`1 < Hkv < Hq`，把 Q heads 分组，每组共享一个 KV head

### 3) head → KV head 的映射（关键公式）

当 `Hq` 能被 `Hkv` 整除时，设组大小：

$$
g = H_q / H_{kv}
$$

那么对任意 Q head `h ∈ [0, Hq)`：

- **MQA**：`kv(h) = 0`
- **GQA**：

$$
kv(h) = \left\lfloor \frac{h}{g} \right\rfloor
$$

这个映射是你所有实现正确性的根：
**只要 `kv(h)` 对了，你就没有把数学改坏。**

---

## Feasibility / Lower Bound 直觉：FlashAttention 没改变什么、改变了什么

### 1) 没改变的：计算下界（精确注意力）

精确注意力要算 `QK^T`，其乘加次数大致随 `Tq*Tk*D` 增长。
在不做近似（稀疏/线性化）的前提下：

- **计算量仍然是二次的**（随 `T` 增长很快）

FlashAttention 的关键不是把 `O(T^2)` 变成 `O(T)`，而是：

- 不把 `QK^T`/softmax 矩阵落地到显存
- 用更好的缓存局部性把访存压力压下去

### 2) 改变的：中间态与带宽

- FlashAttention：把中间态从“显存里的巨大矩阵”变成“寄存器/共享内存里的局部 tile”
- MQA/GQA：把 **K/V 的头数从 Hq 降到 Hkv**，使得 KV cache 的存储与读取量线性下降

这两者叠加：你同时减少了

- “需要读多少 K/V”（由 Hkv 决定）
- “读到之后能不能在 kernel 里重复利用”（由 tiling/fusion 决定）

---

## Problem Framing（你到底在实现什么）

你的实现通常要回答三个问题：

1) **数学上**：每个 head 的注意力定义是什么？（用 `kv(h)` 选 K/V）
2) **数据上**：K/V 的 layout 是什么？（`[B,Hkv,T,D]` 还是 `[B,T,Hkv,D]`）
3) **kernel 上**：在一个 tile 生命周期内，K/V 能被多少个 Q heads 复用？（理想是 g 次）

现实约束（常见但容易被忽略）：

- 很多高性能实现会假设 `Hq % Hkv == 0`（否则分组不均匀、对齐变差）
- `D` 往往要求是 8/16 的倍数（向量化加载）
- `T` 很长时更偏 memory-bound；`T` 很短时收益会缩水

---

## Baseline & Bottleneck（朴素实现为什么慢）

### 朴素 baseline：把 GQA 当成“每个 head 独立算”

数学是对的，但工程上你可能会写出这样的访问模式：

- 对每个 Q head `h`：从显存读一遍 `K[kv(h)]` 和 `V[kv(h)]`

当 `g > 1` 时，这里出现了明显的重复：

- 同组 g 个 head 读的是同一份 K/V
- 但你仍然从显存读了 g 次

### 可量化的瓶颈：KV 读取字节数

以 decode 的极端场景（`Tq=1`）为例，读 K/V 的字节量近似：

$$
\text{bytes per step} \approx 2 \times H_{kv} \times T \times D \times \text{dtype_bytes}
$$

例如 `T=4096, D=128, dtype=fp16(2 bytes)`：

- 每个 KV head 的 K+V 大约 `2 * 4096 * 128 * 2 ≈ 2 MB`
- **MHA（Hkv=32）**：约 `64 MB/step`
- **GQA（Hkv=8）**：约 `16 MB/step`

这就是为什么很多推理引擎里，GQA/MQA 会带来“非常实在”的吞吐提升。

---

## Decode vs Prefill：为什么 GQA 在 decode 更“香”（带数字算账）

很多人第一次看 GQA/MQA 会有疑惑：既然计算量不变，那为什么推理吞吐会涨得这么明显？
关键在于：**decode 的 `Tq` 很小（常见是 1），但 `Tk` 很大（历史上下文）**，于是整段计算更偏向“读 KV”而不是“算矩阵乘”。

### 1) decode（Tq=1）：典型是 memory-bound

设 `T=4096, D=128, Hq=32, dtype=fp16(2 bytes)`。

- 每个 head 需要做一次 `q(1×D) · K(T×D)^T`：大约 `T·D = 524,288` 次乘加
- 输出 `p(1×T) · V(T×D)`：同样大约 `T·D = 524,288` 次乘加

粗略把每次乘加记作 2 FLOPs，则每个 head 的 FLOPs 量级约：

`2 * 2 * T * D ≈ 2.1 MFLOPs/head`，32 heads 约 `67 MFLOPs`。

再看带宽：每个 KV head 的 K+V 大约 `2 MB`（上一节已算）。

- MHA（Hkv=32）：`≈ 64 MB/step`
- GQA（Hkv=8）：`≈ 16 MB/step`

你可以把它理解成“算术强度”（FLOPs/byte）的提升：

- MHA：`67e6 / 64e6 ≈ 1.0 FLOP/byte`
- GQA：`67e6 / 16e6 ≈ 4.2 FLOP/byte`

这就是 decode 下 GQA/MQA 的直观收益来源：**同样的计算量，配上更少的 K/V 读取字节**，更容易把 GPU 从“等显存”里拉出来。

### 2) prefill（Tq=Tk=T）：计算更重，但仍然受益

prefill 时 `Tq≈Tk≈T`，每个 head 的 `QK^T` 是 `T^2·D` 量级。
例如 `T=4096, D=128` 时，单 head 的乘加量级约 `4096^2*128 ≈ 2.1e9`（十亿级），32 heads 更是几十亿到百亿级。

这时系统更可能偏向 compute-bound，但：  
GQA 仍然有价值，因为它会降低：

- KV cache 的存储（影响显存峰值与可批量大小）
- K/V 的 global load 量（尤其当你能在 tile 生命周期内复用给组内 heads）

因此一个务实结论是：

- **想提升 decode 吞吐/省 KV cache**：GQA/MQA 往往是第一优先级
- **想提升 prefill**：FlashAttention 的 tile/fusion 是主力，GQA 是锦上添花

---

## Key Observation（FlashAttention 在 GQA 下的关键转折点）

GQA 给了你一个可利用的结构：

> 同一个 KV head 的 K/V，将被同组的 g 个 Q heads 使用。

FlashAttention 的 tiling 让你把这个结构变成性能：

- **先把 K/V tile 读入 shared memory**（一次）
- **在 tile 还热的时候，对 g 个 Q heads 依次/并行计算**（g 次使用）
- tile 生命周期结束后再换下一块

你会得到一个非常直观的收益上界：

> 如果一份 K/V tile 能被完整复用给 g 个 Q heads，那么 K/V 的 global load 次数理论上可以减少到 1/g。

当然真实 kernel 还要受寄存器/共享内存/warp 排布影响，但这个上界给了你正确的方向感。

---

## Algorithm Steps（工程可落地的分组计算流程）

这里给一个“足够接近真实 kernel”的流程（不依赖具体实现版本）：

1) **定义分组**：`g = Hq / Hkv`，并保证 Q heads 以组为连续维度（`[kv, g]`）
2) **以 KV head 为外层循环粒度**：一次处理一个 `kv`（或一个 kv tile block）
3) **加载 K/V tile**：从显存把 `K[kv]`、`V[kv]` 的一段（长度 `Bk`）读到 shared memory
4) **计算同组的 g 个 Q heads**：
   - 对每个 head：计算局部 `S = Q · K^T / sqrt(D)`
   - 用在线 softmax 更新 `(m, l)` 并累积输出 `O`
5) **滑动到下一段 K/V tile**，直到覆盖全 `Tk`

一个小的形状示意（只看 head 维度）：

```text
Q heads:  [0,1,2,3 | 4,5,6,7 | ...]
KV heads: [0        1        ...]
          ^ group=4 ^ group=4
```

---

## Decision Criteria（什么时候选 MQA / GQA / MHA）

下面给可操作的选择逻辑（不是“唯一正确”，但足够工程化）：

1) **你的瓶颈是 KV cache / 带宽吗？**
- 典型信号：长上下文、decode 吞吐受限、显存紧张
- 如果是：优先考虑 GQA → MQA

2) **你能接受多大质量/可训练成本？**
- MQA（Hkv=1）压得最狠，但更可能影响质量，需要模型/训练策略配合
- GQA（例如 `g=4` 或 `g=8`）通常更平衡

3) **你是否受实现约束？**
- 若推理引擎/内核要求 `Hq % Hkv == 0`，那就别选奇怪的 `Hkv`

一个“先算账再决定”的简单表格（示例）：

| 设定 | Hq | Hkv | g=Hq/Hkv | KV cache 相对 MHA | 备注 |
| --- | ---:| ---:| ---:| ---:| --- |
| MHA | 32 | 32 | 1 | 1× | 质量/实现最简单 |
| GQA | 32 | 8  | 4 | 1/4× | 性能/质量常用折中 |
| MQA | 32 | 1  | 32| 1/32× | 压到极致，需评估质量 |

---

## Worked Example（Trace：最小例子走一遍）

我们用最小但非平凡的例子：

- `Hq=4`（4 个 Q heads）
- `Hkv=2`（2 个 KV heads）
- `g=2`

映射关系：

```text
h:    0  1  2  3
kv(h) 0  0  1  1
```

这意味着：

- head 0 和 head 1 共享 `K[0], V[0]`
- head 2 和 head 3 共享 `K[1], V[1]`

如果你把 `K/V` 复制成 “每个 head 一份”，得到：

```text
K_expanded[0]=K[0], K_expanded[1]=K[0], K_expanded[2]=K[1], K_expanded[3]=K[1]
V_expanded 同理
```

那么 **GQA 的输出应当与用 `K_expanded/V_expanded` 做 MHA 的输出完全一致**。
这就是下面可运行代码要验证的等价。

---

## Correctness（Proof Sketch：为什么复用不会改变结果）

不变式（对每个 head 都成立）：

> 对任意 batch `b`、head `h`、query 位置 `i`，GQA 的注意力只使用 `K[b, kv(h), :, :]` 与 `V[b, kv(h), :, :]`。

因此如果定义“复制后的”张量：

- `K_expanded[b, h, :, :] = K[b, kv(h), :, :]`
- `V_expanded[b, h, :, :] = V[b, kv(h), :, :]`

那么对每个 head 的注意力计算式完全相同：

$$
O[b,h] = \text{softmax}(Q[b,h]K[b,kv(h)]^T / \sqrt{D}) \; V[b,kv(h)]
$$

换句话说：

- **GQA/MQA 改的是参数/缓存的共享方式**
- **FlashAttention 改的是计算顺序与中间态的落地方式**

二者只要不改变 `kv(h)` 的选择关系，就不会改变数学结果（只影响速度与数值误差的微小差异）。

---

## Complexity（算量 vs 带宽）

### 时间复杂度（乘加次数）

精确注意力的主项不变：

- `O(B · Hq · Tq · Tk · D)`

GQA/MQA 不会神奇地减少这个乘加次数（每个 Q head 仍要和全部 K 做点积）。

### 空间与访存（关键收益点）

- KV cache 存储：`O(B · Hkv · Tk · D)`（由 Hkv 线性决定）
- K/V 读取带宽：理想情况下也随 `Hkv` 线性下降

如果 kernel 能把同一 KV tile 在组内复用 g 次，K/V 的 global load 次数会进一步按 1/g 摊薄。

---

## Constant Factors & Engineering Realities（为什么“tile 复用”有现实约束）

FlashAttention 在 GPU 上的关键是 shared memory/寄存器的预算。
给一个非常具体的锚点：

- 假设 `Bk=128, D=128, dtype=fp16(2 bytes)`
- 一个 K tile 大小：`128*128*2 ≈ 32 KB`
- 一个 V tile 大小：同样 `≈ 32 KB`
- K+V 合计：`≈ 64 KB`

这意味着：

- 如果你想同时把 K 和 V tile 放进 shared memory，tile 不能无限大
- 如果再叠加“同时算多个 Q heads”（更高复用），寄存器压力会上升，可能降低 occupancy

工程上常见的权衡：

- tile 大：更少的 loop 次数，但更吃 shared memory（可能挤掉并发）
- tile 小：更容易并发，但 loop 次数更多（指令/调度开销上升）

这也是为什么不同版本/不同实现的 FlashAttention 会在 tile 大小、head 并行度上做不同取舍。

---

## 可运行实现（Python / Numpy）：验证 GQA/MQA 的数学等价

下面的代码做两件事：

1) 实现一个“参考版 MHA”（每个 head 都有自己的 K/V）
2) 实现 GQA/MQA（`K/V` 只有 `Hkv` 个 heads，用 `kv(h)` 共享）

并验证：

- 把 `K/V` 复制成 `K_expanded/V_expanded` 后，参考 MHA 输出 == GQA 输出

```python
import numpy as np


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mha_reference(q, k, v):
    """Reference multi-head attention.

    q: [B, Hq, Tq, D]
    k/v: [B, Hq, Tk, D]
    out: [B, Hq, Tq, D]
    """
    b, hq, tq, d = q.shape
    _, hk, tk, _ = k.shape
    assert hk == hq

    out = np.zeros((b, hq, tq, d), dtype=q.dtype)
    scale = 1.0 / np.sqrt(d)

    for bi in range(b):
        for h in range(hq):
            scores = (q[bi, h] @ k[bi, h].T) * scale  # [Tq, Tk]
            p = softmax_stable(scores, axis=-1)
            out[bi, h] = p @ v[bi, h]

    return out


def gqa_mqa_attention(q, k, v):
    """Grouped/Multi-Query attention.

    q: [B, Hq, Tq, D]
    k/v: [B, Hkv, Tk, D]

    Requirement: Hq % Hkv == 0
    """
    b, hq, tq, d = q.shape
    _, hkv, tk, _ = k.shape
    assert v.shape == (b, hkv, tk, d)
    if hq % hkv != 0:
        raise ValueError(f"Hq % Hkv must be 0, got Hq={hq}, Hkv={hkv}")

    g = hq // hkv  # group size
    out = np.zeros((b, hq, tq, d), dtype=q.dtype)
    scale = 1.0 / np.sqrt(d)

    for bi in range(b):
        for h in range(hq):
            kv = h // g
            scores = (q[bi, h] @ k[bi, kv].T) * scale
            p = softmax_stable(scores, axis=-1)
            out[bi, h] = p @ v[bi, kv]

    return out


def expand_kv_for_reference(k, v, hq: int):
    """Expand [B, Hkv, T, D] to [B, Hq, T, D] by repeating heads."""
    b, hkv, t, d = k.shape
    if hq % hkv != 0:
        raise ValueError("Hq % Hkv must be 0")

    g = hq // hkv
    k_exp = np.repeat(k, repeats=g, axis=1)
    v_exp = np.repeat(v, repeats=g, axis=1)
    return k_exp, v_exp


if __name__ == "__main__":
    np.random.seed(0)

    # Minimal non-trivial example: Hq=4, Hkv=2 -> group=2
    B, Hq, Hkv, Tq, Tk, D = 1, 4, 2, 3, 3, 4

    q = np.random.randn(B, Hq, Tq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, Tk, D).astype(np.float32)
    v = np.random.randn(B, Hkv, Tk, D).astype(np.float32)

    out_gqa = gqa_mqa_attention(q, k, v)

    k_exp, v_exp = expand_kv_for_reference(k, v, hq=Hq)
    out_ref = mha_reference(q, k_exp, v_exp)

    diff = np.max(np.abs(out_ref - out_gqa))
    print("max_abs_diff=", diff)
    print("out shape=", out_gqa.shape)

    # MQA case: Hkv=1
    k_mqa = k[:, :1]
    v_mqa = v[:, :1]
    out_mqa = gqa_mqa_attention(q, k_mqa, v_mqa)
    k_exp2, v_exp2 = expand_kv_for_reference(k_mqa, v_mqa, hq=Hq)
    out_ref2 = mha_reference(q, k_exp2, v_exp2)
    print("mqa max_abs_diff=", np.max(np.abs(out_mqa - out_ref2)))
```

你预期看到的结果：`max_abs_diff` 接近 0（浮点误差范围内）。

补充说明（非常重要）：

- 上面这个示例里，“参考 MHA”与“GQA/MQA”使用的是**同一种** softmax 与矩阵乘顺序，所以差异会非常小。
- 真实的 FlashAttention kernel 为了性能会改变归约顺序、使用 block-wise 累加、以及混合精度（例如用 fp16/bf16 输入、fp32 累加）。这会带来**数值上**的小偏差：  
  - 常见量级：`1e-4 ~ 1e-3`（取决于 D、T、数据分布与实现）  
  - 这通常是“数值等价”（numerically close），而不是“逐 bit 相等”。

如果你在做实现验收，建议用三步把问题收敛：

1) 用 fp32 的 reference（或更高精度）做对照  
2) 同时看 `max_abs_diff` 和相对误差（避免被尺度误导）  
3) 用极端输入做稳定性测试（例如 logits 很大时是否溢出）
---

## E — Engineering（工程应用：3 个真实场景）

### 场景 1：推理服务的 KV cache 预算（先算账再动手）

**背景**：你要把上下文从 4k 拉到 16k，但显存不够。  
**为什么适用**：GQA/MQA 直接线性减少 KV cache。  

```python
# Quick estimator: KV cache size in MB

def kv_cache_mb(T: int, D: int, Hkv: int, dtype_bytes: int = 2) -> float:
    return (2 * Hkv * T * D * dtype_bytes) / (1024 * 1024)

print("MHA  (Hkv=32):", kv_cache_mb(T=8192, D=128, Hkv=32), "MB")
print("GQA  (Hkv= 8):", kv_cache_mb(T=8192, D=128, Hkv=8), "MB")
print("MQA  (Hkv= 1):", kv_cache_mb(T=8192, D=128, Hkv=1), "MB")
```

### 场景 2：自研/改造 kernel 时的“复用机会”判断

**背景**：你想让一个 KV tile 被同组多个 Q heads 使用。  
**为什么适用**：GQA 的组结构提供了天然的复用单位。  

```python
# Example: map each Q head to KV head
Hq, Hkv = 32, 8
assert Hq % Hkv == 0

g = Hq // Hkv
kv_map = [h // g for h in range(Hq)]
print("group size=", g)
print("head->kv (first 16)=", kv_map[:16])
```

### 场景 3：线上排错：内网路由式的“静态断言”

**背景**：你怀疑实现把 `kv(h)` 搞错了，但模型还能跑，只是效果异常。  
**为什么适用**：GQA/MQA 最容易出现“形状对、语义错”。  

建议你加一个 cheap 的断言（开发/测试环境）：

```python
# For GQA: heads in the same group must map to same KV head.
Hq, Hkv = 32, 8
assert Hq % Hkv == 0

g = Hq // Hkv
for kv in range(Hkv):
    heads = list(range(kv * g, (kv + 1) * g))
    assert len(set([h // g for h in heads])) == 1
```

---

## Alternatives & Tradeoffs（对比与取舍：别只看“省显存”）

| 方案 | KV cache | 典型收益 | 典型代价/风险 |
| --- | ---:| --- | --- |
| MHA | 1× | 质量/表达力最好，兼容性最好 | KV cache 大，decode 容易被带宽卡死 |
| GQA | 1/g× | 显存/带宽明显下降，通常更稳 | 组太大可能影响质量；实现需正确映射 |
| MQA | 1/Hq× | KV cache 极小，吞吐潜力最大 | 更可能损失质量，需训练/结构配合 |
| 近似注意力（稀疏/线性） | 取决于方法 | 可把 `T^2` 变小 | 这是“换算法”，会改变数学与质量 |

一个务实的结论：

- 你只想解决“长上下文推理带宽/显存” → **优先 GQA**
- 你被显存逼到墙上、愿意为吞吐牺牲一定质量 → 再考虑 MQA

---

## Migration Path（学会这一篇之后，下一步学什么）

- 想更懂 FlashAttention：继续看 **online softmax 的数值稳定性** 与 **block scheduling**
- 想更懂推理引擎：看 **KV cache layout**、**PagedAttention/分页缓存**、以及连续批处理下的 cache 管理
- 想更懂模型结构：看不同模型为何选择 GQA/MQA（训练稳定性、质量、吞吐的平衡）

---

## 常见坑与边界情况（带失败样例）

1) **`Hq % Hkv != 0`（分组无法整除）**  
失败样例：`Hq=32, Hkv=6`，此时 `g = Hq/Hkv` 不是整数，很多高性能实现会直接不支持（或性能很差）。

工程上通常有三种处理方式（按推荐顺序）：

- **方案 A：把 Hkv 调整为 Hq 的因子**（最推荐）  
  例如 `Hq=32` 时，常见可选 `Hkv∈{1,2,4,8,16,32}`。  
  如果你原本想要 `Hkv=6`，往往会落到 `Hkv=8 (g=4)` 或 `Hkv=4 (g=8)` 这种“可整除且更好对齐”的配置。

- **方案 B：padding 到可整除**（能跑，但要算清楚代价）  
  例如把 `Hkv=6` padding 到 `Hkv=8`，相当于“多出 2 个 KV heads 的存储与带宽”。  
  这类 padding 在训练/推理上都要保证：多出来的 heads 不会引入语义错误（通常意味着你得显式处理权重/缓存）。

- **方案 C：不等组映射（显式 kv_map）**（最不推荐）  
  你可以人为指定 `kv(h)`，让有的 KV head 对应 5 个 Q heads、有的对应 6 个 Q heads。数学上可行，但会破坏很多 kernel 的假设：  
  - group 不均匀 → warp 排布/向量化加载更难做  
  - 复用粒度不稳定 → 性能更难预期  

一句话：**如果你追求性能，优先把 head 配置选成“整除 + 对齐友好”。**

2) **K/V 的 layout 不连续导致性能崩**  
你数学没错，但 K/V 在内存里跳跃访问，tile 加载无法合并，吞吐会很差。

3) **“共享 KV”≠“共享 softmax”**  
Q 不同，softmax 权重不同；你只能共享 K/V 的加载，不能共享注意力权重。

4) **精度/质量回归只看 perplexity 不够**  
GQA/MQA 的影响往往是“能力边界”变化（长文本一致性、检索、指令遵循），要做有代表性的评测。

---

## 最佳实践与建议

- 先用代码验证等价：GQA == 复制 KV 后的 MHA（本文代码就是模板）
- 优先选择整除的 head 配置：让 `Hq % Hkv == 0`
- 关注 decode 场景：`Tq=1` 时 KV 带宽是最直观的收益来源
- 若你在写 kernel：用“KV head 作为外层粒度”，最大化 tile 复用；同时关注 shared memory/寄存器预算

---

## Summary / Takeaways（至少 4 条可执行收获）

- GQA/MQA 的实现核心是 `kv(h)`：只要映射对，数学就对；错了输出会“静悄悄地错”。
- GQA/MQA **不减少乘加次数**，主要减少的是 **KV cache/带宽**（随 Hkv 线性下降）。
- FlashAttention 的 tiling/fusion 能把“共享 KV”的优势放大：K/V tile 读一次，在组内复用 g 次。
- 工程落地要同时看：`Hq%Hkv`、K/V layout 连续性、以及 shared memory/寄存器带来的 tile 限制。

---

## 参考与延伸阅读

- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2305.13245

---

## 元信息

- **阅读时长**：约 15 分钟
- **标签**：flash-attention、mqa、gqa、kv-cache、gpu
- **SEO 关键词**：FlashAttention, MQA, GQA, KV cache, Grouped Query Attention
- **元描述**：FlashAttention 在 MQA/GQA 下如何共享 KV：映射等价、带宽收益与实现要点，附可运行验证。

---

## 行动号召（CTA）

如果你愿意贴一下你模型的 `Hq/Hkv/T/D`（不含任何业务信息），我可以帮你：

- 估算 KV cache 体积与带宽压力
- 给出更贴近你配置的组大小建议
- 指出最可能踩坑的 layout/整除问题
