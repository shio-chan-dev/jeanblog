---
title: "Softmax 工程实现与 GPU 访存优化：在线更新、融合与带宽算账（含可运行验证）"
subtitle: "把 softmax 当作 IO 问题：读几遍、写几遍、何时必须两遍，何时可以一遍（fusion）"
date: 2026-01-25T12:51:13+08:00
summary: "从标准两遍 softmax 的访存模式出发，推导在线 softmax（m,l）更新与正确性；进一步解释在 attention/cross-entropy 中如何通过融合避免落地概率矩阵，并用可运行代码验证等价与估算带宽收益。"
description: "拆解 softmax 标准计算的访存问题，并给出在线 softmax 与融合实现的工程优化思路，包含可运行示例与带宽算账。"
categories: ["AI", "Multimodal"]
tags: ["softmax", "gpu", "memory", "performance", "attention", "online-softmax", "kernel-fusion", "logsumexp"]
keywords: ["Softmax", "GPU", "访存优化", "Online Softmax", "Kernel Fusion", "LogSumExp", "Attention"]
readingTime: 16
draft: false
---

## 副标题 / 摘要

softmax 的公式很短，但 GPU 上跑得慢往往不是因为算不动 exp，而是因为**读写内存的次数太多**。
这篇文章把 softmax 当成一个“IO + 归约（reduction）”问题来拆：

- **标准稳定 softmax**为什么天然是“两遍”（至少两次读输入）
- **在线 softmax**如何用一个不变式维护 `(m, l)`，把“数值稳定 + 归约”做成可组合的 streaming 更新
- 当 softmax 的输出**不需要被显式保存**（attention 的 `P@V`、交叉熵的 `logsumexp`）时，为什么可以通过 **kernel fusion** 避免写回概率矩阵，从而把带宽压力降一个数量级

文末给出可运行的 Numpy 代码：

1) 验证在线 softmax 的更新正确性（含最小 trace）
2) 验证“融合版”（不落地 softmax 概率）与“朴素版”（先 softmax 再乘 V / 再算 loss）数值一致
3) 给出一个可复制的带宽算账函数，帮助你判断优化是否值得做

---

## 目标读者

- 想理解 softmax 在 GPU 上“慢在哪里”的工程读者
- 关注训练/推理吞吐、带宽瓶颈、算子融合（fusion）的优化者
- 需要实现或排查 attention / cross-entropy 融合算子的开发者

---

## 背景 / 动机（先把账算清楚）

GPU 上 softmax 常见的性能事实是：**softmax 很容易变成 memory-bound**。
原因不复杂：softmax 是“按行归一化”的操作，包含至少两个归约（max 与 sum），并且要写出每个元素的输出。

先给一个带数字的锚点：

- 行长度 `N = 4096`
- dtype = fp16（2 bytes）

如果你要输出整行 softmax 概率，那么无论如何你都至少要：

- 读 `N` 个输入（`~ 8 KB`）
- 写 `N` 个输出（`~ 8 KB`）

这只是理论下界。
现实里“稳定 softmax”还要做 max/sum 两个归约，很多实现会导致：

- **至少两次读输入**（第一次求 max/sum，第二次写输出）
- 如果你把 `exp(x-m)` 暂存到全局内存再归一化，甚至会有额外的写回与再读取

而在 attention 里，softmax 的输入/输出规模更大：

- 输入是 `S = QK^T`，形状常见是 `[B, H, Tq, Tk]`
- 输出 `P` 同形状（概率矩阵）
- 若你把 `P` 落地到显存，写回的字节量是 `B*H*Tq*Tk*dtype_bytes`，这是最恐怖的一项

这也是 FlashAttention 的出发点：**不要写回 `P`**。

---

## 快速掌握地图（60–120 秒）

- 问题形状：对矩阵 `X ∈ R^{M×N}` 做 row-wise softmax（每行独立归一化）
- 核心一句话：
  - 要输出 softmax 向量：通常需要 **2-pass**（或 1-pass + 存临时）
  - 不需要输出向量：通过 **fusion** 把 softmax 融到后续归约里，可避免写回 `P`
- 什么时候收益最大：`N` 大、`M` 大、dtype 小（fp16/bf16），且算子链长（attention / xent）
- 常见踩坑：mask/全 -inf 行导致 `l=0`，在线更新若不做保护会出 NaN
- 复杂度 headline：
  - 计算量：每行 `O(N)`（softmax 本身）
  - IO：朴素实现常见 `2×read + 1×write`（输出 softmax），融合可把“写回概率矩阵”变成 0

---

## Deepening Focus（PDKH：只深挖两件事）

本文只深挖两个核心概念：

1) **在线 softmax 的更新（m,l）与正确性**
- P：把问题重述为“维护一个随流更新的 logsumexp 归约”
- D：用最小向量 `[3, 1, -2, 5]` 逐步更新 `(m,l)`
- K：给出不变式：`l = Σ exp(x_i - m)`（对已扫描前缀）
- H：给出可运行代码验证输出等价

2) **融合 softmax：不落地概率矩阵 P 的 IO 模型**
- P：把目标重述为“避免写回/再读取中间态（P 或 exp）”
- D：用 attention 的 `O=P@V` 举最小例子，说明“只要 O，不要 P”
- K：给出不变式：在在线 softmax 更新时同时维护 `o = Σ exp(x_i - m) v_i`
- H：用代码验证“融合版 vs 朴素版”的数值一致，并给出字节算账

---

## Master Mental Model（把 softmax 看成两个归约 + 一个归一化）

稳定 softmax 的等价形式是：

$$
\operatorname{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},\quad m = \max_j x_j
$$

你可以把它拆成三步：

1) **归约 1：max**（得到 `m`，防溢出）
2) **归约 2：sumexp**（得到 `l = Σ exp(x-m)`）
3) **逐元素归一化**（输出 `exp(x-m)/l`）

GPU 优化的核心问题是：

- 这三步要读几遍输入？
- 是否要把中间态写回显存？（写回一次就可能让你直接 memory-bound）
- 能不能把 softmax 的“输出需求”改写成更容易融合的形式？

---

## 核心概念与术语

### 1) logsumexp（本质上就是 softmax 的“分母”）

$$
\operatorname{LSE}(x) = \log \sum_j \exp(x_j)
$$

稳定写法：

$$
\operatorname{LSE}(x) = m + \log \sum_j \exp(x_j - m),\quad m=\max_j x_j
$$

### 2) 在线 softmax 的状态变量（m, l）

- `m`：已扫描元素的最大值
- `l`：已扫描元素在“以 m 为基准”下的指数和（sumexp）

关键是：当 `m` 更新时，`l` 必须做重标定（rescale），否则会错。

---

## Feasibility & Lower Bound：为什么“输出 softmax 向量”很难一遍搞定

这里给一个非常工程化的结论：

- **如果你必须输出完整的 softmax 向量**（每个元素都要写出去），那么要想只读一遍输入，你通常需要把 `exp(x-m)` 暂存下来。
- 暂存的代价就是：你把“第二次读输入”变成了“额外写/读一个临时缓冲区”。

这就是 softmax 的一个典型 IO 不可能三角：

1) 只读一遍输入
2) 不用额外临时存储
3) 输出完整 softmax 向量

三者通常不能同时满足。

所以 GPU 上常见的做法是：

- **2-pass softmax**：第一次得到 `m,l`，第二次再读输入写输出
- 或者 **1-pass + 临时**：第一次读输入写 `exp(x-m)` 到临时，第二次读临时做归一化

当你把 softmax 融合进后续归约（例如 `P@V`、cross-entropy）时，情况就变了：

- 你不需要输出 `P`（概率向量/矩阵）
- 你只需要一个更小的输出（例如 attention 的 `O`，shape `[Tq,D]`）

这时 fusion 才能把 IO 压下去。

---

## Problem Framing（attention 里 softmax 的输入/输出规模）

在 attention 里，softmax 的输入是 score：

$$
S = \frac{QK^T}{\sqrt{D}},\quad S\in\mathbb{R}^{T_q\times T_k}
$$

朴素 attention 的输出是：

$$
P = \operatorname{softmax}(S)\in\mathbb{R}^{T_q\times T_k},\quad O = PV\in\mathbb{R}^{T_q\times D}
$$

关键点：

- `P` 的元素个数是 `Tq*Tk`，而 `O` 的元素个数是 `Tq*D`
- 在长上下文里 `Tk` 可能远大于 `D`

因此：

> 如果你把 `P` 写回显存，你是在写一个比最终输出大得多的中间态。

这就是 FlashAttention/融合 softmax 的“必然性”。

---

## Baseline & Bottleneck（朴素实现的 IO 长什么样）

### Baseline A：输出完整 softmax（典型 2-pass）

每行长度为 `N`，输出 `N` 个概率：

- Pass 1：读 `x` → 求 `m` 与 `l`
- Pass 2：再读 `x` → 写 `y = exp(x-m)/l`

IO 近似：

- read：`2N`
- write：`N`

### Baseline B：attention 朴素写回 P（最贵的一项）

如果你在 attention 中显式构造 `P`：

- 你要写 `Tq*Tk` 的概率矩阵
- 后续算 `O=PV` 时，还要再读 `P`

对长上下文来说，这一步几乎注定把你拖进 memory-bound。

---

## Key Observation 1：在线 softmax 把“稳定性 + 归约”变成可组合的更新

在线 softmax 的核心是维护 `(m,l)`，并保证一个不变式：

> 扫描到第 i 个元素后，`m` 是前缀最大值，且 `l = Σ_{j≤i} exp(x_j - m)`。

当加入一个新元素 `x` 时：

- 新最大值：`m' = max(m, x)`
- 旧的 `l` 需要按新的基准 `m'` 重标定：`l * exp(m - m')`

因此更新式为：

$$
\begin{aligned}
m' &= \max(m, x)\\
l' &= l\cdot\exp(m-m') + \exp(x-m')
\end{aligned}
$$

这组更新式有两个非常工程化的意义：

1) 它是 **streaming** 的：你可以按块扫描一行（tile），不断合并
2) 它是 **可并行归约** 的：每个线程/warp 可先算局部 `(m,l)`，再做合并（类似“分治”）

---

## Worked Example（Trace：用最小例子走一遍）

向量：`x = [3, 1, -2, 5]`

我们从 `m=-inf, l=0` 开始，逐个更新：

| step | x | m（更新后） | l（更新后） | 解释 |
| ---:| ---:| ---:| ---:| --- |
| 1 | 3 | 3 | 1 | `exp(3-3)=1` |
| 2 | 1 | 3 | `1 + exp(1-3)=1+e^{-2}` | m 不变，累加 |
| 3 | -2 | 3 | `1+e^{-2}+e^{-5}` | m 不变，累加 |
| 4 | 5 | 5 | `(旧l)*exp(3-5) + exp(5-5)` | m 变大，先重标定再加 1 |

最后输出：

$$
\operatorname{softmax}(x)_i = \exp(x_i - m) / l
$$

这个 trace 是你验收实现的第一把尺子：如果你写的更新在 step=4（m 变大）时没有 rescale，输出一定错。

---

## Correctness（Proof Sketch：为什么更新式是对的）

不变式：处理完前缀集合 `A` 后，

$$
\begin{aligned}
 m &= \max_{j\in A} x_j\\
 l &= \sum_{j\in A}\exp(x_j - m)
\end{aligned}
$$

加入新元素 `x`，令 `m'=max(m,x)`。

- 若 `m' = m`：显然 `l' = l + exp(x-m)`
- 若 `m' = x`：旧项要从基准 `m` 迁移到基准 `x`，即：

$$
\sum_{j\in A}\exp(x_j - x) = \sum_{j\in A}\exp(x_j - m)\cdot\exp(m-x) = l\cdot\exp(m-m')
$$

再加上新元素 `exp(x-m') = 1`，得到更新式。

---

## Block-wise 合并：把一行拆成多个 tile 还能保持数值稳定吗？

GPU 上几乎不可能“一个线程负责整行”。真实 kernel 会把一行拆成多个块（tile/chunk）：

- 每个线程/warp 先处理自己那一段，得到局部状态 `(m, l)`
- 再把这些局部状态合并成全局 `(m, l)`

关键在于：**合并也必须遵守同一个不变式**。

### 1) 合并两个局部状态（m, l）

假设你把一行分成两段 `A` 与 `B`，分别计算得到：

- `m_A = max(A)`，`l_A = Σ_{i∈A} exp(x_i - m_A)`
- `m_B = max(B)`，`l_B = Σ_{i∈B} exp(x_i - m_B)`

把它们合并成全局状态的正确公式是：

$$
\\begin{aligned}
m &= \\max(m_A, m_B)\\\\
l &= l_A\\cdot\\exp(m_A-m) + l_B\\cdot\\exp(m_B-m)
\\end{aligned}
$$

这就是“重标定（rescale）”在并行归约里的版本：谁的 `m` 更小，谁就要先乘一个 `exp(m_small - m_big)` 把基准抬到同一个 `m` 上。

### 2) 最小数值例子（两段合并）

还是用 `x=[3,1,-2,5]`，分两段：

- `A=[3,1]`：`m_A=3`，`l_A=exp(3-3)+exp(1-3)=1+e^{-2}≈1.1353`
- `B=[-2,5]`：`m_B=5`，`l_B=exp(-2-5)+exp(5-5)=e^{-7}+1≈1.0009`

合并时 `m=max(3,5)=5`：

`l = 1.1353*e^{-2} + 1.0009 ≈ 0.1536 + 1.0009 ≈ 1.1545`

而全量扫描的 `l = e^{-2}+e^{-4}+e^{-7}+1 ≈ 1.1545`，一致。

### 3) 融合版（m,l,o）同样可合并

在融合 attention 时，你还会维护 `o = Σ exp(x_i-m) v_i`。两段合并同理：

$$
o = o_A\\cdot\\exp(m_A-m) + o_B\\cdot\\exp(m_B-m)
$$

这条式子是“fusion 能并行化”的关键：每个线程块先算局部 `(m,l,o)`，再归约合并。

### 4) 一个典型错误（能跑但结果错）

如果你直接做 `l = l_A + l_B`（不做 rescale），当 `m_A != m_B` 时结果必错。

工程上建议你把“分块合并”也写成一个单元测试：先分块算，再合并，必须等价于全量扫描（本文代码部分也给了示例）。
---

## Key Observation 2：融合 softmax（不落地 P）才是注意力里真正的 IO 杀手锏

如果你的目标不是输出 softmax 向量，而是输出一个“softmax 加权后的结果”，例如：

$$
O = \operatorname{softmax}(S)\,V
$$

那么你可以边扫描 `S` 的列（keys），边更新 `(m,l)`，同时维护一个向量累加器 `o`：

$$
 o = \sum_j \exp(S_j - m)\,V_j
$$

当 `m` 更新时，`o` 也要做同样的 rescale：

$$
\begin{aligned}
 m' &= \max(m, s)\\
 l' &= l\cdot\exp(m-m') + \exp(s-m')\\
 o' &= o\cdot\exp(m-m') + \exp(s-m')\,v
\end{aligned}
$$

最终：

$$
 O = o / l
$$

这就是 FlashAttention 的核心“在线融合”结构：

- 你从来不需要把 `P` 写回显存
- 你只维护 `(m,l,o)` 这三个小状态（每个 query 一份）

---

## Decision Criteria（怎么选：2-pass softmax vs fusion）

1) **你是否需要输出完整 softmax 概率？**

- 需要（例如要喂给后续非融合算子、要做 top-k/采样、要做可解释性可视化）：
  - 现实里基本绕不开 2-pass（或 1-pass + 临时）
- 不需要（例如最终只要 `P@V`、只要 `logsumexp`、只要 loss）：
  - 优先考虑 fusion

2) **N 的大小与 shared memory 预算**

- `N` 小（例如 <= 1024）：有机会把一整行（或大部分）放到 shared memory / registers，在一次 global read 内完成更多工作
- `N` 大（例如 4096/8192/16384）：通常要按块扫描，2-pass 输出概率更常见，但 fusion 仍然能避免写回大中间态

3) **数值稳定性要求**

- fp16/bf16 输入时，务必用 fp32 累加 `l` 与 `o`（否则非常容易 NaN 或严重误差）

---

## 实践指南 / 步骤（从需求到可验收实现）

你可以把 softmax 优化当作一个非常可执行的流程：

1) **先明确“我需要输出什么”**  
   - 需要输出完整概率（`y=softmax(x)`）：走 2-pass 稳定 softmax（或 1-pass+临时）  
   - 不需要概率，只需要下游结果（`softmax(x)@v`、cross-entropy）：优先 fusion

2) **写出 IO 账本（读几遍、写几遍）**  
   - 2-pass softmax：`2×read(x) + 1×write(y)`  
   - attention 朴素：`write(P) + read(P)` 这两项往往是最大的额外开销  
   - fusion：目标是把 `P` 的写回/读取变成 0

3) **实现时抓住三个“稳定性硬约束”**  
   - `m`（max）必须用 fp32 维护（fp16 很容易溢出/精度不够）  
   - `l`（sumexp）建议 fp32 累加（尤其是 `Tk` 很大时）  
   - mask 行要有 `l==0` 的保护（全 mask 行很常见）

4) **把验收写成单元测试（强烈建议）**  
   - 最小 trace：`[3,1,-2,5]`，检查 m 变大时是否正确 rescale（本文 Worked Example）  
   - 分块合并：把一行拆两段算 `(m,l)` 再 merge，必须等价于全量扫描（本文 Block-wise 合并）  
   - 数值容忍：GPU kernel 常见 `1e-4~1e-3` 的误差量级（归约顺序/混合精度导致），不要用“逐 bit 相等”验收

这套流程的好处是：你不会把问题留到“跑起来不对再猜”，而是从一开始就把正确性与性能都写进验收标准里。

---

## Complexity（别只写 O(n)，把 IO 写出来）

对每行长度 `N`：

- 计算量：`O(N)`（exp + add）
- IO：
  - 输出 softmax：典型 `2×read(x) + 1×write(y)`
  - 融合 `softmax(x)·v`：典型 `1×read(x) + 1×read(v) + 1×write(o)`，且 **不写回 y**

在 attention 中，关键差异是：

- `y`（概率矩阵）是 `Tq*Tk`
- `o`（输出）是 `Tq*D`

当 `Tk >> D` 时，避免写 `y` 的收益非常大。

---

## Constant Factors & Engineering Realities（GPU 上真正决定速度的细节）

这里列一些“决定你是不是能跑到峰值”的现实约束（每条都尽量给一个可操作锚点）：

1) **归约（max/sum）要在 warp/block 内完成**  
避免全局原子加（atomic add）去累加 `l`，那会直接把你拖进序列化。

2) **exp 的实现不是主要瓶颈，但数值稳定是**  
工程上通常做：输入 fp16/bf16，内部用 fp32 计算 `m,l,o`。

3) **tile 大小受 shared memory 限制**  
例如 `Bk=128, dtype=fp16, D=128`，一个 tile 大小约 `32 KB`；
K/V 各一份就是 `~64 KB`，再加上其他临时变量，很容易顶到一个 SM 的 shared memory 上限。

4) **融合会增加寄存器压力，可能降低 occupancy**  
fusion 不是“总是更快”，它的 tradeoff 是：更少的 global memory IO vs 更高的寄存器/共享内存占用。

---

## 可运行实现（Python / Numpy）：在线 softmax + 融合验证 + 带宽算账

下面的代码分三部分：

- Part A：在线 softmax（m,l）与稳定 softmax 对比
- Part B：融合版 `softmax(scores) @ values`（不落地概率）与朴素版对比
- Part C：带宽算账（估算 bytes），帮助你判断“写回 P”到底多贵

```python
import numpy as np


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def online_softmax_1d(x: np.ndarray):
    """Online softmax for 1D vector.

    Returns:
      m: max
      l: sumexp(x-m)
      p: softmax(x)
    """
    m = -np.inf
    l = 0.0
    for xi in x:
        m_new = max(m, float(xi))
        l = l * np.exp(m - m_new) + np.exp(float(xi) - m_new)
        m = m_new
    p = np.exp(x - m) / l
    return m, l, p


def online_softmax_trace(x: np.ndarray):
    """Return (m,l) trace for debugging."""
    m = -np.inf
    l = 0.0
    trace = []
    for xi in x:
        m_new = max(m, float(xi))
        l_new = l * np.exp(m - m_new) + np.exp(float(xi) - m_new)
        trace.append((float(xi), float(m_new), float(l_new)))
        m, l = m_new, l_new
    return trace


def online_stats_1d(x: np.ndarray):
    """Return (m, l) where l = sum exp(x - m)."""
    m = -np.inf
    l = 0.0
    for xi in x:
        m_new = max(m, float(xi))
        l = l * np.exp(m - m_new) + np.exp(float(xi) - m_new)
        m = m_new
    return m, l


def merge_stats(m_a: float, l_a: float, m_b: float, l_b: float):
    """Merge two (m,l) states into one."""
    m = max(m_a, m_b)
    l = l_a * np.exp(m_a - m) + l_b * np.exp(m_b - m)
    return m, l


def softmax_bug_no_rescale(x: np.ndarray):
    """A common bug: update m but do not rescale l when m increases."""
    m = -np.inf
    l = 0.0
    for xi in x:
        m_new = max(m, float(xi))
        # BUG: missing l = l * exp(m - m_new)
        l = l + np.exp(float(xi) - m_new)
        m = m_new
    return np.exp(x - m) / l


def fused_softmax_weighted_sum(scores: np.ndarray, values: np.ndarray):
    """Compute softmax(scores) @ values without materializing softmax.

    scores: [Tk]
    values: [Tk, D]
    returns: [D]

    This mimics the online (m,l,o) update used in fused attention.
    """
    m = -np.inf
    l = 0.0
    o = np.zeros(values.shape[1], dtype=np.float64)

    for s, v in zip(scores, values):
        s = float(s)
        v = v.astype(np.float64, copy=False)

        m_new = max(m, s)
        alpha = np.exp(m - m_new)  # rescale old state
        p = np.exp(s - m_new)

        l = l * alpha + p
        o = o * alpha + p * v
        m = m_new

    return (o / l).astype(values.dtype)


def bytes_softmax_output(M: int, N: int, dtype_bytes: int, passes_read: int = 2):
    """Rough global memory bytes for outputting a full softmax matrix.

    - reads: passes_read * M*N
    - writes: 1 * M*N
    """
    reads = passes_read * M * N * dtype_bytes
    writes = M * N * dtype_bytes
    return reads + writes


def bytes_attention_with_p(B: int, H: int, Tq: int, Tk: int, D: int, dtype_bytes: int):
    """Rough bytes if you materialize P and then compute O = P @ V.

    Assume:
    - write P once
    - read P once
    - read V once (for matmul)
    - write O once

    This is a simplification, but enough to see scale.
    """
    p_elems = B * H * Tq * Tk
    o_elems = B * H * Tq * D
    v_elems = B * H * Tk * D

    return (
        (p_elems * dtype_bytes)  # write P
        + (p_elems * dtype_bytes)  # read P
        + (v_elems * dtype_bytes)  # read V
        + (o_elems * dtype_bytes)  # write O
    )


def bytes_attention_fused(B: int, H: int, Tq: int, Tk: int, D: int, dtype_bytes: int):
    """Rough bytes for fused attention (do not materialize P).

    You still need to read V and write O.
    Scores S may be computed on the fly from Q and K tiles; here we ignore Q/K read,
    and focus on the difference: no P write/read.
    """
    o_elems = B * H * Tq * D
    v_elems = B * H * Tk * D
    return (v_elems * dtype_bytes) + (o_elems * dtype_bytes)


if __name__ == "__main__":
    np.random.seed(0)

    # Part A: online softmax correctness + trace
    x = np.array([3.0, 1.0, -2.0, 5.0], dtype=np.float64)
    trace = online_softmax_trace(x)
    print("trace (x, m, l):")
    for row in trace:
        print(row)

    m_full, l_full, p_online = online_softmax_1d(x)
    p_ref = softmax_stable(x)
    print("online:", p_online)
    print("ref   :", p_ref)
    print("max_abs_diff:", np.max(np.abs(p_online - p_ref)))

    # Part A2: block-wise stats merge check
    m_a, l_a = online_stats_1d(x[:2])
    m_b, l_b = online_stats_1d(x[2:])
    m_merge, l_merge = merge_stats(m_a, l_a, m_b, l_b)
    print("\nblock-merge m,l:", (m_merge, l_merge), "full m,l:", (m_full, l_full))
    print("merge abs diff:", abs(l_merge - l_full))

    # Part A3: demonstrate the common bug (no rescale)
    p_bug = softmax_bug_no_rescale(x)
    print("\nbug (no rescale) max_abs_diff:", np.max(np.abs(p_bug - p_ref)))

    # Part B: fused softmax-weighted-sum correctness
    Tk, D = 8, 4
    scores = np.random.randn(Tk).astype(np.float32)
    values = np.random.randn(Tk, D).astype(np.float32)

    out_fused = fused_softmax_weighted_sum(scores, values)
    out_naive = softmax_stable(scores) @ values

    print("fused vs naive max_abs_diff:", np.max(np.abs(out_fused - out_naive)))

    # Part C: bandwidth bookkeeping
    B, H, Tq = 1, 32, 1
    Tk, D = 4096, 128
    dtype_bytes = 2  # fp16/bf16

    bytes_with_p = bytes_attention_with_p(B, H, Tq, Tk, D, dtype_bytes)
    bytes_fused = bytes_attention_fused(B, H, Tq, Tk, D, dtype_bytes)

    print("\nattention bandwidth (rough, decode Tq=1):")
    print("materialize P bytes:", bytes_with_p / (1024 * 1024), "MB")
    print("fused (no P) bytes:", bytes_fused / (1024 * 1024), "MB")
    print("ratio:", bytes_with_p / bytes_fused)

    # full softmax output example (M rows)
    M, N = 1024, 4096
    softmax_bytes = bytes_softmax_output(M, N, dtype_bytes, passes_read=2)
    print("\nfull softmax output bytes:", softmax_bytes / (1024 * 1024), "MB")
```

你可以先看两个验收信号：

- `online` 与 `ref` 的 `max_abs_diff` 应接近 0（浮点误差范围内）
- `fused vs naive max_abs_diff` 应接近 0

注意：在真实 GPU kernel 中，由于归约顺序与混合精度，误差常见量级可能到 `1e-4 ~ 1e-3`，这是正常的“数值等价”。

---

## E — Engineering（工程场景：三种你真的会遇到的地方）

### 场景 1：FlashAttention / Attention 内核（只要 O，不要 P）

**背景**：`P` 是 `Tq×Tk` 的大矩阵，但最终只需要 `O=P@V`。  
**为什么适用**：fusion 避免写回 `P`，带宽直接省掉一大截。  

最小化心智模型：维护 `(m,l,o)`，扫完 key 维就得到 `O=o/l`。

### 场景 2：Cross-Entropy（只要 logsumexp，不要 softmax 概率）

**背景**：loss 常写成 `-log softmax(x)[y]`。  
**为什么适用**：你根本不需要输出完整概率向量，只需要 `logsumexp(x)` 与 `x[y]`。  

可写成：

$$
\text{loss} = -x_y + \operatorname{LSE}(x)
$$

这让融合 kernel（logits + reduce + loss）成为自然选择。

### 场景 3：你确实需要概率向量（采样/可视化/后处理）

**背景**：例如要做 top-k / nucleus sampling、或者把概率分布输出给其他模块。  
**为什么适用**：这时无法彻底避免输出 `P`，但你仍然可以：  

- 使用 2-pass（读两遍输入）避免临时写回 `exp(x-m)`
- 或者在 N 较小的情况下用 shared memory 缓存一行，减少 global read

---

## Alternatives & Tradeoffs（选择不是二选一）

| 方案 | 你得到什么 | 你付出什么 | 何时合适 |
| --- | --- | --- | --- |
| 2-pass 稳定 softmax（输出概率） | 完整概率向量 | 2×读输入 + 1×写输出 | 需要概率输出 |
| 1-pass + 临时缓冲 | 读输入一次 | 写/读临时（可能更差） | N 小或临时在 SRAM |
| 融合 softmax（attention/xent） | 不写回大中间态 | 寄存器/共享内存压力上升 | `Tk>>D`，链路可融合 |
| 近似 softmax | IO/算量都降 | 改数学、改质量 | 明确接受近似 |

务实建议：

- 先问“我真的需要 softmax 概率吗？”——很多情况下不需要
- 不需要时，fusion 几乎总是 ROI 更高

---

## 常见问题与注意事项

1) **在线 softmax 和 2-pass softmax 结果完全一样吗？**  
在同样的精度与归约顺序下等价；真实 GPU kernel 可能因为混合精度/归约顺序不同出现 `1e-4~1e-3` 的数值差异。

2) **mask（-inf）怎么处理？**  
必须保证：全 mask 的行不会出现 `l=0` 导致 NaN。工程上常见做法是对全 mask 行输出 0，或在 `l==0` 时做保护。

3) **为什么 fusion 有时反而变慢？**  
fusion 可能显著增加寄存器使用，导致 occupancy 下降；当 `N` 很小或算子链很短时，收益会缩水。

---

## 最佳实践与建议

- 先用最小 trace 验证 `(m,l)` 更新是否正确（尤其是 m 变大时的 rescale）
- fp16/bf16 输入时，内部用 fp32 维护 `m,l,o`（否则容易 NaN/误差大）
- attention/cross-entropy 优先考虑 fusion，避免写回大中间态
- 写性能分析时把 IO 写出来（读几遍、写几遍），不要只写 `O(N)`

---

## 小结 / 结论

softmax 的 GPU 优化路线可以记成三句话：

1) 输出概率向量时，稳定 softmax 基本绕不开 2-pass（除非你愿意写临时缓冲）
2) 在线 softmax 的 `(m,l)` 更新把归约变成可组合的 streaming 过程，是 fusion 的基础
3) 真正的性能大头来自“别写回大中间态”：attention 的 `P`、xent 的概率向量，都能通过 fusion 避免

---

## 参考与延伸阅读

- FlashAttention (online softmax + fusion): https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2305.13245
- NVIDIA Blog（softmax 优化）：https://developer.nvidia.com/blog/optimizing-softmax

---

## 元信息

- **阅读时长**：约 16 分钟
- **标签**：softmax、gpu、memory、kernel-fusion
- **SEO 关键词**：Softmax, GPU, 访存优化, Online Softmax, Kernel Fusion
- **元描述**：softmax 的 GPU 访存优化：在线更新、融合与带宽算账，含可运行示例。

---

## 行动号召（CTA）

如果你愿意提供你的场景参数（不含业务信息）：

- `B/H/Tq/Tk/D`
- dtype（fp16/bf16/fp32）
- 你是否需要输出概率矩阵（是/否）

我可以帮你做一份更贴近你模型的“IO 算账 + 决策建议”，告诉你：2-pass、fusion、还是其他路线更划算。
