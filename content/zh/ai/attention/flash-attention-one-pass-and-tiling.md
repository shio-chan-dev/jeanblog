---
title: "FlashAttention 为什么能 one-pass：在线 softmax（m/l）与 Tiling 的核心思想"
subtitle: "把 Attention 从“存矩阵”改成“流式归约”：不落地 $QK^\\top$ / $P$，但结果等价"
date: 2026-01-25T12:51:14+08:00
summary: "从标准注意力的显存 IO 账本出发，解释 FlashAttention 的核心：在线 softmax 维护 (m,l) 并流式累积输出，再配合 tiling 把数据驻留在片上存储，从而避免显式存储 $QK^\\top$ 与 softmax 概率矩阵。本文给出可运行的 Numpy 分块注意力实现与数值等价验证，并用可复制的字节算账方法说明它为什么会快。"
description: "解释 FlashAttention 的 one-pass 计算原理与 tiling 策略：在线 softmax（m,l）更新、流式累积输出、不落地 $QK^\\top$ 与概率矩阵，并给出可运行的块级注意力验证与访存算账。"
draft: false
categories: ["AI", "Multimodal"]
tags: ["flash-attention", "attention", "online-softmax", "tiling", "gpu", "memory", "kernel-fusion"]
keywords: ["FlashAttention", "Online Softmax", "LogSumExp", "Tiling", "Shared Memory", "HBM", "One-pass", "Attention"]
readingTime: 15
---

## 副标题 / 摘要

FlashAttention 的“one-pass”不是在说 **attention 的数学公式只扫一遍就结束**（你仍然要看完所有 K/V），而是在说：
**对每个 Q tile，你只需要“流式扫过”一次 K/V，就能同时完成 softmax 与输出累积，不必把 $QK^\top$ 或 softmax 概率矩阵 $P$ 落到显存**。

它靠两件事合体：

1) **在线 softmax（online softmax）**：维护每一行的 `(m, l)`（max 与 exp-sum 的统计量），支持分块更新，且数值稳定；  
2) **Tiling（分块驻留）**：把 Q/K/V 切成能装进寄存器/共享内存的小块，在片上完成“算分数→归一化→乘 V→累积”的闭环，避免写回中间矩阵。

- **预计阅读时长**：约 15 分钟  
- **标签**：`flash-attention`、`online-softmax`、`tiling`、`gpu`、`memory`  
- **SEO 关键词**：FlashAttention, Online Softmax, Tiling, One-pass, IO  
- **元描述**：拆解 FlashAttention one-pass 的本质：在线 softmax + tiling，含可运行验证与访存算账。  

---

## 目标读者

- 想把 FlashAttention 从“听说很快”落到“为什么快、快在哪里”的工程读者
- 关心 GPU HBM 带宽、共享内存、kernel fusion 的性能优化者
- 需要实现/排查“分块 attention、在线 softmax、因果 mask”的开发者

---

## 背景 / 动机（先把 $T^2$ 的账算清楚）

标准 attention（以单 head 为例）：

$$
S = \\frac{QK^\\top}{\\sqrt{D}},\\quad P = \\mathrm{softmax}(S),\\quad O = PV
$$

看起来只有三步，但对长序列来说，真正致命的是 **$T^2$ 级别的中间矩阵**：

- $S \\in \\mathbb{R}^{T\\times T}$（score 矩阵）
- $P \\in \\mathbb{R}^{T\\times T}$（softmax 概率）

给一个带数字的锚点（非常常见的规模）：

- 序列长度 `T = 8192`
- head dim `D = 128`
- dtype = fp16（2 bytes）

那么单 head 的 `T×T` 矩阵大小是：

$$
T^2 \\times 2\\text{B} \\approx 8192^2 \\times 2 \\approx 128\\text{MB}
$$

如果你还要把 $P$ 也落到显存，那就是 **额外再来一个 128MB**。
更糟的是：这些中间结果还会被“写一次、再读一次”（下一步要用），所以总的 HBM IO 会飙升。

FlashAttention 的核心目标因此不是“减少计算量”（$T^2D$ 级别的乘加并不会消失），而是：

> **不落地 $S$ / $P$，把 attention 从“存矩阵”改成“流式归约”，把瓶颈从 HBM IO 拉回到片上计算与复用。**

---

## 快速掌握地图（60–120 秒）

- 问题形状：`Q,K,V: [T, D]`（单 head；多 head 只是多一维）
- 核心一句话：**对每个 query 行（或一个 Q tile），扫过 K/V 的 tile，一边做在线 softmax，一边累积输出 $O$**
- 什么时候收益大：`T >= 2048`、HBM 带宽吃紧、显存不够/想更长上下文
- 什么时候收益小：`T <= 512` 或 CPU/小 batch 上，kernel 启动与 tiling 开销可能淹没收益
- 复杂度抬头：FLOPs 仍是 $O(T^2D)$，但 **HBM 读写从“多次 $T^2$”降到“接近 0 次 $T^2$”**
- 常见踩坑：在线 softmax 忘记“max 变大时重标定”（没有乘上 `exp(m_old - m_new)`）会直接算错

---

## 深挖重点（PDKH Ladder：本文只深挖两件事）

本文只围绕两条主线做深挖（避免把文章写成“FlashAttention 百科”）：

1) **在线 softmax 的 `(m, l)` 不变式**：如何分块更新、为什么数值稳定、为什么等价  
2) **Tiling 的 IO/共享内存预算**：tile 该怎么想、怎么“算账”判断值不值得做

我会在后文明确走完 PDKH 的关键台阶：最小例子 → 不变式 → 形式化更新 → 正确性 → 阈值与工程现实 → 失败模式。

---

## 主心智模型（Master Mental Model）

把注意力看成一个 **对 Key 维度做归约（reduction）** 的问题：

- 对每个 query 行 $i$，你要计算的是：
  - $\\mathrm{softmax}$ 的分母：$\\sum_j \\exp(s_{ij})$
  - 输出的加权和：$\\sum_j \\exp(s_{ij}) v_j$

这本质是两类“可流式累积”的量。
唯一障碍是：softmax 需要“全局 max”来稳定数值。
在线 softmax 的技巧就是：把“全局 max”也变成一种可流式更新的状态（`m`），并在 `m` 改变时重标定旧累积（`l` 与 `o`）。

一旦你接受“attention 是归约”，tiling 就变成自然的工程实现：把 `j` 维度切块，块内在片上计算与累积。

---

## 核心概念与术语（把变量说清楚）

### Attention 的基本量

- 序列长度：`T`
- head dim：`D`
- 单 head（为简化）：`Q,K,V ∈ R^{T×D}`
- 分数（score）：  
  $$
  S_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{D}}
  $$
- softmax 概率：  
  $$
  P_{ij} = \\frac{\\exp(S_{ij})}{\\sum_{t=1}^{T} \\exp(S_{it})}
  $$
- 输出：  
  $$
  o_i = \\sum_{j=1}^{T} P_{ij} v_j
  $$

### 在线 softmax 的状态量（每行一份）

我们维护三个状态（都是“到目前为止的归约结果”）：

- `m`：到目前为止的最大值（用于数值稳定）
- `l`：到目前为止的 `exp` 之和（在 `m` 的坐标系下）
- `o`：到目前为止的加权和（同样在 `m` 的坐标系下）

直觉：`(m,l,o)` 是“softmax 归一化 + 输出加权”的**可组合中间状态**。

### Tiling 的块大小

你可以把 Q 切成 `Bq` 行一块，把 K/V 切成 `Bk` 列一块：

- `Q_tile: [Bq, D]`
- `K_tile,V_tile: [Bk, D]`
- `S_tile: [Bq, Bk]`

每次只在片上处理一个 tile，处理完就丢掉 `S_tile`（不落地）。

---

## 可行性与下界直觉：什么“必然要做”，什么“可以不做”

### 你不可能逃掉的下界（非正式）

无论你用什么 attention 实现，只要你要得到精确的 $O$：

- 至少要读一遍 `Q,K,V`：$\\Omega(TD)$
- 至少要写出 `O`：$\\Omega(TD)$
- 并且对全 attention（非稀疏）来说，分数涉及所有 `i×j` 对，FLOPs 量级仍是 $\\Omega(T^2D)$

FlashAttention 不承诺“把 $T^2$ 变成 $T$”，它做的是：**把 $T^2$ 级别的“显存写回/再读”去掉**。

### 你无法避免落地的反例（失败模式）

如果你的需求是：

- 需要显式保存注意力矩阵 $P$（可解释性可视化、某些蒸馏/约束项、或后续模块要复用 P）

那么你就不得不把 $P$ 写到显存（或至少写到某个可复用的存储中）。
FlashAttention 的“省 IO”优势会显著下降，甚至失去意义。

---

## 基线与瓶颈（朴素实现为什么会被 HBM 拖死）

把标准 attention 朴素地拆成三段 kernel（或三段大的算子）：

1) GEMM：写出 $S = QK^\\top$  
2) softmax：读 $S$、写 $P$  
3) GEMM：读 $P$ 与 $V$、写 $O$

只从“是否落地”看，$S$ 与 $P$ 都是 **$T^2$ 级别** 的全量矩阵。
而且它们都会被至少“写一次+读一次”。

### 一个可复制的字节账本（用它判断优化值不值）

用非常粗粒度但足够有用的模型估算（单 head，fp16 存储）：

- 写一次 `T×T`：$T^2×2$ bytes  
- 读一次 `T×T`：$T^2×2$ bytes

朴素三段里最显眼的四项是：

- 写 `S`：$T^2×2$
- 读 `S`：$T^2×2$
- 写 `P`：$T^2×2$
- 读 `P`：$T^2×2$

合计约：

$$
\\text{HBM bytes on } S/P \\approx 4\\times T^2 \\times 2\\text{B}
$$

代入 `T=8192`：

$$
4\\times 8192^2 \\times 2\\text{B} \\approx 512\\text{MB (per head)}
$$

注意：这还没算读 `Q/K/V`、写 `O`，也没算多 head、多 batch。
它解释了为什么很多时候 attention 的瓶颈不是算力，而是“写来写去、读来读去”。

---

## 关键观察：softmax 和输出都可以“在线更新”

softmax 的稳定实现通常要先求 max 再求 sum，看起来像“两遍”：

1) $m = \\max_j s_j$  
2) $l = \\sum_j \\exp(s_j-m)$  

但是注意：当你把 `s` 分成多个块时，你并不需要“先见全局再开始算”。
你只需要维护一个能被块级更新的状态 `(m,l)`，并在 max 变大时把旧的累积重标定。

更进一步：attention 输出并不是 softmax 的完整向量，而是 $\\sum_j P_j v_j$。
如果你能在线维护 $\\sum_j \\exp(s_j-m) v_j$，那你就不必显式保存 $P$。

这就是 FlashAttention 的算法基石。

---

## 在线 softmax（m/l）更新：最小例子 → 不变式 → 形式化

这部分是全文第一个深挖重点（PDKH）。
先把 attention 去掉，只看“一行 softmax”的在线更新，理解之后再把它嵌回 attention。

### P：把问题重述成“可组合的归约”

你要计算：

$$
\\mathrm{softmax}(s)_j = \\frac{\\exp(s_j)}{\\sum_t \\exp(s_t)}
$$

但我们希望支持流式输入：`s` 一段一段来（比如每次来 `Bk` 个）。
因此我们希望有一个状态 `(m,l)`，满足：

- 处理完当前段后，状态就能代表“到目前为止”的 softmax 归一化信息
- 下一段到来时，能在不回看旧数据的情况下更新状态

### D：最小可工作的数值例子（手算 2 步）

设分数向量 `s = [2, 1, 0]`，分两块处理：`[2,1]` 与 `[0]`。

初始化：`m=-inf, l=0`。

处理第一块 `[2,1]`：

- 块 max：`m_b = 2`
- 新 max：`m' = max(m, m_b) = 2`
- 更新 sum：  
  $$
  l' = l\\cdot e^{m-m'} + \\sum_{x\\in \\{2,1\\}} e^{x-m'}
     = 0 + (e^0 + e^{-1}) = 1 + 0.3679
  $$

处理第二块 `[0]`：

- 块 max：`m_b = 0`
- 新 max：`m' = max(2,0) = 2`（max 不变）
- 更新 sum：  
  $$
  l' = l + e^{0-2} = (1+e^{-1}) + e^{-2}
  $$

最后 softmax 分母就是 `l`，数值等价于稳定 softmax 的 `sum(exp(s-m))`。

关键点：**如果第二块里出现了更大的 max（比如出现 3），我们就必须把旧的 l 乘上 `exp(2-3)` 做重标定。**

### K：不变式（这句写清楚，后面都顺了）

当你已经处理了某个集合的元素 $J$ 时，维护：

$$
m = \\max_{j\\in J} s_j,\\quad
l = \\sum_{j\\in J} \\exp(s_j - m)
$$

这是一个非常强的不变式：它把“稳定 softmax”从一次性计算，变成了可以分块维护的状态。

### H：形式化更新（把“重标定”写成公式）

当新来一块分数集合 $B$，令块内最大值为 $m_B = \\max_{x\\in B} x$，则新 max：

$$
m' = \\max(m, m_B)
$$

新 sum：

$$
l' = l\\cdot \\exp(m - m') + \\sum_{x\\in B} \\exp(x - m')
$$

这就是 online softmax 的核心更新公式。

---

## 把在线 softmax 嵌回 attention：多维护一个 `o`

现在把 `s_j` 具体化成 attention 的分数 `s_{ij}`，并且我们最终要的是：

$$
o_i = \\sum_j \\mathrm{softmax}(s_i)_j \\cdot v_j
$$

如果我们沿用上面的 `m,l`，并定义（同样在 `m` 的坐标系下）：

$$
o = \\sum_{j\\in J} \\exp(s_j - m) \\cdot v_j
$$

那么最终输出就是：

$$
\\frac{o}{l}
$$

当新来一块 `B`，更新公式变成：

$$
o' = o\\cdot \\exp(m - m') + \\sum_{x\\in B} \\exp(x - m')\\cdot v(x)
$$

其中 `v(x)` 是该分数对应的 value 向量。
这条公式看起来像“多了一项”，但本质仍然是：**max 变大要把旧累积重标定**。

到这里，你已经拿到了 FlashAttention 的“数学发动机”。
接下来只剩“怎么在 GPU 上把它喂饱”——也就是 tiling。

---

## 算法步骤（Practice Guide：从公式到可实现的步骤）

以单 head、非 causal 为例（mask 只是在分数上加 `-inf`，后面会专门说坑）：

1. 选择块大小 `Bq, Bk`（受共享内存/寄存器预算约束）
2. 对每个 `Q_tile`（`[Bq, D]`）初始化每行的 `(m, l, o)`：
   - `m = -inf`，`l = 0`，`o = 0`
3. 按顺序扫描 `K_tile, V_tile`（每块 `[Bk, D]`）：
   - 计算 `S_tile = Q_tile @ K_tile^T / sqrt(D)`（形状 `[Bq, Bk]`）
   - （可选）对 `S_tile` 应用 mask（padding/causal）
   - 对每一行计算块 max `m_B`
   - 更新 `m' = max(m, m_B)`（逐行）
   - 重标定：`scale = exp(m - m')`
   - 更新 `l = l*scale + sum(exp(S_tile - m'))`
   - 更新 `o = o*scale + exp(S_tile - m') @ V_tile`
4. 扫完所有 K/V 块后输出：`O_tile = o / l`

这个流程的关键性质是：你只需要在片上短暂存在 `S_tile`，用完就丢，不需要把 `T×T` 的 `S` 或 `P` 写回显存。

---

## Worked Example（Trace：两块 K/V，手算一次在线更新）

为了把“重标定”看得更清楚，我们用最小但非平凡的例子：`T=3, D=1`，并把 K/V 分两块：前两列 + 最后一列。

设某一行的分数（已经除过 `sqrt(D)`）为：

- `s = [2, 1, 0]`
- 对应的 value（标量）为：`v = [10, 0, -10]`

我们分块处理：`B1 = [(2,10), (1,0)]`，`B2 = [(0,-10)]`。

初始化：`m=-inf, l=0, o=0`

处理块 B1：

- `m_B=2`，`m'=2`
- `scale = exp(m-m') = exp(-inf) = 0`
- `l = 0*0 + (exp(2-2)+exp(1-2)) = 1 + e^{-1}`
- `o = 0*0 + (exp(0)*10 + exp(-1)*0) = 10`

处理块 B2：

- `m_B=0`，`m'=2`（max 不变）
- `scale = exp(2-2)=1`
- `l = (1+e^{-1})*1 + exp(0-2) = 1 + e^{-1} + e^{-2}`
- `o = 10*1 + exp(0-2)*(-10) = 10 - 10e^{-2}`

最终输出：

$$
\\frac{o}{l} = \\frac{10 - 10e^{-2}}{1 + e^{-1} + e^{-2}}
$$

如果你用“全量 softmax”直接算同样的注意力加权和，会得到完全一致的值。
这个例子展示了两个要点：

1) 在线更新确实只需要一次扫过分块输入  
2) `m` 不变时很直观；`m` 变大时重标定是必须的（后文会给一个失败例子）

---

## Correctness（Proof Sketch：为什么分块更新等价于全量 softmax）

第二个深挖点仍然围绕在线 softmax（PDKH 的“不变式→正确性”）。

### 不变式（再写一遍，但这次带上 `o`）

当已经处理集合 $J$ 时，维护：

$$
m = \\max_{j\\in J} s_j
$$

$$
l = \\sum_{j\\in J} \\exp(s_j - m)
$$

$$
o = \\sum_{j\\in J} \\exp(s_j - m)\\, v_j
$$

### 保持性（为什么更新式能保持不变式）

设新来的块为 $B$，新 max 为 $m' = \\max(m, \\max B)$。

对于旧集合 $J$ 的每一项：

$$
\\exp(s_j - m) = \\exp(s_j - m') \\cdot \\exp(m' - m)
$$

因此把旧的 `l` 和 `o` 转换到新坐标系（以 `m'` 为基准）时，只需要乘一个统一系数：

$$
\\exp(m - m')
$$

这就是更新里 `l*exp(m-m')` 与 `o*exp(m-m')` 的来源。
然后再把新块 $B$ 的贡献（以 `m'` 为基准）加上即可。

### 终止性（为什么扫完就得到正确答案）

当 $J$ 覆盖了所有位置 `1..T`，根据定义：

- `l` 就是稳定 softmax 的分母（在 `m` 的基准下）
- `o/l` 就是 softmax 加权的 value 和

因此分块在线更新与全量 softmax 完全等价。

---

## 复杂度：FLOPs 没变，但空间与 IO 变了

- 时间复杂度：$O(T^2D)$（本质仍是 `Q@K^T` 与 `P@V` 的代数）
- 额外常数：在线更新需要 `exp`、逐行 `max/sum` 归约、以及重标定乘法
- 空间复杂度（中间矩阵）：
  - 朴素：显式存 `S,P` → $O(T^2)$
  - FlashAttention：不落地 `S,P` → 中间仅需要 tile + 状态 → 近似 $O(TD)$（外加每行的 `m,l`）

---

## 常数项与工程现实：Tiling 怎么“算得过账”

这部分是全文第二个深挖重点（PDKH：阈值/工程现实/失败模式）。

### 1) 共享内存预算（最粗但最实用）

一个常见的近似预算（忽略 score tile 的存储，因为很多实现会把 score 留在寄存器/分段计算）：

$$
\\text{bytes} \\approx (B_q + 2B_k) \\cdot D \\cdot \\text{bytes\\_per\\_elem}
$$

解释：

- `Q_tile`：`Bq×D`
- `K_tile`：`Bk×D`
- `V_tile`：`Bk×D`

举例（`D=128`, fp16=2 bytes）：

| Bq | Bk | 估算 bytes | 直观感受 |
|---:|---:|---:|---|
| 64 | 64 | `(64+128)*128*2 ≈ 49KB` | 很多 GPU 轻松装下 |
| 128 | 64 | `(128+128)*128*2 ≈ 65KB` | 接近/略超某些配置 |
| 128 | 128 | `(128+256)*128*2 ≈ 96KB` | 需要更大 shared memory 配置 |

注意：不同 GPU/驱动对每个 SM 的可用共享内存大小不同（并且会和 L1 配置互相影响）。
工程上你通常要做的是：

1) 选几组 `Bq/Bk` 候选  
2) 让 kernel 自动调参或基于硬件 query 选择能跑的最大块  
3) 用 profiler 看“HBM 吞吐 vs SM 占用”，找到甜点

### 2) IO 角度：为什么 tiling 能减少 HBM 访问

对一个 `Q_tile`：

- 朴素：可能会把 `S_tile` 写回 HBM（如果拆 kernel），后面 softmax 再读回来
- tiling + fusion：`S_tile` 不落地，`P_tile` 也不落地，直接做 `P_tile@V_tile` 贡献并累积到 `o`

因此你在 HBM 上“反复读写”的 $T^2$ 项被消掉了。
这就是很多场景下 FlashAttention 看起来像“魔法”的根因：**它其实是在做 IO 消元。**

### 3) Prefill vs Decode：one-pass 的收益不是一刀切

同样是 attention，但两种常见阶段的形状差别巨大：

- **Prefill**（提示词一次性喂入）：`Tq ≈ Tk ≈ T`，score 是 `T×T`  
  - 省掉 `S/P` 落地非常关键，收益大
- **Decode**（自回归每步生成 1 token）：`Tq=1, Tk≈T`，score 是 `1×T`  
  - `S/P` 本来就不是 `T×T`，收益点更多来自：
    - KV cache 读带宽（尤其多 head）
    - 更好的内存访问模式与融合

你可以用一个非常粗的阈值判断：

- 如果你看到 profiler 里 attention 的 bottleneck 是 “HBM 写回/读回” 大矩阵 → FlashAttention 极值  
- 如果 bottleneck 是 “读 KV cache” → 再结合 MQA/GQA 更明显（可对照同目录下的 MQA/GQA 文章）

### 4) 训练反向：不存 P 仍然能反传，但通常要存 (m,l)

很多人第一次看到“不存 P”会问：训练反向要用 softmax 概率，怎么办？

工程上常见做法是：**前向存每行的 `m` 与 `l`（或 `logsumexp`），反向时在需要时重算局部分数并恢复概率。**

带一个数量级锚点（单 head，`T=8192`）：

- 存 `P`：`T^2` 个 fp16 → `~128MB`
- 存 `(m,l)`：每行 2 个 fp32 → `T×2×4B ≈ 64KB`

这解释了为什么 FlashAttention 在训练里也能显著省显存：它把“必须保存的东西”从 $T^2$ 压到了 $T$。

---

## Runnable Implementation（Python / NumPy：在线 softmax + 分块 attention 验证）

下面的代码做三件事：

1) 实现一个“块级在线 softmax 加权”（`online_softmax_weighted_sum`）  
2) 实现一个“按 K/V 分块扫描”的 attention（`attention_block_online`）  
3) 用随机数据验证：分块实现与朴素实现数值一致（`allclose`）

你可以把它保存成 `demo_flash_attention.py` 直接运行。

运行方式示例：

```bash
python3 -m pip install numpy
python3 demo_flash_attention.py
```

```python
import math
from dataclasses import dataclass

import numpy as np


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k.T) / math.sqrt(d)
    p = softmax_stable(scores, axis=-1)
    return p @ v


@dataclass
class OnlineState:
    m: float
    l: float
    o: np.ndarray


def online_softmax_weighted_sum(scores: np.ndarray, values: np.ndarray, block: int) -> np.ndarray:
    """
    scores: [T]
    values: [T, D]
    return: [D] = sum softmax(scores)[j] * values[j]
    """
    assert scores.ndim == 1
    assert values.ndim == 2 and values.shape[0] == scores.shape[0]
    d = values.shape[1]

    state = OnlineState(m=-np.inf, l=0.0, o=np.zeros(d, dtype=np.float64))
    for start in range(0, scores.shape[0], block):
        sb = scores[start : start + block]
        vb = values[start : start + block]  # [Bk, D]

        m_b = float(np.max(sb, initial=-np.inf))
        m_new = max(state.m, m_b)
        scale = math.exp(state.m - m_new) if np.isfinite(state.m) else 0.0

        p = np.exp(sb - m_new)  # [Bk]
        state.l = state.l * scale + float(np.sum(p))
        state.o = state.o * scale + (p[:, None] * vb).sum(axis=0)
        state.m = m_new

    return (state.o / state.l).astype(values.dtype)


def attention_block_online(q: np.ndarray, k: np.ndarray, v: np.ndarray, bk: int = 128) -> np.ndarray:
    """
    A minimal FlashAttention-like formulation:
    - stream K/V in blocks along the sequence length
    - keep (m,l,o) per query row
    q,k,v: [T, D]
    """
    t, d = q.shape
    out = np.zeros((t, d), dtype=q.dtype)
    inv_sqrt_d = 1.0 / math.sqrt(d)

    for i in range(t):
        m = -np.inf
        l = 0.0
        o = np.zeros(d, dtype=np.float64)

        for start in range(0, t, bk):
            kb = k[start : start + bk]  # [Bk, D]
            vb = v[start : start + bk]  # [Bk, D]

            s = (q[i] @ kb.T) * inv_sqrt_d  # [Bk]
            m_b = float(np.max(s, initial=-np.inf))
            m_new = max(m, m_b)
            scale = math.exp(m - m_new) if np.isfinite(m) else 0.0

            p = np.exp(s - m_new)  # [Bk]
            l = l * scale + float(np.sum(p))
            o = o * scale + (p[:, None] * vb).sum(axis=0)
            m = m_new

        out[i] = (o / l).astype(out.dtype)

    return out


def bytes_accounting(T: int, dtype_bytes: int = 2) -> dict:
    """
    Extremely rough IO accounting for one head:
    - naive: materialize S and P, each written and read once
    - flash: do not materialize S/P
    """
    t2 = T * T
    return {
        "naive_S_write": t2 * dtype_bytes,
        "naive_S_read": t2 * dtype_bytes,
        "naive_P_write": t2 * dtype_bytes,
        "naive_P_read": t2 * dtype_bytes,
        "naive_SP_total": 4 * t2 * dtype_bytes,
        "flash_SP_total": 0,
    }


if __name__ == "__main__":
    np.random.seed(0)
    T, D = 64, 32
    q = np.random.randn(T, D).astype(np.float32)
    k = np.random.randn(T, D).astype(np.float32)
    v = np.random.randn(T, D).astype(np.float32)

    out_naive = attention_naive(q, k, v)
    out_block = attention_block_online(q, k, v, bk=16)

    max_abs = float(np.max(np.abs(out_naive - out_block)))
    print("max_abs_diff:", max_abs)
    print("allclose:", np.allclose(out_naive, out_block, rtol=1e-5, atol=1e-6))

    scores = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    values = np.array([[10.0], [0.0], [-10.0]], dtype=np.float64)
    y = online_softmax_weighted_sum(scores, values, block=2)
    y_ref = (softmax_stable(scores)[:, None] * values).sum(axis=0)
    print("online_weighted_sum:", y[0], "ref:", y_ref[0])

    T_big = 8192
    acc = bytes_accounting(T_big, dtype_bytes=2)
    print("S/P bytes per head:", acc["naive_SP_total"] / (1024**2), "MiB")
```

---

## 工程应用场景（Engineering Scenarios）

### 场景 1：长上下文 Prefill（训练/推理都常见）

当 `T` 上到 `4k/8k/16k`，朴素 attention 的 `S/P` 既吃显存又吃带宽。
FlashAttention 的价值是让你在同样显存预算下：

- 让 batch 更大（吞吐更好），或
- 让上下文更长（能力更强）

### 场景 2：推理 Decode（配合 KV cache / MQA/GQA）

decode 阶段 `Tq=1`，`S/P` 本身不再是 `T×T`，但你会遇到另一个硬瓶颈：读 KV cache 的带宽。
这时 FlashAttention 的 fused kernel + MQA/GQA 的共享 KV 往往是组合拳：

- MQA/GQA 先把“必须读的 KV”变少  
- FlashAttention 再把“读到片上以后怎么用”做得更高效

### 场景 3：显存紧张但又想稳定训练（反向需要状态）

在训练里，很多实现会额外保存 `m/l`（或 `logsumexp`）用于反向。
这仍然是 $O(T)$ 级别，不会把你拉回 $O(T^2)$ 的内存深坑。
你需要关心的是：fp16/bf16 下 `exp` 与累积最好在 fp32 做，否则数值误差会放大。

---

## Alternatives and Tradeoffs（替代方案与取舍）

| 方案 | 中间矩阵是否落地 | 显存/IO 形态 | 典型取舍 |
|---|---|---|---|
| 朴素三段（S→P→O） | 落地 `S`、`P` | 大量 `T^2` 读写 | 实现简单，但长序列很痛 |
| 融合版（FlashAttention 思路） | 不落地 `S/P` | 主要是 `Q/K/V/O` 的 `TD` IO | 实现复杂，但对带宽/显存更友好 |
| 稀疏/线性 attention（近似） | 通常不需要 `T^2` | 计算/精度取舍 | 可把复杂度降到近线性，但不是精确 softmax |

如果你要的是“精确 softmax attention + 更长上下文”，FlashAttention 通常是最实用的工程路线。

---

## 常见坑与边界条件（Pitfalls）

1) **忘记重标定（scale）会算错**  
   当新块出现更大的 `m_new`，必须把旧的 `l/o` 乘上 `exp(m_old - m_new)`；漏掉这一项会导致输出系统性偏差。

2) **mask 的时机不对**  
   causal/padding mask 本质是把某些分数设为 `-inf`。  
   你必须在求块 max 和 exp 之前应用 mask，否则 max/sum 会把不该参与的项算进去。

3) **fp16 下直接累计 (l,o) 会有精度坑**  
   实践里常用 fp32 维护 `m,l` 与累积，再在最后 cast 回去；否则长序列 `exp` 的动态范围会让误差变得可见。

4) **tile 过大导致 shared memory / 寄存器溢出**  
   不是越大越好：tile 太大会让单个 SM 同时驻留的 block 变少，吞吐反而下降。
   你需要用 profiler 找“带宽饱和但 SM 空闲”或“寄存器压力过大”的信号。

---

## 最佳实践（Best Practices）

- 先用“字节账本”判断是不是 IO 瓶颈：如果 `S/P` 的读写占主导，FlashAttention 值得做
- 在线 softmax 的状态（`m,l,o`）用 fp32；最终输出再 cast（尤其在 `T>=4096`）
- mask 在 tile 内尽早应用，并用 `-inf` 语义保持一致（避免用大负数导致溢出/不稳定）
- tile 大小不要凭感觉：用共享内存预算 + profiler 双校验

---

## 总结 / Takeaways

1) FlashAttention 的“one-pass”本质是：**对每个 Q tile 只流式扫一遍 K/V，就完成 softmax + 输出累积**  
2) 在线 softmax 用 `(m,l)` 把稳定 softmax 变成可组合的流式更新；max 变化时的重标定是关键  
3) tiling 的价值是把 `S_tile/P_tile` 留在片上，用完就丢，从而消掉 `T^2` 级别的 HBM 读写  
4) FLOPs 没变，但 IO 形态变了：从“反复写回/读回大矩阵”变成“读 Q/K/V、写 O 为主”  
5) 训练反向通常只需保存 `m/l`（$O(T)$），不需要保存 `P`（$O(T^2)$），这就是显存收益来源  

---

## 参考与延伸阅读

- FlashAttention (arXiv): https://arxiv.org/abs/2205.14135  
- Dao-AILab/flash-attention: https://github.com/Dao-AILab/flash-attention  
-（可对照阅读）同目录文章：`softmax-gpu-memory-io-optimization.md`（在线 softmax 与 IO 思路更细）  

---

## 元信息

- **阅读时长**：约 15 分钟  
- **标签**：flash-attention、online-softmax、tiling、gpu、memory  
- **SEO 关键词**：FlashAttention, Online Softmax, Tiling, One-pass, IO  
- **元描述**：拆解 FlashAttention one-pass 的本质：在线 softmax + tiling，含可运行验证与访存算账。  

---

## 行动号召（CTA）

把上面的代码跑起来，做两件事：

1) 把 `T` 从 64 改到 1024/4096，看 `attention_block_online` 仍然能和 `attention_naive` 对齐（数值等价）  
2) 用 `bytes_accounting(8192)` 算一次账，直观看到“落地 S/P”会带来多少 `T^2` 级别 IO  

然后你再去看 profiler 里的 attention 时间占比，会更容易判断：你的系统到底是算力瓶颈还是 IO 瓶颈。
