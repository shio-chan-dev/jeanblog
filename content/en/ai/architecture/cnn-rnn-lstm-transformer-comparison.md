---
title: "CNN, RNN, LSTM, and Transformer: Differences and When to Use Each"
subtitle: "Compare dependency path length and resource complexity to build a measurable selection model"
date: 2026-01-26T16:24:56+08:00
draft: false
categories: ["AI", "Architecture"]
tags: ["cnn", "rnn", "lstm", "transformer", "sequence-modeling"]
summary: "Using dependency path length and resource complexity, this article compares CNN, RNN, LSTM, and Transformer, and provides runnable examples plus a selection workflow."
description: "Using dependency path length and resource complexity, this article compares CNN, RNN, LSTM, and Transformer, and provides runnable examples plus a selection workflow."
keywords: ["CNN", "RNN", "LSTM", "Transformer", "architecture comparison"]
readingTime: "Approx. 27 min"
---

> **Subtitle / Abstract**
> Not a horizontal list. We use two core ideas: dependency path length and resource complexity.
> Treat "path length" as how far information can travel, and "resource complexity" as the hard constraint on trainability.
> Once you understand both, you can judge when CNN/RNN/LSTM/Transformer fits best and make measurable tradeoffs.

- **Estimated reading time**: Approx. 27 min
- **Tags**: `cnn`, `rnn`, `lstm`, `transformer`
- **SEO keywords**: CNN, RNN, LSTM, Transformer
- **Meta description**: Compare CNN, RNN, LSTM, and Transformer via path length and resource complexity.

---

## Target readers

- Beginners who want a fast comparison of major neural architectures
- Practitioners who must choose a model for production
- Developers working on sequence modeling or multimodal systems

## Background / Motivation

Choosing a model answers two questions:
1) How far and how long can information travel in a sequence (dependency path length)
2) Can compute and memory budgets support it (resource complexity)

This article stays on these two axes and avoids a wide, shallow overview.

A concrete example: when n = 1024, an RNN needs 1024 sequential steps for one forward pass.
A Transformer can achieve global interaction within 6 to 12 layers, but the attention matrix has n^2 = 1,048,576 elements.
These two hard facts almost decide the outcome: you are blocked by path length or by memory/throughput.
Ignore either axis and you will pay with accuracy or cost.

## Fast mastery map (60-120s)

- **Problem shape**: images/grids -> CNN; sequences -> RNN/LSTM/Transformer.
- **Core difference**: RNN/LSTM path length grows with n; Transformer path length is near 1 but cost grows as n^2.
- **When to use/avoid**: n <= 256 and low compute -> LSTM/RNN; n >= 512 and need parallelism -> Transformer; pure vision -> CNN.
- **Complexity keywords**: CNN O(HWk^2); RNN O(n d^2) serial; LSTM O(4 n d^2); Transformer O(n^2 d).
- **Common traps**: ignore n^2 memory, misjudge dependency range, mismatch masks or shapes.

## Master-level mental model

- **Core abstraction**: treat sequence modeling as information routing on a computation graph. Path length decides reachability; resource complexity decides feasibility.
- **Problem family**: local connections (CNN), chain propagation (RNN/LSTM), global similarity aggregation (Transformer) are different graphs with different shortest paths.
- **Isomorphic template**: `information routing = aggregate(neighbors)`. RNN uses linear neighbors, CNN uses fixed-radius neighbors, attention uses all-to-all neighbors.
- **Key invariant**: if shortest path L grows with n, long dependencies are hard to learn; if interactions are n^2, memory and time costs are unavoidable.

## Core concepts and terms (only two deep dives)

1. **Dependency path length and parallelism**: decides whether long dependencies can be modeled.
2. **Resource complexity (time/memory) vs n**: decides whether the model can be trained or deployed.

Key terms (used throughout):
- **Path length L**: shortest number of edges from position i to j in the computation graph.
- **Sequential steps S**: number of ordered steps in a forward pass. RNN has S ~= n, CNN/Transformer has S ~= number of layers.
- **Receptive field R**: span covered by CNN in input space, `R = 1 + (k - 1) * L` (no dilation).
- **Sequence length n / hidden size d**: dominant variables in complexity.

These four values are enough to write the core formulas for path length and resource complexity.

A directly usable estimate:
If each layer only connects neighbors within radius r, then spanning distance d needs
`L >= ceil(d / r)`.
Example: r = 2, d = 256 -> L >= 128, which is expensive in depth and gradients.

## Problem abstraction (inputs/outputs)

- **Image input**: `X in R^{B x C x H x W}`, output is classification/detection logits.
- **Sequence input**: `X in R^{B x n x d}`, output is per-step prediction or a sequence representation.
- **Optimization target**: maximize accuracy and throughput within compute/memory budget while meeting latency.

Typical engineering ranges:
- **Sequence length**: `n in [128, 8192]`, and `n >= 1024` is "long sequence".
- **Memory budget**: 16 to 80 GB per GPU; `n >= 4096` often triggers OOM with full attention.
- **Latency target**: online inference often wants P95 < 200 ms, amplifying serial bottlenecks.

## Feasibility and lower bound intuition

**Path length lower bound**: if each layer only connects neighbors within radius r (RNN has r = 1, CNN has r = (k - 1)/2),
then spanning distance d needs `L >= ceil(d / r)` layers.
Example: 1D CNN with k = 3, r = 1, covering d = 512 requires L >= 512 layers.
Even with k = 5 (r = 2), you still need L >= 256. Depth cost remains huge.

**Attention lower bound**: full attention computes similarity for any i, j,
which implies at least Omega(n^2) interactions or memory reads.
Unless you drop interactions (window, sparse, approximate), this upper bound is unavoidable.

A common compromise is **downsample then attend**:
If you reduce n from 2048 to 1024, attention cost drops to 1/4,
but each token covers more information, effectively changing the "path length".
You always trade between the two axes: compress length or pay quadratic cost.

## Naive baselines and bottlenecks

- **Baseline 1: RNN on long sequences**
  When n = 1024, you need 1024 sequential steps; GPU utilization is low.
  Backprop must keep all intermediate states, so training time rises sharply.
- **Baseline 2: shallow CNN for long dependencies**
  With k = 3 and L = 8, receptive field is only R = 17, which is almost blind for n = 512 tasks.
  Stack more layers to expand R and parameters and training time explode.

Even if per-step compute is cheap, **sequential steps decide latency**:
If one step is 0.3 ms, n = 512 RNN forward is about 154 ms.
Transformer has only as many steps as layers (for example 6 layers ~ 1.8 ms).
This is why baselines are usable but hard to scale.

## Key observation

Dependency is not the time order itself, but the strength of relationships between positions.
If all positions can "see" each other in one layer, path length drops from O(n) to O(1).
The cost is interaction count jumping from O(n) to O(n^2), i.e., resource complexity.

## Deep concept 1: dependency path length and parallelism (PDKH)

### 1) Restate the problem (Polya)

If information from position i must affect position j, it must travel along the computation graph.
The longer the path, the more gradients decay and the slower training becomes.

Treat each layer-position as a node, and each valid connection as an edge.
Path length L is the shortest path length.
Short paths mean fast aggregation; long paths mean repeated transforms before information arrives.
This is why path length nearly decides whether long dependencies can be learned.

### 2) Minimal example (Bentley)

Let sequence length n = 6, and position 1 must influence position 6:

- **RNN/LSTM**: must pass step by step, path length = 5.
- **CNN (k = 3, L = 2)**: receptive field is 1 + (k - 1)L = 5, still cannot reach position 6.
  You need L = 3 layers to cover the full range.
- **Transformer**: any position can attend within the same layer, path length = 1.

#### Path length and parallelism comparison

| Structure | Path length L (dependency distance d) | Parallelism | Notes |
| --- | --- | --- | --- |
| RNN | `L = d` | Low | Serial dependency, hard to parallelize |
| LSTM | `L = d` | Low | Gates mitigate gradient decay |
| CNN | `L >= ceil((d - 1)/(k - 1))` | Medium-High | Depends on depth and kernel width |
| Transformer | `L = 1` | High | Global attention in parallel |

#### Sequential step examples (S)

Assume n = 1024 for a single forward pass:
- **RNN/LSTM**: 1024 ordered steps, S ~= 1024.
- **Transformer (6 layers)**: 6 ordered steps, S ~= 6.
- **CNN (20 layers)**: 20 ordered steps, S ~= 20.

This explains why RNN throughput is low on GPU: not slow ops, but too many serial steps.

A rough estimate: if each step is ~0.2 ms,
S = 1024 gives ~205 ms per forward pass;
S = 6 gives ~1.2 ms (ignoring communication and memory bottlenecks).

#### Worked example: how many CNN layers for long dependencies?

To cover dependency distance d = 512 with k = 3,
`L >= (d - 1)/(k - 1) = 255.5`, so at least 256 layers.
This is why CNNs are often replaced by attention for long sequences.

#### Micro-trace: n = 4 dependency propagation

Sequence `[x1, x2, x3, x4]`, make x1 influence x4:
- **RNN**: x1 -> h2 -> h3 -> h4, path length = 3.
- **CNN (k = 3, L = 2)**: layer 1 lets x1 affect {x1, x2}, layer 2 reaches x3 but not x4.
- **Transformer**: x1 directly participates in x4 attention, path length = 1.

This tiny example shows path length differences exist even at the smallest scale.

### 3) Invariants / contracts (Dijkstra/Hoare)

> To stably capture dependencies of distance d, the graph must provide paths with length L <= d.
> When L grows with n, long-dependency training becomes substantially harder.

#### Gradient decay intuition

RNN gradients are products of Jacobians:
`d h_t / d h_{t-k} = product_{i=t-k+1}^{t} J_i`.
When k is large, the product quickly shrinks or explodes. This is the root of long dependency difficulty.
LSTM uses the cell state as a more direct path, but the length is still O(n).

A numeric intuition: if the average spectral radius is ~0.9,
then after 100 steps the gradient magnitude is about 0.9^100 ~= 0.000026.
Even at 0.99, 0.99^100 ~= 0.366, still decaying.
So the longer the path, the more you rely on gates or residuals to keep training stable.

#### Dependency span examples (why long dependencies are hard)

**Copy task**: sequence length n = 512, output the first token at the end.
RNN/LSTM must carry information for 511 steps.
Transformer can connect start and end in one attention pass.

**Bracket matching**: matching outer parentheses often spans the entire sequence.
Such tasks are extremely sensitive to path length and often favor Transformers.

#### Estimating dependency span

- **Text**: measure dependency distances within a sentence (often < 128).
  If cross-paragraph dependencies are common, spans can reach 512 or more.
- **Time series**: use autocorrelation length as "effective memory".
- **Video/visual sequences**: span is driven by object trajectories across frames.

Practically, use P90 as a "safe span":
If 90% of dependencies are below 256, CNN/LSTM is often enough.
If P90 exceeds 512, Transformer advantages are usually stable.

Once you estimate a typical span d, model selection has a direction.

### 4) Formalization (Knuth)

- **RNN/LSTM**: path length `L = |i - j|`.
- **1D CNN**: receptive field `R = 1 + (k - 1)L`, so covering distance d needs `L >= (d - 1)/(k - 1)`.
- **Transformer**: one layer connects any positions, `L = 1`.

Parallelism can be seen as "how many sequential steps are required":
RNN/LSTM need n steps; CNN/Transformer mainly depend on depth.
This directly explains why Transformers have high training throughput.

### 5) Correctness sketch (Dijkstra/Hoare)

- RNN state only moves from t-1 to t, so crossing distance d needs d steps.
- CNN expands receptive field by (k - 1) per layer, so L layers give R = 1 + (k - 1)L.
- Transformer attention builds global dependencies, so path length is 1.

#### Structure-by-structure deepening (path length view)

**CNN:**
Receptive field grows linearly. For k = 3, it goes 3, 5, 7, 9...
At L = 6, R = 13; at L = 20, R = 41.
This shows why CNNs need extreme depth for long dependencies.

With dilation, the formula becomes
`R = 1 + (k - 1) * sum d_l`.
Example: 4 layers with d_l = [1, 2, 4, 8] gives
`R = 1 + 2 * (1 + 2 + 4 + 8) = 31`.
This is larger but still far from n = 512.
It improves path length but does not change the linear-growth nature.

**RNN:**
Path length equals time steps. For n = 512, the furthest dependency needs 511 state transfers.
Even if per-step compute is cheap, long chains amplify gradient decay.

**LSTM:**
Gating stabilizes "effective memory", but path length is still O(n).
In practice, tricks like setting forget bias b_f = 1 extend memory but do not change the order of growth.

**Transformer:**
Path length is 1, turning long dependencies into global parallel matrix ops.
The cost is higher memory and compute (see concept 2).

#### How LSTM gating extends "effective memory"

LSTM centers on cell state c_t with three gates:
`f_t = sigma(W_f [x_t, h_{t-1}])` (forget gate)
`i_t = sigma(W_i [x_t, h_{t-1}])` (input gate)
`o_t = sigma(W_o [x_t, h_{t-1}])` (output gate)
`c_t = f_t * c_{t-1} + i_t * tanh(W_c [x_t, h_{t-1}])`

Here `*` is elementwise multiplication.
When f_t is near 1, c_t preserves information longer.
This explains why LSTM handles medium-long sequences better than vanilla RNN,
but it does not change the fact that path length grows with n.

If the mean f_t is ~0.95, the memory factor after 200 steps is 0.95^200 ~= 0.00034.
Even at 0.99, after 200 steps it is 0.99^200 ~= 0.133.
Gates extend "effective path length" but cannot change the order.

#### Transformer "short path" still needs order signals

Attention is permutation-invariant; without positional encoding,
Transformer treats the sequence as a set.
So path length = 1 does not automatically solve order.
Positional encoding is required.

A common sinusoidal form:
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

From a graph view, attention matrix `A = softmax(QK^T)` is a weighted fully connected graph.
Each row sums to 1, output is a convex combination of values.
So a single attention layer can route information between any positions,
which is why path length collapses to 1.

#### Worked example: CNN receptive field growth table (k = 3)

| Layers L | Receptive field R |
| --- | --- |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 8 | 17 |
| 16 | 33 |

```python
def receptive_field(k, layers):
    return 1 + (k - 1) * layers

for L in [1, 2, 4, 8, 16]:
    print(L, receptive_field(3, L))
```

### 6) Thresholds and scale (Knuth)

- When dependency span > 256, RNN/LSTM often struggle.
- When span > 512, Transformer advantages become clear.
- But this also introduces n^2 cost (see concept 2).

These thresholds are empirical, not theoretical limits.
In speech and short text (n ~ 128-256), LSTM can still be stable.
In long documents and code (n >= 512), path length dominates,
and if you also need high throughput, attention parallelism becomes valuable.

### 7) Counterexamples / failure modes (Bentley/Sedgewick)

If the task is local dependency (for example n <= 128 short text classification),
Transformer can overfit due to excessive global modeling.
In that case LSTM or 1D CNN is often more stable.

Example: in a sentiment task with n = 64 and only tens of thousands of samples,
Transformer capacity can be too high, so path-length advantage does not translate to accuracy.

### 8) Engineering reality (Knuth)

Shorter path is not always better:
Transformer needs positional encoding to express order;
RNN/LSTM can still keep stable memory for n = 200-500.

Common "mitigations":
- CNN uses residual or pyramid structures to extend receptive field.
- RNN/LSTM uses truncated BPTT to control training cost.
- Transformer uses relative positional encoding to strengthen locality.
All of these shorten the effective path without changing the main structure.

Truncated BPTT impact: if you truncate backprop to 256 steps,
you effectively cap dependency span at 256.
This is fine for speech or short text, but hurts long-doc summarization or code understanding.
So truncation length is your "engineering path length budget".

## Deep concept 2: resource complexity as n grows (PDKH)

### 1) Restate the problem (Polya)

When n grows, can the model still be trained and deployed?
This is decided by time and memory complexity.

Split resource complexity into three dimensions:
1) **Compute (FLOPs)**: determines training/inference speed.
2) **Memory footprint**: decides OOM risk.
3) **Memory bandwidth**: determines whether throughput is limited by reads/writes.
Transformer is often not compute-bound; it is memory-bandwidth bound.

### 2) Minimal example (Bentley)

Let n = 2048, d_model = 512, h = 8:

- Attention matrix has n^2 = 4,194,304 elements.
- One head in FP16 is about 8 MB; 8 heads about 64 MB.
- Training also needs activations and gradients, often 3 to 5 times peak.

#### Resource estimate (attention weights)

If batch is B, heads h, dtype FP16 (2 bytes):
`memory ~= B * h * n^2 * 2 bytes`.
Example: B = 4, h = 8, n = 2048:
`4 * 8 * 2048^2 * 2 ~= 512 MB` (attention weights only).

This scales brutally:
- **B doubles** -> memory doubles.
- **n doubles** -> memory becomes 4x.
- **h doubles** -> memory doubles.
So increasing n from 2k to 4k is often more dangerous than increasing layers.

A more practical estimate is to solve for max n:
`n_max ~= sqrt(memory_budget / (B * h * 2 bytes))`.
If memory budget is 8 GB, B = 2, h = 8, then
`n_max ~= sqrt(8 GB / 32 bytes) ~= 16k`.
But with 4x to 8x peak overhead, real n is usually 3x to 4x smaller.

#### n and memory scale (single head FP16)

| n | n^2 elements | Approx memory |
| --- | --- | --- |
| 512 | 262,144 | ~0.5 MB |
| 1024 | 1,048,576 | ~2 MB |
| 2048 | 4,194,304 | ~8 MB |
| 4096 | 16,777,216 | ~32 MB |
| 8192 | 67,108,864 | ~128 MB |

You must also consider **memory bandwidth**:
At n = 2048, one head's weights are ~8 MB; with 12 layers that is ~96 MB of reads/writes.
Training also reads/writes gradients and activations, so bandwidth pressure grows.
This is why FlashAttention speeds up by reducing reads/writes.

```python
def attn_memory_mb(n, h=8, batch=4, bytes_per_elem=2):
    return batch * h * n * n * bytes_per_elem / (1024 ** 2)

for n in [512, 1024, 2048, 4096]:
    print(n, f"{attn_memory_mb(n):.1f} MB")
```

### 3) Invariants / contracts (Dijkstra/Hoare)

> With full attention, you must compute n^2 interactions explicitly or implicitly.
> Without approximation, this cost cannot be avoided.

### 4) Formalization (Knuth)

- **CNN**: `O(HWk^2)` (or `O(n k d^2)` for sequences)
- **RNN**: `O(n d^2)` (serial)
- **LSTM**: `O(4 n d^2)`
- **Transformer**: `O(n^2 d)` + `O(n d^2)` (FFN)

#### Rough compute estimate (n = 1024, d = 512)

- **RNN**: per-step d^2, total `1024 * 512^2 ~= 268M` MACs.
- **LSTM**: about 4x, `~1.07B` MACs.
- **Transformer attention**: `n^2 * d_k`. If d_k = 64, `1024^2 * 64 ~= 67M` MACs,
  but FFN adds `2 * n * d * d_ff` (d_ff = 2048 gives ~2.1B).

Conclusion: Transformer bottlenecks are often FFN and attention memory, not pure FLOPs.

The dominance boundary is:
`n^2 d` (attention) vs `2 n d d_ff` (FFN).
Simplify to `n > 2 d_ff` for attention to dominate.
If d_ff = 2048, attention dominates only when n > 4096.
This explains why FFN is the bottleneck at moderate length and attention dominates at very long length.

### 5) Correctness sketch (Dijkstra/Hoare)

Transformer computes QK^T for all token pairs,
so time and memory must scale with n^2.

### 6) Thresholds and scale (Knuth)

- n <= 2048: full attention is usually feasible.
- 2048 < n <= 8192: use FlashAttention or block attention.
- n > 8192: require sparse/linear attention or retrieval.

A practical upper bound:
Single GPU 24 GB, B = 2, h = 8, attention weights at n = 4096 are ~512 MB.
With activations and optimizer states, you often approach 16 to 24 GB.
So n = 4k is already a warning line for single-GPU training.

### 7) Counterexamples / failure modes (Bentley/Sedgewick)

On a 16 GB single GPU, forcing n = 8k full attention
will often OOM or require tiny batch sizes, lowering efficiency.

### 8) Engineering reality (Knuth)

Common fixes: FlashAttention, block attention, KV cache, gradient checkpointing.
These trade engineering complexity for trainability and throughput.

#### Training vs inference complexity

- **Training**: full attention needs n^2 matrix, high memory and compute.
- **Inference (autoregressive)**: with KV cache, each step interacts with history,
  per-step complexity is ~O(n), and memory is more manageable.

This is why Transformer can be barely workable in inference,
but needs strong compute and memory optimization during training.

KV cache memory estimate:
`memory ~= B * h * n * d_k * 2 bytes`.
If B = 1, h = 8, n = 4096, d_k = 64, memory is ~4 MB.
But if B = 8 or n = 16k, memory grows linearly and must be planned.

#### Worked example: n = 1024 vs n = 4096

For single head FP16:
- n = 1024, attention weights ~2 MB.
- n = 4096, ~32 MB, **16x larger**.

If B = 4, h = 8, n = 4096, attention weights alone exceed 1 GB,
not counting gradients or activations.
So "double length" is not linear cost, it is geometric.

## Complexity and scale summary (two axes together)

| Structure | Path length L | Sequential steps S | Time complexity (dominant) | Memory complexity (dominant) |
| --- | --- | --- | --- | --- |
| CNN | `~(d/(k-1))` | `~layers` | `O(n k d^2)` or `O(HWk^2)` | `O(n d)` |
| RNN | `d` | `~n` | `O(n d^2)` | `O(n d)` |
| LSTM | `d` | `~n` | `O(4 n d^2)` | `O(n d)` |
| Transformer | `1` | `~layers` | `O(n^2 d) + O(n d^2)` | `O(n^2) + O(n d)` |

This table puts path length and resource complexity on one plane:
short-path structures (Transformer) are resource-heavy;
resource-stable structures (RNN/LSTM) suffer on path length.

Remember sequential steps S are a hard ceiling.
Even with more machines, S is hard to parallelize away.
For n = 1024, RNN needs 1024 ordered steps; multi-GPU only increases batch, not reduces steps.

## Constant factors and engineering reality (related to the two axes)

- **Operator granularity**: RNN has many small matmuls, GPU utilization is low; Transformer has fewer large matmuls, but bandwidth is the bottleneck.
- **Precision and memory**: FP16/BF16 halves attention and activation memory, but does not change path length or dependency span.
- **Residuals and caching**: residuals shorten effective path but increase activation storage; short-path models rely more on cache and bandwidth.

## Worked example (trace): same task, path and cost

Toy task: n = 8, make token 1 influence token 8. Compare path and cost:

1) **RNN/LSTM**
   Path length L = 7, must pass 7 state transfers.
   Sequential steps S = 8, cannot be parallelized.

2) **CNN (k = 3)**
   Receptive field R = 1 + 2L.
   L = 1 -> R = 3, L = 2 -> R = 5, L = 3 -> R = 7, L = 4 -> R = 9.
   Only L >= 4 covers x1 -> x8.

3) **Transformer (1 layer)**
   Path length L = 1, x1 can directly influence x8.
   But attention matrix has n^2 = 64 elements per head.
   This is tiny at n = 8, but jumps to 4,194,304 at n = 2048.

This shows: path advantage exists at small scale; resource disadvantage explodes at large scale.

## Practical guide / steps (selection workflow)

1. **Estimate dependency span**: for text, use dependency or sentence span; frequent cross-paragraph links suggest d >= 512.
2. **Estimate sequence length n**: use P50/P90/Max because n determines n^2 cost.
3. **Check budget**: use `B * h * n^2 * 2 bytes` to estimate attention memory and leave 4x to 8x headroom.
4. **Check parallelism needs**: if online inference needs P95 < 200 ms, serial models are likely out.
5. **Run a light baseline**: small CNN/LSTM to verify learnability and set a minimum accuracy bar.
6. **Upgrade structure**: if d is large and budget allows, move to Transformer; if budget is tight, consider local attention or hybrid models.

You can compress this into two must-answer questions:
- **Dependency span**: is the furthest dependency d clearly > 256?
- **Budget**: can memory handle `B * h * n^2` attention?

If both are clear, selection is usually on track.

A simplified 2x2 decision matrix:
- **small d + small budget** -> CNN/LSTM
- **small d + large budget** -> small Transformer or CNN
- **large d + small budget** -> local attention or hybrid
- **large d + large budget** -> full Transformer

If d is unclear, train a small attention model and inspect attention span distribution first.

## Selection guide

- **Dependency span threshold**: if d <= 128, CNN or small RNN is often enough; d >= 512 favors Transformer.
- **Sequence length threshold**: n <= 256 makes full attention cheap; n >= 2048 requires memory planning.
- **Memory budget threshold**: on a 24 GB GPU, B = 2, h = 8, n = 4096 attention weights are ~512 MB.
  Add activations and optimizer state and you can hit 16 to 24 GB quickly.
- **Implementation complexity tolerance**: if your team cannot optimize kernels, use mature implementations (standard Transformer + FlashAttention).

## Runnable example (minimal contrast)

The code below only contrasts structure-level behavior, not training or loss.
It helps you see **CNN local aggregation**, **LSTM sequential state**, and **Transformer global interaction**.
Run it to observe output shapes.

```python
import torch
import torch.nn as nn

# CNN
cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10),
)
img = torch.randn(2, 3, 32, 32)
print("cnn:", cnn(img).shape)

# LSTM
lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
seq = torch.randn(2, 5, 16)
out, _ = lstm(seq)
print("lstm:", out.shape)

# Transformer
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True),
    num_layers=2,
)
seq = torch.randn(2, 6, 32)
print("transformer:", encoder(seq).shape)
```

## Explanation and principles (two axes)

- **Dependency path**: Transformer is shortest, RNN/LSTM is longest, CNN depends on depth and kernel size.
- **Resource cost**: Transformer is the most expensive (n^2), RNN/LSTM cost is linear but serial.

Other differences (gates, positional encoding) are reinforcement for these two axes.

If you place models on a 2D chart:
- **X-axis** = path length (shorter to the left)
- **Y-axis** = resource complexity (lower down)
RNN/LSTM sit lower but to the right, Transformer sits higher but to the left,
and CNN position varies with kernel size and depth.

The two axes are not independent:
reduce n and you lower cost but increase each token's semantic coverage;
increase layers and you shorten path but raise compute and training difficulty.
Real-world solutions often mix compression, local modules, and a small amount of global attention.

## Engineering scenarios (only 3, tied to two axes)

1. **Short text classification (n <= 128)**: small dependency span -> LSTM or 1D CNN is usually enough.
   With n <= 128, attention matrix is only ~16k elements, so Transformer advantage is limited.
2. **Long document summarization (n >= 1024)**: large dependency span -> Transformer, but you must manage n^2 cost.
   At n = 2048, attention weights are 4.2M elements, often requiring FlashAttention or block strategies.
3. **Streaming speech recognition**: low latency requirement -> CNN + LSTM hybrid is more stable.
   Serial steps hurt real-time latency, so local CNN compresses first and LSTM preserves mid-range dependencies.

## Alternatives and tradeoffs (only around the two axes)

- **Full attention vs local attention**:
  Full attention is O(n^2), local window attention is O(n w).
  If n = 2048, w = 256, cost is ~8x lower, but path length grows to about n/w ~= 8.
  If dependency span d = 2048 and window w = 256,
  you need at least L >= ceil(d / w) = 8 layers for global reach.
  You trade lower memory for deeper paths and harder training.
- **Deeper CNN vs adding attention**:
  With k = 3, CNN needs 256 layers to cover d = 512.
  Attention can reduce path length to 1 but adds n^2 memory cost.
- **RNN/LSTM vs Transformer**:
  The former has linear resources but long path; the latter has short path but quadratic resources.
  When n is small and d is not large, RNN/LSTM can have better cost-performance.
- **Increase kernel size vs increase depth**:
  Larger k shortens required depth but compute grows as O(k d^2).
  More layers keeps small kernels but still increases path length and training difficulty.

## Skill ladder

1. **Master local structures**: understand CNN receptive field and path length.
2. **Master chain propagation**: understand RNN/LSTM state transfer and gradient decay.
3. **Master global routing**: understand Transformer global interaction and n^2 cost.
4. **Extend in practice**: for very long n or tight budget, try local attention or hybrid models.

## Common questions and notes

- Underestimating n^2 memory leads to failed training.
- Missing positional encoding makes Transformer unable to represent order.
- Large LSTM hidden size overfits on small data.
- Shallow CNNs "cannot see" global dependencies, so performance stalls.
- Truncated BPTT too short caps dependencies at 128/256 and hurts long-range tasks.
- Doubling n without more data raises overfitting risk and memory cost sharply.
- Looking only at parameter count often underestimates real memory use.
- Using average n for budgeting is risky: if P90 doubles, attention memory is 4x.
- Excessive padding wastes attention on useless tokens; use length bucketing.

## Best practices and recommendations

- Treat dependency span as the first decision axis.
- Treat memory/throughput budget as the second axis.
- Validate learnability with a light baseline, then upgrade.
- If span is large but budget is small, prefer sparse/block attention or hybrids.
- For long sequences, compute n percentiles before selecting full attention.
- Tune n and h first; they have the largest effect on memory and throughput.
- Track peak memory during training, not just parameter count.
- For ultra-long sequences, try chunking/downsampling and check accuracy.
- Log P90 n, peak memory, and throughput in training runs.

## Summary / Conclusion

- CNN suits local patterns and vision grids.
- RNN/LSTM suits short to mid sequences and low compute.
- Transformer excels at long dependencies and parallel training but has n^2 cost.
- Model selection hinges on two axes: **dependency path length** and **resource complexity**.
- When d > 512, path length often dominates; when n > 2048, memory dominates.
- If budget is limited, shorten n or use local attention before deepening the model.

## References and further reading

- https://arxiv.org/abs/1409.2329
- https://arxiv.org/abs/1706.03762
- https://arxiv.org/abs/2010.11929

## Call to Action (CTA)

Run the same dataset with an LSTM and a Transformer, compare dependency span and memory cost, and write down your conclusion.
