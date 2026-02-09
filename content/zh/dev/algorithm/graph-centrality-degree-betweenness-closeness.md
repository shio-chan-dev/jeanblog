---
title: "图中心性三件套：Degree、Betweenness、Closeness 工程 ACERS 解析"
date: 2026-02-09T09:56:11+08:00
draft: false
categories: ["逻辑与算法"]
tags: ["图论", "中心性", "Degree", "Betweenness", "Closeness", "图数据库", "工程实践"]
description: "系统讲解图中心性三大指标：Degree、Betweenness、Closeness。重点给出工程可落地结论：大多数系统优先支持 Degree 和近似 Betweenness，并说明复杂度、近似策略与上线取舍。"
keywords: ["graph centrality", "degree centrality", "betweenness centrality", "closeness centrality", "approximate betweenness", "Brandes"]
---

> **副标题 / 摘要**  
> 中心性不是论文概念，而是图系统里的“节点重要性排序器”。本文按 ACERS 结构讲透 **Degree / Betweenness / Closeness**，并给出一条务实结论：**线上大多数系统只稳定支持 Degree + 近似 Betweenness**。

- **预计阅读时长**：12~16 分钟  
- **标签**：`图论`、`中心性`、`Degree`、`Betweenness`、`Closeness`  
- **SEO 关键词**：图中心性, Degree Centrality, Betweenness, Closeness, 近似 Betweenness  
- **元描述**：图中心性工程指南：三大指标定义、复杂度、近似算法与落地策略，附可运行代码。  

---

## 目标读者

- 做关系图分析、知识图谱、图数据库查询优化的工程师
- 需要把“节点重要性”从概念变成上线指标的开发者
- 想知道为何 Betweenness 工程上昂贵、如何做近似替代的同学

## 背景 / 动机

你在图系统里迟早会遇到这类问题：

- 哪些节点是“社交大 V”或“交易枢纽”？
- 哪些节点是关键桥梁，断开就会让图显著分裂？
- 哪些节点整体上离其他节点更近，适合作为入口/缓存热点？

对应到中心性指标：

1. Degree Centrality：连接数多不多（本地重要性）
2. Betweenness Centrality：是否位于大量最短路径中（桥梁重要性）
3. Closeness Centrality：到全图平均距离是否更短（全局接近性）

现实里最大的坑不是“不会定义”，而是“算不动”：

- Degree 非常便宜，几乎所有系统都能实时支持
- Betweenness 精确计算很贵，通常只能离线或近似
- Closeness 需要大量最短路，图一大就难在线

## 核心概念

### 1) Degree Centrality

无向图中节点 `v` 的度中心性常写为：

```text
C_D(v) = deg(v) / (n - 1)
```

含义：节点局部连接活跃度。

### 2) Betweenness Centrality

```text
C_B(v) = Σ_{s≠v≠t} (σ_st(v) / σ_st)
```

- `σ_st`：从 `s` 到 `t` 的最短路径条数
- `σ_st(v)`：经过 `v` 的最短路径条数

含义：节点作为“通道/桥梁”的中介能力。

### 3) Closeness Centrality

```text
C_C(v) = (n - 1) / Σ_{u≠v} d(v, u)
```

含义：节点到全图其它节点整体有多近。

> 实务补充：不连通图常用 harmonic closeness，避免不可达导致分母异常。

---

## A — Algorithm（题目与算法）

### 题目还原（工程化版本）

给定图 `G=(V,E)`，计算每个节点的中心性分数并返回 Top-K 节点：

1. Degree 中心性
2. Betweenness 中心性（允许近似）
3. Closeness 中心性（或 harmonic 变体）

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| graph | 邻接表 | `graph[u] = [v1, v2, ...]`（无权） |
| k | int | 输出 Top-K 数量 |
| mode | str | `degree` / `betweenness` / `closeness` |
| 返回 | List[(node, score)] | 排序后的节点得分 |

### 示例 1（小图）

```text
A-B-C-D 以及 B-E

直觉：
- B 度数高 -> Degree 高
- B/C 位于多条最短路 -> Betweenness 高
- B/C 到其他点平均更近 -> Closeness 较高
```

### 示例 2（桥接节点）

```text
两个团簇通过 X 相连

X 的 Betweenness 通常极高，即使 Degree 不一定最高
```

---

## 思路推导（从朴素到工程）

### 朴素做法

- 对每对节点都求最短路，再统计经过节点次数
- 复杂度极高，大图不可用

### 关键观察

1. Degree 只看局部邻接，复杂度接近线性
2. Betweenness 可以用 Brandes 算法显著优化，但仍然偏贵
3. Closeness 本质上要做多源最短路，图大时成本高

### 工程决策

- 在线：优先 Degree，必要时加采样近似 Betweenness
- 离线批处理：可做更完整 Betweenness / Closeness
- 大图：统一做 Top-K + 采样 + 分层缓存

---

## C — Concepts（核心思想）

### 方法归类

- Degree：局部统计
- Betweenness：全局最短路依赖分摊
- Closeness：全局距离聚合

### 复杂度直觉（无权图）

| 指标 | 常见算法 | 粗略复杂度 |
| --- | --- | --- |
| Degree | 遍历邻接表 | `O(V+E)` |
| Betweenness | Brandes | `O(VE)` |
| Closeness | 对每点做 BFS | `O(V(V+E))` |

### 现实结论（重点）

> 大部分线上系统只稳定支持 **Degree + 近似 Betweenness**。  
> Closeness 常放离线或仅在小子图计算。

原因很直接：

- Degree 成本低、解释性强、增量更新容易
- Betweenness 精确版太贵，近似可控
- Closeness 对全图连通性和图规模敏感，在线 SLA 难保证

---

## 实践指南 / 步骤

### 步骤 1：先定义业务问题

- 想找“连接多”的节点：Degree
- 想找“关键桥梁”：Betweenness
- 想找“全局接近中心”：Closeness

### 步骤 2：选择在线 or 离线

- 在线服务：Degree + 近似 Betweenness
- 离线报表：补齐 Closeness / 精细 Betweenness

### 步骤 3：可运行 Python 基础实现

```python
from collections import deque
import random


def degree_centrality(graph):
    n = max(len(graph), 1)
    return {u: len(graph.get(u, [])) / max(n - 1, 1) for u in graph}


def bfs_dist(graph, s):
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def closeness_centrality(graph):
    n = len(graph)
    cc = {}
    for u in graph:
        d = bfs_dist(graph, u)
        if len(d) <= 1:
            cc[u] = 0.0
            continue
        s = sum(d.values())
        cc[u] = (len(d) - 1) / s if s > 0 else 0.0
        # 可按业务改成 harmonic closeness
    return cc


def approx_betweenness_by_sampling(graph, samples=8, seed=0):
    random.seed(seed)
    nodes = list(graph.keys())
    if not nodes:
        return {}

    score = {u: 0.0 for u in nodes}
    sample_sources = random.sample(nodes, min(samples, len(nodes)))

    for s in sample_sources:
        # 单源最短路 DAG + 依赖回传（Brandes 思路）
        stack = []
        pred = {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}
        dist = {v: -1 for v in nodes}

        sigma[s] = 1.0
        dist[s] = 0
        q = deque([s])

        while q:
            v = q.popleft()
            stack.append(v)
            for w in graph.get(v, []):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                score[w] += delta[w]

    # 采样归一化（近似）
    factor = len(nodes) / max(len(sample_sources), 1)
    return {u: score[u] * factor for u in nodes}


if __name__ == "__main__":
    g = {
        "A": ["B"],
        "B": ["A", "C", "E"],
        "C": ["B", "D"],
        "D": ["C"],
        "E": ["B"],
    }

    print("degree", degree_centrality(g))
    print("closeness", closeness_centrality(g))
    print("approx_betweenness", approx_betweenness_by_sampling(g, samples=3, seed=42))
```

---

## E — Engineering（工程应用）

### 场景 1：反欺诈“枢纽账户”识别（Degree）

**背景**：资金关系图里，高度连接账户往往是中转中心。  
**为什么适用**：Degree 计算快，适合在线风控特征。

```python
# online feature: out-degree / in-degree
risk_score = out_degree * 0.6 + in_degree * 0.4
```

### 场景 2：关键桥梁节点预警（近似 Betweenness）

**背景**：社交/交易图中某些节点是群体之间“唯一通道”。  
**为什么适用**：Betweenness 能发现桥梁，但精确版太贵，采样近似更可落地。

```go
// pseudo-go style: run sampled Brandes in batch job
// 1) sample K sources
// 2) accumulate dependency scores
// 3) write top-k bridge nodes to Redis/OLAP
```

### 场景 3：关系解释路径入口筛选（Closeness）

**背景**：解释系统希望优先从“整体更接近核心区域”的点展开路径展示。  
**为什么适用**：Closeness 能刻画“平均距离短”的节点。

```javascript
// 用离线 closeness 排名前 N 作为解释入口候选
const candidates = centralityRank.slice(0, N);
```

---

## R — Reflection（反思与深入）

### 精确 vs 近似

| 指标 | 精确成本 | 近似策略 | 工程建议 |
| --- | --- | --- | --- |
| Degree | 低 | 不需要 | 在线直接算 |
| Betweenness | 高 | 采样源点、Top-K 估计 | 在线读离线结果/批量更新 |
| Closeness | 中-高 | 子图计算、harmonic 变体 | 多用于离线分析 |

### 常见错误思路

1. 把 Betweenness 当在线实时指标全量计算
2. 在大规模不连通图直接用标准 Closeness，不做变体处理
3. 忽视图有向/无向差异，导致指标解释偏差

### 为什么“Degree + 近似 Betweenness”最常见

- 成本可控：能满足线上 SLA
- 解释性强：产品和业务容易理解
- 可演进：先上线可用版，再补离线精细指标

---

## 解释与原理（为什么这么做）

中心性的工程本质是“用可接受成本，提取稳定可解释的重要性信号”。

- Degree 给你局部活跃度
- Betweenness 给你桥梁控制力
- Closeness 给你全局接近性

现实系统中，不是“哪个指标最优”，而是“哪个指标在当前规模与时延预算下可持续”。

---

## 常见问题与注意事项

1. **有向图和无向图能混用同一公式吗？**  
   可以共享思路，但统计口径不同（入度/出度、最短路方向）。

2. **Betweenness 一定要精确吗？**  
   不一定。很多场景近似排序就够用，尤其是只要 Top-K。

3. **Closeness 在不连通图怎么处理？**  
   推荐 harmonic closeness 或限制在连通子图内计算。

4. **是否需要实时更新中心性？**  
   多数系统采用“离线批更新 + 在线缓存”，仅 Degree 可做轻量实时增量。

---

## 最佳实践与建议

- 把中心性计算拆成两层：离线主计算 + 在线特征服务
- 先服务业务问题，再选指标，不要“指标先行”
- 对 Betweenness 设预算：采样数、窗口周期、只产出 Top-K
- 对大图先做连通分量切分，避免全图无差别计算

---

## S — Summary（总结）

### 核心收获

- Degree、Betweenness、Closeness 分别对应本地连接、桥梁中介、全局接近三类重要性
- Betweenness 工程上昂贵，精确全量通常不适合在线
- 大多数系统的务实组合是：Degree + 近似 Betweenness
- Closeness 更适合离线分析或小子图计算
- 指标选型必须服从规模、时延和可解释性约束

### 推荐延伸阅读

- Ulrik Brandes (2001): A Faster Algorithm for Betweenness Centrality
- NetworkX centrality 文档（快速实验）
- 图数据库中的 GDS 中心性算子设计（离线批计算实践）

---

## 元信息

- **阅读时长**：12~16 分钟
- **标签**：图论、中心性、Degree、Betweenness、Closeness
- **SEO 关键词**：图中心性, Degree Centrality, Betweenness, Closeness, 近似 Betweenness
- **元描述**：图中心性三件套工程指南：定义、复杂度、近似与上线策略，重点解释为何多数系统只支持 Degree 和近似 Betweenness。

---

## 行动号召（CTA）

建议你下一步直接做两件事：

1. 先上线 `Degree + Top-K`，验证业务可解释性
2. 再做“采样 Betweenness”离线任务，对比排序稳定性

如果你愿意，我可以下一篇直接写“PageRank + 社区发现（Louvain）”的工程版接续文。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
# Degree centrality (unweighted graph)
def degree_centrality(graph):
    n = max(len(graph), 1)
    return {u: len(graph.get(u, [])) / max(n - 1, 1) for u in graph}
```

```c
/* degree centrality for adjacency matrix (undirected) */
#include <stdio.h>
#define N 5

int main(void) {
    int g[N][N] = {
        {0,1,0,0,1},
        {1,0,1,0,0},
        {0,1,0,1,0},
        {0,0,1,0,0},
        {1,0,0,0,0}
    };
    for (int i = 0; i < N; i++) {
        int deg = 0;
        for (int j = 0; j < N; j++) deg += g[i][j];
        double cd = (double)deg / (N - 1);
        printf("node %d degree_c=%.3f\n", i, cd);
    }
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<vector<int>> g = {
        {1,4}, {0,2}, {1,3}, {2}, {0}
    };
    int n = (int)g.size();
    for (int u = 0; u < n; ++u) {
        double c = (double)g[u].size() / max(n - 1, 1);
        cout << "node " << u << " degree_c=" << c << "\n";
    }
}
```

```go
package main

import "fmt"

func main() {
	g := map[int][]int{0: {1,4}, 1: {0,2}, 2: {1,3}, 3: {2}, 4: {0}}
	n := len(g)
	for u, nbrs := range g {
		cd := float64(len(nbrs)) / float64(n-1)
		fmt.Printf("node %d degree_c=%.3f\n", u, cd)
	}
}
```

```rust
use std::collections::HashMap;

fn main() {
    let mut g: HashMap<i32, Vec<i32>> = HashMap::new();
    g.insert(0, vec![1, 4]);
    g.insert(1, vec![0, 2]);
    g.insert(2, vec![1, 3]);
    g.insert(3, vec![2]);
    g.insert(4, vec![0]);

    let n = g.len() as f64;
    for (u, nbrs) in &g {
        let cd = nbrs.len() as f64 / (n - 1.0);
        println!("node {} degree_c={:.3}", u, cd);
    }
}
```

```javascript
const g = new Map([
  ["A", ["B", "E"]],
  ["B", ["A", "C"]],
  ["C", ["B", "D"]],
  ["D", ["C"]],
  ["E", ["A"]],
]);

const n = g.size;
for (const [u, nbrs] of g.entries()) {
  const cd = nbrs.length / (n - 1);
  console.log(u, "degree_c=", cd.toFixed(3));
}
```
