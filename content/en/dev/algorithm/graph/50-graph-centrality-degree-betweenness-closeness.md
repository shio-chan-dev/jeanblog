---
title: "The Graph Centrality Trio: Degree, Betweenness, and Closeness - ACERS Engineering Analysis"
date: 2026-02-09T09:56:11+08:00
draft: false
categories: ["Logic and Algorithms"]
tags: ["Graph Theory", "Centrality", "Degree", "Betweenness", "Closeness", "Graph Database", "Engineering Practice"]
description: "A systematic guide to the three core graph centrality metrics: Degree, Betweenness, and Closeness. The key engineering takeaway: most production systems prioritize Degree and approximate Betweenness, with clear complexity and rollout tradeoffs."
keywords: ["graph centrality", "degree centrality", "betweenness centrality", "closeness centrality", "approximate betweenness", "Brandes"]
---

> **Subtitle / Abstract**  
> Centrality is not just a paper concept. In graph systems, it is a practical node-importance ranking engine. This article follows the ACERS structure to explain **Degree / Betweenness / Closeness** and gives one pragmatic conclusion: **most online systems reliably support only Degree + approximate Betweenness**.

- **Estimated reading time**: 12-16 minutes  
- **Tags**: `Graph Theory`, `Centrality`, `Degree`, `Betweenness`, `Closeness`  
- **SEO keywords**: graph centrality, Degree Centrality, Betweenness, Closeness, approximate Betweenness  
- **Meta description**: Engineering guide to graph centrality: definitions, complexity, approximation methods, and production strategies, with runnable code.  

---

## Target Audience

- Engineers working on relationship graph analysis, knowledge graphs, or graph-database query optimization
- Developers who need to turn "node importance" from concept into production metric
- Practitioners who want to understand why Betweenness is expensive in production and how to approximate it

## Background / Motivation

In graph systems, you will eventually face questions like these:

- Which nodes are social influencers or transaction hubs?
- Which nodes are key bridges whose removal significantly fragments the graph?
- Which nodes are globally closer to others and better suited as entry points or cache hotspots?

These map directly to centrality metrics:

1. Degree Centrality: how many connections a node has (local importance)
2. Betweenness Centrality: whether a node lies on many shortest paths (bridge importance)
3. Closeness Centrality: whether a node has shorter average distance to the full graph (global proximity)

In practice, the biggest challenge is not definition - it is compute cost:

- Degree is very cheap and almost always supports real-time use
- Exact Betweenness is expensive and is usually offline or approximate
- Closeness requires many shortest-path computations and quickly becomes hard to run online on large graphs

## Core Concepts

### 1) Degree Centrality

For node `v` in an undirected graph, Degree centrality is commonly written as:

```text
C_D(v) = deg(v) / (n - 1)
```

Meaning: local connectivity activity of the node.

### 2) Betweenness Centrality

```text
C_B(v) = Σ_{s≠v≠t} (σ_st(v) / σ_st)
```

- `σ_st`: number of shortest paths from `s` to `t`
- `σ_st(v)`: number of those shortest paths that pass through `v`

Meaning: mediation power of the node as a channel or bridge.

### 3) Closeness Centrality

```text
C_C(v) = (n - 1) / Σ_{u≠v} d(v, u)
```

Meaning: how close the node is, overall, to all other nodes in the graph.

> Practical note: disconnected graphs often use harmonic closeness to avoid denominator issues caused by unreachable nodes.

---

## A — Algorithm (Problem and Algorithm)

### Problem Restatement (Engineering Version)

Given graph `G=(V,E)`, compute centrality scores for each node and return Top-K nodes:

1. Degree centrality
2. Betweenness centrality (approximation allowed)
3. Closeness centrality (or harmonic variant)

### Input/Output

| Name | Type | Description |
| --- | --- | --- |
| graph | adjacency list | `graph[u] = [v1, v2, ...]` (unweighted) |
| k | int | number of Top-K outputs |
| mode | str | `degree` / `betweenness` / `closeness` |
| return | List[(node, score)] | sorted node scores |

### Example 1 (Small Graph)

```text
A-B-C-D and B-E

Intuition:
- B has high degree -> high Degree
- B/C lie on many shortest paths -> high Betweenness
- B/C are closer on average to other nodes -> higher Closeness
```

### Example 2 (Bridge Node)

```text
Two clusters are connected through X

X usually has very high Betweenness, even if its Degree is not the highest
```

---

## Reasoning Path (From Naive to Production)

### Naive Approach

- Compute shortest paths for every pair of nodes, then count how often each node is traversed
- Complexity is too high for large graphs

### Key Observations

1. Degree uses only local adjacency and is near linear complexity
2. Betweenness can be significantly optimized with Brandes, but remains relatively expensive
3. Closeness fundamentally needs many-source shortest paths, so cost rises rapidly with graph size

### Engineering Decisions

- Online: prioritize Degree, add sampled approximate Betweenness when necessary
- Offline batch: compute fuller Betweenness / Closeness
- Large graphs: use Top-K + sampling + layered cache together

---

## C — Concepts (Core Ideas)

### Method Categories

- Degree: local statistics
- Betweenness: dependency accumulation over global shortest paths
- Closeness: global distance aggregation

### Complexity Intuition (Unweighted Graph)

| Metric | Common Algorithm | Rough Complexity |
| --- | --- | --- |
| Degree | traverse adjacency list | `O(V+E)` |
| Betweenness | Brandes | `O(VE)` |
| Closeness | run BFS from every node | `O(V(V+E))` |

### Practical Conclusion (Key Point)

> Most online systems reliably support only **Degree + approximate Betweenness**.  
> Closeness is often moved offline or computed only on small subgraphs.

The reasons are straightforward:

- Degree is low-cost, highly interpretable, and easy to update incrementally
- Exact Betweenness is too expensive, while approximation is controllable
- Closeness is sensitive to connectivity and graph size, making online SLA hard to guarantee

---

## Practical Guide / Steps

### Step 1: Define the Business Question First

- Looking for highly connected nodes: Degree
- Looking for critical bridges: Betweenness
- Looking for global proximity centers: Closeness

### Step 2: Choose Online or Offline

- Online services: Degree + approximate Betweenness
- Offline reporting: add Closeness / refined Betweenness

### Step 3: Runnable Python Baseline

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
        # Can be switched to harmonic closeness depending on the use case
    return cc


def approx_betweenness_by_sampling(graph, samples=8, seed=0):
    random.seed(seed)
    nodes = list(graph.keys())
    if not nodes:
        return {}

    score = {u: 0.0 for u in nodes}
    sample_sources = random.sample(nodes, min(samples, len(nodes)))

    for s in sample_sources:
        # Single-source shortest-path DAG + dependency back-propagation (Brandes style)
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

    # Sampling normalization (approximation)
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

## E — Engineering (Engineering Applications)

### Scenario 1: Anti-Fraud Hub Account Detection (Degree)

**Background**: in money-transfer graphs, highly connected accounts are often transfer hubs.  
**Why this fits**: Degree is fast to compute and suitable as an online risk-control feature.

```python
# online feature: out-degree / in-degree
risk_score = out_degree * 0.6 + in_degree * 0.4
```

### Scenario 2: Critical Bridge Node Alerts (Approximate Betweenness)

**Background**: in social or transaction graphs, some nodes are the only channel between communities.  
**Why this fits**: Betweenness finds bridges, but exact computation is expensive; sampled approximation is easier to deploy.

```go
// pseudo-go style: run sampled Brandes in batch job
// 1) sample K sources
// 2) accumulate dependency scores
// 3) write top-k bridge nodes to Redis/OLAP
```

### Scenario 3: Entry Selection for Path Explanation (Closeness)

**Background**: explanation systems may prefer to start path rendering from nodes that are globally closer to core regions.  
**Why this fits**: Closeness captures nodes with shorter average distances.

```javascript
// Use top-N offline closeness nodes as explanation entry candidates
const candidates = centralityRank.slice(0, N);
```

---

## R — Reflection (Reflection and Deeper Analysis)

### Exact vs Approximate

| Metric | Exact Cost | Approximation Strategy | Engineering Recommendation |
| --- | --- | --- | --- |
| Degree | Low | Not required | Compute online directly |
| Betweenness | High | Source-node sampling, Top-K estimation | Read offline results online / batch refresh |
| Closeness | Medium-high | Subgraph computation, harmonic variant | Mostly used for offline analysis |

### Common Wrong Approaches

1. Treating Betweenness as a fully real-time online metric
2. Applying standard Closeness directly to large disconnected graphs without variant handling
3. Ignoring directed vs undirected differences, causing interpretation errors

### Why "Degree + Approximate Betweenness" Is Most Common

- Controllable cost: can satisfy online SLA
- Strong interpretability: easy for product and business teams to reason about
- Easy evolution path: launch a usable version first, then add refined offline metrics

---

## Explanation and Principles (Why This Works)

The engineering essence of centrality is extracting stable, interpretable importance signals at acceptable cost.

- Degree gives local activity
- Betweenness gives bridge-control power
- Closeness gives global proximity

In real systems, the question is not "which metric is theoretically best," but "which metric is sustainable under current scale and latency budget."

---

## Frequently Asked Questions and Notes

1. **Can directed and undirected graphs use the same formulas?**  
   The high-level idea is shared, but counting conventions differ (in-degree/out-degree, shortest-path direction).

2. **Does Betweenness have to be exact?**  
   Not necessarily. In many cases, approximate ranking is enough, especially for Top-K.

3. **How should Closeness be handled on disconnected graphs?**  
   Harmonic closeness is recommended, or restrict computation to connected subgraphs.

4. **Do centrality scores need real-time updates?**  
   Most systems use "offline batch refresh + online cache." Only Degree is usually feasible for lightweight real-time increment.

---

## Best Practices and Recommendations

- Split centrality into two layers: offline primary computation + online feature service
- Start from business questions, then pick metrics; avoid a "metric-first" mindset
- Put budgets on Betweenness: sample size, time window, Top-K-only outputs
- For large graphs, split by connected components first to avoid indiscriminate full-graph computation

---

## S — Summary (Summary)

### Core Takeaways

- Degree, Betweenness, and Closeness capture local connectivity, bridge mediation, and global proximity
- Betweenness is expensive in production; exact full computation is usually unsuitable online
- The pragmatic combination in most systems is Degree + approximate Betweenness
- Closeness is better for offline analysis or small subgraphs
- Metric choice must obey constraints of scale, latency, and interpretability

### Recommended Further Reading

- Ulrik Brandes (2001): A Faster Algorithm for Betweenness Centrality
- NetworkX centrality documentation (quick experiments)
- GDS centrality operator design in graph databases (offline batch practice)

---

## Metadata

- **Reading time**: 12-16 minutes
- **Tags**: Graph Theory, Centrality, Degree, Betweenness, Closeness
- **SEO keywords**: graph centrality, Degree Centrality, Betweenness, Closeness, approximate Betweenness
- **Meta description**: Engineering guide to the graph centrality trio: definitions, complexity, approximation, and rollout strategy, with a focus on why most systems only support Degree and approximate Betweenness.

---

## Call To Action (CTA)

A practical next step is to do two things:

1. Launch `Degree + Top-K` first to validate business interpretability
2. Add an offline sampled-Betweenness job and compare ranking stability

If you want, I can write the next engineering continuation on "PageRank + Community Detection (Louvain)."

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

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
