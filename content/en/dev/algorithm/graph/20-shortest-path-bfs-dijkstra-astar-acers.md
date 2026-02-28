---
title: "Shortest Path Core Trio: BFS, Dijkstra, and A* ACERS Engineering Breakdown"
date: 2026-02-09T09:48:00+08:00
draft: false
categories: ["Logic and Algorithms"]
tags: ["Graph Theory", "shortest path", "BFS", "Dijkstra", "A*", "bidirectional search", "engineering practice"]
description: "A full engineering walkthrough of the shortest-path core trio: BFS for unweighted graphs, Dijkstra for non-negative weights, and heuristic A*. Includes multi-source BFS, bidirectional search, path pruning, and runnable multi-language templates."
keywords: ["shortest path", "BFS", "Dijkstra", "A*", "bidirectional BFS", "bidirectional Dijkstra", "multi-source BFS", "max depth"]
---

> **Subtitle / Abstract**  
> Shortest path is not one question. It is an engineering skill set: choose the right algorithm by graph conditions. This ACERS article breaks down **BFS (unweighted) / Dijkstra (non-negative weights) / A* (heuristic)** and gives optimization templates you actually use in relationship graphs, recommendation paths, and path explanations.

- **Estimated reading time**: 14-18 minutes  
- **Tags**: `Graph Theory`, `shortest path`, `BFS`, `Dijkstra`, `A*`  
- **SEO keywords**: shortest path, BFS, Dijkstra, A*, bidirectional search, multi-source BFS  
- **Meta description**: Engineering guide to the shortest-path core trio: algorithm boundaries, complexity, runnable code, optimization strategies, and practical scenarios.

---

## Target Audience

- Learners reinforcing graph fundamentals who want reusable engineering templates
- Backend/algorithm engineers working on social links, recommendation paths, or graph-query explanations
- Developers who know BFS, Dijkstra, and A* by name but still struggle with robust selection and optimization

## Background / Motivation

Shortest-path problems are common in:

- Minimal relationship chains in social networks (how many hops apart)
- Minimum-cost paths in recommendation systems (multi-objective trade-offs)
- "Why this was recommended" path displays in explainability systems

The most common production mistake is forcing one algorithm onto every scenario:

1. Running BFS on weighted graphs (wrong result, no explicit error)
2. Running Dijkstra on negative edges (unreliable result)
3. Using A* with a poor heuristic (performance degrades to Dijkstra)

In essence, shortest-path solutions should start with **graph-condition classification**, then algorithm selection.

## Core Concepts

| Algorithm | Suitable Graph | Optimality Condition | Typical Complexity | Keywords |
| --- | --- | --- | --- | --- |
| BFS | Unweighted / equal-weight graph | First arrival by layer gives minimum edge count | `O(V+E)` | queue, level |
| Dijkstra | Non-negative weighted graph | Node popped from heap already has optimal distance | `O((V+E)logV)` | relaxation, min-heap |
| A* | Non-negative weighted graph + heuristic | `h(n)` is admissible (never overestimates) | Worst case same as Dijkstra, usually faster | `f=g+h` |

Key formulas:

- **Dijkstra relaxation**: update when `dist[v] > dist[u] + w(u,v)`
- **A* evaluation function**: `f(n) = g(n) + h(n)`

Where:
- `g(n)` is known cost from start to `n`
- `h(n)` is heuristic estimated cost from `n` to target

---

## A - Algorithm (Problem and Algorithm)

### Unified Problem Model

Given graph `G=(V,E)`, start node `s`, and target node `t`, find both shortest path length and path from `s` to `t`.  
The graph may be unweighted, or weighted with non-negative edge weights.

### Input and Output

| Name | Type | Description |
| --- | --- | --- |
| graph | Adjacency list | Graph structure; `graph[u]` is neighbors or `(neighbor, weight)` |
| s | Node ID | Start node |
| t | Node ID | Target node |
| Return | Distance + path | Return `INF/null` or empty path when unreachable |

### Example 1 (Unweighted graph)

```text
A -> B -> D
A -> C -> D

minimum edge count from A to D = 2
valid path: A-B-D or A-C-D
```

### Example 2 (Non-negative weighted graph)

```text
A -> B (2)
A -> C (5)
B -> C (1)
B -> D (4)
C -> D (1)

minimum cost from A to D = 4
path: A-B-C-D
```

---

## Derivation (From naive to production-ready)

### Naive idea: enumerate all paths

- Use DFS to enumerate all `s -> t` paths, then choose minimum
- In cyclic graphs, dedup logic is complex and path count can be exponential

Conclusion: not practical except on tiny graphs.

### Key Observation 1: if edge weights are equal, layer index is cost

- Shortest path reduces to "fewest edges"
- BFS expands by layers; first reach of target is optimal

### Key Observation 2: with non-negative weights, greedy shortest-prefix expansion works

- Dijkstra pops the node with current minimum `dist`
- With non-negative weights, popped nodes cannot be improved later

### Key Observation 3: if you can estimate how far a node is from target, search shrinks

- A* adds heuristic `h(n)` on top of Dijkstra
- Search is guided toward target, reducing irrelevant expansion

---

## C - Concepts (Core Ideas)

### Method Categories

- **BFS**: layered traversal + minimum hop count
- **Dijkstra**: shortest-path tree + relaxation + min-heap
- **A***: shortest path + heuristic best-first search

### Relationship Among the Three

1. Dijkstra is A* with `h(n)=0`
2. BFS is Dijkstra when all edge weights are 1
3. A* performance depends heavily on heuristic quality:
   - Too weak: degrades to Dijkstra
   - Too aggressive and overestimating: may lose optimality

### Engineering Selection Matrix

| Problem Feature | Preferred Algorithm | Notes |
| --- | --- | --- |
| Unweighted graph / minimum hops | BFS | Relationship chains, k-hop search |
| Minimum cost with non-negative weights | Dijkstra | Stable default for backend services |
| Non-negative weights + usable heuristic | A* | Road networks, spatial graphs, explanation paths |
| Negative edges present | Bellman-Ford/Johnson | Do not use Dijkstra/A* |

---

## Practical Guide / Steps

### Step 1: classify graph conditions first

1. Unweighted or equal-weight? Yes -> BFS
2. Any negative edges? Yes -> cannot use Dijkstra/A*
3. Usable heuristic available? Yes -> prefer A*

### Step 2: unify path reconstruction interface

Maintain `parent` mapping: `parent[v] = u`, then backtrack from `t` to `s`.

### Step 3: implement runnable template (Python)

```python
from collections import deque
import heapq
from math import inf


def reconstruct_path(parent, s, t):
    if t not in parent and s != t:
        return []
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs_shortest_path(graph, s, t, max_depth=None):
    """graph[u] = [v1, v2, ...]"""
    q = deque([(s, 0)])
    parent = {s: s}
    visited = {s}

    while q:
        u, d = q.popleft()
        if u == t:
            return d, reconstruct_path(parent, s, t)
        if max_depth is not None and d >= max_depth:
            continue
        for v in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append((v, d + 1))

    return inf, []


def dijkstra_shortest_path(graph, s, t, max_cost=None):
    """graph[u] = [(v, w), ...], w >= 0"""
    dist = {s: 0.0}
    parent = {s: s}
    pq = [(0.0, s)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u, inf):
            continue
        if max_cost is not None and du > max_cost:
            continue
        if u == t:
            return du, reconstruct_path(parent, s, t)
        for v, w in graph.get(u, []):
            nd = du + w
            if nd < dist.get(v, inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return inf, []


def astar_shortest_path(graph, s, t, h):
    """h(u) is admissible heuristic estimate from u to t"""
    g = {s: 0.0}
    parent = {s: s}
    pq = [(h(s), s)]  # (f, node)

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return g[u], reconstruct_path(parent, s, t)

        for v, w in graph.get(u, []):
            ng = g[u] + w
            if ng < g.get(v, inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), v))

    return inf, []


if __name__ == "__main__":
    unweighted = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    print(bfs_shortest_path(unweighted, "A", "D"))  # (2, ['A', 'B', 'D']) or C path

    weighted = {
        "A": [("B", 2), ("C", 5)],
        "B": [("C", 1), ("D", 4)],
        "C": [("D", 1)],
        "D": [],
    }
    print(dijkstra_shortest_path(weighted, "A", "D"))  # (4.0, ['A', 'B', 'C', 'D'])

    heuristic = {"A": 3, "B": 2, "C": 1, "D": 0}
    print(astar_shortest_path(weighted, "A", "D", lambda x: heuristic[x]))
```

---

## E - Engineering (Production Applications)

### Scenario 1: shortest social link chain (BFS + bidirectional BFS)

**Background**: given user A and user B, find a shortest relationship chain for explainability display.  
**Why it fits**: this is an unweighted graph and objective is minimum hops; BFS matches naturally, and bidirectional BFS further reduces expansions.

```python
from collections import deque


def bidirectional_bfs(graph, s, t, max_depth=6):
    if s == t:
        return 0

    qa, qb = deque([s]), deque([t])
    da, db = {s: 0}, {t: 0}

    while qa and qb:
        # expand smaller frontier first
        if len(qa) <= len(qb):
            q, dcur, dother = qa, da, db
        else:
            q, dcur, dother = qb, db, da

        u = q.popleft()
        if dcur[u] >= max_depth:
            continue

        for v in graph.get(u, []):
            if v in dcur:
                continue
            dcur[v] = dcur[u] + 1
            if v in dother:
                return dcur[v] + dother[v]
            q.append(v)

    return -1
```

### Scenario 2: recommendation path (Dijkstra)

**Background**: edge weights represent "cost" (latency, risk, penalty); we need the lowest total cost path.  
**Why it fits**: with non-negative weights, Dijkstra is stable and straightforward to service-ify.

```go
package main

import (
	"container/heap"
	"fmt"
)

type Edge struct{ To string; W float64 }

type Item struct{ D float64; U string }
type PQ []Item
func (p PQ) Len() int { return len(p) }
func (p PQ) Less(i, j int) bool { return p[i].D < p[j].D }
func (p PQ) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p *PQ) Push(x interface{}) { *p = append(*p, x.(Item)) }
func (p *PQ) Pop() interface{} { old := *p; x := old[len(old)-1]; *p = old[:len(old)-1]; return x }

func dijkstra(g map[string][]Edge, s, t string) float64 {
	const INF = 1e18
	dist := map[string]float64{s: 0}
	pq := &PQ{{0, s}}
	heap.Init(pq)

	for pq.Len() > 0 {
		it := heap.Pop(pq).(Item)
		if it.D != dist[it.U] { continue }
		if it.U == t { return it.D }
		for _, e := range g[it.U] {
			nd := it.D + e.W
			if d, ok := dist[e.To]; !ok || nd < d {
				dist[e.To] = nd
				heap.Push(pq, Item{nd, e.To})
			}
		}
	}
	return INF
}

func main() {
	g := map[string][]Edge{
		"A": {{"B", 2}, {"C", 5}},
		"B": {{"C", 1}, {"D", 4}},
		"C": {{"D", 1}},
	}
	fmt.Println(dijkstra(g, "A", "D")) // 4
}
```

### Scenario 3: explanation path in relationship graphs (A* + path pruning)

**Background**: to show users "why X was recommended to Y," we need explainable paths with controlled query latency.  
**Why it fits**: A* can use domain priors (similarity distance) to cut expansions; combine with `maxDepth` pruning to control cost.

```javascript
function astar(graph, s, t, h, maxDepth = 6) {
  const g = new Map([[s, 0]]);
  const pq = [[h(s), 0, s]]; // [f, depth, node]

  while (pq.length) {
    pq.sort((a, b) => a[0] - b[0]);
    const [f, depth, u] = pq.shift();
    if (u === t) return g.get(u);
    if (depth >= maxDepth) continue;

    for (const [v, w] of (graph.get(u) || [])) {
      const ng = g.get(u) + w;
      if (!g.has(v) || ng < g.get(v)) {
        g.set(v, ng);
        pq.push([ng + h(v), depth + 1, v]);
      }
    }
  }
  return Infinity;
}
```

---

## Optimization Essentials (Must-know)

### 1) Multi-source BFS

Queue multiple sources at once and run one unified BFS.
Useful for tasks like "distance to nearest interest point" or "batch infection radius spread."

```python
from collections import deque

def multi_source_bfs(graph, sources):
    q = deque(sources)
    dist = {s: 0 for s in sources}
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

### 2) Bidirectional BFS / bidirectional Dijkstra

- Bidirectional BFS: usually cuts search depth significantly on unweighted graphs
- Bidirectional Dijkstra: can reduce state expansion on non-negative weighted graphs, with higher implementation complexity

### 3) Path pruning (`max_depth` / `max_cost`)

In online services, guarantee usable latency first, then optimize coverage:

- BFS: `max_depth`
- Dijkstra: `max_cost`
- A*: `max_depth + heuristic`

### 4) visited bitmap / bloom

- **bitmap**: exact and memory-controllable (prefer when node IDs can be mapped to contiguous integers)
- **bloom**: more space-efficient but has false positives; suitable for recall-oriented prefiltering, not for strict-optimality main decision chains

---

## R - Reflection (Deep Dive)

### Complexity Comparison

| Algorithm | Time Complexity | Space Complexity |
| --- | --- | --- |
| BFS | `O(V+E)` | `O(V)` |
| Dijkstra (heap) | `O((V+E)logV)` | `O(V)` |
| A* | Worst case same as Dijkstra | `O(V)` |

### Alternatives and Trade-offs

| Approach | Conditions | Cost | When to Choose |
| --- | --- | --- | --- |
| Bellman-Ford | Negative weights allowed | `O(VE)` | Must support negative edges |
| Floyd-Warshall | All-pairs shortest paths | `O(V^3)` | Small offline graph, full-pair queries |
| Core trio in this article | High-frequency online queries | Low to medium | Most online path problems in engineering |

### Common Wrong Approaches

1. Using BFS on weighted graphs
2. Ignoring negative-edge checks and running Dijkstra directly
3. Using an unreasonable heuristic in A*, causing heavy invalid expansion
4. Marking visited too early (can miss better paths in weighted graphs)

### Why This Set Is the Most Practical in Engineering

- Covers the most common graph conditions (unweighted + non-negative weights + heuristics)
- Supports unified interface abstraction so business layers only care about a "path query service"
- Composes naturally with bidirectional search and pruning for SLA control

---

## Explanation and Principles (Why this works)

You can view these three methods as one evolution line:

1. BFS: expand by layers for equal edge-cost settings
2. Dijkstra: expand by minimum known true cost for non-negative weighted settings
3. A*: add target-oriented heuristics to Dijkstra to reduce irrelevant expansion

The core difference is not coding style, but **what determines expansion order**:

- BFS uses level
- Dijkstra uses `g`
- A* uses `g+h`

---

## FAQ and Notes

1. **What if the graph is disconnected?**  
   Return unreachable (`INF` or empty path). Do not force path backtracking.

2. **When should visited be finalized in Dijkstra?**  
   Recommended: finalize when the node is popped and confirmed as the current optimal `dist`.

3. **How to choose `h(n)` in A*?**  
   Road networks often use Manhattan/Euclidean distance. Recommendation graphs can use embedding-distance lower bounds. Avoid systematic overestimation.

4. **When should I use bidirectional search?**  
   When both endpoints are known and the graph is large with high branching factor, benefits are usually significant.

---

## Best Practices and Recommendations

- Validate graph conditions first (unweighted? negative weights? usable heuristic?)
- Make path reconstruction, pruning, and logging metrics a shared middleware layer
- In online services, prioritize tail-latency guarantees before maximum coverage
- On large graphs, prefer adjacency lists + ID compression + bitmap visited

---

## S - Summary

### Key Takeaways

- BFS, Dijkstra, and A* are the shortest-path engineering core trio, and the key is condition-based selection
- Use BFS for unweighted graphs, Dijkstra for non-negative weighted graphs, and A* when reliable heuristics exist
- Multi-source, bidirectional search, and pruning are not optional polish; they are primary tools for online performance/cost control
- A* performance ceiling depends on heuristic quality; weak heuristics degrade performance
- A unified path-service interface significantly lowers switching cost among algorithms

### Recommended Follow-up Reading

- LeetCode 127 (Word Ladder, bidirectional BFS)
- LeetCode 743 (Network Delay Time, Dijkstra)
- Classic A* paper: Hart, Nilsson, Raphael (1968)
- Negative-weight scenarios: Bellman-Ford / Johnson

---

## Metadata

- **Reading time**: 14-18 minutes
- **Tags**: Graph Theory, shortest path, BFS, Dijkstra, A*, bidirectional search
- **SEO keywords**: shortest path, BFS, Dijkstra, A*, bidirectional BFS, multi-source BFS
- **Meta description**: Engineering guide to the shortest-path core trio: algorithm boundaries, complexity, optimization strategies, and runnable code.

---

## Call To Action (CTA)

Recommended next steps using the same template:

1. Refactor your current graph-query API into a pluggable algorithm interface (switchable BFS/Dijkstra/A*)
2. Add online metrics: expanded node count, average path length, and P95 query latency

If you want, I can write the next article directly: "Engineering shortest paths on negative-weight graphs (Bellman-Ford/Johnson)."

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
# Dijkstra (non-negative weights), adjacency list
import heapq
from math import inf


def dijkstra(graph, s, t):
    dist = {s: 0.0}
    parent = {s: s}
    pq = [(0.0, s)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u, inf):
            continue
        if u == t:
            break
        for v, w in graph.get(u, []):
            nd = du + w
            if nd < dist.get(v, inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if t not in dist:
        return inf, []
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return dist[t], path
```

```c
// Dijkstra O(V^2) demo for dense/small graphs (non-negative weights)
#include <stdio.h>
#define N 5
#define INF 1000000000

int main(void) {
    int g[N][N] = {
        {0, 2, 5, 0, 0},
        {0, 0, 1, 4, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    int s = 0, t = 3;
    int dist[N], vis[N] = {0};

    for (int i = 0; i < N; i++) dist[i] = INF;
    dist[s] = 0;

    for (int i = 0; i < N; i++) {
        int u = -1;
        for (int j = 0; j < N; j++)
            if (!vis[j] && (u == -1 || dist[j] < dist[u])) u = j;
        if (u == -1 || dist[u] == INF) break;
        vis[u] = 1;
        for (int v = 0; v < N; v++) {
            if (g[u][v] > 0 && dist[v] > dist[u] + g[u][v])
                dist[v] = dist[u] + g[u][v];
        }
    }

    if (dist[t] >= INF) printf("unreachable\n");
    else printf("dist=%d\n", dist[t]);
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;

pair<long long, vector<int>> dijkstra(int n, vector<vector<pair<int,int>>>& g, int s, int t) {
    const long long INF = (1LL<<60);
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;

    dist[s] = 0;
    parent[s] = s;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (du != dist[u]) continue;
        if (u == t) break;
        for (auto [v, w] : g[u]) {
            long long nd = du + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                parent[v] = u;
                pq.push({nd, v});
            }
        }
    }

    if (dist[t] == INF) return {INF, {}};
    vector<int> path;
    for (int x = t; x != s; x = parent[x]) path.push_back(x);
    path.push_back(s);
    reverse(path.begin(), path.end());
    return {dist[t], path};
}
```

```go
package main

import (
	"container/heap"
	"fmt"
)

type Edge struct{ To int; W int64 }
type Item struct{ D int64; U int }
type PQ []Item
func (p PQ) Len() int { return len(p) }
func (p PQ) Less(i, j int) bool { return p[i].D < p[j].D }
func (p PQ) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p *PQ) Push(x interface{}) { *p = append(*p, x.(Item)) }
func (p *PQ) Pop() interface{} { old := *p; x := old[len(old)-1]; *p = old[:len(old)-1]; return x }

func dijkstra(g [][]Edge, s, t int) int64 {
	const INF int64 = 1<<60
	dist := make([]int64, len(g))
	for i := range dist { dist[i] = INF }
	dist[s] = 0

	pq := &PQ{{0, s}}
	heap.Init(pq)

	for pq.Len() > 0 {
		it := heap.Pop(pq).(Item)
		if it.D != dist[it.U] { continue }
		if it.U == t { return it.D }
		for _, e := range g[it.U] {
			nd := it.D + e.W
			if nd < dist[e.To] {
				dist[e.To] = nd
				heap.Push(pq, Item{nd, e.To})
			}
		}
	}
	return INF
}

func main() {
	g := make([][]Edge, 4)
	g[0] = []Edge{{1, 2}, {2, 5}}
	g[1] = []Edge{{2, 1}, {3, 4}}
	g[2] = []Edge{{3, 1}}
	fmt.Println(dijkstra(g, 0, 3)) // 4
}
```

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

fn dijkstra(graph: &Vec<Vec<(usize, i64)>>, s: usize, t: usize) -> i64 {
    let inf = i64::MAX / 4;
    let mut dist = vec![inf; graph.len()];
    let mut pq = BinaryHeap::new();

    dist[s] = 0;
    pq.push((Reverse(0_i64), s));

    while let Some((Reverse(du), u)) = pq.pop() {
        if du != dist[u] { continue; }
        if u == t { return du; }
        for &(v, w) in &graph[u] {
            let nd = du + w;
            if nd < dist[v] {
                dist[v] = nd;
                pq.push((Reverse(nd), v));
            }
        }
    }
    inf
}

fn main() {
    let g = vec![
        vec![(1,2),(2,5)],
        vec![(2,1),(3,4)],
        vec![(3,1)],
        vec![]
    ];
    println!("{}", dijkstra(&g, 0, 3)); // 4
}
```

```javascript
// BFS shortest hops in unweighted graph
function bfsShortest(graph, s, t) {
  const q = [[s, 0]];
  const seen = new Set([s]);

  while (q.length) {
    const [u, d] = q.shift();
    if (u === t) return d;
    for (const v of (graph.get(u) || [])) {
      if (!seen.has(v)) {
        seen.add(v);
        q.push([v, d + 1]);
      }
    }
  }
  return Infinity;
}

const g = new Map([
  ["A", ["B", "C"]],
  ["B", ["D"]],
  ["C", ["D"]],
  ["D", []],
]);
console.log(bfsShortest(g, "A", "D")); // 2
```
