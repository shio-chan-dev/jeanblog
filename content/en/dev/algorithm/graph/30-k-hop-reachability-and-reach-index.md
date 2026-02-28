---
title: "k-hop and Reachability Queries: BFS Limits, Reachability Indexes, and 2-hop Labeling ACERS Analysis"
subtitle: "From Online Search to Offline Indexing: Explainable Tradeoffs Across Latency, Memory, and Update Cost"
date: 2026-02-09T09:52:17+08:00
draft: false
summary: "This article walks through k-hop and reachability queries in practice: BFS+hop limits, transitive-closure tradeoffs, and engineering rollout paths for bitmap indexes and 2-hop labeling."
categories: ["Logic and Algorithms"]
tags: ["Graph", "BFS", "Reachability", "k-hop", "Transitive Closure", "2-hop labeling", "Bitmap Index"]
description: "A systematic engineering guide to k-hop and reachability queries: BFS+hop limits, why full transitive closure is usually not maintained, and practical bitmap/2-hop reach-index strategies with runnable multi-language code."
keywords: ["k-hop", "Reachability", "Transitive Closure", "BFS hop limit", "2-hop labeling", "reach index", "bitset"]
readingTime: 15
---

> **Subtitle / Abstract**  
> The hard part of graph querying is not "whether you can find a path". It is whether you can find it **reliably within latency and memory budgets**. This article breaks reachability into three layers: **online BFS + hop limit, offline closure (usually not fully materialized), and indexed queries (2-hop / reach index)**, then gives a directly usable engineering decision template.

- **Estimated reading time**: 12-16 minutes  
- **Tags**: `k-hop`, `Reachability`, `BFS`, `Bitmap Index`  
- **SEO keywords**: k-hop, Reachability, Transitive Closure, 2-hop labeling, reach index  
- **Meta description**: From online BFS to indexed reachability queries: hop limits, closure cost tradeoffs, and 2-hop/bitmap index selection.

---

## Target Audience

- Engineers working on graph databases, risk-control graphs, dependency analysis, or call-chain troubleshooting
- Developers who need to turn "path existence" from interview logic into production capability
- System designers facing the three-way tension of high query volume, large graphs, and frequent updates

## Background / Motivation

Reachability queries are a core graph-system capability, but production systems face three practical tensions:

1. Queries must be fast: typically synchronous inside API paths (millisecond-level)
2. Graphs are large: from millions to hundreds of millions of nodes/edges
3. Updates are frequent: index maintenance cost cannot grow without bound

So you should not focus on one algorithm only. You need layered strategies by scenario:

- Online low-latency: BFS + hop limit + early stop
- Offline exactness: transitive closure (usually not fully materialized)
- Query-heavy workloads: bitmap indexes, 2-hop labeling, reach indexes

## Core Concepts

| Concept | Definition | Key Cost |
| --- | --- | --- |
| Reachability | Whether a path `u -> v` exists | Query latency |
| k-hop query | Reachable set with path length `<= k` | Frontier expansion size |
| Transitive Closure | Full pairwise reachability matrix | Precompute and storage cost |
| 2-hop Labeling | Reachability decision via hub labels | Label build and maintenance complexity |
| Reach Index | Family of query-oriented reachability indexes | Index size and update cost |

---

## A — Algorithm (Problem and Algorithm)

### Problem Restatement (Engineering Abstraction)

Given a directed graph `G=(V,E)`, support two query types:

1. `reachable(u, v)`: decide whether `u` can reach `v`
2. `k_hop(u, k)`: return nodes reachable from `u` within `k` hops

Constraints:

- Queries must support early stop (target hit, hop limit reached, budget reached)
- No recursion (deep-graph risk); use iterative implementations
- Optional: introduce indexes to accelerate high-frequency queries

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| graph | List[List[int]] | Adjacency list, node IDs are 0..n-1 |
| u, v | int | Source and target |
| k | int | Max hops |
| Return 1 | bool | Reachable or not |
| Return 2 | Set[int] | k-hop neighborhood |

### Example 1: k-hop

```text
graph = [
  [1,2],   # 0
  [3],     # 1
  [3,4],   # 2
  [5],     # 3
  [],      # 4
  []       # 5
]
query: k_hop(0, 2)
result: {0,1,2,3,4}
```

### Example 2: Reachability

```text
query: reachable(0, 5)
result: true
query: reachable(4, 5)
result: false
```

---

## Reasoning Path (From Naive to Engineering-Ready)

### Naive Option 1: Full-graph BFS for Every Query

- Correct but not economical
- Too much repeated computation under high query frequency

### Naive Option 2: Full Transitive Closure (TC)

- Query can be `O(1)`
- But build and storage are usually too heavy (especially large graph + frequent updates)

### Key Observations

1. Most online queries need only local scope (k-hop) or early-stop hits
2. Not every graph is worth full closure materialization
3. Indexing should follow the query/update ratio, not blind theoretical optimality

### Method Selection

- **Prioritize online queries**: BFS + hop limit + early stop
- **Static graph with high query density**: consider reach indexes (2-hop/bitmap)
- **Dynamic graph with frequent updates**: favor lightweight indexes + online search hybrid

---

## C — Concepts (Core Ideas)

### 1) BFS + Hop Limit

For `k-hop`, BFS is the natural model because BFS layers are hop counts.

State definition: `(node, depth)`

Pruning rules:

- `depth == k`: do not expand neighbors further
- `node == target`: return true immediately for reachability
- `visited_budget` hits cap: return partial result or degrade

### 2) Reachability and Transitive Closure

Transitive closure can be viewed as a boolean reachability matrix `R`:

- `R[u][v] = 1` iff `u` reaches `v`

Advantage: extremely fast queries.  
Cost: expensive build, large storage, expensive updates.

Engineering conclusion: **usually do not fully materialize** unless the graph is relatively static and query volume is high enough to amortize the cost.

### 3) Bitmap Indexes / 2-hop Labeling / Reach Indexes

2-hop labeling decision form (directed reachability):

- For each node `x`, maintain `L_out(x)` and `L_in(x)`
- `u` reaches `v` iff `L_out(u) ∩ L_in(v) != ∅` (plus reflexive rules)

Pros: very fast queries.  
Challenges: label construction and incremental maintenance are complex, and label size is highly graph-structure dependent.

Common engineering compromises:

- Bitmap reach index (compressed storage)
- Hierarchical indexes + online BFS verification
- Landmark/Bloom prefilter + exact search fallback

### 4) Minimal Hand-Worked 2-hop Labeling Example

Consider the directed graph:

```text
0 -> 1 -> 3
 \\        ^
  -> 2 ----|
```

A simplified label set (for demonstration):

- `L_out(0) = {1,2,3}`
- `L_out(1) = {3}`
- `L_out(2) = {3}`
- `L_in(3) = {0,1,2}`

For `reachable(0,3)`, check only:

```text
L_out(0) ∩ L_in(3) = {1,2,3} ∩ {0,1,2} = {1,2} != ∅
```

So you can return reachable without expanding the full online frontier.  
This is why 2-hop is common in read-heavy, write-light workloads: move query cost into offline preprocessing.

---

## Practical Guide / Steps

1. Quantify workload first: QPS, P99, graph scale, update frequency
2. Build a baseline: iterative BFS + hop limit + early stop
3. Add indexes only after load testing: prioritize bitmap index or lightweight reach index
4. If index hits fail, fall back to online BFS
5. In strict-correctness scenarios, Bloom can only prefilter, never decide alone

Runnable Python example (`python3 reachability_demo.py`):

```python
from collections import deque
from typing import List, Set


def bfs_k_hop(graph: List[List[int]], s: int, k: int) -> Set[int]:
    vis = bytearray(len(graph))
    vis[s] = 1
    q = deque([(s, 0)])
    out = {s}

    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = 1
                out.add(v)
                q.append((v, d + 1))

    return out


def reachable_bfs(graph: List[List[int]], s: int, t: int, hop_limit: int | None = None) -> bool:
    vis = bytearray(len(graph))
    vis[s] = 1
    q = deque([(s, 0)])

    while q:
        u, d = q.popleft()
        if u == t:
            return True
        if hop_limit is not None and d == hop_limit:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = 1
                q.append((v, d + 1))

    return False


def transitive_closure_small(graph: List[List[int]]) -> List[int]:
    """Small-graph demo: one bitset row per node (Python int)."""
    n = len(graph)
    rows = [0] * n
    for u in range(n):
        rows[u] |= 1 << u
        for v in graph[u]:
            rows[u] |= 1 << v

    # Warshall-bitset: if u reaches k, then u also reaches all nodes that k reaches
    for k in range(n):
        mk = 1 << k
        rk = rows[k]
        for u in range(n):
            if rows[u] & mk:
                rows[u] |= rk

    return rows


def reachable_by_tc(rows: List[int], u: int, v: int) -> bool:
    return ((rows[u] >> v) & 1) == 1


if __name__ == "__main__":
    g = [[1, 2], [3], [3, 4], [5], [], []]

    print("k<=2 from 0:", sorted(bfs_k_hop(g, 0, 2)))
    print("reachable 0->5:", reachable_bfs(g, 0, 5))
    print("reachable 4->5:", reachable_bfs(g, 4, 5))

    tc = transitive_closure_small(g)
    print("tc 0->5:", reachable_by_tc(tc, 0, 5))
```

---

## E — Engineering (Applications)

### Scenario 1: k-hop Expansion on Risk-Control Graphs (Python)

**Background**: Expand from risky seed accounts to accounts within `k` hops for real-time blocking.  
**Why this fits**: BFS layer semantics align naturally with hop rules and budget control.

```python
from collections import deque

def risk_expand(graph, seeds, k):
    vis = set(seeds)
    q = deque((s, 0) for s in seeds)
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if v not in vis:
                vis.add(v)
                q.append((v, d + 1))
    return vis
```

### Scenario 2: Fast Service-Call Reachability Checks (Go)

**Background**: During incident debugging, determine whether service A can reach service B via call chains.  
**Why this fits**: Reachability can stop immediately on hit, which works well for online diagnostics.

```go
package main

import "fmt"

func Reachable(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	q := []int{s}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		u := q[head]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(Reachable(g, 0, 5))
}
```

### Scenario 3: Bitmap Index for Static Dependency Graphs (C++)

**Background**: Build/compile dependency graphs update infrequently, but dependency-existence checks are very frequent.  
**Why this fits**: Build bitmap closure once, then answer queries with `O(1)` bit checks.

```cpp
#include <iostream>
#include <vector>

std::vector<unsigned long long> closure6(const std::vector<std::vector<int>>& g) {
    int n = (int)g.size();
    std::vector<unsigned long long> row(n, 0);
    for (int u = 0; u < n; ++u) {
        row[u] |= 1ULL << u;
        for (int v : g[u]) row[u] |= 1ULL << v;
    }
    for (int k = 0; k < n; ++k) {
        unsigned long long mk = 1ULL << k;
        for (int u = 0; u < n; ++u) {
            if (row[u] & mk) row[u] |= row[k];
        }
    }
    return row;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    auto r = closure6(g);
    std::cout << (((r[0] >> 5) & 1ULL) ? "reachable" : "not") << "\n";
}
```

---

## R — Reflection (Deeper Analysis)

### Complexity Analysis

Let the actually touched subgraph during query be `V'` nodes and `E'` edges:

- Online BFS query: `O(V' + E')`
- k-hop query: worst-case still `O(V'+E')`, but usually much smaller due to `k`
- Full closure:
  - BFS from every node: `O(n*(n+m))`
  - Boolean-matrix / bitset optimization still has high precompute and storage cost

### Alternatives and Tradeoffs

| Approach | Query | Build | Update | Best Fit |
| --- | --- | --- | --- | --- |
| BFS per query | Medium | None | None | Frequent updates, low/medium query volume |
| Full closure | Very fast | Very high | Very high | Static small/medium graph, high query density |
| 2-hop / reach index | Fast | Medium-high | Medium-high | Query-heavy workloads tolerant of offline build |
| Lightweight index + BFS fallback | Fast (avg) | Medium | Medium | Common compromise for most online systems |

### Common Mistakes

1. Blindly materializing full closure, making build/storage uncontrollable
2. Using Bloom alone for strict-correctness reachability decisions
3. No hop/budget limits, causing long-tail latency explosions online

### Why This Is Engineering-Feasible

- BFS + hop limit gives a low-complexity, low-maintenance baseline
- Indexes are introduced gradually based on query density, avoiding over-design
- "Index hit + search fallback" balances latency and correctness

---

## FAQs and Notes

1. **Is Reachability the same as shortest path?**  
   No. Reachability only asks whether a path exists, not the minimum distance.

2. **Should Transitive Closure never be computed?**  
   Not true. It is valuable on static graphs with high query density; most dynamic online graphs just cannot maintain full closure economically.

3. **Is 2-hop labeling always better than BFS?**  
   No. It is query-friendly but heavier to build/maintain, so it fits read-heavy, write-light scenarios.

---

## Best Practices and Recommendations

- Ship an observable BFS baseline first (with hop limits, budgets, timeouts)
- Use real traffic profiles to decide whether a reach index is worth introducing
- Prioritize maintainability in index design before theoretical optimality
- Keep a downgrade path: fallback to BFS when indexes degrade or fail

---

## S — Summary (Wrap-up)

### Core Takeaways

- Reachability is an "algorithm + system constraints" problem, not a single-optimal-algorithm problem
- For `k-hop`, prefer BFS + hop limit + early stop
- Transitive Closure can make queries fast, but usually should not be fully materialized, especially on dynamic graphs
- 2-hop labeling / reach indexes are best for read-heavy, write-light workloads
- The most stable production pattern is usually "lightweight index + online BFS fallback"

### Recommended Further Reading

- LeetCode 1971 (Find if Path Exists in Graph)
- LeetCode 847 (Shortest Path Visiting All Nodes, state-space search extension)
- Graph database query-optimization docs (Neo4j / JanusGraph neighborhood query strategies)
- Classic reachability-index papers (2-hop labeling / GRAIL)

---

## Metadata

- **Reading time**: 12-16 minutes
- **Tags**: Reachability, k-hop, BFS, 2-hop labeling
- **SEO keywords**: Reachability, k-hop, Transitive Closure, 2-hop labeling, reach index
- **Meta description**: Engineering reachability query strategy: BFS+hop limits, closure tradeoffs, bitmap/2-hop indexing, and online fallback.

---

## Call to Action (CTA)

Two practical next steps:

1. Add `hop_limit` and `visit_budget` parameters to your current reachability API
2. Run an A/B load test on real traffic: "BFS per query" vs "lightweight index + fallback"

If you want, I can write the next post as well:
"Reachability Index Implementation Guide: when to pick 2-hop, when to pick GRAIL, and when to stick with BFS."

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


def reachable(graph, s, t):
    vis = [False] * len(graph)
    q = deque([s])
    vis[s] = True
    while q:
        u = q.popleft()
        if u == t:
            return True
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                q.append(v)
    return False


def k_hop(graph, s, k):
    vis = [False] * len(graph)
    q = deque([(s, 0)])
    vis[s] = True
    out = {s}
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                out.add(v)
                q.append((v, d + 1))
    return out
```

```c
#include <stdbool.h>
#include <stdio.h>

#define N 6

bool reachable(int g[N][N], int s, int t) {
    int q[128], head = 0, tail = 0;
    bool vis[N] = {0};
    q[tail++] = s;
    vis[s] = true;
    while (head < tail) {
        int u = q[head++];
        if (u == t) return true;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                q[tail++] = v;
            }
        }
    }
    return false;
}

int main(void) {
    int g[N][N] = {0};
    g[0][1] = g[0][2] = 1;
    g[1][3] = 1;
    g[2][3] = g[2][4] = 1;
    g[3][5] = 1;
    printf("%d\n", reachable(g, 0, 5));
    return 0;
}
```

```cpp
#include <iostream>
#include <queue>
#include <vector>

bool reachable(const std::vector<std::vector<int>>& g, int s, int t) {
    std::vector<char> vis(g.size(), 0);
    std::queue<int> q;
    vis[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (u == t) return true;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                q.push(v);
            }
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    std::cout << reachable(g, 0, 5) << "\n";
}
```

```go
package main

import "fmt"

func reachable(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	q := []int{s}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		u := q[head]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(reachable(g, 0, 5))
}
```

```rust
use std::collections::VecDeque;

fn reachable(graph: &Vec<Vec<usize>>, s: usize, t: usize) -> bool {
    let mut vis = vec![false; graph.len()];
    let mut q = VecDeque::new();
    vis[s] = true;
    q.push_back(s);

    while let Some(u) = q.pop_front() {
        if u == t {
            return true;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                q.push_back(v);
            }
        }
    }
    false
}

fn main() {
    let g = vec![vec![1, 2], vec![3], vec![3, 4], vec![5], vec![], vec![]];
    println!("{}", reachable(&g, 0, 5));
}
```

```javascript
function reachable(graph, s, t) {
  const vis = Array(graph.length).fill(false);
  const q = [s];
  let head = 0;
  vis[s] = true;

  while (head < q.length) {
    const u = q[head++];
    if (u === t) return true;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        q.push(v);
      }
    }
  }
  return false;
}

const g = [[1, 2], [3], [3, 4], [5], [], []];
console.log(reachable(g, 0, 5));
```
