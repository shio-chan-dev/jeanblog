---
title: "Connected Components and Strongly Connected Components: Tarjan / Kosaraju ACERS Engineering Analysis"
date: 2026-02-09T09:50:22+08:00
draft: false
description: "A systematic guide to undirected Connected Components and directed SCCs, with emphasis on Tarjan (common in production) and Kosaraju, mapped to graph-database scenarios such as community pre-grouping, subgraph splitting, and partition hints."
tags: ["Graph Theory", "Connected Components", "SCC", "Tarjan", "Kosaraju", "BFS", "DFS", "Graph Database"]
categories: ["Logic and Algorithms"]
keywords: ["Connected Components", "Strongly Connected Components", "Tarjan", "Kosaraju", "Graph Partitioning", "Community Detection"]
---

> **Subtitle / Abstract**  
> Components are foundational for graph algorithms: undirected graphs ask "are nodes connected," while directed graphs ask "are nodes mutually reachable." Following the ACERS template, this article moves from naive methods to Tarjan / Kosaraju, then shows production graph-database use cases with runnable multi-language code.

- **Estimated reading time**: 14-18 minutes  
- **Tags**: `Graph Theory`, `Connected Components`, `SCC`, `Tarjan`  
- **SEO keywords**: Connected Components, SCC, Tarjan, Kosaraju, graph database  
- **Meta description**: From undirected connected components to directed SCCs, with clear Tarjan/Kosaraju mechanics, complexity, and production rollout guidance.  

---

## Target Audience

- Learners who need BFS/DFS to become second nature
- Engineers doing subgraph analysis and partition planning in graph-database systems
- Intermediate developers who want one unified framework for "undirected CC + directed SCC"

## Background / Motivation

In production, you quickly hit three types of questions:

1. Do these nodes naturally split into disconnected groups? (undirected connected components)
2. Which nodes form mutually reachable strong cycles? (directed SCC)
3. How can a large graph be split into subgraphs that are more parallelizable, cache-friendly, and shard-friendly?

If you only know BFS/DFS but not the "component view," you end up repeating reachability queries with high cost and weak maintainability.  
The value of component algorithms is: **one full-graph scan turns many local queries into O(1) component-ID comparisons**.

## Core Concepts

- **Connected Components (CC)**: maximal node sets in an undirected graph where any two nodes are reachable
- **Strongly Connected Components (SCC)**: maximal node sets in a directed graph where any two nodes are mutually reachable
- **Condensation DAG**: DAG obtained by contracting each SCC into a single node
- **Tarjan core state**: `dfn[u]` (timestamp), `low[u]` (minimum reachable timestamp), stack and `in_stack`
- **Kosaraju core flow**: DFS finish-order on original graph + second DFS on reversed graph

---

## A — Algorithm (Problem and Algorithm)

### Problem Restatement (Engineering Formulation)

Given a graph `G=(V,E)`:

- If `G` is undirected, output all **Connected Components**;
- If `G` is directed, output all **Strongly Connected Components**.

Also return:

- The number of components
- The component ID of every node

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| n | int | Number of nodes (`0..n-1`) |
| edges | List[(u,v)] | Edge list |
| directed | bool | Directed or not |
| Return | (k, comp_id[]) | `k` is component count; `comp_id[i]` is node i's component ID |

### Example 1 (Undirected CC)

```text
n = 7
edges = [(0,1),(1,2),(3,4),(5,6)]

Connected components:
{0,1,2}, {3,4}, {5,6}
k = 3
```

### Example 2 (Directed SCC)

```text
n = 6
edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,3),(4,5)]

Strongly connected components:
{0,1,2}, {3,4}, {5}
k = 3
```

---

## Reasoning Path (From Naive to Optimal)

### Naive Approach

- Run one reachability search (BFS/DFS) from every node
- Then merge or cross-compare the resulting sets

Problems:

- Time complexity inflates to `O(V*(V+E))`
- Repeated scanning over the same edges hurts cache locality and throughput

### Key Observations

1. **Undirected graph**: from one unvisited node, a single BFS/DFS can consume one full connected component.  
2. **Directed graph**: one-way reachability is not enough; you need equivalence classes of mutual reachability (SCC).

### Method Selection

- Undirected graph: iterative BFS/DFS + visited (most robust)
- Directed graph: Tarjan (single DFS pass, more common in production)
- Kosaraju: very intuitive implementation, useful for cross-checking and education

---

## C — Concepts (Core Ideas)

### Method Categories

- Graph traversal: BFS / DFS
- Component decomposition: Connected Components / SCC
- Condensation modeling: SCC -> DAG

### Tarjan Invariants

Maintain during DFS:

- `dfn[u]`: timestamp when node `u` is first visited
- `low[u]`: smallest reachable `dfn` from `u` via tree edges + back edges

When `dfn[u] == low[u]`, `u` is the root of one SCC. Pop until `u` to materialize that SCC.

### Essence of Kosaraju

1. Record postorder by DFS finish time on the original graph
2. Build the reversed graph
3. Run DFS on the reversed graph in reverse postorder; each DFS yields one SCC

### Why Tarjan Is Common in Engineering

- One DFS pass for SCC decomposition (no explicit reverse-graph build required)
- Smaller constant factors and more direct memory behavior
- Easier to combine with online stats (for example SCC size thresholds)

---

## Practical Guide / Steps

### Undirected Connected Components (Iterative)

1. Build adjacency list
2. Start one stack/queue traversal from every unvisited node
3. Assign `comp_id` during traversal
4. Optional early stop:
   - If only checking whether two nodes are in the same component, stop once confirmed
   - If only building a k-hop subgraph, limit traversal depth

### Directed SCC (Tarjan)

1. Maintain global timestamp `time`
2. Push node on DFS entry and initialize `dfn/low`
3. Recurse into unvisited neighbors; for stack neighbors, update `low`
4. When `dfn==low`, pop a full SCC

### Engineering Choice for `visited`

- **bitmap**: exact, predictable, good for fixed ID spaces
- **bloom filter**: memory-saving but with false positives; suitable for approximate dedup, not strict-correctness traversal

---

## Runnable Example (Python)

```python
from collections import deque
from typing import List, Tuple


def connected_components_undirected(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    comp = [-1] * n
    cid = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        queue = deque([start])
        comp[start] = cid
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    queue.append(v)
        cid += 1
    return cid, comp


def scc_tarjan(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    dfn = [-1] * n
    low = [0] * n
    in_stack = [False] * n
    stack = []
    comp = [-1] * n

    time = 0
    scc_id = 0

    def dfs(u: int) -> None:
        nonlocal time, scc_id
        dfn[u] = low[u] = time
        time += 1
        stack.append(u)
        in_stack[u] = True

        for v in graph[u]:
            if dfn[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif in_stack[v]:
                low[u] = min(low[u], dfn[v])

        if dfn[u] == low[u]:
            while True:
                x = stack.pop()
                in_stack[x] = False
                comp[x] = scc_id
                if x == u:
                    break
            scc_id += 1

    for i in range(n):
        if dfn[i] == -1:
            dfs(i)

    return scc_id, comp


if __name__ == "__main__":
    n1 = 7
    e1 = [(0, 1), (1, 2), (3, 4), (5, 6)]
    k1, c1 = connected_components_undirected(n1, e1)
    print("Undirected CC count:", k1, "comp:", c1)

    n2 = 6
    e2 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)]
    k2, c2 = scc_tarjan(n2, e2)
    print("Directed SCC count:", k2, "comp:", c2)
```

Run:

```bash
python3 connected_components_demo.py
```

---

## E — Engineering (Applications)

### Scenario 1: Community Pre-grouping in Graph Databases (Python)

**Background**: Before community analysis on a user-relationship graph, first remove isolated disconnected blocks.  
**Why this fits**: Running CC first directly narrows the scope for later algorithms (for example Louvain).  

```python
def group_by_component(node_ids, comp_ids):
    groups = {}
    for node, cid in zip(node_ids, comp_ids):
        groups.setdefault(cid, []).append(node)
    return groups
```

### Scenario 2: Subgraph Splitting for Parallel Task Dispatch (Go)

**Background**: Split offline graph-compute tasks by component across workers to reduce cross-worker communication.  
**Why this fits**: Components are naturally independent, so tasks parallelize without cross-dependencies.  

```go
package main

import "fmt"

func bucketByComp(comp []int) map[int][]int {
	b := map[int][]int{}
	for node, cid := range comp {
		b[cid] = append(b[cid], node)
	}
	return b
}

func main() {
	comp := []int{0, 0, 1, 1, 2}
	fmt.Println(bucketByComp(comp))
}
```

### Scenario 3: Partition Hints from Component IDs (JavaScript)

**Background**: In online graph services, you want highly coupled nodes placed on the same shard as much as possible.  
**Why this fits**: SCC/CC IDs are strong signals that reduce cross-shard edge ratio.  

```javascript
function assignShardByComp(compIds, shardCount) {
  return compIds.map((cid) => cid % shardCount);
}

console.log(assignShardByComp([0, 0, 1, 1, 2, 2], 2));
```

---

## R — Reflection (Deeper Analysis)

### Complexity Analysis

- Undirected CC (BFS/DFS): `O(V+E)`, space `O(V)`
- Tarjan SCC: `O(V+E)`, space `O(V)`
- Kosaraju SCC: `O(V+E)`, space `O(V+E)` (includes reverse graph)

### Alternatives and Tradeoffs

| Method | Graph Type | Time Complexity | Pros | Limits |
| --- | --- | --- | --- | --- |
| BFS/DFS components | Undirected | O(V+E) | Intuitive, stable | Cannot handle SCC |
| Tarjan | Directed | O(V+E) | Single pass, production-friendly | Harder than plain BFS to implement |
| Kosaraju | Directed | O(V+E) | Clear mental model | Needs reverse graph and two DFS passes |
| Union-Find | Static undirected connectivity | approx O(E α(V)) | Quick to implement | Not suitable for SCC |

### Why Tarjan Is More Engineering-Feasible

- Better fit for online pipelines: one pass can emit SCC IDs directly
- No reverse-graph build required, reducing extra memory and data movement
- Easy to attach additional metrics: SCC size, out-edge count, inter-SCC edge ratio

---

## Explanation and Principles (Why This Works)

- The essence of CC is "undirected reachability equivalence classes"; one traversal can fully cover one class.
- The essence of SCC is "directed mutual-reachability equivalence classes"; Tarjan uses `dfn/low + stack` to identify cycle roots online.
- After mapping nodes to `comp_id`, many queries are reduced in dimension:
  - "Are these in the same group?" => `comp_id[u] == comp_id[v]`
  - "Partition hint?" => `hash(comp_id)`

---

## FAQs and Notes

1. **Can Tarjan be used to compute SCC on undirected graphs?**  
   Yes, but unnecessary; direct CC is simpler for undirected graphs.

2. **Must Tarjan be recursive?**  
   No. It can be converted to an explicit-stack iterative version, but implementation complexity is higher.

3. **Can Bloom filter replace visited?**  
   Not for strict-correctness scenarios; false positives can skip nodes that should be traversed.

4. **Why is my SCC ordering different?**  
   As long as SCC partitioning is correct, numbering order can vary with traversal order.

---

## Best Practices and Recommendations

- Normalize node IDs to `0..n-1` before graph algorithms to avoid mapping bugs
- Prefer iterative BFS/DFS for undirected CC to avoid deep-recursion stack risk
- For large directed graphs, prioritize Tarjan; add Kosaraju when teaching or cross-validating
- Persist `comp_id` in production and reuse it for query, cache, and sharding decisions

---

## S — Summary (Wrap-up)

### Core Takeaways

- Components are the first "dimensionality reduction layer" in graph computing; one computation supports many query types.
- Undirected CC and directed SCC are different problems and cannot be mixed.
- Tarjan identifies SCC online with `dfn/low` in `O(V+E)`, which is why it is production-common.
- Kosaraju is great for understanding and cross-validation; Tarjan is usually better for production rollout.
- In graph databases, `comp_id` can directly support coarse community grouping, subgraph splitting, and partition hints.

### Recommended Further Reading

- Tarjan, R. (1972). Depth-first search and linear graph algorithms.
- CLRS graph chapters (SCC, topological sorting)
- Neo4j Graph Data Science docs: Connected Components / SCC

### Closing Note

If you already know BFS/DFS, the next required step is component thinking.  
In engineering, the real value is not "can traverse," but turning traversal outputs into stable, reusable structured labels (`comp_id`).

---

## Metadata

- **Reading time**: 14-18 minutes
- **Tags**: Graph Theory, Connected Components, SCC, Tarjan, Graph Database
- **SEO keywords**: Connected Components, SCC, Tarjan, Kosaraju, Graph Partitioning
- **Meta description**: A systematic guide to undirected CC and directed SCC, focused on Tarjan/Kosaraju and graph-database engineering rollout.

---

## Call to Action (CTA)

Two immediate actions:

1. Run CC / SCC on your production graph and output a histogram of `comp_id` distribution.
2. Measure cross-component edge ratio and evaluate partitioning or subgraph-level parallelization.

If you want, I can continue with "3) Shortest Paths (Dijkstra / A* / Multi-source BFS)" in the same ACERS style.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


def connected_components_undirected(n, edges):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    comp = [-1] * n
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        q = deque([s])
        comp[s] = cid
        while q:
            u = q.popleft()
            for v in g[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return cid, comp
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int* data;
    int size;
    int cap;
} Vec;

void push(Vec* v, int x) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 4;
        v->data = (int*)realloc(v->data, sizeof(int) * v->cap);
    }
    v->data[v->size++] = x;
}

int main(void) {
    int n = 5;
    int comp[5] = {-1, -1, -1, -1, -1};
    // Demo placeholder: in real systems, build adjacency by edges and run BFS/DFS.
    comp[0] = comp[1] = 0;
    comp[2] = comp[3] = 1;
    comp[4] = 2;
    for (int i = 0; i < n; ++i) printf("node %d -> comp %d\n", i, comp[i]);
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;

pair<int, vector<int>> connectedComponentsUndirected(int n, const vector<pair<int,int>>& edges) {
    vector<vector<int>> g(n);
    for (auto [u,v] : edges) {
        g[u].push_back(v);
        g[v].push_back(u);
    }
    vector<int> comp(n, -1);
    int cid = 0;
    queue<int> q;
    for (int s = 0; s < n; ++s) {
        if (comp[s] != -1) continue;
        comp[s] = cid;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : g[u]) {
                if (comp[v] == -1) {
                    comp[v] = cid;
                    q.push(v);
                }
            }
        }
        cid++;
    }
    return {cid, comp};
}

int main() {
    vector<pair<int,int>> edges = {{0,1},{1,2},{3,4}};
    auto [k, comp] = connectedComponentsUndirected(5, edges);
    cout << "k=" << k << "\n";
    for (int i = 0; i < (int)comp.size(); ++i) cout << i << ":" << comp[i] << " ";
    cout << "\n";
}
```

```go
package main

import "fmt"

func connectedComponentsUndirected(n int, edges [][2]int) (int, []int) {
	g := make([][]int, n)
	for _, e := range edges {
		u, v := e[0], e[1]
		g[u] = append(g[u], v)
		g[v] = append(g[v], u)
	}
	comp := make([]int, n)
	for i := range comp {
		comp[i] = -1
	}
	cid := 0
	q := make([]int, 0)
	for s := 0; s < n; s++ {
		if comp[s] != -1 {
			continue
		}
		comp[s] = cid
		q = append(q, s)
		for len(q) > 0 {
			u := q[0]
			q = q[1:]
			for _, v := range g[u] {
				if comp[v] == -1 {
					comp[v] = cid
					q = append(q, v)
				}
			}
		}
		cid++
	}
	return cid, comp
}

func main() {
	edges := [][2]int{{0, 1}, {1, 2}, {3, 4}}
	k, comp := connectedComponentsUndirected(5, edges)
	fmt.Println(k, comp)
}
```

```rust
use std::collections::VecDeque;

fn connected_components_undirected(n: usize, edges: &[(usize, usize)]) -> (usize, Vec<i32>) {
    let mut g = vec![vec![]; n];
    for &(u, v) in edges {
        g[u].push(v);
        g[v].push(u);
    }
    let mut comp = vec![-1; n];
    let mut cid: i32 = 0;

    for s in 0..n {
        if comp[s] != -1 {
            continue;
        }
        let mut q = VecDeque::new();
        comp[s] = cid;
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            for &v in &g[u] {
                if comp[v] == -1 {
                    comp[v] = cid;
                    q.push_back(v);
                }
            }
        }
        cid += 1;
    }

    (cid as usize, comp)
}

fn main() {
    let edges = vec![(0, 1), (1, 2), (3, 4)];
    let (k, comp) = connected_components_undirected(5, &edges);
    println!("{} {:?}", k, comp);
}
```

```javascript
function connectedComponentsUndirected(n, edges) {
  const g = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    g[u].push(v);
    g[v].push(u);
  }

  const comp = Array(n).fill(-1);
  let cid = 0;

  for (let s = 0; s < n; s += 1) {
    if (comp[s] !== -1) continue;
    const queue = [s];
    comp[s] = cid;

    while (queue.length) {
      const u = queue.shift();
      for (const v of g[u]) {
        if (comp[v] === -1) {
          comp[v] = cid;
          queue.push(v);
        }
      }
    }
    cid += 1;
  }

  return [cid, comp];
}

console.log(connectedComponentsUndirected(5, [[0, 1], [1, 2], [3, 4]]));
```
