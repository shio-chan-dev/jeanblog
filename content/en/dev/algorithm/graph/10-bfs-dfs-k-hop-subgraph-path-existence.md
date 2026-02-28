---
title: "BFS / DFS Engineering Primer: k-hop Queries, Subgraph Extraction, and Path Existence ACERS Breakdown"
date: 2026-02-09T09:44:11+08:00
draft: false
categories: ["Logic and Algorithms"]
tags: ["Graph", "BFS", "DFS", "k-hop", "subgraph extraction", "path existence"]
description: "Centered on three high-frequency graph tasks (k-hop query, subgraph extraction, and path existence), this article explains practical BFS/DFS iteration templates, early-stop pruning, and visited bitmap/bloom choices with runnable multi-language implementations."
keywords: ["BFS", "DFS", "k-hop", "subgraph extraction", "path existence", "visited bitmap", "bloom filter"]
---

> **Subtitle / Abstract**  
> BFS / DFS are not just about "being able to code them." You need production-ready behavior, predictable cost, and provable correctness. Following the ACERS structure, this article breaks three common tasks (k-hop query, subgraph extraction, and path existence) into reusable templates: **iterative implementation + early stop + visited structure selection**.

- **Estimated reading time**: 12-16 minutes  
- **Tags**: `Graph`, `BFS`, `DFS`, `k-hop`, `subgraph extraction`  
- **SEO keywords**: BFS, DFS, k-hop query, subgraph extraction, path existence, visited bitmap, bloom filter  
- **Meta description**: BFS/DFS for engineering scenarios: iterative implementations to avoid stack overflow, early stop to cut search cost, and visited bitmap/bloom to optimize memory and dedup performance.

---

## Target Audience

- Engineers working on graph databases, risk-control relationship graphs, or call-chain analysis
- Learners who can write "problem-solution style BFS/DFS" but do not yet have engineering templates
- Developers who want traversal code that is stable, observable, and extensible

## Background / Motivation

In production systems, BFS/DFS is usually not a one-off offline script. It is part of an online request path:

- `k-hop` neighborhood queries need latency control
- Subgraph extraction needs memory and output-size control
- Path existence checks need fast true/false responses

If you stop at textbook recursive templates, you quickly hit issues:

1. Deep graphs cause recursive stack overflow
2. No pruning causes unnecessary expansion
3. The wrong visited structure hurts both memory and throughput

So this article has one focus:
upgrade BFS/DFS to a level where you can use them fluently in production.

## Core Concepts

| Concept | Purpose | Engineering Focus |
| --- | --- | --- |
| BFS (queue) | Layer-by-layer expansion, natural support for hop levels | Good for k-hop, minimum edge count, layered subgraphs |
| DFS (stack) | Deep exploration, efficient for path existence checks | Good for fast reachability decisions and depth-based pruning |
| early stop | Stop search as soon as conditions are met | Controls P99 latency and resource usage |
| visited bitmap | Exact dedup with compact memory | Requires node ID compression first |
| bloom filter | Probabilistic dedup / prefilter | Has false positives; cannot be used alone in strict-correctness tasks |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement (LeetCode-style training problem)

Given an unweighted graph `G` (adjacency list), a start node `s`, a maximum hop count `K`, and an optional target node `t`:

1. Return the set of nodes reachable from `s` within `K` hops (k-hop query)
2. Return the subgraph formed by visited nodes and edges (subgraph extraction)
3. Determine whether a path `s -> t` exists (path existence)

Requirements:

- Use **iterative** BFS/DFS (no recursion)
- Support early stop (for example: beyond K hops, target hit, or business predicate hit)
- Maintain visited state to avoid repeated expansion

### Input and Output

| Name | Type | Description |
| --- | --- | --- |
| graph | List[List[int]] | Adjacency list, node IDs are 0..n-1 |
| s | int | Start node |
| K | int | Maximum hop count (for BFS) |
| t | int | Target node (for reachability) |
| Return 1 | Set[int] | Nodes reachable within K hops |
| Return 2 | List[Tuple[int,int]] | Extracted edge set (optional) |
| Return 3 | bool | Reachable or not |

### Example 1: k-hop query

```text
graph = [
  [1,2],    # 0
  [3],      # 1
  [3,4],    # 2
  [5],      # 3
  [],       # 4
  []        # 5
]
s = 0, K = 2

output nodes: {0,1,2,3,4}
```

Explanation: within 2 hops, you can reach `0 (0 hop), 1/2 (1 hop), 3/4 (2 hops)`. Node 5 needs 3 hops.

### Example 2: path existence

```text
same graph as above
s = 0, t = 5

output: true
```

---

## Derivation (From naive approach to engineering template)

### Naive version: recursive DFS / BFS without pruning

- Recursive DFS can hit stack-depth limits on deep graphs
- BFS without hop limits may scan the full graph
- Without visited state, expansions can repeat exponentially

### Key Observations

1. In practice, you usually want not "full-graph traversal" but the minimal traversal that satisfies business constraints
2. Search order can be templated (queue/BFS, stack/DFS), but pruning must be business-aware
3. visited is not one fixed implementation. You must choose based on graph scale and correctness needs

### Method Selection

- `k-hop`: prefer BFS (naturally layered)
- Path existence: prefer iterative DFS (stack + early stop)
- Large graphs: ID compression + bitmap; for high-throughput, weak-consistency dedup, add bloom prefiltering

---

## C - Concepts (Core Ideas)

### Method Categories

- Graph Traversal
- Layered Search (BFS)
- Depth Search (DFS)
- Pruned Search

### Engineering Invariants

1. `visited[u] = true` means node `u` has been queued/stacked (or consumed, depending on policy)
2. In BFS, `(node, depth).depth` never exceeds `K`
3. After an early-stop condition triggers, returned results still satisfy business-defined correctness

### Early-Stop Design Template

- **hop limit**: stop expanding neighbors when `depth == K`
- **target hit**: return immediately when `node == t`
- **budget control**: stop when visited-node count exceeds threshold and return partial results
- **predicate pruning**: skip expansion when node attributes do not satisfy business rules

### visited Structure Selection

| Structure | Correctness | Memory | Speed | Suitable Scenarios |
| --- | --- | --- | --- | --- |
| HashSet | Exact | Medium-high | Fast | Sparse node IDs, dynamic IDs |
| Bitmap | Exact | Lowest (bit-level) | Fast | Node IDs can be compressed to contiguous integers |
| Bloom Filter | Approximate (false positives) | Very low | Fast | Prefiltering and dedup acceleration (error-tolerant) |

Key conclusions:

- **Strict correctness tasks** (for example permission checks, risk-control hits) cannot rely on bloom alone
- The safest bloom usage is "prefilter + exact-structure confirmation"

---

## Practical Guide / Steps

1. Normalize node IDs first (compress to `0..n-1` when needed)
2. For `k-hop`, use BFS with `depth` in queue entries
3. For path existence, use iterative DFS with a stack of pending nodes
4. Apply early-stop checks at the top of each loop
5. Prefer bitmap for visited (when compressible), otherwise HashSet
6. If dedup checks are the throughput bottleneck, add bloom prefiltering

Runnable Python example (`python3 bfs_dfs_demo.py`):

```python
from collections import deque
from typing import List, Set


class SimpleBloom:
    """Demo Bloom filter: prefilter only, not a standalone correctness guarantee."""

    def __init__(self, m: int = 1 << 15):
        self.m = m
        self.bits = bytearray(m // 8 + 1)

    def _idx(self, x: int, salt: int) -> int:
        return hash((x, salt)) & (self.m - 1)

    def _set(self, i: int) -> None:
        self.bits[i >> 3] |= 1 << (i & 7)

    def _get(self, i: int) -> bool:
        return (self.bits[i >> 3] >> (i & 7)) & 1 == 1

    def add(self, x: int) -> None:
        for salt in (17, 31, 73):
            self._set(self._idx(x, salt))

    def maybe_contains(self, x: int) -> bool:
        return all(self._get(self._idx(x, salt)) for salt in (17, 31, 73))


def bfs_k_hop(graph: List[List[int]], s: int, k: int) -> Set[int]:
    n = len(graph)
    visited = bytearray(n)  # bitmap
    q = deque([(s, 0)])
    visited[s] = 1
    result = {s}

    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not visited[v]:
                visited[v] = 1
                result.add(v)
                q.append((v, d + 1))

    return result


def dfs_path_exists(graph: List[List[int]], s: int, t: int) -> bool:
    n = len(graph)
    visited = bytearray(n)
    stack = [s]
    visited[s] = 1

    while stack:
        u = stack.pop()
        if u == t:  # early stop
            return True
        for v in graph[u]:
            if not visited[v]:
                visited[v] = 1
                stack.append(v)

    return False


def bfs_with_bloom_prefilter(graph: List[List[int]], s: int, limit: int = 100000) -> int:
    """Example: bloom reduces set lookups; exact set still guarantees correctness."""
    q = deque([s])
    exact = {s}
    bloom = SimpleBloom()
    bloom.add(s)
    visited_count = 0

    while q and visited_count < limit:
        u = q.popleft()
        visited_count += 1
        for v in graph[u]:
            # bloom says "not seen" => definitely unseen, enqueue directly
            if not bloom.maybe_contains(v):
                bloom.add(v)
                exact.add(v)
                q.append(v)
                continue
            # bloom says "maybe seen" => confirm with exact set
            if v not in exact:
                exact.add(v)
                q.append(v)

    return visited_count


if __name__ == "__main__":
    graph = [
        [1, 2],  # 0
        [3],     # 1
        [3, 4],  # 2
        [5],     # 3
        [],      # 4
        [],      # 5
    ]

    print("k-hop<=2:", sorted(bfs_k_hop(graph, 0, 2)))
    print("path 0->5:", dfs_path_exists(graph, 0, 5))
    print("bloom+exact visits:", bfs_with_bloom_prefilter(graph, 0, limit=100))
```

---

## E - Engineering (Production Applications)

### Scenario 1: k-hop neighborhood query in graph databases (Python)

**Background**: users provide seed nodes, and the system returns neighborhood nodes within N hops.  
**Why it fits**: BFS is naturally layered, and `depth` directly maps to k-hop business semantics.

```python
from collections import deque

def k_hop_nodes(graph, s, k):
    q = deque([(s, 0)])
    vis = {s}
    out = {s}
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if v not in vis:
                vis.add(v)
                out.add(v)
                q.append((v, d + 1))
    return out
```

### Scenario 2: call-chain fault tracing (Go)

**Background**: determine whether service A can reach faulty service B in a call graph.  
**Why it fits**: iterative DFS with target-hit early stop often returns faster than full-graph scans.

```go
package main

import "fmt"

func pathExists(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	stack := []int{s}
	vis[s] = true
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				stack = append(stack, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(pathExists(g, 0, 5)) // true
}
```

### Scenario 3: online dedup prefiltering in relationship graphs (C++)

**Background**: under high QPS, visited-set lookups become a hotspot.  
**Why it fits**: use bloom for fast "possibly unseen" routing, then confirm with exact bitmap/set to reduce average dedup cost.

```cpp
#include <bitset>
#include <iostream>
#include <unordered_set>

struct Bloom {
    static const int M = 1 << 16;
    std::bitset<M> bits;
    int h1(int x) const { return (x * 1315423911u) & (M - 1); }
    int h2(int x) const { return (x * 2654435761u) & (M - 1); }
    void add(int x) { bits.set(h1(x)); bits.set(h2(x)); }
    bool maybe(int x) const { return bits.test(h1(x)) && bits.test(h2(x)); }
};

int main() {
    Bloom b;
    std::unordered_set<int> exact;
    for (int x : {1, 2, 3}) { b.add(x); exact.insert(x); }
    int q = 4;
    if (!b.maybe(q) || exact.find(q) != exact.end()) {
        std::cout << "not visited yet\n";
    }
}
```

---

## R - Reflection (Deep Dive)

### Complexity Analysis

Let `V'` and `E'` be the node and edge counts of the visited subgraph:

- BFS / DFS time complexity: `O(V' + E')`
- Extra space for visited:
  - HashSet: `O(V')`
  - Bitmap: `O(N)` bits (`N` is the full-graph node upper bound)
  - Bloom: `O(m)` bits (`m` is bit-array size, tunable approximation)

For k-hop tasks, `V'` and `E'` are often much smaller than full-graph size, which is where early stop provides most value.

### Alternatives and Trade-offs

| Approach | Pros | Cons | Best For |
| --- | --- | --- | --- |
| Recursive DFS | Short code | Stack risk on deep graphs, weaker controllability | Small offline scripts |
| Iterative DFS | Controllable, easy to add early stop | Manual stack management | Path existence / online checks |
| BFS | Clear layering, suitable for hop constraints | Peak memory may be higher than DFS | k-hop / layered retrieval |
| Bidirectional BFS | Faster point-to-point path queries | Higher implementation complexity | Sparse graph, single-source to single-target |

### Common Wrong Approaches

1. **Mark visited only at dequeue time**: can cause repeated enqueues and queue blowup
2. **Use bloom alone as visited**: false positives can skip nodes that should be visited
3. **No budget limits**: online requests can suffer long-tail latency at high-degree nodes

### Why This Is the Most Practical Engineering Strategy

- Iterative implementation avoids recursion risks
- early stop constrains search cost inside business boundaries
- bitmap/bloom make visited strategy flexible by graph scale

---

## FAQ and Notes

1. **Which is faster, BFS or DFS?**  
   There is no absolute winner. BFS is common for k-hop. For reachability where the target may appear deep, DFS often hits faster.

2. **Can bloom false positives affect correctness?**  
   Yes. If bloom is used alone for dedup, false positives can skip valid search branches. Strict-correctness tasks must use exact confirmation.

3. **When should visited be marked?**  
   Usually at enqueue/push time, so the same node is not inserted repeatedly.

---

## Best Practices and Recommendations

- Define business stop conditions before writing traversal code
- Default to iterative versions; use recursion only for small offline tools
- Prefer bitmap when node IDs can be compressed, balancing speed and memory
- Use bloom only as a prefilter, not as a standalone correctness guarantee
- Add visit caps and latency monitoring to traversal to avoid online cascades

---

## S - Summary

### Key Takeaways

- Engineering-grade BFS/DFS is about iterative containers, clear invariants, and explicit early-stop conditions
- Prefer BFS for k-hop queries and subgraph extraction; prefer iterative DFS for path existence
- visited has no universal answer: HashSet, bitmap, and bloom each have boundaries
- Bloom has false positives; use it as "prefilter + exact confirmation," not as a standalone strict decision source
- Make search budgets (hop count, node budget, time budget) explicit parameters for stable production behavior

### Recommended Follow-up Reading

- LeetCode 200 (Number of Islands): graph traversal templates
- LeetCode 127 (Word Ladder): BFS + pruning
- Graph500 / graph computing benchmarks: ideas for large-scale traversal performance
- Classic Bloom-filter papers and engineering parameter sizing (false-positive rate vs bit-array size)

---

## Metadata

- **Reading time**: 12-16 minutes
- **Tags**: Graph, BFS, DFS, k-hop, subgraph extraction
- **SEO keywords**: BFS, DFS, k-hop query, path existence, visited bitmap, bloom filter
- **Meta description**: Engineering BFS/DFS templates with iterative implementation, early stop, visited bitmap/bloom selection, and runnable multi-language code.

---

## Call To Action (CTA)

I recommend locking in two actions immediately:

1. Refactor one online graph-query API to expose explicit early-stop parameters (hop budget, node budget, time budget)
2. Benchmark HashSet vs bitmap on real data (add bloom prefilter if needed), and record throughput and memory curves

If you want, I can write the next post:
"Union-Find + BFS/DFS selection checklist for graph problems (when to traverse vs when to merge)."

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


def bfs_k_hop(graph, s, k):
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


def dfs_path_exists(graph, s, t):
    vis = [False] * len(graph)
    st = [s]
    vis[s] = True
    while st:
        u = st.pop()
        if u == t:
            return True
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                st.append(v)
    return False


if __name__ == "__main__":
    g = [[1, 2], [3], [3, 4], [5], [], []]
    print(sorted(bfs_k_hop(g, 0, 2)))
    print(dfs_path_exists(g, 0, 5))
```

```c
#include <stdio.h>
#include <stdbool.h>

#define N 6

void bfs_k_hop(int g[N][N], int s, int k) {
    int q[128][2], head = 0, tail = 0;
    bool vis[N] = {0};
    vis[s] = true;
    q[tail][0] = s; q[tail][1] = 0; tail++;

    while (head < tail) {
        int u = q[head][0], d = q[head][1];
        head++;
        if (d == k) continue;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                q[tail][0] = v; q[tail][1] = d + 1; tail++;
            }
        }
    }

    for (int i = 0; i < N; ++i) if (vis[i]) printf("%d ", i);
    printf("\n");
}

bool dfs_path_exists(int g[N][N], int s, int t) {
    int st[128], top = 0;
    bool vis[N] = {0};
    st[top++] = s;
    vis[s] = true;

    while (top) {
        int u = st[--top];
        if (u == t) return true;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                st[top++] = v;
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

    bfs_k_hop(g, 0, 2);          // 0 1 2 3 4
    printf("%d\n", dfs_path_exists(g, 0, 5)); // 1
    return 0;
}
```

```cpp
#include <iostream>
#include <queue>
#include <vector>

std::vector<int> bfsKHop(const std::vector<std::vector<int>>& g, int s, int k) {
    std::vector<char> vis(g.size(), 0);
    std::queue<std::pair<int, int>> q;
    vis[s] = 1;
    q.push({s, 0});

    while (!q.empty()) {
        auto [u, d] = q.front();
        q.pop();
        if (d == k) continue;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                q.push({v, d + 1});
            }
        }
    }

    std::vector<int> out;
    for (int i = 0; i < (int)g.size(); ++i) if (vis[i]) out.push_back(i);
    return out;
}

bool dfsPathExists(const std::vector<std::vector<int>>& g, int s, int t) {
    std::vector<char> vis(g.size(), 0);
    std::vector<int> st = {s};
    vis[s] = 1;
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        if (u == t) return true;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                st.push_back(v);
            }
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    auto nodes = bfsKHop(g, 0, 2);
    for (int x : nodes) std::cout << x << " ";
    std::cout << "\n" << dfsPathExists(g, 0, 5) << "\n";
}
```

```go
package main

import "fmt"

func bfsKHop(graph [][]int, s, k int) []bool {
	vis := make([]bool, len(graph))
	type Node struct{ u, d int }
	q := []Node{{s, 0}}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		cur := q[head]
		if cur.d == k {
			continue
		}
		for _, v := range graph[cur.u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, Node{v, cur.d + 1})
			}
		}
	}
	return vis
}

func dfsPathExists(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	stack := []int{s}
	vis[s] = true
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				stack = append(stack, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(bfsKHop(g, 0, 2))
	fmt.Println(dfsPathExists(g, 0, 5))
}
```

```rust
use std::collections::VecDeque;

fn bfs_k_hop(graph: &Vec<Vec<usize>>, s: usize, k: usize) -> Vec<bool> {
    let mut vis = vec![false; graph.len()];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    vis[s] = true;
    q.push_back((s, 0));

    while let Some((u, d)) = q.pop_front() {
        if d == k {
            continue;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                q.push_back((v, d + 1));
            }
        }
    }
    vis
}

fn dfs_path_exists(graph: &Vec<Vec<usize>>, s: usize, t: usize) -> bool {
    let mut vis = vec![false; graph.len()];
    let mut st = vec![s];
    vis[s] = true;

    while let Some(u) = st.pop() {
        if u == t {
            return true;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                st.push(v);
            }
        }
    }
    false
}

fn main() {
    let graph = vec![vec![1, 2], vec![3], vec![3, 4], vec![5], vec![], vec![]];
    println!("{:?}", bfs_k_hop(&graph, 0, 2));
    println!("{}", dfs_path_exists(&graph, 0, 5));
}
```

```javascript
function bfsKHop(graph, s, k) {
  const vis = Array(graph.length).fill(false);
  const q = [[s, 0]];
  let head = 0;
  vis[s] = true;

  while (head < q.length) {
    const [u, d] = q[head++];
    if (d === k) continue;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        q.push([v, d + 1]);
      }
    }
  }
  return vis;
}

function dfsPathExists(graph, s, t) {
  const vis = Array(graph.length).fill(false);
  const st = [s];
  vis[s] = true;

  while (st.length) {
    const u = st.pop();
    if (u === t) return true;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        st.push(v);
      }
    }
  }
  return false;
}

const g = [[1, 2], [3], [3, 4], [5], [], []];
console.log(bfsKHop(g, 0, 2));
console.log(dfsPathExists(g, 0, 5));
```
