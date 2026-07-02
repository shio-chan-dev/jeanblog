---
title: "LeetCode 684: Redundant Connection With Union-Find"
date: 2026-07-02T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Union-Find", "DSU", "graph", "tree", "cycle detection", "LeetCode 684"]
description: "Solve LeetCode 684 Redundant Connection in Python by deriving the already-connected edge pressure, 1-indexed parent array, bool-returning union, and ordered edge scan."
keywords: ["LeetCode 684", "Redundant Connection", "Union-Find", "DSU", "cycle detection", "graph", "Python"]
---

> **Subtitle / Summary**
> The key signal in this problem is not a component count. It is a failed `union`: if two endpoints are already connected, adding the current edge closes a cycle.

- **Reading time**: 8-10 min
- **Tags**: `Union-Find`, `DSU`, `graph`, `tree`, `cycle detection`
- **SEO keywords**: LeetCode 684, Redundant Connection, Union-Find, DSU, cycle detection
- **Meta description**: A pressure-first Python guide to LeetCode 684 that derives 1-indexed Union-Find, bool-returning union, ordered edge scanning, and final checks.

---

## Problem Requirement

You are given an undirected graph. It started as a tree with `n` nodes labeled from `1` to `n`, then one extra edge was added.

A tree is:

- connected
- acyclic

After adding one extra edge, the graph is still connected, but now it contains a cycle.

The input is `edges`, where `edges[i] = [a, b]` means there is an undirected edge between nodes `a` and `b`.

Return an edge that can be removed so the graph becomes a tree again.

If multiple answers are possible, return the one that appears last in the input.

### Input and Output

- Input: `edges: List[List[int]]`
- Output: one edge as `List[int]`
- `n == len(edges)`
- nodes are labeled `1..n`
- there are no repeated edges
- the given graph is connected

### Examples

```text
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]
```

The first two edges form a tree:

```text
1 - 2
|
3
```

Before adding `[2,3]`, node `2` can already reach node `3` through `2 -> 1 -> 3`. Adding a direct edge closes a cycle.

```text
Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
Output: [1,4]
```

Before adding `[1,4]`, node `1` can already reach node `4` through `1 -> 2 -> 3 -> 4`, so `[1,4]` is redundant.

### Constraints

- `n == edges.length`
- `3 <= n <= 1000`
- `edges[i].length == 2`
- `1 <= ai < bi <= edges.length`
- `ai != bi`
- there are no repeated edges
- the graph is connected

## Step 1: When Does an Edge Become Redundant?

Start with the smallest useful example:

```text
edges = [[1,2],[1,3],[2,3]]
```

Process the edges in order.

Add `[1,2]`:

```text
1 - 2
3
```

No cycle.

Add `[1,3]`:

```text
2 - 1 - 3
```

Still a tree.

Now prepare to add `[2,3]`.

The current baseline is:

```text
Whenever we see an edge, add it to the graph.
```

This baseline breaks because:

> If the two endpoints are already connected before adding the edge, the edge creates a cycle.

In this example, before adding `[2,3]`:

```text
2 -> 1 -> 3
```

So `2` and `3` are already connected. Edge `[2,3]` is redundant.

The change in this step is the question we ask:

> For every edge `[a, b]`, before adding it, ask whether `a` and `b` are already connected.

If they are not connected, the edge is safe and should merge two components.

If they are already connected, this edge creates a cycle and is the answer.

Check this step:

```text
[1,2]: 1 and 2 are not connected, safe
[1,3]: 1 and 3 are not connected, safe
[2,3]: 2 and 3 are already connected, return [2,3]
```

Now this version can:

- define redundancy as "the endpoints were already connected"
- reduce the problem to dynamic connectivity

It still lacks:

- a data structure that can answer connectivity while edges are scanned

## Step 2: Use parent to Maintain Components

The current baseline is:

```text
Before adding [a, b], ask whether a and b are already connected.
```

This breaks because:

> We do not yet maintain which component each node belongs to.

Union-Find is the right state for that question.

Because nodes are labeled from `1` to `n`, create a parent array of size `n + 1`:

```python
n = len(edges)
parent = list(range(n + 1))
```

Now `parent[1]` through `parent[n]` correspond directly to real nodes. `parent[0]` is unused.

At the beginning, every node is its own representative:

```text
parent[1] = 1
parent[2] = 2
parent[3] = 3
```

`find(x)` returns the representative of the component containing `x`:

```python
def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

The `find` invariant is:

> As we follow parent pointers, `x` remains inside the same connected component; the self-parent node we reach is the representative of that component.

Path compression only shortens future routes to the representative. It does not change which nodes are connected.

Now this version can:

- allocate DSU state for nodes `1..n`
- ask for the representative of any node

It still lacks:

- a way to merge components when a safe edge appears
- a way for the merge operation to tell us whether the edge created a cycle

## Step 3: Make union Report Cycle Pressure

The current baseline is:

```text
find(x) returns the representative of x's component.
```

Now process one edge `[a, b]`.

This baseline breaks because:

> Knowing roots is not enough. We need to turn root comparison into a decision: safe edge or cycle edge.

Write `union` so it returns a boolean:

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

The return value is the contract:

- `True`: the endpoints were in different components, so the edge is safe and the merge happened
- `False`: the endpoints were already connected, so this edge creates a cycle

Check the triangle example:

```text
edges = [[1,2],[1,3],[2,3]]
```

Process `[1,2]`:

```text
find(1) != find(2)
union(1,2) -> True
```

Process `[1,3]`:

```text
find(1) != find(3)
union(1,3) -> True
```

Process `[2,3]`:

```text
find(2) == find(3)
union(2,3) -> False
```

So `[2,3]` is the answer.

Now this version can:

- merge two components for safe edges
- return `False` for cycle-closing edges
- use failed `union` as the redundant-edge signal

It still lacks:

- scanning the whole input in the order the problem gives it

## Step 4: Scan Edges in Input Order

The current baseline is:

```text
union(a, b) can decide whether one edge creates a cycle.
```

This breaks because:

> The input is an entire edge list, and the problem asks for the returned edge. If multiple answers are possible, return the one that appears last in the input.

Under this problem's promise, the graph is a tree plus one extra edge. If we rebuild the graph from an empty DSU and scan edges in input order:

- before a cycle appears, each edge connects two different components
- the first edge whose endpoints are already connected is the edge that closes the cycle in this input order

The scan rule is short:

```python
for a, b in edges:
    if not union(a, b):
        return [a, b]
```

Check example 2:

```text
edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
```

Process in order:

```text
[1,2]: merge succeeds
[2,3]: merge succeeds
[3,4]: merge succeeds
[1,4]: 1 and 4 are already connected through 1-2-3-4, union fails
```

Return `[1,4]`.

We do not need to inspect `[1,5]` after that, because the problem guarantees this graph is a tree with one additional edge.

Now this version can:

- scan edges in order
- return the current edge when `union` fails
- explain why example 2 returns `[1,4]`

It still lacks:

- the final LeetCode wrapper, checks, and complexity

## Step 5: Complete Code and Verification

The current baseline is:

```text
parent + find + union + ordered scan
```

This breaks because:

> The code still needs to be packaged as `Solution.findRedundantConnection`.

Complete code:

```python
from typing import List


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        parent = list(range(n + 1))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> bool:
            root_a = find(a)
            root_b = find(b)

            if root_a == root_b:
                return False

            parent[root_b] = root_a
            return True

        for a, b in edges:
            if not union(a, b):
                return [a, b]

        return []
```

The scan invariant is:

> At the start of each loop, every previous edge that did not return has been safely added to the DSU. If the current edge has endpoints with the same root, adding it would create a cycle.

Check the code:

```python
def check() -> None:
    s = Solution()

    assert s.findRedundantConnection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert s.findRedundantConnection([
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 4],
        [1, 5],
    ]) == [1, 4]
    assert s.findRedundantConnection([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 5],
    ]) == [1, 5]


check()
```

Now this version can:

- align the parent array with `1..n` labels by using `n + 1`
- use `find` to answer whether two endpoints are in the same component
- use the `union` return value to separate safe edges from redundant edges
- return the first failed union while scanning input order

## Complexity

Let `n = len(edges)`.

- Time complexity: `O(n * alpha(n))`, where `alpha` is the inverse Ackermann function and is effectively constant in practice.
- Space complexity: `O(n)` for the `parent` array.

## Summary

LeetCode 684 is not a count-based Union-Find problem.

The core signal is:

```text
union(a, b) fails
```

That means:

```text
a and b were already connected
adding [a, b] would create a cycle
the current edge is redundant
```

So keep the `union` return contract precise:

- `True`: merge succeeded, edge is safe
- `False`: endpoints were already connected, edge is redundant
