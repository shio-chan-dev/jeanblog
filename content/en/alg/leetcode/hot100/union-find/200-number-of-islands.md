---
title: "LeetCode 200: Number of Islands With Union-Find"
date: 2026-07-02T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Union-Find", "DSU", "graph", "matrix", "connected components", "LeetCode 200"]
description: "Solve LeetCode 200 Number of Islands in Python by deriving land-only count, 2D-to-1D ids, Union-Find merges, and exactly when count decreases."
keywords: ["LeetCode 200", "Number of Islands", "Union-Find", "DSU", "connected components", "matrix", "Python"]
---

> **Subtitle / Summary**
> Do not start this problem by memorizing a Union-Find template. Start from the pressure: each land cell looks like one island at first, but adjacent land cells must collapse into one connected component.

- **Reading time**: 10-12 min
- **Tags**: `Hot100`, `Union-Find`, `DSU`, `matrix`, `connected components`
- **SEO keywords**: LeetCode 200, Number of Islands, Union-Find, DSU, connected components
- **Meta description**: A pressure-first Python guide to LeetCode 200 that derives land-only count, 2D-to-1D ids, Union-Find merges, and the final runnable solution.

---

## Problem Requirement

You are given an `m x n` 2D character grid:

- `"1"` means land
- `"0"` means water

Return the number of islands.

An island is formed by land cells connected horizontally or vertically. Diagonal contact does not connect islands. You may assume the four edges of the grid are surrounded by water.

### Input and Output

- Input: `grid: List[List[str]]`
- Output: the number of islands as an `int`
- Only four directions matter: up, down, left, and right.
- Water cells never count as island components.

### Examples

```text
Input:
[
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

Output: 1
```

All land cells are connected through four-direction moves, so there is one island.

```text
Input:
[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

Output: 3
```

There are three disconnected land groups, so the answer is `3`.

### Constraints

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` is `"0"` or `"1"`

DFS and BFS are both valid approaches. This tutorial uses Union-Find because the training goal is to model the answer as connected component changes.

## Step 1: Do Not Count Land Cells Directly

Start with a tiny task.

```text
grid = [
  ["1", "1"],
  ["0", "1"]
]
```

If we only count `"1"` cells, we get `3`.

But these three land cells are connected:

```text
(0,0) -- (0,1)
           |
         (1,1)
```

They form one island, so the answer should be `1`.

The current baseline is:

```text
Whenever we see a land cell, add one to the answer.
```

This baseline breaks when land cells are adjacent.

The exact break is:

> One island may contain multiple land cells. Counting land cells splits one island into many islands.

Now compare another 2x2 grid:

```text
grid = [
  ["1", "0"],
  ["0", "1"]
]
```

The two land cells touch only diagonally, so they are not connected. The answer is `2`.

The change in this step is conceptual:

> Number of islands = number of four-direction connected components among land cells.

Check this step:

```text
[["1","1"],["0","1"]] -> 1
[["1","0"],["0","1"]] -> 2
```

Now this version can:

- avoid confusing land-cell count with island count
- distinguish four-direction connection from diagonal contact
- define the target as connected component count among land cells

It still lacks:

- a way to represent land cells as nodes that Union-Find can maintain

## Step 2: Give Only Land Cells an Initial Count

The current baseline is:

```text
Number of islands = number of connected components among land cells.
```

Union-Find uses an array to store each node's parent. But grid positions are two-dimensional coordinates like `(r, c)`.

This baseline breaks because:

> Union-Find needs one-dimensional node ids, and water cells must not enter the initial island count.

First map each grid coordinate to one flat id.

If the grid has `n` columns:

```python
def cell_id(r: int, c: int) -> int:
    return r * n + c
```

For a 2x3 grid:

```text
(0,0)->0  (0,1)->1  (0,2)->2
(1,0)->3  (1,1)->4  (1,2)->5
```

Now initialize the state:

```python
m, n = len(grid), len(grid[0])
parent = list(range(m * n))
count = 0
```

`parent` may reserve a slot for every cell, but `count` must not start from `m * n`.

Water is not an island.

So the initial `count` only counts land:

```python
for r in range(m):
    for c in range(n):
        if grid[r][c] == "1":
            count += 1
```

Check with this small grid:

```text
grid = [
  ["1", "0", "1"],
  ["0", "1", "0"]
]
```

The land cells and ids are:

```text
(0,0) -> 0
(0,2) -> 2
(1,1) -> 4
```

Before any merges:

```text
count = 3
```

Now this version can:

- assign every cell a stable one-dimensional id
- count only land cells as initial island candidates
- treat every land cell as one separate component at the beginning

It still lacks:

- the merge rule for adjacent land cells

## Step 3: Merge Adjacent Land Cells

The current baseline is:

```text
Every land cell starts as one island candidate.
```

This breaks on a 2x2 all-land grid:

```text
grid = [
  ["1", "1"],
  ["1", "1"]
]
```

The initial `count` is `4`, but the answer should be `1`.

The break is:

> Adjacent land cells belong to the same connected component. Initial count alone keeps overcounting.

Add the two Union-Find operations.

`find(x)` returns the representative of `x`'s current component:

```python
def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

The `find` invariant is:

> `find(x)` returns the representative of the component containing `x`; path compression shortens the route to the representative without changing connectivity.

`union(a, b)` merges the components containing two cells:

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

The return value answers the important question:

```text
Did this operation really merge two different islands?
```

If `root_a == root_b`, the two cells are already in the same island, so `count` must not decrease.

If `root_a != root_b`, two different islands just became one island:

```python
if union(a, b):
    count -= 1
```

Trace the 2x2 all-land grid:

```text
initial count = 4

merge (0,0) with (0,1): count 4 -> 3
merge (0,0) with (1,0): count 3 -> 2
merge (0,1) with (1,1): count 2 -> 1
already-connected neighbor pairs do not decrease count again
```

In the final scan, we can check only right and down neighbors to avoid duplicate undirected edges.

Now this version can:

- decide whether two land cells already belong to the same island
- merge adjacent land components
- decrease `count` only on a real merge

It still lacks:

- the full grid scan and the LeetCode wrapper

## Step 4: Scan the Grid and Submit

The current baseline is:

```text
We have cell_id, find, union, and the rule count -= 1 only on a real merge.
```

This breaks because:

> We still have not fed every adjacent-land relation into Union-Find, and the code is not yet in the required `Solution.numIslands` shape.

For each land cell, check only two directions:

```text
right: (r, c + 1)
down:  (r + 1, c)
```

Why only right and down?

An undirected neighbor pair only needs to be processed once. If `(0,0)` checks its right neighbor `(0,1)`, then `(0,1)` does not need to check left back to `(0,0)`.

Complete code:

```python
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        parent = list(range(m * n))
        count = 0

        def cell_id(r: int, c: int) -> int:
            return r * n + c

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

        for r in range(m):
            for c in range(n):
                if grid[r][c] == "1":
                    count += 1

        for r in range(m):
            for c in range(n):
                if grid[r][c] != "1":
                    continue

                current = cell_id(r, c)

                if c + 1 < n and grid[r][c + 1] == "1":
                    if union(current, cell_id(r, c + 1)):
                        count -= 1

                if r + 1 < m and grid[r + 1][c] == "1":
                    if union(current, cell_id(r + 1, c)):
                        count -= 1

        return count
```

The scan invariant is:

> After processing a land cell's right and down neighbors, every adjacent-land relation seen so far is reflected in Union-Find; `count` equals the number of land connected components under those processed relations.

After all neighbor relations have been processed, `count` is the number of islands.

Check the code:

```python
def check() -> None:
    s = Solution()

    assert s.numIslands([
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]) == 1

    assert s.numIslands([
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]) == 3

    assert s.numIslands([["0", "0"], ["0", "0"]]) == 0
    assert s.numIslands([["1", "1"], ["1", "1"]]) == 1
    assert s.numIslands([["1", "0"], ["0", "1"]]) == 2


check()
```

Now this version can:

- keep water out of the initial island count
- merge horizontally or vertically adjacent land
- keep diagonal land separate
- decrement `count` only when two different components really merge

## Complexity

Let the grid size be `m x n`.

- Time complexity: `O(mn * alpha(mn))`, where `alpha` is the inverse Ackermann function and is effectively constant in practice.
- Space complexity: `O(mn)` for the `parent` array.

## Summary

For the Union-Find version of Number of Islands, keep this chain fixed:

```text
number of islands
= number of connected components among land cells
= initial land count
- number of real adjacent-land merges
```

So the `count` rule is:

- each land cell starts as one island candidate
- adjacent land cells try to `union`
- `count` decreases only when `union` merges two different components
