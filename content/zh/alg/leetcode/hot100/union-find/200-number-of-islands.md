---
title: "LeetCode 200：岛屿数量，把网格陆地看成连通分量"
date: 2026-07-02T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "并查集", "Union-Find", "DSU", "图", "矩阵", "LeetCode 200"]
description: "从网格中相邻陆地的合并压力出发，用并查集推导 LeetCode 200 岛屿数量，讲清二维坐标映射、陆地初始 count、union 后 count 什么时候减少。"
keywords: ["LeetCode 200", "岛屿数量", "Number of Islands", "并查集", "Union-Find", "连通分量", "矩阵"]
---

## 题目要求

给你一个 `m x n` 的二维字符网格 `grid`：

- `"1"` 表示陆地
- `"0"` 表示水

题目要求返回岛屿数量。

一个岛屿由水平或垂直相邻的陆地组成。对角线相邻不算连通。可以认为网格四周都被水包围。

### 输入输出

- 输入：`grid: List[List[str]]`
- 输出：岛屿数量 `int`
- 只看上下左右四个方向。
- `"0"` 水格子不能算作岛屿的一部分。

### 示例

```text
输入：
[
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

输出：1
```

这些陆地通过上下左右连成一整块，所以答案是 `1`。

```text
输入：
[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

输出：3
```

这里有三块互不连通的陆地，所以答案是 `3`。

### 约束

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` 只会是 `"0"` 或 `"1"`

这一题可以用 DFS 或 BFS 做。这里的目标是练并查集：把每块陆地当成一个节点，把相邻陆地合并，最后留下的陆地连通分量数量就是岛屿数量。

## Step 1：先不要数陆地格子

先看一个很小的任务。

```text
grid = [
  ["1", "1"],
  ["0", "1"]
]
```

如果只数 `"1"`，会得到 `3`。

但这三个陆地格子通过上下左右连在一起：

```text
(0,0) -- (0,1)
           |
         (1,1)
```

所以它们是同一个岛，答案应该是 `1`。

当前 baseline 是：

```text
看到一个陆地格子，就把答案加一。
```

这个 baseline 会在相邻陆地上出错。

它的 break 是：

> 同一个岛可能包含多个陆地格子。数陆地格子会把一个岛拆成多个岛。

再看另一个 2x2 网格：

```text
grid = [
  ["1", "0"],
  ["0", "1"]
]
```

这里两个陆地只是在对角线上相邻，不属于上下左右连通，所以答案是 `2`。

这一步要改的不是代码，而是目标：

> 岛屿数量 = 陆地格子形成的四方向连通分量数量。

检查这一步：

```text
[["1","1"],["0","1"]] -> 1
[["1","0"],["0","1"]] -> 2
```

现在这一版能做到：

- 不再把岛屿理解成陆地格子数量。
- 知道只有上下左右相邻才会合并成同一个岛。
- 知道问题本质是数陆地连通分量。

它还缺：

- 怎么把网格里的陆地格子变成并查集能维护的节点。

## Step 2：只给陆地建立节点

当前 baseline 是：

```text
岛屿数量 = 陆地连通分量数量
```

并查集用数组维护节点的父节点。但网格里的位置是二维坐标，比如 `(r, c)`。

这个 baseline 的 break 是：

> 并查集需要一维下标；而且水格子不能进入初始岛屿数量。

先把二维坐标映射成一维编号。

如果网格有 `n` 列，那么：

```python
def cell_id(r: int, c: int) -> int:
    return r * n + c
```

例如一个 2x3 网格：

```text
(0,0)->0  (0,1)->1  (0,2)->2
(1,0)->3  (1,1)->4  (1,2)->5
```

接下来初始化两个东西：

```python
m, n = len(grid), len(grid[0])
parent = list(range(m * n))
count = 0
```

`parent` 可以给所有格子准备位置，但 `count` 不能从 `m * n` 开始。

因为水不是岛。

所以初始 `count` 只统计陆地：

```python
for r in range(m):
    for c in range(n):
        if grid[r][c] == "1":
            count += 1
```

用这个小网格检查：

```text
grid = [
  ["1", "0", "1"],
  ["0", "1", "0"]
]
```

陆地坐标和 id 是：

```text
(0,0) -> 0
(0,2) -> 2
(1,1) -> 4
```

初始时它们还没有合并，所以：

```text
count = 3
```

这一步之后，当前版本能做到：

- 给每个格子一个稳定的一维编号。
- 只把陆地计入初始岛屿数量。
- 把每块陆地先当成一个独立岛屿候选。

它还缺：

- 相邻陆地应该合并成同一个岛。

## Step 3：相邻陆地要合并

当前 baseline 是：

```text
每块陆地一开始都是一个独立岛屿候选。
```

这会在 2x2 全陆地上出错：

```text
grid = [
  ["1", "1"],
  ["1", "1"]
]
```

初始 `count = 4`，但答案应该是 `1`。

break 是：

> 相邻陆地属于同一个连通分量。只初始化 count，不合并相邻陆地，会持续过数。

现在加入并查集的两个动作。

`find(x)` 找到 `x` 所在集合的代表：

```python
def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

它的 invariant 是：

> `find(x)` 返回的是 `x` 当前所在连通分量的代表；路径压缩只改变指向代表的路径长度，不改变连通关系。

`union(a, b)` 合并两个格子所在的连通分量：

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

这里让 `union` 返回 `True / False`，是为了回答一个关键问题：

```text
这次合并有没有真的把两个不同岛屿变成一个岛屿？
```

如果 `root_a == root_b`，说明两个格子已经属于同一个岛。再处理这条相邻关系时，`count` 不能减少。

如果 `root_a != root_b`，说明两个不同岛屿被合并成一个岛，`count` 才减少：

```python
if union(a, b):
    count -= 1
```

用 2x2 全陆地手推：

```text
初始 count = 4

合并 (0,0) 和 (0,1): count 4 -> 3
合并 (0,0) 和 (1,0): count 3 -> 2
合并 (0,1) 和 (1,1): count 2 -> 1
再遇到已经连通的相邻关系时，不再减少
```

实际代码里我们可以只看右边和下边，避免重复处理同一对相邻格子。这样 2x2 全陆地也会从 `4` 降到 `1`。

这一步之后，当前版本能做到：

- 判断两个陆地格子是否已经属于同一个岛。
- 合并两个相邻陆地连通分量。
- 只在真实合并时减少 `count`。

它还缺：

- 把这些规则接到完整网格扫描和 LeetCode 提交接口里。

## Step 4：扫描网格并提交

当前 baseline 是：

```text
有 cell_id、find、union，并且知道真实合并时 count -= 1。
```

它的 break 是：

> 还没有把所有相邻陆地关系喂给 union，也还不是 LeetCode 要求的 `Solution.numIslands`。

现在只补最后一层扫描规则。

对每个陆地格子，只检查两个方向：

```text
右边: (r, c + 1)
下边: (r + 1, c)
```

为什么只检查右和下？

因为无向相邻关系不用处理两次。比如 `(0,0)` 和 `(0,1)`，从 `(0,0)` 看右边已经处理过，从 `(0,1)` 再看左边就是重复边。

完整代码：

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

扫描循环的 invariant 是：

> 每次处理完一个陆地格子的右边和下边之后，已经被扫描过的相邻陆地关系都已经反映在并查集里；`count` 等于当前已建立相邻关系下的陆地连通分量数量。

最后所有相邻关系都处理完，`count` 就是岛屿数量。

检查这份代码：

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

现在这个版本能做到：

- 水格子不进入初始岛屿数量。
- 上下左右相邻陆地会合并。
- 对角线陆地不会合并。
- `count` 只在两个不同连通分量真实合并时减少。

## 复杂度

设网格大小是 `m x n`。

- 时间复杂度：`O(mn * alpha(mn))`，其中 `alpha` 是反阿克曼函数，实际可以近似看成常数。
- 空间复杂度：`O(mn)`，来自 `parent` 数组。

## 小结

这题用并查集时，不要从模板开始背。

先抓住这条线：

```text
岛屿数量
= 陆地连通分量数量
= 初始陆地 count
- 相邻陆地真实合并次数
```

所以 `count` 的规则是：

- 看到陆地，初始多一个候选岛屿
- 看到相邻陆地，尝试 `union`
- 只有 `union` 真的合并了两个不同集合，`count` 才减少
