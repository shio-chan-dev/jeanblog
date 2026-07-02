---
title: "LeetCode 684：冗余连接，用 union 失败找到成环边"
date: 2026-07-02T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["并查集", "Union-Find", "DSU", "图", "树", "环检测", "LeetCode 684"]
description: "从树加一条边必然成环的压力出发，用并查集推导 LeetCode 684 冗余连接，讲清 1-indexed parent、union 返回值和为什么 union 失败就是答案。"
keywords: ["LeetCode 684", "冗余连接", "Redundant Connection", "并查集", "Union-Find", "环检测", "图"]
---

## 题目要求

题目给你一个无向图。这个图原本是一棵有 `n` 个节点的树，节点编号是 `1..n`，后来额外加了一条边。

树的定义是：

- 连通
- 没有环

加上一条额外边之后，图仍然连通，但会出现一个环。

现在给定边数组 `edges`，其中 `edges[i] = [a, b]` 表示节点 `a` 和节点 `b` 之间有一条无向边。题目要求返回一条可以删除的边，使剩下的图重新变成树。

如果有多个答案，返回在输入中最后出现的那条。

### 输入输出

- 输入：`edges: List[List[int]]`
- 输出：一条边 `List[int]`
- `n == len(edges)`
- 节点编号是 `1..n`
- 图中没有重复边
- 给定图是连通的

### 示例

```text
输入：edges = [[1,2],[1,3],[2,3]]
输出：[2,3]
```

前两条边形成一棵树：

```text
1 - 2
|
3
```

再加入 `[2,3]`，`2` 和 `3` 之间已经能通过 `2 -> 1 -> 3` 连通。现在再加直接边，就形成环。

```text
输入：edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
输出：[1,4]
```

加入 `[1,4]` 之前，`1` 和 `4` 已经能通过 `1 -> 2 -> 3 -> 4` 连通，所以 `[1,4]` 是冗余边。

### 约束

- `n == edges.length`
- `3 <= n <= 1000`
- `edges[i].length == 2`
- `1 <= ai < bi <= edges.length`
- `ai != bi`
- 没有重复边
- 图是连通的

## Step 1：什么时候一条边会多余？

先只看最小例子：

```text
edges = [[1,2],[1,3],[2,3]]
```

按顺序加入边。

加入 `[1,2]`：

```text
1 - 2
3
```

没有环。

加入 `[1,3]`：

```text
2 - 1 - 3
```

还是一棵树，没有环。

现在准备加入 `[2,3]`。

当前 baseline 是：

```text
只要看到一条边，就把它加入图。
```

这个 baseline 的 break 是：

> 如果边的两个端点在加入之前已经连通，再加这条边就会形成环。

在这个例子里，加入 `[2,3]` 之前：

```text
2 -> 1 -> 3
```

`2` 和 `3` 已经连通。所以 `[2,3]` 是多余的。

这一步要改变的是问题问法：

> 对每条边 `[a, b]`，在加入它之前，先问 `a` 和 `b` 是否已经连通。

如果不连通，这条边是安全的，可以合并两个连通块。

如果已经连通，这条边会制造环，就是答案。

检查这一步：

```text
[1,2]：1 和 2 未连通，可以加入
[1,3]：1 和 3 未连通，可以加入
[2,3]：2 和 3 已连通，返回 [2,3]
```

现在这一版能做到：

- 知道冗余边不是看边本身，而是看加入之前端点是否已经连通。
- 知道问题核心是动态维护连通性。

它还缺：

- 怎样快速回答“两个节点是否已经连通”。

## Step 2：用 parent 维护连通块

当前 baseline 是：

```text
加入边 [a, b] 前，先判断 a 和 b 是否已连通。
```

break 是：

> 现在还没有数据结构维护节点属于哪个连通块。

并查集正好维护这个信息。

因为节点编号是 `1..n`，所以 `parent` 开到 `n + 1`：

```python
n = len(edges)
parent = list(range(n + 1))
```

这样 `parent[1]` 到 `parent[n]` 正好对应真实节点，`parent[0]` 不用。

一开始每个节点都是自己的连通块代表：

```text
parent[1] = 1
parent[2] = 2
parent[3] = 3
```

`find(x)` 返回 `x` 当前所在连通块的代表：

```python
def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

`find` 的 invariant 是：

> 沿着 `parent` 往上走时，`x` 始终没有离开原来的连通块；最后停下来的自指节点就是这个连通块的代表。

路径压缩只把节点更直接地挂到代表下面，不改变哪些点互相连通。

这一步之后，当前版本能做到：

- 给 `1..n` 的节点准备并查集状态。
- 用 `find(x)` 查询节点所在连通块代表。

它还缺：

- 看到一条边时，怎样把两个连通块合并。
- 怎样让合并动作告诉我们“这条边是否成环”。

## Step 3：让 union 告诉我们是否成环

当前 baseline 是：

```text
find(x) 可以返回 x 所在连通块的代表。
```

现在处理一条边 `[a, b]`。

break 是：

> 只会 find 还不够。我们需要把“根是否相同”变成一个明确决策：安全加入，还是形成环。

写一个返回布尔值的 `union`：

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

返回值语义很重要：

- `True`：两个端点原来不连通，这条边安全，合并成功
- `False`：两个端点原来已经连通，这条边会成环

用三角形例子检查：

```text
edges = [[1,2],[1,3],[2,3]]
```

处理 `[1,2]`：

```text
find(1) != find(2)
union(1,2) -> True
```

处理 `[1,3]`：

```text
find(1) != find(3)
union(1,3) -> True
```

处理 `[2,3]`：

```text
find(2) == find(3)
union(2,3) -> False
```

所以 `[2,3]` 是答案。

现在这一版能做到：

- 安全边会合并两个连通块。
- 成环边会让 `union` 返回 `False`。
- `union` 失败就是冗余边信号。

它还缺：

- 按输入顺序处理所有边，并返回题目要求的那条边。

## Step 4：按输入顺序扫描边

当前 baseline 是：

```text
union(a, b) 可以判断一条边是否会成环。
```

break 是：

> 题目给的是一整个 edges 数组，而且要求返回符合条件的边。如果有多个答案，要返回输入中最后出现的那条。

在这题的前提下，图是树加一条额外边。按输入顺序从空图开始加边时：

- 没有成环之前，每条边都把两个不同连通块连起来。
- 第一次遇到端点已连通的边，就是这条额外边在当前输入顺序下关闭了环。

所以扫描规则很短：

```python
for a, b in edges:
    if not union(a, b):
        return [a, b]
```

用示例 2 检查：

```text
edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
```

逐条处理：

```text
[1,2]：合并成功
[2,3]：合并成功
[3,4]：合并成功
[1,4]：1 和 4 已经通过 1-2-3-4 连通，union 失败
```

所以返回 `[1,4]`。

后面的 `[1,5]` 不需要再看，因为题目保证只有一条额外边造成这次冗余。

现在这一版能做到：

- 按输入顺序处理边。
- 在第一条 `union` 失败的边处返回答案。
- 解释为什么示例 2 返回 `[1,4]`。

它还缺：

- 完整 LeetCode 包装、测试和复杂度。

## Step 5：完整代码和验证

当前 baseline 是：

```text
parent + find + union + 顺序扫描
```

break 是：

> 还没有整理成 LeetCode 要求的 `Solution.findRedundantConnection`。

完整代码：

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

扫描循环的 invariant 是：

> 每次循环开始时，之前所有没有返回的边都已经安全加入并查集；如果当前边的两个端点已经同根，那么加入它会形成环。

检查：

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

现在这个版本能做到：

- 用 `n + 1` 的 `parent` 对齐 `1..n` 节点编号。
- 用 `find` 判断两个端点是否属于同一个连通块。
- 用 `union` 的返回值区分安全边和成环边。
- 按输入顺序返回第一条 `union` 失败的边。

## 复杂度

设 `n = len(edges)`。

- 时间复杂度：`O(n * alpha(n))`，其中 `alpha` 是反阿克曼函数，实际可以近似看成常数。
- 空间复杂度：`O(n)`，来自 `parent` 数组。

## 小结

684 题不是用并查集维护 `count`。

它的核心信号是：

```text
union(a, b) 失败
```

含义是：

```text
a 和 b 已经连通
再加 [a, b] 会形成环
当前边就是冗余边
```

所以写代码时要特别固定 `union` 的返回值语义：

- `True`：合并成功，边安全
- `False`：已经连通，边冗余
