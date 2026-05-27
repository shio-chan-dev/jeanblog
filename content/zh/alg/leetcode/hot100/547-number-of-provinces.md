---
title: "LeetCode 547：省份数量，把邻接矩阵看成图的连通分量"
date: 2026-05-27T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "图", "DFS", "BFS", "并查集", "连通分量", "LeetCode 547"]
description: "从邻接矩阵的直接连接关系出发，把 LeetCode 547 省份数量转成无向图连通分量计数问题，再逐步推出 DFS、BFS 和并查集解法。"
keywords: ["LeetCode 547", "省份数量", "Number of Provinces", "连通分量", "DFS", "BFS", "并查集"]
---

## 题目要求

给你一个 `n x n` 的矩阵 `isConnected`，其中有 `n` 个城市。

如果 `isConnected[i][j] == 1`，说明城市 `i` 和城市 `j` **直接相连**；如果两个城市可以通过若干个直接相连的城市互相到达，它们就属于同一个省份。

题目要求返回省份的总数。

这里最容易误解的一点是：题目不是让我们数矩阵里有多少个 `1`，也不是只看直接相连的城市对。它真正要数的是：

> 直接或间接连接在一起的城市组有多少个。

换成图的语言，就是：

> 给定一个无向图的邻接矩阵，返回这个图的连通分量数量。

### 输入输出

- 输入：`isConnected: List[List[int]]`
- 输出：省份数量 `int`
- 城市编号可以按 `0..n-1` 理解。
- `isConnected[i][j] == 1` 表示城市 `i` 和城市 `j` 之间有边。
- `isConnected[i][j] == 0` 表示城市 `i` 和城市 `j` 没有直接边。

### 示例 1

```text
输入：isConnected = [
  [1, 1, 0],
  [1, 1, 0],
  [0, 0, 1]
]
输出：2
```

城市 `0` 和城市 `1` 直接相连，所以它们属于同一个省份。

城市 `2` 只和自己相连，不能通过城市 `0` 或城市 `1` 到达，所以它单独属于另一个省份。

因此一共有两个省份：

```text
{0, 1}
{2}
```

答案是 `2`。

### 示例 2

```text
输入：isConnected = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
]
输出：3
```

三个城市之间都没有直接连接。每个城市只能和自己连通，所以每个城市都是一个独立省份：

```text
{0}
{1}
{2}
```

答案是 `3`。

### 约束

- `1 <= n <= 200`
- `isConnected.length == n`
- `isConnected[i].length == n`
- `isConnected[i][j]` 只会是 `0` 或 `1`
- `isConnected[i][i] == 1`
- `isConnected[i][j] == isConnected[j][i]`

### 这一节先冻结什么

现在我们已经把题目从“矩阵问题”翻译成了“图的连通分量计数问题”。

当前 checkpoint 能做到：

- 知道矩阵里的 `1` 表示直接连接。
- 知道省份允许通过中间城市间接连接。
- 知道目标不是数边，而是数连通分量。
- 能解释两个官方示例为什么分别输出 `2` 和 `3`。

它还缺：

- 如何从一个城市找出它所在的完整省份。
- 如何避免重复统计同一个省份。
- 如何把这个思路写成可运行代码。

## Step 1：先解决一个更小的问题

现在的 baseline 是：我们已经知道目标是数连通分量，但还没有办法“拿到一个省份”。

直接问“总共有多少省份”还是太大。先把问题缩小：

> 给定一个还没处理过的城市，怎么找出和它属于同一个省份的所有城市？

看示例 1：

```text
isConnected = [
  [1, 1, 0],
  [1, 1, 0],
  [0, 0, 1]
]
```

如果从城市 `0` 开始：

- `isConnected[0][0] == 1`，城市 `0` 属于当前省份。
- `isConnected[0][1] == 1`，城市 `1` 和城市 `0` 直接相连，也属于当前省份。
- `isConnected[0][2] == 0`，城市 `2` 不能从城市 `0` 直接到达。

继续看城市 `1` 的连接：

- `isConnected[1][0] == 1`，城市 `0` 已经在当前省份里。
- `isConnected[1][1] == 1`，城市 `1` 已经在当前省份里。
- `isConnected[1][2] == 0`，城市 `2` 仍然不属于当前省份。

所以从城市 `0` 出发，我们能找到完整省份：

```text
{0, 1}
```

这一步的关键不是最终答案，而是一个中间能力：

> 从一个城市出发，把所有能直接或间接到达的城市都标记出来。

为了做到这件事，需要一个 `visited` 数组：

```text
visited[i] == True  表示城市 i 已经被某个省份处理过
visited[i] == False 表示城市 i 还没有归入任何已发现的省份
```

如果我们从一个未访问城市出发，把能到达的城市全部标记成已访问，那么这一次遍历就找到了一个完整省份。

可以先把这个子问题写成伪代码：

```text
visit(city):
    标记 city 已访问
    for next_city in 所有城市:
        如果 city 和 next_city 相连，并且 next_city 没访问过:
            visit(next_city)
```

这就是图遍历的入口。它可以用 DFS 写，也可以用 BFS 写。

这一节先只冻结这个子问题，不急着写完整答案。

当前 checkpoint 能做到：

- 知道要从“找一个省份”开始，而不是直接写最终计数。
- 知道 `visited` 的作用是避免重复处理城市。
- 知道一次遍历会把同一省份里的城市全部标记掉。

它还缺：

- 怎么把 `visit(city)` 写成 Python 代码。
- 怎么扫描所有城市并统计省份数量。

## Step 2：用 DFS 找到每一个省份

上一节的 baseline 是：

```text
visit(city) 可以从一个城市出发，标记同一个省份里的所有城市。
```

但它还不能回答最终问题。因为题目要的是省份总数，而不是只找出城市 `0` 所在的省份。

这会在示例 1 里暴露出来：

```text
{0, 1}
{2}
```

如果我们只从城市 `0` 出发，只能得到 `{0, 1}`，城市 `2` 永远不会被处理。

所以需要在上一版基础上增加一个外层扫描：

> 从左到右检查每个城市。如果这个城市还没访问过，说明发现了一个新省份，于是答案加一，并从它开始 DFS 标记整个省份。

这个逻辑可以直接写成 LeetCode 代码：

```python
from typing import List


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n

        def dfs(city: int) -> None:
            visited[city] = True
            for next_city in range(n):
                if isConnected[city][next_city] == 1 and not visited[next_city]:
                    dfs(next_city)

        provinces = 0
        for city in range(n):
            if not visited[city]:
                provinces += 1
                dfs(city)

        return provinces
```

这段代码里有两个动作。

第一个动作是 `dfs(city)`：

```python
def dfs(city: int) -> None:
    visited[city] = True
    for next_city in range(n):
        if isConnected[city][next_city] == 1 and not visited[next_city]:
            dfs(next_city)
```

它只负责一件事：从 `city` 出发，把同一个省份里的城市全部标记成已访问。

第二个动作是外层扫描：

```python
provinces = 0
for city in range(n):
    if not visited[city]:
        provinces += 1
        dfs(city)
```

这里的 `if not visited[city]` 很关键。

如果一个城市已经访问过，说明它已经属于之前某个省份，不能再重复计数。

如果一个城市还没访问过，说明之前发现的所有省份都到不了它。它一定属于一个新省份，所以先 `provinces += 1`，再用 `dfs(city)` 把这个新省份整体标记掉。

用示例 1 手动走一遍：

```text
visited = [False, False, False]
provinces = 0

city = 0:
  没访问过，provinces = 1
  dfs(0) 标记 0 和 1
  visited = [True, True, False]

city = 1:
  已访问过，跳过

city = 2:
  没访问过，provinces = 2
  dfs(2) 标记 2
  visited = [True, True, True]
```

最后返回 `2`。

再看示例 2：

```text
isConnected = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
]
```

每个城市都无法到达其他城市，所以外层扫描会在城市 `0`、`1`、`2` 各启动一次 DFS，答案就是 `3`。

当前 checkpoint 能做到：

- 用 DFS 从一个城市找出完整省份。
- 用外层扫描统计所有省份。
- 正确处理连在一起的城市和孤立城市。

它还缺：

- 为什么这个计数一定不会漏、不重。
- 时间复杂度和空间复杂度是多少。

## Step 3：为什么 DFS 计数不会漏也不会重

现在我们已经有了一份能通过示例的 DFS 代码。但只会跑示例还不够，因为这道题真正容易错的地方不是语法，而是计数逻辑：

```python
for city in range(n):
    if not visited[city]:
        provinces += 1
        dfs(city)
```

为什么看到一个未访问城市时，就可以立刻把 `provinces` 加一？

先看一个错误想法：每遇到一条连接边，就把答案加一。

比如：

```text
0 -- 1 -- 2
```

在矩阵里可能看到 `0` 连 `1`，`1` 连 `2`。如果按边来数，会得到两个连接关系。但这三个城市其实属于同一个省份，答案应该是 `1`。

所以省份不是“边的数量”，而是“从一个入口能扩散到的一整组城市”。

DFS 代码依赖三个不变量。

第一个不变量：

> `dfs(city)` 结束后，所有能从 `city` 到达的城市都会被标记为 `visited = True`。

原因是 `dfs` 会检查 `city` 到每个 `next_city` 的连接。如果有边，并且 `next_city` 没访问过，就继续对 `next_city` 做 DFS。这样一层一层扩散，直接连接和间接连接都会被覆盖。

第二个不变量：

> 如果外层扫描看到 `visited[city] == False`，那么 `city` 不属于之前发现过的任何省份。

因为之前每发现一个省份，都会立刻调用 DFS 把这个省份里的所有城市都标记掉。如果 `city` 仍然没被标记，说明之前所有 DFS 都到不了它。它必须开启一个新省份。

第三个不变量：

> 一个省份只会被计数一次。

一个省份第一次被外层扫描遇到时，答案加一，然后 DFS 把整个省份标记为已访问。之后外层扫描再遇到这个省份里的其他城市，它们已经是 `visited = True`，会被跳过。

把这三个不变量合起来，就是这段代码的正确性：

- 不会漏：外层扫描会检查每个城市。任何未访问城市都会启动一次 DFS。
- 不会重：一个城市被 DFS 标记后，后面不会再触发新的计数。
- 数的是省份：每次计数都对应一个完整连通分量，而不是一条边或一个矩阵里的 `1`。

### 复杂度

`isConnected` 是邻接矩阵。对一个城市做 DFS 时，代码会扫描这一整行：

```python
for next_city in range(n):
```

每个城市最多真正进入 `dfs` 一次，因为进入后会被标记为 `visited`。每次进入都扫描 `n` 个可能邻居，所以总时间复杂度是：

```text
O(n * n) = O(n^2)
```

这里不要写成 `O(n + m)`。`O(n + m)` 更适合邻接表。题目给的是矩阵，矩阵本身就有 `n^2` 个位置，扫描成本按矩阵大小计算更直接。

空间复杂度来自两部分：

- `visited` 数组：`O(n)`
- DFS 递归栈：最坏情况下所有城市连成一条链，递归深度 `O(n)`

所以空间复杂度是：

```text
O(n)
```

当前 checkpoint 能做到：

- 解释为什么看到未访问城市时可以把省份数加一。
- 解释为什么 DFS 不会漏掉间接连接城市。
- 解释为什么同一个省份不会被重复统计。
- 给出邻接矩阵下的 `O(n^2)` 时间复杂度和 `O(n)` 空间复杂度。

它还缺：

- 如果不想用递归，怎么把同样的图遍历写成 BFS。
- 如果想从“合并集合”的角度理解，怎么写并查集。
