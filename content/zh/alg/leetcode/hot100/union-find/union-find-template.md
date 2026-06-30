---
title: "并查集模板：find / union / count 从零推导"
date: 2026-06-30T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "并查集", "Union-Find", "DSU", "图", "连通性", "模板"]
description: "从一个集合合并小任务出发，推导并查集模板中的 parent、find、union、count 和路径压缩，最终得到可直接背写的 Python 模板。"
keywords: ["并查集", "Union-Find", "DSU", "find", "union", "count", "路径压缩", "连通分量", "Hot100"]
---

> **副标题 / 摘要**
> 并查集不是从 `find` 和 `union` 这两个函数名开始背。它要解决的问题是：怎样判断两个点是否已经属于同一个集合，并在看到一条连接关系时把两个集合合并。

- **预计阅读时长**：10~12 分钟
- **标签**：`Hot100`、`并查集`、`Union-Find`、`DSU`、`图`
- **SEO 关键词**：并查集, Union-Find, DSU, find, union, count, 路径压缩
- **元描述**：用 Python 从零推导并查集模板，讲清 parent、find、union、count、路径压缩和连通分量计数。

---

## A — Algorithm（从集合合并压力开始）

### 小任务：不断合并集合，并回答连通性

假设有 `5` 个点：

```text
0, 1, 2, 3, 4
```

一开始，每个点都是一个独立集合：

```text
{0}, {1}, {2}, {3}, {4}
```

现在依次发生两次合并：

```text
union(0, 1)
union(1, 2)
```

我们想回答三个问题：

- `0` 和 `2` 是否在同一个集合？
- `3` 和 `4` 是否在同一个集合？
- 当前一共有几个集合？

手工看答案是：

```text
{0, 1, 2}, {3}, {4}
```

所以：

```text
0 和 2 在同一个集合
3 和 4 不在同一个集合
当前 count = 3
```

这就是并查集要解决的核心任务：

- `find(x)`：找到 `x` 所在集合的代表
- `union(a, b)`：把 `a` 和 `b` 所在的两个集合合并
- `count`：记录当前还剩多少个集合

### 为什么不用每次扫描所有点？

一个直接做法是维护很多集合，然后每次查询时扫描集合里有没有两个点。

这个方法的问题是：当点很多、合并很多、查询很多时，扫描会反复做重复工作。

并查集换了一个表示法：

> 不直接保存“集合里有哪些元素”，而是让每个点指向一个父节点，最终指向同一个代表的点属于同一个集合。

---

## 目标读者

- 想把并查集基础模板背熟的人
- 做图连通性、连通分量、岛屿数量、省份数量时卡在 `find/union/count` 的人
- 已经见过 LeetCode 547，但想单独补一篇并查集模板的人

## C — Concepts（一步一步长出模板）

### Step 1：先让每个点自成集合

当前压力是：一开始有 `n` 个点，每个点都是自己的集合。

所以最小状态就是：

```python
n = 5
parent = list(range(n))
```

此时：

```text
parent = [0, 1, 2, 3, 4]
```

含义是：

```text
parent[x] == x
```

表示 `x` 现在是自己所在集合的代表。

这一步之后，我们能表达初始状态：

```text
{0}, {1}, {2}, {3}, {4}
```

但它还缺：

- 如何沿着 `parent` 找到一个点最终属于哪个集合

### Step 2：写出第一个 find

现在问一个具体问题：

> 如果 `1` 已经被挂到 `0` 下面，怎么知道 `1` 属于哪个集合？

比如：

```text
parent = [0, 0, 2, 3, 4]
```

这里 `parent[1] = 0`，说明 `1` 指向 `0`。
而 `parent[0] = 0`，说明 `0` 是代表。

所以 `find(1)` 应该返回 `0`。

最小版本：

```python
def find(x: int) -> int:
    while parent[x] != x:
        x = parent[x]
    return x
```

`find` 的循环 invariant 是：

> 每次循环开始时，`x` 仍然在原来那个集合里；沿着 `parent[x]` 往上走，不会改变集合，只会更接近代表。

如果 `parent[x] == x`，说明已经走到代表，返回它。

检查一下：

```python
parent = [0, 0, 2, 3, 4]

assert find(0) == 0
assert find(1) == 0
assert find(2) == 2
```

现在这个版本能：

- 找到一个点所在集合的代表
- 用 `find(a) == find(b)` 判断两个点是否同集合

它还缺：

- 看到一条连接关系时，怎么把两个集合合并

### Step 3：union 只合并两个代表

现在处理第一个合并：

```text
union(0, 1)
```

当前 baseline 是：

```python
parent = [0, 1, 2, 3, 4]
```

如果直接写：

```python
parent[1] = 0
```

这个例子可以工作，但它只适合 `1` 本身就是代表的情况。

更一般的问题是：

```text
union(a, b)
```

`a` 和 `b` 可能不是代表。
所以合并前必须先找到它们各自集合的代表：

```python
root_a = find(a)
root_b = find(b)
```

如果两个代表相同，说明已经在同一个集合里，不需要合并。
如果两个代表不同，就让一个代表指向另一个代表：

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

这里返回 `True / False` 是为了让调用方知道这次合并是否真的发生。

检查：

```python
parent = list(range(5))

assert union(0, 1) is True
assert find(0) == find(1)

assert union(1, 2) is True
assert find(0) == find(2)

assert find(3) != find(4)
```

现在这个版本能：

- 合并两个点所在的集合
- 判断两个点是否连通
- 避免重复合并同一个集合

它还缺：

- 当前一共有几个集合

### Step 4：count 只在真正合并时减少

一开始有 `n` 个点，每个点都是独立集合。
所以：

```python
count = n
```

每次 `union(a, b)` 真正把两个不同集合合并时，集合数量减少 `1`。

但如果 `a` 和 `b` 本来就在同一个集合里，`count` 不能再减少。

反例很小：

```text
n = 3
union(0, 1)  # count 从 3 变成 2
union(0, 1)  # 这次是重复合并，count 仍然应该是 2
```

所以把 `count` 放进对象里更清楚：

```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        self.parent[root_b] = root_a
        self.count -= 1
        return True
```

检查重复合并：

```python
uf = UnionFind(3)

assert uf.count == 3
assert uf.union(0, 1) is True
assert uf.count == 2
assert uf.union(0, 1) is False
assert uf.count == 2
```

现在这个版本能：

- 判断两个点是否属于同一个集合
- 合并两个不同集合
- 维护当前集合数量

它还缺：

- 当 parent 链很长时，`find` 会反复走同一条路径

### Step 5：路径压缩让 find 越查越短

当前 `find` 是正确的，但可能慢。

看一条极端链：

```text
0 -> 1 -> 2 -> 3
```

如果每次 `find(0)` 都从 `0` 走到 `3`，后续查询会重复走同一条链。

路径压缩的想法是：

> 找到代表以后，顺手把沿途节点直接挂到代表下面。

递归写法最短：

```python
def find(self, x: int) -> int:
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]
```

关键行是：

```python
self.parent[x] = self.find(self.parent[x])
```

它做的事情不是改变集合关系。
它只是把 `x` 的父节点改成这个集合的最终代表。

检查一条链：

```python
uf = UnionFind(4)
uf.parent = [1, 2, 3, 3]

assert uf.find(0) == 3
assert uf.parent[0] == 3
assert uf.parent[1] == 3
assert uf.parent[2] == 3
```

路径压缩之后，下一次 `find(0)` 就可以更快到达代表。

---

## Runnable Example（Python）

下面是基础并查集模板。
这版包含：

- `parent`
- `find`
- `union`
- `count`
- `connected`
- 路径压缩

```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        self.parent[root_b] = root_a
        self.count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


if __name__ == "__main__":
    uf = UnionFind(5)

    assert uf.count == 5
    assert uf.union(0, 1) is True
    assert uf.union(1, 2) is True

    assert uf.connected(0, 2) is True
    assert uf.connected(3, 4) is False
    assert uf.count == 3

    assert uf.union(0, 2) is False
    assert uf.count == 3
```

如果你想用函数模板，而不是类模板，可以记成这样：

```python
n = 5
parent = list(range(n))
count = n


def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(a: int, b: int) -> bool:
    global count

    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    count -= 1
    return True


def connected(a: int, b: int) -> bool:
    return find(a) == find(b)
```

---

## Explanation（为什么这样设计）

### parent 存的不是“直接连边”

`parent[x]` 不是说 `x` 和 `parent[x]` 在原图里一定有一条边。

它只表示：

> 为了管理集合，我们让 `x` 指向一个更靠近集合代表的节点。

所以并查集维护的是集合关系，不是原图结构。

这也是为什么并查集适合回答：

```text
a 和 b 是否连通？
当前有几个连通分量？
```

但不适合回答：

```text
a 到 b 的具体路径是什么？
最短路径是多少？
```

### 为什么 union 要先 find？

因为 `a` 和 `b` 可能不是集合代表。

如果直接写：

```python
parent[b] = a
```

可能只是把一个普通节点挂到另一个普通节点下面，甚至破坏已有结构。

稳定写法永远是：

```python
root_a = find(a)
root_b = find(b)
```

然后只连接两个代表。

### count 的 invariant

`count` 的 invariant 是：

> `count` 等于当前集合代表的数量。

初始化时，每个点都是代表，所以 `count = n`。

当且仅当 `root_a != root_b` 时，两个代表合并成一个代表，所以 `count -= 1`。

如果 `root_a == root_b`，代表数量没有变化，`count` 不能变。

---

## R — Reflection（复杂度、取舍和常见坑）

### Complexity

设有 `n` 个点，执行 `m` 次 `find/union` 操作。

带路径压缩的并查集，单次操作均摊接近 `O(1)`。
更正式地说，如果再配合按秩合并或按大小合并，复杂度是：

```text
O(alpha(n))
```

这里 `alpha(n)` 是反阿克曼函数，在实际数据规模下可以近似看成常数。

这篇基础模板只保留路径压缩，不把 rank/size 放进主线。
原因是：

- 基础题里这版通常已经足够
- 模板更短，更容易背写
- rank/size 是优化树高的增强项，不是理解 `find/union/count` 的第一步

### 常见错误

- `union(a, b)` 里直接改 `parent[b] = a`，没有先找代表
- 两个点已经同集合时，仍然让 `count -= 1`
- `find(x)` 只返回 `parent[x]`，没有继续找到最终代表
- 把并查集当成可以恢复具体路径的数据结构
- 一开始就加入 rank/size，反而把基础模板写乱

### 什么时候用并查集？

适合：

- 不断给出连接关系
- 需要判断两个点是否连通
- 需要统计连通分量数量
- 不关心具体路径，只关心集合归属

不适合：

- 需要最短路径
- 需要输出从 `a` 到 `b` 的路径
- 图中有删除边并且要求实时准确连通性
- 有方向图强连通分量问题

---

## S — Summary

- 并查集解决的是“集合归属”和“集合合并”问题。
- `parent[x] == x` 表示 `x` 是一个集合代表。
- `find(x)` 的目标是沿着 `parent` 找到最终代表。
- `union(a, b)` 必须先找两个代表，再合并代表。
- `count` 只在两个不同集合真正合并时减少。
- 路径压缩不改变答案，只让后续 `find` 更快。

### 进一步练习

- LeetCode 547：省份数量
- 岛屿数量类问题：把二维坐标映射成一维编号
- 冗余连接：用并查集判断一条边是否连接了同一集合
- Kruskal 最小生成树：按边权排序后用并查集判断是否成环
