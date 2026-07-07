---
title: "LeetCode 452：用最少数量的箭引爆气球，从共同交集推出区间贪心"
date: 2026-07-06T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "贪心", "区间", "排序", "LeetCode 452"]
description: "从一支箭能覆盖哪些气球出发，推导 LeetCode 452 用最少数量的箭引爆气球：先写共同交集扫描基线，再压缩成按结束位置排序的 arrow_pos 贪心。"
keywords: ["LeetCode 452", "用最少数量的箭引爆气球", "Minimum Number of Arrows to Burst Balloons", "贪心", "区间贪心", "排序", "Python"]
---

## 题目要求

给你一个气球数组 `points`。

每个气球是一个水平区间：

```text
[start, end]
```

如果一支箭射在横坐标 `x` 上，并且：

```text
start <= x <= end
```

那么这支箭可以射爆这个气球。

一支箭会一直向上飞，所以同一个 `x` 上能覆盖到的所有气球都会被射爆。

题目要求返回：

```text
射爆所有气球需要的最少箭数
```

### 输入输出

- 输入：`points: List[List[int]]`
- 输出：`int`
- 每个气球是一个区间 `[start, end]`
- 箭的位置 `x` 可以落在端点上
- 不需要返回每支箭的位置，只需要返回最少箭数

### 示例

```text
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
```

一种射法是：

```text
x = 6  射爆 [2,8] 和 [1,6]
x = 11 射爆 [10,16] 和 [7,12]
```

再看两个边界例子：

```text
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
```

这些气球互不相交，每个气球都需要一支箭。

```text
输入：points = [[1,2],[2,3],[3,4],[4,5]]
输出：2
```

因为箭可以射在端点上，所以：

```text
x = 2 可以射爆 [1,2] 和 [2,3]
x = 4 可以射爆 [3,4] 和 [4,5]
```

### 约束

- `1 <= points.length <= 10^5`
- `points[i].length == 2`
- `-2^31 <= start < end <= 2^31 - 1`

## Step 1：一支箭到底能覆盖哪些气球？

先看一个很小的问题：

```text
[1,2] 和 [2,3]
```

这两个气球能不能用一支箭射爆？

可以。

因为箭可以射在端点上：

```text
x = 2
```

同时满足：

```text
1 <= 2 <= 2
2 <= 2 <= 3
```

当前 baseline 是：

```text
一个气球需要一支箭。
```

这个 baseline 的 break 是：

> 示例 1 有 4 个气球，但答案是 2。说明一支箭不一定只服务一个气球。

真正要问的是：

```text
哪些气球有共同的 x 位置？
```

只要一组气球的区间存在公共交集，就可以用一支箭射爆这一组。

例如：

```text
[1,6] 和 [2,8]
```

公共交集是：

```text
[2,6]
```

在这个范围内任意选一个 `x`，都能射爆它们。

现在这一版能做到：

- 知道一支箭对应的是一组有公共交集的区间。
- 知道端点相接也可以共享一支箭。
- 能解释为什么箭数可能小于气球数量。

它还缺：

- 一个能把所有气球分成若干组的可运行方法。

## Step 2：先写共同交集扫描 baseline

现在 baseline 是：

```text
一支箭可以覆盖一组有公共交集的气球。
```

break 是：

> 我们还不能对任意 `points` 算出最少箭数。

先写一个正确但状态稍重的版本。

把气球按起点排序：

```python
points.sort(key=lambda p: p[0])
```

扫描时维护当前这一组气球的共同可射区间：

```text
[left, right]
```

当读到一个新区间 `[start, end]`：

- 如果 `start <= right`，说明它和当前组仍有交集。
- 共同可射区间收缩成 `[max(left, start), min(right, end)]`。
- 如果 `start > right`，说明当前组已经无法覆盖它，必须新开一支箭。

代码：

```python
from typing import List


def arrows_by_intersection(points: List[List[int]]) -> int:
    points.sort(key=lambda p: p[0])

    arrows = 1
    left, right = points[0]

    for start, end in points[1:]:
        if start <= right:
            left = max(left, start)
            right = min(right, end)
        else:
            arrows += 1
            left, right = start, end

    return arrows
```

用示例检查：

```text
points = [[10,16],[2,8],[1,6],[7,12]]
```

按起点排序：

```text
[[1,6],[2,8],[7,12],[10,16]]
```

扫描过程：

```text
当前组 [1,6]
读到 [2,8]，仍有交集，公共可射区间变成 [2,6]
读到 [7,12]，7 > 6，必须新开一支箭，当前组变成 [7,12]
读到 [10,16]，仍有交集，公共可射区间变成 [10,12]
```

一共需要 `2` 支箭。

现在这一版能做到：

- 正确统计需要多少个有公共交集的气球组。
- 能解释每一支箭覆盖的是哪个区间组。
- 能跑通题目的核心示例。

它还缺：

- 状态里维护了 `left` 和 `right` 两端，但最终判断新开一箭时真正关键的是右边界。

## Step 3：把共同交集压缩成 arrow_pos

现在 baseline 是：

```text
维护当前组的共同可射区间 [left, right]。
```

break 是：

> 这个 baseline 正确，但状态还可以更小。判断下一个气球能不能被当前箭覆盖，只需要知道当前箭最晚能射在哪里。

在一组气球里，最危险的是结束位置最早的气球。

如果当前组里最早结束的位置是：

```text
right
```

那么当前箭的位置不能超过 `right`。

为了给后续气球留下机会，直接把箭放在这个最早结束位置：

```text
arrow_pos = right
```

这和 435 里的“保留结束更早的区间”是同一类贪心压力：

```text
越早冻结右边界，越不会错过当前必须覆盖的区间。
```

于是可以先按结束位置排序。

每次遇到一个气球 `[start, end]`：

- 如果 `start <= arrow_pos`，当前箭能射爆它。
- 如果 `start > arrow_pos`，当前箭射不到它，必须新开一支箭，并把新箭放在 `end`。

注意这里新开箭的条件是：

```text
start > arrow_pos
```

不是：

```text
start >= arrow_pos
```

因为箭可以射在端点上。

看边界例子：

```text
points = [[1,2],[2,3],[3,4],[4,5]]
```

如果第一支箭放在：

```text
arrow_pos = 2
```

那么 `[2,3]` 的 `start = 2`，仍然能被这支箭射爆。

所以 `[1,2]` 和 `[2,3]` 共享一支箭。

现在这一版能做到：

- 把共同交集的右边界压缩成 `arrow_pos`。
- 知道按结束位置排序的原因。
- 知道什么时候必须新开一支箭。

它还缺：

- 完整的 LeetCode 函数和可运行检查。

## Step 4：用 arrow_pos 统计最少箭数

现在 baseline 是：

```text
按结束位置排序，把当前箭放在当前组最早结束的位置。
```

break 是：

> 还没有把这个规则组装成 LeetCode 要的 `findMinArrowShots`。

完整代码：

```python
from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])

        arrows = 1
        arrow_pos = points[0][1]

        for start, end in points[1:]:
            if start > arrow_pos:
                arrows += 1
                arrow_pos = end

        return arrows
```

用示例走一遍：

```text
points = [[10,16],[2,8],[1,6],[7,12]]
```

按结束位置排序：

```text
[[1,6],[2,8],[7,12],[10,16]]
```

扫描：

```text
第一支箭放在 6
[2,8] 的 start = 2 <= 6，被当前箭覆盖
[7,12] 的 start = 7 > 6，新开第二支箭，放在 12
[10,16] 的 start = 10 <= 12，被第二支箭覆盖
```

最后需要 `2` 支箭。

现在这一版能做到：

- 用最少状态统计箭数。
- 正确处理端点相接。
- 满足 LeetCode 的函数签名。

## 正确性直觉

当前箭要覆盖当前组里的所有气球。

只要当前组里有一个气球最早结束在 `arrow_pos`，这支箭就不能放到 `arrow_pos` 右边。

所以把箭放在 `arrow_pos` 是安全且最宽松的选择：

```text
它不会错过当前最早结束的气球，
同时尽可能给后面还没处理的气球留下覆盖机会。
```

按结束位置排序后，每次选择当前最早结束的气球来决定箭的位置。

如果后面的气球起点仍然不超过 `arrow_pos`，它就被当前箭顺带覆盖。

如果后面的气球起点已经大于 `arrow_pos`，当前箭无论放在哪里都不可能同时覆盖它和当前组，因此必须新开一支箭。

## 复杂度

排序需要：

```text
O(n log n)
```

扫描一次：

```text
O(n)
```

总时间复杂度：

```text
O(n log n)
```

额外只用了 `arrows` 和 `arrow_pos`：

```text
O(1)
```

## 常见错误

### 1. 把 `start == arrow_pos` 当成需要新箭

这题的覆盖条件是：

```text
start <= x <= end
```

所以如果：

```text
start == arrow_pos
```

当前箭仍然可以射爆这个气球。

新开箭条件必须是：

```python
start > arrow_pos
```

### 2. 一开始按起点排序后直接套最终写法

按起点排序适合写共同交集 baseline。

最终贪心写法更适合按结束位置排序，因为 `arrow_pos` 来自当前最早结束的气球。

如果排序依据和状态含义对不上，很容易写出看似能过部分样例、但证明不清楚的代码。

### 3. 忘记题目保证至少一个气球

约束里有：

```text
1 <= points.length
```

所以 LeetCode 版本里可以直接初始化：

```python
arrows = 1
arrow_pos = points[0][1]
```

不需要额外处理空数组。

## 可以直接运行的检查

```python
from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])

        arrows = 1
        arrow_pos = points[0][1]

        for start, end in points[1:]:
            if start > arrow_pos:
                arrows += 1
                arrow_pos = end

        return arrows


def check() -> None:
    s = Solution()
    assert s.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert s.findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert s.findMinArrowShots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2
    assert s.findMinArrowShots([[1, 2]]) == 1


check()
```

## 小结

452 的贪心点是：

```text
每一支箭放在当前组最早结束的位置。
```

扫描 invariant 是：

> 当前箭放在 `arrow_pos`，它能覆盖当前已经归入这一组的所有气球；如果下一个气球的 `start > arrow_pos`，当前箭组已经结束，必须新开一支箭。

和 435 的关系是：

- 435：保留尽量多的不重叠区间。
- 452：用尽量少的点覆盖所有区间。

两题都在训练同一个区间贪心直觉：

```text
按结束位置排序，优先处理最早结束的约束。
```
