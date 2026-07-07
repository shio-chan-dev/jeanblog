---
title: "LeetCode 435：无重叠区间，从删除最少转成保留最多"
date: 2026-07-06T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "贪心", "区间", "排序", "LeetCode 435"]
description: "从删除哪个区间的选择压力出发，推导 LeetCode 435 无重叠区间：先把最少删除转换成最多保留，再逐步推出按结束位置排序的区间贪心。"
keywords: ["LeetCode 435", "无重叠区间", "Non-overlapping Intervals", "贪心", "区间贪心", "排序", "Python"]
---

## 题目要求

给你一个区间数组 `intervals`。

每个区间写成：

```text
[start, end]
```

题目要求删除尽量少的区间，使剩下的区间互不重叠。

最后返回：

```text
最少需要删除多少个区间
```

### 输入输出

- 输入：`intervals: List[List[int]]`
- 输出：`int`
- 每个区间满足 `start < end`
- 如果两个区间只是在端点相接，不算重叠

也就是说：

```text
[1,2] 和 [2,3]
```

可以同时保留。

### 示例

```text
输入：intervals = [[1,2],[2,3],[3,4],[1,3]]
输出：1
```

删除 `[1,3]` 后，剩下：

```text
[[1,2],[2,3],[3,4]]
```

这些区间互不重叠。

再看两个边界例子：

```text
输入：intervals = [[1,2],[1,2],[1,2]]
输出：2
```

三个完全相同的区间最多只能保留一个，所以要删除两个。

```text
输入：intervals = [[1,2],[2,3]]
输出：0
```

这两个区间只在端点 `2` 相接，不算重叠，所以不用删除。

### 约束

- `1 <= intervals.length <= 10^5`
- `intervals[i].length == 2`
- `-5 * 10^4 <= start_i < end_i <= 5 * 10^4`

## Step 1：不要先问删哪个，先问最多能留几个

先看这个例子：

```text
intervals = [[1,2],[2,3],[3,4],[1,3]]
```

题目问的是：

```text
最少删除几个区间？
```

当前 baseline 是：

```text
直接判断应该删除哪个区间。
```

这个 baseline 的 break 是：

> 当多个区间互相冲突时，直接盯着“删哪个”很容易乱。我们还不知道应该根据什么标准删除。

换一个角度看：

```text
如果一共有 n 个区间，最后最多能保留 max_kept 个互不重叠区间，
那么最少删除数量就是 n - max_kept。
```

也就是：

```text
min_removed = n - max_kept
```

在这个例子里，可以保留：

```text
[[1,2],[2,3],[3,4]]
```

这三个区间互不重叠。

原数组一共有 `4` 个区间，最多保留 `3` 个，所以最少删除：

```text
4 - 3 = 1
```

这一步只改变目标，不改变题目答案：

```text
求最少删除数量
```

等价于：

```text
求最多能保留多少个互不重叠区间
```

现在这一版能做到：

- 知道不要一开始就猜“删哪个”。
- 知道可以先算 `max_kept`，最后再用 `n - max_kept` 转回答案。
- 能解释 `[[1,2],[2,3],[3,4],[1,3]]` 为什么答案是 `1`。

它还缺：

- 当两个区间冲突时，应该保留哪一个。

## Step 2：冲突时，保留结束更早的那个

现在 baseline 是：

```text
先求最多能保留多少个互不重叠区间。
```

break 是：

> 目标变成了“最多保留”，但遇到重叠区间时，我们还不知道应该保留哪个。

看一个更小的冲突：

```text
[1,2], [1,3], [2,3]
```

如果先保留 `[1,3]`：

```text
保留：[1,3]
```

那么 `[2,3]` 会和 `[1,3]` 重叠，后面就很难继续保留。

如果先保留 `[1,2]`：

```text
保留：[1,2]
```

因为题目说端点相接不算重叠，所以 `[2,3]` 还能继续保留：

```text
[1,2], [2,3]
```

这说明冲突时，应该优先保留：

```text
结束位置更早的区间
```

原因不是它看起来更短，而是它给后面的区间留下了更大的空间。

换句话说：

```text
如果两个区间都可以作为当前保留区间，
结束更早的那个不会让后续选择变少。
```

所以后面扫描时，应该让区间按结束位置从小到大排列。

现在这一版能做到：

- 知道冲突时优先保留结束更早的区间。
- 知道排序依据来自“给后面留空间”，不是随便套模板。
- 能解释为什么 `[1,2]` 比 `[1,3]` 更适合作为当前保留区间。

它还缺：

- 一个能把这个规则跑完整个数组的扫描过程。

## Step 3：按结束位置扫描，统计最多能保留几个

现在 baseline 是：

```text
冲突时，保留结束更早的区间。
```

break 是：

> 这还是一个局部选择，还没有变成可运行的整体算法。

我们把区间按结束位置排序：

```python
intervals.sort(key=lambda x: x[1])
```

扫描过程中只需要维护两件事：

```text
last_end: 上一个被保留区间的结束位置
kept:     当前已经保留了多少个区间
```

什么时候当前区间可以保留？

如果当前区间是：

```text
[start, end]
```

只要：

```text
start >= last_end
```

它就不会和上一个保留区间重叠。

注意这里是 `>=`，不是 `>`。

因为：

```text
[1,2] 和 [2,3]
```

端点相接，不算重叠。

先写一个只计算 `max_kept` 的版本：

```python
from typing import List


def max_non_overlapping(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[1])

    kept = 0
    last_end = float("-inf")

    for start, end in intervals:
        if start >= last_end:
            kept += 1
            last_end = end

    return kept
```

检查两个关键例子。

第一个：

```text
intervals = [[1,2],[2,3]]
```

排序后还是：

```text
[[1,2],[2,3]]
```

扫描过程：

```text
保留 [1,2]，last_end = 2
[2,3] 的 start = 2，满足 start >= last_end
继续保留 [2,3]
```

所以：

```text
max_kept = 2
```

第二个：

```text
intervals = [[1,2],[1,2],[1,2]]
```

扫描过程：

```text
保留第一个 [1,2]，last_end = 2
第二个 [1,2] 的 start = 1，不满足 start >= 2，跳过
第三个 [1,2] 的 start = 1，不满足 start >= 2，跳过
```

所以：

```text
max_kept = 1
```

现在这一版能做到：

- 按结束位置从小到大扫描。
- 用 `last_end` 判断当前区间能不能保留。
- 计算最多能保留多少个互不重叠区间。

它还缺：

- LeetCode 要的是最少删除数量，不是最多保留数量。

## Step 4：把最多保留转回最少删除

现在 baseline 是：

```text
max_non_overlapping(intervals) 可以算出最多能保留多少个区间。
```

break 是：

> 如果直接返回 `kept`，回答的是转换后的问题，不是原题。

原题要返回：

```text
最少删除多少个区间
```

前面已经得到关系：

```text
min_removed = n - max_kept
```

所以最终只需要把 `kept` 转回删除数量。

完整 LeetCode 写法：

```python
from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])

        kept = 0
        last_end = float("-inf")

        for start, end in intervals:
            if start >= last_end:
                kept += 1
                last_end = end

        return len(intervals) - kept
```

用最开始的例子检查：

```text
intervals = [[1,2],[2,3],[3,4],[1,3]]
```

按结束位置排序后：

```text
[[1,2],[2,3],[1,3],[3,4]]
```

扫描：

```text
保留 [1,2]，last_end = 2
保留 [2,3]，last_end = 3
[1,3] 重叠，跳过
保留 [3,4]，last_end = 4
```

最多保留 `3` 个。

原来一共有 `4` 个，所以最少删除：

```text
4 - 3 = 1
```

这和题目输出一致。

现在这一版能做到：

- 按结束位置排序，优先保留结束更早的区间。
- 用 `start >= last_end` 正确处理端点相接。
- 返回原题要求的最少删除数量。

## 正确性直觉

这题的核心不是“看到重叠就随便删一个”。

真正的贪心选择是：

> 在所有可以作为当前保留区间的选择里，保留结束最早的那个。

结束越早，后面的区间越容易接上。

如果保留了一个结束更晚的区间，它不会让当前保留数量变多，却可能挡住后面的区间。

所以按结束位置排序后，只要当前区间能接在 `last_end` 后面，就立刻保留它。

这个过程得到的是最多保留数量。

最后再用：

```text
最少删除 = 总数 - 最多保留
```

转回原题答案。

## 复杂度

排序需要：

```text
O(n log n)
```

扫描每个区间一次：

```text
O(n)
```

所以总时间复杂度是：

```text
O(n log n)
```

除了排序本身，额外只用了 `kept` 和 `last_end`：

```text
O(1)
```

## 常见错误

### 1. 把端点相接当成重叠

这题明确说：

```text
[1,2] 和 [2,3]
```

不重叠。

所以判断能否保留时，要写：

```python
start >= last_end
```

不能写成：

```python
start > last_end
```

### 2. 返回 `kept`

`kept` 是最多保留数量。

原题要的是最少删除数量，所以最后必须返回：

```python
len(intervals) - kept
```

### 3. 先按起点排序，然后遇到重叠随便删

按起点排序也可以写出别的版本，但本题最稳定的贪心解释是按结束位置排序。

因为“结束更早”直接对应：

```text
给后面的区间留下更多空间
```

这比“遇到重叠后再修正”更容易证明，也更不容易写错边界。

## 可以直接运行的检查

```python
from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])

        kept = 0
        last_end = float("-inf")

        for start, end in intervals:
            if start >= last_end:
                kept += 1
                last_end = end

        return len(intervals) - kept


def check() -> None:
    s = Solution()
    assert s.eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert s.eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert s.eraseOverlapIntervals([[1, 2], [2, 3]]) == 0
    assert s.eraseOverlapIntervals([[1, 100], [11, 22], [1, 11], [2, 12]]) == 2


check()
```

## 小结

435 的贪心点是：

```text
最少删除 = 总数 - 最多保留
```

而最多保留的策略是：

```text
按结束位置排序，能接上就保留。
```

扫描 invariant 是：

> 已保留的区间互不重叠，并且 `last_end` 尽可能早，从而给后面的区间留下尽可能多的空间。

掌握这题后，再看 452 射气球会自然很多：那里不是“保留最多区间”，而是“一支箭覆盖一个有公共交点的区间组”。
