---
title: "LeetCode 42：一个位置上方为什么能留住水？"
date: 2026-01-24T10:27:35+08:00
draft: true
categories: ["LeetCode"]
tags: ["Hot100", "数组", "LeetCode 42"]
description: "从一个位置能接多少水开始，逐步构造 LeetCode 42 接雨水的可运行解法。"
keywords: ["LeetCode 42", "Trapping Rain Water", "接雨水", "数组", "Python"]
---

## 题目要求

给定 `n` 个非负整数 `height`。每个整数表示一根宽度为 `1` 的柱子高度，所有柱子从左到右相邻排列。

下雨后，有些较矮的柱子上方会被两侧较高的柱子围住水。返回整张高度图最终能接住的雨水总量。

LeetCode 要求实现：

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ...
```

### 示例 1

```text
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
```

### 示例 2

```text
输入：height = [4,2,0,3,2,5]
输出：9
```

### 约束

```text
n == len(height)
1 <= n <= 2 * 10^4
0 <= height[i] <= 10^5
```

## Step 1：先回答一个位置能接多少水

先不计算整张高度图，只看一个位置：

```text
height = [3,0,2]
            ^
           i = 1
```

下标 `1` 的柱子高度是 `0`。它左边有高度为 `3` 的柱子，右边有高度为 `2` 的柱子。

如果只看左边，似乎可以把水加到高度 `3`。但右边的墙只有高度 `2`，超过 `2` 的水会从右边流走。因此这个位置的水面最高只能到：

```text
min(左侧最高柱子, 右侧最高柱子)
= min(3, 2)
= 2
```

这个位置上方的水量是：

```text
水面高度 - 当前柱子高度
= 2 - 0
= 2
```

当前 baseline 是：

> 找到当前位置两边的墙，再判断水能留到多高。

但“看两边的墙”还不够精确。如果一侧有多根柱子，我们需要知道这一侧真正能提供的最高边界；如果直接采用较高一侧，水又会从较矮一侧流走。

因此，对一个下标 `i`，只增加一个可执行规则：

1. 在 `0..i` 中找到 `left_highest`。
2. 在 `i..n-1` 中找到 `right_highest`。
3. 较矮的边界决定 `water_level`。
4. 用 `water_level - height[i]` 得到当前位置的水量。

左右范围都包含 `i`。这样两个最高值都不会低于 `height[i]`，计算结果不会变成负数。

把这个局部规则写成第一个可运行版本：

```python
from typing import List


def trapped_at(height: List[int], i: int) -> int:
    left_highest = max(height[: i + 1])
    right_highest = max(height[i:])
    water_level = min(left_highest, right_highest)
    return water_level - height[i]
```

用刚才的低洼位置检查它：

```python
assert trapped_at([3, 0, 2], 1) == 2
```

两端没有完整的左右包围，因此都接不到水：

```python
assert trapped_at([3, 0, 2], 0) == 0
assert trapped_at([3, 0, 2], 2) == 0
```

再检查一个两侧等高的低洼位置：

```python
assert trapped_at([2, 1, 2], 1) == 1
```

现在这个版本可以：

- 计算任意一个下标上方的雨水量
- 解释为什么水面由两侧最高边界中较矮的一侧决定
- 通过包含当前位置的左右范围保证结果非负

它还不能：

- 计算整张高度图的雨水总量
- 避免不同位置反复扫描相同的左右区间
