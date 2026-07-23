---
title: "LeetCode 42：一张高度图能接住多少雨水？"
date: 2026-01-24T10:27:35+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "双指针", "前后最大值", "LeetCode 42"]
description: "从单个位置的水量开始，先构造 O(n²) 正确解，再经过边界数组推导 O(n) 时间、O(1) 额外空间的双指针解法。"
keywords: ["LeetCode 42", "Trapping Rain Water", "接雨水", "双指针", "前后最大值", "Python"]
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

## Step 2：把每个位置的水量加起来

当前版本已经可以回答：

```text
下标 i 上方有多少水？
```

但题目要求的是整张高度图的总水量。只调用一次 `trapped_at` 会漏掉其他位置。

当前 baseline 是上一节的局部函数：

```python
def trapped_at(height: List[int], i: int) -> int:
    left_highest = max(height[: i + 1])
    right_highest = max(height[i:])
    water_level = min(left_highest, right_highest)
    return water_level - height[i]
```

这个 baseline 的局部答案是正确的，缺少的只是一个完成规则：每个下标都处理一次，并把局部水量累加到 `total`。

在上一版后面增加：

```python
def trap_by_scanning(height: List[int]) -> int:
    total = 0

    for i in range(len(height)):
        total += trapped_at(height, i)

    return total
```

在 `[3,0,2]` 中，三个位置的贡献分别是：

| `i` | `trapped_at(height, i)` | 累计 `total` |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 2 | 2 |
| 2 | 0 | 2 |

现在可以检查完整输入：

```python
assert trap_by_scanning([3, 0, 2]) == 2
assert trap_by_scanning([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_by_scanning([4, 2, 0, 3, 2, 5]) == 9
assert trap_by_scanning([1]) == 0
```

现在这个版本可以：

- 正确计算整张高度图的雨水总量
- 让每个位置恰好贡献一次局部水量
- 处理单根柱子和没有低洼位置的输入

它还不能：

- 避免为不同下标重复寻找左右最高柱子
- 在最大规模输入上保持高效

`trapped_at` 对一个位置要查看 O(n) 个元素，外层又处理 n 个位置，因此最坏时间复杂度是 O(n²)。Python 切片还会产生 O(n) 的临时空间；即使改用显式循环，重复扫描造成的 O(n²) 时间也不会消失。

## Step 3：保存已经找过的边界

考虑：

```text
height = [4,2,0,3,2,5]
```

计算下标 `1` 的左侧最高值时，我们查看 `[4,2]`。计算下标 `2` 时又查看 `[4,2,0]`，其中 `[4,2]` 的工作完全重复。

右侧也有相同问题。输入最多有 `2 * 10^4` 个位置，O(n²) 的 baseline 会反复回答相同的前缀和后缀问题。

当前 baseline 是：

```python
for i in range(len(height)):
    total += trapped_at(height, i)
```

它的问题不是局部公式错误，而是每次调用都重新寻找边界。

把这一部分替换为两个可复用的数组：

- `left_highest[i]`：`0..i` 中的最高柱子
- `right_highest[i]`：`i..n-1` 中的最高柱子

每个新位置只依赖相邻位置已经保存的结果：

```text
left_highest[i] = max(left_highest[i - 1], height[i])
right_highest[i] = max(right_highest[i + 1], height[i])
```

这两个名字在这里第一次直接参与更新和最终水量计算：

```python
def trap_with_boundaries(height: List[int]) -> int:
    n = len(height)

    left_highest = [0] * n
    left_highest[0] = height[0]
    for i in range(1, n):
        left_highest[i] = max(left_highest[i - 1], height[i])

    right_highest = [0] * n
    right_highest[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_highest[i] = max(right_highest[i + 1], height[i])

    total = 0
    for i in range(n):
        water_level = min(left_highest[i], right_highest[i])
        total += water_level - height[i]

    return total
```

两个构造循环分别保持：

> 写入 `left_highest[i]` 后，它等于 `height[0..i]` 的最大值。

> 写入 `right_highest[i]` 后，它等于 `height[i..n-1]` 的最大值。

因此最后一个循环使用的仍然是 Step 1 已经验证过的两个真实边界，只是不再重复寻找它们。

检查中间版本：

```python
assert trap_with_boundaries([3, 0, 2]) == 2
assert trap_with_boundaries([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_with_boundaries([4, 2, 0, 3, 2, 5]) == 9
assert trap_with_boundaries([5, 4, 3, 2, 1]) == 0
assert trap_with_boundaries([2, 2, 2]) == 0
```

现在这个版本可以：

- 在 O(n) 时间内计算总水量
- 复用所有前缀和后缀的最高边界
- 保持每个位置的局部水量公式不变

它还不能：

- 避免保存两个长度为 n 的边界数组
- 把额外空间从 O(n) 降到 O(1)

三个线性循环的总时间是 O(n)，两个边界数组使用 O(n) 额外空间。

## Step 4：只结算边界已经确定的一侧

边界数组版本已经把时间降到 O(n)，但它为每个下标保存了两份信息：

```text
left_highest[0..n-1]
right_highest[0..n-1]
```

最后求和时，一个位置处理完就不会再使用。真正的压力是：

> 能否只保留当前需要的两个边界，而不是保存所有位置的边界？

当前 baseline 依赖：

```text
water[i] = min(left_highest[i], right_highest[i]) - height[i]
```

要删掉数组，我们必须先知道当前哪个位置的较矮边界已经确定。

现在才引入四个状态：

- `left`、`right`：尚未结算区间的两端
- `left_highest`：从开头到 `left` 见过的最高柱子
- `right_highest`：从 `right` 到结尾见过的最高柱子

每轮先更新两个最高值。如果：

```text
left_highest <= right_highest
```

那么 `left` 右侧的真实最高值至少是 `right_highest`，也就至少是 `left_highest`。因此 `left` 的较矮边界已经确定为 `left_highest`，可以立即计算：

```text
left_highest - height[left]
```

反过来，如果 `left_highest > right_highest`，`right` 的较矮边界已经确定为 `right_highest`，可以计算：

```text
right_highest - height[right]
```

每轮只结算一侧，并移动对应指针。把边界数组版本替换为最终 LeetCode 实现：

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        left_highest = 0
        right_highest = 0
        total = 0

        while left <= right:
            left_highest = max(left_highest, height[left])
            right_highest = max(right_highest, height[right])

            if left_highest <= right_highest:
                total += left_highest - height[left]
                left += 1
            else:
                total += right_highest - height[right]
                right -= 1

        return total
```

用 `[5,0,1,0,2]` 检查容易忽略的右侧分支：

| `left` | `right` | `left_highest` | `right_highest` | 本轮结算 | 新增水量 | `total` |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 4 | 5 | 2 | 右侧下标 4 | 0 | 0 |
| 0 | 3 | 5 | 2 | 右侧下标 3 | 2 | 2 |
| 0 | 2 | 5 | 2 | 右侧下标 2 | 1 | 3 |
| 0 | 1 | 5 | 2 | 右侧下标 1 | 2 | 5 |
| 0 | 0 | 5 | 5 | 左侧下标 0 | 0 | 5 |

最终检查：

```python
solution = Solution()

assert solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert solution.trap([4, 2, 0, 3, 2, 5]) == 9
assert solution.trap([5, 0, 1, 0, 2]) == 5
assert solution.trap([3, 0, 2]) == 2
assert solution.trap([5, 4, 3, 2, 1]) == 0
assert solution.trap([2, 2, 2]) == 0
assert solution.trap([1]) == 0
```

循环 invariant 是：

> 每轮开始时，`left..right` 之外的下标都已经恰好结算一次。

更新两个最高值后：

```text
left_highest = max(height[0..left])
right_highest = max(height[right..n-1])
```

- 如果 `left_highest <= right_highest`，右侧真实边界不可能低于 `left_highest`，所以结算 `left` 正确。
- 否则左侧真实边界不可能低于 `right_highest`，所以结算 `right` 正确。
- 每轮恰好移动一个指针，invariant 被带到下一轮。
- 当 `left > right` 时，每个下标都已结算一次，`total` 就是总水量。

现在这个版本已经满足题目要求：

- 时间复杂度：O(n)。每轮移动一个指针，每个下标只结算一次。
- 额外空间复杂度：O(1)。只保存指针、边界最高值和总量。
- 不修改输入数组。

## 常见错误

### 1. 先计算水量，再更新当前边界

当前柱子本身也属于这一侧的最高边界候选。必须先执行：

```python
left_highest = max(left_highest, height[left])
right_highest = max(right_highest, height[right])
```

然后才能计算非负的局部水量。

### 2. 直接把比较条件换成当前柱子高度

比较 `height[left]` 与 `height[right]` 也可以构造另一种正确写法，但它使用不同的移动规则和证明。本文的结算依据是已经维护好的 `left_highest` 与 `right_highest`，不要只替换条件而保留原证明。

### 3. 一轮同时移动两个指针

每轮只有较矮边界一侧已经可以安全结算。两个指针同时移动会跳过另一侧尚未确定的位置。

### 4. 忘记边界范围包含当前位置

如果最高值不包含当前柱子，`water_level - height[i]` 可能为负。三个版本都让左右最高值包含当前位置，因此不需要额外套 `max(0, ...)`。

### 5. 把 `left <= right` 随意改成 `left < right`

其他循环边界也可能得到正确实现，但本文的 invariant 是“每个下标恰好结算一次”。使用 `left <= right` 会显式处理最后相遇的位置；更换循环条件时必须同步修改终止证明。

## 总结

这道题的推导路线是：

```text
先计算一个位置的水量
-> 对所有位置重复扫描，得到 O(n²) 正确解
-> 保存每个前缀和后缀的最高值，得到 O(n) 时间、O(n) 空间解
-> 证明当前较矮边界的一侧可以立即结算
-> 每轮移动一个指针，得到 O(n) 时间、O(1) 额外空间解
```

双指针版本最关键的不是记住 `if left_highest <= right_highest`，而是解释为什么这个条件足以确定一侧的真实较矮边界。只要这一步能够独立推导，四个状态和移动规则就不再是需要死记的模板。
