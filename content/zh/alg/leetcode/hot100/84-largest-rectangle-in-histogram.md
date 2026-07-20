---
title: "LeetCode 84：柱状图中最大的矩形，连续区间的高度由谁决定"
date: 2026-07-20T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "栈", "单调栈", "LeetCode 84"]
description: "从连续区间的最矮柱限制出发，先构造枚举区间 baseline，再推导按柱高结算最大宽度的线性单调栈解法。"
keywords: ["LeetCode 84", "Largest Rectangle in Histogram", "柱状图中最大的矩形", "单调栈", "Hot100", "Python"]
---

## 题目要求

给定一个非负整数数组 `heights`。每个 `heights[i]` 表示宽度为 `1` 的柱子高度，所有柱子紧挨排列。

题目要求返回柱状图中能够形成的最大矩形面积。

一个合法矩形必须覆盖一段连续柱子。它的宽度是这段区间包含的柱子数量，高度不能超过区间中的最矮柱。

LeetCode 提供的方法接口是：

```text
largestRectangleArea(heights: List[int]) -> int
```

### 示例 1

```text
输入：heights = [2,1,5,6,2,3]
输出：10
```

下标 `2` 和 `3` 的两根柱子高度分别是 `5` 和 `6`，可以形成高度 `5`、宽度 `2`、面积 `10` 的矩形。

### 示例 2

```text
输入：heights = [2,4]
输出：4
```

可以选择高度 `4`、宽度 `1`，也可以选择高度 `2`、宽度 `2`，最大面积都是 `4`。

### 约束

- `1 <= heights.length <= 10^5`
- `0 <= heights[i] <= 10^4`

## Step 1：矩形面积不是柱高之和

先看一个小柱状图：

```text
heights = [2,1,2]
```

如果矩形覆盖全部三根柱子，它的宽度是 `3`，但高度最多只能是 `1`：

```text
height = 1
width = 3
area = 1 * 3 = 3
```

不能把三根柱高相加得到 `5`。柱状图矩形覆盖的是一块完整矩形区域，中间高度为 `1` 的柱子会限制整个区间的矩形高度。

当前 baseline 是：

```text
选择一段连续区间，用区间中的最矮柱作为矩形高度。
```

对任意连续区间 `[left, right]`：

```text
width = right - left + 1
height = 区间内最矮柱高度
area = height * width
```

这个 baseline 的 break 是：

> 我们已经知道如何计算一个给定区间的面积，但还没有一个可以执行的过程来比较所有连续区间并找到最大值。

现在这一版能做到：

- 区分矩形面积与柱高之和。
- 明确矩形必须覆盖连续柱子。
- 用区间最矮柱确定矩形高度。
- 正确解释 `[2,1,2]` 的面积 `3` 和标准示例中的面积 `10`。

它还缺：

- 一个可以枚举并比较所有合法矩形的正确算法。

## Step 2：先枚举连续区间，写出正确 baseline

当前 baseline 已经知道一个区间的面积计算方式：

```text
区间最矮高度 * 区间宽度
```

现在需要把“比较所有连续区间”写成可运行代码。

固定左边界 `left` 后，逐步向右扩展 `right`。每扩展一根柱子，只需要更新当前区间的最矮高度：

```python
min_height = min(min_height, heights[right])
```

先写一个正确版本：

```python
from typing import List


def largest_rectangle_quadratic(heights: List[int]) -> int:
    n = len(heights)
    best = 0

    for left in range(n):
        min_height = heights[left]

        for right in range(left, n):
            min_height = min(min_height, heights[right])
            width = right - left + 1
            best = max(best, min_height * width)

    return best
```

用 `[2,1,2]` 中 `left = 0` 的扩展过程检查：

| `right` | 区间 | `min_height` | 宽度 | 面积 | 当前最大值 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `[2]` | 2 | 1 | 2 | 2 |
| 1 | `[2,1]` | 1 | 2 | 2 | 2 |
| 2 | `[2,1,2]` | 1 | 3 | 3 | 3 |

运行检查：

```python
assert largest_rectangle_quadratic([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle_quadratic([2, 4]) == 4
assert largest_rectangle_quadratic([2, 1, 2]) == 3
assert largest_rectangle_quadratic([1]) == 1
assert largest_rectangle_quadratic([0]) == 0
assert largest_rectangle_quadratic([1, 2, 3, 4]) == 6
```

现在这一版能做到：

- 枚举每个连续区间。
- 在扩展右边界时增量维护区间最矮高度。
- 正确计算并比较所有合法矩形面积。
- 处理单柱和零高度边界。

它还缺：

- 大量重叠区间会反复计算相同柱子之间的高低关系。

左边界有 O(n) 种选择，每个左边界最多扩展 O(n) 个右边界，因此时间复杂度是 O(n²)。除局部变量外，额外空间复杂度是 O(1)。

## Step 3：改为让每根柱子充当限制高度

当前 baseline 的观察顺序是：

```text
先选择区间
-> 再找区间中的最矮柱
-> 用最矮高度计算面积
```

它的 break 是：

> 同一根柱子可能是许多重叠区间的限制高度，但 baseline 会在不同区间中反复发现这件事。

可以把问题反过来：固定下标 `i`，假设 `heights[i]` 是矩形的限制高度，然后问它最多能向左右延伸多远。

它可以一直延伸，直到遇到第一根严格更矮的柱子：

- `left`：`i` 左侧第一个满足 `heights[left] < heights[i]` 的下标。
- `right`：`i` 右侧第一个满足 `heights[right] < heights[i]` 的下标。

真正能覆盖的区间是：

```text
left + 1 到 right - 1
```

所以：

```text
width = right - left - 1
area = heights[i] * width
```

现在让这个公式实际参与标准示例。对：

```text
heights = [2,1,5,6,2,3]
```

固定下标 `i = 2`、高度 `5`：

- 左侧第一个严格更矮值是下标 `1` 的高度 `1`。
- 右侧第一个严格更矮值是下标 `4` 的高度 `2`。
- 中间可以覆盖下标 `2..3`。

计算：

```text
left = 1
right = 4
width = 4 - 1 - 1 = 2
area = 5 * 2 = 10
```

如果某一侧没有更矮柱子，就使用柱状图外的虚拟边界：

```text
左侧没有更矮柱：left = -1
右侧没有更矮柱：right = n
```

例如 `[2,4]` 中高度 `2` 的柱子，左右边界分别是 `-1` 和 `2`：

```text
width = 2 - (-1) - 1 = 2
area = 2 * 2 = 4
```

这里使用“严格更矮”边界很重要。相等高度不会阻止矩形继续延伸，因此可以包含在同一个候选区间中。

现在这一版能做到：

- 把区间枚举改写成“每根柱子作为限制高度”的候选模型。
- 用左右第一个严格更矮位置确定最大宽度。
- 正确处理延伸到柱状图边缘的虚拟边界。
- 在标准示例中实际得到最大面积 `10`。

它还缺：

- 为每根柱子单独向左右寻找边界仍然可能需要 O(n) 时间。
- 一个能够共享边界查找工作的状态结构。

## Step 4：当前更矮柱出现时，结算栈顶矩形

当前 baseline 已经知道每根柱子需要左右边界，但单独向两侧扫描仍然会重复工作。

关键压力来自扫描过程本身：

> 当当前位置 `i` 的高度低于此前某根柱子时，`i` 就是那根较高柱子右侧第一个严格更矮的位置。此时它的右边界已经确定，可以立即结算面积。

现在才加入一个索引栈 `stack`。栈中下标对应的高度从栈底到栈顶保持非递减：

```text
heights[stack[0]] <= heights[stack[1]] <= ...
```

遇到当前高度 `current_height` 时，持续弹出比它更高的栈顶柱子：

```python
while stack and heights[stack[-1]] > current_height:
    height = heights[stack.pop()]
    left = stack[-1] if stack else -1
    width = i - left - 1
    best = max(best, height * width)
```

弹出以后：

- 当前下标 `i` 是被弹柱子的第一个严格更矮右边界。
- 新栈顶是当前可用区间左侧不能跨过的位置。
- 如果栈为空，左侧可以一直延伸到柱状图开头，所以使用 `-1`。

先把这个机制接到真实柱子的扫描上：

```python
best = 0
stack = []

for i, current_height in enumerate(heights):
    while stack and heights[stack[-1]] > current_height:
        height = heights[stack.pop()]
        left = stack[-1] if stack else -1
        width = i - left - 1
        best = max(best, height * width)

    stack.append(i)
```

用 `[2,1,5,6,2]` 检查：

| `i` | 当前高度 | 操作 | `left` | `width` | 面积 | 操作后的栈 |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 0 | 2 | 压入 0 | - | - | - | `[0]` |
| 1 | 1 | 弹出高度 2，再压入 1 | -1 | 1 | 2 | `[1]` |
| 2 | 5 | 压入 2 | - | - | - | `[1,2]` |
| 3 | 6 | 压入 3 | - | - | - | `[1,2,3]` |
| 4 | 2 | 弹出高度 6 | 2 | 1 | 6 | `[1,2]` |
| 4 | 2 | 弹出高度 5 | 1 | 2 | 10 | `[1]` |
| 4 | 2 | 压入 4 | - | - | - | `[1,4]` |

高度 `5` 在下标 `4` 遇到第一个严格更矮值，左侧不能越过下标 `1` 的高度 `1`，因此宽度是：

```text
4 - 1 - 1 = 2
```

得到面积 `5 * 2 = 10`。

### 相等高度为什么可以留在栈中

弹栈条件使用严格 `>`，所以相等高度会同时留在栈中。此时新栈顶可能与被弹柱子等高，不一定是“严格更矮”的左边界。

这不会漏掉最大面积：较晚的等高柱可能先以较窄宽度结算，而更早的等高柱仍留在栈中，之后会覆盖包含这些等高柱的更宽区间。至少有一个相同高度的代表会得到完整可用宽度。

因此这里准确的 invariant 是“栈中高度非递减”，而不是严格递增。

现在这一版能做到：

- 在第一个严格更矮右边界出现时结算较高柱子的矩形。
- 通过弹栈后的栈顶计算可用左边界和宽度。
- 在一次扫描中共享不同柱子的边界信息。
- 正确保留相等高度并覆盖它们的最大候选宽度。

它还缺：

- 如果输入以递增高度结束，栈中柱子不会遇到更矮的真实右边界，扫描结束后仍未结算。
- 完整 LeetCode 封装、最终 invariant 和复杂度证明。

## Step 5：用尾部 0 结算最后一批柱子

看严格递增输入：

```text
heights = [1,2,3,4]
```

当前 baseline 只有在遇到更矮柱子时才弹栈。这个输入扫描结束时，四个下标仍然全部留在栈中，面积还没有结算。

break 很具体：

> 真实输入结束后，剩余柱子缺少一个更矮的右边界 `n`。

增加一个虚拟的尾部高度 `0`：

```python
bars = heights + [0]
```

这里创建新数组，不修改调用方传入的 `heights`。虚拟 `0` 不属于答案中的真实矩形，它只负责在最后一轮成为所有正高度柱子的更矮右边界，触发剩余弹栈。

完整的 LeetCode 实现是：

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        bars = heights + [0]
        best = 0
        stack = []

        for i, current_height in enumerate(bars):
            while stack and bars[stack[-1]] > current_height:
                height = bars[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                best = max(best, height * width)

            stack.append(i)

        return best
```

循环 invariant 是：

> 每轮开始前，`stack` 按下标递增保存尚未遇到严格更矮右边界的柱子，并且对应高度从栈底到栈顶非递减。

当当前高度更低时：

- 当前下标 `i` 是被弹柱子的第一个严格更矮右边界。
- 弹出后的新栈顶给出当前可用的左侧阻挡位置。
- `i - left - 1` 是被弹高度能够覆盖的宽度。

弹栈结束后，栈顶高度小于等于当前高度，压入 `i` 后仍保持非递减 invariant。

最后的虚拟 `0` 会结算所有剩余正高度柱子。即使真实输入包含高度 `0`，严格 `>` 条件也能保持正确结果。

### 运行检查

```python
solution = Solution()

assert solution.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
assert solution.largestRectangleArea([2, 4]) == 4
assert solution.largestRectangleArea([2, 1, 2]) == 3
assert solution.largestRectangleArea([1]) == 1
assert solution.largestRectangleArea([0]) == 0
assert solution.largestRectangleArea([1, 2, 3, 4]) == 6
assert solution.largestRectangleArea([4, 3, 2, 1]) == 6
assert solution.largestRectangleArea([2, 2, 2]) == 6
```

### 复杂度

虽然代码中存在嵌套 `while`，但每个下标只会：

- 压栈一次。
- 至多弹栈一次。

因此所有弹栈操作加起来是 O(n)：

- 时间复杂度：O(n)。
- 额外空间复杂度：O(n)，用于 `bars` 副本和索引栈。

## 常见错误

### 1. 忘记结算扫描结束后仍在栈中的柱子

严格递增输入不会触发真实弹栈。可以显式写第二个清理循环，也可以像本文一样使用尾部 `0` 统一结算。

### 2. 直接执行 `heights.append(0)`

这样会修改调用方传入的数组。使用：

```python
bars = heights + [0]
```

可以保持输入不变。

### 3. 宽度少减或多减一个位置

`left` 和当前 `i` 都是不能包含的阻挡位置，所以宽度是：

```text
i - left - 1
```

### 4. 栈中只保存高度

计算宽度需要左右下标，因此栈必须保存索引。

### 5. 随意把 `>` 改成 `>=`

使用 `>=` 也可以构造正确方案，但相等高度的保留规则和 invariant 会改变。本文使用严格 `>`，栈中高度保持非递减。

### 6. 看到嵌套循环就判断为 O(n²)

复杂度应按每个下标的总入栈、出栈次数计算。一个下标不可能被弹出两次。

## 小结

这道题的推导路线是：

```text
固定连续区间的最矮高度和宽度
-> 枚举所有区间，维护 min_height
-> 改为让每根柱子充当限制高度
-> 用左右第一个严格更矮位置确定最大宽度
-> 当前更矮柱出现时弹栈结算面积
-> 用尾部 0 结算扫描结束后的剩余柱子
-> 每个下标最多入栈、出栈一次
```

84 比 739 和 503 更进一步：弹栈时不只是写入一个“下一个更大值”，而是同时利用弹出后的新栈顶和当前下标确定左右边界与矩形宽度。
