---
title: "LeetCode 739：每日温度，如何找到右侧第一个更高温度"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "栈", "单调栈", "LeetCode 739"]
description: "从向右扫描的 O(n²) baseline 出发，推导维护未解决日期的单调索引栈，在线性时间内解决 LeetCode 739 每日温度。"
keywords: ["LeetCode 739", "Daily Temperatures", "每日温度", "单调栈", "下一个更大元素", "Hot100"]
---

## 题目要求

给定一个整数数组 `temperatures`，其中 `temperatures[i]` 表示第 `i` 天的温度。

返回数组 `answer`，其中：

```text
answer[i] = 从第 i 天开始，需要等待多少天才会遇到更高温度
```

如果之后没有更高温度，`answer[i] = 0`。

这里的“更高”是严格大于。相同温度不能结算等待中的日期。

LeetCode 提供的方法接口是：

```text
dailyTemperatures(temperatures: List[int]) -> List[int]
```

### 示例

```text
输入：temperatures = [73,74,75,71,69,72,76,73]
输出：[1,1,4,2,1,1,0,0]
```

### 约束

- `1 <= temperatures.length <= 10^5`
- `30 <= temperatures[i] <= 100`

## Step 1：答案不是更高温度，而是等待天数

先看一个更小的输入：

```text
temperatures = [73,71,72,76]
```

逐天回答：

- 第 0 天是 `73`，右侧第一个更高温度是第 3 天的 `76`，等待 `3` 天。
- 第 1 天是 `71`，第 2 天的 `72` 更高，等待 `1` 天。
- 第 2 天是 `72`，第 3 天的 `76` 更高，等待 `1` 天。
- 第 3 天右侧没有日期，答案是 `0`。

所以结果是：

```text
[3,1,1,0]
```

当前 baseline 是：

```text
对每一天，观察它右侧什么时候第一次出现更高温度。
```

这个 baseline 必须先固定三个细节：

1. 要找的是右侧第一个更高温度，不是右侧最高温度。
2. 答案是两个下标之差，不是温度差。
3. 相等不算更高，例如 `[70,70]` 的答案是 `[0,0]`。

现在这一版能做到：

- 准确解释每个输出位置的含义。
- 处理“之后没有更高温度”的零值。
- 区分严格更高与相等温度。

它还缺：

- 一个可以为每一天执行查找的正确算法。

## Step 2：先向右扫描，写出正确 baseline

当前 baseline 是“从某一天向右寻找第一个更高温度”。最直接的实现就是把这个动作写成双重循环。

```python
from typing import List


def daily_temperatures_scan(temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    answer = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if temperatures[j] > temperatures[i]:
                answer[i] = j - i
                break

    return answer
```

这个版本中的每一部分都对应题意：

- `answer` 初始为 `0`，自然覆盖“之后没有更高温度”的情况。
- `j` 从 `i + 1` 开始，只检查右侧日期。
- 使用 `>`，保证相等温度不会被当成答案。
- 找到第一个更高温度后立即 `break`。
- `j - i` 是等待天数。

运行检查：

```python
assert daily_temperatures_scan([73, 71, 72, 76]) == [3, 1, 1, 0]
assert daily_temperatures_scan([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures_scan([60, 50, 40]) == [0, 0, 0]
assert daily_temperatures_scan([70, 70]) == [0, 0]
```

现在这一版能做到：

- 正确找到每一天右侧第一个严格更高温度。
- 对不存在答案的位置保留 `0`。
- 直接对应题目的定义，便于验证。

它还缺：

- 最坏情况下会反复扫描相同的后缀。

例如温度严格递减时，每个 `i` 都会一直扫描到数组末尾，时间复杂度达到 O(n²)。当 `n` 可以达到 `10^5` 时，这个代价不能接受。

## Step 3：把还没等到升温的日期留下来

重新观察：

```text
[73,71,72,76]
```

扫描到 `71` 时，我们还不知道它什么时候会遇到更高温度。扫描到 `72` 时，答案立刻出现了：`72 > 71`。

这说明可以换一个方向思考：

> 不让每个旧日期主动向右查找，而是让今天的温度结算此前仍在等待的日期。

为了在结算时计算等待天数，必须保存日期下标，而不只是温度值。增加一个栈 `stack`，保存还没有找到更高温度的下标。

当前日期 `i`、温度 `temperature` 到来时，检查栈顶：

```python
while stack and temperature > temperatures[stack[-1]]:
    previous = stack.pop()
    answer[previous] = i - previous
```

这个栈现在真正参与了判断、弹出和答案更新。完成结算后，再把当前下标加入栈：

```python
stack.append(i)
```

用 `[73,71,72,76]` 完整检查：

| `i` | 当前温度 | 操作 | 更新 | 操作后的栈 |
| ---: | ---: | --- | --- | --- |
| 0 | 73 | 栈空，压入 0 | 无 | `[0]` |
| 1 | 71 | 71 不高于 73，压入 1 | 无 | `[0,1]` |
| 2 | 72 | 弹出 1，再压入 2 | `answer[1] = 2 - 1 = 1` | `[0,2]` |
| 3 | 76 | 依次弹出 2、0，再压入 3 | `answer[2] = 1`，`answer[0] = 3` | `[3]` |

扫描过程中，栈具有三个性质：

- 下标从栈底到栈顶递增，因为日期按顺序压入。
- 栈中的日期都还没有找到更高温度。
- 对应温度从栈底到栈顶非递增。

最后一条不是“严格递减”。相等温度不会触发 `>` 条件，因此可以同时留在栈中。

现在这一版能做到：

- 保存尚未解决的日期，而不是重复扫描它们的右侧区间。
- 用一个较高温度一次结算多个旧日期。
- 直接根据下标差写入等待天数。

它还缺：

- 证明弹栈时的当前日期一定是第一个更高温度。
- 完整的 LeetCode 实现和复杂度分析。

## Step 4：为什么弹栈时一定是第一个更高温度

在扫描到日期 `i` 前，某个下标 `previous` 仍然留在栈中，说明：

```text
previous + 1 到 i - 1 之间，没有温度严格高于 temperatures[previous]
```

否则它早就在那个更早的日期被弹出了。

当今天满足：

```text
temperatures[i] > temperatures[previous]
```

今天就是 `previous` 右侧第一个严格更高温度，因此可以安全写入：

```text
answer[previous] = i - previous
```

完整实现是：

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0] * len(temperatures)
        stack = []

        for i, temperature in enumerate(temperatures):
            while stack and temperature > temperatures[stack[-1]]:
                previous = stack.pop()
                answer[previous] = i - previous

            stack.append(i)

        return answer
```

循环 invariant 是：

> 每轮开始前，`stack` 按日期递增保存所有尚未找到更高温度的下标，它们对应的温度从栈底到栈顶非递增。

弹栈会解决所有比当前温度低的栈顶日期。弹出结束后，栈顶温度大于等于当前温度，因此压入当前下标后，非递增性质仍然成立。

循环结束后仍留在栈中的日期，右侧没有更高温度。`answer` 的初始值已经是 `0`，不需要额外处理。

### 运行检查

```python
solution = Solution()

assert solution.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert solution.dailyTemperatures([73, 71, 72, 76]) == [3, 1, 1, 0]
assert solution.dailyTemperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert solution.dailyTemperatures([60, 50, 40]) == [0, 0, 0]
assert solution.dailyTemperatures([70, 70]) == [0, 0]
```

### 复杂度

- 时间复杂度：O(n)。每个下标最多压栈一次、弹栈一次。
- 额外空间复杂度：O(n)。最坏情况下所有下标都会留在栈中。

虽然代码中有嵌套的 `while`，但同一个下标不可能被重复弹出，因此所有 `while` 迭代加起来仍然是 O(n)。

## 常见错误

### 1. 栈里只保存温度

题目返回等待天数，需要计算 `i - previous`，因此必须保存下标。

### 2. 使用 `>=` 弹栈

相等温度不是更高温度。使用 `>=` 会错误结算 `[70,70]` 中的第 0 天。

### 3. 弹栈后忘记压入当前日期

今天解决了旧日期，但今天自己仍可能等待未来更高温度，因此最终必须执行 `stack.append(i)`。

### 4. 把栈描述为严格递减

由于相等温度会保留，准确说法是“栈中对应温度非递增”。

### 5. 看到嵌套循环就判断为 O(n²)

复杂度应按元素总压栈和总弹栈次数计算，而不是只看语法嵌套。

## 小结

这道题的推导路线是：

```text
明确右侧第一个严格更高温度
-> 对每一天向右扫描
-> 发现多个日期重复扫描相同后缀
-> 保存尚未解决的日期下标
-> 当前温度弹出并结算更低的栈顶日期
-> 每个下标最多入栈、出栈一次
```

单调栈在这里不是为了“让栈看起来有序”，而是为了维护仍未解决的候选日期，并在答案第一次出现时立即完成结算。
