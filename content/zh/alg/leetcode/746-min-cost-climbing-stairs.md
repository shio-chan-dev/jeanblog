---
title: "LeetCode 746：使用最小花费爬楼梯，从 top 位置推出 dp"
date: 2026-05-03T14:33:39+08:00
draft: false
categories: ["LeetCode"]
tags: ["动态规划", "一维DP", "爬楼梯", "LeetCode 746"]
---

## 从 `[10,15,20]` 的 top 位置开始

题目给一个数组 `cost`，`cost[i]` 表示踩到第 `i` 阶需要付出的代价。每次付费后可以继续爬 `1` 阶或 `2` 阶。你可以从第 `0` 阶或第 `1` 阶开始，问到达楼顶的最小花费。

先看最小例子：

```text
cost = [10,15,20]
```

楼顶不是下标 `2`，而是在最后一阶之后的位置，可以记为位置 `3`。

到达 top 位置 `3` 的最后一步只可能来自：

- 位置 `2`，付 `cost[2]`
- 位置 `1`，付 `cost[1]`

这题最容易错的地方就在这里：我们要求的是“到达 top 的花费”，不是“到达最后一个下标的花费”。

## 题目事实

- 输入：整数数组 `cost`
- `cost[i]` 表示踩到第 `i` 阶的代价
- 每次可以爬 `1` 或 `2` 阶
- 可以从下标 `0` 或下标 `1` 开始
- 约束：`2 <= cost.length <= 1000`，`0 <= cost[i] <= 999`

示例：

```text
输入：cost = [10,15,20]
输出：15
解释：从下标 1 开始，付 15 后直接到达 top
```

```text
输入：cost = [1,100,1,1,1,100,1,1,100,1]
输出：6
```

## Step 1：先定义位置，而不是台阶下标

如果只盯着 `cost[i]`，很容易把答案错放在最后一个台阶。我们先定义位置：

```text
位置 0：第 0 阶
位置 1：第 1 阶
...
位置 n：top，最后一阶之后
```

定义状态：

```text
dp[i] = 到达位置 i 的最小花费
```

先写最小骨架：

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
```

现在这个版本能做到：

- 给 `0..n` 这些位置预留状态。
- 明确 `dp[n]` 才是 top 的答案。

它还缺：

- 起点花费怎么定义。
- 如何从前两个位置转移到当前位置。

## Step 2：起点 0 和 1 的花费都是 0

题目允许从下标 `0` 或下标 `1` 开始。开始站上去这件事本身不计入 `dp`；代价是在你从某个台阶继续往上爬时支付。

所以 base case 是：

```python
dp[0] = 0
dp[1] = 0
```

当前代码是：

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0
```

现在这个版本能做到：

- 表达“可以从 0 或 1 免费开始”。
- 为后续位置提供前两个来源。

它还缺：

- 到达第 `i` 个位置需要从哪里付费过来。

## Step 3：到达位置 i，只可能从 i-1 或 i-2 付费跳来

现在看 `i >= 2`。

如果最后一步从 `i - 1` 来，需要先到达位置 `i - 1`，然后支付 `cost[i - 1]`。
如果最后一步从 `i - 2` 来，需要先到达位置 `i - 2`，然后支付 `cost[i - 2]`。

在上一版基础上新增转移：

```python
for i in range(2, n + 1):
    dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
```

这条转移里的两项含义是：

- `dp[i - 1] + cost[i - 1]`：从前一阶付费走一步到 `i`
- `dp[i - 2] + cost[i - 2]`：从前两阶付费走两步到 `i`

第一版完整正确代码是：

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0

    for i in range(2, n + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

    return dp[n]
```

现在这个版本能做到：

- 正确计算到达每个位置的最小花费。
- 把答案放在 top 位置 `dp[n]`。

它还缺：

- 空间还能优化，因为每次只依赖前两个位置。

## Step 4：慢速走一遍表

用 `cost = [10,15,20]`：

| i | 位置含义 | 从 i-1 来 | 从 i-2 来 | dp[i] |
| --- | --- | --- | --- | --- |
| 0 | 起点 0 | - | - | 0 |
| 1 | 起点 1 | - | - | 0 |
| 2 | 第 2 个位置 | 0 + 15 | 0 + 10 | 10 |
| 3 | top | 10 + 20 | 0 + 15 | 15 |

答案是：

```text
dp[3] = 15
```

这张表要确认两件事：

- top 是位置 `n`，不是 `n - 1`。
- 支付的是“离开某个台阶时”的 `cost`。

## Step 5：把 dp 数组压成两个变量

上一版转移只依赖：

- `dp[i - 2]`
- `dp[i - 1]`

所以用两个变量保存即可：

```python
prev2 = 0
prev1 = 0
```

在上一版基础上，替换表数组：

```python
for i in range(2, n + 1):
    cur = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
    prev2 = prev1
    prev1 = cur
```

最终完整代码是：

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        n = len(cost)
        prev2 = 0
        prev1 = 0

        for i in range(2, n + 1):
            cur = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
            prev2 = prev1
            prev1 = cur

        return prev1


if __name__ == "__main__":
    print(Solution().minCostClimbingStairs([10, 15, 20]))
    print(Solution().minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))
```

现在这个版本能做到：

- 保持 `dp[i]` 的位置含义不变。
- 把额外空间从 `O(n)` 降到 `O(1)`。

它还缺：

- 如果题目改成能跳更多步，就需要保留更多来源；本题只允许 `1` 或 `2` 步。

## 正确性

不变量：

- 处理到位置 `i` 后，`prev1` 等于 `dp[i]`，`prev2` 等于 `dp[i - 1]`。

为什么转移正确：

- 到达位置 `i` 的最后一步只能来自位置 `i - 1` 或 `i - 2`。
- 从 `i - 1` 来必须支付 `cost[i - 1]`。
- 从 `i - 2` 来必须支付 `cost[i - 2]`。
- 题目要最小花费，所以取两者最小。

为什么返回 `prev1`：

- 循环结束时已经处理到位置 `n`。
- `prev1` 此时就是 `dp[n]`，也就是到达 top 的最小花费。

## 复杂度

- 时间复杂度：`O(n)`。
- 额外空间：`O(1)`。

## 常见错误

- 把答案写成到达最后一个台阶 `n - 1` 的花费，而不是 top 位置 `n`。
- 把 `dp[0]` 和 `dp[1]` 初始化成 `cost[0]`、`cost[1]`，导致“可以免费从 0/1 开始”的条件被破坏。
- 转移时写成 `cost[i]`，但到达位置 `i` 时支付的是来源台阶的费用。

## 小结

- 这题的关键是把 top 当作位置 `n`。
- `dp[i]` 表示到达位置 `i` 的最小花费。
- 到达 `i` 只能从 `i - 1` 或 `i - 2` 付费跳来。
- 空间优化只是把 `dp[i - 2]` 和 `dp[i - 1]` 留成两个变量。

## 参考与延伸

- LeetCode 70：Climbing Stairs
- LeetCode 746：Min Cost Climbing Stairs
- LeetCode 198：House Robber

## Notes

- 题意、示例和约束参考 LeetCode 746 的公开题目摘要。
- 代码语言按本仓库当前 LeetCode 教程默认选择 Python。
