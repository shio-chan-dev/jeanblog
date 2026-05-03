---
title: "LeetCode 198：打家劫舍，从偷或不偷推出一维 DP"
date: 2026-05-03T14:33:39+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "动态规划", "一维DP", "打家劫舍", "LeetCode 198"]
---

## 从 `[1,2,3,1]` 的相邻冲突开始

题目给一个数组 `nums`，`nums[i]` 表示第 `i` 间房子的金额。不能偷相邻两间房子，问最多能偷到多少钱。

看例子：

```text
nums = [1,2,3,1]
```

如果偷下标 `2` 的房子，金额是 `3`，那么下标 `1` 和下标 `3` 都不能偷。
如果不偷下标 `2`，答案可能来自前面下标 `0..1` 的最优结果。

所以走到某一间房时，核心选择只有两个：

- 偷当前房子：必须跳过前一间
- 不偷当前房子：沿用前一间之前的最优结果

这篇只用 Python，从这个二选一冲突推出一维 DP。

## 题目事实

- 输入：整数数组 `nums`
- `nums[i]` 表示第 `i` 间房子的金额
- 不能偷相邻房子
- 约束：`1 <= nums.length <= 100`，`0 <= nums[i] <= 400`

示例：

```text
输入：nums = [1,2,3,1]
输出：4
解释：偷下标 0 和下标 2，金额 1 + 3 = 4
```

```text
输入：nums = [2,7,9,3,1]
输出：12
解释：偷下标 0、2、4，金额 2 + 9 + 1 = 12
```

## Step 1：先定义“看到第 i 间为止”的最优值

直接问“整条街最多偷多少”太大。先定义：

```text
dp[i] = 只考虑下标 0..i 的房子时，最多能偷多少钱
```

这个定义有一个好处：处理第 `i` 间房子时，前面已经有一个稳定的最优结果。

先写最小骨架：

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
```

现在这个版本能做到：

- 给每个前缀 `0..i` 预留一个最优值。
- 明确 `dp[i]` 是“看到第 i 间为止”，不是“必须偷第 i 间”。

它还缺：

- 前一两间房子的 base case。
- 当前房子偷或不偷的转移。

## Step 2：先处理第一间和第二间

在上一版基础上，先填 base case。

只看第 `0` 间时，最多只能偷它：

```python
dp[0] = nums[0]
```

只看第 `0..1` 间时，因为相邻不能一起偷，只能取更大的那间：

```python
if n == 1:
    return dp[0]

dp[1] = max(nums[0], nums[1])
```

当前代码是：

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    if n == 1:
        return dp[0]

    dp[1] = max(nums[0], nums[1])
```

现在这个版本能做到：

- 正确处理 `n = 1`。
- 正确处理只有两间房子的相邻冲突。

它还缺：

- `i >= 2` 时如何做“偷或不偷”的选择。

## Step 3：第 i 间只有偷或不偷两种选择

现在看 `i >= 2`。

如果不偷第 `i` 间，那么最优值就是：

```text
dp[i - 1]
```

如果偷第 `i` 间，那么第 `i - 1` 间不能偷，只能接上：

```text
dp[i - 2] + nums[i]
```

在上一版基础上新增转移：

```python
for i in range(2, n):
    dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
```

第一版完整正确代码是：

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    if n == 1:
        return dp[0]

    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[n - 1]
```

现在这个版本能做到：

- 正确计算每个前缀里的最大可偷金额。
- 明确每一步都在比较“偷当前”和“不偷当前”。

它还缺：

- 空间还能优化，因为每次只依赖前两个前缀状态。

## Step 4：慢速走一遍表

用 `nums = [2,7,9,3,1]`：

| i | nums[i] | 不偷当前 dp[i-1] | 偷当前 dp[i-2]+nums[i] | dp[i] |
| --- | --- | --- | --- | --- |
| 0 | 2 | - | 2 | 2 |
| 1 | 7 | - | - | 7 |
| 2 | 9 | 7 | 2 + 9 = 11 | 11 |
| 3 | 3 | 11 | 7 + 3 = 10 | 11 |
| 4 | 1 | 11 | 11 + 1 = 12 | 12 |

答案是：

```text
dp[4] = 12
```

这张表要看的重点是：

> `dp[i]` 不要求第 i 间一定被偷，它表示看到第 i 间为止的全局最优。

## Step 5：把 dp 数组压成两个变量

上一版转移只依赖：

- `dp[i - 2]`
- `dp[i - 1]`

所以保留两个变量：

- `prev2`：上一轮的 `dp[i - 2]`
- `prev1`：上一轮的 `dp[i - 1]`

在上一版基础上替换数组：

```python
prev2 = nums[0]
prev1 = max(nums[0], nums[1])

for i in range(2, n):
    cur = max(prev1, prev2 + nums[i])
    prev2 = prev1
    prev1 = cur
```

最终完整代码是：

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        n = len(nums)

        if n == 1:
            return nums[0]

        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])

        for i in range(2, n):
            cur = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = cur

        return prev1


if __name__ == "__main__":
    print(Solution().rob([1, 2, 3, 1]))
    print(Solution().rob([2, 7, 9, 3, 1]))
```

现在这个版本能做到：

- 保持和 `dp` 表版本一样的状态含义。
- 把额外空间从 `O(n)` 降到 `O(1)`。

它还缺：

- 如果房子围成环，就会变成 `213. House Robber II`，需要拆成两个线性区间；本题只有一条直线。

## 正确性

不变量：

- 处理到房子 `i` 后，`prev1` 等于 `dp[i]`，即下标 `0..i` 内最多能偷的钱。
- `prev2` 等于上一轮的 `dp[i - 1]`，用于下一步转移。

为什么转移正确：

- 任意最优方案对第 `i` 间房子只有两种情况：偷或不偷。
- 不偷第 `i` 间，最优值就是前缀 `0..i-1` 的最优值 `dp[i - 1]`。
- 偷第 `i` 间，第 `i - 1` 间不能偷，只能加上 `dp[i - 2] + nums[i]`。
- 两类情况覆盖所有合法方案，取最大值即可。

为什么返回 `prev1`：

- 循环结束后，`prev1` 对应最后一个下标的前缀最优值。
- 这正是整条街的最大可偷金额。

## 复杂度

- 时间复杂度：`O(n)`。
- 额外空间：`O(1)`。

## 常见错误

- 把 `dp[i]` 定义成“必须偷第 i 间”，但转移又写成 `max(dp[i - 1], ...)`，状态含义冲突。
- 忘记处理 `n = 1`。
- 偷当前房子时接 `dp[i - 1]`，违反相邻不能偷的约束。
- 把这题和环形打家劫舍混在一起；本题首尾不相邻。

## 小结

- `dp[i]` 表示看到第 `i` 间为止的最大收益。
- 每个位置只有两种决策：偷当前或不偷当前。
- 偷当前必须接 `dp[i - 2]`，不偷当前就是 `dp[i - 1]`。
- 空间优化只是保留前两个 DP 状态。

## 参考与延伸

- LeetCode 198：House Robber
- LeetCode 213：House Robber II
- LeetCode 70：Climbing Stairs
- LeetCode 746：Min Cost Climbing Stairs

## Notes

- 题意、示例和约束参考 LeetCode 198 的公开题目摘要。
- 代码语言按本仓库当前 LeetCode 教程默认选择 Python。
