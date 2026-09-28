---
title: "LeetCode 213：打家劫舍 II，环形拆成两个线性问题"
date: 2026-09-22T10:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "动态规划", "一维DP", "环形数组", "打家劫舍", "LeetCode 213"]
---

## 题目要求

### 输入输出

- 输入：整数数组 `nums`，`nums[i]` 表示第 `i` 间房子的金额。
- 房子首尾相连，不能偷相邻的两间房子，因此第 `0` 间和第 `n - 1` 间也不能同时偷。
- 输出：在不触发警报的前提下，最多能偷到的金额。
- 约束：`1 <= nums.length <= 100`，`0 <= nums[i] <= 1000`。

### 示例

```text
输入：nums = [2,3,2]
输出：3
解释：第 0 间和第 2 间首尾相邻，不能同时偷；选择金额 3 的房子。
```

```text
输入：nums = [1,2,3,1]
输出：4
解释：可以偷下标 0 和下标 2，也可以偷下标 1 和下标 3。
```

```text
输入：nums = [1,2,3]
输出：3
```

这篇只用 Python，先从最小的首尾冲突开始。

## Step 1：先处理首尾相邻的冲突

先问一个具体问题：`nums = [2,3,2]` 时，能不能沿用直线街道的做法，直接在整个数组上选择不相邻的房子？

当前基线只有一条规则：相邻房子不能同时偷。若把这三个数当成直线，可能会把下标 `0` 和下标 `2` 当成彼此不相邻，从而错误地选择两端，得到金额 `4`。

这在环形输入上就会失败：首尾也相邻，`0` 和 `2` 必须互斥。要消除这一个冲突，只增加一个分支规则：任何合法方案至少属于下面一个候选分支。

- **排除最后一间**：只看下标 `0..n-2`，也就是 `nums[0:n-1]`。
- **排除第一间**：只看下标 `1..n-1`，也就是 `nums[1:n]`。

这两个区间分别代表“排除最后一间”和“排除第一间”的候选分支，因此每个区间内部已经是一条直线。它们共同覆盖所有合法方案：由于两端不能同时偷，每个合法方案至少有一端不被选择；如果两端都不偷，它可能同时落在两个候选区间里，这不影响最后取最大值。

用最小例子检查这个拆分：

```text
[2,3,2]

排除最后一间 -> [2,3]，区间内最多取 3
排除第一间   -> [3,2]，区间内最多取 3
两个结果取较大值 -> 3
```

现在这个版本能做到：

- 看见环形约束为什么不能直接当作一条直线。
- 把原题改写成两个候选、共同覆盖的线性区间问题。

它还缺：

- 如何计算任意一个线性区间的最优金额。
- 只有一间或两间房子时，线性区间的边界应该怎样处理。

## Step 2：先让一个线性区间处理一、两间房

现在只看 Task 1 产生的一个候选区间，例如 `[2,3]`。我们已经知道它是一条直线，但还没有一个函数能回答“这个区间最多能偷多少”。

当前基线是两个区间的边界描述：`[0..n-2]` 或 `[1..n-1]`。它能告诉我们该看哪一段，却不能保存这段前缀已经得到的最好结果。

这在区间只有一间或两间房时就必须先解决：

- 一间房只能取这一间。
- 两间房不能同时取，只能取两者较大的金额。

在上一版的两个区间模型上，只增加一个线性辅助函数，并先放入这两个 base case：

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size

    # dp[k]：只看 nums[left..left+k] 时的最大金额
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    return dp[1]
```

这里的 `dp[k]` 使用的是区间内的偏移量，而不是原数组下标：`dp[0]` 对应 `nums[left]`，`dp[1]` 对应 `nums[left..left+1]`。这个状态现在已经参与了两种 base case 的赋值和返回。

检查最小输入：

```python
assert rob_line([5], 0, 0) == 5
assert rob_line([2, 3], 0, 1) == 3
```

现在这个版本能做到：

- 求出任意长度为 1 或 2 的线性区间最优值。
- 让 `dp[k]` 的含义和原数组下标明确分开。

它还缺：

- 三间及以上的房子如何从前两个状态继续得到当前状态。
- 这两个 base case 之后的“偷当前或不偷当前”转移。

## Step 3：补上线性区间的偷或不偷转移

现在把区间扩大到 `[2,7,9]`。Task 2 的版本可以初始化前两间房，但在第三间 `9` 到来时就停住了：它没有说明应该把 `9` 接到哪个前缀结果上。

当前基线是一个只处理长度 1 或 2 的 `rob_line`。这在三间房上会失败，因为没有 `dp[2]`，也就无法回答“偷 `9`，还是沿用前两间的最好结果”。这里继续约定 `rob_line` 只接收非空区间（`left <= right`）；环形入口的单房 guard 留到后面处理。

在上一版中，只替换函数结尾的固定 `return dp[1]`，让每个后续位置都比较两种来源：

- 不偷当前房，沿用 `dp[offset - 1]`。
- 偷当前房，只能接上 `dp[offset - 2]`，再加上当前金额。

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size

    # dp[k]：只看 nums[left..left+k] 时的最大金额
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])

    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)

    return dp[-1]
```

检查这条转移：

```python
assert rob_line([2, 7, 9], 0, 2) == 11
assert rob_line([2, 7, 9, 3, 1], 0, 4) == 12
```

第一条断言的状态是 `2 -> 7 -> max(7, 2 + 9) = 11`。第二条继续得到 `11 -> max(11, 7 + 3) = 11 -> max(11, 11 + 1) = 12`。

现在这个版本能做到：

- 正确求出任意非空线性区间的最大可偷金额。
- 每个位置都明确比较“跳过当前”和“接上前两格”两种合法来源。

它还缺：

- 还没有把这个线性 helper 接回环形数组的两个候选分支。
- `n == 1` 时不能直接构造两个排除区间。

## Step 4：把线性 helper 接回环形数组

现在回到原题的环形输入。Task 3 已经能求一个非空线性区间，但还没有入口把两个候选区间都算出来。以 `[2,3,2]` 为压力：只调用一次 `rob_line` 会漏掉“排除第一间”或“排除最后一间”的其中一种合法可能。

当前基线是一个正确的 `rob_line(nums, left, right)`。这在环形数组上还不够，因为首尾冲突只能通过两个候选分支消除。这里先把入口前置限定为 `n >= 2`，单房输入留给下一步。

在上一版的 helper 外面只增加一个 `rob` 入口：分别求出排除最后一间和排除第一间的结果，再取较大值。`rob_line` 的线性转移完全不变。

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)
    return dp[-1]


def rob(nums: list[int]) -> int:
    n = len(nums)
    exclude_last = rob_line(nums, 0, n - 2)
    exclude_first = rob_line(nums, 1, n - 1)
    return max(exclude_last, exclude_first)
```

检查三个长度至少为 2 的环形例子：

```python
assert rob([2, 3, 2]) == 3
assert rob([1, 2, 3, 1]) == 4
assert rob([1, 2, 3]) == 3
```

现在这个版本能做到：

- 对 `n >= 2` 的环形数组，完整比较两个线性候选分支。
- 复用同一个线性 helper，而不是复制两套转移逻辑。

它还缺：

- `n == 1` 时两个候选区间并不都合法，入口需要先处理单房边界。
- 当前额外空间仍是 `O(n)`，因为线性 helper 保留了整张 `dp` 表。

## Step 5：先补上只有一间房的边界

现在用题目允许的最小输入 `[5]` 检查 Task 4。当前 `rob` 会直接构造两个排除区间，但长度为 `1` 时不存在两个合法的非空区间；问题不是线性转移，而是入口在拆分前少了边界判断。

当前基线是：`n >= 2` 时两个候选区间已经能给出正确结果。它在 `[5]` 上会把 `right` 算成 `-1`，因此无法安全调用 `rob_line`。

在上一版的完整代码中只增加一个早返回，必须放在两个 helper 调用之前；下面把保持不变的线性 helper 和更新后的环形入口放在同一个可运行代码块里：

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)
    return dp[-1]


def rob(nums: list[int]) -> int:
    n = len(nums)
    if n == 1:
        return nums[0]

    exclude_last = rob_line(nums, 0, n - 2)
    exclude_first = rob_line(nums, 1, n - 1)
    return max(exclude_last, exclude_first)
```

检查边界和常规样例：

```python
assert rob([5]) == 5
assert rob([2, 3]) == 3
assert rob([2, 3, 2]) == 3
assert rob([1, 2, 3, 1]) == 4
assert rob([1, 2, 3]) == 3
```

现在这个版本能做到：

- 覆盖题目约束中的所有非空长度，包括只有一间房的环。
- 在 `n >= 2` 时继续比较两个线性候选分支，得到第一版完整正确解。
- 时间复杂度为 `O(n)`，额外空间仍为 `O(n)`。

它还缺：

- 线性 helper 每次只读取前两个 `dp` 状态，却保留了整张数组；下一步可以把空间压缩到 `O(1)`。

## Step 6：只保留线性转移需要的两个状态

Task 5 已经得到第一版正确解，但线性 helper 的每一步只读取两个旧值：`dp[offset - 2]` 和 `dp[offset - 1]`。整张 `dp` 表不会再被读取，因此继续保存它只是额外空间。

当前基线是可运行的数组版：它的正确性已经覆盖单房、两房和环形样例，但空间复杂度为 `O(n)`。这一步只替换线性 helper 的存储方式，不改变两个环形候选区间，也不改变 `n == 1` guard。

在上一版 `rob_line` 中，用两个滚动状态保存最近的两个前缀结果：

- `prev2` 表示上一轮的 `dp[offset - 2]`。
- `prev1` 表示上一轮的 `dp[offset - 1]`。
- 先算当前值 `cur`，再把 `prev1` 平移到 `prev2`，把 `cur` 平移到 `prev1`。

```python
class Solution:
    @staticmethod
    def rob_line(nums: list[int], left: int, right: int) -> int:
        size = right - left + 1
        prev2 = nums[left]
        if size == 1:
            return prev2

        prev1 = max(nums[left], nums[left + 1])
        for offset in range(2, size):
            current = nums[left + offset]
            cur = max(prev1, prev2 + current)
            prev2, prev1 = prev1, cur

        return prev1

    def rob(self, nums: list[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        exclude_last = self.rob_line(nums, 0, n - 2)
        exclude_first = self.rob_line(nums, 1, n - 1)
        return max(exclude_last, exclude_first)
```

检查最终版本：

```python
solver = Solution()
assert solver.rob([2, 3, 2]) == 3
assert solver.rob([1, 2, 3, 1]) == 4
assert solver.rob([1, 2, 3]) == 3
assert solver.rob([5]) == 5
assert solver.rob([2, 7, 9, 3, 1]) == 11
```

最后一个例子是环形数组：排除最后一间得到 `[2,7,9,3]` 的线性结果 `11`，排除第一间得到 `[7,9,3,1]` 的结果 `10`，所以答案为 `11`。

正确性不变量是：处理完某个 `offset` 后，`prev1` 等于该区间前缀的最优值，`prev2` 等于前一个前缀的最优值；因此下一轮仍能准确计算“跳过当前”与“偷当前”。两个候选区间覆盖首尾冲突，单房 guard 覆盖最小输入。

现在这个版本能做到：

- 用两个线性候选区间解决环形首尾冲突。
- 覆盖全部非空输入，并保持与数组版相同的结果。
- 时间复杂度为 `O(n)`，额外空间降为 `O(1)`。

这就是本题最后一个增量 checkpoint；后续只需要独立审查教学链和最终代码，不再增加另一份会引入新逻辑的参考答案。
