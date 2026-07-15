---
title: "LeetCode 338：比特位计数，如何复用较小数字的结果"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "位运算", "动态规划", "二进制", "LeetCode 338"]
description: "从对 0..n 每个数字重复统计的 baseline 出发，推导 answer[i] = answer[i & (i - 1)] + 1，在线性时间内完成比特位计数。"
keywords: ["LeetCode 338", "Counting Bits", "比特位计数", "动态规划", "n & (n - 1)", "Hot100"]
---

## 题目要求

给定一个非负整数 `n`，返回一个长度为 `n + 1` 的数组 `answer`。

其中：

```text
answer[i] = 整数 i 的二进制表示中 1 的数量
```

需要回答的范围包含 `0` 和 `n`。

LeetCode 提供的方法接口是：

```text
countBits(n: int) -> List[int]
```

### 示例 1

```text
输入：n = 2
输出：[0,1,1]
```

对应关系是：

```text
0 -> 0   -> 0 个 1
1 -> 1   -> 1 个 1
2 -> 10  -> 1 个 1
```

### 示例 2

```text
输入：n = 5
输出：[0,1,1,2,1,2]
```

### 约束

- `0 <= n <= 10^5`

## Step 1：这次要回答 0 到 n 的所有数字

LeetCode 191 只要求统计一个整数。现在看 `n = 5`：

| `i` | 二进制 | `1` 的数量 |
| ---: | --- | ---: |
| 0 | `0` | 0 |
| 1 | `1` | 1 |
| 2 | `10` | 1 |
| 3 | `11` | 2 |
| 4 | `100` | 1 |
| 5 | `101` | 2 |

所以输出不是一个数字，而是：

```text
[0,1,1,2,1,2]
```

当前 baseline 是：

```text
已经会统计一个整数中 1 的数量。
```

这个 baseline 的 break 是：

> 单次统计只能得到一个结果，而本题需要按顺序返回 `0..n` 的所有结果。

现在这一版能做到：

- 准确说明 `answer[i]` 与整数 `i` 的对应关系。
- 知道输出长度必须是 `n + 1`。
- 知道 `answer[0]` 必须是 `0`。

它还缺：

- 一个能够生成整张结果表的正确版本。

## Step 2：先对每个数字重复一次 191

当前 baseline 已经会用下面的操作统计一个数：

```text
value & (value - 1)
```

它每次删除最低位的一个 `1`。最直接的批量方案，就是对 `0..n` 中每个数字都独立执行一次。

```python
from typing import List


def count_bits_repeated(n: int) -> List[int]:
    answer = []

    for value in range(n + 1):
        current = value
        count = 0

        while current:
            current &= current - 1
            count += 1

        answer.append(count)

    return answer
```

运行检查：

```python
assert count_bits_repeated(0) == [0]
assert count_bits_repeated(2) == [0, 1, 1]
assert count_bits_repeated(5) == [0, 1, 1, 2, 1, 2]
```

这个版本的循环关系很清楚：

- 外层循环依次处理 `0..n`。
- 内层循环独立统计当前 `value`。
- `answer` 只负责按顺序保存每次统计结果。

现在这一版能做到：

- 正确生成完整输出。
- 复用 191 中已经证明过的清位过程。
- 处理 `n = 0` 的边界情况。

它还缺：

- 每个数字都从头统计，没有使用 `answer` 中已经算好的较小数字结果。

如果整数需要 O(log n) 个二进制位，这个 baseline 的时间复杂度上界是 O(n log n)。

## Step 3：删除最低位的 1 后，结果已经算过

当前 baseline 对每个 `value` 反复执行：

```text
value = value & (value - 1)
```

但这里有一个此前没有利用的信息：对任意 `i > 0`，都有：

```text
i & (i - 1) < i
```

因为这个操作删除了 `i` 中最低位的一个 `1`。当我们按从小到大的顺序处理 `i` 时，这个更小数字的答案已经存在于 `answer` 中。

用 `i = 12` 检查：

```text
i             = 1100₂
i & (i - 1)   = 1000₂ = 8
```

`12` 比 `8` 恰好多一个 `1`，所以：

```text
answer[12] = answer[8] + 1
```

一般化以后得到：

```text
answer[i] = answer[i & (i - 1)] + 1
```

现在这个公式真正参与计算。按顺序检查 `1..5`：

| `i` | `i & (i - 1)` | 已有结果 | `answer[i]` |
| ---: | ---: | ---: | ---: |
| 1 | 0 | `answer[0] = 0` | 1 |
| 2 | 0 | `answer[0] = 0` | 1 |
| 3 | 2 | `answer[2] = 1` | 2 |
| 4 | 0 | `answer[0] = 0` | 1 |
| 5 | 4 | `answer[4] = 1` | 2 |

这时才需要把方法归类为动态规划：

- 当前状态：`answer[i]`。
- 更小子问题：`answer[i & (i - 1)]`。
- 状态转移：在更小结果上加 `1`。

现在这一版能做到：

- 用一次查表和一次加法得到每个新结果。
- 解释为什么依赖位置一定已经计算完成。
- 把 191 中的清位操作变成跨数字复用关系。

它还缺：

- 按正确顺序初始化并填完整张表的实现。

## Step 4：按从小到大的顺序填完整张表

首先创建输出数组：

```python
answer = [0] * (n + 1)
```

`answer[0] = 0` 已经是正确 base case。对 `i = 0` 使用递推式反而会得到自我依赖，因此循环从 `1` 开始。

完整的 LeetCode 实现是：

```python
from typing import List


class Solution:
    def countBits(self, n: int) -> List[int]:
        answer = [0] * (n + 1)

        for i in range(1, n + 1):
            answer[i] = answer[i & (i - 1)] + 1

        return answer
```

循环 invariant 是：

> 进入下标 `i` 的迭代前，`answer[0..i-1]` 已经全部正确。

对 `i > 0`，`i & (i - 1)` 严格小于 `i`，因此读取的一定是前缀中已经正确的结果。写入 `answer[i]` 后，正确前缀扩展一个位置，invariant 继续成立。

### 运行检查

```python
solution = Solution()

assert solution.countBits(0) == [0]
assert solution.countBits(2) == [0, 1, 1]
assert solution.countBits(5) == [0, 1, 1, 2, 1, 2]
```

### 复杂度

- 时间复杂度：O(n)。每个 `i` 只进行一次状态转移。
- 输出空间：O(n)。题目要求返回这张数组。
- 额外空间复杂度：O(1)。除输出数组外，没有使用随 `n` 增长的额外存储。

## 常见错误

### 1. 从 `i = 0` 开始套递推式

`0 & (0 - 1)` 在 Python 中仍会得到 `0`，从而把 `answer[0]` 错误地更新成 `answer[0] + 1`。应保留 base case，并从 `1` 开始循环。

### 2. 只写公式，不解释为什么能查表

关键证明是 `i & (i - 1) < i`。没有这个关系，就不能保证依赖已经计算完成。

### 3. 每个数字继续调用一次完整统计函数

这种写法是正确 baseline，但没有达到题目希望的 O(n) 批量复用。

### 4. 把输出数组算作额外空间

`answer` 是题目要求返回的结果。复杂度分析应区分输出空间 O(n) 与额外空间 O(1)。

## 小结

这道题的推导路线是：

```text
定义 answer[i]
-> 对每个数字重复执行 191
-> 发现清位后的更小数字已经算过
-> answer[i] = answer[i & (i - 1)] + 1
-> 按从小到大的顺序线性填表
```

191 教会我们删除一个整数最低位的 `1`；338 则进一步利用“删除后的结果已经计算过”，把位运算变成动态规划状态转移。
