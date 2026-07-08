---
title: "LeetCode 45：跳跃游戏 II，从可达边界升级到最少跳数"
date: 2026-07-08T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "贪心", "数组", "可达边界", "LeetCode 45"]
description: "从 LeetCode 55 的 farthest 可达边界出发，推导 LeetCode 45 跳跃游戏 II：用 current_end 表示当前跳数覆盖边界，用 farthest 表示下一跳最远覆盖，最终得到最少跳数贪心。"
keywords: ["LeetCode 45", "跳跃游戏 II", "Jump Game II", "贪心", "可达边界", "最少跳数", "Python"]
---

## 题目要求

给你一个整数数组 `nums`。

你一开始站在下标 `0`。

`nums[i]` 表示：

```text
从下标 i 最多可以向右跳多少步
```

题目要求返回：

```text
到达最后一个下标所需的最少跳跃次数
```

### 输入输出

- 输入：`nums: List[int]`
- 输出：`int`
- 从下标 `0` 出发
- 每个数字表示最大跳跃长度，不是必须跳这么远
- 题目保证一定可以到达最后一个下标
- 只需要返回最少跳数，不需要返回具体路径

### 示例

```text
输入：nums = [2,3,1,1,4]
输出：2
```

一种最少跳法是：

```text
0 -> 1 -> 4
```

从下标 `0` 跳到下标 `1`，再从下标 `1` 跳到最后一个下标。

```text
输入：nums = [2,3,0,1,4]
输出：2
```

虽然中间有 `0`，但仍然可以：

```text
0 -> 1 -> 4
```

### 约束

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 1000`
- 题目保证 `nums[n - 1]` 可达

## Step 1：45 不是问能不能到，而是问最少几步到

如果刚做完 LeetCode 55，已经知道可以维护：

```text
farthest = 当前所有可达位置能覆盖到的最远下标
```

这足够回答：

```text
能不能到终点？
```

但 45 的问题变了。

它问的是：

```text
最少跳几次到终点？
```

当前 baseline 是：

```text
只维护一个 farthest，判断终点是否可达。
```

这个 baseline 的 break 是：

> `farthest` 可以告诉我们覆盖范围能不能碰到终点，但它没有告诉我们什么时候应该把跳数加一。

看例子：

```text
nums = [2,3,1,1,4]
```

答案是 `2`，因为：

```text
0 -> 1 -> 4
```

这不是在问随便找一条路径，也不是在问终点是否可达。

真正的问题是：

```text
需要几层覆盖范围，才能覆盖到最后一个下标？
```

现在这一版能做到：

- 知道 45 是 55 的升级：从“是否可达”变成“最少跳数”。
- 知道单独的 `farthest` 不够，因为它没有计步边界。
- 知道后面要解释“每一跳覆盖到哪里”。

它还缺：

- 一个能说明“这一跳”和“下一跳”怎么分层的模型。

## Step 2：把一次跳跃看成一层覆盖范围

现在 baseline 是：

```text
需要计算最少跳数，也就是最少覆盖层数。
```

break 是：

> 如果只猜一条路径，比如先猜 `0 -> 1 -> 4`，可以得到一个答案，但不能说明它为什么是最少。

还是看：

```text
nums = [2,3,1,1,4]
```

从下标 `0` 出发。

第 `0` 层是起点：

```text
step 0: [0]
```

从下标 `0` 最多能跳 `2` 步，所以第 `1` 层可以覆盖：

```text
step 1: [1..2]
```

也就是下标 `1` 和下标 `2`。

现在不要急着选一条具体路径。

我们扫描第 `1` 层里的所有位置：

```text
i = 1, nums[1] = 3, 最远可以到 4
i = 2, nums[2] = 1, 最远可以到 3
```

这一层里最好的扩展来自下标 `1`，下一层可以覆盖到下标 `4`。

所以：

```text
step 2 可以到达终点
```

这个过程像 BFS，但我们不需要真的建图。

因为每一层在数组上都是一段连续范围：

```text
step 0: [0]
step 1: [1..2]
step 2: [3..4] 或更远
```

现在这一版能做到：

- 不猜单条路径，而是看当前跳数能覆盖的一整层。
- 知道最少跳数就是最少覆盖层数。
- 能解释 `[2,3,1,1,4]` 为什么两跳能到。

它还缺：

- 如果显式保存每一层范围，状态还是有点重；我们只需要边界。

## Step 3：用 current_end 和 farthest 表示两层边界

现在 baseline 是：

```text
每一跳覆盖一层连续下标范围。
```

break 是：

> 层的思想是对的，但没有必要真的保存每一层有哪些下标。扫描数组时，只维护边界就够了。

需要两个边界：

```text
current_end = 当前这一跳能覆盖到的最右边界
farthest    = 扫描当前层时，下一跳最远能覆盖到哪里
```

注意这两个变量职责不同。

`current_end` 回答：

```text
当前这一步的范围什么时候扫描完？
```

`farthest` 回答：

```text
扫描完当前这一层后，下一步最远能到哪里？
```

先只看 `farthest` 怎么更新。

当扫描到下标 `i` 时，如果 `i` 在当前层里，那么从 `i` 再跳一步，最远可以到：

```text
i + nums[i]
```

所以更新：

```python
farthest = max(farthest, i + nums[i])
```

用 `[2,3,1,1,4]`  trace：

```text
一开始：
current_end = 0
farthest = 0

i = 0:
farthest = max(0, 0 + nums[0]) = 2
```

这说明：

```text
从当前层出发，下一跳最远可以覆盖到下标 2。
```

继续扫描下一层时：

```text
i = 1:
farthest = max(2, 1 + nums[1]) = 4

i = 2:
farthest = max(4, 2 + nums[2]) = 4
```

这说明：

```text
扫描完这一层后，下一跳最远可以到下标 4。
```

现在这一版能做到：

- 用 `current_end` 表示当前跳数覆盖边界。
- 用 `farthest` 表示下一跳最远覆盖边界。
- 知道 `farthest` 的更新公式来自当前层里的每个位置。

它还缺：

- 什么时候说明当前层扫描完了，应该把跳数加一。

## Step 4：当 i == current_end，当前这一跳扫描完了

现在 baseline 是：

```text
扫描过程中维护 current_end 和 farthest。
```

break 是：

> `farthest` 会不断扩张，但跳数 `steps` 不能每扫描一个位置都加一。只有当前这一跳覆盖的范围扫描完，才需要进入下一跳。

当前这一跳的右边界是：

```text
current_end
```

所以当扫描下标满足：

```text
i == current_end
```

说明：

```text
当前这一跳能覆盖的所有位置都已经扫描完了。
```

这时必须使用下一跳：

```python
steps += 1
current_end = farthest
```

完整扫描逻辑已经出来了：

```python
steps = 0
current_end = 0
farthest = 0

for i in range(len(nums) - 1):
    farthest = max(farthest, i + nums[i])

    if i == current_end:
        steps += 1
        current_end = farthest
```

为什么循环只到：

```python
range(len(nums) - 1)
```

因为到达最后一个下标后，不需要再跳。

如果把最后一个下标也放进循环，可能会在终点处多加一次 `steps`。

用 `[2,3,1,1,4]` trace：

```text
start:
steps = 0, current_end = 0, farthest = 0

i = 0:
farthest = max(0, 0 + 2) = 2
i == current_end，当前第 0 层扫描完
steps = 1
current_end = 2

i = 1:
farthest = max(2, 1 + 3) = 4
i != current_end，不加 steps

i = 2:
farthest = max(4, 2 + 1) = 4
i == current_end，当前第 1 层扫描完
steps = 2
current_end = 4
```

此时已经知道两跳可以覆盖到终点。

现在这一版能做到：

- 在当前层结束时才增加跳数。
- 用 `current_end = farthest` 进入下一层。
- 避免在终点多加一步。

它还缺：

- LeetCode 完整函数、检查用例、复杂度和常见错误。

## Step 5：完整 LeetCode 写法

现在 baseline 是：

```text
用 current_end 和 farthest 扫描层边界，并在 i == current_end 时增加 steps。
```

break 是：

> 还没有把这个规则放进 LeetCode 要的 `jump(nums)` 方法里，也没有用边界用例验证。

完整代码：

```python
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                steps += 1
                current_end = farthest

        return steps
```

题目保证终点可达，所以这里不需要写不可达分支。

再看第二个例子：

```text
nums = [2,3,0,1,4]
```

trace：

```text
i = 0:
farthest = 2
到达 current_end，steps = 1，current_end = 2

i = 1:
farthest = max(2, 1 + 3) = 4

i = 2:
farthest = max(4, 2 + 0) = 4
到达 current_end，steps = 2，current_end = 4
```

虽然下标 `2` 是 `0`，但同一层里的下标 `1` 已经把下一层扩展到了终点。

现在这一版能做到：

- 返回最少跳数。
- 正确处理数组长度为 `1` 的情况。
- 正确处理当前层里有 `0`，但其他位置能扩展更远的情况。

## 正确性直觉

这题可以理解成压缩版 BFS。

每一次跳跃对应一层：

```text
当前 steps 能覆盖的下标范围
```

扫描这一层时，我们不急着选具体从哪个下标跳。

我们只记录：

```text
下一层最远能覆盖到哪里
```

这就是 `farthest`。

当 `i == current_end` 时，当前层已经扫描完。

这时如果还没到终点，就必须多用一跳，进入下一层：

```text
steps += 1
current_end = farthest
```

因为每一层代表“多跳一次后能到达的所有位置”，所以第一次覆盖到终点时，对应的 `steps` 就是最少跳数。

## 复杂度

只扫描数组一次：

```text
O(n)
```

只维护三个变量：

```text
steps
current_end
farthest
```

空间复杂度：

```text
O(1)
```

## 常见错误

### 1. 循环扫到最后一个下标

不要写：

```python
for i in range(len(nums)):
```

因为到达最后一个下标后，不需要再跳。

正确写法是：

```python
for i in range(len(nums) - 1):
```

### 2. 每次更新 farthest 就加 steps

`farthest` 更新的是下一层的最远边界。

它变大不代表已经完成了一次跳跃。

只有当：

```text
i == current_end
```

当前层扫描完，才应该加 `steps`。

### 3. 混淆 current_end 和 farthest

`current_end` 是：

```text
当前这一跳能覆盖到的边界
```

`farthest` 是：

```text
扫描当前层时，下一跳最远能覆盖到哪里
```

二者不能在每个位置都同步更新。

只有当前层结束时，才执行：

```python
current_end = farthest
```

## 可以直接运行的检查

```python
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                steps += 1
                current_end = farthest

        return steps


def check() -> None:
    s = Solution()
    assert s.jump([2, 3, 1, 1, 4]) == 2
    assert s.jump([2, 3, 0, 1, 4]) == 2
    assert s.jump([0]) == 0
    assert s.jump([1, 2]) == 1
    assert s.jump([1, 1, 1, 1]) == 3


check()
```

## 小结

45 的核心不是“每一步跳到当前能跳最远的位置”这么粗糙。

更准确的说法是：

```text
扫描当前跳数覆盖的一整层，
用 farthest 记录下一跳最远能覆盖到哪里；
当 i == current_end 时，当前层结束，steps 加一。
```

三个变量的职责是：

```text
steps       = 已经使用了多少次跳跃
current_end = 当前这一次跳跃能覆盖到的右边界
farthest    = 下一次跳跃最远能覆盖到的右边界
```

这就是从 55 的“可达边界”升级到 45 的“最少跳数边界”。
