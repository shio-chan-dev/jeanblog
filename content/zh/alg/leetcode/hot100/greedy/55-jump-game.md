---
title: "LeetCode 55：跳跃游戏，用最远覆盖范围判断能否到达终点"
date: 2026-07-03T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "贪心", "数组", "可达性", "LeetCode 55"]
description: "从会卡在 0 的失败样例出发，推导 LeetCode 55 跳跃游戏：先用 reachable 数组建立正确基线，再压缩成维护 farthest 最远覆盖范围的贪心。"
keywords: ["LeetCode 55", "跳跃游戏", "Jump Game", "贪心", "最远可达", "数组", "Python"]
---

## 题目要求

给你一个整数数组 `nums`。

你一开始站在下标 `0`。`nums[i]` 表示从位置 `i` 最多可以向右跳多少步。

题目要求判断：能不能到达最后一个下标。

### 输入输出

- 输入：`nums: List[int]`
- 输出：`bool`
- 从下标 `0` 出发。
- 每个位置的数字表示最大跳跃长度，不是必须跳这么远。
- 只需要判断能否到达最后一个下标，不需要返回具体路径。

### 示例

```text
输入：nums = [2,3,1,1,4]
输出：true
```

一种跳法是：

```text
0 -> 1 -> 4
```

从下标 `0` 可以跳到下标 `1`，再从下标 `1` 跳到最后一个下标。

```text
输入：nums = [3,2,1,0,4]
输出：false
```

无论怎么跳，都会被下标 `3` 的 `0` 卡住，无法到达最后一个下标 `4`。

### 约束

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 10^5`

## Step 1：不要先猜路径，先看覆盖范围

先看失败样例：

```text
nums = [3,2,1,0,4]
```

从下标 `0` 最多可以跳到下标 `3`。

看起来选择很多：

```text
0 -> 1
0 -> 2
0 -> 3
```

当前 baseline 是：

```text
试着选择一条跳跃路径，看能不能到终点。
```

这个 baseline 的 break 是：

> 题目不要求返回路径。盯着某一条路径，会忽略“当前整体能覆盖到哪里”这个更关键的问题。

在 `[3,2,1,0,4]` 里，能到达的位置最终都无法越过下标 `3`：

```text
index:  0  1  2  3  4
nums:   3  2  1  0  4
cover:  ----------^
```

下标 `4` 不在覆盖范围内，所以答案是 `False`。

这一步要改变的是视角：

> 不先猜具体路径，而是维护当前能覆盖到的最远下标。

现在这一版能做到：

- 知道题目是可达性判断，不是路径构造。
- 知道失败来自覆盖范围断掉，而不是某一步跳得不够漂亮。
- 知道后面要维护“最远可达位置”。

它还缺：

- 一个先正确、可运行的可达性判断。

## Step 2：先写可达性数组 baseline

当前 baseline 是：

```text
需要判断每个下标是否可达。
```

break 是：

> 只有“覆盖范围”的想法还不够具体，我们需要先写一个能跑的正确版本。

可以先用一个数组：

```text
reachable[i] == True 表示下标 i 可以从 0 到达
```

初始：

```python
reachable[0] = True
```

如果 `i` 可达，那么从 `i` 可以跳到：

```text
i + 1, i + 2, ..., i + nums[i]
```

先写一个正确 baseline：

```python
from typing import List


def can_jump_reachable(nums: List[int]) -> bool:
    n = len(nums)
    reachable = [False] * n
    reachable[0] = True

    for i in range(n):
        if not reachable[i]:
            continue

        right = min(n - 1, i + nums[i])
        for nxt in range(i + 1, right + 1):
            reachable[nxt] = True

    return reachable[-1]
```

检查：

```python
assert can_jump_reachable([2, 3, 1, 1, 4]) is True
assert can_jump_reachable([3, 2, 1, 0, 4]) is False
```

现在这个版本能做到：

- 明确哪些下标可达。
- 从可达下标向右标记新的可达位置。
- 正确处理官方两个例子。

它还缺：

- `reachable` 数组和内部标记循环比较重。我们其实只需要知道最右边能覆盖到哪里。

## Step 3：把可达数组压成 farthest

当前 baseline 是：

```text
reachable[i] 记录每个下标是否可达。
```

看成功样例：

```text
nums = [2,3,1,1,4]
```

从下标 `0` 能覆盖到 `2`：

```text
farthest = 2
```

只要扫描位置 `i <= farthest`，说明这个位置可达，就可以用它继续扩张右边界：

```text
farthest = max(farthest, i + nums[i])
```

break 是：

> 如果已经知道 `[0..farthest]` 这一段都可达，就不需要保存每个下标的布尔值。只维护右边界就够了。

于是把 `reachable` 压成一个变量：

```python
farthest = 0

for i, jump in enumerate(nums):
    if i <= farthest:
        farthest = max(farthest, i + jump)
```

用 `[2,3,1,1,4]` trace：

```text
start: farthest = 0

i = 0, jump = 2: i <= 0, farthest = max(0, 2) = 2
i = 1, jump = 3: i <= 2, farthest = max(2, 4) = 4
```

此时 `farthest` 已经到达最后一个下标，可以返回 `True`。

这一步之后，当前版本能做到：

- 用一个右边界表示当前可达覆盖范围。
- 从每个可达下标扩张覆盖范围。
- 用局部最远覆盖推进全局可达性。

它还缺：

- 如果扫描到一个不可达下标，应该立刻失败。

## Step 4：遇到断层就失败

当前 baseline 是：

```text
扫描可达下标，并更新 farthest。
```

break 是：

> 如果 `i > farthest`，说明当前位置不可达。继续用这个位置扩张覆盖范围是不合法的。

所以最终规则是：

- 如果 `i > farthest`，返回 `False`
- 否则用 `i + nums[i]` 更新 `farthest`
- 如果 `farthest >= last`，可以返回 `True`

完整代码：

```python
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last = len(nums) - 1
        farthest = 0

        for i, jump in enumerate(nums):
            if i > farthest:
                return False

            farthest = max(farthest, i + jump)

            if farthest >= last:
                return True

        return True
```

循环 invariant 是：

> 每次循环开始时，`farthest` 表示之前可达下标能覆盖到的最远位置；如果当前 `i` 超过它，当前下标不可达，后面也不能通过之前的位置补救。

检查：

```python
def check() -> None:
    s = Solution()

    assert s.canJump([2, 3, 1, 1, 4]) is True
    assert s.canJump([3, 2, 1, 0, 4]) is False
    assert s.canJump([0]) is True
    assert s.canJump([2, 0, 0]) is True
    assert s.canJump([1, 0, 1, 0]) is False


check()
```

现在这个版本能做到：

- 不构造具体路径。
- 不保存完整可达数组。
- 用 `farthest` 表示当前整体覆盖范围。
- 在覆盖范围断掉时返回 `False`。

## 复杂度

设 `n = len(nums)`。

- 时间复杂度：`O(n)`，每个下标最多扫描一次。
- 空间复杂度：`O(1)`，只维护 `farthest`。

## 小结

55 的贪心点是：

```text
只要当前下标 i 还在覆盖范围内，
它就可以用 i + nums[i] 尝试扩张覆盖右边界。
```

所以我们维护的不是路径，而是：

```text
farthest = 目前所有可达位置能覆盖到的最远下标
```

如果某一刻：

```text
i > farthest
```

说明扫描已经到达断层，后面的位置不可能被之前的跳跃覆盖，答案就是 `False`。
