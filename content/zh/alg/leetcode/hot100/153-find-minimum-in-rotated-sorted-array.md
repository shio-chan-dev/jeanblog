---
title: "LeetCode 153：寻找旋转排序数组中的最小值"
date: 2026-07-28T16:13:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二分查找", "旋转数组", "LeetCode 153"]
description: "从线性扫描基线推导包含最小值的闭区间与二分更新，在 O(log n) 时间、O(1) 额外空间内返回旋转数组的最小值。"
keywords: ["Find Minimum in Rotated Sorted Array", "寻找旋转排序数组中的最小值", "二分查找", "旋转数组", "LeetCode 153", "Hot100"]
---

## 题目要求

输入给出一个非空整数数组 `nums`。数组中的元素互不相同；旋转前，数组按严格递增顺序排列。

数组会被旋转 `1` 到 `nums.length` 次。每旋转一次，就把最后一个元素移到数组最前面；因此，旋转 `nums.length` 次后，数组恢复为原来的递增顺序。返回旋转后数组中的最小值。题目要求算法的运行时间为 `O(log n)`。

LeetCode 使用以下方法契约：

```text
class Solution:
    def findMin(self, nums: List[int]) -> int:
```

### 官方示例

```text
输入：nums = [3,4,5,1,2]
输出：1

输入：nums = [4,5,6,7,0,1,2]
输出：0

输入：nums = [11,13,15,17]
输出：11
```

### 约束

- `1 <= nums.length <= 5000`
- `-5000 <= nums[i] <= 5000`
- `nums` 中的所有整数互不相同。
- 旋转前，`nums` 按严格递增顺序排列。
- `nums` 被旋转 `1` 到 `nums.length` 次。

## Step 1：先得到一个肯定正确的最小值

对于 `[3,4,5,1,2]`，为什么不能直接返回第一个元素？

### 压力

这个数组的第一个元素是 `3`，但最小值是旋转后移到中间的 `1`。旋转保留了所有值，却不保证最小值仍在下标 `0`，所以直接返回 `nums[0]` 会得到错误答案。

### 上一个基线

当前基线只有非空输入、严格递增后旋转、元素唯一、返回最小值、官方示例、约束和 `Solution.findMin` 方法契约，还没有可执行的查找规则。

### 断点

方法契约只说明应该返回什么。它没有提供一个同时适用于任意合法旋转和旋转整整 `nums.length` 次后恢复原顺序这种情况的执行过程。

### 改动

在上一个基线中加入一次线性扫描。因为题目保证 `nums` 非空，`find_min_scan` 可以用 `nums[0]` 初始化当前最小值 `minimum`，再访问剩余的每个值。遇到更小的值时就更新 `minimum`，扫描结束后返回它。

```python
from typing import List


def find_min_scan(nums: List[int]) -> int:
    minimum = nums[0]

    for index in range(1, len(nums)):
        if nums[index] < minimum:
            minimum = nums[index]

    return minimum


assert find_min_scan([3, 4, 5, 1, 2]) == 1
assert find_min_scan([4, 5, 6, 7, 0, 1, 2]) == 0
assert find_min_scan([11, 13, 15, 17]) == 11
assert find_min_scan([7]) == 7
assert find_min_scan([4, 1, 2, 3]) == 1
assert find_min_scan([1, 2, 3, 4]) == 1
```

### 检查

前三个断言覆盖全部官方示例。后面三个断言依次检查单元素数组、把末尾元素移到最前面的一次旋转，以及旋转 `nums.length` 次后得到的原递增顺序。运行整个代码块时没有断言失败，就说明扫描不依赖最小值在旋转后的具体位置。

### 正确性

初始化后，`minimum` 是第一个元素的最小值。每访问一个剩余元素，若它更小就替换 `minimum`，否则保留原值。因此，每轮结束时，`minimum` 都是目前已经访问过的所有元素中的最小值。全部元素访问完后，它就是整个数组的最小值。

### 复杂度

设数组长度为 `n`。这个版本访问每个元素一次，所以时间复杂度是 `O(n)`；它只保存 `minimum` 和循环下标，额外空间复杂度是 `O(1)`。

### Step 1 结果

现在这个版本可以正确返回每个合法非空输入的最小值，包括单元素、一次旋转和旋转整整 `nums.length` 次的情况。

它仍然缺少：`O(n)` 线性扫描不满足题目要求的 `O(log n)` 运行时间。

## Step 2：哪一侧还能包含最小值

如果不再检查每个值，那么一次判断必须排除一部分位置，同时证明最小值仍留在没有被排除的位置中。怎样做到这一点？

### 压力

线性扫描之所以需要 `O(n)` 时间，是因为它从不排除尚未访问的值。要超过扫描，每一步都必须丢弃一部分值；但只缩小范围还不够，缩小后的范围必须继续包含真正的最小值，否则后面的判断都失去了正确性基础。

### 上一个基线

上一个版本用 `find_min_scan` 访问所有元素，并用运行中的 `minimum` 保证答案正确。它没有记录哪些位置仍可能是最小值，也没有任何经过证明的排除规则。

### 断点

一个容易想到的做法是把 `nums[mid]` 与数组的第一个元素比较，但这个比较不能单独给出适用于当前范围的稳定更新规则。对于 `[4,5,6,7,0,1,2]` 和 `[11,13,15,17]`，初始中点都大于第一个元素，前者的最小值却在中点右侧，后者的最小值在中点左侧。并且当范围左端移动后，原来的第一个元素可能已经不在当前范围内。

因此，我们需要一个始终属于当前范围的比较基准，并且要直接证明每次更新不会丢掉最小值。

### 改动：维护包含最小值的闭区间

用闭区间 `[left, right]` 表示当前仍可能包含最小值的下标，两个端点都包含在内。它的含义是：

> 真正最小值的下标一定在 `[left, right]` 中。

完整数组对应初始范围 `[0, len(nums) - 1]`。一步缩小只用于至少包含两个位置的范围，因此 `left < right`。中点为：

```text
mid = left + (right - left) // 2
```

此时 `mid < right`。题目保证所有值互不相同，所以 `nums[mid]` 与 `nums[right]` 不会相等，只需要处理下面两种情况。

#### `nums[mid] > nums[right]`

如果从 `mid` 到 `right` 仍保持普通递增顺序，就应该有 `nums[mid] < nums[right]`。现在关系相反，说明旋转产生的下降点位于这段范围中，下降点之后才是整个数组的最小值。因此，最小值只能在 `[mid + 1, right]` 中，`mid` 以及它左侧的当前位置都不可能是最小值，可以执行：

```text
left = mid + 1
```

#### `nums[mid] < nums[right]`

这时从 `mid` 到 `right` 没有跨过旋转产生的下降点，所以最小值不在 `[mid + 1, right]` 中。它仍在 `[left, mid]` 中，而且 `mid` 自己可能就是最小值。例如，在 `[3,4,5,1,2]` 的当前范围 `[3,4]` 中，`mid = 3`，对应的值 `1` 就是最小值。因此不能丢弃 `mid`，必须执行：

```text
right = mid
```

下面的辅助函数只执行一次比较和一次范围更新。它返回更新后的闭区间，不包含循环或最终答案的返回规则。

```python
from typing import List, Tuple


def shrink_minimum_interval_once(
    nums: List[int], left: int, right: int
) -> Tuple[int, int]:
    mid = left + (right - left) // 2

    if nums[mid] > nums[right]:
        left = mid + 1
    else:
        right = mid

    return left, right


first = [4, 5, 6, 7, 0, 1, 2]
first_interval = shrink_minimum_interval_once(first, 0, 6)
assert first_interval == (4, 6)
assert min(first) in first[first_interval[0] : first_interval[1] + 1]

second = [3, 4, 5, 1, 2]
second_interval = shrink_minimum_interval_once(second, 0, 4)
assert second_interval == (3, 4)
assert min(second) in second[second_interval[0] : second_interval[1] + 1]

second_interval = shrink_minimum_interval_once(
    second, second_interval[0], second_interval[1]
)
assert second_interval == (3, 3)
assert min(second) in second[second_interval[0] : second_interval[1] + 1]

third = [11, 13, 15, 17]
third_interval = shrink_minimum_interval_once(third, 0, 3)
assert third_interval == (0, 1)
assert min(third) in third[third_interval[0] : third_interval[1] + 1]
```

### 检查

代码中的每次调用都与下面的手动轨迹一致。最后一列直接检查更新后的闭区间仍包含真正的最小值。

| 数组 | 当前范围 | `mid` | 比较 | 下一范围 | 保留最小值的证据 |
| --- | --- | ---: | --- | --- | --- |
| `[4,5,6,7,0,1,2]` | `[0,6]` | `3` | `7 > 2` | `[4,6]` | `[0,1,2]` 包含 `0` |
| `[3,4,5,1,2]` | `[0,4]` | `2` | `5 > 2` | `[3,4]` | `[1,2]` 包含 `1` |
| `[3,4,5,1,2]` | `[3,4]` | `3` | `1 < 2` | `[3,3]` | 保留 `mid` 处的最小值 `1` |
| `[11,13,15,17]` | `[0,3]` | `1` | `13 < 17` | `[0,1]` | `[11,13]` 包含 `11` |

前两次大于分支都把 `left` 移到 `mid + 1`，排除了不可能成为最小值的中点及左侧位置。后两次小于分支都把 `right` 移到 `mid`；其中第三行具体证明了丢弃 `mid` 会直接丢掉答案。

### Step 2 结果

现在这个版本可以根据 `nums[mid]` 与当前 `nums[right]` 的比较安全地缩小一次闭区间，并保证真正的最小值仍在下一范围中。

它仍然缺少：这些更新还没有被组装成完整循环，也没有终止条件和最终返回规则。

## Step 3：候选区间何时就是答案

两个区间更新已经能够安全地保留最小值，但怎样重复执行它们，并在正确的时刻返回答案？

### 压力

`shrink_minimum_interval_once` 只完成一步。它没有说明何时继续、何时停止，也没有把最后保留的下标转换为题目要求的最小值。安全的一步更新还不是可提交的 `Solution.findMin`。

### 上一个基线

上一个版本已经维护闭区间 `[left, right]`，并保证真正最小值的下标始终在区间内。它也已经证明了两个更新：

- `nums[mid] > nums[right]` 时执行 `left = mid + 1`。
- 否则执行 `right = mid`；由元素唯一可知，此时实际关系是 `nums[mid] < nums[right]`。

### 断点

如果没有循环条件，就不知道应该执行多少次更新。如果没有严格变小的证明，循环可能无法结束。如果没有结束状态的含义，即使更新停下，也不知道应该返回哪个值。

### 改动：循环直到只剩一个下标

在上一个版本中，把已经证明的两条更新放进 `while left < right`。每轮都根据当前端点重新计算 `mid`；当条件不再成立时，闭区间只剩下 `left == right` 这一个下标。区间始终保留最小值，所以返回 `nums[left]`。

下面是文章中唯一的最终 `Solution.findMin` 实现。固定断言和随机差分检查继续复用 Step 1 的 `find_min_scan` 作为正确性基线，因此按顺序运行本文的 Python 代码块即可执行全部检查。

```python
from random import Random
from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        return nums[left]


solution = Solution()

assert solution.findMin([3, 4, 5, 1, 2]) == 1
assert solution.findMin([4, 5, 6, 7, 0, 1, 2]) == 0
assert solution.findMin([11, 13, 15, 17]) == 11
assert solution.findMin([7]) == 7
assert solution.findMin([4, 1, 2, 3]) == 1
assert solution.findMin([1, 2, 3, 4]) == 1

nums = [4, 5, 6, 7, 0, 1, 2]
snapshot = nums.copy()
assert solution.findMin(nums) == 0
assert nums == snapshot

rng = Random(153)

for length in range(1, 33):
    original = sorted(rng.sample(range(-5000, 5001), length))

    for rotations in range(1, length + 1):
        rotated = original[-rotations:] + original[:-rotations]
        snapshot = rotated.copy()

        assert solution.findMin(rotated) == find_min_scan(rotated)
        assert rotated == snapshot
```

### 严格推进

循环开始时有 `left < right`，所以中点满足 `left <= mid < right`。

- 执行 `left = mid + 1` 时，新 `left` 严格大于旧 `left`，并且不超过 `right`。
- 执行 `right = mid` 时，新 `right` 严格小于旧 `right`，并且不小于 `left`。

两个分支都会严格缩短闭区间，同时保持 `left <= right`。区间长度是有限的，因此循环一定会结束。

### 正确性证明

循环维持下面的不变量：

> 每轮开始时，真正最小值的下标都在闭区间 `[left, right]` 中。

**初始化：** 题目保证数组非空。初始区间 `[0, len(nums) - 1]` 包含所有下标，因此包含最小值的下标。

**保持：** 如果 `nums[mid] > nums[right]`，Step 2 已证明最小值在 `[mid + 1, right]` 中，执行 `left = mid + 1` 后不变量成立。否则，元素唯一使实际关系成为 `nums[mid] < nums[right]`；Step 2 已证明最小值在 `[left, mid]` 中，执行 `right = mid` 后不变量仍成立。

**终止与返回：** 严格推进保证循环结束。结束时 `left == right`，而不变量仍保证最小值下标在 `[left, right]` 中。这个区间只有一个下标，所以它必然是最小值下标，返回 `nums[left]` 正确。

### 较慢分支轨迹

在未旋转的 `[11,13,15,17]` 中，每次都进入保留 `mid` 的 `right = mid` 分支：

| `[left, right]` | `mid` | 比较 | 更新后的区间 |
| --- | ---: | --- | --- |
| `[0,3]` | `1` | `13 < 17` | `right = mid = 1`，得到 `[0,1]` |
| `[0,1]` | `0` | `11 < 13` | `right = mid = 0`，得到 `[0,0]` |

此时 `left == right == 0`，不变量说明唯一保留的下标就是最小值下标，因此返回 `nums[0] == 11`。

### 边界情况与检查

- **单元素：** 初始就是 `left == right == 0`，循环不会执行，直接返回唯一元素。
- **未旋转或完整旋转：** 当前范围始终严格递增，所以每轮都有 `nums[mid] < nums[right]`，`right` 不断移动到 `mid`，最终保留下标 `0`。
- **固定断言：** 六个断言覆盖全部官方示例、单元素、一次旋转和完整旋转。
- **随机差分：** 固定种子 `153` 生成 32 个取值互不相同的递增数组，并检查每个数组从 `1` 到 `length` 的全部旋转，共 528 个输入；最终实现必须与 Step 1 的线性扫描结果一致。
- **输入不变性：** 固定案例和全部随机案例都对调用前后的数组快照进行比较，确认方法不会修改 `nums`。

### 复杂度

设当前闭区间包含 `m = right - left + 1` 个候选。中点取下整后，无论更新哪一侧，下一轮保留的候选数都不超过 `ceil(m / 2)`。候选数量由此从 `n` 递减到 `1`，所以时间复杂度是 `O(log n)`。算法只保存 `left`、`right` 和 `mid`，额外空间复杂度是 `O(1)`。

### 总结

线性扫描先提供了不受旋转位置影响的正确基线。随后，包含最小值的闭区间把 `nums[mid]` 与当前 `nums[right]` 的比较转化为两条安全更新。最终循环通过严格推进收缩到唯一的最小值下标，在 `O(log n)` 时间和 `O(1)` 额外空间内返回答案，并且不修改输入数组。
