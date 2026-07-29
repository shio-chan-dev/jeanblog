---
title: "LeetCode 34：在排序数组中查找元素的第一个和最后一个位置"
date: 2025-12-04T11:00:00+08:00
draft: false
aliases: ["/alg/leetcode/binary-search/34-find-first-and-last-position-of-element-in-sorted-array/"]
categories: ["LeetCode"]
tags: ["Hot100", "二分查找", "有序数组", "边界查找", "LeetCode 34"]
description: "从线性扫描基线出发，推导两个半开区间二分边界，在 O(log n) 时间内返回目标值的完整范围。"
keywords: ["Find First and Last Position of Element in Sorted Array", "Search Range", "二分查找", "边界查找", "有序数组", "LeetCode 34"]
---

## 题目要求

先看官方示例中的输入 `nums = [5,7,7,8,8,10]` 和 `target = 8`。答案必须是完整范围 `[3,4]`；只返回下标 `3` 或 `4`，都没有回答目标值第一次和最后一次出现在哪里。

给定一个按非递减顺序排列的整数数组 `nums` 和一个整数 `target`：

- 如果 `target` 存在，返回它第一次和最后一次出现的下标 `[first, last]`。
- 如果 `target` 不存在，返回 `[-1, -1]`。
- 题目最终要求算法的运行时间为 `O(log n)`。

LeetCode 使用以下方法契约：

```text
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
```

### 官方示例

```text
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

输入：nums = [], target = 0
输出：[-1,-1]
```

### 约束

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `-10^9 <= target <= 10^9`
- `nums` 按非递减顺序排列。

## Step 1：先得到一个肯定正确的范围

当目标值连续出现多次时，怎样保证同时记录最早和最晚的下标？

### 上一个基线

当前只有题目要求、示例、约束和 `Solution.searchRange` 方法契约，还没有可执行的方法。

### 断点

找到一个等于 `target` 的元素只能得到一个下标。对于 `[5,7,7,8,8,10]` 中的 `8`，这个下标可能是 `3`，也可能是 `4`；任何一个都不能单独表示完整范围。当前基线没有规则保证两个端点都被记录。

### 改动

在上一个基线中加入一次完整扫描：

- `first` 表示目前看到的第一次匹配，初始值为 `-1`，只在第一次匹配时更新。
- `last` 表示目前看到的最后一次匹配，初始值为 `-1`，每次匹配时都更新。

这样，如果没有匹配，两个值会保持为 `-1`；如果存在重复目标，第一次匹配和最后一次匹配会分别留下两个端点。

```python
from typing import List


def search_range_scan(nums: List[int], target: int) -> List[int]:
    first = -1
    last = -1

    for index, value in enumerate(nums):
        if value == target:
            if first == -1:
                first = index
            last = index

    return [first, last]


assert search_range_scan([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert search_range_scan([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert search_range_scan([], 0) == [-1, -1]
assert search_range_scan([2, 2, 2, 2], 2) == [0, 3]
assert search_range_scan([7], 7) == [0, 0]
assert search_range_scan([7], 8) == [-1, -1]
assert search_range_scan([1, 2, 3], 1) == [0, 0]
assert search_range_scan([1, 2, 3], 3) == [2, 2]
```

### 检查

上面的固定断言覆盖了三个官方示例、全部元素相等、单元素命中与未命中，以及目标位于数组第一个或最后一个位置的情况。运行整个代码块时没有断言失败，就说明这次改动处理了重复目标带来的两个端点问题。

### 复杂度

设数组长度为 `n`。这个版本会检查每个元素，因此时间复杂度是 `O(n)`；它只保存 `first`、`last`、`index` 和 `value`，额外空间复杂度是 `O(1)`。

### Step 1 结果

现在这个版本可以通过一次扫描正确返回目标值的第一次和最后一次出现位置，也能正确处理目标不存在的情况。

它仍然缺少：`O(n)` 的扫描不满足题目要求的 `O(log n)` 运行时间。

## Step 2：迫使搜索停在第一个候选位置

线性扫描已经能得到正确答案，但在长度达到 `10^5` 时，它可能检查每个元素。数组已经按非递减顺序排列，怎样利用这个顺序把搜索范围每次缩小一半？

### 上一个基线

上一个版本从左到右扫描整个数组，时间复杂度是 `O(n)`。它不会漏掉重复值，却没有利用数组有序这个条件。

一个只判断 `nums[mid] == target` 的普通二分也不够。以 `[5,7,7,8,8,10]` 和 `target = 8` 为例，搜索可能先在下标 `4` 命中并立即返回，但下标 `3` 也是 `8`。等值命中只能说明找到了某一个目标值，不能迫使搜索停在最靠左的候选位置。

### 断点

我们暂时把问题缩小为：

> 找到第一个满足 `nums[i] >= target` 的下标；如果所有元素都小于 `target`，就返回 `len(nums)`。

因为数组有序，判断 `nums[i] >= target` 的结果只会从 `False` 变成 `True` 一次。例如：

```text
nums:       [5,    7,    7,    8,    8,    10]
>= 8:       F     F     F     T     T      T
                                 ^
                           第一个 True
```

现在要找的不是任意一个等于 `target` 的元素，而是这组判断第一次变成 `True` 的边界。

### 改动：维护半开搜索区间

用 `[left, right)` 表示还没有被排除的实际数组下标。右端不包含在区间内，因此初始化为：

```python
left = 0
right = len(nums)
```

设真正要找的边界是 `boundary`。如果数组里没有满足条件的元素，`boundary` 就是 `len(nums)`。每轮循环开始时维持下面的不变量：

- `left <= boundary <= right`。
- 所有小于 `left` 的下标都已确定不满足条件，也就是对应值小于 `target`。
- 所有不小于 `right` 的实际数组下标都已确定满足条件，也就是对应值不小于 `target`。
- 尚未判断的实际数组下标位于半开区间 `[left, right)`。

取中点后只有两种情况：

```python
mid = left + (right - left) // 2

if nums[mid] >= target:
    right = mid
else:
    left = mid + 1
```

当 `nums[mid] >= target` 时，`mid` 自己可能就是第一个满足条件的位置，不能丢掉它，所以令 `right = mid`。

当 `nums[mid] < target` 时，`mid` 以及它左边的位置都不可能是边界，所以令 `left = mid + 1`。

这两个更新都保留 `left <= boundary <= right`。同时，`left < right` 时一定有 `left <= mid < right`，所以无论走哪个分支，`right - left` 都会严格减小。循环最终会在 `left == right` 时停止；结合不变量，此时只能有 `left == boundary`。

把这条规则写成当前阶段的辅助函数：

```python
from typing import List


def first_not_less(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] >= target:
            right = mid
        else:
            left = mid + 1

    return left


assert first_not_less([5, 7, 7, 8, 8, 10], 8) == 3
assert first_not_less([1, 3, 5], 4) == 2
assert first_not_less([], 0) == 0
assert first_not_less([2, 2, 2], 2) == 0
assert first_not_less([1, 3, 5], 0) == 0
assert first_not_less([1, 3, 5], 6) == 3
```

### 检查 1：重复目标

对 `[5,7,7,8,8,10]` 和 `target = 8` 手动执行：

| `left` | `right` | `mid` | `nums[mid] >= 8` | 更新后区间 |
| ---: | ---: | ---: | :---: | :--- |
| 0 | 6 | 3 | `True` | `[0, 3)` |
| 0 | 3 | 1 | `False` | `[2, 3)` |
| 2 | 3 | 2 | `False` | `[3, 3)` |

区间收缩到空时返回 `3`，它是第一个满足 `nums[i] >= 8` 的下标。即使第一次检查的下标 `3` 已经满足条件，搜索仍然保留左半侧，确认前面不存在更早的候选位置。

### 检查 2：目标夹在两个值之间

对 `[1,3,5]` 和 `target = 4`：

| `left` | `right` | `mid` | `nums[mid] >= 4` | 更新后区间 |
| ---: | ---: | ---: | :---: | :--- |
| 0 | 3 | 1 | `False` | `[2, 3)` |
| 2 | 3 | 2 | `True` | `[2, 2)` |

函数返回 `2`，表示 `4` 应该插入到下标 `2` 才能保持数组有序。这个结果只描述插入边界，不说明下标 `2` 的值等于 `4`。

### 检查 3：空数组

空数组初始化时就是 `left = right = 0`。循环一次也不会执行，函数直接返回 `0`。这里的 `0` 同时等于 `len(nums)`，仍然是合法的插入边界。

### Step 2 结果

现在这个版本可以在 `O(log n)` 时间内找到第一个值不小于 `target` 的位置；这个位置也就是保持有序时的插入边界。

它仍然缺少：这个插入位置本身不能证明 `target` 存在，也没有给出重复目标块的最后一个位置。

## Step 3：候选位置不等于真实命中

考虑 `nums = [1,3,5]` 和 `target = 4`。Step 2 的 `first_not_less` 返回 `2`，因为下标 `2` 是插入 `4` 后仍能保持有序的第一个位置。但 `nums[2]` 是 `5`，数组里根本没有 `4`。

怎样区分“可以插在这里”和“目标值确实从这里开始”？

### 上一个基线

当前版本已经有 `first_not_less(nums, target)`。它返回第一个满足 `nums[i] >= target` 的下标；如果所有值都小于 `target`，则返回 `len(nums)`。

这正是一个插入边界，却还不是一次确认过的命中。

### 断点

直接把插入边界当作起始位置会出现两类错误：

1. 对 `[1,3,5]` 和 `target = 4`，`start = 2` 仍在数组内，但 `nums[2] != 4`。
2. 对 `[1,3,5]` 和 `target = 6`，`start = 3`，恰好等于 `len(nums)`；此时访问 `nums[start]` 会越界。空数组也有同样的问题。

所以验证顺序不能交换。必须先判断 `start` 是否等于数组长度，只有它仍是实际下标时，才能读取 `nums[start]`。

### 改动：在读取数组之前排除越界位置

复用 Step 2 的 `first_not_less`，在它返回后加入这一条判断：

```python
if start == len(nums) or nums[start] != target:
    return [-1, -1]
```

Python 从左到右计算 `or`，并且会短路：

- 如果 `start == len(nums)` 为真，整个条件已经为真，不会执行 `nums[start]`。
- 只有 `start < len(nums)` 时，才会继续判断 `nums[start] != target`。
- 两个条件都为假时，`start` 才是一个实际下标，并且 `nums[start] == target`。

把这条判断接到上一步的辅助函数后面：

```python
from typing import List


def verified_start_or_absent(nums: List[int], target: int):
    start = first_not_less(nums, target)

    if start == len(nums) or nums[start] != target:
        return [-1, -1]

    return start


assert verified_start_or_absent([1, 3, 5], 0) == [-1, -1]
assert verified_start_or_absent([1, 3, 5], 6) == [-1, -1]
assert verified_start_or_absent([1, 3, 5], 4) == [-1, -1]
assert verified_start_or_absent([], 0) == [-1, -1]
assert verified_start_or_absent([1, 2, 2, 2, 4], 2) == 1
```

这个中间函数还不是 LeetCode 的完整方法。目标不存在时，它先沿用题目要求返回 `[-1, -1]`；目标存在时，它只暴露已经验证过的 `start`，不编造尚未求出的另一个下标。

### 检查：两个条件分别挡住什么

| 输入 | `first_not_less` 返回值 | 验证结果 |
| :--- | ---: | :--- |
| `[1,3,5]`, `target = 0` | `0` | `nums[0]` 是 `1`，返回 `[-1, -1]` |
| `[1,3,5]`, `target = 6` | `3` | `start == len(nums)`，短路后返回 `[-1, -1]` |
| `[1,3,5]`, `target = 4` | `2` | `nums[2]` 是 `5`，返回 `[-1, -1]` |
| `[]`, `target = 0` | `0` | `start == len(nums)`，短路后返回 `[-1, -1]` |
| `[1,2,2,2,4]`, `target = 2` | `1` | `nums[1] == 2`，保留已验证的起始下标 `1` |

最后一行还说明了为什么继续复用 `first_not_less`：通过验证的下标不仅包含目标值，而且仍然是目标值第一次出现的位置。

### Step 3 结果

现在这个版本可以安全地区分插入位置和真实命中：目标不存在时返回 `[-1, -1]`，目标存在时得到经过验证的第一次出现位置。

它仍然缺少：目标存在时，我们还没有计算它最后一次出现的位置，因此还不能返回完整范围。

## Step 4：找到目标块之后的第一个位置

Step 3 已经确认了 `start`，但它只告诉我们目标块从哪里开始，不知道这个块延伸到哪里。

一个直接做法是从 `start` 向右扫描，直到遇到不同的值：

```python
last = start
while last + 1 < len(nums) and nums[last + 1] == target:
    last += 1
```

如果数组的 `n` 个元素全都等于 `target`，这段扫描仍然要走过几乎整个数组，最坏时间复杂度是 `O(n)`。这样会丢掉前面二分查找已经获得的 `O(log n)` 优势。

### 断点

我们需要的不是从目标块内部逐个走到末尾，而是另一个可以二分查找的边界：

> 找到第一个满足 `nums[i] > target` 的下标；如果没有更大的元素，就返回 `len(nums)`。

对于 `[5,7,7,8,8,10]` 和 `target = 8`，判断结果是：

```text
nums:       [5,    7,    7,    8,    8,    10]
> 8:        F     F     F     F     F      T
                                             ^
                                        第一个 True
```

这个位置是 `5`。目标块占据它前面的连续下标，因此最后一个目标下标就是：

```text
end = first_greater - 1 = 5 - 1 = 4
```

### 改动：等于目标时继续向右

第二次搜索仍然使用半开区间 `[left, right)`，但谓词必须严格变成 `nums[mid] > target`：

```python
mid = left + (right - left) // 2

if nums[mid] > target:
    right = mid
else:
    left = mid + 1
```

设 `greater_boundary` 是第一个值大于 `target` 的位置，没有这样的值时取 `len(nums)`。每轮开始时维持：

- `left <= greater_boundary <= right`。
- 所有小于 `left` 的下标都已确定满足 `nums[i] <= target`。
- 所有不小于 `right` 的实际数组下标都已确定满足 `nums[i] > target`。
- 尚未判断的实际数组下标位于 `[left, right)`。

当 `nums[mid] > target` 时，`mid` 可能就是第一个更大值，所以保留它并令 `right = mid`。否则 `nums[mid] <= target`，包括所有等于 `target` 的位置，都不可能是第一个更大值，因此令 `left = mid + 1`。

这正是它与 `first_not_less` 的区别：

- 第一次搜索把 `nums[mid] == target` 视为谓词成立，执行 `right = mid`，保留这个位置并继续寻找更早的候选。
- 第二次搜索把 `nums[mid] == target` 视为谓词不成立，执行 `left = mid + 1`，越过这个位置并继续向右寻找。

两个分支都会严格缩小 `right - left`。循环在 `left == right` 时终止，再由不变量得到 `left == greater_boundary`。

### 检查：官方重复示例的两个边界

先找第一个满足 `nums[i] >= 8` 的位置：

| `left` | `right` | `mid` | 判断 | 更新后区间 |
| ---: | ---: | ---: | :--- | :--- |
| 0 | 6 | 3 | `8 >= 8` | `[0, 3)` |
| 0 | 3 | 1 | `7 < 8` | `[2, 3)` |
| 2 | 3 | 2 | `7 < 8` | `[3, 3)` |

第一次搜索返回 `start = 3`。

再找第一个满足 `nums[i] > 8` 的位置：

| `left` | `right` | `mid` | 判断 | 更新后区间 |
| ---: | ---: | ---: | :--- | :--- |
| 0 | 6 | 3 | `8 <= 8` | `[4, 6)` |
| 4 | 6 | 5 | `10 > 8` | `[4, 5)` |
| 4 | 5 | 4 | `8 <= 8` | `[5, 5)` |

第二次搜索返回 `first_greater = 5`，所以 `end = 5 - 1 = 4`，最终范围是 `[3,4]`。

### 组装最终的 LeetCode 方法

下面把已经验证过的第一次边界搜索直接展开到 `searchRange`，保留安全的存在性判断，再加入第二次边界搜索。两个循环没有合并成带配置参数的通用辅助函数，因为它们最重要的区别正是可见的 `>=` 与 `>`。

```python
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums)

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1

        start = left

        if start == len(nums) or nums[start] != target:
            return [-1, -1]

        left = 0
        right = len(nums)

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1

        first_greater = left
        end = first_greater - 1
        return [start, end]


solution = Solution()

assert solution.searchRange([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert solution.searchRange([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert solution.searchRange([], 0) == [-1, -1]
assert solution.searchRange([2, 2, 2, 2], 2) == [0, 3]
assert solution.searchRange([7], 7) == [0, 0]
assert solution.searchRange([7], 8) == [-1, -1]
assert solution.searchRange([1, 2, 3], 1) == [0, 0]
assert solution.searchRange([1, 2, 3], 3) == [2, 2]

unchanged = [5, 7, 7, 8, 8, 10]
snapshot = unchanged.copy()
assert solution.searchRange(unchanged, 8) == [3, 4]
assert unchanged == snapshot
```

### 随机差分检查

固定示例容易漏掉边界组合。复用 Step 1 的 `search_range_scan` 作为正确但较慢的基线，用固定随机种子生成有序数组，比较两个版本的结果。同时检查每次调用前后输入数组保持不变。

在同一个 Python 会话中运行 Step 1 的基线、上面的最终实现，再运行：

```python
import random


rng = random.Random(34)
solution = Solution()

for _ in range(1000):
    length = rng.randint(0, 50)
    nums = sorted(rng.randint(-10, 10) for _ in range(length))
    target = rng.randint(-12, 12)
    snapshot = nums.copy()

    assert solution.searchRange(nums, target) == search_range_scan(nums, target)
    assert nums == snapshot
```

### 正确性证明

**起始位置正确。** 第一次循环始终把第一个满足 `nums[i] >= target` 的边界保留在 `left` 和 `right` 之间。终止时 `start` 就是这个边界，所以 `start` 之前的所有值都小于 `target`。

**缺失判断正确且安全。** 如果 `start == len(nums)`，数组中没有值不小于 `target`，目标必然不存在，并且短路判断不会读取越界位置。如果 `start < len(nums)` 但 `nums[start] != target`，由边界定义可知 `nums[start] > target`，而它之前的值都小于 `target`，所以目标同样不存在。反过来，guard 未返回时有 `nums[start] == target`，且更早位置都小于目标，因此 `start` 是第一次出现的位置。

**结束位置正确。** 第二次循环返回第一个满足 `nums[i] > target` 的 `first_greater`。已经确认 `nums[start] == target`；由数组有序可知，从 `start` 到 `first_greater - 1` 的值既不小于 `target`，又不大于 `target`，所以都等于目标。`first_greater` 是第一个越过目标的边界，因此 `end = first_greater - 1` 正好是最后一次出现的位置。

三部分合起来，存在目标时返回 `[start, end]`，不存在时返回 `[-1, -1]`。

### 复杂度

两次半开区间二分查找各需要 `O(log n)` 时间，常数次检查和计算需要 `O(1)` 时间，因此总时间复杂度是 `O(log n)`。算法只使用若干整数变量，不创建随输入增长的辅助结构，所以辅助空间复杂度是 `O(1)`。

## 总结

- 线性扫描先提供了一个可以验证优化结果的正确基线，但最坏需要 `O(n)` 时间。
- 第一个满足 `nums[i] >= target` 的位置给出目标的起始候选；安全的长度与相等性检查负责判断目标是否存在。
- 第一个满足 `nums[i] > target` 的位置落在目标块之后，减一得到最后一次出现的位置。
- 两次搜索都使用 `[left, right)`，但等于目标时的处理不同：第一次保留当前下标并继续向左找，第二次丢弃当前下标并继续向右找。
- 最终 `Solution.searchRange` 在 `O(log n)` 时间和 `O(1)` 辅助空间内返回完整范围，并且不修改输入数组。
