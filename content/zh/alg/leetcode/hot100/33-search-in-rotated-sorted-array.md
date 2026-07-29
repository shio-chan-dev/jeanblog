---
title: "LeetCode 33：搜索旋转排序数组"
date: 2026-07-28T15:40:34+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二分查找", "旋转数组", "LeetCode 33"]
description: "从线性扫描基线推导旋转数组中的有序半区与目标范围判断，在 O(log n) 时间、O(1) 额外空间内完成搜索。"
keywords: ["Search in Rotated Sorted Array", "搜索旋转排序数组", "二分查找", "旋转数组", "LeetCode 33", "Hot100"]
---

## 题目要求

输入给出一个整数数组 `nums` 和一个整数 `target`。`nums` 中的元素互不相同；旋转前，数组严格递增。调用方法前，数组可能在某个未知下标 `k`（`0 <= k < nums.length`）处旋转为：

```text
[nums[k], ..., nums[n-1], nums[0], ..., nums[k-1]]
```

如果 `target` 存在，返回它在旋转后数组中的下标；否则返回 `-1`。题目要求算法的运行时间为 `O(log n)`。

LeetCode 使用以下方法契约：

```text
class Solution:
    def search(self, nums: List[int], target: int) -> int:
```

### 官方示例

```text
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

输入：nums = [1], target = 0
输出：-1
```

### 约束

- `1 <= nums.length <= 5000`
- `-10^4 <= nums[i], target <= 10^4`
- `nums` 中的每个值都互不相同。
- `nums` 旋转前按严格递增顺序排列。

## Step 1：先正确搜索每一种旋转

对于 `[4,5,6,7,0,1,2]`，怎样先得到一个不受旋转位置影响的正确答案？

### 压力

这个数组包含 `[4,5,6,7]` 和 `[0,1,2]` 两个递增片段，但从 `7` 到 `0` 的跳变说明整个数组已经不再按普通递增顺序排列。因此，普通二分查找依赖的整体有序前提在旋转后不再成立，不能直接照搬。

### 上一个基线

当前基线只有题目输入、输出、官方示例、约束和 `Solution.search` 方法契约，还没有可执行的搜索方法。

### 断点

方法契约只规定了应该返回什么，却没有给出一个能在每一种合法旋转中都保持正确的执行过程。

### 改动

先在上一个基线中加入一次线性扫描。`search_scan` 按下标依次检查每个值；遇到 `target` 就返回当前下标，扫描结束仍未命中则返回 `-1`。旋转只改变元素的排列位置，不会让这次逐项检查漏掉目标。

```python
from typing import List


def search_scan(nums: List[int], target: int) -> int:
    for index, value in enumerate(nums):
        if value == target:
            return index
    return -1


assert search_scan([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_scan([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert search_scan([1], 0) == -1
assert search_scan([1, 3, 5, 7], 5) == 2
assert search_scan([9], 9) == 0
assert search_scan([9], 4) == -1
assert search_scan([6, 7, 1, 2, 3, 4, 5], 1) == 2
assert search_scan([6, 7, 1, 2, 3, 4, 5], 8) == -1
```

### 检查

这些固定断言覆盖了三个官方示例、未旋转数组、单元素命中与未命中、目标正好位于旋转点，以及目标不存在的情况。整个代码块运行时没有断言失败，就说明线性扫描处理了旋转破坏整体有序性带来的正确性问题。

### 复杂度

设数组长度为 `n`。最坏情况下需要检查全部 `n` 个元素，因此时间复杂度是 `O(n)`；除循环变量外没有使用随输入规模增长的存储，因此额外空间复杂度是 `O(1)`。

### Step 1 结果

线性扫描可以在任意合法旋转中返回目标下标，并在目标不存在时返回 `-1`。它的答案正确，但 `O(n)` 的运行时间还不满足题目要求。

## Step 2：每轮哪一半仍然有序

线性扫描已经正确，但它没有利用旋转数组中仍然保留的局部顺序。要走向对数时间搜索，每个当前区间中都必须有一段连续的半区能够被证明保持普通递增顺序。

### 压力

以 `[4,5,6,7,0,1,2]` 为例，整个数组不递增，但连续片段 `[4,5,6,7]` 仍然递增。现在需要的不是假设整个数组有序，而是在当前区间中找到一段可以证明按普通顺序排列的连续半区。

### 上一个基线

上一个版本是正确的 `O(n)` 线性扫描。它逐个检查元素，不依赖任何有序性，因此适用于每一种合法旋转，但最坏情况下必须检查整个数组。

### 断点

当前区间可能跨过旋转点，因此整个区间并非全局单调。只知道原数组旋转前严格递增，还不足以把当前整个区间当作普通有序数组。

### 改动：判断有序半区

引入闭合候选区间 `[left, right]`：

- `left` 是当前区间包含的第一个下标。
- `right` 是当前区间包含的最后一个下标。
- `mid = left + (right - left) // 2` 是当前区间的中点。
- 左半区是 `[left, mid]`，右半区是 `[mid, right]`；两个半区都包含 `mid`。

对于合法的旋转数组，左半区有序当且仅当 `nums[left] <= nums[mid]`；否则，右半区有序。

这个结论依赖两个题目事实。旋转前严格递增且只旋转一次，意味着整个数组最多只有一处从大值跳到小值的断点。把一个连续区间在 `mid` 处分开，这个断点不可能同时落在两个半区内部，所以至少一个半区保持普通递增顺序。

元素互不相同让端点比较能够识别断点：如果 `[left, mid]` 没有跨过断点，它按严格递增顺序排列，因此 `nums[left] <= nums[mid]`；如果它跨过断点，断点前的值都大于断点后的值，因此 `nums[left] > nums[mid]`。后一种情况下，唯一的断点已经出现在左半区，右半区必然有序。

下面的函数只做一次分类，不改变 `left` 或 `right`：

```python
from typing import List


def classify_sorted_half(nums: List[int], left: int, right: int) -> str:
    mid = left + (right - left) // 2
    if nums[left] <= nums[mid]:
        return "left"
    return "right"


assert classify_sorted_half([6, 7, 0, 1, 2, 4, 5], 0, 6) == "right"
assert classify_sorted_half([4, 5, 6, 7, 0, 1, 2], 0, 6) == "left"
assert classify_sorted_half([0, 1, 2, 4, 5, 6, 7], 0, 6) == "left"
```

### 检查

三个断言对应下面三种形状。每一行都只计算当前中点并判断有序半区，没有改变区间。

| 情况 | `nums` | `[left, right]` | `mid` | 端点比较 | 分类依据与结果 |
| --- | --- | --- | ---: | --- | --- |
| 旋转点在 `mid` 左侧 | `[6,7,0,1,2,4,5]` | `[0, 6]` | 3 | `6 <= 1` 为假 | 左半区 `[6,7,0,1]` 跨过旋转点；右半区 `[1,2,4,5]` 有序 |
| 旋转点在 `mid` 右侧 | `[4,5,6,7,0,1,2]` | `[0, 6]` | 3 | `4 <= 7` 为真 | 左半区 `[4,5,6,7]` 有序 |
| 未旋转 | `[0,1,2,4,5,6,7]` | `[0, 6]` | 3 | `0 <= 4` 为真 | 两个半区都有序；分类规则返回左半区 |

第一行的旋转点下标是 `2`，位于 `mid = 3` 左侧，所以左半区跨过断点，分类结果是右半区。第二行的旋转点下标是 `4`，位于 `mid = 3` 右侧，所以左半区没有跨过断点。第三行没有旋转断点，端点比较仍然稳定地识别出一个有序半区。

### Step 2 结果

对于任意当前闭区间，现在可以判断左半区是否有序；若不是，则能确定右半区有序。不过，有序半区标签本身还不能说明哪一半可以包含 `target`。

## Step 3：保留可能包含目标的半区

有序半区标签只有在能够证明 `target` 是否落入该半区的值域时才有用。在完成这个判断之前，任何位置都不能被安全排除。

### 压力

在 `[4,5,6,7,0,1,2]` 中，第一次取到 `mid = 3` 时可以识别出左半区 `[4,5,6,7]` 有序。但是，搜索 `6` 时应该保留左侧，搜索 `0` 时却应该保留右侧。只有“左半区有序”这个标签，无法支持任何一次区间缩小。

### 上一个基线

上一个版本使用闭区间 `[left, right]` 表示当前候选位置，并通过 `nums[left] <= nums[mid]` 判断左半区是否有序；否则可以确定右半区有序。它还没有比较 `target` 与有序半区的端点。

### 断点

当前规则没有证明哪一半包含目标，因此还没有任何区间更新能够保证保留一个已存在的 `target`。在这个证明完成前，丢弃任意一半都可能漏掉答案。

### 改动：用目标值域更新闭区间

在每轮循环开始时维持下面的候选区间不变量：

> 如果 `target` 存在于数组中，那么它的下标一定包含在当前闭区间 `[left, right]` 中。

初始化为 `[0, len(nums) - 1]`，因此所有数组下标都在候选区间内。取出 `mid` 后，必须先判断 `nums[mid] == target` 并立即返回；后面的值域判断可以排除 `mid`，正是因为已经确认它不等于目标。

接下来把 Task 2 的有序半区分类转换为区间更新：

- 如果左半区有序，使用 `nums[left] <= target < nums[mid]` 判断目标是否落在左半区的值域中。
  - 条件成立时，保留 `[left, mid - 1]`，即令 `right = mid - 1`。
  - 否则保留 `[mid + 1, right]`，即令 `left = mid + 1`。
- 否则右半区有序，使用 `nums[mid] < target <= nums[right]` 判断目标是否落在右半区的值域中。
  - 条件成立时，保留 `[mid + 1, right]`，即令 `left = mid + 1`。
  - 否则保留 `[left, mid - 1]`，即令 `right = mid - 1`。

两个值域都排除 `nums[mid]`，因为中点相等分支已经先执行。闭区间端点仍然包含在判断中，所以等于 `nums[left]` 或 `nums[right]` 的目标不会被错误排除。

### 最终实现

把已经推导出的中点相等、有序半区分类、目标值域判断和边界更新整合进唯一的 LeetCode 实现：

```python
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid

            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1
```

### 为什么更新保持候选区间不变量

- **初始化**：`[0, len(nums) - 1]` 包含所有合法下标，所以存在的目标一定在区间内。
- **中点命中**：如果 `nums[mid] == target`，直接返回正确下标，不需要继续维持区间。
- **左半区有序**：严格递增使 `nums[left] <= target < nums[mid]` 精确描述目标是否位于 `[left, mid - 1]`。条件成立时更新 `right`；否则，中点又已排除，存在的目标只能位于 `[mid + 1, right]`。
- **右半区有序**：同理，`nums[mid] < target <= nums[right]` 精确描述目标是否位于 `[mid + 1, right]`。条件成立时更新 `left`；否则，存在的目标只能位于 `[left, mid - 1]`。
- **目标不存在**：不变量只约束“目标存在”的情况。目标不存在时，区间仍会持续缩小，最终返回 `-1`。

这些推理只覆盖题目规定的互不相同元素。唯一性既让 Task 2 的有序半区判断没有相等歧义，也让有序半区内的值域判断是严格的；这里不加入题目范围之外的重复值回退分支。

每次未命中时都会执行 `right = mid - 1` 或 `left = mid + 1`，所以新的闭区间严格短于旧区间。当 `left > right` 时候选区间为空，循环必然终止。

### 检查 1：目标在左侧有序半区

对 `[4,5,6,7,0,1,2]` 搜索 `6`：

| `left` | `right` | `mid` | `nums[mid]` | 判断 | 结果 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | 左半区有序，`4 <= 6 < 7` | `right = 2` |
| 0 | 2 | 1 | 5 | 左半区有序，但 `4 <= 6 < 5` 为假 | `left = 2` |
| 2 | 2 | 2 | 6 | 中点命中 | 返回 `2` |

### 检查 2：目标在右侧有序半区

对 `[6,7,0,1,2,4,5]` 搜索 `4`：

| `left` | `right` | `mid` | `nums[mid]` | 判断 | 结果 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 1 | 右半区有序，`1 < 4 <= 5` | `left = 4` |
| 4 | 6 | 5 | 4 | 中点命中 | 返回 `5` |

### 检查 3：目标位于旋转点

对 `[4,5,6,7,0,1,2]` 搜索旋转点处的 `0`：

| `left` | `right` | `mid` | `nums[mid]` | 判断 | 结果 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | 左半区有序，但 `4 <= 0 < 7` 为假 | `left = 4` |
| 4 | 6 | 5 | 1 | 左半区有序，`0 <= 0 < 1` | `right = 4` |
| 4 | 4 | 4 | 0 | 中点命中 | 返回 `4` |

### 检查 4：目标不存在

对 `[4,5,6,7,0,1,2]` 搜索 `3`：

| `left` | `right` | `mid` | `nums[mid]` | 判断 | 结果 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | 左半区有序，但 `4 <= 3 < 7` 为假 | `left = 4` |
| 4 | 6 | 5 | 1 | 左半区有序，但 `0 <= 3 < 1` 为假 | `left = 6` |
| 6 | 6 | 6 | 2 | 单元素左半区有序，但 `2 <= 3 < 2` 为假 | `left = 7` |

此时 `left = 7 > right = 6`，候选区间为空，返回 `-1`。

### 可执行验证

下面的检查接在前面 Task 1 的 `search_scan` 和最终 `Solution` 之后运行。固定断言覆盖官方示例、单元素、未旋转数组、旋转点、两元素数组和缺失目标；随后用固定种子生成长度 `1` 到 `40` 的严格递增数组，检查每一种旋转，并确认搜索不会修改输入。

```python
from random import Random


solution = Solution()

assert solution.search([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert solution.search([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert solution.search([1], 0) == -1
assert solution.search([1], 1) == 0
assert solution.search([1, 3, 5, 7], 1) == 0
assert solution.search([1, 3, 5, 7], 7) == 3
assert solution.search([6, 7, 1, 2, 3, 4, 5], 1) == 2
assert solution.search([6, 7, 0, 1, 2, 4, 5], 4) == 5
assert solution.search([6, 7, 1, 2, 3, 4, 5], 8) == -1
assert solution.search([3, 1], 3) == 0
assert solution.search([3, 1], 1) == 1

nums = [4, 5, 6, 7, 0, 1, 2]
before = nums.copy()
assert solution.search(nums, 0) == 4
assert nums == before

rng = Random(33)
checked_cases = 0

for length in range(1, 41):
    original = sorted(rng.sample(range(-10_000, 10_001), length))
    missing = []

    while len(missing) < 3:
        candidate = rng.randint(-10_000, 10_000)
        if candidate not in original and candidate not in missing:
            missing.append(candidate)

    for rotation in range(length):
        rotated = original[rotation:] + original[:rotation]

        for target in original + missing:
            before = rotated.copy()
            assert solution.search(rotated, target) == search_scan(rotated, target)
            assert rotated == before
            checked_cases += 1

assert checked_cases == 24_600
```

### 复杂度

每轮只做常数次比较，并把候选闭区间缩小到中点一侧，区间长度至多约减半。因此最多执行 `O(log n)` 轮，时间复杂度是 `O(log n)`。算法只保存 `left`、`right` 和 `mid` 等常数个变量，不修改输入数组，额外空间复杂度是 `O(1)`。

## 总结

旋转破坏了整个数组的普通有序性，但任意当前区间在中点两侧至少有一个普通有序半区。先处理中点相等，再用有序半区的端点判断 `target` 是否落入其值域，就能让每次边界更新都保留一个已存在的目标。候选区间持续缩小，最终得到符合题目要求的 `O(log n)` 时间、`O(1)` 额外空间搜索。
