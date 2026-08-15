---
title: "二分查找区间怎么选：从候选集合到边界更新"
date: 2026-08-14
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "搜索区间", "循环不变式", "边界问题"]
description: "从候选集合出发，理解二分查找的闭区间、左闭右开区间和开区间哨兵写法，以及初始化、循环条件与边界更新为什么必须配套。"
keywords: ["二分查找区间", "闭区间", "左闭右开", "循环不变式", "binary search interval"]
---

写二分查找时，我们经常先看到这样的初始化：

```python
left = 0
right = len(nums) - 1
```

本文讨论的输入是按非递减顺序排列、允许重复值的 `list[int]`。精确查找要返回某个等于 `target` 的实际下标，不存在时返回 `-1`；左边界查找要返回 `[0, n]` 中第一个满足 `nums[i] >= target` 的位置，如果不存在这样的数组元素，就返回 `n`。

它看起来像是在声明一个闭区间，但只看这两行还不够。真正决定区间含义的是下面四件事是否一致：

1. `left` 和 `right` 是候选下标、未分类边界，还是哨兵；
2. 循环在什么条件下说明仍有实际元素需要检查；
3. 检查 `mid` 后，更新是否正确排除了它；
4. 循环结束时，哪个位置承载最终答案。

最可靠的判断方法不是背代码，而是先问一句：

> 当前哪些实际下标还没有被排除或分类，答案位置又被限制在哪个范围内？

这些持续成立的事实就是二分查找的循环不变式。区间符号、初始化和更新规则都应该从它推出来。

## 闭区间 `[left, right]`

当 `left` 和 `right` 都指向尚未排除的实际数组下标时，候选集合是闭区间：

```python
left = 0
right = len(nums) - 1
```

此时两个端点都可能是答案，所以只要 `left <= right`，区间就仍然非空：

```python
while left <= right:
```

如果 `mid` 不是答案，下一轮必须把它排除：

```python
if nums[mid] < target:
    left = mid + 1
else:
    right = mid - 1
```

因此，这套约定是：

| 项目 | 规则 |
| --- | --- |
| 候选集合 | `[left, right]` |
| 初始化 | `left = 0`, `right = n - 1` |
| 非空条件 | `left <= right` |
| 排除 `mid` | `left = mid + 1` 或 `right = mid - 1` |
| 空区间 | `left > right` |

它很适合“找到任意一个等于 `target` 的位置”：命中时直接返回，区间为空时返回不存在。

## 左闭右开区间 `[left, right)`

对于左边界查找，`[left, right)` 表示尚未分类的实际数组下标，而不是所有可能返回的位置。循环始终维护三个事实：

- 所有 `i < left` 的实际元素都满足 `nums[i] < target`；
- 所有数组内 `i >= right` 的实际元素都满足 `nums[i] >= target`；
- 第一个满足条件的位置位于整数位置范围 `[left, right]`，其中 `right` 可以等于 `n`。

初始化为：

```python
left = 0
right = len(nums)
```

`right` 可以等于数组长度，因为代码不会读取 `nums[right]`。初始化时还没有实际元素被分类，三个事实自然成立。只要 `left < right`，`[left, right)` 中就还有实际元素需要检查：

```python
while left < right:
```

例如寻找第一个满足 `nums[i] >= target` 的位置时：

```python
if nums[mid] < target:
    left = mid + 1
else:
    right = mid
```

当 `nums[mid] >= target` 时，`mid` 仍可能是第一个符合条件的位置，所以不能用 `right = mid - 1` 把它排除。两种更新都会扩大已经分类的区域，同时保留上述不变式。循环结束时 `left == right`，尚未分类的实际元素区间为空，答案范围也收缩到这一个位置；如果所有值都小于目标，它就是 `len(nums)`。

| 项目 | 规则 |
| --- | --- |
| 尚未分类的实际元素 | `[left, right)` |
| 可能的答案位置 | `[left, right]` |
| 初始化 | `left = 0`, `right = n` |
| 非空条件 | `left < right` |
| 更新 | `left = mid + 1` 或 `right = mid` |
| 结束状态 | `left == right == 答案位置` |

## 开区间 `(left, right)` 与哨兵

还可以让两个端点都表示已经确定的状态，只搜索它们之间尚未确定的位置。以寻找第一个 `nums[i] >= target` 的位置为例：

```python
left = -1
right = len(nums)
```

这里不访问 `nums[-1]` 或 `nums[len(nums)]`。`-1` 和 `n` 是扩展边界上的状态哨兵，不是实际数组元素。循环始终维护：

- 所有 `0 <= i <= left` 的实际元素都满足 `nums[i] < target`；
- 所有 `right <= i < n` 的实际元素都满足 `nums[i] >= target`；
- `(left, right)` 中的实际下标尚未分类。

初始化时 `left = -1` 和 `right = n`，两个已经分类的实际元素范围都是空集，所以不变式成立。可以把左哨兵理解为谓词为假的边界，把右哨兵理解为谓词为真的边界，但代码永远不会读取这两个哨兵位置。

只有当二者之间至少还有一个实际下标时才继续：

```python
while left + 1 < right:
```

更新时，已经检查过的 `mid` 加入对应的已分类区域，并成为新的状态哨兵：

```python
if nums[mid] < target:
    left = mid
else:
    right = mid
```

两种更新都保持量化后的不变式。循环结束时 `left + 1 == right`，中间已经没有未分类的实际下标；左侧都小于目标，右侧从 `right` 开始都大于等于目标，因此 `right` 就是第一个大于等于目标的位置。

| 项目 | 规则 |
| --- | --- |
| 未确定位置 | `(left, right)` |
| 初始化 | `left = -1`, `right = n` |
| 非空条件 | `left + 1 < right` |
| 更新 | `left = mid` 或 `right = mid` |
| 结束状态 | `left + 1 == right` |

## 为什么不能混用规则

假设要返回第一个大于等于 `target` 的位置，却把闭区间初始化和左闭右开更新拼在一起：

```python
def broken_first_not_less(nums: list[int], target: int) -> int:
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

对 `nums = [1, 3, 5, 7]`、`target = 8`，正确插入位置是 `4`。但这段代码把 `right` 初始化为 `3`，答案位置范围从一开始就没有包含位置 `4`，最后只能错误地返回 `3`。

问题不在某一行语法，而在循环不变式已经互相矛盾：更新规则允许答案等于 `n`，初始化却提前排除了这个候选位置。

## 三种写法的可运行对照

下面的代码分别实现闭区间精确查找，以及两种区间表示下的左边界查找：

```python
def search_closed(nums: list[int], target: int) -> int:
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def first_not_less_half_open(nums: list[int], target: int) -> int:
    left = 0
    right = len(nums)

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def first_not_less_open(nums: list[int], target: int) -> int:
    left = -1
    right = len(nums)

    while left + 1 < right:
        mid = (left + right) // 2

        if nums[mid] < target:
            left = mid
        else:
            right = mid

    return right


nums = [1, 3, 5, 7]
nums_with_duplicates = [1, 1, 3, 3]

assert search_closed(nums, 5) == 2
assert search_closed(nums, 4) == -1
assert search_closed([], 1) == -1

assert first_not_less_half_open(nums, 4) == 2
assert first_not_less_half_open(nums, 8) == 4
assert first_not_less_half_open([], 1) == 0

assert first_not_less_open(nums, 4) == 2
assert first_not_less_open(nums, 8) == 4
assert first_not_less_open([], 1) == 0

assert first_not_less_half_open(nums_with_duplicates, 1) == 0
assert first_not_less_half_open(nums_with_duplicates, 2) == 2
assert first_not_less_open(nums_with_duplicates, 1) == 0
assert first_not_less_open(nums_with_duplicates, 2) == 2

for target in range(-1, 10):
    assert first_not_less_half_open(nums, target) == first_not_less_open(
        nums, target
    )
```

三种写法每轮都会严格缩小未确定范围，因此时间复杂度都是 `O(log n)`，额外空间复杂度都是 `O(1)`。区别不在性能，而在候选集合和最终返回语义。

## 写之前先回答四个问题

不要先决定使用哪段模板。先写下这四个答案：

1. `left` 指向的位置仍可能是答案吗？
2. `right` 指向的位置仍可能是答案吗？
3. 检查 `mid` 后，它应该保留还是排除？
4. 循环结束时，要返回命中的元素、插入位置，还是某一侧边界？

只要这四个答案明确，区间符号就不再是需要死记的标签：初始化建立不变式，循环条件判断是否还有实际元素需要检查，边界更新扩大已排除或已分类的范围，退出状态再给出最终答案。
