---
title: "LeetCode 74：搜索二维矩阵"
date: 2026-08-13
draft: true
categories:
  - LeetCode
tags:
  - 二分查找
  - 矩阵
  - 一维下标映射
  - 有序数组
  - LeetCode 74
---

给你一个 `m x n` 的整数矩阵 `matrix` 和一个整数 `target`。如果 `target` 在矩阵中，返回 `True`；否则返回 `False`。

题目给出的矩阵满足两个条件：

- 每一行都按非递减顺序排列。
- 每一行的第一个整数都严格大于上一行的最后一个整数。

要求算法的时间复杂度为 `O(log(m * n))`。

例如，对于下面的矩阵：

```text
1   3   5   7
10  11  16  20
23  30  34  60
```

- `target = 3` 时返回 `True`。
- `target = 13` 时返回 `False`。

约束如下：

- `1 <= m, n <= 100`
- `-10^4 <= matrix[i][j], target <= 10^4`

LeetCode 提供的方法签名是：

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        pass
```

## 第一步：扫描每一个元素

现在只有题目条件和方法签名，还没有一段能判断目标值是否存在的代码。先解决最基本的问题：怎样得到一个可以直接运行、结果确定正确的版本？

当前基线无法对示例中的 `3` 或 `13` 给出答案。加入一层遍历每一行的循环，再在当前行中检查每一个值。遇到目标值就立即返回 `True`；只有检查完所有元素仍未命中时，才返回 `False`。

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            for value in row:
                if value == target:
                    return True

        return False


solution = Solution()
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60],
]

# 官方示例：普通命中与缺失。
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False

# 单个元素：命中与缺失。
assert solution.searchMatrix([[1]], 1) is True
assert solution.searchMatrix([[1]], 0) is False

# 第一个元素与最后一个元素。
assert solution.searchMatrix(matrix, 1) is True
assert solution.searchMatrix(matrix, 60) is True

# 只有一行。
assert solution.searchMatrix([[1, 3, 5, 7]], 5) is True
assert solution.searchMatrix([[1, 3, 5, 7]], 6) is False

# 只有一列。
assert solution.searchMatrix([[1], [3], [5]], 3) is True
assert solution.searchMatrix([[1], [3], [5]], 4) is False
```

这个版本最多检查 `m * n` 个元素，因此时间复杂度是 `O(mn)`。循环只保存当前行和当前值，额外空间复杂度是 `O(1)`。

到这个检查点，我们已经能通过逐个扫描正确判断目标值是否存在。不过，它在最慢情况下会检查所有元素，还没有满足题目要求的 `O(log(m * n))` 时间复杂度。

## 第二步：把所有行接成一个有序数组

上一步的逐个扫描是正确的，但它没有使用题目给出的两个排列条件。先看一个更小的矩阵：

```text
1  3  5
7  9  11
```

在每一行内部，元素按非递减顺序排列。跨过行边界时，第二行的第一个元素 `7` 又严格大于第一行的最后一个元素 `5`。因此，如果按从上到下、每行从左到右的顺序接起所有元素，会得到：

```text
[1, 3, 5, 7, 9, 11]
```

这个数组仍然有序。同样的关系存在于任意两行之间：行内顺序保证一行中的元素不会逆序，相邻两行首尾之间的严格大小关系保证跨过行边界时也不会逆序。所以，整个矩阵按行展开后是一个全局有序数组。

当前版本直接在两层循环中比较元素，二维的行边界让这个连续顺序没有显现出来。在上一个版本中，先把元素按行写入一个真实的 `flat` 数组，再线性检查 `flat`：

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        flat = [value for row in matrix for value in row]

        for value in flat:
            if value == target:
                return True

        return False


# 小矩阵按行展开后的结果必须完全一致。
tiny_matrix = [
    [1, 3, 5],
    [7, 9, 11],
]
tiny_flat = [value for row in tiny_matrix for value in row]
assert tiny_flat == [1, 3, 5, 7, 9, 11]

solution = Solution()
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60],
]

# 官方示例：展开后的线性检查保持相同的命中与缺失结果。
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False
```

构造 `flat` 会复制 `m * n` 个元素，随后最慢仍要检查其中的 `m * n` 个元素，因此完整方法的时间复杂度是 `O(mn)`，额外空间复杂度也是 `O(mn)`。

到这个检查点，我们已经能把矩阵实际展开为一个全局有序数组，并在搜索中使用它。不过，当前搜索仍然逐个检查元素，而且 `flat` 保存了矩阵中的所有元素。

## 第三步：在真实的一维数组上二分查找

当前基线已经得到全局有序的 `flat`，但搜索时仍然从头到尾逐个比较。如果目标不存在，这个线性搜索可能检查 `flat` 中的全部 `m * n` 个值。

这个版本的具体问题是：数组的有序性已经建立，却还没有任何规则借助它排除一半候选下标。在上一个版本中，只替换搜索 `flat` 的部分：用 `left` 和 `right` 表示当前仍可能包含目标的闭区间 `[left, right]`，再用 `mid` 检查该区间的中间值。

- 如果 `flat[mid] == target`，直接返回 `True`。
- 如果 `flat[mid] < target`，由于 `mid` 及其左边的值都不可能是目标，令 `left = mid + 1`。
- 如果 `flat[mid] > target`，由于 `mid` 及其右边的值都不可能是目标，令 `right = mid - 1`。

两种更新都会排除已检查的 `mid`，所以闭区间每轮都严格缩小。当 `left > right` 时，区间为空，搜索结束并返回 `False`。

对官方矩阵搜索不存在的 `13` 时，二分过程如下：

| 轮次 | `left` | `right` | `mid` | `flat[mid]` | 更新 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 11 | 5 | 11 | `11 < 13`，令 `left = 6` |
| 2 | 6 | 11 | 8 | 23 | `23 > 13`，令 `right = 7` |
| 3 | 6 | 7 | 6 | 16 | `16 > 13`，令 `right = 5` |

第三轮后 `left = 6` 而 `right = 5`，候选区间已经为空，因此可以确定 `13` 不在数组中。

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        flat = [value for row in matrix for value in row]

        left = 0
        right = len(flat) - 1

        while left <= right:
            mid = (left + right) // 2
            value = flat[mid]

            if value == target:
                return True
            if value < target:
                left = mid + 1
            else:
                right = mid - 1

        return False


# 真实展开的数组仍与按行读取的顺序一致。
tiny_matrix = [
    [1, 3, 5],
    [7, 9, 11],
]
tiny_flat = [value for row in tiny_matrix for value in row]
assert tiny_flat == [1, 3, 5, 7, 9, 11]

solution = Solution()
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60],
]

# 官方示例：普通命中与缺失。
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False

# 单个元素：命中与缺失。
assert solution.searchMatrix([[1]], 1) is True
assert solution.searchMatrix([[1]], 0) is False

# 第一个元素与最后一个元素。
assert solution.searchMatrix(matrix, 1) is True
assert solution.searchMatrix(matrix, 60) is True

# 只有一行。
assert solution.searchMatrix([[1, 3, 5, 7]], 5) is True
assert solution.searchMatrix([[1, 3, 5, 7]], 6) is False

# 只有一列。
assert solution.searchMatrix([[1], [3], [5]], 3) is True
assert solution.searchMatrix([[1], [3], [5]], 4) is False

# 目标在最小值之下或最大值之上。
assert solution.searchMatrix(matrix, 0) is False
assert solution.searchMatrix(matrix, 61) is False
```

二分搜索阶段每轮排除至少一半候选下标，时间复杂度是 `O(log(mn))`。但是，完整方法仍然要先复制 `m * n` 个元素来构造 `flat`，因此完整方法的时间复杂度仍是 `O(mn)`，额外空间复杂度也是 `O(mn)`。

到这个检查点，我们已经能在实际展开的有序数组上使用标准闭区间二分查找。不过，构造 `flat` 仍然需要复制和保存矩阵中的全部 `m * n` 个元素。
