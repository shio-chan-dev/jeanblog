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
