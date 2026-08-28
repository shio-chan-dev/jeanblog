---
title: "LeetCode 74: Search a 2D Matrix"
date: 2026-08-13
draft: false
categories:
  - LeetCode
tags:
  - Binary Search
  - Matrix
  - Virtual Index Mapping
  - Sorted Array
  - LeetCode 74
---

You are given an `m x n` integer matrix `matrix` and an integer `target`. Return `True` if `target` is in the matrix; otherwise, return `False`.

The matrix satisfies two conditions:

- Every row is sorted in non-decreasing order.
- The first integer of each row is strictly greater than the last integer of the previous row.

The required time complexity is `O(log(m * n))`.

For example, consider this matrix:

```text
1   3   5   7
10  11  16  20
23  30  34  60
```

- When `target = 3`, return `True`.
- When `target = 13`, return `False`.

The constraints are:

- `1 <= m, n <= 100`
- `-10^4 <= matrix[i][j], target <= 10^4`

LeetCode provides this method signature:

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        pass
```

## Step 1: Scan Every Element

At this point, we only have the problem constraints and the method signature. We do not yet have code that can decide whether the target exists. Start with the simplest question: how can we produce a directly runnable version whose result is clearly correct?

The current baseline cannot answer either example. Add one loop over the rows and another over the values in each row. Return `True` as soon as a value equals the target. Return `False` only after every element has been checked without a match.

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

# Official examples: a normal hit and a missing target.
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False

# A single element: hit and miss.
assert solution.searchMatrix([[1]], 1) is True
assert solution.searchMatrix([[1]], 0) is False

# The first and last elements.
assert solution.searchMatrix(matrix, 1) is True
assert solution.searchMatrix(matrix, 60) is True

# A single row.
assert solution.searchMatrix([[1, 3, 5, 7]], 5) is True
assert solution.searchMatrix([[1, 3, 5, 7]], 6) is False

# A single column.
assert solution.searchMatrix([[1], [3], [5]], 3) is True
assert solution.searchMatrix([[1], [3], [5]], 4) is False
```

This version checks at most `m * n` elements, so its time complexity is `O(mn)`. The loops store only the current row and value, so the extra space complexity is `O(1)`.

At this checkpoint, we can correctly determine whether the target exists by scanning every element. However, the worst case still examines the entire matrix, so it does not yet satisfy the required `O(log(m * n))` time complexity.

## Step 2: Join All Rows into One Sorted Array

The element-by-element scan is correct, but it does not use either ordering condition. Consider a smaller matrix:

```text
1  3  5
7  9  11
```

Within each row, the elements are sorted in non-decreasing order. Across the row boundary, the first element of the second row, `7`, is strictly greater than the last element of the first row, `5`. Therefore, reading every row from left to right and from top to bottom produces:

```text
[1, 3, 5, 7, 9, 11]
```

This array is still sorted. The same relationship holds between every pair of adjacent rows: the within-row order prevents inversions inside a row, while the strict boundary condition prevents an inversion when moving to the next row. Therefore, the entire matrix is globally sorted when read in row-major order.

The nested loops in the previous version hide this continuous order behind two-dimensional row boundaries. Modify that version by first writing the elements into a real `flat` array in row-major order, then scanning `flat` linearly:

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        flat = [value for row in matrix for value in row]

        for value in flat:
            if value == target:
                return True

        return False


# Flattening this small matrix must preserve the exact row-major order.
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

# The flattened linear scan preserves the same hit and miss behavior.
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False
```

Building `flat` copies `m * n` elements, and the subsequent scan may still inspect all `m * n` values. Therefore, the complete method has `O(mn)` time complexity and `O(mn)` extra space complexity.

At this checkpoint, we have made the globally sorted row-major sequence explicit and used it for searching. However, the search still checks values one at a time, and `flat` stores every element in the matrix.

## Step 3: Binary Search the Real One-Dimensional Array

The current baseline has a globally sorted `flat` array, but it still compares values from beginning to end. If the target is missing, this linear search may inspect all `m * n` values.

The concrete problem is now clear: the array is sorted, but no rule uses that order to discard half of the candidate indices. Replace only the search over `flat`. Let `left` and `right` represent the closed interval `[left, right]` that may still contain the target, and inspect the middle value at `mid`.

- If `flat[mid] == target`, return `True`.
- If `flat[mid] < target`, neither `mid` nor any index to its left can contain the target, so set `left = mid + 1`.
- If `flat[mid] > target`, neither `mid` nor any index to its right can contain the target, so set `right = mid - 1`.

Both updates remove the inspected `mid`, so the closed interval strictly shrinks on every iteration. When `left > right`, the interval is empty and the search returns `False`.

Searching for the missing value `13` in the example matrix proceeds as follows:

| Round | `left` | `right` | `mid` | `flat[mid]` | Update |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 11 | 5 | 11 | `11 < 13`, set `left = 6` |
| 2 | 6 | 11 | 8 | 23 | `23 > 13`, set `right = 7` |
| 3 | 6 | 7 | 6 | 16 | `16 > 13`, set `right = 5` |

After the third round, `left = 6` and `right = 5`. The candidate interval is empty, so `13` cannot be in the array.

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


# The real flattened array still matches the row-major sequence.
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

# Official examples: a normal hit and a missing target.
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False

# A single element: hit and miss.
assert solution.searchMatrix([[1]], 1) is True
assert solution.searchMatrix([[1]], 0) is False

# The first and last elements.
assert solution.searchMatrix(matrix, 1) is True
assert solution.searchMatrix(matrix, 60) is True

# A single row.
assert solution.searchMatrix([[1, 3, 5, 7]], 5) is True
assert solution.searchMatrix([[1, 3, 5, 7]], 6) is False

# A single column.
assert solution.searchMatrix([[1], [3], [5]], 3) is True
assert solution.searchMatrix([[1], [3], [5]], 4) is False

# Targets below the minimum and above the maximum.
assert solution.searchMatrix(matrix, 0) is False
assert solution.searchMatrix(matrix, 61) is False
```

The binary-search phase discards at least half of the remaining indices each round, so it takes `O(log(mn))` time. However, the complete method still has to copy `m * n` elements into `flat`. Its total time complexity therefore remains `O(mn)`, and its extra space complexity is also `O(mn)`.

At this checkpoint, we can use standard closed-interval binary search on a real sorted array. However, constructing `flat` still requires copying and storing every element in the matrix.

## Step 4: Access `flat[mid]` Without Creating `flat`

The current baseline already has a complete and correct binary search. Looking at its loop, each iteration reads only one value: `flat[mid]`. Yet the method first copies all `m * n` elements just to access those individual values.

That copy is exactly why the complete method still needs `O(mn)` time and extra space. We do not need another search algorithm. We need a direct way to answer this question: given an index `mid` in the row-major sequence, where is the same element in the original matrix?

Suppose each row contains `cols` elements. Every complete group of `cols` elements before `mid` accounts for one entire row. Therefore:

```text
row = mid // cols
col = mid % cols
```

Integer division counts how many complete rows come before the element, and the remainder gives its column inside the current row. For the three-row, four-column example matrix, the boundary indices map as follows:

| Virtual index | `row = mid // cols` | `col = mid % cols` | Matrix position |
| --- | ---: | ---: | --- |
| `0` | 0 | 0 | first element of the first row |
| `cols - 1 = 3` | 0 | 3 | last element of the first row |
| `cols = 4` | 1 | 0 | first element of the second row |
| `rows * cols - 1 = 11` | 2 | 3 | last element of the last row |

There is no need to create `flat`. Starting from the previous version, replace the right boundary `len(flat) - 1` with `rows * cols - 1`, and replace `flat[mid]` with `matrix[row][col]`. The rest of the binary search remains unchanged:

```python
import random
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])

        left = 0
        right = rows * cols - 1

        while left <= right:
            mid = (left + right) // 2
            row = mid // cols
            col = mid % cols
            value = matrix[row][col]

            if value == target:
                return True
            if value < target:
                left = mid + 1
            else:
                right = mid - 1

        return False


solution = Solution()
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60],
]

# Official examples: a normal hit and a missing target.
assert solution.searchMatrix(matrix, 3) is True
assert solution.searchMatrix(matrix, 13) is False

# A single element: hit and miss.
assert solution.searchMatrix([[1]], 1) is True
assert solution.searchMatrix([[1]], 0) is False

# The first and last elements, plus targets outside the value range.
assert solution.searchMatrix(matrix, 1) is True
assert solution.searchMatrix(matrix, 60) is True
assert solution.searchMatrix(matrix, 0) is False
assert solution.searchMatrix(matrix, 61) is False

# A single row or column.
assert solution.searchMatrix([[1, 3, 5, 7]], 5) is True
assert solution.searchMatrix([[1, 3, 5, 7]], 6) is False
assert solution.searchMatrix([[1], [3], [5]], 3) is True
assert solution.searchMatrix([[1], [3], [5]], 4) is False

# Rows may contain duplicates, but the next row must still begin above
# the previous row's final value.
matrix_with_duplicates = [
    [1, 1, 3],
    [5, 5, 8],
]
assert solution.searchMatrix(matrix_with_duplicates, 1) is True
assert solution.searchMatrix(matrix_with_duplicates, 5) is True
assert solution.searchMatrix(matrix_with_duplicates, 4) is False

# Searching must not modify the input matrix.
snapshot = [row[:] for row in matrix]
solution.searchMatrix(matrix, 16)
assert matrix == snapshot

# Generate 1,200 valid matrices with a fixed seed and compare the result
# with an element-by-element scan.
rng = random.Random(74)

for _ in range(1_200):
    test_rows = rng.randint(1, 8)
    test_cols = rng.randint(1, 8)
    next_value = rng.randint(-100, 100)
    test_matrix = []

    for _ in range(test_rows):
        test_row = [next_value]
        for _ in range(1, test_cols):
            test_row.append(test_row[-1] + rng.randint(0, 3))
        test_matrix.append(test_row)
        next_value = test_row[-1] + rng.randint(1, 3)

    test_target = rng.randint(test_matrix[0][0] - 2, test_matrix[-1][-1] + 2)
    expected = any(
        value == test_target
        for test_row in test_matrix
        for value in test_row
    )
    actual = solution.searchMatrix(test_matrix, test_target)
    assert actual is expected
```

### Why the Mapped Binary Search Is Still Correct

Step 2 established the key fact: reading the matrix in row-major order produces a globally sorted sequence. We no longer store that sequence, but each virtual index still maps to exactly one matrix coordinate. For every `0 <= mid < rows * cols`:

- `mid // cols` is between `0` and `rows - 1`.
- `mid % cols` is between `0` and `cols - 1`.
- Converting the coordinate back to a one-dimensional index gives `(mid // cols) * cols + mid % cols == mid`.

Therefore, `matrix[mid // cols][mid % cols]` reads exactly the element that `flat[mid]` would contain, without going out of bounds.

The binary-search loop maintains the same invariant: if the target exists, its virtual index is inside the closed interval `[left, right]`. When the middle value is smaller than the target, global order lets us discard `mid` and every index to its left. When the middle value is greater, we can discard `mid` and every index to its right. Each iteration removes `mid` and strictly shrinks the interval. Equality returns `True`; an empty interval means every candidate index has been eliminated, so the method returns `False`.

### Complexity

The virtual sequence contains `m * n` indices. Binary search reduces the candidate range by at least half on every iteration, so the time complexity is `O(log(mn))`. The algorithm stores only the matrix dimensions, three indices, and the current value. It never creates `flat`, so the extra space complexity is `O(1)`.

At this final checkpoint, we have transferred standard binary search from a real one-dimensional array directly to the matrix. Quotient and remainder map each virtual index to its matrix element, so the solution avoids copying the input while meeting the required `O(log(mn))` time and `O(1)` extra space bounds.
