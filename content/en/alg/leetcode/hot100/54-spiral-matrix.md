---
title: "Hot100: Spiral Matrix (Boundary Shrinking Simulation ACERS Guide)"
date: 2026-02-03T10:01:50+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "matrix", "simulation", "boundary shrinking", "array", "LeetCode 54", "ACERS"]
description: "Traverse a matrix in clockwise spiral order in O(mn) using boundary shrinking. Includes engineering scenarios, pitfalls, and multi-language implementations."
keywords: ["Spiral Matrix", "clockwise spiral traversal", "boundary shrinking", "O(mn)", "LeetCode 54", "Hot100"]
---

> **Subtitle / Summary**  
> Spiral traversal looks like “just printing in a fancy order”, but the real difficulty is getting boundaries and invariants right. This ACERS guide gives a reusable boundary-shrinking template and runnable multi-language solutions.

- **Reading time**: 12–15 min  
- **Tags**: `Hot100`, `matrix`, `simulation`, `boundary shrinking`  
- **SEO keywords**: Hot100, Spiral Matrix, clockwise traversal, boundary shrinking, LeetCode 54  
- **Meta description**: O(mn) spiral order traversal using boundary shrinking, with pitfalls, engineering scenarios, and runnable code.  

---

## Target Readers

- Hot100 learners who want a reliable “matrix simulation” template  
- Intermediate engineers who often get boundary cases wrong  
- Anyone working with grids (visualization, raster data, path generation)

## Background / Motivation

Matrix problems are notorious for being “easy to code, hard to get 100% correct”.  
One extra loop or one missed boundary check can break single-row/single-column cases or cause duplicated output.

Spiral Matrix is a great training problem because it forces you to make the **loop invariant** explicit:

- What region is still unvisited?
- How do we shrink it safely after finishing an edge?

If you can express that invariant clearly, the code becomes short and robust.

## Core Concepts

- **Boundaries**: `top / bottom / left / right` define the current unvisited rectangle  
- **Layer**: each iteration peels one outer “ring” (top row, right col, bottom row, left col)  
- **Shrink**: after finishing an edge, move the boundary inward (`top++`, `right--`, `bottom--`, `left++`)  
- **Loop invariant**: the unvisited region is always `top..bottom × left..right`

---

## A — Algorithm

### Problem Restatement

Given an `m × n` matrix `matrix`, return all elements in **clockwise spiral order**.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| matrix | int[][] | an `m × n` matrix |
| return | int[] | elements in clockwise spiral order |

### Example 1

```text
matrix =
[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

### Example 2

```text
matrix =
[
  [ 1,  2,  3,  4],
  [ 5,  6,  7,  8],
  [ 9, 10, 11, 12]
]
output: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

---

## C — Concepts

### Thought Process: from `visited` to boundary shrinking

1) **Naive**: direction array + `visited`  
   Start at `(0,0)`, move right/down/left/up, rotate when hitting a boundary or visited cell.  
   - Pros: intuitive  
   - Cons: needs `m×n` extra space; more branching and easier to get wrong

2) **Key observation**: spiral traversal is “peel the onion”  
   Each layer is exactly four edges: top row, right column, bottom row, left column.

3) **Chosen method**: boundary shrinking (O(1) extra space)  
   Maintain `top, bottom, left, right`. Traverse edges, then shrink the boundary.

### Method Type

- Matrix simulation  
- Boundary shrinking  
- Loop invariant + boundary conditions

### The key invariant (the reason it’s correct)

At the start of each loop, the unvisited region is a rectangle:

```text
rows: top .. bottom
cols: left .. right
```

After finishing an edge, we shrink the corresponding boundary by 1:

- top row done → `top += 1`  
- right col done → `right -= 1`  
- bottom row done → `bottom -= 1` (only if `top <= bottom`)  
- left col done → `left += 1` (only if `left <= right`)

Those two conditional checks are what prevent duplicates when only one row/column remains.

---

## Practical Steps

1. Handle empty matrix: return `[]`  
2. Initialize boundaries: `top=0, bottom=m-1, left=0, right=n-1`  
3. While `top <= bottom` and `left <= right`:
   - Traverse top row (left → right), `top++`
   - Traverse right col (top → bottom), `right--`
   - If `top <= bottom`, traverse bottom row (right → left), `bottom--`
   - If `left <= right`, traverse left col (bottom → top), `left++`
4. Return the result

Runnable Python example (save as `spiral_matrix.py`):

```python
from typing import List


def spiral_order(matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])
    top, bottom, left, right = 0, m - 1, 0, n - 1
    res: List[int] = []

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            res.append(matrix[top][j])
        top += 1

        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(matrix[bottom][j])
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1

    return res


if __name__ == "__main__":
    print(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
```

---

## E — Engineering

The “engineering value” of this problem is that it’s a reusable **grid path generator**.  
You can replace “append value” with “emit coordinates”, and map coordinates to any domain object (pixels, tiles, warehouse bins, table cells).

### Scenario 1: spiral feature extraction for image/raster patches (Python)

**Background**: flatten a small patch (e.g., 7×7, 11×11) into a 1D feature vector.  
**Why it fits**: spiral order preserves “outer-to-inner” structure in the sequence.

```python
from typing import List


def spiral_order(matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    top, bottom, left, right = 0, m - 1, 0, n - 1
    res: List[int] = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            res.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1
    return res


print(spiral_order([[0, 0, 1], [0, 1, 1], [1, 1, 1]]))
```

### Scenario 2: backend service streaming a grid “layer by layer” (Go)

**Background**: maps/tiles/seat grids often want progressive loading: outer ring first for faster “first screen”.  
**Why it fits**: boundary shrinking naturally yields a layer-by-layer traversal.

```go
package main

import "fmt"

func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }

    m, n := len(matrix), len(matrix[0])
    top, bottom, left, right := 0, m-1, 0, n-1
    res := make([]int, 0, m*n)

    for top <= bottom && left <= right {
        for j := left; j <= right; j++ {
            res = append(res, matrix[top][j])
        }
        top++

        for i := top; i <= bottom; i++ {
            res = append(res, matrix[i][right])
        }
        right--

        if top <= bottom {
            for j := right; j >= left; j-- {
                res = append(res, matrix[bottom][j])
            }
            bottom--
        }

        if left <= right {
            for i := bottom; i >= top; i-- {
                res = append(res, matrix[i][left])
            }
            left++
        }
    }

    return res
}

func main() {
    grid := [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
    fmt.Println(spiralOrder(grid))
}
```

### Scenario 3: spiral scan path for robots / automation (C)

**Background**: coverage scanning in a discrete grid (inspection, cleaning, sampling).  
**Why it fits**: O(1) state; no `visited` buffer needed (good for embedded).

```c
#include <stdio.h>

static void spiral_path(int m, int n) {
    int top = 0, bottom = m - 1, left = 0, right = n - 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) printf("(%d,%d) ", top, j);
        ++top;

        for (int i = top; i <= bottom; ++i) printf("(%d,%d) ", i, right);
        --right;

        if (top <= bottom) {
            for (int j = right; j >= left; --j) printf("(%d,%d) ", bottom, j);
            --bottom;
        }

        if (left <= right) {
            for (int i = bottom; i >= top; --i) printf("(%d,%d) ", i, left);
            ++left;
        }
    }
    printf("\\n");
}

int main(void) {
    spiral_path(3, 4);
    return 0;
}
```

### Scenario 4: spiral highlight animation in a grid UI (JavaScript)

**Background**: highlight cells in a spiral order for tutorials/animations on a grid.  
**Why it fits**: you only need the traversal order as a frame sequence.

```javascript
function spiralOrder(matrix) {
  if (!matrix.length || !matrix[0].length) return [];
  let top = 0, bottom = matrix.length - 1;
  let left = 0, right = matrix[0].length - 1;
  const res = [];

  while (top <= bottom && left <= right) {
    for (let j = left; j <= right; j++) res.push(matrix[top][j]);
    top++;

    for (let i = top; i <= bottom; i++) res.push(matrix[i][right]);
    right--;

    if (top <= bottom) {
      for (let j = right; j >= left; j--) res.push(matrix[bottom][j]);
      bottom--;
    }

    if (left <= right) {
      for (let i = bottom; i >= top; i--) res.push(matrix[i][left]);
      left++;
    }
  }

  return res;
}

console.log(spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
```

---

## R — Reflection

### Complexity

- **Time**: O(mn) (each element is output exactly once)  
- **Space**: O(1) extra (excluding the output array); `visited` approach uses O(mn)

### Alternatives

| Method | Idea | Extra Space | Typical issues |
| --- | --- | --- | --- |
| `visited` + direction turns | “walk and turn” | O(mn) | more conditionals, easy to get off-by-one |
| recursion / per-layer slicing | “peel layers” | depends | may introduce slicing copies or recursion overhead |
| **boundary shrinking (this post)** | traverse 4 edges + shrink | **O(1)** | must handle single-row/single-col carefully |

### Why boundary shrinking is engineering-friendly

- Minimal state: four integers describe the progress  
- No extra matrix: good for memory-constrained environments  
- Easy to batch: each layer is naturally a “batch” for progressive output

---

## Explanation / Why it works

Think of the unvisited part as a shrinking rectangle:

1) top edge: output row `top`, then `top++`  
2) right edge: output col `right`, then `right--`  
3) bottom edge: only if `top <= bottom`, output row `bottom`, then `bottom--`  
4) left edge: only if `left <= right`, output col `left`, then `left++`

Those two checks are critical: when only one row or one column is left, you must skip the opposite edges to avoid duplicates.

---

## Common Pitfalls and Notes

1) **Why do we need `if top <= bottom` and `if left <= right`?**  
   They handle the “single remaining row/column” cases and prevent duplicates.

2) **Should we handle empty inputs?**  
   LeetCode usually guarantees `m,n >= 1`, but production code should return `[]` for empty input.

3) **What about jagged arrays (rows with different lengths)?**  
   The problem assumes a rectangular matrix. Validate/normalize in real systems.

4) **How to output coordinates instead of values?**  
   Replace `res.append(matrix[i][j])` with emitting `(i, j)` (or pushing into a queue/channel).

---

## Best Practices

- Write the invariant first: unvisited region is `top..bottom × left..right`  
- Shrink boundaries immediately after finishing an edge  
- Keep the two safety checks for bottom/left edges  
- For streaming output, replace “append” with “yield/send”

---

## S — Summary

### Key Takeaways

- Spiral traversal is “peel layers”, not “random turns”  
- `top/bottom/left/right` boundaries give O(1) extra space  
- The two checks (`top<=bottom`, `left<=right`) are the correctness key  
- The same template works as a generic grid path generator  

### Conclusion

The goal is not “pass the sample”, but “never break on single-row/single-col edge cases”.  
Once you master boundary shrinking, many matrix simulation problems become straightforward.

### References & Further Reading

- LeetCode 54. Spiral Matrix
- LeetCode 59. Spiral Matrix II (generate spiral matrix)
- LeetCode 885. Spiral Matrix III (expanding spiral path)

---

## Meta

- **Reading time**: 12–15 min  
- **Tags**: Hot100, matrix, simulation, boundary shrinking, LeetCode 54  
- **SEO keywords**: Hot100, Spiral Matrix, boundary shrinking, O(mn), LeetCode 54  
- **Meta description**: O(mn) spiral traversal using boundary shrinking, with pitfalls and multi-language code.  

---

## Call to Action

Turn the boundary-shrinking logic into your personal “matrix simulation template”.  
Next time you see a spiral/layered traversal, copy the template and only swap the “output action”.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import List


def spiral_order(matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])
    top, bottom, left, right = 0, m - 1, 0, n - 1
    res: List[int] = []

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            res.append(matrix[top][j])
        top += 1

        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(matrix[bottom][j])
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1

    return res


if __name__ == "__main__":
    print(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
```

```c
#include <stdio.h>
#include <stdlib.h>

int* spiral_order(int** matrix, int m, int n, int* returnSize) {
    if (m <= 0 || n <= 0) {
        *returnSize = 0;
        return NULL;
    }

    int total = m * n;
    int* res = (int*)malloc((size_t)total * sizeof(int));
    int idx = 0;

    int top = 0, bottom = m - 1, left = 0, right = n - 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) res[idx++] = matrix[top][j];
        ++top;

        for (int i = top; i <= bottom; ++i) res[idx++] = matrix[i][right];
        --right;

        if (top <= bottom) {
            for (int j = right; j >= left; --j) res[idx++] = matrix[bottom][j];
            --bottom;
        }

        if (left <= right) {
            for (int i = bottom; i >= top; --i) res[idx++] = matrix[i][left];
            ++left;
        }
    }

    *returnSize = idx;
    return res;
}

int main(void) {
    int a0[] = {1, 2, 3, 4};
    int a1[] = {5, 6, 7, 8};
    int a2[] = {9, 10, 11, 12};
    int* matrix[] = {a0, a1, a2};

    int returnSize = 0;
    int* res = spiral_order(matrix, 3, 4, &returnSize);
    for (int i = 0; i < returnSize; ++i) {
        if (i) printf(", ");
        printf("%d", res[i]);
    }
    printf("\\n");
    free(res);
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

std::vector<int> spiralOrder(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return {};

    int m = (int)matrix.size();
    int n = (int)matrix[0].size();
    int top = 0, bottom = m - 1, left = 0, right = n - 1;

    std::vector<int> res;
    res.reserve((size_t)m * (size_t)n);

    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) res.push_back(matrix[top][j]);
        ++top;

        for (int i = top; i <= bottom; ++i) res.push_back(matrix[i][right]);
        --right;

        if (top <= bottom) {
            for (int j = right; j >= left; --j) res.push_back(matrix[bottom][j]);
            --bottom;
        }

        if (left <= right) {
            for (int i = bottom; i >= top; --i) res.push_back(matrix[i][left]);
            ++left;
        }
    }

    return res;
}

int main() {
    std::vector<std::vector<int>> m = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    auto res = spiralOrder(m);
    for (size_t i = 0; i < res.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << res[i];
    }
    std::cout << \"\\n\";
    return 0;
}
```

```go
package main

import \"fmt\"

func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }

    m, n := len(matrix), len(matrix[0])
    top, bottom, left, right := 0, m-1, 0, n-1
    res := make([]int, 0, m*n)

    for top <= bottom && left <= right {
        for j := left; j <= right; j++ {
            res = append(res, matrix[top][j])
        }
        top++

        for i := top; i <= bottom; i++ {
            res = append(res, matrix[i][right])
        }
        right--

        if top <= bottom {
            for j := right; j >= left; j-- {
                res = append(res, matrix[bottom][j])
            }
            bottom--
        }

        if left <= right {
            for i := bottom; i >= top; i-- {
                res = append(res, matrix[i][left])
            }
            left++
        }
    }

    return res
}

func main() {
    grid := [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
    fmt.Println(spiralOrder(grid))
}
```

```rust
fn spiral_order(matrix: &[Vec<i32>]) -> Vec<i32> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![];
    }

    let m = matrix.len() as i32;
    let n = matrix[0].len() as i32;
    let (mut top, mut bottom, mut left, mut right) = (0i32, m - 1, 0i32, n - 1);
    let mut res: Vec<i32> = Vec::with_capacity((m * n) as usize);

    while top <= bottom && left <= right {
        for j in left..=right {
            res.push(matrix[top as usize][j as usize]);
        }
        top += 1;

        for i in top..=bottom {
            res.push(matrix[i as usize][right as usize]);
        }
        right -= 1;

        if top <= bottom {
            for j in (left..=right).rev() {
                res.push(matrix[bottom as usize][j as usize]);
            }
            bottom -= 1;
        }

        if left <= right {
            for i in (top..=bottom).rev() {
                res.push(matrix[i as usize][left as usize]);
            }
            left += 1;
        }
    }

    res
}

fn main() {
    let matrix = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
    println!(\"{:?}\", spiral_order(&matrix));
}
```

```javascript
function spiralOrder(matrix) {
  if (!matrix.length || !matrix[0].length) return [];
  let top = 0, bottom = matrix.length - 1;
  let left = 0, right = matrix[0].length - 1;
  const res = [];

  while (top <= bottom && left <= right) {
    for (let j = left; j <= right; j++) res.push(matrix[top][j]);
    top++;

    for (let i = top; i <= bottom; i++) res.push(matrix[i][right]);
    right--;

    if (top <= bottom) {
      for (let j = right; j >= left; j--) res.push(matrix[bottom][j]);
      bottom--;
    }

    if (left <= right) {
      for (let i = bottom; i >= top; i--) res.push(matrix[i][left]);
      left++;
    }
  }

  return res;
}

console.log(spiralOrder([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]));
```
