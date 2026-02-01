---
title: "Hot100：搜索二维矩阵 II（Search a 2D Matrix II）右上角阶梯搜索 O(m+n) ACERS 解析"
date: 2026-02-01T13:56:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "矩阵", "单调性", "指针", "剪枝", "LeetCode 240"]
description: "行列均升序的矩阵查找 target：从右上角出发每步排除一行或一列，O(m+n) 时间、O(1) 空间；含推导、工程迁移与多语言实现。"
keywords: ["Search a 2D Matrix II", "搜索二维矩阵 II", "staircase search", "monotone matrix", "O(m+n)", "in-place"]
---

> **副标题 / 摘要**  
> 这题的关键不是二分，而是利用“行列都单调”的结构，从**右上角（或左下角）**像走楼梯一样移动：每一步都能排除一整行或一整列，从而把复杂度降到 O(m+n)。

- **预计阅读时长**：10~13 分钟  
- **标签**：`Hot100`、`矩阵`、`单调性`、`指针`  
- **SEO 关键词**：搜索二维矩阵 II, 单调矩阵查找, 右上角搜索, O(m+n), LeetCode 240  
- **元描述**：在行列均升序的矩阵中搜索 target：从右上角阶梯式移动，每步排除一行或一列，O(m+n)/O(1) 解法与多语言实现。  

---

## 目标读者

- 刷 Hot100，希望掌握“二维单调结构剪枝”模板的学习者  
- 写过二分但总在二维问题里迷路的中级开发者  
- 在工程中需要查询/裁剪/定位二维单调表格的工程师

## 背景 / 动机

二维表在工程里很常见：费率表、校准表、阈值表、网格配置表等。  
当一个表满足“横向递增 + 纵向递增”的 **二维单调（monotone matrix）** 特性时，很多查询不需要 O(mn) 全扫。  
这题就是经典入门：**用单调性做剪枝**，把搜索降成线性级别。

## 核心概念

| 概念 | 含义 | 在本题的作用 |
| --- | --- | --- |
| 二维单调矩阵 | 行升序、列升序 | 保证“比较一次就能排除一行/列” |
| 右上角起点 | 右上角元素：左边更小、下边更大 | 决策方向天然明确 |
| 剪枝 | 排除不可能包含 target 的行/列 | 每步减少搜索空间 |
| O(1) 额外空间 | 只用 i/j 指针 | 适合大矩阵与性能场景 |

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个 `m x n` 矩阵 `matrix` 和一个目标值 `target`。矩阵满足：

- 每行从左到右升序排列  
- 每列从上到下升序排列  

请判断 `target` 是否存在于矩阵中，返回 `true/false`。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| matrix | int[][] | `m x n` 矩阵（行列升序） |
| target | int | 目标值 |
| 返回 | bool | 是否存在 |

### 示例 1

```text
matrix =
[
  [1,  4,  7, 11, 15],
  [2,  5,  8, 12, 19],
  [3,  6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
target = 5
输出: true
```

### 示例 2

```text
matrix 同上
target = 20
输出: false
```

---

## 思路推导：从全扫到“走楼梯”

### 朴素解：全矩阵扫描

直接遍历所有元素，时间 O(mn)，一定能过，但没有利用单调性。

### 次优解：对每一行做二分

每行升序，所以可以对每行二分：时间 O(m log n)。  
这是合理的，但仍然没有把“列也升序”这条信息榨干。

### 关键观察：右上角是“决策最清晰”的位置

把指针放在右上角 `(0, n-1)`，记当前值 `x = matrix[i][j]`：

- 若 `x == target`：找到  
- 若 `x > target`：**这一列**里，当前位置下面的元素只会更大，不可能有 target  
  - 所以可以排除当前列，`j--`（向左移动）  
- 若 `x < target`：**这一行**里，当前位置左边的元素只会更小，不可能有 target  
  - 所以可以排除当前行，`i++`（向下移动）

每一步都至少排除一整行或一整列，最多走 `m+n` 步结束。

---

## C — Concepts（核心思想）

### 方法归类

- **单调结构搜索（Monotone Search）**  
- **指针剪枝（Pointer + Pruning）**  
- 也常被称为 **“阶梯搜索（Staircase Search）”**

### 正确性直觉（为什么能排除一整行/列）

右上角元素的邻域性质：

- 左侧都 <= 当前值（同一行升序）  
- 下侧都 >= 当前值（同一列升序）

因此当 `x > target` 时，当前列从 `i` 往下都 >= x > target，整列不可能包含 target；  
当 `x < target` 时，当前行从 `0` 到 `j` 都 <= x < target，整行不可能包含 target。  
这就是“比较一次 -> 排除一大片”的根源。

---

## 实践指南 / 步骤

1. 处理空矩阵边界  
2. 初始化 `i=0, j=n-1`（右上角）  
3. 循环直到越界：
   - 命中则返回 `true`
   - `> target` 则 `j--`
   - `< target` 则 `i++`
4. 越界仍未命中则返回 `false`

Python 可运行示例（保存为 `search_2d_matrix_ii.py`）：

```python
from typing import List


def search_matrix(matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    while i < m and j >= 0:
        x = matrix[i][j]
        if x == target:
            return True
        if x > target:
            j -= 1
        else:
            i += 1
    return False


if __name__ == "__main__":
    mat = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    print(search_matrix(mat, 5))   # True
    print(search_matrix(mat, 20))  # False
```

---

## E — Engineering（工程应用）

### 场景 1：费率/报价二维表的快速校验（Go）

**背景**：一些业务会维护“重量 x 距离 -> 费用”的费率表；重量越大费用越高、距离越远费用越高，表自然满足行列单调。  
**为什么适用**：你可能需要在上线前检查某个费用值是否被配置进表（例如排查重复、对账或验证缓存命中）。  

```go
package main

import "fmt"

func existsFee(matrix [][]int, target int) bool {
    m := len(matrix)
    if m == 0 || len(matrix[0]) == 0 {
        return false
    }
    n := len(matrix[0])
    i, j := 0, n-1
    for i < m && j >= 0 {
        x := matrix[i][j]
        if x == target {
            return true
        }
        if x > target {
            j--
        } else {
            i++
        }
    }
    return false
}

func main() {
    fees := [][]int{
        {10, 20, 30},
        {15, 25, 35},
        {18, 28, 40},
    }
    fmt.Println(existsFee(fees, 25)) // true
}
```

### 场景 2：嵌入式校准表查询（C）

**背景**：设备可能用一个 `n x n` 的校准表描述某参数在两个变量同时增大时的单调变化（例如温度/湿度对补偿值的影响）。  
**为什么适用**：C 环境往往不想分配额外内存；O(1) 指针移动更可控。  

```c
#include <stdio.h>

int search(int m, int n, int a[m][n], int target) {
    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
        int x = a[i][j];
        if (x == target) return 1;
        if (x > target) --j;
        else ++i;
    }
    return 0;
}

int main(void) {
    int a[3][4] = {
        {1, 4, 7, 10},
        {2, 5, 8, 12},
        {3, 6, 9, 15},
    };
    printf("%d\n", search(3, 4, a, 8));  // 1
    printf("%d\n", search(3, 4, a, 11)); // 0
    return 0;
}
```

### 场景 3：前端表格中的“单调矩阵快速定位”（JavaScript）

**背景**：某些 UI 会展示一个按行列维度递增的二维表（例如“档位 x 级别”的权益表）；需要快速判断某值是否存在来高亮/提示。  
**为什么适用**：线性移动比全表扫描更省时，且逻辑简单、易维护。  

```javascript
function existsInMonotoneMatrix(matrix, target) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  if (m === 0 || n === 0) return false;

  let i = 0, j = n - 1;
  while (i < m && j >= 0) {
    const x = matrix[i][j];
    if (x === target) return true;
    if (x > target) j--;
    else i++;
  }
  return false;
}

console.log(existsInMonotoneMatrix([[1, 2], [3, 4]], 3)); // true
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(m+n)  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 备注 |
| --- | --- | --- | --- | --- |
| 全扫 | 遍历所有元素 | O(mn) | O(1) | 最简单但最慢 |
| 行二分 | 每行二分 | O(m log n) | O(1) | 合理但没用到列单调 |
| **阶梯搜索** | 右上角/左下角指针移动 | **O(m+n)** | **O(1)** | 模板最优解 |

### 常见坑

1. **起点选错导致方向不确定**：右上角/左下角是关键；左上角/右下角无法同时拥有“一个方向变大、一个方向变小”的性质。  
2. **边界没处理**：空矩阵、空行要先返回。  
3. **把“>= / <=”写错**：遇到 `x > target` 只能左移，遇到 `x < target` 只能下移（顺序别反了）。

---

## 常见问题与注意事项

1. **为什么不能用“整体二分”像 LeetCode 74 那样？**  
   因为本题只保证行列各自升序，并不保证“上一行的末尾 < 下一行的开头”。不能直接当成一维有序数组。

2. **有重复元素怎么办？**  
   不影响。算法只依赖单调（非严格也可），命中就返回 true。

3. **能不能返回坐标而不是 bool？**  
   可以：命中时返回 `(i, j)`；否则返回 `(-1, -1)`。复杂度不变。

---

## 最佳实践与建议

- 把该题背成“二维单调矩阵的 stair-search 模板”，以后遇到类似剪枝题直接套  
- 优先选择右上角/左下角作为起点，让“比较后能删一行/列”  
- 在工程里如果要做“找第一个 >= target 的位置”等变体，可在该模板上做小改造

---

## S — Summary（总结）

### 核心收获

- 行列同时升序 => 矩阵具有二维单调结构，可用于剪枝  
- 从右上角出发，`>` 左移、`<` 下移，每步排除一整列或一整行  
- 最多移动 `m+n` 次，时间 O(m+n)，空间 O(1)  
- 行二分是次优方案；全扫是兜底方案  
- 该模板能迁移到费率表、校准表、阈值表等二维单调数据结构

### 参考与延伸阅读

- LeetCode 240. Search a 2D Matrix II
- Monotone matrix / Staircase search 相关讨论
- 二维数据结构的剪枝与“搜索空间单调性”思想

---

## 元信息

- **阅读时长**：10~13 分钟  
- **标签**：Hot100、矩阵、单调性、剪枝  
- **SEO 关键词**：Search a 2D Matrix II, 搜索二维矩阵 II, 阶梯搜索, O(m+n)  
- **元描述**：在行列升序矩阵中搜索 target：右上角阶梯搜索，每步排除一行或一列，O(m+n)/O(1) 解法与多语言实现。  

---

## 行动号召（CTA）

建议你把这题当作“二维单调剪枝模板”的起点：  
写完后再尝试两个变体：返回坐标、以及统计 target 出现次数（若有重复）。  
如果你希望我把这些变体也整理成同风格的短文或追加到本文里，告诉我即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def search_matrix(matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    while i < m and j >= 0:
        x = matrix[i][j]
        if x == target:
            return True
        if x > target:
            j -= 1
        else:
            i += 1
    return False


if __name__ == "__main__":
    mat = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    print(search_matrix(mat, 5))
    print(search_matrix(mat, 20))
```

```c
#include <stdio.h>

int search(int m, int n, int a[m][n], int target) {
    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
        int x = a[i][j];
        if (x == target) return 1;
        if (x > target) --j;
        else ++i;
    }
    return 0;
}

int main(void) {
    int a[5][5] = {
        {1, 4, 7, 11, 15},
        {2, 5, 8, 12, 19},
        {3, 6, 9, 16, 22},
        {10, 13, 14, 17, 24},
        {18, 21, 23, 26, 30},
    };
    printf("%d\n", search(5, 5, a, 5));
    printf("%d\n", search(5, 5, a, 20));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

bool searchMatrix(const std::vector<std::vector<int>>& matrix, int target) {
    int m = (int)matrix.size();
    if (m == 0) return false;
    int n = (int)matrix[0].size();
    if (n == 0) return false;

    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
        int x = matrix[i][j];
        if (x == target) return true;
        if (x > target) --j;
        else ++i;
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> mat{
        {1, 4, 7, 11, 15},
        {2, 5, 8, 12, 19},
        {3, 6, 9, 16, 22},
        {10, 13, 14, 17, 24},
        {18, 21, 23, 26, 30},
    };
    std::cout << std::boolalpha << searchMatrix(mat, 5) << "\n";
    std::cout << std::boolalpha << searchMatrix(mat, 20) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func searchMatrix(matrix [][]int, target int) bool {
    m := len(matrix)
    if m == 0 {
        return false
    }
    n := len(matrix[0])
    if n == 0 {
        return false
    }

    i, j := 0, n-1
    for i < m && j >= 0 {
        x := matrix[i][j]
        if x == target {
            return true
        }
        if x > target {
            j--
        } else {
            i++
        }
    }
    return false
}

func main() {
    mat := [][]int{
        {1, 4, 7, 11, 15},
        {2, 5, 8, 12, 19},
        {3, 6, 9, 16, 22},
        {10, 13, 14, 17, 24},
        {18, 21, 23, 26, 30},
    }
    fmt.Println(searchMatrix(mat, 5))
    fmt.Println(searchMatrix(mat, 20))
}
```

```rust
fn search_matrix(matrix: &Vec<Vec<i32>>, target: i32) -> bool {
    let m = matrix.len();
    if m == 0 {
        return false;
    }
    let n = matrix[0].len();
    if n == 0 {
        return false;
    }

    let mut i: usize = 0;
    let mut j: i32 = (n as i32) - 1;
    while i < m && j >= 0 {
        let x = matrix[i][j as usize];
        if x == target {
            return true;
        }
        if x > target {
            j -= 1;
        } else {
            i += 1;
        }
    }
    false
}

fn main() {
    let mat = vec![
        vec![1, 4, 7, 11, 15],
        vec![2, 5, 8, 12, 19],
        vec![3, 6, 9, 16, 22],
        vec![10, 13, 14, 17, 24],
        vec![18, 21, 23, 26, 30],
    ];
    println!("{}", search_matrix(&mat, 5));
    println!("{}", search_matrix(&mat, 20));
}
```

```javascript
function searchMatrix(matrix, target) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  if (m === 0 || n === 0) return false;

  let i = 0;
  let j = n - 1;
  while (i < m && j >= 0) {
    const x = matrix[i][j];
    if (x === target) return true;
    if (x > target) j--;
    else i++;
  }
  return false;
}

const mat = [
  [1, 4, 7, 11, 15],
  [2, 5, 8, 12, 19],
  [3, 6, 9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30],
];
console.log(searchMatrix(mat, 5));
console.log(searchMatrix(mat, 20));
```

