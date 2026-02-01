---
title: "矩阵置零：用首行首列做标记实现原地 O(1) 空间（LeetCode 73）"
date: 2026-02-01T12:53:35+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "矩阵", "数组", "原地", "标记", "LeetCode 73"]
description: "把首行首列当作标记位，在不额外开集合的前提下原地完成矩阵置零；含思路推导、常见坑与多语言实现。"
keywords: ["Set Matrix Zeroes", "矩阵置零", "in-place", "O(1) space", "first row first column marker"]
---

> **副标题 / 摘要**  
> “矩阵置零”是典型的二维标记传播问题：某个位置为 0，会影响整行整列。本文用 ACERS 结构讲清楚为什么不能直接改、如何用首行首列做标记实现原地 O(1) 额外空间，并给出多语言可运行代码。

- **预计阅读时长**：12~15 分钟  
- **标签**：`矩阵`、`原地算法`、`标记位`  
- **SEO 关键词**：矩阵置零, 原地 O(1) 空间, 首行首列标记, LeetCode 73  
- **元描述**：用首行首列作标记位，原地将含 0 的行与列全部置零；包含推导、复杂度对比、工程迁移与多语言实现。  

---

## 目标读者

- 刷 LeetCode，想把“二维数组原地技巧”沉淀成稳定模板的同学  
- 需要在工程里做二维网格/表格/矩阵数据清洗与传播标记的开发者  
- 对空间优化敏感（嵌入式、性能场景、内存受限）的工程师

## 背景 / 动机

二维数据在工程里到处都是：表格、图像、传感器网格、关联矩阵……  
“某个单元格触发规则 -> 影响整行整列”这种联动，本质就是 **行列传播（row/col propagation）**。  
这题额外要求“原地”，逼你掌握一个非常通用的技巧：**用数据结构本身的某些位置当作标记位**，避免额外内存。

## 核心概念

- **传播标记**：发现 0 后，不是立刻改整行整列，而是先记录“哪些行/列要被清零”  
- **原地（in-place）**：只允许 O(1) 额外空间（不算输入矩阵本身）  
- **标记位复用**：把 `matrix[0][j]` 当作“第 j 列要清零”的标记，把 `matrix[i][0]` 当作“第 i 行要清零”的标记  
- **首行/首列特判**：首行/首列既是数据又是标记位，因此需要单独用两个布尔量记录它们是否本来就该清零

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个 `m x n` 矩阵 `matrix`：如果某个元素为 `0`，则将该元素所在的 **整行** 与 **整列** 的所有元素都设置为 `0`。  
要求 **原地修改** `matrix`（通常不需要返回值）。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| matrix | int[][] | `m x n` 矩阵 |
| 输出 | void | 原地修改 `matrix` |

### 示例 1

```text
输入:
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]

输出:
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
```

### 示例 2

```text
输入:
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]

输出:
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

---

## 思路推导：从“直接改”到“原地标记”

### 朴素误区：看到 0 就立刻把行列改成 0

这会把“后续产生的 0”也当成“原始 0”继续传播，导致过度清零。

反例（说明“立刻改”会连锁污染）：

```text
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
```

如果你在遍历时遇到中间的 0，立刻把第二行/第二列清零，那么矩阵里会新增很多 0；  
当遍历继续走到这些新增 0 时，你又会继续清零别的行列，最终整矩阵可能都变 0（错误）。

### 正确方向：先“记录”要清零的行/列，再统一写回

最直接做法是：

- 第一遍扫矩阵：用集合 `rows`/`cols` 记录出现 0 的行号/列号  
- 第二遍扫矩阵：若 `i in rows` 或 `j in cols` 就置 0

这很稳，但额外空间是 O(m+n)。

### 空间优化关键：把首行首列当作 rows/cols 的“集合”

观察：

- `matrix[i][0]` 这 m 个格子足够记录“第 i 行要不要清零”
- `matrix[0][j]` 这 n 个格子足够记录“第 j 列要不要清零”

于是：

1. 先用两个布尔量记住：首行是否本来就有 0？首列是否本来就有 0？  
2. 从 `(1,1)` 开始扫描：遇到 `matrix[i][j]==0`，就写标记：
   - `matrix[i][0]=0`（第 i 行要清零）
   - `matrix[0][j]=0`（第 j 列要清零）
3. 第二遍从 `(1,1)` 扫：看标记位决定置 0
4. 最后按两个布尔量决定是否清空首行/首列

---

## C — Concepts（核心思想）

### 方法归类

- **二维数组原地标记（In-place Marker）**
- **哨兵位/复用存储（Sentinel / Storage Reuse）**
- **两遍扫描（Two-pass）**：一遍打标记，一遍按标记写回

### 为什么一定需要“首行/首列的两个布尔量”？

因为首行首列被我们拿来当标记位，它们原本的数据会被覆盖。  
举个最典型的冲突：

- 如果 `matrix[0][0]` 是 0，它既可能表示“首行要清零”，也可能表示“首列要清零”，单靠一个格子区分不了两种信息。

因此必须额外保存：

- `row0_zero`: 首行是否包含 0
- `col0_zero`: 首列是否包含 0

这两个布尔量是整个算法唯一的额外空间（O(1)）。

---

## 实践指南 / 步骤

1. 扫首行：若存在 0，记 `row0_zero = True`  
2. 扫首列：若存在 0，记 `col0_zero = True`  
3. 扫内部区域 `(1..m-1, 1..n-1)`：遇到 0，则设置行标记/列标记  
4. 再扫内部区域：若行标记为 0 或列标记为 0，则把该格设为 0  
5. 若 `row0_zero` 为真：清空首行  
6. 若 `col0_zero` 为真：清空首列

Python 可运行示例（保存为 `set_matrix_zeroes.py`）：

```python
from typing import List


def set_zeroes(matrix: List[List[int]]) -> None:
    if not matrix or not matrix[0]:
        return

    m, n = len(matrix), len(matrix[0])

    row0_zero = any(matrix[0][j] == 0 for j in range(n))
    col0_zero = any(matrix[i][0] == 0 for i in range(m))

    # Use first row/col as markers.
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Apply markers to inner cells.
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if row0_zero:
        for j in range(n):
            matrix[0][j] = 0

    if col0_zero:
        for i in range(m):
            matrix[i][0] = 0


if __name__ == "__main__":
    a = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    set_zeroes(a)
    print(a)
```

---

## E — Engineering（工程应用）

### 场景 1：数据清洗（样本-特征矩阵的“失效传播”，Python）

**背景**：`m` 条样本、`n` 个特征形成矩阵；某个值为 0 代表“该样本/该特征出现硬失效”。  
**为什么适用**：一旦某样本出现硬失效，往往整行都不可用；而某个特征列失效时，整列都要被屏蔽。  

```python
def invalidate_rows_cols(mat):
    # 直接复用 set_zeroes：0 作为失效哨兵
    from typing import List

    def set_zeroes(matrix: List[List[int]]) -> None:
        if not matrix or not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])
        row0 = any(matrix[0][j] == 0 for j in range(n))
        col0 = any(matrix[i][0] == 0 for i in range(m))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row0:
            for j in range(n):
                matrix[0][j] = 0
        if col0:
            for i in range(m):
                matrix[i][0] = 0

    set_zeroes(mat)


if __name__ == "__main__":
    mat = [
        [10, 20, 30],
        [40,  0, 50],
        [60, 70, 80],
    ]
    invalidate_rows_cols(mat)
    print(mat)
```

### 场景 2：产线网格质检（缺陷传播到行列，C）

**背景**：`m x n` 网格传感器里，读数为 0 表示该点缺陷；为了快速定位问题批次，会把整行整列标为 0。  
**为什么适用**：原地算法常见于内存受限设备（MCU/边缘计算），O(1) 额外空间很重要。

```c
#include <stdio.h>

void setZeroes(int m, int n, int a[m][n]) {
    int row0 = 0, col0 = 0;
    for (int j = 0; j < n; ++j) if (a[0][j] == 0) row0 = 1;
    for (int i = 0; i < m; ++i) if (a[i][0] == 0) col0 = 1;

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (a[i][j] == 0) {
                a[i][0] = 0;
                a[0][j] = 0;
            }
        }
    }

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (a[i][0] == 0 || a[0][j] == 0) a[i][j] = 0;
        }
    }

    if (row0) for (int j = 0; j < n; ++j) a[0][j] = 0;
    if (col0) for (int i = 0; i < m; ++i) a[i][0] = 0;
}

int main(void) {
    int a[3][4] = {
        {0, 1, 2, 0},
        {3, 4, 5, 2},
        {1, 3, 1, 5},
    };
    setZeroes(3, 4, a);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) printf("%d ", a[i][j]);
        printf("\n");
    }
    return 0;
}
```

### 场景 3：前端表格联动（输入 0 后清空所在行列，JavaScript）

**背景**：表格编辑器里，把 0 当作“清空信号”；用户在某格输入 0 时，整行整列都要被清空并刷新 UI。  
**为什么适用**：直接在原数组上改，避免复制大矩阵带来的卡顿。

```javascript
function setZeroes(matrix) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  if (m === 0 || n === 0) return;

  let row0 = false;
  let col0 = false;
  for (let j = 0; j < n; j++) if (matrix[0][j] === 0) row0 = true;
  for (let i = 0; i < m; i++) if (matrix[i][0] === 0) col0 = true;

  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      if (matrix[i][j] === 0) {
        matrix[i][0] = 0;
        matrix[0][j] = 0;
      }
    }
  }

  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      if (matrix[i][0] === 0 || matrix[0][j] === 0) matrix[i][j] = 0;
    }
  }

  if (row0) for (let j = 0; j < n; j++) matrix[0][j] = 0;
  if (col0) for (let i = 0; i < m; i++) matrix[i][0] = 0;
}

const grid = [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1],
];
setZeroes(grid);
console.log(grid);
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(mn)（两遍扫描矩阵）  
- **空间复杂度**：O(1)（只用两个布尔量；标记位复用矩阵首行首列）

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 备注 |
| --- | --- | --- | --- | --- |
| 直接改（错误） | 遇到 0 立刻清行列 | O(mn) | O(1) | 会被新增 0 污染 |
| 集合记录 | 记录 rows/cols 再写回 | O(mn) | O(m+n) | 最直观、最不容易写错 |
| **首行首列标记** | 用 matrix[0][j]/matrix[i][0] 存标记 | O(mn) | **O(1)** | 模板解法，但要处理首行首列 |

### 为什么当前方法最工程可行

- 不需要额外分配与 m/n 相关的内存，适合大矩阵与内存敏感场景  
- 逻辑稳定：标记阶段与写回阶段分离，不会被“新增 0”污染  
- 模板化强：二维原地题（如打标记、染色、传播）经常复用类似结构

---

## 解释与原理（为什么这么做）

核心点只有一句话：**“先打标记，再按标记写回。”**  
用首行首列当标记位，相当于把 `rows`/`cols` 两个集合“嵌入”进矩阵本身；  
而额外的 `row0_zero/col0_zero` 用来解决首行首列在“数据”和“标记”之间的冲突。

---

## 常见问题与注意事项

1. **为什么要先扫首行/首列？**  
   因为后续我们会改动首行首列来写标记，如果不提前记录，就会丢失“它们是否本来包含 0”的信息。

2. **能不能用 `matrix[0][0]` 同时表示首行和首列？**  
   不行。`matrix[0][0]` 只有一个比特的信息量，区分不了“首行要清零”和“首列要清零”两件事，所以需要两个布尔量。

3. **遍历顺序有讲究吗？**  
   标记阶段从 `(1,1)` 开始；写回阶段也从 `(1,1)` 开始。最后才处理首行/首列。

4. **矩阵为空怎么办？**  
   先判断 `matrix` 或 `matrix[0]` 为空，直接返回（各语言实现里都要注意）。

---

## 最佳实践与建议

- 把“首行首列作为标记位 + 两个布尔量”的模式背成模板  
- 写代码时强制分成三个阶段：`scan row0/col0 -> mark -> apply -> handle row0/col0`  
- 调试时优先构造包含以下情况的用例：
  - 0 在首行
  - 0 在首列
  - 0 在 `matrix[0][0]`
  - 多个 0 分布在不同区域

---

## S — Summary（总结）

### 核心收获

- “矩阵置零”不能边遍历边清零，否则会被新增 0 污染  
- 正确解法是“两阶段”：先记录要清零的行/列，再统一写回  
- 用首行首列可把 `rows/cols` 的标记嵌入矩阵本身，实现 O(1) 额外空间  
- 首行首列的信息会被覆盖，必须用 `row0_zero/col0_zero` 额外保存  
- 该模板可迁移到很多二维原地题与工程里的联动传播场景

### 小结 / 结论

这题的价值不在“置零”本身，而在于：你掌握了一个非常高频的二维原地技巧。  
后续遇到类似“按行/列传播标记”的题，优先尝试用首行首列复用存储。

### 参考与延伸阅读

- LeetCode 73. Set Matrix Zeroes
- “in-place algorithm / sentinel marker” 相关讲解（任意算法教材的空间优化章节）
- 你也可以对照做一遍“集合记录 O(m+n)”版本，帮助理解为什么首行首列能替代集合

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：矩阵、原地、标记位、空间优化  
- **SEO 关键词**：矩阵置零, Set Matrix Zeroes, 原地 O(1), 首行首列标记, LeetCode 73  
- **元描述**：用首行首列作标记位，原地将含 0 的行与列全部置零；含推导、复杂度对比与多语言实现。  

---

## 行动号召（CTA）

建议你把这题的代码写成一个“二维原地标记模板”，之后刷到相似题直接套。  
如果你在项目里遇到过“某格触发 -> 整行整列联动”的真实场景，也欢迎留言交流。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def set_zeroes(matrix: List[List[int]]) -> None:
    if not matrix or not matrix[0]:
        return

    m, n = len(matrix), len(matrix[0])

    row0_zero = any(matrix[0][j] == 0 for j in range(n))
    col0_zero = any(matrix[i][0] == 0 for i in range(m))

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if row0_zero:
        for j in range(n):
            matrix[0][j] = 0

    if col0_zero:
        for i in range(m):
            matrix[i][0] = 0


if __name__ == "__main__":
    mat = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes(mat)
    print(mat)
```

```c
#include <stdio.h>

void setZeroes(int m, int n, int a[m][n]) {
    int row0 = 0, col0 = 0;
    for (int j = 0; j < n; ++j) if (a[0][j] == 0) row0 = 1;
    for (int i = 0; i < m; ++i) if (a[i][0] == 0) col0 = 1;

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (a[i][j] == 0) {
                a[i][0] = 0;
                a[0][j] = 0;
            }
        }
    }

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (a[i][0] == 0 || a[0][j] == 0) a[i][j] = 0;
        }
    }

    if (row0) for (int j = 0; j < n; ++j) a[0][j] = 0;
    if (col0) for (int i = 0; i < m; ++i) a[i][0] = 0;
}

int main(void) {
    int a[3][3] = {{1,1,1},{1,0,1},{1,1,1}};
    setZeroes(3, 3, a);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) printf("%d ", a[i][j]);
        printf("\n");
    }
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

void setZeroes(std::vector<std::vector<int>>& matrix) {
    int m = (int)matrix.size();
    int n = m == 0 ? 0 : (int)matrix[0].size();
    if (m == 0 || n == 0) return;

    bool row0 = false, col0 = false;
    for (int j = 0; j < n; ++j) if (matrix[0][j] == 0) row0 = true;
    for (int i = 0; i < m; ++i) if (matrix[i][0] == 0) col0 = true;

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }

    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
        }
    }

    if (row0) for (int j = 0; j < n; ++j) matrix[0][j] = 0;
    if (col0) for (int i = 0; i < m; ++i) matrix[i][0] = 0;
}

int main() {
    std::vector<std::vector<int>> mat{{1,1,1},{1,0,1},{1,1,1}};
    setZeroes(mat);
    for (auto& row : mat) {
        for (int x : row) std::cout << x << " ";
        std::cout << "\n";
    }
    return 0;
}
```

```go
package main

import "fmt"

func setZeroes(matrix [][]int) {
    m := len(matrix)
    if m == 0 {
        return
    }
    n := len(matrix[0])
    if n == 0 {
        return
    }

    row0 := false
    col0 := false
    for j := 0; j < n; j++ {
        if matrix[0][j] == 0 {
            row0 = true
        }
    }
    for i := 0; i < m; i++ {
        if matrix[i][0] == 0 {
            col0 = true
        }
    }

    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if matrix[i][j] == 0 {
                matrix[i][0] = 0
                matrix[0][j] = 0
            }
        }
    }

    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if matrix[i][0] == 0 || matrix[0][j] == 0 {
                matrix[i][j] = 0
            }
        }
    }

    if row0 {
        for j := 0; j < n; j++ {
            matrix[0][j] = 0
        }
    }
    if col0 {
        for i := 0; i < m; i++ {
            matrix[i][0] = 0
        }
    }
}

func main() {
    mat := [][]int{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}
    setZeroes(mat)
    fmt.Println(mat)
}
```

```rust
fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let m = matrix.len();
    if m == 0 {
        return;
    }
    let n = matrix[0].len();
    if n == 0 {
        return;
    }

    let mut row0 = false;
    let mut col0 = false;
    for j in 0..n {
        if matrix[0][j] == 0 {
            row0 = true;
        }
    }
    for i in 0..m {
        if matrix[i][0] == 0 {
            col0 = true;
        }
    }

    for i in 1..m {
        for j in 1..n {
            if matrix[i][j] == 0 {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }

    for i in 1..m {
        for j in 1..n {
            if matrix[i][0] == 0 || matrix[0][j] == 0 {
                matrix[i][j] = 0;
            }
        }
    }

    if row0 {
        for j in 0..n {
            matrix[0][j] = 0;
        }
    }
    if col0 {
        for i in 0..m {
            matrix[i][0] = 0;
        }
    }
}

fn main() {
    let mut mat = vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]];
    set_zeroes(&mut mat);
    println!("{:?}", mat);
}
```

```javascript
function setZeroes(matrix) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  if (m === 0 || n === 0) return;

  let row0 = false;
  let col0 = false;
  for (let j = 0; j < n; j++) if (matrix[0][j] === 0) row0 = true;
  for (let i = 0; i < m; i++) if (matrix[i][0] === 0) col0 = true;

  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      if (matrix[i][j] === 0) {
        matrix[i][0] = 0;
        matrix[0][j] = 0;
      }
    }
  }

  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      if (matrix[i][0] === 0 || matrix[0][j] === 0) matrix[i][j] = 0;
    }
  }

  if (row0) for (let j = 0; j < n; j++) matrix[0][j] = 0;
  if (col0) for (let i = 0; i < m; i++) matrix[i][0] = 0;
}

const mat = [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1],
];
setZeroes(mat);
console.log(mat);
```
