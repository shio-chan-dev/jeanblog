---
title: "Hot100：旋转图像（Rotate Image）转置 + 行反转实现原地 90 度旋转 ACERS 解析"
date: 2026-02-01T13:51:51+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "矩阵", "原地", "转置", "反转", "LeetCode 48"]
description: "顺时针旋转 n×n 矩阵 90 度：用“先转置、再反转每一行”在 O(n^2) 时间、O(1) 额外空间原地完成；含推导、工程场景与多语言实现。"
keywords: ["Rotate Image", "旋转图像", "in-place", "matrix transpose", "reverse rows", "O(1) space"]
---

> **副标题 / 摘要**  
> 旋转图像的核心不是“算新坐标”，而是把映射拆成两个可原地执行的操作：**转置（transpose）+ 反转每一行（reverse rows）**。本文按 ACERS 模板给出从朴素解到原地解的推导、常见坑与多语言可运行实现。

- **预计阅读时长**：10~14 分钟  
- **标签**：`Hot100`、`矩阵`、`原地`、`转置`  
- **SEO 关键词**：旋转图像, Rotate Image, 原地旋转 90 度, 转置, 行反转, LeetCode 48  
- **元描述**：顺时针原地旋转 n×n 矩阵 90 度：转置 + 行反转模板解；含思路推导、复杂度对比、工程迁移与多语言实现。  

---

## 目标读者

- 刷 Hot100，想把“矩阵原地技巧”整理成可复用模板的学习者  
- 需要在工程里处理二维网格（图像/棋盘/地图/热力图）变换的开发者  
- 对空间敏感，希望避免额外矩阵拷贝的工程师

## 背景 / 动机

在很多场景里，“旋转”是高频操作：  
图像增强、棋盘/地图方向变换、传感器方向校正、UI 表格视图旋转等。  
如果每次旋转都新建一个矩阵，空间开销是 O(n^2)，在大矩阵或高频调用时会非常“吃内存”，甚至触发 GC/内存抖动。  
因此这题的关键约束是：**必须原地（in-place）完成 90 度旋转**。

## 核心概念

| 概念 | 含义 | 为什么重要 |
| --- | --- | --- |
| 坐标映射 | 旋转后的新坐标与旧坐标之间的关系 | 让你知道“最终要变成什么” |
| 转置（Transpose） | `matrix[i][j]` 与 `matrix[j][i]` 交换 | 原地可做、且能把行列关系对齐 |
| 行反转（Reverse Row） | 把每一行左右翻转 | 与转置组合后刚好等价于顺时针 90 度 |
| 原地算法 | 只用常数额外空间完成变换 | 适合大矩阵与性能场景 |

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个 `n x n` 的二维矩阵 `matrix` 表示图像。请将图像 **顺时针旋转 90 度**。  
要求 **原地修改** `matrix`，不要使用另一个矩阵。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| matrix | int[][] | `n x n` 矩阵 |
| 输出 | void | 原地修改 `matrix` |

### 示例 1（3x3）

```text
输入:
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
]

输出:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

### 示例 2（2x2）

```text
输入:
[
  [1,2],
  [3,4]
]

输出:
[
  [3,1],
  [4,2]
]
```

---

## 思路推导：从“新矩阵”到“原地两步走”

### 朴素方案（不符合原地）：新建一个矩阵

顺时针旋转 90 度的坐标映射是：

```
new[i][j] = old[n - 1 - j][i]
```

用这个公式写起来很直接，但你会立刻遇到问题：  
如果直接覆盖到 `matrix` 上，会把后面还需要用到的旧值覆盖掉；  
所以朴素方案通常会创建 `new` 矩阵，最后再复制回去，空间 O(n^2)，不符合题意。

### 关键观察：把映射拆成两次原地可执行的变换

如果我们先对矩阵 **转置**：

```
transpose: T[i][j] = old[j][i]
```

再对转置后的每一行做 **反转**：

```
reverse row: R[i][j] = T[i][n - 1 - j]
```

组合起来：

```
R[i][j] = T[i][n - 1 - j]
       = old[n - 1 - j][i]
```

这正好等于顺时针旋转 90 度的映射。  
于是我们得到最常用的原地模板：

1) 转置（沿主对角线交换）  
2) 反转每一行

---

## C — Concepts（核心思想）

### 方法归类

- **矩阵原地变换（In-place Matrix Transform）**  
- **分解式等价变换（Decomposition by Equivalence）**：把一个复杂变换拆成多个可原地执行的简单变换  
- **对称交换（Symmetric Swap）**：转置时只交换上三角/下三角，避免重复

### 转置怎么做才是“原地”？

转置沿主对角线交换：`(i, j)` 与 `(j, i)` 是同一对。  
所以只需要遍历上三角（`j > i`）：

```text
for i in [0..n-1]:
  for j in [i+1..n-1]:
    swap(matrix[i][j], matrix[j][i])
```

### 反转每一行怎么做才是“原地”？

每行用左右双指针交换：

```text
l = 0, r = n-1
while l < r:
  swap(row[l], row[r]); l++; r--
```

---

## 实践指南 / 步骤

1. 若 `n <= 1`，直接返回  
2. 转置：只遍历 `j > i` 的上三角并交换  
3. 对每一行做原地反转  
4. 完成顺时针 90 度旋转

Python 可运行示例（保存为 `rotate_image.py`）：

```python
from typing import List


def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    if n <= 1:
        return

    # 1) transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 2) reverse each row
    for i in range(n):
        matrix[i].reverse()


if __name__ == "__main__":
    mat = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    rotate(mat)
    print(mat)
```

---

## E — Engineering（工程应用）

### 场景 1：训练数据增强（旋转方形 patch，Python）

**背景**：CV 训练里经常对图像做旋转增强；在某些 pipeline 中，你会先裁剪出 `n x n` 的方形 patch 再做变换。  
**为什么适用**：对 patch 做原地旋转可避免频繁分配新数组，减少内存峰值与 GC 压力。  

```python
def rotate_patch_inplace(patch):
    n = len(patch)
    for i in range(n):
        for j in range(i + 1, n):
            patch[i][j], patch[j][i] = patch[j][i], patch[i][j]
    for row in patch:
        row.reverse()


if __name__ == "__main__":
    patch = [[1, 2], [3, 4]]
    rotate_patch_inplace(patch)
    print(patch)  # [[3, 1], [4, 2]]
```

### 场景 2：嵌入式传感器网格方向校正（热成像/ToF 传感器，C）

**背景**：很多传感器输出固定朝向的网格（例如 32x32 热成像），设备安装方向可能不同，需要旋转校正。  
**为什么适用**：设备端内存紧张，原地 O(1) 额外空间很关键。  

```c
#include <stdio.h>

void rotate90(int n, int a[n][n]) {
    // transpose
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int tmp = a[i][j];
            a[i][j] = a[j][i];
            a[j][i] = tmp;
        }
    }
    // reverse each row
    for (int i = 0; i < n; ++i) {
        for (int l = 0, r = n - 1; l < r; ++l, --r) {
            int tmp = a[i][l];
            a[i][l] = a[i][r];
            a[i][r] = tmp;
        }
    }
}

int main(void) {
    int a[2][2] = {{1,2},{3,4}};
    rotate90(2, a);
    printf("%d %d\n%d %d\n", a[0][0], a[0][1], a[1][0], a[1][1]);
    return 0;
}
```

### 场景 3：前端棋盘/地图视图旋转（JavaScript）

**背景**：拼图/棋类/网格编辑器常见“旋转棋盘视角”功能，底层通常就是二维数组。  
**为什么适用**：直接改原数组，避免复制大棋盘导致卡顿，尤其在移动端更明显。  

```javascript
function rotate90(matrix) {
  const n = matrix.length;
  if (n <= 1) return;

  // transpose
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const tmp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = tmp;
    }
  }

  // reverse each row
  for (let i = 0; i < n; i++) {
    matrix[i].reverse();
  }
}

const board = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];
rotate90(board);
console.log(board);
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n^2)（转置 O(n^2) + 行反转 O(n^2)）  
- **空间复杂度**：O(1)（只用常数个临时变量交换）

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 评价 |
| --- | --- | --- | --- | --- |
| 新矩阵映射 | 按 `new[i][j] = old[n-1-j][i]` 写新矩阵 | O(n^2) | O(n^2) | 好写但不符合题意 |
| **转置 + 行反转** | 拆成两次原地操作 | O(n^2) | **O(1)** | 最常用模板，易实现 |
| 分层四元交换 | 一层一层做 4 个点循环换位 | O(n^2) | O(1) | 也很好，但实现更易写错边界 |

### 常见错误思路

1. **直接按映射覆盖原矩阵**：会覆盖掉未来还要读取的旧值  
2. **转置写错遍历范围**：`j` 从 `0` 开始会把对称元素交换两次，等于没做  
3. **只反转列/只反转行**：与转置组合顺序不对会变成逆时针或镜像

---

## 解释与原理（为什么这么做）

顺时针 90 度旋转的本质是坐标映射：

```
(i, j) -> (j, n - 1 - i)
```

而“转置 + 行反转”恰好等价于这个映射：  
转置负责把 `i/j` 对调；行反转负责把列索引变成 `n-1-列`。  
这就是为什么该模板既原地又正确。

---

## 常见问题与注意事项

1. **为什么只适用于 n×n？**  
   因为 90 度旋转会把宽高互换；若不是方阵，就不可能在同一块二维数组里原地完成（除非改变存储结构）。

2. **如何做逆时针 90 度？**  
   也可以原地：先转置，再反转每一列（或先反转每一行，再转置）。  

3. **如何旋转 180 度？**  
   旋转两次 90 度即可；或直接“每行反转 + 行顺序反转”。

4. **n 很大时会超时吗？**  
   O(n^2) 已是下界级别（你需要访问大部分元素），通常不会是瓶颈；真正的大瓶颈在内存访问与缓存局部性，但本算法已经足够顺序友好。

---

## 最佳实践与建议

- 把“转置 + 行反转”记成矩阵顺时针 90 度旋转的模板解  
- 转置时只遍历 `j > i` 的上三角，避免重复交换  
- 写测试时覆盖奇偶 n（例如 1、2、3、4）以及包含重复值的矩阵  
- 工程里如果不是方阵：优先改数据结构或接受额外矩阵（空间换清晰）

---

## S — Summary（总结）

### 核心收获

- 顺时针 90 度旋转的坐标映射是 `new[i][j] = old[n-1-j][i]`  
- 直接覆盖会污染未读数据，必须拆分成可原地执行的步骤  
- **转置 + 反转每一行** 与旋转 90 度完全等价，且可做到 O(1) 额外空间  
- 该技巧可迁移到棋盘、网格、图像 patch 等二维数据变换  
- 分层四元交换是同复杂度的替代方案，但更容易写错边界

### 小结 / 结论

把这题背成一句话：  
**“顺时针 90 度旋转 = 转置 + 每行反转”**。  
以后遇到矩阵旋转/变换类题，先想能否拆成若干原地操作，往往能一把过。

### 参考与延伸阅读

- LeetCode 48. Rotate Image
- 线性代数中关于转置与对称变换的基础概念
- 图像处理中的几何变换（旋转/翻转/仿射）

---

## 元信息

- **阅读时长**：10~14 分钟  
- **标签**：Hot100、矩阵、原地、转置、反转  
- **SEO 关键词**：Rotate Image, 旋转图像, 原地旋转 90 度, 转置, 行反转, LeetCode 48  
- **元描述**：顺时针原地旋转 n×n 矩阵 90 度：转置 + 行反转模板解；含推导、工程迁移与多语言实现。  

---

## 行动号召（CTA）

建议你做两件事巩固：  
1) 手写一遍“分层四元交换”的版本；2) 写几个奇偶 n 的用例自测。  
如果你希望我把“逆时针/180 度/任意 k 次旋转”的统一模板也整理成一篇短文，告诉我即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    if n <= 1:
        return

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for row in matrix:
        row.reverse()


if __name__ == "__main__":
    mat = [[1, 2], [3, 4]]
    rotate(mat)
    print(mat)
```

```c
#include <stdio.h>

void rotate(int n, int a[n][n]) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int tmp = a[i][j];
            a[i][j] = a[j][i];
            a[j][i] = tmp;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int l = 0, r = n - 1; l < r; ++l, --r) {
            int tmp = a[i][l];
            a[i][l] = a[i][r];
            a[i][r] = tmp;
        }
    }
}

int main(void) {
    int a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
    rotate(3, a);
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

void rotate(std::vector<std::vector<int>>& matrix) {
    int n = (int)matrix.size();
    if (n <= 1) return;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        int l = 0, r = n - 1;
        while (l < r) {
            std::swap(matrix[i][l], matrix[i][r]);
            ++l;
            --r;
        }
    }
}

int main() {
    std::vector<std::vector<int>> mat{{1,2,3},{4,5,6},{7,8,9}};
    rotate(mat);
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

func rotate(matrix [][]int) {
    n := len(matrix)
    if n <= 1 {
        return
    }

    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }

    for i := 0; i < n; i++ {
        for l, r := 0, n-1; l < r; l, r = l+1, r-1 {
            matrix[i][l], matrix[i][r] = matrix[i][r], matrix[i][l]
        }
    }
}

func main() {
    mat := [][]int{{1, 2}, {3, 4}}
    rotate(mat)
    fmt.Println(mat)
}
```

```rust
fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();
    if n <= 1 {
        return;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let tmp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tmp;
        }
    }

    for i in 0..n {
        let mut l = 0usize;
        let mut r = n - 1;
        while l < r {
            matrix[i].swap(l, r);
            l += 1;
            r -= 1;
        }
    }
}

fn main() {
    let mut mat = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    rotate(&mut mat);
    println!("{:?}", mat);
}
```

```javascript
function rotate(matrix) {
  const n = matrix.length;
  if (n <= 1) return;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const tmp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = tmp;
    }
  }

  for (let i = 0; i < n; i++) {
    matrix[i].reverse();
  }
}

const mat = [
  [1, 2],
  [3, 4],
];
rotate(mat);
console.log(mat);
```

