---
title: "Hot100：螺旋矩阵（Spiral Matrix）边界收缩模拟 ACERS 解析"
date: 2026-02-01T13:55:22+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "矩阵", "模拟", "边界收缩", "数组", "LeetCode 54"]
description: "用“边界收缩”在 O(mn) 时间输出矩阵的顺时针螺旋序列，附工程场景、易错点与多语言实现（Hot100）。"
keywords: ["Spiral Matrix", "螺旋矩阵", "顺时针螺旋遍历", "边界收缩", "LeetCode 54", "Hot100", "O(mn)"]
---

> **副标题 / 摘要**  
> “顺时针螺旋遍历”看似只是打印顺序，实则考验你对边界与循环不变量的掌控。本文用 ACERS 结构给出可直接复用的边界收缩模板，并给出多语言可运行实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`矩阵`、`模拟`、`边界收缩`  
- **SEO 关键词**：Hot100, Spiral Matrix, 螺旋矩阵, 顺时针螺旋遍历, 边界收缩, LeetCode 54  
- **元描述**：用边界收缩法输出矩阵的顺时针螺旋序列，包含推导、工程场景、复杂度对比与多语言代码。  

---

## 目标读者

- 正在刷 Hot100、想把“矩阵模拟题”沉淀成模板的同学  
- 对边界条件容易写错、希望提升代码稳健性的中级开发者  
- 做可视化/栅格数据处理/网格路径相关任务的工程师

## 背景 / 动机

矩阵类题目最容易“写得出来，但写不对”：  
多一层循环、多一个边界判断，就可能在单行/单列、奇偶层数时出错或重复输出。

螺旋遍历是一个很好的训练题：它逼你把 **循环不变量**（哪些行列还没被处理）和 **边界收缩**（每处理完一条边就把边界往里缩）描述清楚，代码才能既短又不炸。

## 核心概念

- **边界（Boundaries）**：用 `top/bottom/left/right` 表示当前还未处理的矩形外框  
- **层（Layer）**：每次循环处理一圈外框（上边、右边、下边、左边）  
- **收缩（Shrink）**：每处理完一条边就移动对应边界：`top++`、`right--`、`bottom--`、`left++`  
- **循环不变量**：始终保证未输出区域是 `top..bottom` × `left..right`

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个 `m` 行 `n` 列的矩阵 `matrix`，请按照 **顺时针螺旋顺序**，返回矩阵中的所有元素。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| matrix | int[][] | `m × n` 的矩阵 |
| 返回 | int[] | 按顺时针螺旋顺序输出的所有元素 |

### 示例 1（自拟）

```text
matrix =
[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
输出: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

### 示例 2（自拟）

```text
matrix =
[
  [ 1,  2,  3,  4],
  [ 5,  6,  7,  8],
  [ 9, 10, 11, 12]
]
输出: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

---

## C — Concepts（核心思想）

### 思路推导：从“标记访问”到“边界收缩”

1. **朴素思路：方向数组 + visited 标记**  
   从 `(0,0)` 出发按右/下/左/上转向；走到越界或已访问就转向。  
   - 优点：直观  
   - 缺点：需要 `m×n` 的 `visited`，空间浪费；代码更容易写出越界/重复判断

2. **关键观察：螺旋遍历 = 一圈圈“剥洋葱”**  
   每一圈只需要处理四条边：上边一行、右边一列、下边一行、左边一列。  
   处理完外圈，就把边界往里缩，问题规模变小。

3. **方法选择：边界收缩（O(1) 额外空间）**  
   用四个指针维护未处理矩形：`top, bottom, left, right`。  
   每轮输出四条边（注意边界交叉时跳过），直到 `top > bottom` 或 `left > right`。

### 方法归类

- **矩阵模拟（Matrix Simulation）**  
- **边界收缩（Boundary Shrinking）**  
- **循环不变量 + 边界条件处理**

### 关键不变量（写对的核心）

进入每轮循环时，未输出区域一定是一个矩形：

```text
行范围: top .. bottom
列范围: left .. right
```

每输出一条边，就把对应边界向内收缩 1：

- 输出上边：`top += 1`  
- 输出右边：`right -= 1`  
- 输出下边：`bottom -= 1`（前提：`top <= bottom`）  
- 输出左边：`left += 1`（前提：`left <= right`）

---

## 实践指南 / 步骤

1. 处理空矩阵：`matrix == []` 或 `matrix[0] == []` 直接返回空数组  
2. 初始化四个边界：`top=0, bottom=m-1, left=0, right=n-1`  
3. 当 `top <= bottom` 且 `left <= right` 时循环：  
   - 走上边（从左到右）  
   - 走右边（从上到下）  
   - 若仍有剩余行：走下边（从右到左）  
   - 若仍有剩余列：走左边（从下到上）  
4. 返回结果

Python 可运行示例（保存为 `spiral_matrix.py`）：

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

## E — Engineering（工程应用）

> 这一题的“工程价值”在于：它是一个**可复用的网格路径生成器**。  
> 你可以把“遍历矩阵”换成“遍历坐标”，再把坐标映射到任何业务对象（像素、瓦片、货架格、内存页、表格单元）。

### 场景 1：图像/栅格数据的螺旋特征提取（Python）

**背景**：在简单的图像特征工程中，常把一个小 patch（如 7×7、11×11）展平成一维向量做特征。  
**为什么适用**：螺旋顺序能把“从外到内”的结构编码进序列，有时比逐行展开更贴近形状边界信息。

（假设你已把上文 `spiral_order` 实现保存为 `spiral_matrix.py`）

```python
from spiral_matrix import spiral_order


def spiral_vector(patch):
    return spiral_order(patch)


if __name__ == "__main__":
    patch = [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    print(spiral_vector(patch))
```

### 场景 2：后端服务按螺旋顺序渐进返回网格数据（Go）

**背景**：地图瓦片、热力图、排座位等网格数据，常需要“从外圈往里”逐步加载/渲染。  
**为什么适用**：边界收缩天然给出了渐进顺序；你甚至可以按圈（layer）分批返回，提高首屏速度。

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

### 场景 3：机器人/自动化巡检的螺旋扫描路径（C）

**背景**：在离散网格上做覆盖式扫描（巡检、清洁、采样），螺旋路径能保证“外圈优先”，且易于分段执行。  
**为什么适用**：算法只需 O(1) 状态；在嵌入式环境里不需要额外 visited 缓冲区。

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
    printf("\n");
}

int main(void) {
    spiral_path(3, 4);
    return 0;
}
```

### 场景 4：前端表格/棋盘的螺旋高亮动画（JavaScript）

**背景**：在 Canvas 或 DOM Grid 中做“螺旋高亮/引导动画”，需要一个稳定的格子访问序列。  
**为什么适用**：直接复用螺旋顺序即可得到动画帧序列。

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

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(mn)，每个元素只进结果一次  
- **空间复杂度**：O(1)（不计返回数组）；若用 visited 则是 O(mn)

### 替代方案对比

| 方法 | 思路 | 额外空间 | 典型问题 |
| --- | --- | --- | --- |
| visited + 方向数组 | “走路 + 转向” | O(mn) | 需要 visited；越界与转向条件更复杂 |
| 递归/按层切片 | 每层取四条边 | 取决于实现 | 代码可读但容易产生切片拷贝或递归栈 |
| **边界收缩（本文）** | 四边遍历 + 收缩 | **O(1)** | 需严谨处理单行/单列边界 |

### 为什么边界收缩更工程可行

- 状态少：四个整数就能描述进度  
- 无需额外矩阵：更省内存、更容易迁移到嵌入式/低资源环境  
- 易做“分批输出”：每一圈天然是一个批次（layer）

---

## 解释与原理（为什么这么做）

把未输出区域看作一个不断缩小的矩形框：

1. **上边（top 行）**：从 `left → right` 输出一整行，说明这一行已完成，因此 `top++`  
2. **右边（right 列）**：从 `top → bottom` 输出一整列，完成后 `right--`  
3. **下边（bottom 行）**：必须保证还有未处理行（`top <= bottom`），再从 `right → left` 输出，完成后 `bottom--`  
4. **左边（left 列）**：必须保证还有未处理列（`left <= right`），再从 `bottom → top` 输出，完成后 `left++`

两个条件判断的意义很关键：  
当只剩一行或一列时，如果不做判断，就会在“下边/左边”阶段重复输出已经输出过的元素。

---

## 常见问题与注意事项

1. **为什么需要 `if top <= bottom` 和 `if left <= right`？**  
   用来处理“只剩一行”或“只剩一列”的情况，避免重复输出。

2. **空矩阵要不要处理？**  
   题目通常保证 `m,n >= 1`，但工程代码建议对空输入返回 `[]`，更健壮。

3. **矩阵行长度不一致怎么办？**  
   题意通常是规则矩阵（每行长度相同）。如果你在工程里拿到“锯齿数组”，需要先做校验或补齐。

4. **如何把“输出元素”改成“输出坐标”？**  
   把 `res.append(matrix[i][j])` 替换成 `res.append((i,j))`（或写到 channel/队列）即可，工程里常用这一招做路径生成。

---

## 最佳实践与建议

- 先写不变量：未处理区域永远是 `top..bottom × left..right`  
- 每移动一次边界都立刻收缩，避免“忘了 top++/right--”  
- 单行/单列的重复输出问题，用两条 if 一次性兜住  
- 需要流式输出时，可以把四段遍历改成“边遍历边 yield/发送”

---

## S — Summary（总结）

### 核心收获

- 螺旋遍历本质是“按层剥离外框”，不是随机转向  
- 用 `top/bottom/left/right` 维护边界，能做到 O(1) 额外空间  
- 两个边界判断（`top<=bottom`、`left<=right`）是避免重复输出的关键  
- 该模板可直接迁移为“网格路径生成器”，适用可视化、巡检、栅格数据处理等场景

### 小结 / 结论

这题写对的标准不是“能过样例”，而是：  
**边界清晰、没有特判地狱、单行/单列稳如老狗**。  
把边界收缩模板背熟，你会发现很多矩阵模拟题都能一把梭。

### 参考与延伸阅读

- LeetCode 54. Spiral Matrix  
- LeetCode 59. Spiral Matrix II（生成螺旋矩阵）  
- LeetCode 885. Spiral Matrix III（按步长扩张的螺旋路径）  
- 矩阵遍历相关：边界、分层、方向数组等经典技巧

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：Hot100、矩阵、模拟、边界收缩、LeetCode 54  
- **SEO 关键词**：Hot100, Spiral Matrix, 螺旋矩阵, 顺时针螺旋遍历, 边界收缩, LeetCode 54  
- **元描述**：用边界收缩法输出矩阵的顺时针螺旋序列（Hot100），包含推导、复杂度与多语言实现。  

---

## 行动号召（CTA）

建议你把本文的“边界收缩模板”封装成一个小工具：  
下一次遇到矩阵模拟题，先把模板复制过来，再把“输出动作”替换成你的业务逻辑。  
如果你在工程里用过螺旋遍历（比如可视化、路径规划），欢迎在评论区分享你的场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

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
    printf("\n");
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
    std::cout << "\n";
    return 0;
}
```

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
    println!("{:?}", spiral_order(&matrix));
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
