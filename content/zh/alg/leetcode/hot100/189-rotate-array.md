---
title: "Hot100：轮转数组（Rotate Array）三次反转 ACERS 解析"
date: 2026-01-24T12:29:42+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "旋转", "反转", "双指针", "LeetCode 189"]
description: "用三次反转在 O(n) 时间完成轮转数组，含工程场景、常见误区与多语言实现。"
keywords: ["Rotate Array", "轮转数组", "数组旋转", "反转", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 轮转数组是典型的数组变换题：把数组整体向右移动 k 位。本文用 ACERS 拆解“三次反转”的核心思路，并给出工程场景迁移与多语言可运行实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`数组`、`旋转`  
- **SEO 关键词**：Rotate Array, 轮转数组, 数组旋转, 反转, O(n)  
- **元描述**：三次反转法解决轮转数组，含复杂度对比、工程场景与多语言代码。  

---

## 目标读者

- 正在刷 Hot100 的学习者  
- 想掌握“数组原地变换”模板的中级开发者  
- 需要处理时间序列对齐、轮值偏移的工程师

## 背景 / 动机

轮转数组在工程中非常常见：  
轮值排班、时间序列对齐、环形缓冲区、前端轮播等都可以抽象为“整体右移 k 位”。  
如果用逐步移动会变成 O(nk)，在数据量稍大时就不可用，因此需要更高效的原地方案。

## 核心概念

- **轮转（rotate）**：把数组向右移动 k 位，后 k 个元素移到最前  
- **k 取模**：`k %= n`，避免 k 超过数组长度  
- **反转（reverse）**：用双指针交换来原地反转区间  
- **原地（in-place）**：在原数组上操作，额外空间 O(1)

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| k | int | 向右轮转步数 |
| 返回 | int[] | 轮转后的数组 |

### 示例 1（官方）

```text
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
```

### 示例 2（官方）

```text
输入: nums = [-1,-100,3,99], k = 2
输出: [3,99,-1,-100]
```

---

## C — Concepts（核心思想）

### 关键思路：三次反转

1. 反转整个数组  
2. 反转前 k 个  
3. 反转后 n-k 个  

反转后的位置关系刚好等价于右移 k 位。

### 方法归类

- 数组原地操作  
- 双指针反转  
- 贪心式局部处理（每段各自就位）

### 关键公式

```
k = k % n
```

### 概念模型

```text
[1,2,3,4,5,6,7], k=3
整体反转:        [7,6,5,4,3,2,1]
反转前 k=3:       [5,6,7,4,3,2,1]
反转后 n-k=4:     [5,6,7,1,2,3,4]
```

---

## 实践指南 / 步骤

1. 获取数组长度 n，若 n 为 0 直接返回  
2. 计算 `k %= n`，处理 k 大于 n 的情况  
3. 反转 `[0, n-1]`  
4. 反转 `[0, k-1]`  
5. 反转 `[k, n-1]`  

运行方式示例：

```bash
python3 rotate_array.py
```

## 可运行示例（Python）

```python
from typing import List


def rotate(nums: List[int], k: int) -> List[int]:
    n = len(nums)
    if n == 0:
        return nums
    k %= n
    if k == 0:
        return nums

    def rev(i: int, j: int) -> None:
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

    rev(0, n - 1)
    rev(0, k - 1)
    rev(k, n - 1)
    return nums


if __name__ == "__main__":
    print(rotate([1, 2, 3, 4, 5, 6, 7], 3))
    print(rotate([-1, -100, 3, 99], 2))
```

---

## E — Engineering（工程应用）

### 场景 1：时间序列对齐（Python，数据分析）

**背景**：对跨时区指标做对齐，需要把序列整体平移。  
**为什么适用**：轮转操作可直接完成“整体偏移”，逻辑清晰。

```python
def rotate_series(values, k):
    n = len(values)
    if n == 0:
        return values
    k %= n
    return values[-k:] + values[:-k]


print(rotate_series([10, 20, 30, 40, 50], 2))
```

### 场景 2：环形缓冲区起点调整（C，系统编程）

**背景**：日志采集使用环形缓冲区，需要调整读起点以便对齐采样窗口。  
**为什么适用**：原地反转能避免额外内存拷贝。

```c
#include <stdio.h>

static void reverse(int *nums, int l, int r) {
    while (l < r) {
        int tmp = nums[l];
        nums[l] = nums[r];
        nums[r] = tmp;
        l++;
        r--;
    }
}

static void rotate(int *nums, int n, int k) {
    if (n <= 1) return;
    k %= n;
    if (k == 0) return;
    reverse(nums, 0, n - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, n - 1);
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5, 6, 7};
    int n = (int)(sizeof(nums) / sizeof(nums[0]));
    rotate(nums, n, 3);
    for (int i = 0; i < n; ++i) {
        printf("%d%s", nums[i], i + 1 == n ? "\n" : " ");
    }
    return 0;
}
```

### 场景 3：前端轮播与推荐排序（JavaScript，前端）

**背景**：商品/内容列表需要轮播展示，从不同起点开始。  
**为什么适用**：数组轮转可快速生成新的展示顺序。

```javascript
function rotateList(nums, k) {
  const n = nums.length;
  if (n === 0) return nums;
  k %= n;
  return nums.slice(-k).concat(nums.slice(0, n - k));
}

console.log(rotateList(["A", "B", "C", "D"], 1));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
- **空间复杂度**：O(1) 额外空间（原地反转）

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 暴力逐步移动 | 每次移动 1 位，重复 k 次 | O(nk) | k 大时不可用 |
| 额外数组 | `new[(i+k)%n] = old[i]` | O(n) | 需要额外空间 |
| 环状替换 | 按循环交换 | O(n) | 实现更复杂 |
| 三次反转 | 反转分段到位 | O(n) | 简洁、稳定 |

### 为什么当前方法最优 / 最工程可行

三次反转只需一次全局扫描 + 两次局部扫描，  
同时保持原地操作，既高效又易于维护，是工程上最常用的做法。

---

## 解释与原理（为什么这么做）

把数组整体反转后，原本在末尾的元素移动到了前面，但顺序被颠倒。  
再对前 k 个与后 n-k 个分别反转，即可恢复各自内部顺序。  
这相当于把 “右移 k 位” 转换成 “三次反转”，逻辑更清晰且可复用。

---

## 常见问题与注意事项

1. **k 可能大于 n**：必须先做 `k %= n`  
2. **空数组或单元素**：直接返回即可  
3. **k 为 0**：无需做任何反转  
4. **语言差异**：注意原地修改 vs 返回新数组

---

## 最佳实践与建议

- 把反转写成独立函数，避免重复代码  
- 先处理 k 的取模，避免越界  
- 如果业务不要求原地，可用新数组实现更直观的写法  
- 单测覆盖 k=0、k=n、k>n 的边界

---

## S — Summary（总结）

### 核心收获

- 轮转数组本质是位置映射 `(i + k) % n`  
- 三次反转实现 O(n) 时间、O(1) 额外空间  
- 先取模再反转是正确性的关键  
- 该模型可迁移到轮值、窗口对齐、轮播排序等场景  

### 推荐延伸阅读

- LeetCode 189. Rotate Array  
- C++ `std::reverse` 与区间反转  
- 环形缓冲区与数组旋转的工程应用  

### 小结 / 结论

轮转数组是一道非常典型的“原地变换”题。  
掌握三次反转后，很多区间变换问题都会变得直接可解。

---

## 参考与延伸阅读

- https://leetcode.com/problems/rotate-array/  
- https://en.cppreference.com/w/cpp/algorithm/reverse  
- https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types  
- https://pkg.go.dev/sort  

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、数组、旋转、反转  
- **SEO 关键词**：Rotate Array, 轮转数组, 数组旋转, 反转, O(n)  
- **元描述**：三次反转法解决轮转数组，含复杂度对比与工程场景。  

---

## 行动号召（CTA）

如果你正在刷 Hot100，建议把“反转 + 区间变换”整理成模板库，  
欢迎留言分享你在工程中的轮转应用场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def rotate(nums: List[int], k: int) -> None:
    n = len(nums)
    if n == 0:
        return
    k %= n
    if k == 0:
        return

    def rev(i: int, j: int) -> None:
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

    rev(0, n - 1)
    rev(0, k - 1)
    rev(k, n - 1)


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7]
    rotate(data, 3)
    print(data)
```

```c
#include <stdio.h>

static void reverse(int *nums, int l, int r) {
    while (l < r) {
        int tmp = nums[l];
        nums[l] = nums[r];
        nums[r] = tmp;
        l++;
        r--;
    }
}

static void rotate(int *nums, int n, int k) {
    if (n <= 1) return;
    k %= n;
    if (k == 0) return;
    reverse(nums, 0, n - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, n - 1);
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5, 6, 7};
    int n = (int)(sizeof(nums) / sizeof(nums[0]));
    rotate(nums, n, 3);
    for (int i = 0; i < n; ++i) {
        printf("%d%s", nums[i], i + 1 == n ? "\n" : " ");
    }
    return 0;
}
```

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

void rotate(std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    if (n == 0) return;
    k %= n;
    if (k == 0) return;
    std::reverse(nums.begin(), nums.end());
    std::reverse(nums.begin(), nums.begin() + k);
    std::reverse(nums.begin() + k, nums.end());
}

int main() {
    std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
    rotate(nums, 3);
    for (int x : nums) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func reverse(nums []int, l, r int) {
	for l < r {
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
}

func rotate(nums []int, k int) {
	n := len(nums)
	if n == 0 {
		return
	}
	k %= n
	if k == 0 {
		return
	}
	reverse(nums, 0, n-1)
	reverse(nums, 0, k-1)
	reverse(nums, k, n-1)
}

func main() {
	nums := []int{1, 2, 3, 4, 5, 6, 7}
	rotate(nums, 3)
	fmt.Println(nums)
}
```

```rust
fn rotate(nums: &mut Vec<i32>, k: usize) {
    let n = nums.len();
    if n == 0 {
        return;
    }
    let k = k % n;
    if k == 0 {
        return;
    }
    nums.reverse();
    nums[..k].reverse();
    nums[k..].reverse();
}

fn main() {
    let mut nums = vec![1, 2, 3, 4, 5, 6, 7];
    rotate(&mut nums, 3);
    println!("{:?}", nums);
}
```

```javascript
function reverseRange(nums, l, r) {
  while (l < r) {
    const tmp = nums[l];
    nums[l] = nums[r];
    nums[r] = tmp;
    l += 1;
    r -= 1;
  }
}

function rotate(nums, k) {
  const n = nums.length;
  if (n === 0) return;
  k %= n;
  if (k === 0) return;
  reverseRange(nums, 0, n - 1);
  reverseRange(nums, 0, k - 1);
  reverseRange(nums, k, n - 1);
}

const data = [1, 2, 3, 4, 5, 6, 7];
rotate(data, 3);
console.log(data);
```
