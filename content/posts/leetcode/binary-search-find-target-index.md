---
title: "经典 Binary Search：在排序数组中查找目标值索引的统一模板"
date: 2025-12-04T11:20:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "有序数组", "基础算法", "面试高频题"]
description: "在有序数组中查找目标值：存在返回下标，不存在返回 -1。本文以 LeetCode 704 为例，用统一的二分模板讲清楚边界处理，并给出多语言实现和工程实践示例。"
keywords: ["Binary Search", "LeetCode 704", "二分查找", "有序数组查找", "基础算法模板"]
---

> **副标题 / 摘要**  
> 二分查找是所有算法面试和工程系统中的“必修课”。本文以最基础的「在有序数组中查找目标值」为例，从题意、边界到统一模板，系统整理 Binary Search 的写法，并配套多语言实现，帮助你彻底告别二分边界恐惧症。

- **预计阅读时长**：8~10 分钟  
- **适用场景标签**：`二分查找基础`、`数组检索`、`性能优化`  
- **SEO 关键词**：binary search, LeetCode 704, 二分查找模板, 有序数组目标索引  

---

## 目标读者与背景

**目标读者**

- 刚开始系统刷题、希望夯实基础二分查找的同学；
- 在工程中经常需要在有序列表中查找、定位数据的后端 / 前端工程师；
- 曾经被二分查找的边界条件困扰、希望形成统一模板的开发者。

**为什么这题值得认真学？**

- 它是 LeetCode 704：Binary Search，二分查找的最基础版本；
- 几乎所有高级二分题（Search Range、插入位置、求上下界）都以此为内核；
- 大量工程场景（有序列表查找、策略表、时间线等）都可以套用这个模板。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定一个按非降序排序的整数数组 `nums` 和一个整数 `target`。  
> 请你在数组中查找 `target`，如果存在，则返回其下标；否则，返回 `-1`。  
> 要求算法的时间复杂度为 **O(log n)**。

**输入**

- `nums`: 已排序（非降序）的整数数组，长度为 `n`
- `target`: 要查找的整数

**输出**

- 若 `target` 存在于 `nums` 中，则返回其下标；
- 否则返回 `-1`。

### 示例 1

```text
nums   = [-1, 0, 3, 5, 9, 12]
target = 9
```

数组中存在 9，且在下标 4：

**输出**：`4`

### 示例 2

```text
nums   = [-1, 0, 3, 5, 9, 12]
target = 2
```

数组中不存在 2，应该返回：

**输出**：`-1`

---

## C — Concepts（核心思想）

### 1. 为什么可以用二分查找？

使用二分查找需要满足两个关键条件：

1. **数据有序**（单调）：  
   题目明确说 `nums` 已按非降序排序。
2. **目标是定位某个值 / 边界**：  
   要么找到 `target`，要么确认它不存在。

在有序数组上做查找，用二分查找可以：

- 把搜索区间每次缩小一半，达到 O(log n) 的复杂度；
- 避免 O(n) 线性扫描带来的性能问题。

### 2. 经典二分查找模板（左闭右闭区间）

为了贴近很多语言标准库和常见写法，这里用左闭右闭 `[l, r]` 模板：

```text
初始化：l = 0, r = n - 1
循环条件：l <= r
mid = l + (r - l) // 2
比较 nums[mid] 与 target：
  - 若相等：返回 mid
  - 若 nums[mid] < target：目标在右半边 → l = mid + 1
  - 若 nums[mid] > target：目标在左半边 → r = mid - 1
循环结束：没找到，返回 -1
```

这一模板是最常见的「找某个等于目标的点」的二分写法。

### 3. 另一种选择：左闭右开（下界）模板

你也可以使用左闭右开 `[l, r)` 模板 + `lower_bound`：

1. 找到第一个满足 `nums[i] >= target` 的位置 `l`；
2. 若 `l < n` 且 `nums[l] == target` 则返回 `l`，否则 `-1`。

这与上一节的 Search Insert Position 一脉相承，适合统一为一个模板。

本篇代码中，我们用更直观的 `[l, r]` 版本，方便入门；  
后续可以根据需要切换到 `[l, r)` 风格。

### 4. 时间与空间复杂度

- 时间复杂度：O(log n)，每一步都把搜索区间缩小一半；
- 空间复杂度：O(1)，只用到几个整型变量。

---

## 实践指南 / 实现步骤

1. **确认边界给定方式**  
   - 决定使用 `[l, r]` 还是 `[l, r)`；  
   - 本文代码示例多采用 `[l, r]`，更符合很多教科书写法。

2. **写出循环不变式**

```text
while l <= r:
    mid = l + (r - l) // 2
    ...
```

3. **处理三种比较情况**

```text
if nums[mid] == target:  返回 mid
if nums[mid] < target:   去右半边 → l = mid + 1
if nums[mid] > target:   去左半边 → r = mid - 1
```

4. **退出循环**

- 当 `l > r` 时，说明整个数组已经被搜索完毕，未找到 `target`；
- 返回 `-1`。

5. **验证边界**

- `nums` 为空时：`r = -1`，循环不会进入，直接返回 `-1`；
- 目标在最左 / 最右位置的情况；
- 数组只包含一个元素的情况。

---

## E — Engineering（工程应用）

二分查找不仅是刷题常客，更是工程系统中高频使用的“基础设施”。

### 场景 1：配置 / 策略表查找（Python）

**背景**  
假设你维护了一张按 key 排序的配置表（比如限流阈值、定价策略等），希望在内存中快速找到某个 key 对应的配置。

虽然实际场景中可能会用哈希表，但在一些「范围型策略」里，使用有序数组 + 二分更适合做范围定位。

**示例代码**

```python
from typing import List


def binary_search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


if __name__ == "__main__":
    print(binary_search([-1, 0, 3, 5, 9, 12], 9))  # 4
```

---

### 场景 2：后端日志 / 指标查找（Go）

**背景**  
在一些内存索引结构里，你可能会维护一个**按时间戳排序的数组**，需要快速判断某个时间点是否有数据。

**示例代码（Go）**

```go
package main

import "fmt"

func binarySearch(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := l + (r-l)/2
		if nums[mid] == target {
			return mid
		}
		if nums[mid] < target {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return -1
}

func main() {
	fmt.Println(binarySearch([]int{-1, 0, 3, 5, 9, 12}, 9)) // 4
	fmt.Println(binarySearch([]int{-1, 0, 3, 5, 9, 12}, 2)) // -1
}
```

---

### 场景 3：前端版本 / 功能开关列表查找（JavaScript）

**背景**  
在前端，你可能会维护一个按版本排序的数组，用于决定某个版本是否已经支持某项特性：

```text
supportedVersions = [1, 2, 4, 6, 8]
```

需要快速判断 `currentVersion` 是否已经在列表中。

**示例代码**

```js
function binarySearch(nums, target) {
  let l = 0, r = nums.length - 1;
  while (l <= r) {
    const mid = (l + r) >> 1;
    if (nums[mid] === target) return mid;
    if (nums[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}

console.log(binarySearch([-1, 0, 3, 5, 9, 12], 9)); // 4
console.log(binarySearch([-1, 0, 3, 5, 9, 12], 2)); // -1
```

---

## R — Reflection（反思与深入）

### 1. 时间与空间复杂度

- 每轮循环将搜索区间大小缩小一半；
- 需要大约 `log₂(n)` 轮；
- 时间复杂度：**O(log n)**；
- 空间复杂度：**O(1)**。

相比线性扫描 O(n)，当 `n` 很大（如 1e5、1e6）时，二分查找优势非常明显。

---

### 2. 常见错误与陷阱

1. **死循环**
   - 原因多为 `mid` 计算或左右边界更新写错；
   - 比如在 `[l, r)` 模式下错误地写成 `l = mid` 而不是 `l = mid + 1`；

2. **越界访问**
   - `nums[mid]` 时 mid 计算错误或上下界调整不当；
   - `r` 初始化为 `len(nums)` 却仍然以 `[l, r]`（左闭右闭）处理。

3. **区间风格混用**
   - 一会儿用 `[l, r]`，一会儿用 `[l, r)`，条件也在 `<=` 和 `<` 间切换；
   - 建议在一个项目内统一一种风格，常见的两种：
     - `[l, r]` + `while l <= r`
     - `[l, r)` + `while l < r`

4. **未正确处理空数组**
   - `nums` 为空时，`r = -1`，需要保证循环不会进入且不会访问越界。

---

### 3. 与其他二分变种的关系

- 本题：目标是**找到任意一个等于 target 的位置**；
- Search Insert Position：目标是**找到第一个 ≥ target 的位置**；
- Search Range：目标是**找到 target 的起始位置和结束位置**；
- Maximum Count of Positive/Negative：通过二分找出“负数结束 / 正数开始”的边界。

一旦理解本题的二分逻辑，再在此基础上做小改动，就能自然延展到各种「边界型二分」题目。

---

## S — Summary（总结）

- 本题是二分查找中最基础的一类：在有序数组中查找等于目标值的索引。
- 使用左闭右闭 `[l, r]` 模板，可以非常清晰地写出 O(log n) 的解法。
- 二分查找的核心在于：维护好搜索区间、正确更新左右边界，并保证循环收敛。
- 统一的 Binary Search 模板，不仅对刷题有帮助，在工程中也广泛适用。
- 掌握本题后，你可以自然过渡到「插入位置」「起始/结束位置」「上下界」等进阶题目。

---

## 参考与延伸阅读

- LeetCode 704. Binary Search（原题）  
- LeetCode 35, 34 等二分变种题  
- 各语言标准库搜索函数：`bisect`（Python）、`std::binary_search` / `lower_bound`（C++）、`sort.Search`（Go）  
- 《算法导论》第二部分关于排序与查找的内容

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

### Python 实现

```python
from typing import List


def binary_search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


if __name__ == "__main__":
    print(binary_search([-1, 0, 3, 5, 9, 12], 9))  # 4
```

---

### C 实现

```c
#include <stdio.h>

int binarySearch(int *nums, int numsSize, int target) {
    int l = 0, r = numsSize - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[mid] < target) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return -1;
}

int main(void) {
    int nums[] = {-1, 0, 3, 5, 9, 12};
    int n = sizeof(nums) / sizeof(nums[0]);
    printf("%d\n", binarySearch(nums, n, 9)); // 4
    printf("%d\n", binarySearch(nums, n, 2)); // -1
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int binarySearch(const vector<int> &nums, int target) {
    int l = 0, r = (int)nums.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1;
}

int main() {
    vector<int> nums{-1, 0, 3, 5, 9, 12};
    cout << binarySearch(nums, 9) << endl; // 4
    cout << binarySearch(nums, 2) << endl; // -1
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func binarySearch(nums []int, target int) int {
	l, r := 0, len(nums)-1
	for l <= r {
		mid := l + (r-l)/2
		if nums[mid] == target {
			return mid
		}
		if nums[mid] < target {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return -1
}

func main() {
	fmt.Println(binarySearch([]int{-1, 0, 3, 5, 9, 12}, 9)) // 4
	fmt.Println(binarySearch([]int{-1, 0, 3, 5, 9, 12}, 2)) // -1
}
```

---

### Rust 实现

```rust
fn binary_search(nums: &[i32], target: i32) -> i32 {
    if nums.is_empty() {
        return -1;
    }
    let mut l: i32 = 0;
    let mut r: i32 = nums.len() as i32 - 1;
    while l <= r {
        let mid = l + (r - l) / 2;
        let value = nums[mid as usize];
        if value == target {
            return mid;
        }
        if value < target {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    -1
}

fn main() {
    let nums = vec![-1, 0, 3, 5, 9, 12];
    println!("{}", binary_search(&nums, 9)); // 4
    println!("{}", binary_search(&nums, 2)); // -1
}
```

---

### JavaScript 实现

```js
function binarySearch(nums, target) {
  let l = 0, r = nums.length - 1;
  while (l <= r) {
    const mid = (l + r) >> 1;
    if (nums[mid] === target) return mid;
    if (nums[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}

console.log(binarySearch([-1, 0, 3, 5, 9, 12], 9)); // 4
console.log(binarySearch([-1, 0, 3, 5, 9, 12], 2)); // -1
```

---

## 行动号召（CTA）

- 把本文的二分查找模板抄进你的笔记或代码仓库，尝试不看代码自己写一遍。
- 用同一个模板实现「Search Insert Position」和「Search Range」，体会它们之间的联系。
- 在你的工程代码中，找一个使用线性搜索的有序列表逻辑，试着用二分查找替换，看能否提升性能或简化代码。 

