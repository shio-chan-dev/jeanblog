---
title: "最大正负数计数：用二分在排序数组中统计正整数和负整数数量的最大值"
date: 2025-12-04T11:40:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "有序数组", "计数问题", "边界查找"]
description: "在排序数组中统计正整数和负整数数量的最大值（Maximum Count of Positive & Negative Integers）。本文用上下界二分模板一次性找到负数结束位置和正数起始位置，并给出多语言实现与工程应用示例。"
keywords: ["Maximum Count of Positive and Negative Integers", "二分查找", "有序数组计数", "上下界查找"]
---

> **副标题 / 摘要**  
> 给定一个有序整数数组，如何在 O(log n) 时间内分别统计负数和正数的个数，并返回两者中的较大值？这道「Maximum Count of Positive & Negative Integers」正是边界型二分的练习题。本文用上下界二分一次性搞定负数结束和正数起点。

- **预计阅读时长**：8~10 分钟  
- **适用场景标签**：`二分查找`、`边界计数`、`排序数组`  
- **SEO 关键词**：maximum count, positive negative, 二分统计, 上下界, 有序数组计数  

---

## 目标读者与背景

**目标读者**

- 已经会写 basic binary search，希望进阶到“计数型二分”的同学；
- 在工程中有基于排序数据做区间计数需求的工程师；
- 准备面试，想把二分查找的上下界技巧练熟的开发者。

**背景 / 动机**

在各种日志 / 指标 / 数据分析场景中，我们经常会对**有序数据**做计数：

- 比如统计小于 0 的条目数量；
- 统计大于某个阈值的条目数量；
- 找到“负数段结束”和“正数段开始”的位置。

这道 LeetCode 题「Maximum Count of Positive & Negative Integers」是这类需求的简化模型，非常适合作为上下界二分的练习。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定一个按非降序排序的整数数组 `nums`。  
> 数组中可能包含负数、0 和正数。  
> 定义：  
> - `countNeg` = 数组中小于 0 的元素数量；  
> - `countPos` = 数组中大于 0 的元素数量。  
> 请返回 `max(countNeg, countPos)`。

**输入**

- `nums`: 已排序的整数数组，长度为 `n`，元素可以是负数、0 或正数。

**输出**

- 整数：`max(countNeg, countPos)`。

### 示例 1

```text
nums = [-3, -2, -1, 0, 0, 1, 2]
```

- 负数有 3 个：`[-3, -2, -1]`
- 正数有 2 个：`[1, 2]`

最大值为 `3`。

**输出**：`3`

### 示例 2

```text
nums = [-2, -1, -1, 1, 2, 3]
```

- 负数有 3 个：`[-2, -1, -1]`
- 正数也有 3 个：`[1, 2, 3]`

最大值为 `3`。

**输出**：`3`

### 示例 3

```text
nums = [0, 0, 0]
```

- 负数有 0 个；
- 正数有 0 个；

最大值为 `0`。

**输出**：`0`

---

## C — Concepts（核心思想）

### 1. 用边界点表示计数

数组是按非降序排序的，我们知道：

- 所有负数（< 0）一定出现在左侧；
- 所有正数（> 0）一定出现在右侧；
- 中间可能有连续的一段 0。

把数组大致画一下：

```text
[ 负数 ... 负数 ][ 0 ... 0 ][ 正数 ... 正数 ]
        ^              ^
       分界1          分界2
```

我们只要找到两个分界点：

1. **第一个 ≥ 0 的位置**：
   - 这个下标之前全是 < 0 的负数；
   - 负数数量 = 这个位置的下标值。
2. **第一个 > 0 的位置**：
   - 这个下标开始到末尾全是 > 0 的正数；
   - 正数数量 = `n - 该下标`。

这两个位置本质就是：

- `lower_bound(nums, 0)`：第一个 `>= 0` 的位置；
- `upper_bound(nums, 0)`：第一个 `> 0` 的位置。

于是：

```text
countNeg = lower_bound(nums, 0)
countPos = n - upper_bound(nums, 0)
答案 = max(countNeg, countPos)
```

### 2. 下界 / 上界二分模板回顾

**下界（≥ target 的第一个位置）**

```text
int lower_bound(nums, target):
    l = 0, r = n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l
```

**上界（> target 的第一个位置）**

```text
int upper_bound(nums, target):
    l = 0, r = n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l
```

### 3. 算法类型与复杂度

- 算法类型：**上 / 下界二分查找**
- 核心操作：在有序数组中找到满足条件的边界位置，然后根据下标算数量；
- 时间复杂度：O(log n)
- 空间复杂度：O(1)

---

## 实践指南 / 实现步骤

1. **实现 `lower_bound(nums, 0)`**
   - 返回第一个 `nums[i] >= 0` 的下标；
   - 若所有元素都小于 0，则返回 `n`，此时 `countNeg = n`。

2. **实现 `upper_bound(nums, 0)`**
   - 返回第一个 `nums[i] > 0` 的下标；
   - 若没有正数，则返回 `n`，此时 `countPos = 0`。

3. **计算正负数量**

```text
countNeg = index_first_ge_0
countPos = n - index_first_gt_0
ans = max(countNeg, countPos)
```

4. **检查边界情况**

- 数组全为负数：`lower_bound(0) == n`，`upper_bound(0) == n`，`countNeg = n`，`countPos = 0`；
- 数组全为正数：`lower_bound(0) == 0`，`upper_bound(0) == 0`，`countNeg = 0`，`countPos = n`；
- 数组全为 0：`lower_bound(0) == 0`，`upper_bound(0) == n`，两个计数都为 0。

---

## E — Engineering（工程应用）

这种“负数/正数计数”的模式，在工程里对应的是各种「阈值计数」。

### 场景 1：监控指标偏差统计（Python）

**背景**  
假设你有一组按照大小排序的偏差值（实际值减期望值）：

- 偏差 < 0 表示低于预期；
- 偏差 > 0 表示高于预期；
- 偏差 = 0 表示刚好。

你想知道**某个时间段内，“低于预期”的次数和“高于预期”的次数哪个更多**。

**示例代码**

```python
from typing import List


def maximum_count(nums: List[int]) -> int:
    n = len(nums)

    # 第一个 >= 0 的下标 => 负数个数
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= 0:
            r = mid
        else:
            l = mid + 1
    count_neg = l

    # 第一个 > 0 的下标 => 正数个数 = n - 该下标
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > 0:
            r = mid
        else:
            l = mid + 1
    count_pos = n - l

    return max(count_neg, count_pos)


if __name__ == "__main__":
    print(maximum_count([-3, -2, -1, 0, 0, 1, 2]))  # 3
```

---

### 场景 2：风控得分的正负分布分析（Go）

**背景**  
风控模型输出一组得分（可以为负、0、正），你希望：

- 快速统计“负向得分样本”和“正向得分样本”的数量；
- 看哪个「风险方向」的样本更多，以辅助调参。

如果你把得分排序后，就可以用本题的二分方法快速计算。

**示例代码（Go）**

```go
package main

import (
	"fmt"
	"sort"
)

func maximumCount(nums []int) int {
	n := len(nums)
	// 第一个 >= 0 的位置
	l, r := 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= 0 {
			r = mid
		} else {
			l = mid + 1
		}
	}
	countNeg := l

	// 第一个 > 0 的位置
	l, r = 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > 0 {
			r = mid
		} else {
			l = mid + 1
		}
	}
	countPos := n - l

	if countNeg > countPos {
		return countNeg
	}
	return countPos
}

func main() {
	nums := []int{-3, -2, -1, 0, 0, 1, 2}
	sort.Ints(nums) // 题目保证已排序，这里演示一下
	fmt.Println(maximumCount(nums)) // 3
}
```

---

### 场景 3：前端评分分布可视化（JavaScript）

**背景**  
前端拿到一组用户打分偏差（已排序），用来做简单的图表，比如：

- 正向反馈 vs 负向反馈数量比较；
- 显示“正向反馈更多”还是“负向反馈更多”。

可以直接在前端用二分统计出两边的数量，然后渲染图表。

**示例代码**

```js
function maximumCount(nums) {
  const n = nums.length;

  // 第一个 >= 0
  let l = 0, r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= 0) r = mid;
    else l = mid + 1;
  }
  const countNeg = l;

  // 第一个 > 0
  l = 0; r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > 0) r = mid;
    else l = mid + 1;
  }
  const countPos = n - l;

  return Math.max(countNeg, countPos);
}

console.log(maximumCount([-3, -2, -1, 0, 0, 1, 2])); // 3
```

---

## R — Reflection（反思与深入）

### 1. 复杂度分析

- 两次二分查找，每次 O(log n)；
- 总时间复杂度：**O(log n)**；
- 空间复杂度：**O(1)**。

相比直接线性扫描 O(n)，二分在大规模数据上优势明显（特别是频繁查询时）。

---

### 2. 替代方案与常见错误

**线性扫描**

- 遍历一遍数组，统计 `< 0` 和 `> 0` 的数量；
- 时间 O(n)，逻辑简单，但没有利用“已排序”这个重要信息；
- 在 n 不大时是完全可行的，但本题更鼓励用二分练手。

**常见错误 1：把 0 统计进正数 / 负数**

- 题目明确 `countNeg` 只统计 `< 0`，`countPos` 只统计 `> 0`；
- 有些实现会错误地把 0 归到某一边。

**常见错误 2：上界 / 下界条件写错**

- 下界应使用 `>= target`，这里 target = 0；
- 上界应使用 `> target`，也是 target = 0。

**常见错误 3：忽略数组全负 / 全正 / 全零的情况**

- 若不仔细处理 `l == n` 等边界，可能误算数量或造成越界访问。

---

### 3. 与其他二分题的关系

本题可以看成是前面若干二分题（Search Range、Next Greatest Letter 等）的一个综合练习：

- 用 `lower_bound(0)` 找负数结束位置；
- 用 `upper_bound(0)` 找正数开始位置；
- 再用简单算术把下标转换为数量。

掌握本题后，可更自然地想到用二分来做 “≤ / ≥ / < / >” 条件的计数，而不是只用线性扫描。

---

## S — Summary（总结）

- 本题的本质是：在有序数组中找到「负数段」和「正数段」的边界，然后分别计算两边的长度。
- 使用下界 / 上界二分，可以在 O(log n) 时间内找到第一个 `>= 0` 和第一个 `> 0` 的位置。
- 负数数量 = 第一个 `>= 0` 的位置下标，正数数量 = `n - 第一个 > 0 的位置下标`。
- 相比线性扫描，二分方案充分利用了数组「已排序」的前提，更具工程推广价值。
- 该技巧可广泛迁移到各种“按阈值对已排序数组做计数”的场景，如偏差分析、风险得分统计等。

---

## 参考与延伸阅读

- LeetCode 2529. Maximum Count of Positive Integer and Negative Integer  
- 二分上下界相关题目：Search Insert Position、Search Range、Next Greatest Letter  
- C++ `std::lower_bound` / `std::upper_bound` 文档与用法示例  
- 《算法导论》关于有序结构上的搜索与统计章节

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

### Python 实现

```python
from typing import List


def maximum_count(nums: List[int]) -> int:
    n = len(nums)

    # 第一个 >= 0 的位置
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= 0:
            r = mid
        else:
            l = mid + 1
    count_neg = l

    # 第一个 > 0 的位置
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > 0:
            r = mid
        else:
            l = mid + 1
    count_pos = n - l

    return max(count_neg, count_pos)


if __name__ == "__main__":
    print(maximum_count([-3, -2, -1, 0, 0, 1, 2]))  # 3
```

---

### C 实现

```c
#include <stdio.h>

int maximumCount(int *nums, int numsSize) {
    int n = numsSize;
    int l = 0, r = n;
    // 第一个 >= 0
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= 0) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    int countNeg = l;

    // 第一个 > 0
    l = 0; r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > 0) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    int countPos = n - l;

    return (countNeg > countPos) ? countNeg : countPos;
}

int main(void) {
    int nums[] = {-3, -2, -1, 0, 0, 1, 2};
    int n = sizeof(nums) / sizeof(nums[0]);
    printf("%d\n", maximumCount(nums, n)); // 3
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int maximumCount(vector<int> &nums) {
    int n = (int)nums.size();

    // 第一个 >= 0
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= 0) r = mid;
        else l = mid + 1;
    }
    int countNeg = l;

    // 第一个 > 0
    l = 0; r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > 0) r = mid;
        else l = mid + 1;
    }
    int countPos = n - l;

    return max(countNeg, countPos);
}

int main() {
    vector<int> nums{-3, -2, -1, 0, 0, 1, 2};
    cout << maximumCount(nums) << endl; // 3
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func maximumCount(nums []int) int {
	n := len(nums)

	// 第一个 >= 0
	l, r := 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= 0 {
			r = mid
		} else {
			l = mid + 1
		}
	}
	countNeg := l

	// 第一个 > 0
	l, r = 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > 0 {
			r = mid
		} else {
			l = mid + 1
		}
	}
	countPos := n - l

	if countNeg > countPos {
		return countNeg
	}
	return countPos
}

func main() {
	fmt.Println(maximumCount([]int{-3, -2, -1, 0, 0, 1, 2})) // 3
}
```

---

### Rust 实现

```rust
fn maximum_count(nums: &[i32]) -> i32 {
    let n = nums.len();

    // 第一个 >= 0
    let mut l: usize = 0;
    let mut r: usize = n;
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] >= 0 {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    let count_neg = l as i32;

    // 第一个 > 0
    l = 0;
    r = n;
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] > 0 {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    let count_pos = (n - l) as i32;

    count_neg.max(count_pos)
}

fn main() {
    let nums = vec![-3, -2, -1, 0, 0, 1, 2];
    println!("{}", maximum_count(&nums)); // 3
}
```

---

### JavaScript 实现

```js
function maximumCount(nums) {
  const n = nums.length;

  // 第一个 >= 0
  let l = 0, r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= 0) r = mid;
    else l = mid + 1;
  }
  const countNeg = l;

  // 第一个 > 0
  l = 0; r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > 0) r = mid;
    else l = mid + 1;
  }
  const countPos = n - l;

  return Math.max(countNeg, countPos);
}

console.log(maximumCount([-3, -2, -1, 0, 0, 1, 2])); // 3
```

---

## 行动号召（CTA）

- 把 `lower_bound`/`upper_bound` 模板加进你的二分查找笔记，并练习用它们做「计数」而不仅仅是「查找」。
- 尝试为“统计大于某阈值的元素个数”“统计在区间 [L, R] 内的元素个数”等问题设计类似的二分方案。
- 回顾你项目中的一些统计逻辑，如果输入数据已经是有序的，考虑用二分加速计数。 

