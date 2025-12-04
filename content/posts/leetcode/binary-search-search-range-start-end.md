---
title: "在排序数组中查找元素的起始和结束位置：一套二分模板搞定 Search Range"
date: 2025-12-04T11:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "有序数组", "边界问题", "面试高频题"]
description: "在有序数组中找到目标值的起始和结束位置（Search Range），要求 O(log n) 时间。本文用下界/上界二分模板彻底解决边界问题，并给出多语言代码和工程实践示例。"
keywords: ["Search Range", "first and last position", "二分查找", "有序数组", "lower_bound", "upper_bound"]
---

> **副标题 / 摘要**  
> 很多同学会写“找一个等于目标的二分”，但一到“找目标的起始和结束位置”就容易被边界条件卡住。本文用统一的下界 / 上界二分模板，彻底吃透 Search Range 类型问题，并给出多语言实现和工程场景示例。

- **预计阅读时长**：10~15 分钟  
- **适用场景标签**：`二分查找`、`日志区间查询`、`时间序列检索`  
- **SEO 关键词**：search range, first and last position, 二分查找边界, lower_bound, upper_bound  

---

## 目标读者与背景

**目标读者**

- 已经知道二分查找基本写法，但一到“找起始位置/结束位置”就容易出错的同学；
- 经常对日志、监控指标做时间区间检索的工程师；
- 准备面试时希望掌握一套可复用二分模板的开发者。

**背景 / 动机**

几乎所有互联网系统里都有“按时间排序的日志 / 事件 / 指标”：

- 比如按时间排序的访问日志；
- 按上报时间排序的监控数据点；
- 按 ID 排序的业务记录。

在这些有序数据上，最常见的操作之一就是：

> 找出“所有值等于 X 的记录”的区间 `[start, end]`。

这道 LeetCode 经典题「Search for a Range」正是这个需求的抽象版本。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定一个按非降序排序的整数数组 `nums` 和一个目标值 `target`。  
> 请在数组中找到目标值的**起始位置和结束位置**，以数组 `[start, end]` 形式返回。  
> 如果数组中不存在目标值，返回 `[-1, -1]`。  
> 要求时间复杂度为 **O(log n)**。

**输入**

- `nums`: 已按非降序排序的整数数组，长度为 `n`
- `target`: 要查找的目标整数

**输出**

- 长度为 2 的整数数组 `[start, end]`：
  - `start`: 目标在数组中第一次出现的下标
  - `end`: 目标在数组中最后一次出现的下标
  - 若不存在目标值，则为 `[-1, -1]`

### 示例 1

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8
```

- 目标值 `8` 出现的位置为下标 3 和 4；
- 起始位置 `start = 3`，结束位置 `end = 4`。

**输出**：

```text
[3, 4]
```

### 示例 2

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 6
```

数组中不存在 6，应该返回：

```text
[-1, -1]
```

### 示例 3

```text
nums   = []
target = 0
```

空数组中任何值都不存在，因此返回：

```text
[-1, -1]
```

---

## C — Concepts（核心思想）

### 1. 起始 / 结束位置如何建模？

目标值的起始位置 `start` 是：

> 数组中 **第一个 ≥ target 的位置**，并且该位置的值等于 `target`。

结束位置 `end` 是：

> 数组中 **最后一个 ≤ target 的位置**。  
> 也可以写成 `end = 第一个 > target 的位置 - 1`。

这引出两个经典概念：

- **下界（Lower Bound）**：第一个满足 `nums[mid] >= target` 的位置；
- **上界（Upper Bound）**：第一个满足 `nums[mid] > target` 的位置。

一旦这两个位置明确了：

```text
start = lower_bound(target)
end   = upper_bound(target) - 1
```

如果：

- `start == n`（越界）或 `nums[start] != target`，说明不存在目标值 → 返回 `[-1, -1]`。

### 2. 二分模板：下界 / 上界

我们使用统一的左闭右开区间 `[l, r)` 写法。

**下界（第一个 ≥ target 的位置）**

```text
l = 0, r = n
while (l < r):
    mid = (l + r) // 2
    if nums[mid] >= target:
        r = mid
    else:
        l = mid + 1
return l
```

**上界（第一个 > target 的位置）**

```text
l = 0, r = n
while (l < r):
    mid = (l + r) // 2
    if nums[mid] > target:
        r = mid
    else:
        l = mid + 1
return l
```

利用这两个模板，可以非常稳定地解决起点 / 终点 / 插入位置等一系列问题。

### 3. 算法类型与复杂度

- 算法类型：**二分查找（Binary Search）**
- 核心思想：在有序数组中，快速找到满足某种单调条件的边界位置；
- 时间复杂度：O(log n)；
- 空间复杂度：O(1)。

---

## 实践指南 / 实现步骤

1. **实现 `lower_bound(nums, target)`**
   - 返回第一个满足 `nums[i] >= target` 的下标；
   - 若不存在这样的元素，返回 `n`。

2. **实现 `upper_bound(nums, target)`**
   - 返回第一个满足 `nums[i] > target` 的下标；
   - 若不存在这样的元素，返回 `n`。

3. **用它们构造答案**

```text
start = lower_bound(nums, target)
end   = upper_bound(nums, target) - 1
if start == n or nums[start] != target:
    return [-1, -1]
else:
    return [start, end]
```

4. **检查边界情况**
   - 空数组：`n == 0` 时 `lower_bound` 和 `upper_bound` 都返回 0，但 `start == n` → 正确返回 `[-1, -1]`；
   - 所有元素都小于 target / 大于 target；
   - 只有一个元素的数组。

---

## E — Engineering（工程应用）

这道题在工程中对应的是各种**时间范围 / 值范围定位**需求。

### 场景 1：日志系统中查找某类请求的范围（Python）

**背景**  
假设你有一个按时间排序的日志数组 `timestamps`，记录了某类请求的发生时间（以秒为单位）。你想快速找出：

- 所有时间等于 `t` 的日志；
- 或者一个时间区间 `[start_t, end_t]` 的所有日志范围。

在简化场景下，可以先看“等于某个时间戳”的区间，就和本题几乎一样。

**示例代码**

```python
from typing import List, Tuple


def lower_bound(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l


def upper_bound(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l


def search_range(nums: List[int], target: int) -> Tuple[int, int]:
    start = lower_bound(nums, target)
    end = upper_bound(nums, target) - 1
    if start == len(nums) or nums[start] != target:
        return -1, -1
    return start, end


if __name__ == "__main__":
    print(search_range([5, 7, 7, 8, 8, 10], 8))  # (3, 4)
```

---

### 场景 2：后端服务中按 ID 区间批量处理（Go / Rust）

**背景**  
表数据往往按主键 ID 排序存储（如某些 KV 存储 / 内存索引）。要批量处理 ID 等于某个值（或某个区间）的所有记录时，可以先用二分找到范围。

下面的 Go 代码演示如何在有序数组中找到 target 的 `[start, end]` 区间。

```go
package main

import "fmt"

func lowerBound(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func upperBound(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func searchRange(nums []int, target int) (int, int) {
	start := lowerBound(nums, target)
	end := upperBound(nums, target) - 1
	if start == len(nums) || nums[start] != target {
		return -1, -1
	}
	return start, end
}

func main() {
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 8)) // 3 4
}
```

---

### 场景 3：前端配置中查找等值区间（JavaScript）

**背景**  
在前端，有时会把一系列版本号、时间戳或权重排序后放在数组里，用于做分流、灰度控制或配置生效时间段。

你可能需要快速找出「所有值等于 X 的项」在数组中的区间，用于 UI 高亮或后续 API 请求。

**示例代码**

```js
function lowerBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) r = mid;
    else l = mid + 1;
  }
  return l;
}

function upperBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l;
}

function searchRange(nums, target) {
  const start = lowerBound(nums, target);
  const end = upperBound(nums, target) - 1;
  if (start === nums.length || nums[start] !== target) {
    return [-1, -1];
  }
  return [start, end];
}

console.log(searchRange([5, 7, 7, 8, 8, 10], 8)); // [3, 4]
```

---

## R — Reflection（反思与深入）

### 1. 时间与空间复杂度

- 下界 / 上界二分各自是 O(log n)；
- Search Range 需要调用两次 → 总体仍然是 O(log n)；
- 空间复杂度 O(1)，只使用几个整型变量。

完全满足题目要求。

---

### 2. 替代方案与常见错误

**线性扫描方案**

- 直接从头到尾扫描数组，记录第一个和最后一个 target 出现的位置；
- 时间复杂度 O(n)，在 n 较大时比 O(log n) 慢；
- 更重要的是，不符合题目“必须 O(log n)”的要求。

**错误的二分写法 1：只找一个等于 target 的位置**

- 很多同学写的是“存在返回一个位置，不存在返回 -1”的标准二分；
- 然后从该位置向两边线性扩展找起点 / 终点，这样最坏情况下仍是 O(n)。

**错误的二分写法 2：边界条件混乱**

- 左闭右开 `[l, r)` 与左闭右闭 `[l, r]` 写法混用，容易产生死循环或 off-by-one；
- 在同一项目中建议统一一种写法（本文统一用 `[l, r)`）。

---

### 3. 为什么统一下界 / 上界模板更工程可行？

- **降低记忆负担**：  
  不用为每道题从头推边界，只需记住两个稳定的模板。

- **可复用性强**：  
  起始位置、结束位置、插入位置、计数范围等问题，都可以用这两个模板直接解决。

- **便于团队协作**：  
  当团队约定“二分一律用 lower_bound / upper_bound 模板”后，代码可读性、可维护性大幅提升。

---

## S — Summary（总结）

- 本题的本质是：在有序数组中找到目标值的**左边界（起始位置）**和**右边界（结束位置）**。
- 使用 `lower_bound`（第一个 ≥ target）和 `upper_bound`（第一个 > target）可以稳定地找到这两个边界。
- 二分查找的关键是保持区间不变式和收敛规则的一致性，推荐统一使用 `[l, r)` 模板。
- 相比线性扫描或错误的“先找到一个位置再向两边扩展”，下界 / 上界方案时间复杂度更优且更稳健。
- 这套模板不仅适用于本题，也适用于各种日志 / 指标 / 配置的区间查找问题。

---

## 参考与延伸阅读

- LeetCode 34. Find First and Last Position of Element in Sorted Array  
- C++ 标准库 `std::lower_bound` / `std::upper_bound` 文档  
- 二分查找专题题单：包含“搜索插入位置”“旋转数组最小值”“求平方根”等  
- 《算法导论》关于二分搜索与基于比较的排序章节

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

下面给出多语言版本的完整实现，统一采用“先求下界 / 再求上界”的风格。

### Python 实现

```python
from typing import List


def search_range(nums: List[int], target: int) -> List[int]:
    n = len(nums)

    def lower_bound(x: int) -> int:
        l, r = 0, n
        while l < r:
            mid = (l + r) // 2
            if nums[mid] >= x:
                r = mid
            else:
                l = mid + 1
        return l

    def upper_bound(x: int) -> int:
        l, r = 0, n
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > x:
                r = mid
            else:
                l = mid + 1
        return l

    start = lower_bound(target)
    end = upper_bound(target) - 1
    if start == n or start < 0 or nums[start] != target:
        return [-1, -1]
    return [start, end]


if __name__ == "__main__":
    print(search_range([5, 7, 7, 8, 8, 10], 8))  # [3, 4]
```

---

### C 实现

```c
#include <stdio.h>

int lower_bound_search(int *nums, int n, int target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

int upper_bound_search(int *nums, int n, int target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

void searchRange(int *nums, int n, int target, int *out0, int *out1) {
    int start = lower_bound_search(nums, n, target);
    int end = upper_bound_search(nums, n, target) - 1;
    if (start == n || n == 0 || nums[start] != target) {
        *out0 = -1;
        *out1 = -1;
    } else {
        *out0 = start;
        *out1 = end;
    }
}

int main(void) {
    int nums[] = {5, 7, 7, 8, 8, 10};
    int n = sizeof(nums) / sizeof(nums[0]);
    int start, end;
    searchRange(nums, n, 8, &start, &end);
    printf("[%d, %d]\n", start, end);  // [3, 4]
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> searchRange(vector<int> &nums, int target) {
    int n = (int)nums.size();
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target) r = mid;
        else l = mid + 1;
    }
    int start = l;

    l = 0; r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > target) r = mid;
        else l = mid + 1;
    }
    int end = l - 1;

    if (start == n || n == 0 || nums[start] != target) {
        return {-1, -1};
    }
    return {start, end};
}

int main() {
    vector<int> nums{5, 7, 7, 8, 8, 10};
    auto res = searchRange(nums, 8);
    cout << "[" << res[0] << ", " << res[1] << "]\n";  // [3, 4]
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func searchRange(nums []int, target int) []int {
	n := len(nums)
	l, r := 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	start := l

	l, r = 0, n
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	end := l - 1

	if start == n || n == 0 || nums[start] != target {
		return []int{-1, -1}
	}
	return []int{start, end}
}

func main() {
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 8)) // [3 4]
}
```

---

### Rust 实现

```rust
fn search_range(nums: &[i32], target: i32) -> (i32, i32) {
    let n = nums.len();
    let mut l = 0usize;
    let mut r = n;
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] >= target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    let start = l;

    l = 0;
    r = n;
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    let end = if l == 0 { 0 } else { l - 1 };

    if start == n || n == 0 || nums[start] != target {
        (-1, -1)
    } else {
        (start as i32, end as i32)
    }
}

fn main() {
    let nums = vec![5, 7, 7, 8, 8, 10];
    let (start, end) = search_range(&nums, 8);
    println!("[{}, {}]", start, end); // [3, 4]
}
```

---

### JavaScript 实现

```js
function searchRange(nums, target) {
  const n = nums.length;
  let l = 0, r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) r = mid;
    else l = mid + 1;
  }
  const start = l;

  l = 0; r = n;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > target) r = mid;
    else l = mid + 1;
  }
  const end = l - 1;

  if (start === n || n === 0 || nums[start] !== target) {
    return [-1, -1];
  }
  return [start, end];
}

console.log(searchRange([5, 7, 7, 8, 8, 10], 8)); // [3, 4]
```

---

## 行动号召（CTA）

- 把 `lower_bound` / `upper_bound` 模板抄进你的个人算法模板库，并自己实现一遍。
- 尝试用今天的模板重写「搜索插入位置」「最大正负数计数」等二分题，体会统一模板的威力。
- 在你自己的业务代码里，找一处“在有序数组上做范围查找”的逻辑，看看是否可以用这套二分模板让代码更简洁、可维护。 

