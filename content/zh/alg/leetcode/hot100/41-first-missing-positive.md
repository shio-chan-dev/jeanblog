---
title: "Hot100：缺失的第一个正数（First Missing Positive）原地索引定位 ACERS 解析"
date: 2026-01-24T12:42:31+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "原地哈希", "索引定位", "置换", "LeetCode 41"]
description: "用原地索引定位在 O(n) 时间、O(1) 额外空间找到缺失的第一个正数，含工程场景与多语言实现。"
keywords: ["First Missing Positive", "缺失的第一个正数", "原地哈希", "索引映射", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 缺失的第一个正数是经典的“原地哈希/索引定位”题：把值放回它应该在的位置，再线性扫描即可找到答案。本文按 ACERS 拆解思路、工程应用与多语言实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`数组`、`原地哈希`  
- **SEO 关键词**：First Missing Positive, 缺失的第一个正数, 原地哈希, 索引映射, O(n)  
- **元描述**：O(n) 时间、O(1) 额外空间的原地索引定位解法，含工程场景与多语言代码。  

---

## 目标读者

- 正在刷 Hot100 的学习者  
- 想掌握“原地索引定位”模板的中级开发者  
- 需要在原数组内做高效重排与定位的工程师

## 背景 / 动机

“找最小缺失正数”本质是一个**定位问题**：  
如果能把值 `x` 放在索引 `x-1` 上，那么答案就是第一个不匹配的位置。  
题目还要求 O(n) 时间和 O(1) 额外空间，逼迫我们放弃排序与哈希表，  
转而使用原地置换的技巧。

## 核心概念

| 概念 | 含义 | 作用 |
| --- | --- | --- |
| 原地哈希 | 用数组下标充当哈希桶 | O(1) 额外空间 |
| 索引定位 | 值 `x` 应放到 `x-1` | 构造可扫描的结构 |
| 置换交换 | 不断交换直到就位 | 线性时间完成 |

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个未排序的整数数组 `nums`，找出其中**没有出现的最小正整数**。  
请实现 **O(n)** 时间复杂度并且只使用 **常数级别**额外空间的解决方案。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 未排序整数数组 |
| 返回 | int | 最小缺失的正整数 |

### 示例 1（官方）

```text
输入: nums = [1,2,0]
输出: 3
```

### 示例 2（官方）

```text
输入: nums = [3,4,-1,1]
输出: 2
```

### 思路概览

1. 对每个位置 i，把 `nums[i]` 放到它应该去的位置 `nums[i]-1`。  
2. 完成“就位”后，从左到右找到第一个 `nums[i] != i+1` 的位置。  
3. 该位置对应的正整数 `i+1` 即为答案；若全部匹配则答案为 `n+1`。

---

## C — Concepts（核心思想）

### 关键模型

```
值 x 应该放在索引 x-1
```

### 方法归类

- 原地哈希（Index-as-Hash）  
- 数组置换 / 位置归位  
- 线性扫描验证

### 不变量

当置换结束时：  
如果 `nums[i] == i+1`，说明正整数 `i+1` 存在；  
第一个不匹配的 `i`，就是最小缺失正整数的位置。

---

## 实践指南 / 步骤

1. 遍历数组下标 i：  
   - 若 `nums[i]` 在 `[1, n]` 且不在正确位置，则与目标位置交换  
2. 交换后不前进 i，继续处理新值，直到当前位置稳定  
3. 再次扫描数组，找到第一个 `nums[i] != i+1`  
4. 若全部匹配，则返回 `n+1`

运行方式示例：

```bash
python3 first_missing_positive.py
```

## 可运行示例（Python）

```python
from typing import List


def first_missing_positive(nums: List[int]) -> int:
    n = len(nums)
    i = 0
    while i < n:
        v = nums[i]
        if 1 <= v <= n and nums[v - 1] != v:
            nums[i], nums[v - 1] = nums[v - 1], nums[i]
        else:
            i += 1
    for i, v in enumerate(nums):
        if v != i + 1:
            return i + 1
    return n + 1


if __name__ == "__main__":
    print(first_missing_positive([1, 2, 0]))
    print(first_missing_positive([3, 4, -1, 1]))
```

---

## 解释与原理（为什么这么做）

把数组当作一个“索引哈希表”：  
当数值 `x` 在 1..n 范围内时，它应该占据索引 `x-1`。  
通过交换把元素归位后，数组的第一个空洞（不匹配）位置  
就是“最小缺失正整数”。  
每次交换至少会让一个元素就位，因此总体复杂度仍是 O(n)。

---

## E — Engineering（工程应用）

### 场景 1：批量编号补位（Python，数据分析）

**背景**：分析批次中需要寻找最小未占用的正整数编号。  
**为什么适用**：原地算法可在大数组上快速定位缺失编号。

```python
def next_id(nums):
    n = len(nums)
    i = 0
    while i < n:
        v = nums[i]
        if 1 <= v <= n and nums[v - 1] != v:
            nums[i], nums[v - 1] = nums[v - 1], nums[i]
        else:
            i += 1
    for i, v in enumerate(nums):
        if v != i + 1:
            return i + 1
    return n + 1


print(next_id([2, 1, 4, 6, 3]))
```

### 场景 2：分片编号校验（Go，后台服务）

**背景**：配置文件中记录了活跃分片编号，需要快速找到缺失的最小正号。  
**为什么适用**：O(n) 扫描 + 原地置换，适合配置校验任务。

```go
package main

import "fmt"

func firstMissingPositive(nums []int) int {
	n := len(nums)
	i := 0
	for i < n {
		v := nums[i]
		if v >= 1 && v <= n && nums[v-1] != v {
			nums[i], nums[v-1] = nums[v-1], nums[i]
		} else {
			i++
		}
	}
	for i, v := range nums {
		if v != i+1 {
			return i + 1
		}
	}
	return n + 1
}

func main() {
	fmt.Println(firstMissingPositive([]int{3, 4, -1, 1}))
}
```

### 场景 3：前端任务序号补位（JavaScript，前端）

**背景**：前端列表需要为新任务分配最小未用序号。  
**为什么适用**：无需额外存储即可得到最小空位。

```javascript
function firstMissingPositive(nums) {
  const n = nums.length;
  let i = 0;
  while (i < n) {
    const v = nums[i];
    if (v >= 1 && v <= n && nums[v - 1] !== v) {
      const tmp = nums[v - 1];
      nums[v - 1] = nums[i];
      nums[i] = tmp;
    } else {
      i += 1;
    }
  }
  for (let idx = 0; idx < n; idx += 1) {
    if (nums[idx] !== idx + 1) {
      return idx + 1;
    }
  }
  return n + 1;
}

console.log(firstMissingPositive([1, 2, 0]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
- **空间复杂度**：O(1) 额外空间

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 暴力枚举 | 每个正数逐一验证 | O(n^2) | 大数组不可用 |
| 排序 | 先排序再扫描 | O(n log n) | 不满足线性时间 |
| 哈希集合 | 记录出现的正数 | O(n) 时间 / O(n) 空间 | 违背 O(1) 空间 |
| 原地索引定位 | 置换到位再扫描 | O(n) / O(1) | 当前最优 |

### 为什么当前方法最优 / 最工程可行

在“线性时间 + 常数空间”的约束下，  
原地索引定位几乎是唯一可行的通用解法，  
实现简单、可复用且适合大规模数据。

---

## 常见问题与注意事项

1. **为什么会死循环？**  
   交换时必须检查 `nums[v-1] != v`，避免重复值造成无限交换。

2. **负数和 0 怎么处理？**  
   只关心 1..n 的值，其他直接跳过即可。

3. **数组长度为 1 怎么办？**  
   若为 `[1]` 返回 2，否则返回 1。

---

## 最佳实践与建议

- 使用 `while` 交换直到当前位置稳定  
- 边界判断必须包含 `1 <= v <= n`  
- 如果业务不允许修改输入，可先复制数组再操作  
- 单测覆盖重复值、负数、全正连续、缺头/缺尾

---

## S — Summary（总结）

### 核心收获

- 值 `x` 放到索引 `x-1` 是原地哈希的关键  
- 双阶段（置换 + 扫描）可实现 O(n) 时间  
- 该方法满足 O(1) 额外空间的严格要求  
- 适合处理“最小缺失正数”与索引归位类问题

### 推荐延伸阅读

- LeetCode 41. First Missing Positive  
- 原地哈希/索引映射技巧  
- 数组置换与循环不变式分析

### 小结 / 结论

缺失的第一个正数是一道典型的“空间受限”算法题。  
掌握原地索引定位，你将获得一类高频题的通用解法。

---

## 参考与延伸阅读

- https://leetcode.com/problems/first-missing-positive/  
- https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types  
- https://en.cppreference.com/w/cpp/algorithm  
- https://pkg.go.dev/std  

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：Hot100、数组、原地哈希、索引定位  
- **SEO 关键词**：First Missing Positive, 缺失的第一个正数, 原地哈希, 索引映射, O(n)  
- **元描述**：O(n) 时间、O(1) 额外空间的原地索引定位解法与工程应用。  

---

## 行动号召（CTA）

如果你在刷 Hot100，建议把“原地索引定位”整理成自己的模板库。  
欢迎留言分享你在工程中的应用案例。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def first_missing_positive(nums: List[int]) -> int:
    n = len(nums)
    i = 0
    while i < n:
        v = nums[i]
        if 1 <= v <= n and nums[v - 1] != v:
            nums[i], nums[v - 1] = nums[v - 1], nums[i]
        else:
            i += 1
    for i, v in enumerate(nums):
        if v != i + 1:
            return i + 1
    return n + 1


if __name__ == "__main__":
    print(first_missing_positive([1, 2, 0]))
```

```c
#include <stdio.h>

int first_missing_positive(int *nums, int n) {
    int i = 0;
    while (i < n) {
        int v = nums[i];
        if (v >= 1 && v <= n && nums[v - 1] != v) {
            int tmp = nums[v - 1];
            nums[v - 1] = nums[i];
            nums[i] = tmp;
        } else {
            i++;
        }
    }
    for (i = 0; i < n; ++i) {
        if (nums[i] != i + 1) return i + 1;
    }
    return n + 1;
}

int main(void) {
    int nums[] = {3, 4, -1, 1};
    int n = (int)(sizeof(nums) / sizeof(nums[0]));
    printf("%d\n", first_missing_positive(nums, n));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

int first_missing_positive(std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    int i = 0;
    while (i < n) {
        int v = nums[i];
        if (v >= 1 && v <= n && nums[v - 1] != v) {
            std::swap(nums[i], nums[v - 1]);
        } else {
            ++i;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (nums[i] != i + 1) return i + 1;
    }
    return n + 1;
}

int main() {
    std::vector<int> nums{1, 2, 0};
    std::cout << first_missing_positive(nums) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func firstMissingPositive(nums []int) int {
	n := len(nums)
	i := 0
	for i < n {
		v := nums[i]
		if v >= 1 && v <= n && nums[v-1] != v {
			nums[i], nums[v-1] = nums[v-1], nums[i]
		} else {
			i++
		}
	}
	for i, v := range nums {
		if v != i+1 {
			return i + 1
		}
	}
	return n + 1
}

func main() {
	fmt.Println(firstMissingPositive([]int{1, 2, 0}))
}
```

```rust
fn first_missing_positive(nums: &mut Vec<i32>) -> i32 {
    let n = nums.len();
    let mut i = 0;
    while i < n {
        let v = nums[i];
        if v >= 1 && (v as usize) <= n && nums[v as usize - 1] != v {
            let target = (v as usize) - 1;
            nums.swap(i, target);
        } else {
            i += 1;
        }
    }
    for i in 0..n {
        if nums[i] != (i as i32) + 1 {
            return (i as i32) + 1;
        }
    }
    (n as i32) + 1
}

fn main() {
    let mut nums = vec![3, 4, -1, 1];
    println!("{}", first_missing_positive(&mut nums));
}
```

```javascript
function firstMissingPositive(nums) {
  const n = nums.length;
  let i = 0;
  while (i < n) {
    const v = nums[i];
    if (v >= 1 && v <= n && nums[v - 1] !== v) {
      const tmp = nums[v - 1];
      nums[v - 1] = nums[i];
      nums[i] = tmp;
    } else {
      i += 1;
    }
  }
  for (let idx = 0; idx < n; idx += 1) {
    if (nums[idx] !== idx + 1) return idx + 1;
  }
  return n + 1;
}

console.log(firstMissingPositive([3, 4, -1, 1]));
```
