---
title: "Two Sum 两数之和：哈希表一遍扫描与 ACERS 工程化解析"
date: 2025-12-28T10:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["哈希表", "数组", "补数", "两数之和", "面试高频", "ACERS"]
description: "从题意还原到哈希表一遍扫描，系统讲解 Two Sum，并给出工程场景、复杂度对比与多语言实现。"
keywords: ["Two Sum", "两数之和", "hash map", "补数", "O(n)", "LeetCode 1"]
---

> **副标题 / 摘要**  
> Two Sum（两数之和）是最经典的数组哈希题：用“补数 + 哈希表”把 O(n^2) 降到 O(n)。本文按 ACERS 结构拆解题意、原理与工程迁移，并给出多语言可运行实现。

- **预计阅读时长**：10~12 分钟
- **标签**：`哈希表`、`数组`、`补数`、`面试高频`
- **SEO 关键词**：Two Sum, 两数之和, hash map, 补数, O(n)
- **元描述**：两数之和的哈希表解法与工程应用解析，含复杂度对比与多语言代码。

---

## 目标读者

- 刚开始刷题，希望建立“补数 + 哈希表”基本模型的初学者
- 需要把算法思路迁移到工程问题的中级开发者
- 准备面试、想快速掌握高频题的求职者

## 背景 / 动机

“在一堆数字里找出两数之和”等价于一个**快速配对**问题，常见于对账、预算、风控、推荐等场景。  
朴素暴力法虽然简单，但在数据量上来后会直接超时；哈希表一遍扫描能把复杂度从 O(n^2) 降到 O(n)，是最工程可行的做法之一。

## A — Algorithm（题目与算法）

### 题目还原

给定一个整数数组 `nums` 和一个整数目标值 `target`，请在该数组中找出和为目标值的 **两个** 整数，并返回它们的数组下标。  
每种输入只会对应一个答案，并且你不能使用两次相同的元素。答案可以按任意顺序返回。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| target | int | 目标和 |
| 返回 | int[] | 满足 `nums[i] + nums[j] == target` 的下标 |

### 基础示例

| nums | target | 输出 |
| --- | --- | --- |
| [2, 7, 11, 15] | 9 | [0, 1] |
| [3, 2, 4] | 6 | [1, 2] |

**补数图示（示例 1）**

```text
target = 9
i=0, nums[i]=2, need=7, map={}
存入 2->0
i=1, nums[i]=7, need=2, map 中存在 2->0
答案: [0, 1]
```

## C — Concepts（核心思想）

### 核心概念

| 概念 | 含义 | 作用 |
| --- | --- | --- |
| 补数（complement） | need = target - nums[i] | 把求和转成查找 |
| 哈希表（hash map） | 值 → 下标 | O(1) 平均查找 |
| 一遍扫描 | 从左到右一次遍历 | 保证最优时间 |

### 方法类型

哈希表 + 一遍扫描 + 查找型问题。

### 概念模型

核心公式只有一个：

```text
need = target - nums[i]
```

流程模型：

```text
遍历 nums[i] -> 计算 need -> 在哈希表查找 -> 命中则返回 -> 否则记录 nums[i]
```

### 关键数据结构

使用哈希表保存 **值 → 下标** 的映射。  
先查找补数，再插入当前元素，可以避免“重复使用同一元素”的错误。

## 实践指南 / 步骤

1. 初始化哈希表 `seen`（键：数值；值：下标）。
2. 遍历数组，计算 `need = target - nums[i]`。
3. 若 `need` 在 `seen` 中，直接返回 `[seen[need], i]`。
4. 否则记录 `seen[nums[i]] = i`，继续遍历。

运行方式示例：

```bash
python3 two_sum.py
```

## 可运行示例（Python）

```python
from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []


if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))
```

## E — Engineering（工程应用）

### 场景 1：对账异常定位（Python，数据分析）

**背景**：电商对账中，需要在订单金额列表里找出两笔加起来等于某个目标值的订单。  
**为什么适用**：数据量较大时，O(n) 的哈希方案更省时。

```python
def find_pair_amounts(amounts, target):
    seen = {}
    for i, x in enumerate(amounts):
        need = target - x
        if need in seen:
            return seen[need], i
        seen[x] = i
    return None


print(find_pair_amounts([120, 80, 200, 50], 200))
```

### 场景 2：风控阈值组合检查（Go，后台服务）

**背景**：风控服务里需要判断两项指标是否能组合达到阈值，用于快速触发规则。  
**为什么适用**：服务端强调延迟，哈希表一遍扫描最稳妥。

```go
package main

import "fmt"

func twoSum(nums []int, target int) []int {
	seen := map[int]int{}
	for i, x := range nums {
		need := target - x
		if j, ok := seen[need]; ok {
			return []int{j, i}
		}
		seen[x] = i
	}
	return []int{}
}

func main() {
	fmt.Println(twoSum([]int{12, 5, 7, 3}, 10))
}
```

### 场景 3：购物车组合提示（JavaScript，前端）

**背景**：前端需要提示“任选两件商品可达满减门槛”。  
**为什么适用**：在浏览器侧快速计算组合，避免一次次后端请求。

```javascript
function twoSum(nums, target) {
  const seen = new Map();
  for (let i = 0; i < nums.length; i += 1) {
    const need = target - nums[i];
    if (seen.has(need)) {
      return [seen.get(need), i];
    }
    seen.set(nums[i], i);
  }
  return [];
}

console.log(twoSum([39, 21, 10, 60], 70));
```

## R — Reflection（反思与深入）

### 复杂度分析

时间复杂度：O(n)，哈希表查询均摊 O(1)。  
空间复杂度：O(n)，需要存储已遍历的元素。

### 替代方案与取舍

| 方案 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力双循环 | O(n^2) | O(1) | 实现简单但大规模超时 |
| 排序 + 双指针 | O(n log n) | O(n) | 需保留原下标 |
| 哈希表一遍 | O(n) | O(n) | 当前方法，速度最佳 |

### 常见问题与注意事项

- 先插入再查找会导致“同元素重复使用”的错误
- 有重复数字时，必须保证返回的两个下标不同
- 题目保证唯一答案，可以命中即返回
- 大数组注意哈希表负载因子，避免退化

### 为什么当前方法最优 / 最工程可行

数组无序且只需一次配对时，任何比较型方法都需要至少 O(n) 的扫描。  
哈希表把“查找补数”降到均摊 O(1)，因此是时间复杂度上的最优实践。

## 最佳实践与建议

- 坚持“先查找补数，再插入当前值”的顺序
- 统一用 `value -> index` 映射，避免索引丢失
- 单元测试覆盖重复元素、负数、零值场景
- 数据量大时可预估哈希表容量以减少扩容

## S — Summary（总结）

- Two Sum 的核心是“补数 + 哈希表”的一次遍历模型
- 哈希表可把暴力 O(n^2) 降为 O(n)
- 先查找再插入能避免同一元素被重复使用
- 工程场景中常用于对账、风控、组合提示等快速配对问题

### 小结 / 结论

这道题表面简单，但背后体现的是“把求和转为查找”的抽象能力。  
掌握它不仅能提升刷题效率，也能在真实业务里迅速定位高效解法。

## 参考与延伸阅读

- https://leetcode.com/problems/two-sum/
- https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
- https://en.cppreference.com/w/cpp/container/unordered_map
- https://go.dev/ref/spec#Map_types
- https://doc.rust-lang.org/std/collections/struct.HashMap.html

## 行动号召（CTA）

如果你在业务里遇到 Two Sum 变体（Three Sum、子数组求和等），欢迎留言交流；也可以收藏本文作为面试复盘清单。

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []


if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int key;
    int val;
    int used;
} Entry;

static unsigned hash_int(int key) {
    return (uint32_t)key * 2654435761u;
}

static int find_slot(Entry *table, int cap, int key, int *found) {
    unsigned mask = (unsigned)cap - 1u;
    unsigned idx = hash_int(key) & mask;
    while (table[idx].used && table[idx].key != key) {
        idx = (idx + 1u) & mask;
    }
    *found = table[idx].used && table[idx].key == key;
    return (int)idx;
}

int two_sum(const int *nums, int n, int target, int out[2]) {
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 2) cap = 2;
    Entry *table = (Entry *)calloc((size_t)cap, sizeof(Entry));
    if (!table) return 0;

    for (int i = 0; i < n; ++i) {
        int need = target - nums[i];
        int found = 0;
        int pos = find_slot(table, cap, need, &found);
        if (found) {
            out[0] = table[pos].val;
            out[1] = i;
            free(table);
            return 1;
        }
        pos = find_slot(table, cap, nums[i], &found);
        if (!found) {
            table[pos].key = nums[i];
            table[pos].val = i;
            table[pos].used = 1;
        }
    }
    free(table);
    return 0;
}

int main(void) {
    int nums[] = {2, 7, 11, 15};
    int out[2];
    if (two_sum(nums, 4, 9, out)) {
        printf("%d %d\n", out[0], out[1]);
    }
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

std::vector<int> two_sum(const std::vector<int> &nums, int target) {
    std::unordered_map<int, int> seen;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        int need = target - nums[i];
        auto it = seen.find(need);
        if (it != seen.end()) {
            return {it->second, i};
        }
        seen[nums[i]] = i;
    }
    return {};
}

int main() {
    std::vector<int> nums{2, 7, 11, 15};
    auto res = two_sum(nums, 9);
    if (!res.empty()) {
        std::cout << res[0] << " " << res[1] << "\n";
    }
    return 0;
}
```

```go
package main

import "fmt"

func twoSum(nums []int, target int) []int {
	seen := map[int]int{}
	for i, x := range nums {
		need := target - x
		if j, ok := seen[need]; ok {
			return []int{j, i}
		}
		seen[x] = i
	}
	return []int{}
}

func main() {
	fmt.Println(twoSum([]int{2, 7, 11, 15}, 9))
}
```

```rust
use std::collections::HashMap;

fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {
    let mut seen: HashMap<i32, usize> = HashMap::new();
    for (i, &x) in nums.iter().enumerate() {
        let need = target - x;
        if let Some(&j) = seen.get(&need) {
            return Some((j, i));
        }
        seen.insert(x, i);
    }
    None
}

fn main() {
    let nums = vec![2, 7, 11, 15];
    if let Some((i, j)) = two_sum(&nums, 9) {
        println!("{} {}", i, j);
    }
}
```

```javascript
function twoSum(nums, target) {
  const seen = new Map();
  for (let i = 0; i < nums.length; i += 1) {
    const need = target - nums[i];
    if (seen.has(need)) {
      return [seen.get(need), i];
    }
    seen.set(nums[i], i);
  }
  return [];
}

console.log(twoSum([2, 7, 11, 15], 9));
```
