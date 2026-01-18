---
title: "Hot100：和为 K 的子数组（Subarray Sum Equals K）前缀和 + 哈希表 ACERS 解析"
date: 2026-01-17T20:49:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "前缀和", "哈希表", "子数组", "计数"]
description: "用前缀和与哈希表在 O(n) 时间统计和为 K 的子数组数量，含工程场景、多语言实现与常见误区。"
keywords: ["Subarray Sum Equals K", "和为K的子数组", "前缀和", "hash map", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 这是 Hot100 专栏第 1 篇：和为 K 的子数组。本文用“前缀和 + 频次哈希表”把 O(n^2) 降到 O(n)，并按 ACERS 模板给出工程场景与多语言实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`前缀和`、`哈希表`  
- **SEO 关键词**：Subarray Sum Equals K, 和为K的子数组, 前缀和, 哈希表, O(n)  
- **元描述**：和为 K 的子数组计数问题的前缀和解法，含工程迁移、复杂度对比与多语言代码。  

---

## 目标读者

- 正在刷 Hot100，希望建立稳定算法模板的初学者
- 需要把计数类算法迁移到业务数据统计的中级工程师
- 准备面试，想掌握“前缀和 + 哈希表”核心套路的人

## 背景 / 动机

“统计和为 K 的子数组数量”是最经典的计数类问题之一。  
它广泛出现在日志分析、风控阈值命中、交易序列统计等场景。  
朴素的两层遍历虽然直观，但一旦数据规模增大就会明显卡顿，因此需要可扩展的 O(n) 解法。

## 核心概念（必须理解）

- **子数组**：数组中连续、非空的片段  
- **前缀和**：`prefix[i] = nums[0..i]` 的和  
- **差分关系**：若 `prefix[r] - prefix[l-1] = k`，则 `nums[l..r]` 的和为 k  
- **频次哈希表**：统计某个前缀和出现的次数，以 O(1) 均摊时间查询

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个整数数组 `nums` 和一个整数 `k`，请统计并返回 **和为 k 的子数组** 的个数。  
子数组是数组中元素的连续非空序列。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| k | int | 目标和 |
| 返回 | int | 和为 k 的子数组数量 |

### 示例 1

```text
nums = [1, 1, 1], k = 2
```

可行子数组为 `[1,1]`（下标 0..1）和 `[1,1]`（下标 1..2），  
**输出**：`2`

### 示例 2

```text
nums = [1, 2, 3], k = 3
```

可行子数组为 `[1,2]` 和 `[3]`，  
**输出**：`2`

---

## C — Concepts（核心思想）

### 方法类型

**前缀和 + 频次哈希表**，属于典型的计数型算法。

### 关键公式

设前缀和为：

```text
prefix[0] = 0
prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
```

子数组 `nums[l..r]` 的和：

```text
sum(l..r) = prefix[r+1] - prefix[l]
```

要让它等于 `k`，只需满足：

```text
prefix[l] = prefix[r+1] - k
```

### 核心思路

从左到右遍历数组，用 `sum` 记录当前前缀和。  
每走到一个位置，就把 `sum - k` 在哈希表里出现过的次数加到答案中，  
再把当前 `sum` 记入哈希表。

这一步“先统计、再入表”的顺序可以避免遗漏。

---

## 实践指南 / 步骤

1. 初始化：`sum = 0`，哈希表 `count = {0: 1}`  
2. 遍历 `nums` 中每个元素 `x`  
3. 更新 `sum += x`  
4. 累加答案 `ans += count[sum - k]`  
5. 更新 `count[sum] += 1`

---

## 可运行示例（Python）

```python
from typing import List


def subarray_sum(nums: List[int], k: int) -> int:
    count = {0: 1}
    ans = 0
    s = 0
    for x in nums:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


if __name__ == "__main__":
    print(subarray_sum([1, 1, 1], 2))
    print(subarray_sum([1, 2, 3], 3))
```

运行方式示例：

```bash
python3 demo.py
```

---

## 解释与原理（为什么这么做）

这个问题的难点是：子数组必须连续。  
前缀和把“连续子数组的和”转成了**两个前缀和的差**，  
于是计数问题就变成了“查询之前是否出现过某个前缀和”。

这也是为什么滑动窗口不可靠：  
数组里存在负数时，窗口的单调性被破坏，不能保证正确性。

---

## E — Engineering（工程应用）

### 场景 1：交易流水命中阈值统计（Python，数据分析）

**背景**：对一段交易流水 `amounts` 统计“连续天数交易和刚好等于 k”的次数。  
**为什么适用**：交易额有正有负，滑动窗口失效，前缀和更稳。

```python
def count_exact_k(amounts, k):
    count = {0: 1}
    s = 0
    ans = 0
    for x in amounts:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


print(count_exact_k([3, -1, 2, 1, -2, 4], 3))
```

### 场景 2：服务监控窗口命中统计（Go，后台服务）

**背景**：统计连续时间窗口内“错误数总和等于 k”的次数，用于报警策略回放。  
**为什么适用**：日志是离线批处理，O(n) 统计性能最好。

```go
package main

import "fmt"

func countExactK(nums []int, k int) int {
	count := map[int]int{0: 1}
	sum := 0
	ans := 0
	for _, x := range nums {
		sum += x
		ans += count[sum-k]
		count[sum]++
	}
	return ans
}

func main() {
	fmt.Println(countExactK([]int{1, 2, 3, -2, 2}, 3))
}
```

### 场景 3：前端优惠组合提示（JavaScript，前端）

**背景**：在购物车中统计“连续商品价格和恰好等于满减阈值”的组合数量。  
**为什么适用**：前端轻量计算，不依赖后端即可给出提示。

```javascript
function countExactK(nums, k) {
  const count = new Map();
  count.set(0, 1);
  let sum = 0;
  let ans = 0;
  for (const x of nums) {
    sum += x;
    ans += count.get(sum - k) || 0;
    count.set(sum, (count.get(sum) || 0) + 1);
  }
  return ans;
}

console.log(countExactK([5, -1, 2, 4, -2], 4));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
- 空间复杂度：O(n)

### 替代方案与对比

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力双循环 | O(n^2) | O(1) | 简单但易超时 |
| 前缀和 + 哈希表 | O(n) | O(n) | 当前方法，最优可行 |
| 排序前缀和 | O(n log n) | O(n) | 适合求区间数量但实现更复杂 |

### 常见错误思路

- **滑动窗口**：只适用于全正数，负数会破坏单调性  
- **漏掉前缀和 0**：没初始化 `count[0] = 1` 会漏掉从下标 0 开始的解  
- **使用 32 位累加**：和可能溢出，建议使用 64 位

### 为什么这是最优

至少要遍历一次数组才能知道所有子数组信息，  
因此时间复杂度下界是 O(n)。  
哈希表让每一步查找均摊 O(1)，实现了最优可行解。

---

## 常见问题与注意事项

1. **数组包含负数怎么办？**  
   前缀和方案天然支持负数，这是它比滑动窗口更强的关键原因。  
   **反例**：`nums = [1, -1, 1], k = 1`  
   正确答案有 3 个子数组：`[1] (0..0)`、`[1,-1,1] (0..2)`、`[1] (2..2)`。  
   如果用“正数场景”的滑动窗口策略：  
   - `l=0, r=0, sum=1` → 命中 1 次，然后为了找新解收缩 `l++`，`sum=0`  
   - `r=1`，`sum=-1`；`r=2`，`sum=0`  
   结束时只统计到 1 个解，漏掉了 `(0..2)` 和 `(2..2)`。  
   **根因**：负数使得窗口和不具备单调性，  
   “sum > k 收缩 / sum < k 扩张”的规则不再可靠。

2. **k 很大时会溢出吗？**  
   建议用 64 位整数保存前缀和，尤其是语言默认 int 较小的场景。

3. **子数组必须连续吗？**  
   是的，题目要求连续子数组，非连续是子序列概念。

---

## 最佳实践与建议

- 使用“前缀和 + 频次表”作为固定模板  
- 初始化 `count[0] = 1`，不要忘  
- 对大数和用 64 位  
- 写单元测试覆盖负数、全零、k=0 等边界

---

## S — Summary（总结）

- 子数组求和可转化为“前缀和差分”问题  
- 哈希表计数把 O(n^2) 降为 O(n)  
- 负数存在时滑动窗口不可靠  
- 初始化 `count[0]=1` 是关键细节  
- 工程上常用于日志、交易、监控等连续统计任务

### 推荐延伸阅读

- LeetCode 560 — Subarray Sum Equals K  
- Prefix Sum (前缀和) 数据结构  
- Hash Map 在计数问题中的经典用法  
- Sliding Window 适用条件对比

---

## 小结 / 结论

这道题的价值不在“解题技巧”，而在于可复用的前缀和计数模型。  
把它掌握成模板，你会发现一大批“连续区间计数”问题都能一把过。

---

## 参考与延伸阅读

- https://leetcode.com/problems/subarray-sum-equals-k/
- https://cp-algorithms.com/data_structures/prefix_sum.html
- https://en.cppreference.com/w/cpp/container/unordered_map
- https://doc.rust-lang.org/std/collections/struct.HashMap.html

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：Hot100、前缀和、哈希表、计数  
- **SEO 关键词**：Subarray Sum Equals K, 和为K的子数组, 前缀和, 哈希表  
- **元描述**：用前缀和 + 哈希表在线统计和为 K 的子数组数量，含工程应用与多语言实现。  

---

## 行动号召（CTA）

如果你正在刷 Hot100，建议把每题都按“模型 + 工程迁移”的方式沉淀下来。  
也欢迎在评论区分享你的题解或工程化变体。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def subarray_sum(nums: List[int], k: int) -> int:
    count = {0: 1}
    ans = 0
    s = 0
    for x in nums:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


if __name__ == "__main__":
    print(subarray_sum([1, 1, 1], 2))
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long key;
    int val;
    int used;
} Entry;

static unsigned long long hash_ll(long long x) {
    return (unsigned long long)x * 11400714819323198485ull;
}

static int find_slot(Entry *table, int cap, long long key, int *found) {
    unsigned long long mask = (unsigned long long)cap - 1ull;
    unsigned long long idx = hash_ll(key) & mask;
    while (table[idx].used && table[idx].key != key) {
        idx = (idx + 1ull) & mask;
    }
    *found = table[idx].used && table[idx].key == key;
    return (int)idx;
}

int subarray_sum(const int *nums, int n, int k) {
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 2) cap = 2;
    Entry *table = (Entry *)calloc((size_t)cap, sizeof(Entry));
    if (!table) return 0;

    long long sum = 0;
    int ans = 0;
    int found = 0;
    int pos = find_slot(table, cap, 0, &found);
    table[pos].used = 1;
    table[pos].key = 0;
    table[pos].val = 1;

    for (int i = 0; i < n; ++i) {
        sum += nums[i];
        pos = find_slot(table, cap, sum - k, &found);
        if (found) ans += table[pos].val;
        pos = find_slot(table, cap, sum, &found);
        if (found) {
            table[pos].val += 1;
        } else {
            table[pos].used = 1;
            table[pos].key = sum;
            table[pos].val = 1;
        }
    }
    free(table);
    return ans;
}

int main(void) {
    int nums[] = {1, 1, 1};
    printf("%d\n", subarray_sum(nums, 3, 2));
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

int subarraySum(const std::vector<int> &nums, int k) {
    std::unordered_map<long long, int> count;
    count[0] = 1;
    long long sum = 0;
    int ans = 0;
    for (int x : nums) {
        sum += x;
        auto it = count.find(sum - k);
        if (it != count.end()) {
            ans += it->second;
        }
        count[sum] += 1;
    }
    return ans;
}

int main() {
    std::vector<int> nums{1, 1, 1};
    std::cout << subarraySum(nums, 2) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func subarraySum(nums []int, k int) int {
	count := map[int]int{0: 1}
	sum := 0
	ans := 0
	for _, x := range nums {
		sum += x
		ans += count[sum-k]
		count[sum]++
	}
	return ans
}

func main() {
	fmt.Println(subarraySum([]int{1, 1, 1}, 2))
}
```

```rust
use std::collections::HashMap;

fn subarray_sum(nums: &[i32], k: i32) -> i32 {
    let mut count: HashMap<i64, i32> = HashMap::new();
    count.insert(0, 1);
    let mut sum: i64 = 0;
    let mut ans: i32 = 0;
    for &x in nums {
        sum += x as i64;
        if let Some(v) = count.get(&(sum - k as i64)) {
            ans += *v;
        }
        *count.entry(sum).or_insert(0) += 1;
    }
    ans
}

fn main() {
    let nums = vec![1, 1, 1];
    println!("{}", subarray_sum(&nums, 2));
}
```

```javascript
function subarraySum(nums, k) {
  const count = new Map();
  count.set(0, 1);
  let sum = 0;
  let ans = 0;
  for (const x of nums) {
    sum += x;
    ans += count.get(sum - k) || 0;
    count.set(sum, (count.get(sum) || 0) + 1);
  }
  return ans;
}

console.log(subarraySum([1, 1, 1], 2));
```
