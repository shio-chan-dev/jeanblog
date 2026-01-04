---
title: "数据结构基础：好数对计数（Number of Good Pairs）哈希统计 ACERS 解析"
date: 2025-12-30T11:40:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["哈希表", "计数", "数组", "好数对", "LeetCode 1512", "ACERS"]
description: "用哈希计数一遍扫描解决好数对（Good Pairs）问题，附工程场景、复杂度对比与多语言实现。"
keywords: ["Good Pairs", "好数对", "hash map", "frequency", "计数", "LeetCode 1512"]
---

> **副标题 / 摘要**  
> 这是“数据结构基础”系列第 2 题：好数对计数。通过“频次统计 + 组合计数”，把 O(n^2) 直接降到 O(n)，并给出可直接迁移到工程的实现方式。

- **预计阅读时长**：8~10 分钟
- **标签**：`哈希表`、`计数`、`数组`
- **SEO 关键词**：Good Pairs, 好数对, hash map, frequency, 计数
- **元描述**：好数对计数的哈希表解法与工程化应用，含复杂度分析与多语言代码。

---

## 目标读者

- 刚开始学习哈希表与计数思想的初学者
- 希望把刷题方法迁移到业务统计的中级工程师
- 准备面试，想掌握基础计数模型的同学

## 背景 / 动机

“找出相同元素的两两组合数量”是一个常见的**计数类问题**。  
在数据去重、行为分析、错误归因等场景里，这类问题通常会被反复遇到。  
若用双重循环计算，复杂度是 O(n^2)；一旦数据规模扩大就会变慢。  
因此需要一个可线性扩展的方案。

## A — Algorithm（题目与算法）

### 题目还原

给你一个整数数组 `nums`。  
如果一组数字 `(i, j)` 满足 `nums[i] == nums[j]` 且 `i < j`，就称为一个 **好数对**。  
返回好数对的数目。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| 返回 | int | 好数对数量 |

### 基础示例

| nums | 输出 | 说明 |
| --- | --- | --- |
| [1, 2, 3, 1, 1, 3] | 4 | (0,3) (0,4) (3,4) (2,5) |
| [1, 1, 1, 1] | 6 | C(4,2) = 6 |
| [1, 2, 3] | 0 | 无重复 |

**直观图示（示例 1）**

```text
值 1 出现 3 次 -> 组合数 C(3,2)=3
值 3 出现 2 次 -> 组合数 C(2,2)=1
总计 3 + 1 = 4
```

## C — Concepts（核心思想）

### 核心概念与术语

- **频次统计（frequency count）**：统计每个数字出现的次数  
- **组合数**：同值出现 `c` 次，可形成 `c * (c - 1) / 2` 个好数对  
- **哈希表**：在 O(1) 均摊时间内完成计数

### 算法类型

哈希计数 / 频次统计 / 组合计数。

### 关键公式

```text
对每个值 v，出现次数为 c
好数对数量 = c * (c - 1) / 2
```

### 一遍扫描模型

当遍历到 `nums[i]` 时，如果该值已经出现 `count` 次，那么当前元素与之前的 `count` 个元素都能形成好数对：

```text
ans += count[nums[i]]
count[nums[i]] += 1
```

## 实践指南 / 步骤

1. 初始化哈希表 `count` 和答案 `ans = 0`
2. 逐个遍历数组元素 `x`
3. 把 `count[x]` 加到 `ans` 上（表示新形成的好数对）
4. 再把 `count[x]` 加 1

运行方式示例：

```bash
python3 good_pairs.py
```

## 可运行示例（Python）

```python
from typing import List


def num_identical_pairs(nums: List[int]) -> int:
    count = {}
    ans = 0
    for x in nums:
        ans += count.get(x, 0)
        count[x] = count.get(x, 0) + 1
    return ans


if __name__ == "__main__":
    print(num_identical_pairs([1, 2, 3, 1, 1, 3]))
```

## E — Engineering（工程应用）

### 场景 1：数据质量评估（Python，数据分析）

**背景**：统计同一字段的重复配对数，用于评估某一列的重复程度。  
**为什么适用**：只关心“重复程度”而非具体位置，哈希计数最合适。

```python
def duplicate_pair_score(values):
    count = {}
    score = 0
    for v in values:
        score += count.get(v, 0)
        count[v] = count.get(v, 0) + 1
    return score


print(duplicate_pair_score(["A", "B", "A", "C", "A"]))
```

### 场景 2：批处理任务去重权重（Go，后台服务）

**背景**：对任务 ID 做重复计数，重复次数越高说明越可能是批量重试或异常。  
**为什么适用**：需要线性时间统计，不增加服务延迟。

```go
package main

import "fmt"

func goodPairs(ids []int) int {
	count := map[int]int{}
	ans := 0
	for _, id := range ids {
		ans += count[id]
		count[id]++
	}
	return ans
}

func main() {
	fmt.Println(goodPairs([]int{7, 7, 8, 9, 7}))
}
```

### 场景 3：前端列表重复提示（JavaScript，前端）

**背景**：在表单或导入预览里提示用户有多少重复项。  
**为什么适用**：前端侧一次扫描即可计算，不必等待后端统计。

```javascript
function goodPairs(items) {
  const count = new Map();
  let ans = 0;
  for (const x of items) {
    ans += count.get(x) || 0;
    count.set(x, (count.get(x) || 0) + 1);
  }
  return ans;
}

console.log(goodPairs(["u1", "u2", "u1", "u1"]));
```

## 解释与原理

把“找两两相同”转为“统计出现次数”，就能把问题变成组合计数。  
每新增一个元素，它与之前所有相同元素都构成好数对，因此直接累加已有计数即可。  
这种思路是一种常见的“**在线计数**”模型，适用于数据流、日志、行为序列等场景。

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)
- 空间复杂度：O(n)

### 替代方案与取舍

| 方案 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力双循环 | O(n^2) | O(1) | 简单但易超时 |
| 排序后分组 | O(n log n) | O(1) | 需排序，破坏原顺序 |
| 哈希计数一遍扫描 | O(n) | O(n) | 速度最佳，工程可行 |

### 常见问题与注意事项

- 先累加再自增，才能保证不会把当前元素和自己配对
- 计数值可能很大，建议使用 64 位整数
- 哈希表负载过高会退化，必要时可预估容量

### 为什么当前方法最优

你必须至少遍历一次数组才能知道重复情况；  
哈希表在均摊 O(1) 时间内完成统计，因此整体 O(n) 是最优的工程选择。

## 最佳实践与建议

- 使用“在线计数”模型，避免两层循环
- 结果可能超出 32 位范围时用 `long long` / `int64`
- 需要稳定输入顺序时，避免排序方案
- 对频繁重复的值可预估哈希容量以减少扩容

## S — Summary（总结）

- 好数对数量等价于“相同元素的两两组合数”
- 哈希表统计频次可把复杂度从 O(n^2) 降到 O(n)
- “先累加再自增”的一遍扫描是最简洁安全的写法
- 该模型可直接迁移到数据质量、去重统计、行为分析等场景

### 小结 / 结论

好数对是一个“看似简单但高度工程化”的计数问题。  
掌握哈希计数的思路，可以快速解决一大类重复统计问题。

## 参考与延伸阅读

- https://leetcode.com/problems/number-of-good-pairs/
- https://en.wikipedia.org/wiki/Combination
- https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
- https://en.cppreference.com/w/cpp/container/unordered_map
- https://doc.rust-lang.org/std/collections/struct.HashMap.html

## 行动号召（CTA）

把这题当作“计数模型”的起点，尝试改写为 Three Sum 变体或分组统计问题，并在评论区分享你的思路。

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def num_identical_pairs(nums: List[int]) -> int:
    count = {}
    ans = 0
    for x in nums:
        ans += count.get(x, 0)
        count[x] = count.get(x, 0) + 1
    return ans


if __name__ == "__main__":
    print(num_identical_pairs([1, 2, 3, 1, 1, 3]))
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int key;
    int count;
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

long long num_identical_pairs(const int *nums, int n) {
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 2) cap = 2;
    Entry *table = (Entry *)calloc((size_t)cap, sizeof(Entry));
    if (!table) return 0;

    long long ans = 0;
    for (int i = 0; i < n; ++i) {
        int found = 0;
        int pos = find_slot(table, cap, nums[i], &found);
        if (found) {
            ans += table[pos].count;
            table[pos].count += 1;
        } else {
            table[pos].used = 1;
            table[pos].key = nums[i];
            table[pos].count = 1;
        }
    }
    free(table);
    return ans;
}

int main(void) {
    int nums[] = {1, 2, 3, 1, 1, 3};
    printf("%lld\n", num_identical_pairs(nums, 6));
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

long long num_identical_pairs(const std::vector<int> &nums) {
    std::unordered_map<int, long long> count;
    long long ans = 0;
    for (int x : nums) {
        ans += count[x];
        count[x] += 1;
    }
    return ans;
}

int main() {
    std::vector<int> nums{1, 2, 3, 1, 1, 3};
    std::cout << num_identical_pairs(nums) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func numIdenticalPairs(nums []int) int64 {
	count := map[int]int64{}
	var ans int64 = 0
	for _, x := range nums {
		ans += count[x]
		count[x]++
	}
	return ans
}

func main() {
	fmt.Println(numIdenticalPairs([]int{1, 2, 3, 1, 1, 3}))
}
```

```rust
use std::collections::HashMap;

fn num_identical_pairs(nums: &[i32]) -> i64 {
    let mut count: HashMap<i32, i64> = HashMap::new();
    let mut ans: i64 = 0;
    for &x in nums {
        let c = *count.get(&x).unwrap_or(&0);
        ans += c;
        count.insert(x, c + 1);
    }
    ans
}

fn main() {
    let nums = vec![1, 2, 3, 1, 1, 3];
    println!("{}", num_identical_pairs(&nums));
}
```

```javascript
function numIdenticalPairs(nums) {
  const count = new Map();
  let ans = 0;
  for (const x of nums) {
    ans += count.get(x) || 0;
    count.set(x, (count.get(x) || 0) + 1);
  }
  return ans;
}

console.log(numIdenticalPairs([1, 2, 3, 1, 1, 3]));
```
