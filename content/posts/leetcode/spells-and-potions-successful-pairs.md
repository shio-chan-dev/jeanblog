---
title: "咒语与药水的成功组合：排序 + 二分查找秒杀乘积约束问题"
date: 2025-12-04T10:50:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "排序", "数组", "双指针", "面试高频题"]
description: "给定 spells 和 potions 两个数组以及 success 阈值，如何用排序 + 二分查找在 O((n+m)log m) 时间内求出每个咒语能与多少药水形成乘积 ≥ success 的成功组合，并给出多语言实现与工程应用示例。"
keywords: ["spells and potions", "successful pairs", "二分查找", "排序", "乘积约束", "LeetCode 2300"]
---

> **副标题 / 摘要**  
> 一道典型的“乘积 ≥ 阈值”计数题，看起来像是 O(n²) 的双重循环，实际上用「排序 + 二分查找」就能把复杂度压到 O((n+m)log m)。本文从题意抽象、核心公式到多语言实现，带你把这类阈值匹配问题彻底吃透。

- **预计阅读时长**：10~15 分钟  
- **适用场景标签**：`二分查找`、`排序计数`、`阈值匹配`  
- **SEO 关键词**：spells and potions, successful pairs, 二分查找, lower_bound, 乘积约束  

---

## 目标读者与背景

**目标读者**

- 已熟悉基本二分查找，想提升「在有序数组上做计数」能力的同学
- 后端 / 算法工程师，经常处理阈值判断与配对统计的问题
- 准备技术面试，希望积累“排序 + 二分”模板的开发者

**为什么这题值得单独写一篇？**

- 它把一个表面 O(n²) 的「所有配对」问题，转化成了**对有序数组的二分计数**；
- 公式非常典型：把 `a * b ≥ success` 转成 `b ≥ ceil(success / a)`；
- 这种思路在推荐系统、风控额度、资源匹配等业务里屡见不鲜。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定两个整数数组 `spells` 和 `potions`，以及一个正整数 `success`。  
> 对于每个咒语 `spells[i]`，我们定义它与药水 `potions[j]` 的组合是“成功”的，当且仅当：  
> `spells[i] * potions[j] >= success`  
> 请返回一个数组 `ans`，其中 `ans[i]` 表示第 `i` 个咒语可以与多少个药水形成成功组合。

**输入**

- `spells`: 长度为 `n` 的整数数组
- `potions`: 长度为 `m` 的整数数组
- `success`: 正整数阈值

**输出**

- 整数数组 `ans`，长度为 `n`，`ans[i]` 为每个 `spells[i]` 能匹配的成功药水数量

### 示例

```text
spells  = [5, 1, 3]
potions = [1, 2, 3, 4, 5]
success = 7
```

对每个咒语：

- `spell = 5`：  
  需要 `5 * potion >= 7` → `potion >= 7/5 = 1.4`，向上取整得到 `potion >= 2`  
  在 `potions` 中满足的是 `[2, 3, 4, 5]`，一共 `4` 个

- `spell = 1`：  
  需要 `1 * potion >= 7` → `potion >= 7`  
  `potions` 里最大也只有 5，所以是 `0` 个

- `spell = 3`：  
  需要 `3 * potion >= 7` → `potion >= 7/3 ≈ 2.33`，向上取整得到 `potion >= 3`  
  满足的是 `[3, 4, 5]`，一共 `3` 个

因此答案为：

```text
ans = [4, 0, 3]
```

---

## C — Concepts（核心思想）

### 1. 从乘积约束到「下界」问题

对固定的咒语值 `s = spells[i]`，成功条件是：

```text
s * potions[j] >= success
```

假设我们只考虑 `s > 0`（若题目存在 0 或负数可额外讨论），可以等价变形为：

```text
potions[j] >= success / s
```

由于 `potions[j]` 和 `success` 是整数，我们要满足：

```text
potions[j] >= ceil(success / s)
```

记：

```text
need = ceil(success / s)
```

注意：  
**不要使用浮点数**，用整数安全实现向上取整的公式：

```text
need = (success + s - 1) // s
```

这样，每个咒语的问题就变成了：

> 在数组 `potions` 中，找到第一个 **≥ need** 的位置，下标记为 `idx`，  
> 则从 `idx` 到末尾 `m-1` 的所有药水都满足条件，总数为 `m - idx`。

这正是一个标准的「有序数组上找下界（lower_bound）」的问题。

### 2. 为什么要排序？

二分查找的前提是数组有序。

我们可以：

1. 单独对 `potions` 排序（不影响题意，因为只关心数量，不关心原下标）；
2. 对每个咒语 `s`：
   - 计算 `need = ceil(success / s)`
   - 在排序后的 `potions` 中，二分找到第一个 `>= need` 的位置 `idx`
   - 对应成功数为 `m - idx`

这样就避免了遍历整个 `potions` 数组的 O(m) 操作，每个咒语只需 O(log m)。

### 3. 算法类型与复杂度

- 算法类型：**排序 + 二分查找（lower_bound）**
- 时间复杂度：
  - 排序 `potions`: O(m log m)
  - 对每个 `spells[i]` 做一次二分：O(n log m)
  - 总体：O((n + m) log m)
- 空间复杂度：
  - 如果在原地排序：O(1) 额外空间（忽略递归栈）

与暴力方法 O(n·m) 相比，在 `n, m` 均为 1e5 级别时，差距非常明显。

---

## 实践指南 / 实现步骤

1. **排序 `potions`**
   - 使用语言内置排序即可（如 `sort.Ints`、`std::sort`）。

2. **遍历每个咒语 `s`**

   - 若 `s == 0`，则 `s * potions[j]` 永远是 0，不可能 ≥ 正数 `success`，答案为 0；
   - 否则计算：

     ```text
     need = (success + s - 1) // s
     ```

3. **在排序好的 `potions` 上二分**

   - 找到第一个 `potions[idx] >= need` 的位置；
   - 若 `idx == m`（越界），说明不存在满足条件的药水，结果为 0；
   - 否则结果为 `m - idx`。

4. **收集结果**
   - 为每个咒语记录这一数量，输出数组 `ans`。

5. **边界检查**
   - `spells` 或 `potions` 为空时，直接返回全 0；
   - 注意使用足够大的整数类型保存中间结果（如 `long long` / `int64`）。

---

## E — Engineering（工程应用）

下面用三个实际场景，说明这种「排序 + 阈值二分计数」在工程里的用法。

### 场景 1：定价与优惠组合评估（Python）

**背景**  
你有一批商品价格（咒语）和一批折扣系数（药水，例如 0.9、0.8）。你想知道：  
对每个商品，**有多少种折扣方案会让折后收入仍然大于某个阈值**？

虽然实际业务多是浮点运算，这里可以简化为整数乘积与阈值比较。

**示例代码**

```python
from bisect import bisect_left
from typing import List


def successful_pairs(spells: List[int], potions: List[int], success: int) -> List[int]:
    potions.sort()
    m = len(potions)
    ans = []

    for s in spells:
        if s == 0:
            ans.append(0)
            continue
        need = (success + s - 1) // s  # ceil
        idx = bisect_left(potions, need)
        ans.append(m - idx)

    return ans


if __name__ == "__main__":
    print(successful_pairs([5, 1, 3], [1, 2, 3, 4, 5], 7))  # [4, 0, 3]
```

---

### 场景 2：风控额度组合估算（Go）

**背景**  
在风控系统中：

- `spells[i]` 可以看作借款金额；
- `potions[j]` 可以看作担保倍数 / 抵押倍数；
- `success` 是一个安全阈值，例如“风险缓释程度”。

你希望知道对每一笔借款金额 `spells[i]`，有多少种担保方案可以让：

```text
借款金额 * 担保倍数 >= success
```

**示例代码**

```go
package main

import (
	"fmt"
	"sort"
)

func successfulPairs(spells []int, potions []int, success int64) []int {
	sort.Ints(potions)
	m := len(potions)
	ans := make([]int, len(spells))

	for i, s := range spells {
		if s == 0 {
			ans[i] = 0
			continue
		}
		need := (success + int64(s) - 1) / int64(s)
		idx := sort.Search(m, func(j int) bool {
			return int64(potions[j]) >= need
		})
		ans[i] = m - idx
	}
	return ans
}

func main() {
	fmt.Println(successfulPairs([]int{5, 1, 3}, []int{1, 2, 3, 4, 5}, 7)) // [4 0 3]
}
```

---

### 场景 3：前端优惠券 × 商品匹配（JavaScript）

**背景**  
在前端你有：

- `spells[i]`：商品价格；
- `potions[j]`：折扣倍数（可近似映射为整数、比方说扩大 100 倍做整数算再除回去）；
- 你需要计算每个商品能和多少优惠券组合，使得**折后价乘以某个指标仍然 ≥ success**。

虽然真实逻辑可能更复杂，但核心都是「一个数组排序，另外一个数组的每个元素用二分找到阈值位置」。

**示例代码**

```js
function successfulPairs(spells, potions, success) {
  potions.sort((a, b) => a - b);
  const m = potions.length;
  const ans = [];

  for (const s of spells) {
    if (s === 0) {
      ans.push(0);
      continue;
    }
    const need = Math.ceil(success / s);
    let l = 0, r = m;
    while (l < r) {
      const mid = (l + r) >> 1;
      if (potions[mid] >= need) r = mid;
      else l = mid + 1;
    }
    ans.push(m - l);
  }
  return ans;
}

console.log(successfulPairs([5, 1, 3], [1, 2, 3, 4, 5], 7)); // [4, 0, 3]
```

---

## R — Reflection（反思与深入）

### 1. 复杂度分析

- 排序 `potions`：O(m log m)
- 对每个咒语进行二分：O(n log m)
- 总体时间复杂度：**O((n + m) log m)**  
  在 `n, m ~ 1e5` 的情况下完全可行。

- 空间复杂度：  
  - 原地排序时，额外空间 O(1)（或 O(log m) 递归栈）。

相比之下，**暴力双重循环**的复杂度为 O(n·m)，在大规模数据时会直接超时或导致请求超时。

---

### 2. 替代方案与常见错误

**暴力法**：  

```text
for s in spells:
    cnt = 0
    for p in potions:
        if s * p >= success:
            cnt++
    ans[i] = cnt
```

- 时间复杂度 O(n·m)；
- 完全没利用 potions 数组的可重用性和有序性。

**双指针 + 排序（另一种思路）**

- 也可以同时排序 `spells`（带原下标）和 `potions`，使用双指针从右往左移动，统计每个咒语能匹配的药水个数；
- 这也是常见解法之一，但实现上更容易写错边界，对初学者不如二分方案直观。

**常见错误**

1. **直接计算乘积比较导致溢出**  
   - 如果 `spells[i]` 和 `potions[j]` 都是 1e9 级别，用 32 位整数相乘会溢出；
   - 推荐用除法：`p >= ceil(success / s)`，从而避免 `s * p` 直接计算；

2. **向上取整写错**  
   - 常见错误写法：`need = success / s`（这是向下取整，会漏掉边界值）；  
   - 正确：`need = (success + s - 1) // s`。

3. **忘记排序 `potions` 再二分**  
   - 二分查找必须建立在有序数组之上，否则结果不可预测。

4. **忽略 `s == 0` 的情况**  
   - 若题目允许 `spells` 中出现 0，需要特判：0 乘以任何非负数都不可能 ≥ 正数 `success`。

---

### 3. 为什么“排序 + 二分”更工程可行？

- **可读性好**：  
  代码结构清晰：排序一次 + 每个元素做一次二分；  
  很容易被团队中其他人理解和复用。

- **性能稳健**：  
  二分查找操作的复杂度非常稳定，主要耗时集中在排序上；  
  在数据量很大时也有良好表现。

- **可扩展性强**：  
  许多类似 “a * b ≥ C” / “a + b ≥ C” / “b ≥ f(a)” 的匹配计数问题都可以按同样方式建模。

---

## S — Summary（总结）

- 把 `spells[i] * potions[j] >= success` 转换为 `potions[j] >= ceil(success / spells[i])` 是本题的核心。
- 通过对 `potions` 排序，并对每个咒语用二分查找找“第一个 ≥ need 的位置”，我们可以在 O((n+m)log m) 时间内解题。
- 相比暴力 O(n·m) 的双重循环，排序 + 二分在大数据场景下更具工程价值。
- 注意处理整数向上取整、溢出风险以及 `s == 0` 等边界条件。
- 这种模式可以迁移到价格组合、风险额度、资源匹配等多个业务场景。

---

## 参考与延伸阅读

- LeetCode 2300. Successful Pairs of Spells and Potions（原题）  
- 其他典型的「排序 + 二分计数」问题：
  - 三数之和 / 四数之和中的去重与剪枝
  - 统计区间内小于 / 大于某值的元素个数
- 《算法导论》排序与二分查找章节  
- 各主流语言标准库的二分查找接口：`bisect`（Python）、`std::lower_bound`（C++）、`sort.Search`（Go）等

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

### Python 实现

```python
from bisect import bisect_left
from typing import List


def successful_pairs(spells: List[int], potions: List[int], success: int) -> List[int]:
    potions.sort()
    m = len(potions)
    ans = []

    for s in spells:
        if s == 0:
            ans.append(0)
            continue
        need = (success + s - 1) // s
        idx = bisect_left(potions, need)
        ans.append(m - idx)

    return ans


if __name__ == "__main__":
    print(successful_pairs([5, 1, 3], [1, 2, 3, 4, 5], 7))  # [4, 0, 3]
```

---

### C 实现

```c
#include <stdio.h>
#include <stdlib.h>

int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

int lower_bound(int *arr, int n, int target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] >= target) r = mid;
        else l = mid + 1;
    }
    return l;
}

void successfulPairs(int *spells, int n, int *potions, int m, long long success, int *ans) {
    qsort(potions, m, sizeof(int), cmp_int);

    for (int i = 0; i < n; ++i) {
        int s = spells[i];
        if (s == 0) {
            ans[i] = 0;
            continue;
        }
        long long need_ll = (success + s - 1) / s;
        if (need_ll > potions[m - 1]) {
            ans[i] = 0;
            continue;
        }
        int need = (int)need_ll;
        int idx = lower_bound(potions, m, need);
        ans[i] = m - idx;
    }
}

int main(void) {
    int spells[] = {5, 1, 3};
    int potions[] = {1, 2, 3, 4, 5};
    int n = 3, m = 5;
    int ans[3];

    successfulPairs(spells, n, potions, m, 7, ans);
    for (int i = 0; i < n; ++i) {
        printf("%d ", ans[i]);
    }
    printf("\n"); // 4 0 3
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> successfulPairs(vector<int> spells, vector<int> potions, long long success) {
    sort(potions.begin(), potions.end());
    int m = (int)potions.size();
    vector<int> ans;
    ans.reserve(spells.size());

    for (int s : spells) {
        if (s == 0) {
            ans.push_back(0);
            continue;
        }
        long long need = (success + s - 1) / s;
        auto it = lower_bound(potions.begin(), potions.end(), (int)need);
        ans.push_back((int)(potions.end() - it));
    }
    return ans;
}

int main() {
    vector<int> spells{5, 1, 3};
    vector<int> potions{1, 2, 3, 4, 5};
    auto ans = successfulPairs(spells, potions, 7);
    for (int x : ans) cout << x << " ";
    cout << endl; // 4 0 3
    return 0;
}
```

---

### Go 实现

```go
package main

import (
	"fmt"
	"sort"
)

func successfulPairs(spells []int, potions []int, success int64) []int {
	sort.Ints(potions)
	m := len(potions)
	ans := make([]int, len(spells))

	for i, s := range spells {
		if s == 0 {
			ans[i] = 0
			continue
		}
		need := (success + int64(s) - 1) / int64(s)
		idx := sort.Search(m, func(j int) bool {
			return int64(potions[j]) >= need
		})
		ans[i] = m - idx
	}
	return ans
}

func main() {
	fmt.Println(successfulPairs([]int{5, 1, 3}, []int{1, 2, 3, 4, 5}, 7)) // [4 0 3]
}
```

---

### Rust 实现

```rust
fn successful_pairs(spells: Vec<i32>, mut potions: Vec<i32>, success: i64) -> Vec<i32> {
    potions.sort();
    let m = potions.len();
    let mut ans = Vec::with_capacity(spells.len());

    for s in spells {
        if s == 0 {
            ans.push(0);
            continue;
        }
        let need = (success + s as i64 - 1) / s as i64;
        let idx = potions
            .binary_search_by(|&p| {
                if (p as i64) < need {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|i| i);
        ans.push((m - idx) as i32);
    }
    ans
}

fn main() {
    let spells = vec![5, 1, 3];
    let potions = vec![1, 2, 3, 4, 5];
    let ans = successful_pairs(spells, potions, 7);
    println!("{:?}", ans); // [4, 0, 3]
}
```

---

### JavaScript 实现

```js
function successfulPairs(spells, potions, success) {
  potions.sort((a, b) => a - b);
  const m = potions.length;
  const ans = [];

  for (const s of spells) {
    if (s === 0) {
      ans.push(0);
      continue;
    }
    const need = Math.ceil(success / s);
    let l = 0, r = m;
    while (l < r) {
      const mid = (l + r) >> 1;
      if (potions[mid] >= need) r = mid;
      else l = mid + 1;
    }
    ans.push(m - l);
  }
  return ans;
}

console.log(successfulPairs([5, 1, 3], [1, 2, 3, 4, 5], 7)); // [4, 0, 3]
```

---

## 行动号召（CTA）

- 把这道题按你最熟悉的语言手写一遍，并把「排序 + 下界二分」抽成一个通用工具函数。
- 回顾你项目中的阈值匹配逻辑，看能否用“先排序再二分计数”的方式重构一处性能瓶颈。
- 挑几道类似的「≥ 阈值配对计数」题（如配对和 ≥ K、配对差值 ≤ K），试着用同样思路建模并实现。

