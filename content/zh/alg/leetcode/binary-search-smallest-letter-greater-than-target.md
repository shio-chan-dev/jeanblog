---
title: "比目标字母大的最小字母：有序字符数组上的二分查找技巧"
date: 2025-12-04T11:30:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["二分查找", "有序数组", "字符数组", "边界问题"]
description: "在排序字符数组中查找比目标字母大的最小字母（Find Smallest Letter Greater Than Target），支持环绕。本文用 upper_bound 二分模板稳定处理边界，并提供多语言实现和工程实践示例。"
keywords: ["Find Smallest Letter Greater Than Target", "二分查找", "upper_bound", "字符数组", "有序数组边界"]
---

> **副标题 / 摘要**  
> 这道题看似只是“找一个比目标大的字母”，本质上是经典的上界二分（upper_bound）问题：在有序字符数组中找到第一个 > target 的元素，并在找不到时从头环绕。本文给出完整的二分模板和多语言实现，帮你稳拿这类边界题。

- **预计阅读时长**：8~10 分钟  
- **适用场景标签**：`二分查找进阶`、`字符数组`、`上界查找`  
- **SEO 关键词**：find smallest letter greater than target, upper_bound, 二分查找字符数组  

---

## 目标读者与背景

**目标读者**

- 已经掌握基本二分查找，想进一步熟悉上下界（upper/lower bound）的同学；
- 在工程中需要在有序集合中找到“下一个更大值”的开发者；
- 准备中高级面试，想通过一道题统一上界二分写法的工程师。

**背景 / 动机**

很多系统都会用到“环形有序列表”的概念：

- 比如按字母排序的标签、按时间排序的分片；
- 想要找“比当前值更大的下一个值”，找不到就从头开始。

这道题「Find Smallest Letter Greater Than Target」正是这种模式的简化版，是练习上界二分的好题。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定一个按非降序排序的字符数组 `letters`，数组中的字母都是小写英文字母。  
> 给定一个字符 `target`，请你找到数组中**严格大于** `target` 的最小字母并返回。  
> 注意：`letters` 数组是**环绕**的——如果不存在这样的字母，则返回数组的第一个元素。

**输入**

- `letters`: 排序好的小写字母数组，长度为 `n`，且 `letters` 中至少有两个不同的字母；
- `target`: 一个小写字母。

**输出**

- 字符：数组中**比 `target` 大的最小字母**；若不存在，则为 `letters[0]`。

### 示例 1

```text
letters = ['c', 'f', 'j']
target  = 'a'
```

- 所有比 `'a'` 大的字母有 `['c', 'f', 'j']`；
- 其中最小的是 `'c'`。

**输出**：`'c'`

### 示例 2

```text
letters = ['c', 'f', 'j']
target  = 'c'
```

- 比 `'c'` 大的字母有 `['f', 'j']`；
- 最小的是 `'f'`。

**输出**：`'f'`

### 示例 3（环绕）

```text
letters = ['c', 'f', 'j']
target  = 'j'
```

- 没有比 `'j'` 更大的字母；
- 由于数组是环绕的，答案为第一个元素 `'c'`。

**输出**：`'c'`

---

## C — Concepts（核心思想）

### 1. 本质：上界（upper_bound）二分 + 环绕

问题可以抽象为：

> 在有序数组 `letters` 中找到第一个满足 `letters[i] > target` 的位置 `i`。  
> 如果存在这样的 `i`，返回 `letters[i]`；  
> 否则返回 `letters[0]`。

这就是典型的 **上界（Upper Bound）** 问题：

```text
upper_bound(target) = 第一个满足 letters[i] > target 的下标 i
```

实现上界的二分模板：

```text
l = 0, r = n
while l < r:
    mid = (l + r) // 2
    if letters[mid] > target:
        r = mid
    else:
        l = mid + 1
return l
```

此时：

- 若 `l < n`，则 `letters[l]` 是第一个 > target 的字母；
- 若 `l == n`，说明不存在比 target 更大的字母，需要环绕到 `letters[0]`。

### 2. 算法类型与复杂度

- 算法类型：**二分查找（upper_bound）**
- 特点：在有序数组中查找**严格大于目标**的最小元素；
- 时间复杂度：O(log n)
- 空间复杂度：O(1)

### 3. 与下界（lower_bound）的区别

- 下界：找第一个 `>= target` 的位置；
- 上界：找第一个 `> target` 的位置。

本题要求「严格大于」，因此需要上界二分。

---

## 实践指南 / 实现步骤

1. **写出 upper_bound 模板**

```text
function upper_bound(letters, target):
    l = 0, r = n
    while l < r:
        mid = (l + r) // 2
        if letters[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l
```

2. **处理环绕逻辑**

- 调用 `idx = upper_bound(letters, target)`；
- 若 `idx == n`，返回 `letters[0]`；
- 否则返回 `letters[idx]`。

3. **检查特例**

- 若 `target` 小于 `letters[0]`，则 idx 会是 0，返回 `letters[0]`；
- 若 `target` 大于等于 `letters[n-1]`，则 idx == n → 返回 `letters[0]`，完美处理环绕。

---

## E — Engineering（工程应用）

这种“找比目标大的最小元素，如果没有就从头开始”的模式在工程里也很常见。

### 场景 1：环形分片选择 / 一致性哈希（Python）

**背景**  
在一致性哈希或环形分片中：

- 你有一组按 hash 值排序的节点标记；
- 给定一个 key 的 hash，需要找到「第一个 hash 大于 key 的节点」；
- 如果没有，就从头开始（环绕）。

这与本题几乎完全相同，只不过把字符换成整数。

**示例代码**

```python
from typing import List


def next_greatest_letter(letters: List[str], target: str) -> str:
    n = len(letters)
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if letters[mid] > target:
            r = mid
        else:
            l = mid + 1
    return letters[0] if l == n else letters[l]


if __name__ == "__main__":
    print(next_greatest_letter(["c", "f", "j"], "a"))  # "c"
    print(next_greatest_letter(["c", "f", "j"], "c"))  # "f"
    print(next_greatest_letter(["c", "f", "j"], "j"))  # "c"
```

---

### 场景 2：时间轮 / Cron 表达式中的下一个触发点（Go）

**背景**  
在时间轮或类似 cron 调度中，常常有一组排序好的时间点（例如分钟或小时），你需要：

- 找到「下一个大于当前时间的触发点」；
- 如果当前时间之后没有触发点，则回到当天的第一个触发点。

用整数数组 + 上界二分，就能快速找到下一个触发时间。

**示例代码（Go，示意）**

```go
package main

import "fmt"

func nextGreaterSlot(slots []int, now int) int {
	n := len(slots)
	l, r := 0, n
	for l < r {
		mid := l + (r-l)/2
		if slots[mid] > now {
			r = mid
		} else {
			l = mid + 1
		}
	}
	if l == n {
		return slots[0]
	}
	return slots[l]
}

func main() {
	slots := []int{10, 20, 40, 50}
	fmt.Println(nextGreaterSlot(slots, 5))  // 10
	fmt.Println(nextGreaterSlot(slots, 20)) // 40
	fmt.Println(nextGreaterSlot(slots, 50)) // 10
}
```

---

### 场景 3：前端轮播图 / Banner 轮转（JavaScript）

**背景**  
在前端轮播组件中，你可能有一组按顺序排序的 Banner 编号（或权重阈值），需要根据当前状态找到「下一个」 Banner，若已到末尾则回到第一个。

用上界二分可以在常数时间内找到下一个 Banner 下标（相对于 `log n` 其实差别不大，但逻辑清晰）。

**示例代码**

```js
function nextGreatestLetter(letters, target) {
  let l = 0, r = letters.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (letters[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l === letters.length ? letters[0] : letters[l];
}

console.log(nextGreatestLetter(["c", "f", "j"], "a")); // "c"
console.log(nextGreatestLetter(["c", "f", "j"], "c")); // "f"
console.log(nextGreatestLetter(["c", "f", "j"], "j")); // "c"
```

---

## R — Reflection（反思与深入）

### 1. 复杂度分析

- 由于每次循环都将区间 `[l, r)` 长度缩小一半；
- 所需迭代次数大约为 `log₂(n)`；
- 时间复杂度：**O(log n)**；
- 空间复杂度：**O(1)**。

对于 `letters` 长度在 1e5 级别，它依然可以轻松满足性能要求。

---

### 2. 替代方案与常见错误

**线性扫描**

```text
for ch in letters:
    if ch > target:
        return ch
return letters[0]
```

- 时间复杂度 O(n)，在 n 不大时也能接受，但与题目希望的 O(log n) 相比略逊；
- 更重要的是，错过了训练上界二分的好机会。

**常见错误 1：条件写成 >=**

```text
if letters[mid] >= target: ...
```

- 这会返回第一个 **≥ target** 的字母（下界），而题目要求的是 **> target**；
- 示例中 `target = 'c'`，`letters = ['c', 'f', 'j']`，会错误地返回 `'c'`。

**常见错误 2：忽略环绕逻辑**

- 部分实现只在找到上界时返回，却忘了处理 `idx == n` 的情况；
- 有的同学会访问 `letters[idx]` 而不判断 idx 是否越界。

**常见错误 3：区间边界混乱**

- 和所有二分一样，若不统一使用 `[l, r)` 或 `[l, r]`，极易发生 off-by-one 或死循环。

---

### 3. 与其他二分题的关系

- 本题：找第一个 **>`target`** 的元素（上界）；
- Search Insert Position：找第一个 **≥`target`** 的位置（下界）；
- Search Range 结束位置：`upper_bound(target) - 1`；
- Maximum Count of Positive/Negative：也会通过上界 / 下界二分来找分界点。

可以将它们统一为：

> 在有序数组中，用二分找到“某个条件第一次成立”的位置。

本题是“条件 = `letters[i] > target`”的典型示例。

---

## S — Summary（总结）

- 「比目标字母大的最小字母」本质是一个 **上界（upper_bound）二分 + 环绕** 问题。
- 使用 `[l, r)` 区间和 `letters[mid] > target` 条件，可以稳定找到第一个 > target 的位置。
- 当上界下标等于数组长度时，表示不存在更大的元素，需要返回 `letters[0]` 实现环绕。
- 该模式在一致性哈希、时间轮调度、版本 / Banner 轮转等工程场景中非常常见。
- 与下界（二分找第一个 ≥ target）配合，可以覆盖绝大多数边界查找问题。

---

## 参考与延伸阅读

- LeetCode 744. Find Smallest Letter Greater Than Target  
- 二分查找上下界专题题目：Search Insert Position、Search Range、Maximum Count of Positive/Negative Integers  
- C++ 标准库 `std::upper_bound` 文档  
- 关于环形数组和一致性哈希的设计文章

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

### Python 实现

```python
from typing import List


def next_greatest_letter(letters: List[str], target: str) -> str:
    n = len(letters)
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if letters[mid] > target:
            r = mid
        else:
            l = mid + 1
    return letters[0] if l == n else letters[l]


if __name__ == "__main__":
    print(next_greatest_letter(["c", "f", "j"], "a"))  # "c"
```

---

### C 实现

```c
#include <stdio.h>

char nextGreatestLetter(char *letters, int lettersSize, char target) {
    int l = 0, r = lettersSize;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (letters[mid] > target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    if (l == lettersSize) return letters[0];
    return letters[l];
}

int main(void) {
    char letters[] = {'c', 'f', 'j'};
    printf("%c\n", nextGreatestLetter(letters, 3, 'a')); // c
    printf("%c\n", nextGreatestLetter(letters, 3, 'c')); // f
    printf("%c\n", nextGreatestLetter(letters, 3, 'j')); // c
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

char nextGreatestLetter(const vector<char> &letters, char target) {
    int n = (int)letters.size();
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (letters[mid] > target) r = mid;
        else l = mid + 1;
    }
    return (l == n) ? letters[0] : letters[l];
}

int main() {
    vector<char> letters{'c', 'f', 'j'};
    cout << nextGreatestLetter(letters, 'a') << endl; // c
    cout << nextGreatestLetter(letters, 'c') << endl; // f
    cout << nextGreatestLetter(letters, 'j') << endl; // c
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func nextGreatestLetter(letters []byte, target byte) byte {
	n := len(letters)
	l, r := 0, n
	for l < r {
		mid := l + (r-l)/2
		if letters[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	if l == n {
		return letters[0]
	}
	return letters[l]
}

func main() {
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'a')) // c
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'c')) // f
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'j')) // c
}
```

---

### Rust 实现

```rust
fn next_greatest_letter(letters: &[char], target: char) -> char {
    let n = letters.len();
    let mut l: usize = 0;
    let mut r: usize = n;
    while l < r {
        let mid = l + (r - l) / 2;
        if letters[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    if l == n { letters[0] } else { letters[l] }
}

fn main() {
    let letters = vec!['c', 'f', 'j'];
    println!("{}", next_greatest_letter(&letters, 'a')); // c
    println!("{}", next_greatest_letter(&letters, 'c')); // f
    println!("{}", next_greatest_letter(&letters, 'j')); // c
}
```

---

### JavaScript 实现

```js
function nextGreatestLetter(letters, target) {
  let l = 0, r = letters.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (letters[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l === letters.length ? letters[0] : letters[l];
}

console.log(nextGreatestLetter(["c", "f", "j"], "a")); // "c"
console.log(nextGreatestLetter(["c", "f", "j"], "c")); // "f"
console.log(nextGreatestLetter(["c", "f", "j"], "j")); // "c"
```

---

## 行动号召（CTA）

- 把本文的 upper_bound 模板添加到你的二分查找笔记中，并手写一遍加深印象。
- 尝试用同一个模板解决「Maximum Count of Positive/Negative Integers」中边界点的查找。
- 在你自己的项目里，找到一个“查找下一个更大元素”的逻辑，看看能否用二分 + 环绕模式简化。 

