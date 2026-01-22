---
title: "固定间距 1 检测：一次扫描判断 1 之间至少 k 个间隔（LeetCode 1437）"
date: 2026-01-22T09:12:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["数组", "双指针", "贪心", "LeetCode 1437"]
description: "用一次扫描判断所有 1 之间是否至少相隔 k 个位置，含工程场景、常见误区与多语言实现。"
keywords: ["Check If All 1's Are at Least Length K Places Away", "固定间距 1 检测", "事件间距", "O(n)"]
---

> **副标题 / 摘要**  
> 固定间距 1 检测是典型的“事件间距校验”模型。本文按 ACERS 结构拆解题意、原理与工程迁移，并给出多语言可运行实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`数组`、`双指针`、`事件间距`  
- **SEO 关键词**：固定间距 1 检测, 事件间距, LeetCode 1437, O(n)  
- **元描述**：一次扫描判断所有 1 是否至少相隔 k 个位置，含工程场景、复杂度对比与多语言代码。  

---

## 目标读者

- 刷 LeetCode 并希望沉淀“模板题”的学习者  
- 做监控/风控/行为分析的工程师  
- 需要判断事件间隔是否合规的系统开发者

## 背景 / 动机

许多系统都有“事件不能过密”的约束：例如登录失败、报警事件、敏感操作、API 调用等。  
这类问题的本质是 **“事件间距是否满足阈值”**，与该题完全等价。  
如果能用 O(n) 一次扫描完成校验，就能直接迁移到实时系统。

## 核心概念

- **事件间距**：两个事件之间至少有 `k` 个“空位”  
- **在线校验**：只记住上一次事件的位置即可  
- **边界处理**：初始化 `last = -k-1`，消除首个事件特判

---

## A — Algorithm（题目与算法）

### 题目还原

给定整数数组 `nums` 与整数 `k`，若任意两个 `1` 之间至少有 `k` 个 `0`（等价于两次 `1` 的索引差 `> k`），返回 `true`，否则返回 `false`。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 仅包含 0/1 的数组 |
| k | int | 需要的最小间隔 |
| 返回 | bool | 是否满足间距约束 |

### 示例 1

```text
nums = [1,0,0,0,1,0,0,1], k = 2
输出: true
```

### 示例 2

```text
nums = [1,0,1], k = 2
输出: false
```

---

## C — Concepts（核心思想）

### 关键观察

- 只需要记住 **上一个 1 的索引** `last`  
- 当遇到新的 `1`：若 `i - last <= k`，说明间隔不足  
- 否则更新 `last = i`

### 方法归类

- **单次线性扫描（One-pass Scan）**  
- **事件间距校验（Event Spacing Check）**  
- **双指针 / 贪心（Greedy with last pointer）**

### 数学表达

若 `i` 和 `j` 是两个 1 的索引（`i < j`），要求：

```
(j - i - 1) >= k  ⇔  (j - i) > k
```

因此检查条件为：

```
if i - last <= k: return false
```

---

## 实践指南 / 步骤

1. 初始化 `last = -k - 1`（避免首个 1 特判）  
2. 从左到右扫描数组  
3. 遇到 `1` 时判断间隔：若 `i - last <= k` 返回 `false`  
4. 否则更新 `last = i`  
5. 扫描结束仍未冲突则返回 `true`

Python 可运行示例（保存为 `k_length_apart.py`）：

```python
def k_length_apart(nums, k):
    last = -k - 1
    for i, x in enumerate(nums):
        if x == 1:
            if i - last <= k:
                return False
            last = i
    return True


if __name__ == "__main__":
    print(k_length_apart([1, 0, 0, 0, 1, 0, 0, 1], 2))  # True
    print(k_length_apart([1, 0, 1], 2))                  # False
```

---

## E — Engineering（工程应用）

### 场景 1：风控系统（登录失败间隔校验，Python）

**背景**：登录失败事件过密可能是暴力破解。  
**为什么适用**：只需记录上一条失败事件位置即可，适合流式日志。

```python
def check_login_spacing(events, k):
    last = -k - 1
    for i, x in enumerate(events):
        if x != 1:
            continue
        if i - last <= k:
            return False
        last = i
    return True
```

### 场景 2：系统监控（错误事件密度，Go）

**背景**：服务错误不能在短时间内连续出现。  
**为什么适用**：O(1) 状态即可完成密度检测。

```go
package main

import "fmt"

func okSpacing(log []int, k int) bool {
    last := -k - 1
    for i, x := range log {
        if x == 1 {
            if i-last <= k {
                return false
            }
            last = i
        }
    }
    return true
}

func main() {
    fmt.Println(okSpacing([]int{1, 0, 0, 1}, 2))
}
```

### 场景 3：嵌入式采样去抖（C）

**背景**：传感器触发不能过密，否则认为抖动。  
**为什么适用**：适合资源受限环境，空间 O(1)。

```c
#include <stdio.h>

int k_length_apart(const int *a, int n, int k) {
    int last = -k - 1;
    for (int i = 0; i < n; ++i) {
        if (a[i] == 1) {
            if (i - last <= k) return 0;
            last = i;
        }
    }
    return 1;
}

int main(void) {
    int a[] = {1,0,0,1};
    printf("%d\n", k_length_apart(a, 4, 2));
    return 0;
}
```

### 场景 4：前端防抖触发（JavaScript）

**背景**：按钮点击过密需要抑制。  
**为什么适用**：把点击序列映射为 0/1，直接复用算法。

```javascript
function okSpacing(events, k) {
  let last = -k - 1;
  for (let i = 0; i < events.length; i++) {
    if (events[i] === 1) {
      if (i - last <= k) return false;
      last = i;
    }
  }
  return true;
}

console.log(okSpacing([1, 0, 0, 1], 2));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 记录所有 1 的索引 | 先存索引再遍历 | O(n) | 额外空间浪费 |
| 双重循环 | 每对 1 做检查 | O(n^2) | 规模大时不可用 |
| **单次扫描** | 只记 last | **O(n)** | 最简洁、最稳 |

### 为什么当前方法最优

- 不需要额外数据结构  
- 可以流式处理，适合实时系统  
- 逻辑简洁，边界易处理

---

## 解释与原理（为什么这么做）

只要记住“上一个 1 的位置”即可判断当前 1 是否过近。  
初始化 `last = -k-1` 等价于在数组左侧放一个“虚拟的 1”，这样首个 1 总是合法，避免特判。  
当出现 `i - last <= k` 时，说明两次 1 之间的空位数量小于 k，直接失败。

---

## 常见问题与注意事项

1. **为什么是 `i - last <= k`？**  
   间隔至少 k 个 0 等价于 `(i - last - 1) >= k`，移项后就是 `i - last > k`。

2. **k = 0 合法吗？**  
   合法，表示相邻 1 也允许。

3. **数组不是 0/1 可以用吗？**  
   可以，约定“事件发生”的值为 1 即可。

---

## 最佳实践与建议

- 用 `last = -k-1` 消除首元素特判  
- 将“事件间距校验”封装成通用函数  
- 需要更复杂规则时，可扩展为“最小间隔 + 最大密度”组合检测

---

## S — Summary（总结）

### 核心收获

- 问题本质是“事件间距是否达标”  
- 只需记录上一次 1 的位置即可  
- 初始化 `-k-1` 可简化边界处理  
- 单次扫描即可完成，O(n)/O(1)  
- 工程中常用于风控、监控、限流、防抖

### 小结 / 结论

这道题不仅是 LeetCode 模板题，更是工程里的“事件间距检查器”。  
把它写成通用函数，你会在很多系统里复用它。

### 参考与延伸阅读

- LeetCode 1437. Check If All 1's Are at Least Length K Places Away
- Rate Limiting / Debounce / Throttle 相关文档
- Event Stream Processing 基础

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：数组、事件间距、风控、监控  
- **SEO 关键词**：固定间距 1 检测, 事件间距, LeetCode 1437, O(n)  
- **元描述**：一次扫描判断所有 1 是否至少相隔 k 个位置，含工程场景与多语言实现。  

---

## 行动号召（CTA）

如果你在做监控或风控系统，建议把这类“事件间距模型”整理成工具函数库。  
欢迎在评论区分享你遇到的实际场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def k_length_apart(nums, k):
    last = -k - 1
    for i, x in enumerate(nums):
        if x == 1:
            if i - last <= k:
                return False
            last = i
    return True


if __name__ == "__main__":
    print(k_length_apart([1, 0, 0, 0, 1, 0, 0, 1], 2))
```

```c
#include <stdio.h>

int k_length_apart(const int *a, int n, int k) {
    int last = -k - 1;
    for (int i = 0; i < n; ++i) {
        if (a[i] == 1) {
            if (i - last <= k) return 0;
            last = i;
        }
    }
    return 1;
}

int main(void) {
    int a[] = {1,0,0,1};
    printf("%d\n", k_length_apart(a, 4, 2));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

bool kLengthApart(const std::vector<int>& nums, int k) {
    int last = -k - 1;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (nums[i] == 1) {
            if (i - last <= k) return false;
            last = i;
        }
    }
    return true;
}

int main() {
    std::cout << std::boolalpha << kLengthApart({1,0,0,1}, 2) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func kLengthApart(nums []int, k int) bool {
    last := -k - 1
    for i, x := range nums {
        if x == 1 {
            if i-last <= k {
                return false
            }
            last = i
        }
    }
    return true
}

func main() {
    fmt.Println(kLengthApart([]int{1, 0, 0, 1}, 2))
}
```

```rust
fn k_length_apart(nums: &[i32], k: i32) -> bool {
    let mut last = -k - 1;
    for (i, &x) in nums.iter().enumerate() {
        if x == 1 {
            let i = i as i32;
            if i - last <= k {
                return false;
            }
            last = i;
        }
    }
    true
}

fn main() {
    let nums = vec![1, 0, 0, 1];
    println!("{}", k_length_apart(&nums, 2));
}
```

```javascript
function kLengthApart(nums, k) {
  let last = -k - 1;
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] === 1) {
      if (i - last <= k) return false;
      last = i;
    }
  }
  return true;
}

console.log(kLengthApart([1, 0, 0, 1], 2));
```
