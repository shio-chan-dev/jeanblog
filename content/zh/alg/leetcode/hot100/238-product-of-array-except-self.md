---
title: "Hot100：除自身以外数组的乘积（Product of Array Except Self）前后缀乘积 ACERS 解析"
date: 2026-01-24T12:35:09+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "数组", "前缀乘积", "后缀乘积", "空间优化", "LeetCode 238"]
description: "O(n) 时间、无需除法，用前后缀乘积求解除自身以外数组的乘积，含工程场景与多语言实现。"
keywords: ["Product of Array Except Self", "除自身以外数组的乘积", "前缀乘积", "后缀乘积", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 除自身以外数组的乘积是典型的前后缀乘积题：不使用除法，在 O(n) 时间内完成。本文按 ACERS 结构拆解题意与算法，并给出工程迁移场景与多语言实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`数组`、`前缀乘积`  
- **SEO 关键词**：Product of Array Except Self, 除自身以外数组的乘积, 前缀乘积, 后缀乘积, O(n)  
- **元描述**：用前后缀乘积在 O(n) 时间内解决除自身以外数组的乘积问题，含工程场景与多语言代码。  

---

## 目标读者

- 正在刷 Hot100 的学习者  
- 想掌握“前后缀乘积”模型的中级开发者  
- 需要做序列因子组合与乘积聚合的工程师

## 背景 / 动机

很多业务需要“排除自身的整体乘积”：  
例如组合指标、冗余度评估、批量权重计算等。  
若直接对每个位置做一次全数组相乘，复杂度会退化为 O(n^2)；  
而题目还明确禁止使用除法，因此必须依赖前后缀乘积的线性解法。

## 核心概念

- **前缀乘积**：`prefix[i] = nums[0] * ... * nums[i-1]`  
- **后缀乘积**：`suffix[i] = nums[i+1] * ... * nums[n-1]`  
- **无除法**：只允许乘法与遍历  
- **空间优化**：用结果数组承载前缀，再用后缀补乘

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个整数数组 `nums`，返回数组 `answer`，  
其中 `answer[i]` 等于 `nums` 中**除了** `nums[i]` 之外其余各元素的乘积。  
题目保证任意元素的前缀/后缀乘积都在 32 位整数范围内。  
要求：**不使用除法**，并在 **O(n)** 时间内完成。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| 返回 | int[] | 每个位置为“除自身以外的乘积” |

### 示例 1（官方）

```text
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

### 示例 2（官方）

```text
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
```

---

## C — Concepts（核心思想）

### 核心公式

```
answer[i] = 前缀乘积(i) * 后缀乘积(i)
```

其中：

- `前缀乘积(i) = nums[0] * ... * nums[i-1]`
- `后缀乘积(i) = nums[i+1] * ... * nums[n-1]`

### 方法归类

- 前后缀乘积  
- 线性扫描  
- 空间优化（O(1) 额外空间）

### 直观理解

先从左到右写入“左侧乘积”，  
再从右到左用“右侧乘积”补齐，  
每个位置都能在 O(1) 时间完成更新。

---

## 实践指南 / 步骤

1. 初始化结果数组 `res`，初值为 1  
2. 从左到右累乘，写入每个位置的前缀乘积  
3. 从右到左累乘，把后缀乘积乘到 `res[i]`  
4. 返回 `res`

运行方式示例：

```bash
python3 product_except_self.py
```

## 可运行示例（Python）

```python
from typing import List


def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res


if __name__ == "__main__":
    print(product_except_self([1, 2, 3, 4]))
    print(product_except_self([-1, 1, 0, -3, 3]))
```

---

## 解释与原理（为什么这么做）

题目禁止使用除法，因此不能通过“总乘积 / nums[i]”来计算。  
前缀乘积记录每个位置左侧的累积乘积，后缀乘积记录右侧的累积乘积，  
二者相乘就恰好是“除自身以外的乘积”。  
使用两次线性扫描即可得到所有答案，整体复杂度为 O(n)。

---

## E — Engineering（工程应用）

### 场景 1：指标敏感度评估（Python，数据分析）

**背景**：数据分析里需要评估某项指标被移除后的综合影响。  
**为什么适用**：前后缀乘积能高效计算“排除自身”的组合结果。

```python
def leave_one_out_product(weights):
    res = [1] * len(weights)
    prefix = 1
    for i, x in enumerate(weights):
        res[i] = prefix
        prefix *= x
    suffix = 1
    for i in range(len(weights) - 1, -1, -1):
        res[i] *= suffix
        suffix *= weights[i]
    return res


print(leave_one_out_product([2, 3, 5, 7]))
```

### 场景 2：多因子评分服务（Go，后台服务）

**背景**：推荐或风控系统会合成多个因子分数，有时需要排除单个因子做对照。  
**为什么适用**：O(n) 扫描能在服务端低延迟完成计算。

```go
package main

import "fmt"

func productExceptSelf(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	prefix := 1
	for i := 0; i < n; i++ {
		res[i] = prefix
		prefix *= nums[i]
	}
	suffix := 1
	for i := n - 1; i >= 0; i-- {
		res[i] *= suffix
		suffix *= nums[i]
	}
	return res
}

func main() {
	fmt.Println(productExceptSelf([]int{1, 2, 3, 4}))
}
```

### 场景 3：前端组合权重展示（JavaScript，前端）

**背景**：前端需要展示“移除某项后的综合评分”，用于解释模型或 UI 展示。  
**为什么适用**：前缀/后缀法在浏览器里也能快速计算。

```javascript
function productExceptSelf(nums) {
  const n = nums.length;
  const res = new Array(n).fill(1);
  let prefix = 1;
  for (let i = 0; i < n; i += 1) {
    res[i] = prefix;
    prefix *= nums[i];
  }
  let suffix = 1;
  for (let i = n - 1; i >= 0; i -= 1) {
    res[i] *= suffix;
    suffix *= nums[i];
  }
  return res;
}

console.log(productExceptSelf([1, 2, 3, 4]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
- **空间复杂度**：O(1) 额外空间（不计输出数组）

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 暴力枚举 | 每个 i 扫一遍数组 | O(n^2) | 大数组不可用 |
| 使用除法 | 总乘积 / nums[i] | O(n) | 题目禁止，且有 0 的边界 |
| 左右数组 | 分别存前缀/后缀 | O(n) 空间 | 正确但额外空间多 |
| 前后缀合并 | 一数组两遍扫描 | O(n) / O(1) | 当前最优方案 |

### 为什么当前方法最优 / 最工程可行

前后缀乘积只需两次线性扫描，  
在不使用除法的前提下实现最小额外空间，  
既满足题目约束，也易于在工程中复用。

---

## 常见问题与注意事项

1. **为什么不能用除法？**  
   题目明确禁止，且有 0 的情况下除法也会失效。

2. **k=0 或空数组怎么办？**  
   空数组直接返回空结果，算法对任意长度都成立。

3. **32 位乘积保证有什么意义？**  
   说明乘积不会溢出常规 32 位整数范围，便于工程实现。

---

## 最佳实践与建议

- 结果数组可先存前缀，再乘后缀，节省空间  
- 注意从右向左时的下标边界  
- 遇到相似问题先画出“左侧 / 右侧累积”的模型  
- 单测覆盖含 0、负数、长度为 1 的场景

---

## S — Summary（总结）

### 核心收获

- 题目要求“不用除法 + O(n)”决定了前后缀乘积方案  
- `answer[i] = prefix[i] * suffix[i]` 是核心公式  
- 双遍扫描即可完成，额外空间 O(1)  
- 可迁移到“排除自身”的多种工程问题  

### 推荐延伸阅读

- LeetCode 238. Product of Array Except Self  
- 前后缀数组（Prefix/Suffix）技巧  
- 数组原地空间优化策略  

### 小结 / 结论

前后缀乘积是数组类题目中最通用的技巧之一。  
掌握它，可以快速解决“排除自身”的各类变体。

---

## 参考与延伸阅读

- https://leetcode.com/problems/product-of-array-except-self/  
- https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types  
- https://en.cppreference.com/w/cpp/algorithm  
- https://pkg.go.dev/std  

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、数组、前缀乘积、后缀乘积  
- **SEO 关键词**：Product of Array Except Self, 除自身以外数组的乘积, 前缀乘积, 后缀乘积, O(n)  
- **元描述**：前后缀乘积在 O(n) 时间内解决除自身以外数组的乘积问题。  

---

## 行动号召（CTA）

如果你在刷 Hot100，建议把“前后缀模型”整理成模板题库。  
欢迎分享你在工程里遇到的类似问题与解法。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res


if __name__ == "__main__":
    print(product_except_self([1, 2, 3, 4]))
```

```c
#include <stdio.h>

void product_except_self(const int *nums, int n, int *out) {
    int prefix = 1;
    for (int i = 0; i < n; ++i) {
        out[i] = prefix;
        prefix *= nums[i];
    }
    int suffix = 1;
    for (int i = n - 1; i >= 0; --i) {
        out[i] *= suffix;
        suffix *= nums[i];
    }
}

int main(void) {
    int nums[] = {1, 2, 3, 4};
    int out[4];
    product_except_self(nums, 4, out);
    for (int i = 0; i < 4; ++i) {
        printf("%d%s", out[i], i + 1 == 4 ? "\n" : " ");
    }
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

std::vector<int> product_except_self(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<int> res(n, 1);
    int prefix = 1;
    for (int i = 0; i < n; ++i) {
        res[i] = prefix;
        prefix *= nums[i];
    }
    int suffix = 1;
    for (int i = n - 1; i >= 0; --i) {
        res[i] *= suffix;
        suffix *= nums[i];
    }
    return res;
}

int main() {
    std::vector<int> nums{1, 2, 3, 4};
    auto res = product_except_self(nums);
    for (int x : res) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func productExceptSelf(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	prefix := 1
	for i := 0; i < n; i++ {
		res[i] = prefix
		prefix *= nums[i]
	}
	suffix := 1
	for i := n - 1; i >= 0; i-- {
		res[i] *= suffix
		suffix *= nums[i]
	}
	return res
}

func main() {
	fmt.Println(productExceptSelf([]int{1, 2, 3, 4}))
}
```

```rust
fn product_except_self(nums: &[i32]) -> Vec<i32> {
    let n = nums.len();
    let mut res = vec![1; n];
    let mut prefix = 1;
    for i in 0..n {
        res[i] = prefix;
        prefix *= nums[i];
    }
    let mut suffix = 1;
    for i in (0..n).rev() {
        res[i] *= suffix;
        suffix *= nums[i];
    }
    res
}

fn main() {
    let nums = vec![1, 2, 3, 4];
    println!("{:?}", product_except_self(&nums));
}
```

```javascript
function productExceptSelf(nums) {
  const n = nums.length;
  const res = new Array(n).fill(1);
  let prefix = 1;
  for (let i = 0; i < n; i += 1) {
    res[i] = prefix;
    prefix *= nums[i];
  }
  let suffix = 1;
  for (let i = n - 1; i >= 0; i -= 1) {
    res[i] *= suffix;
    suffix *= nums[i];
  }
  return res;
}

console.log(productExceptSelf([1, 2, 3, 4]));
```
