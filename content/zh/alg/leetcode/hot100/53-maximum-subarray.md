---
title: "Hot100：最大子数组和（Maximum Subarray）Kadane 一维 DP ACERS 解析"
date: 2026-01-23T11:58:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "动态规划", "贪心", "子数组", "LeetCode 53"]
description: "用 Kadane 算法在 O(n) 时间求最大子数组和，含工程场景、常见误区与多语言实现。"
keywords: ["Maximum Subarray", "最大子数组和", "Kadane", "动态规划", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 最大子数组和是最经典的一维 DP / 贪心题。本文用 ACERS 模板拆解 Kadane 算法，给出工程迁移思路与多语言可运行实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`动态规划`、`贪心`  
- **SEO 关键词**：Maximum Subarray, 最大子数组和, Kadane, 动态规划, O(n)  
- **元描述**：Kadane 一维 DP 求最大子数组和，含工程场景、复杂度分析与多语言代码。  

---

## 目标读者

- 正在刷 Hot100 的学习者  
- 想掌握“最大子段和”经典模板的中级开发者  
- 需要做序列区间增益分析的工程师

## 背景 / 动机

最大子数组和不仅是 LeetCode 经典题，也常见于实际系统：  
交易收益区间、指标提升区间、日志峰值段落、吞吐提升区间等都可以抽象为“最大连续收益”。  
朴素 O(n^2) 枚举无法扩展，Kadane 给出 O(n) 的线性解。

## 核心概念

- **子数组**：连续且非空的数组片段  
- **状态转移**：`dp[i]` 表示“以 i 结尾的最大子数组和”  
- **Kadane 思想**：如果前缀和为负，直接丢弃，从当前位置重新开始

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个整数数组 `nums`，找出一个具有最大和的连续子数组（子数组至少包含一个元素），返回其最大和。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| 返回 | int | 最大子数组和 |

### 示例 1（官方）

```text
nums = [-2,1,-3,4,-1,2,1,-5,4]
输出 = 6
解释：子数组 [4,-1,2,1] 和为 6
```

### 示例 2（官方）

```text
nums = [1]
输出 = 1
```

---

## C — Concepts（核心思想）

### 关键公式

设 `dp[i]` 为以 `i` 结尾的最大子数组和：

```
dp[i] = max(nums[i], dp[i-1] + nums[i])
答案 = max(dp[i])
```

### 方法归类

- **一维动态规划（DP）**  
- **贪心（负前缀直接舍弃）**

### 直观解释

如果 `dp[i-1]` 为负数，继续加只会让后面的和变小，直接从 `nums[i]` 重新开始更优。  
这就是 Kadane 的本质。

---

## 实践指南 / 步骤

1. 初始化 `cur = nums[0]`、`best = nums[0]`  
2. 从第 2 个元素开始扫描：  
   - `cur = max(nums[i], cur + nums[i])`  
   - `best = max(best, cur)`  
3. 返回 `best`

Python 可运行示例（保存为 `maximum_subarray.py`）：

```python
def max_subarray(nums):
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


if __name__ == "__main__":
    print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print(max_subarray([1]))
```

---

## E — Engineering（工程应用）

### 场景 1：交易收益区间（Python，数据分析）

**背景**：用日收益序列找“最优连续盈利区间”。  
**为什么适用**：最大子数组和直接给出收益峰值段。

```python
def best_profit_streak(deltas):
    cur = best = deltas[0]
    for x in deltas[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

print(best_profit_streak([3, -2, 5, -1, 2, -4, 6]))
```

### 场景 2：系统监控峰值段（C++，高性能）

**背景**：CPU 使用率变化序列中寻找最大连续上升段。  
**为什么适用**：O(n) 扫描适合高频采样。

```cpp
#include <iostream>
#include <vector>

int maxBurst(const std::vector<int>& deltas) {
    int cur = deltas[0];
    int best = deltas[0];
    for (size_t i = 1; i < deltas.size(); ++i) {
        cur = std::max(deltas[i], cur + deltas[i]);
        best = std::max(best, cur);
    }
    return best;
}

int main() {
    std::cout << maxBurst({3, -2, 5, -1, 2}) << "\n";
    return 0;
}
```

### 场景 3：后端吞吐提升区间（Go，后台服务）

**背景**：请求 QPS 的差分序列中找“最大连续提升”。  
**为什么适用**：Kadane 适合在线聚合。

```go
package main

import "fmt"

func maxIncrease(deltas []int) int {
    cur := deltas[0]
    best := deltas[0]
    for i := 1; i < len(deltas); i++ {
        if cur+deltas[i] > deltas[i] {
            cur += deltas[i]
        } else {
            cur = deltas[i]
        }
        if cur > best {
            best = cur
        }
    }
    return best
}

func main() {
    fmt.Println(maxIncrease([]int{3, -2, 5, -1, 2}))
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 暴力枚举 | 枚举所有子数组 | O(n^2) | 规模大就不可用 |
| 前缀和枚举 | 计算区间和 | O(n^2) | 仍然太慢 |
| 分治 | 左右递归合并 | O(n log n) | 实现复杂 |
| **Kadane** | 一维 DP | **O(n)** | 最优且简单 |

### 为什么当前方法最优

- 单次扫描，常数空间  
- 实现简单、易复用  
- 可直接迁移到流式数据

---

## 解释与原理（为什么这么做）

Kadane 的关键是“负贡献丢弃”：  
如果当前累积和 `cur` 为负，它只会拖累后续区间，因此从当前元素重新开始更优。  
这等价于 `dp[i] = max(nums[i], dp[i-1] + nums[i])` 的状态转移。

---

## 常见问题与注意事项

1. **全是负数怎么办？**  
   依然成立，答案是最大（最不负）的那个数。

2. **能否允许空子数组？**  
   本题不允许，必须至少包含一个元素。

3. **是否需要记录区间位置？**  
   可以额外维护起止索引，但会增加代码复杂度。

---

## 最佳实践与建议

- 用 `cur` / `best` 两变量即可完成  
- 遇到相似问题先做“差分序列”建模，再套 Kadane  
- 若需要区间索引，可在更新 `cur` 时记录起点

---

## S — Summary（总结）

### 核心收获

- 最大子数组和是经典一维 DP 模板题  
- Kadane 通过“负贡献丢弃”实现 O(n)  
- 时间 O(n)、空间 O(1) 可直接工程复用  
- 全负数组也能正确处理  
- 适用于收益、吞吐、波动等连续增益分析

### 小结 / 结论

掌握 Kadane 就掌握了“连续最优区间”的核心套路。  
这是一道刷题与工程迁移价值都很高的题。

### 参考与延伸阅读

- LeetCode 53. Maximum Subarray
- 经典 DP 教材（最大子段和）
- 《算法导论》分治法与 Kadane 对比

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、动态规划、贪心、子数组  
- **SEO 关键词**：Maximum Subarray, 最大子数组和, Kadane, O(n)  
- **元描述**：Kadane 一维 DP 求最大子数组和，含工程场景与多语言实现。  

---

## 行动号召（CTA）

如果你在做 Hot100，建议把这类“连续最优区间”模板整理成自己的刷题工具箱。  
欢迎评论区分享你在工程里的使用场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def max_subarray(nums):
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


if __name__ == "__main__":
    print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
```

```c
#include <stdio.h>

int max_subarray(const int *nums, int n) {
    int cur = nums[0];
    int best = nums[0];
    for (int i = 1; i < n; ++i) {
        int with_cur = cur + nums[i];
        cur = nums[i] > with_cur ? nums[i] : with_cur;
        if (cur > best) best = cur;
    }
    return best;
}

int main(void) {
    int nums[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    printf("%d\n", max_subarray(nums, 9));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

int maxSubArray(const std::vector<int>& nums) {
    int cur = nums[0];
    int best = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        cur = std::max(nums[i], cur + nums[i]);
        best = std::max(best, cur);
    }
    return best;
}

int main() {
    std::vector<int> nums = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    std::cout << maxSubArray(nums) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func maxSubArray(nums []int) int {
    cur := nums[0]
    best := nums[0]
    for i := 1; i < len(nums); i++ {
        if cur+nums[i] > nums[i] {
            cur += nums[i]
        } else {
            cur = nums[i]
        }
        if cur > best {
            best = cur
        }
    }
    return best
}

func main() {
    nums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
    fmt.Println(maxSubArray(nums))
}
```

```rust
fn max_subarray(nums: &[i32]) -> i32 {
    let mut cur = nums[0];
    let mut best = nums[0];
    for &x in &nums[1..] {
        let with_cur = cur + x;
        cur = if x > with_cur { x } else { with_cur };
        if cur > best {
            best = cur;
        }
    }
    best
}

fn main() {
    let nums = vec![-2, 1, -3, 4, -1, 2, 1, -5, 4];
    println!("{}", max_subarray(&nums));
}
```

```javascript
function maxSubArray(nums) {
  let cur = nums[0];
  let best = nums[0];
  for (let i = 1; i < nums.length; i++) {
    cur = Math.max(nums[i], cur + nums[i]);
    best = Math.max(best, cur);
  }
  return best;
}

console.log(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]));
```
