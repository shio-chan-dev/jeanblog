---
title: "Hot100：接雨水（Trapping Rain Water）双指针 / 前后最大值 ACERS 解析"
date: 2026-01-24T10:27:35+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "双指针", "数组", "前缀最大值", "LeetCode 42"]
description: "用双指针与左右最大值在 O(n) 时间计算接雨水总量，含工程场景、常见误区与多语言实现。"
keywords: ["Trapping Rain Water", "接雨水", "双指针", "前后最大值", "O(n)", "Hot100"]
---

> **副标题 / 摘要**  
> 接雨水是最经典的“区间高度约束”题。本文按 ACERS 模板讲清双指针思路、关键公式与工程迁移，并提供多语言可运行实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`双指针`、`数组`  
- **SEO 关键词**：Trapping Rain Water, 接雨水, 双指针, 前后最大值, O(n)  
- **元描述**：双指针 O(n) 求接雨水总量，含工程场景、复杂度分析与多语言代码。  

---

## 目标读者

- 正在刷 Hot100 的学习者  
- 需要掌握“左右边界约束”模板的中级开发者  
- 处理地形/容量/水位等区间分析的工程师

## 背景 / 动机

接雨水问题本质是“每个位置能盛多少水”，与工程中的容量评估、缓冲区盈余、资源占用上限等模型高度相似。  
朴素做法每个位置都向两侧找最高，复杂度 O(n^2)。  
双指针与前后最大值可以把复杂度降到 O(n)。

## 核心概念

- **局部水位**：`water[i] = min(maxLeft[i], maxRight[i]) - height[i]`  
- **左右边界**：当前位置两侧的最高柱子决定水位上限  
- **双指针**：用左/右指针同步维护左右最大值

---

## A — Algorithm（题目与算法）

### 题目还原

给定 n 个非负整数表示每个宽度为 1 的柱子的高度，计算按此排列的柱子能接多少雨水。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| height | int[] | 柱子高度数组 |
| 返回 | int | 能接住的雨水总量 |

### 示例 1（官方）

```text
height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出 = 6
```

### 示例 2（官方）

```text
height = [4,2,0,3,2,5]
输出 = 9
```

---

## C — Concepts（核心思想）

### 关键公式

对任意位置 `i`：

```
water[i] = min(maxLeft[i], maxRight[i]) - height[i]
```

### 方法归类

- **双指针**  
- **前后最大值（边界约束）**

### 直观解释

当前位置能盛多少水由左右最高柱子中较矮的一侧决定。  
因此只要能维护左右最大值，就能在线计算水量。

---

## 实践指南 / 步骤

1. 设置双指针 `l=0, r=n-1`，并维护 `leftMax`、`rightMax`  
2. 比较 `leftMax` 和 `rightMax`：
   - 若 `leftMax <= rightMax`，当前由左侧决定，累加 `leftMax - height[l]` 并 `l++`  
   - 否则由右侧决定，累加 `rightMax - height[r]` 并 `r--`
3. 扫描结束得到总水量

Python 可运行示例（保存为 `trapping_rain_water.py`）：

```python
def trap(height):
    if not height:
        return 0
    l, r = 0, len(height) - 1
    left_max = right_max = 0
    ans = 0
    while l < r:
        left_max = max(left_max, height[l])
        right_max = max(right_max, height[r])
        if left_max <= right_max:
            ans += left_max - height[l]
            l += 1
        else:
            ans += right_max - height[r]
            r -= 1
    return ans


if __name__ == "__main__":
    print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))
    print(trap([4,2,0,3,2,5]))
```

---

## E — Engineering（工程应用）

### 场景 1：缓存容量“被占用”估算（Python，数据分析）

**背景**：将“高度序列”视为资源使用量，计算可容纳的空余容量。  
**为什么适用**：等价于左右边界约束的容量计算。

```python
def free_capacity(usage):
    return trap(usage)

print(free_capacity([2, 0, 2]))
```

### 场景 2：地形高程蓄水估计（C++，系统/仿真）

**背景**：在 1D 剖面上估算积水量。  
**为什么适用**：前后最高点限制水位上限。

```cpp
#include <iostream>
#include <vector>

int trap(const std::vector<int>& h) {
    if (h.empty()) return 0;
    int l = 0, r = (int)h.size() - 1;
    int leftMax = 0, rightMax = 0, ans = 0;
    while (l < r) {
        leftMax = std::max(leftMax, h[l]);
        rightMax = std::max(rightMax, h[r]);
        if (leftMax <= rightMax) {
            ans += leftMax - h[l];
            ++l;
        } else {
            ans += rightMax - h[r];
            --r;
        }
    }
    return ans;
}

int main() {
    std::cout << trap({0,1,0,2,1,0,1,3,2,1,2,1}) << "\n";
    return 0;
}
```

### 场景 3：后端缓冲区水位上限评估（Go，后台服务）

**背景**：请求队列高度序列中估算“可容纳的溢出量”。  
**为什么适用**：双指针 O(n) 适合在线评估。

```go
package main

import "fmt"

func trap(height []int) int {
    if len(height) == 0 {
        return 0
    }
    l, r := 0, len(height)-1
    leftMax, rightMax := 0, 0
    ans := 0
    for l < r {
        if height[l] > leftMax {
            leftMax = height[l]
        }
        if height[r] > rightMax {
            rightMax = height[r]
        }
        if leftMax <= rightMax {
            ans += leftMax - height[l]
            l++
        } else {
            ans += rightMax - height[r]
            r--
        }
    }
    return ans
}

func main() {
    fmt.Println(trap([]int{0,1,0,2,1,0,1,3,2,1,2,1}))
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
| 暴力枚举 | 每个位置向两侧找最高 | O(n^2) | 太慢 |
| 预处理数组 | 预存 maxLeft / maxRight | O(n) | 需要 O(n) 空间 |
| 单调栈 | 用栈维护凹槽 | O(n) | 代码复杂度高 |
| **双指针** | 左右最大值在线维护 | **O(n)** | 最简洁 |

### 为什么当前方法最工程可行

- 不需要额外数组  
- 单次扫描、常数空间  
- 逻辑可解释、易于调试

---

## 解释与原理（为什么这么做）

雨水高度由两侧更低的墙决定，因此本质是寻找左右最大值的下界。  
双指针法保证始终处理“较低一侧”，让已确定的边界直接参与计算。  
这样既避免了重复计算，又能保证每个位置只被处理一次。

---

## 常见问题与注意事项

1. **为什么用 `leftMax <= rightMax` 决定方向？**  
   因为较低边界决定水位，上界已经确定的一侧可以安全结算。

2. **高度为 0 会影响结果吗？**  
   不会，它只是一个普通高度值。

3. **能否允许负数高度？**  
   题目限定非负；若出现负数，需要重新定义模型。

---

## 最佳实践与建议

- 优先使用双指针版本，空间 O(1)  
- 如果需要直观可视化，可先用预处理数组  
- 工程迁移时，先明确“边界上限”的语义

---

## S — Summary（总结）

### 核心收获

- 接雨水问题的关键在于左右边界最小值  
- 双指针能在 O(n) 内完成并节省空间  
- 适用于各种“容量上限”计算场景  
- 预处理与单调栈是可选的替代方案  
- 工程上更推荐双指针

### 小结 / 结论

掌握接雨水的双指针解法，就能快速迁移到容量评估、峰谷填充等问题中。  
这也是 Hot100 必背的经典模板。

### 参考与延伸阅读

- LeetCode 42. Trapping Rain Water
- 单调栈与区间极值问题
- 地形分析与水位建模基础

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：Hot100、双指针、数组、前后最大值  
- **SEO 关键词**：Trapping Rain Water, 接雨水, 双指针, O(n)  
- **元描述**：双指针 O(n) 求接雨水总量，含工程场景与多语言实现。  

---

## 行动号召（CTA）

如果你在刷 Hot100，建议把“边界约束类”问题整理成自己的模板库。  
欢迎评论区分享你的工程迁移思路。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def trap(height):
    if not height:
        return 0
    l, r = 0, len(height) - 1
    left_max = right_max = 0
    ans = 0
    while l < r:
        left_max = max(left_max, height[l])
        right_max = max(right_max, height[r])
        if left_max <= right_max:
            ans += left_max - height[l]
            l += 1
        else:
            ans += right_max - height[r]
            r -= 1
    return ans


if __name__ == "__main__":
    print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))
```

```c
#include <stdio.h>

int trap(const int *h, int n) {
    if (n == 0) return 0;
    int l = 0, r = n - 1;
    int leftMax = 0, rightMax = 0, ans = 0;
    while (l < r) {
        if (h[l] > leftMax) leftMax = h[l];
        if (h[r] > rightMax) rightMax = h[r];
        if (leftMax <= rightMax) {
            ans += leftMax - h[l];
            ++l;
        } else {
            ans += rightMax - h[r];
            --r;
        }
    }
    return ans;
}

int main(void) {
    int h[] = {0,1,0,2,1,0,1,3,2,1,2,1};
    printf("%d\n", trap(h, 12));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

int trap(const std::vector<int>& h) {
    if (h.empty()) return 0;
    int l = 0, r = (int)h.size() - 1;
    int leftMax = 0, rightMax = 0, ans = 0;
    while (l < r) {
        leftMax = std::max(leftMax, h[l]);
        rightMax = std::max(rightMax, h[r]);
        if (leftMax <= rightMax) {
            ans += leftMax - h[l];
            ++l;
        } else {
            ans += rightMax - h[r];
            --r;
        }
    }
    return ans;
}

int main() {
    std::cout << trap({0,1,0,2,1,0,1,3,2,1,2,1}) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func trap(height []int) int {
    if len(height) == 0 {
        return 0
    }
    l, r := 0, len(height)-1
    leftMax, rightMax := 0, 0
    ans := 0
    for l < r {
        if height[l] > leftMax {
            leftMax = height[l]
        }
        if height[r] > rightMax {
            rightMax = height[r]
        }
        if leftMax <= rightMax {
            ans += leftMax - height[l]
            l++
        } else {
            ans += rightMax - height[r]
            r--
        }
    }
    return ans
}

func main() {
    fmt.Println(trap([]int{0,1,0,2,1,0,1,3,2,1,2,1}))
}
```

```rust
fn trap(height: &[i32]) -> i32 {
    if height.is_empty() {
        return 0;
    }
    let mut l: i32 = 0;
    let mut r: i32 = height.len() as i32 - 1;
    let mut left_max = 0;
    let mut right_max = 0;
    let mut ans = 0;
    while l < r {
        let li = l as usize;
        let ri = r as usize;
        if height[li] > left_max {
            left_max = height[li];
        }
        if height[ri] > right_max {
            right_max = height[ri];
        }
        if left_max <= right_max {
            ans += left_max - height[li];
            l += 1;
        } else {
            ans += right_max - height[ri];
            r -= 1;
        }
    }
    ans
}

fn main() {
    let h = vec![0,1,0,2,1,0,1,3,2,1,2,1];
    println!("{}", trap(&h));
}
```

```javascript
function trap(height) {
  if (height.length === 0) return 0;
  let l = 0;
  let r = height.length - 1;
  let leftMax = 0;
  let rightMax = 0;
  let ans = 0;
  while (l < r) {
    leftMax = Math.max(leftMax, height[l]);
    rightMax = Math.max(rightMax, height[r]);
    if (leftMax <= rightMax) {
      ans += leftMax - height[l];
      l++;
    } else {
      ans += rightMax - height[r];
      r--;
    }
  }
  return ans;
}

console.log(trap([0,1,0,2,1,0,1,3,2,1,2,1]));
```
