---
title: "判断一个数是否为 2 的幂（Power of Two）：位运算 O(1) ACERS 解析"
date: 2026-01-21T13:22:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["位运算", "二进制", "数学", "LeetCode 231"]
description: "用位运算 O(1) 判断整数是否为 2 的幂，含工程场景、常见误区与多语言实现。"
keywords: ["Power of Two", "2 的幂", "位运算", "bit manipulation", "LeetCode 231"]
---

> **副标题 / 摘要**  
> 2 的幂判断是位运算最经典的模板题之一。本文按 ACERS 结构讲清原理、工程场景与常见误区，并给出可复用的多语言实现。

- **预计阅读时长**：8~12 分钟  
- **标签**：`位运算`、`二进制`、`数学`  
- **SEO 关键词**：Power of Two, 2 的幂, 位运算, bit manipulation, LeetCode 231  
- **元描述**：用位运算 O(1) 判断 2 的幂，含工程应用、复杂度分析与多语言代码。  

---

## 目标读者

- 刚开始接触位运算的算法学习者  
- 想沉淀“位运算模板题”的中级开发者  
- 在系统/后端中需要对齐、分片、容量判断的工程师

## 背景 / 动机

“2 的幂”是很多工程系统的隐含约束：哈希表容量、内存对齐、任务分片、FFT 窗口大小等。  
如果每次判断都用循环或除法，不仅慢，而且容易写出边界错误。  
位运算提供了 **O(1)** 的稳定判断，是可长期复用的基础能力。

## 核心概念

- **二进制表示**：2 的幂在二进制中只有一个 `1`，其余全是 `0`  
- **位与运算**：`n & (n - 1)` 会清除最低位的 `1`  
- **必要条件**：`n > 0`，排除 0 和负数

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个整数 `n`，判断它是否为 2 的幂。  
如果是返回 `true`，否则返回 `false`。

### 输入输出

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| n | int | 待判断整数 |
| 返回 | bool | 是否为 2 的幂 |

### 示例 1

```text
输入: n = 1
输出: true
解释: 2^0 = 1
```

### 示例 2

```text
输入: n = 12
输出: false
解释: 12 的二进制是 1100，含多个 1
```

---

## C — Concepts（核心思想）

### 核心原理：一次位运算完成判断

- 2 的幂的二进制形态：`1000...000`（只有一个 `1`）
- `n - 1` 会把这个 `1` 变成 `0`，右侧全部变成 `1`
- 因此：

```text
n      = 1000...000
n - 1  = 0111...111
n & (n - 1) = 0000...000
```

结论：

```text
n 是 2 的幂  ⟺  n > 0 且 (n & (n - 1)) == 0
```

### 方法归类

- **位运算（Bit Manipulation）**
- **位技巧（Bit Hacks）**
- **低层优化模式（Low-level Optimization）**

---

## 实践指南 / 步骤

1. **先排除非正数**：`n <= 0` 一定不是 2 的幂。  
2. **用位运算判定**：`(n & (n - 1)) == 0`。  
3. **返回布尔结果**。

Python 可运行示例（保存为 `power_of_two.py` 后运行 `python3 power_of_two.py`）：

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    print(is_power_of_two(1))   # True
    print(is_power_of_two(12))  # False
```

---

## E — Engineering（工程应用）

### 场景 1：数据分析 / 信号处理窗口大小（Python）

**背景**：FFT、卷积等算法常要求窗口长度为 2 的幂。  
**为什么适用**：快速校验窗口参数，避免运行时异常或性能退化。

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

window = 1024
if not is_power_of_two(window):
    raise ValueError("window size must be power of two")
print("ok")
```

### 场景 2：内存对齐 / 分配器块大小（C）

**背景**：内存分配器常把块大小对齐到 2 的幂。  
**为什么适用**：对齐访问更快，且便于位运算索引。

```c
#include <stdio.h>
#include <stdint.h>

int is_pow2(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int main(void) {
    printf("%d\n", is_pow2(64));  // 1
    printf("%d\n", is_pow2(48));  // 0
    return 0;
}
```

### 场景 3：后端服务的分片 / 并发度校验（Go）

**背景**：服务分片或线程数常设为 2 的幂以方便位运算路由。  
**为什么适用**：`index & (shards-1)` 比取模更快，且分布更均匀。

```go
package main

import "fmt"

func isPowerOfTwo(n int) bool {
    return n > 0 && (n&(n-1)) == 0
}

func main() {
    shards := 16
    if !isPowerOfTwo(shards) {
        panic("shards must be power of two")
    }
    fmt.Println("ok")
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(1)  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 复杂度 | 风险/缺点 |
| --- | --- | --- | --- |
| 循环除 2 | 一直除到不能整除 | O(log n) | 慢且易写错边界 |
| 计数 1 的数量 | `popcount(n)==1` | O(1) | 依赖特定指令或库 |
| 取对数 | `log2(n)` 是否为整数 | 取决于实现 | 浮点精度风险 |
| **位运算** | `n & (n - 1)` | **O(1)** | 最简洁、最稳定 |

### 为什么当前方法最优

- 常数级判断，没有循环与除法成本  
- 可迁移到多语言和系统级代码  
- 与工程场景（对齐、容量、分片）强一致

---

## 解释与原理（为什么这么做）

2 的幂在二进制里只有一个 `1`。当你执行 `n - 1` 时，最低位的 `1` 会被借位“清掉”，右侧全部变成 `1`。  
因此，`n` 与 `n-1` 的按位与结果必然为 `0`。  
只要再加上 `n > 0`，就排除了 0 和负数的干扰。

---

## 常见问题与注意事项

1. **n = 0 是否是 2 的幂？**  
   不是，必须显式排除 `n <= 0`。

2. **负数会怎样？**  
   负数在补码下会有很多 `1`，必须直接返回 `false`。

3. **浮点判断是否可靠？**  
   不可靠，`log2` 会有精度误差，不建议用。

---

## 最佳实践与建议

- 始终先判断 `n > 0`，再做位运算  
- 封装成可复用函数，便于工程中重复使用  
- 如果需要最近的 2 的幂，可扩展成“补齐到 2 的幂”的工具函数

---

## S — Summary（总结）

### 核心收获

- 2 的幂在二进制里只有一个 `1`  
- `n & (n - 1)` 能清掉最低位 `1`  
- 结合 `n > 0` 可一行完成判断  
- 工程中常见于哈希表容量、内存对齐、分片数设置  
- 位运算方法稳定、可迁移、O(1)

### 小结 / 结论

这是一道典型的“位运算模板题”。  
一旦掌握，你会在系统工程与性能优化中反复用到它。

### 参考与延伸阅读

- LeetCode 231. Power of Two
- LeetCode 191. Number of 1 Bits
- LeetCode 342. Power of Four
- 《Hacker's Delight》Bit Tricks 章节
- 《Computer Systems: A Programmer’s Perspective》位运算章节

---

## 元信息

- **阅读时长**：8~12 分钟  
- **标签**：位运算、二进制、数学、LeetCode 231  
- **SEO 关键词**：Power of Two, 2 的幂, 位运算, bit manipulation, LeetCode 231  
- **元描述**：用位运算 O(1) 判断 2 的幂，含工程应用、复杂度分析与多语言代码。  

---

## 行动号召（CTA）

如果你正在刷 LeetCode，不妨把“位运算模板题”整理成自己的工具箱。  
欢迎评论区分享你在工程中遇到的 2 的幂使用场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    print(is_power_of_two(1))   # True
    print(is_power_of_two(12))  # False
```

```c
#include <stdio.h>
#include <stdint.h>

int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int main(void) {
    printf("%d\n", is_power_of_two(1));
    printf("%d\n", is_power_of_two(12));
    return 0;
}
```

```cpp
#include <iostream>

bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int main() {
    std::cout << std::boolalpha << isPowerOfTwo(1) << "\n";
    std::cout << std::boolalpha << isPowerOfTwo(12) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func isPowerOfTwo(n int) bool {
    return n > 0 && (n&(n-1)) == 0
}

func main() {
    fmt.Println(isPowerOfTwo(1))
    fmt.Println(isPowerOfTwo(12))
}
```

```rust
fn is_power_of_two(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn main() {
    println!("{}", is_power_of_two(1));
    println!("{}", is_power_of_two(12));
}
```

```javascript
function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

console.log(isPowerOfTwo(1));
console.log(isPowerOfTwo(12));
```
