---
title: "LeetCode 1513：仅含 1 的子串数量（连续 1 子串计数）ACERS 解析"
date: 2026-01-18T18:42:59+08:00
draft: false
categories: ["LeetCode"]
tags: ["计数", "字符串", "连续段", "ACERS", "LeetCode 1513"]
description: "用连续段在线统计在 O(n) 时间计算仅含 1 的子串数量，含工程场景与多语言实现。"
keywords: ["Number of Substrings With Only 1s", "连续1子串", "LeetCode 1513", "计数", "O(n)"]
---

> **副标题 / 摘要**  
> 这是“连续 1 子串计数”的标准题：用 `cur` 维护以当前位置结尾的连续 1 长度即可在线累加答案。本文按 ACERS 模板给出清晰模型、工程场景与多语言实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`计数`、`字符串`、`连续段`  
- **SEO 关键词**：Number of Substrings With Only 1s, 连续1子串, LeetCode 1513  
- **元描述**：在线统计连续 1 子串数量的 O(n) 解法与工程化应用。

---

## 目标读者

- 正在刷 LeetCode / 准备面试的同学  
- 想建立“连续段计数”模板的中级开发者  
- 做日志分析、监控与行为统计的工程师

## 背景 / 动机

“只含 1 的连续子串数量”看似简单，但它对应一类非常常见的工程统计：  
连续事件强度、稳定性评分、连续活跃天数、心跳连续正常等。  
掌握这题等于掌握“连续段贡献计数”的可复用模型。

## 核心概念

- **连续子串**：必须连续，不能跳过元素  
- **连续段（run）**：一段连续的 1  
- **在线累加（cur 模型）**：记录以当前位置结尾的连续 1 长度  
- **取模**：答案可能很大，需要取 `1e9+7`

---

## A — Algorithm（题目与算法）

### 题目重述

给你一个二进制字符串 `s`，请返回 **仅由字符 '1' 组成的子串** 的数量。  
子串要求连续且非空。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| s | string | 只包含 '0' 和 '1' |
| 返回 | int | 仅含 1 的子串数量（取模） |

### 示例

```text
s = "0110111"
输出 = 9
```

解释：连续 1 段为长度 2 和 3，贡献分别为 3 和 6，总和 9。

---

## C — Concepts（核心思想）

### 方法类型

**线性扫描 + 连续段在线计数**。

### 关键模型

设 `cur` 为“以当前位置结尾的连续 1 长度”：

```text
若 s[i] == '1' -> cur += 1
若 s[i] == '0' -> cur = 0
答案累加 ans += cur
```

### 等价公式

每个连续 1 段长度为 `L`，它贡献的子串数为：

```text
L * (L + 1) / 2
```

逐位累加 `cur` 与上述公式等价，但更容易在线处理。

---

## 实践指南 / 步骤

1. 初始化 `ans = 0`，`cur = 0`  
2. 遍历字符串 `s`：
   - `s[i] == '1'`：`cur += 1`，`ans += cur`  
   - `s[i] == '0'`：`cur = 0`  
3. 每步对 `ans` 取模  
4. 返回 `ans`

---

## 可运行示例（Python）

```python
def num_sub(s: str) -> int:
    mod = 1_000_000_007
    ans = 0
    cur = 0
    for ch in s:
        if ch == "1":
            cur += 1
            ans += cur
            ans %= mod
        else:
            cur = 0
    return ans


if __name__ == "__main__":
    print(num_sub("0110111"))
```

运行方式示例：

```bash
python3 demo.py
```

---

## 解释与原理（为什么这么做）

当 `cur = L` 时，以当前位置结尾的合法子串长度可以是 `1..L`，  
因此新增子串数正好是 `L`。  
不断累加 `cur`，就等价于对每个连续段应用 `L(L+1)/2` 的公式。

这就是为什么只需一次遍历即可完成统计。

---

## E — Engineering（工程应用）

### 场景 1：用户连续活跃评分（Python，数据分析）

**背景**：将用户每日活跃标记为 0/1，统计连续活跃强度。  
**为什么适用**：连续段越长，贡献越大，在线统计成本低。

```python
def activity_score(days):
    mod = 1_000_000_007
    ans = 0
    cur = 0
    for x in days:
        if x == 1:
            cur += 1
            ans = (ans + cur) % mod
        else:
            cur = 0
    return ans


print(activity_score([0, 1, 1, 0, 1, 1, 1]))
```

### 场景 2：心跳连续正常统计（C++，系统编程）

**背景**：服务器心跳日志用 0/1 表示异常/正常，统计连续正常贡献。  
**为什么适用**：高频日志需要 O(n) 的线性处理。

```cpp
#include <iostream>
#include <vector>

long long healthScore(const std::vector<int> &beats) {
    const long long MOD = 1000000007LL;
    long long ans = 0, cur = 0;
    for (int x : beats) {
        if (x == 1) {
            cur += 1;
            ans += cur;
            ans %= MOD;
        } else {
            cur = 0;
        }
    }
    return ans;
}

int main() {
    std::vector<int> beats{1, 1, 0, 1, 1, 1};
    std::cout << healthScore(beats) << "\n";
    return 0;
}
```

### 场景 3：订单成功连续段统计（Go，后台服务）

**背景**：订单成功/失败序列用于稳定性打分。  
**为什么适用**：线上批量统计更需要稳定的 O(n) 方案。

```go
package main

import "fmt"

func successScore(flags []int) int {
	const mod = 1000000007
	ans, cur := 0, 0
	for _, x := range flags {
		if x == 1 {
			cur++
			ans += cur
			ans %= mod
		} else {
			cur = 0
		}
	}
	return ans
}

func main() {
	fmt.Println(successScore([]int{1, 0, 1, 1, 1}))
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
- 空间复杂度：O(1)

### 替代方案与对比

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力枚举子串 | O(n^2) | O(1) | 易超时 |
| 先统计连续段长度 | O(n) | O(1) | 需分段再公式 |
| 在线 cur 累加 | O(n) | O(1) | 当前方法，最直接 |

### 常见错误思路

- 忘记取模或使用 32 位整型导致溢出  
- 把“子串”写成“子序列”  
- 遇到 '0' 没有把 `cur` 清零

### 为什么当前方法最优

必须至少扫描一遍字符串才能知道连续段结构，  
因此 O(n) 是最优；在线累加让实现最简单。

---

## 常见问题与注意事项

1. **为什么要取模？**  
   当字符串很长时，答案会超过 32 位整数上限。

2. **全是 0 会怎样？**  
   `cur` 一直为 0，答案自然是 0。

3. **全是 1 会怎样？**  
   答案为 `n(n+1)/2`，这也是公式验证的边界情况。

---

## 最佳实践与建议

- 使用 64 位累加并及时取模  
- 以 `cur` 模型为模板复用到“连续段计数”类问题  
- 做边界用例：空段、全 0、全 1、交替 01

---

## S — Summary（总结）

- 只含 1 的子串统计本质是连续段贡献计数  
- `cur` 在线模型最直观且易实现  
- O(n) 时间 + O(1) 空间已达最优  
- 工程场景可直接迁移到连续事件强度统计  
- 取模与溢出处理是关键细节

### 推荐延伸阅读

- LeetCode 1513 — Number of Substrings With Only 1s  
- Run-Length Encoding（RLE）  
- Online Algorithm（在线算法思想）

---

## 小结 / 结论

掌握 `cur` 在线累加，就掌握了“连续段贡献统计”的通用解法。  
它不仅能过题，更能在工程统计场景中直接复用。

---

## 参考与延伸阅读

- https://leetcode.com/problems/number-of-substrings-with-only-1s/
- https://en.wikipedia.org/wiki/Run-length_encoding
- https://docs.python.org/3/library/stdtypes.html#str

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：计数、字符串、连续段  
- **SEO 关键词**：Number of Substrings With Only 1s, 连续1子串, LeetCode 1513  
- **元描述**：连续 1 子串计数的 O(n) 在线解法与工程应用。  

---

## 行动号召（CTA）

如果你在刷题或做日志统计，建议把这题当作“连续段计数”的模板题。  
欢迎评论区分享你遇到的类似业务场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def num_sub(s: str) -> int:
    mod = 1_000_000_007
    ans = 0
    cur = 0
    for ch in s:
        if ch == "1":
            cur += 1
            ans += cur
            ans %= mod
        else:
            cur = 0
    return ans


if __name__ == "__main__":
    print(num_sub("0110111"))
```

```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int num_sub(const char *s) {
    const int64_t MOD = 1000000007LL;
    int64_t ans = 0;
    int64_t cur = 0;
    for (size_t i = 0; s[i] != '\0'; ++i) {
        if (s[i] == '1') {
            cur += 1;
            ans += cur;
            ans %= MOD;
        } else {
            cur = 0;
        }
    }
    return (int)(ans % MOD);
}

int main(void) {
    char s[200005];
    if (scanf("%200000s", s) != 1) return 0;
    printf("%d\n", num_sub(s));
    return 0;
}
```

```cpp
#include <iostream>
#include <string>

int numSub(const std::string &s) {
    const long long MOD = 1000000007LL;
    long long ans = 0;
    long long cur = 0;
    for (char c : s) {
        if (c == '1') {
            cur += 1;
            ans += cur;
            ans %= MOD;
        } else {
            cur = 0;
        }
    }
    return (int)(ans % MOD);
}

int main() {
    std::string s;
    if (!(std::cin >> s)) return 0;
    std::cout << numSub(s) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func numSub(s string) int {
	const mod = 1000000007
	ans, cur := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == '1' {
			cur++
			ans += cur
			ans %= mod
		} else {
			cur = 0
		}
	}
	return ans
}

func main() {
	fmt.Println(numSub("0110111"))
}
```

```rust
fn num_sub(s: &str) -> i64 {
    const MOD: i64 = 1_000_000_007;
    let mut ans: i64 = 0;
    let mut cur: i64 = 0;
    for &b in s.as_bytes() {
        if b == b'1' {
            cur += 1;
            ans = (ans + cur) % MOD;
        } else {
            cur = 0;
        }
    }
    ans
}

fn main() {
    println!("{}", num_sub("0110111"));
}
```

```javascript
function numSub(s) {
  const MOD = 1000000007;
  let ans = 0;
  let cur = 0;
  for (const ch of s) {
    if (ch === "1") {
      cur += 1;
      ans = (ans + cur) % MOD;
    } else {
      cur = 0;
    }
  }
  return ans;
}

console.log(numSub("0110111"));
```
