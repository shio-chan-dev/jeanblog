---
title: "LeetCode 1456：最大元音子串数量的滑动窗口 ACERS 解析"
date: 2026-01-20T13:40:45+08:00
draft: false
categories: ["LeetCode"]
tags: ["滑动窗口", "字符串", "固定窗口", "LeetCode 1456", "ACERS"]
description: "用固定滑动窗口在 O(n) 时间求最大元音子串数量，含工程迁移与多语言实现。"
keywords: ["Maximum Number of Vowels", "最大元音子串", "滑动窗口", "固定窗口", "O(n)"]
---

> **副标题 / 摘要**  
> 最大元音子串数量是“固定窗口计数”的标准模板题。本文按 ACERS 结构讲清楚滑动窗口的核心思想，并给出工程场景与多语言实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`滑动窗口`、`字符串`、`固定窗口`  
- **SEO 关键词**：Maximum Number of Vowels, 最大元音子串, 滑动窗口, 固定窗口  
- **元描述**：滑动窗口求固定长度子串最大元音数，含工程化应用与多语言代码。  

---

## 目标读者

- 正在刷 LeetCode / Hot100 的同学  
- 想建立“固定窗口计数”模板的中级开发者  
- 需要做日志/指标窗口统计的工程师

## 背景 / 动机

固定长度窗口内的最大计数是工程里极常见的需求：  
监控系统统计异常峰值、运营分析统计活跃峰值、NLP 统计特征峰值。  
如果每次窗口都重新计算，会退化为 O(nk)。  
滑动窗口能让每步更新变成 O(1)，把整体降到 O(n)。

## 核心概念

- **固定滑动窗口**：窗口长度固定为 k，只右移一位  
- **增量更新**：进入右端元素、移除左端元素  
- **条件计数**：只统计满足条件（本题为元音）的元素数量

---

## A — Algorithm（题目与算法）

### 题目重述

给你一个字符串 `s` 和整数 `k`。  
返回长度为 `k` 的子串中，**元音字符数量的最大值**。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| s | string | 只包含小写英文字符 |
| k | int | 窗口长度 |
| 返回 | int | 任意长度为 k 的子串中最大元音数 |

### 示例 1

```text
s = "abciiidef", k = 3
输出 = 3
```

### 示例 2

```text
s = "aeiou", k = 2
输出 = 2
```

---

## C — Concepts（核心思想）

### 方法类型

**固定滑动窗口 + 条件计数**。

### 关键模型

设 `cnt` 为当前窗口内元音数量：

```text
cnt = cnt + isVowel(s[i]) - isVowel(s[i-k])
```

窗口每右移一步，只做 O(1) 的增量更新。

### 数据结构与判定

元音判定可用集合或函数：

```text
isVowel(c) = c in {a, e, i, o, u}
```

---

## 实践指南 / 步骤

1. 先统计首个长度为 `k` 窗口的元音数量  
2. 设置 `ans = cnt`  
3. 右移窗口：加入 `s[i]`，移除 `s[i-k]`  
4. 每步更新 `ans = max(ans, cnt)`  
5. 返回 `ans`

---

## 可运行示例（Python）

```python
def max_vowels(s: str, k: int) -> int:
    vowels = set("aeiou")
    cnt = sum(1 for c in s[:k] if c in vowels)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in vowels:
            cnt += 1
        if s[i - k] in vowels:
            cnt -= 1
        if cnt > ans:
            ans = cnt
    return ans


if __name__ == "__main__":
    print(max_vowels("abciiidef", 3))
```

运行方式示例：

```bash
python3 demo.py
```

---

## 解释与原理（为什么这么做）

窗口长度固定为 `k`，所以每次右移只会：  
**新增一个字符 + 移出一个字符**。  
因此计数更新只需常数时间。

对比暴力法：  
每个窗口扫描 k 个字符，复杂度 O(nk)。  
滑动窗口把它降为 O(n)，在 n 大、k 也不小的场景差距明显。

---

## E — Engineering（工程应用）

### 场景 1：日志异常峰值统计（Go，后台服务）

**背景**：统计任意 k 分钟内异常日志数量最大值。  
**为什么适用**：固定窗口峰值统计就是该模型。

```go
package main

import "fmt"

func maxErrors(flags []int, k int) int {
	cnt, ans := 0, 0
	for i, x := range flags {
		if x == 1 {
			cnt++
		}
		if i >= k {
			if flags[i-k] == 1 {
				cnt--
			}
		}
		if i >= k-1 && cnt > ans {
			ans = cnt
		}
	}
	return ans
}

func main() {
	fmt.Println(maxErrors([]int{0, 1, 1, 0, 1, 0, 1}, 3))
}
```

### 场景 2：文本特征峰值分析（Python，数据分析）

**背景**：统计任意 k 长度窗口内的“情绪词”最大数量。  
**为什么适用**：窗口固定，计数条件可替换。

```python
def max_keyword(text, k, keywords):
    s = list(text)
    cnt = sum(1 for c in s[:k] if c in keywords)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in keywords:
            cnt += 1
        if s[i - k] in keywords:
            cnt -= 1
        if cnt > ans:
            ans = cnt
    return ans


print(max_keyword("happyxxsadxxhappy", 5, set("hs")))
```

### 场景 3：前端输入实时高亮（JavaScript，前端）

**背景**：统计输入框最近 k 个字符中“敏感字符”最大值。  
**为什么适用**：前端实时响应需要 O(1) 更新。

```javascript
function maxFlag(chars, k, flagSet) {
  let cnt = 0;
  for (let i = 0; i < Math.min(k, chars.length); i += 1) {
    if (flagSet.has(chars[i])) cnt += 1;
  }
  let ans = cnt;
  for (let i = k; i < chars.length; i += 1) {
    if (flagSet.has(chars[i])) cnt += 1;
    if (flagSet.has(chars[i - k])) cnt -= 1;
    if (cnt > ans) ans = cnt;
  }
  return ans;
}

console.log(maxFlag("abciiidef", 3, new Set(["a", "e", "i", "o", "u"])));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
- 空间复杂度：O(1)

### 替代方案与对比

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力扫描 | O(nk) | O(1) | 每个窗口重算 |
| 前缀和 | O(n) | O(n) | 额外数组 |
| 滑动窗口 | O(n) | O(1) | 当前方法 |

### 常见错误思路

- 只更新右端元素，忘记移除左端  
- 窗口未形成就更新答案（`i < k-1`）  
- 元音判定写成多次 `if` 导致遗漏

### 为什么当前方法最优

必须扫描每个字符一次，O(n) 是下界。  
滑动窗口达到了最优，并且空间 O(1) 更适合工程场景。

---

## 常见问题与注意事项

1. **k=1 怎么办？**  
   直接统计单字符元音即可，窗口逻辑同样适用。

2. **字符串很长会溢出吗？**  
   计数最大为 k，不会溢出。

3. **用 set 判断元音会慢吗？**  
   常数成本，足够快；也可用位运算优化。

---

## 最佳实践与建议

- 先统计首个窗口，再进入滑动更新  
- 条件判断抽成函数，便于复用  
- 固定窗口问题优先用滑动窗口而非双重循环  
- 当条件复杂时，仍可用“增量更新”思想

---

## S — Summary（总结）

- 固定窗口最大计数是滑动窗口的典型应用  
- 每步只做 O(1) 更新即可完成统计  
- 相比 O(nk) 暴力法，性能提升显著  
- 工程场景可替换条件实现峰值统计  
- `i >= k-1` 是窗口形成的关键边界

### 推荐延伸阅读

- LeetCode 1456 — Maximum Number of Vowels in a Substring of Given Length  
- Sliding Window Pattern  
- Prefix Sum 与窗口法对比

---

## 小结 / 结论

最大元音子串问题本质是固定窗口计数。  
把这套模板记下来，你就能快速解决一类滚动统计问题。

---

## 参考与延伸阅读

- https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/
- https://en.wikipedia.org/wiki/Sliding_window_protocol
- https://docs.python.org/3/library/stdtypes.html#str

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：滑动窗口、字符串、固定窗口  
- **SEO 关键词**：Maximum Number of Vowels, 最大元音子串, 滑动窗口  
- **元描述**：固定窗口滑动统计最大元音数的 O(n) 解法与工程应用。  

---

## 行动号召（CTA）

如果你在做日志或指标统计，建议优先用固定滑动窗口模板。  
欢迎评论区分享你正在处理的窗口类需求。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
def max_vowels(s: str, k: int) -> int:
    vowels = set("aeiou")
    cnt = sum(1 for c in s[:k] if c in vowels)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in vowels:
            cnt += 1
        if s[i - k] in vowels:
            cnt -= 1
        if cnt > ans:
            ans = cnt
    return ans


if __name__ == "__main__":
    print(max_vowels("abciiidef", 3))
```

```c
#include <stdio.h>
#include <string.h>

static int is_vowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

int max_vowels(const char *s, int k) {
    int cnt = 0;
    int ans = 0;
    int n = (int)strlen(s);
    for (int i = 0; i < n; ++i) {
        if (is_vowel(s[i])) cnt++;
        if (i >= k && is_vowel(s[i - k])) cnt--;
        if (i >= k - 1 && cnt > ans) ans = cnt;
    }
    return ans;
}

int main(void) {
    printf("%d\n", max_vowels("abciiidef", 3));
    return 0;
}
```

```cpp
#include <iostream>
#include <string>

static bool isVowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

int maxVowels(const std::string &s, int k) {
    int cnt = 0, ans = 0;
    for (int i = 0; i < (int)s.size(); ++i) {
        if (isVowel(s[i])) cnt++;
        if (i >= k && isVowel(s[i - k])) cnt--;
        if (i >= k - 1 && cnt > ans) ans = cnt;
    }
    return ans;
}

int main() {
    std::cout << maxVowels("abciiidef", 3) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func isVowel(c byte) bool {
	return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
}

func maxVowels(s string, k int) int {
	cnt, ans := 0, 0
	for i := 0; i < len(s); i++ {
		if isVowel(s[i]) {
			cnt++
		}
		if i >= k && isVowel(s[i-k]) {
			cnt--
		}
		if i >= k-1 && cnt > ans {
			ans = cnt
		}
	}
	return ans
}

func main() {
	fmt.Println(maxVowels("abciiidef", 3))
}
```

```rust
fn is_vowel(c: u8) -> bool {
    c == b'a' || c == b'e' || c == b'i' || c == b'o' || c == b'u'
}

fn max_vowels(s: &str, k: usize) -> i32 {
    let bytes = s.as_bytes();
    let mut cnt: i32 = 0;
    let mut ans: i32 = 0;
    for i in 0..bytes.len() {
        if is_vowel(bytes[i]) { cnt += 1; }
        if i >= k && is_vowel(bytes[i - k]) { cnt -= 1; }
        if i + 1 >= k && cnt > ans { ans = cnt; }
    }
    ans
}

fn main() {
    println!("{}", max_vowels("abciiidef", 3));
}
```

```javascript
function maxVowels(s, k) {
  const isVowel = (c) => "aeiou".includes(c);
  let cnt = 0;
  let ans = 0;
  for (let i = 0; i < s.length; i += 1) {
    if (isVowel(s[i])) cnt += 1;
    if (i >= k && isVowel(s[i - k])) cnt -= 1;
    if (i >= k - 1 && cnt > ans) ans = cnt;
  }
  return ans;
}

console.log(maxVowels("abciiidef", 3));
```
