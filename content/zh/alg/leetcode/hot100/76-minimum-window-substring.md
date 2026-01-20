---
title: "LeetCode 76：最小覆盖子串（Minimum Window Substring）滑动窗口 ACERS 解析"
date: 2026-01-20T21:59:19+08:00
draft: false
categories: ["LeetCode"]
tags: ["滑动窗口", "哈希表", "字符串", "最小覆盖子串", "LeetCode 76", "ACERS"]
description: "用可变滑动窗口与计数哈希表在 O(n) 时间找到最小覆盖子串，含工程场景与多语言实现。"
keywords: ["Minimum Window Substring", "最小覆盖子串", "滑动窗口", "哈希表", "O(n)", "LeetCode 76"]
---

> **副标题 / 摘要**  
> 最小覆盖子串是“可变滑动窗口 + 计数哈希表”的经典题。本文按 ACERS 模板解释如何判断窗口有效、如何收缩得到最短答案，并给出工程场景与多语言实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`滑动窗口`、`哈希表`、`字符串`  
- **SEO 关键词**：Minimum Window Substring, 最小覆盖子串, 滑动窗口, 哈希表  
- **元描述**：最小覆盖子串的 O(n) 滑动窗口解法与工程应用，含多语言实现。

---

## 目标读者

- 正在刷 LeetCode 的中级开发者  
- 需要掌握“可变窗口 + 覆盖约束”的算法模板  
- 做文本分析、日志聚合或流式过滤的工程师

## 背景 / 动机

“在一段序列中找到最短区间覆盖目标集合”在工程中非常常见：  
日志告警需要覆盖多种错误码，搜索摘要需要覆盖关键字，  
运营分析需要覆盖多个行为标签。  
本题提供了一个可复用的窗口收缩模板。

## 核心概念

- **可变滑动窗口**：右指针扩张直到满足条件，左指针收缩缩短答案  
- **计数哈希表**：支持重复字符，必须按次数覆盖  
- **满足条件的计数**：判断当前窗口是否“覆盖了全部需要”

---

## A — Algorithm（题目与算法）

### 题目重述

给定字符串 `s` 和 `t`，返回 `s` 中最短的子串，使其包含 `t` 中的每一个字符（包括重复字符）。  
若不存在这样的子串，返回空字符串 `""`。  
测试用例保证答案唯一。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| s | string | 源字符串 |
| t | string | 目标字符串（需要覆盖的字符与次数） |
| 返回 | string | 最短覆盖子串或空串 |

### 示例 1

```text
s = "ADOBECODEBANC", t = "ABC"
输出 = "BANC"
```

### 示例 2

```text
s = "a", t = "a"
输出 = "a"
```

### 示例 3

```text
s = "a", t = "aa"
输出 = ""
```

---

## C — Concepts（核心思想）

### 方法类型

**可变滑动窗口 + 频次覆盖判断**。

### 关键模型

- `need[c]`：t 中每个字符需要的次数  
- `window[c]`：当前窗口中字符次数  
- `required`：需要满足的字符种类数  
- `formed`：当前窗口已经满足的字符种类数  

当 `formed == required` 时，窗口有效，可以尝试收缩。

### 概念模型

```text
扩张右指针 -> 满足覆盖 -> 收缩左指针 -> 更新最短答案
```

---

## 实践指南 / 步骤

1. 统计 `t` 的字符频次 `need`  
2. 初始化 `l = 0`，`formed = 0`，`required = len(need)`  
3. 右指针 `r` 逐步扩张，更新 `window`  
4. 当某字符频次满足 `need`，令 `formed += 1`  
5. 若 `formed == required`，开始收缩 `l` 并更新最短答案  
6. `formed` 不满足时停止收缩，继续扩张右指针

---

## 可运行示例（Python）

```python
from collections import Counter, defaultdict


def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""
    need = Counter(t)
    window = defaultdict(int)
    required = len(need)
    formed = 0
    l = 0
    best_len = 10 ** 18
    best_l = 0

    for r, ch in enumerate(s):
        window[ch] += 1
        if ch in need and window[ch] == need[ch]:
            formed += 1

        while formed == required:
            if r - l + 1 < best_len:
                best_len = r - l + 1
                best_l = l
            left_ch = s[l]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1
            l += 1

    return "" if best_len == 10 ** 18 else s[best_l:best_l + best_len]


if __name__ == "__main__":
    print(min_window("ADOBECODEBANC", "ABC"))
```

运行方式示例：

```bash
python3 demo.py
```

---

## 解释与原理（为什么这么做）

窗口需要满足“字符种类 + 次数”两层约束。  
`formed` 只在某字符数量 **达到** 需求时加 1，  
因此 `formed == required` 等价于“窗口已覆盖全部需求”。  

一旦覆盖成立，继续右移只会让窗口更长，所以必须尝试收缩左端，  
直到覆盖被破坏，再继续扩张右端。  
这保证了每个字符只进出窗口一次，整体 O(n)。

---

## E — Engineering（工程应用）

### 场景 1：搜索摘要最短覆盖（Python，数据分析）

**背景**：在文档中找到最短片段覆盖所有关键词。  
**为什么适用**：关键词可重复，窗口需要计数覆盖。

```python
from collections import Counter, defaultdict


def min_span(text, keywords):
    need = Counter(keywords)
    window = defaultdict(int)
    required = len(need)
    formed = 0
    l = 0
    best = (10 ** 18, 0)
    for r, ch in enumerate(text):
        if ch in need:
            window[ch] += 1
            if window[ch] == need[ch]:
                formed += 1
        while formed == required:
            if r - l + 1 < best[0]:
                best = (r - l + 1, l)
            left = text[l]
            if left in need:
                window[left] -= 1
                if window[left] < need[left]:
                    formed -= 1
            l += 1
    return "" if best[0] == 10 ** 18 else text[best[1]:best[1] + best[0]]
```

### 场景 2：日志最短覆盖区间（Go，后台服务）

**背景**：在时间序列日志中找到最短区间覆盖所有错误类型。  
**为什么适用**：错误类型可能重复出现，必须按次数覆盖。

```go
package main

import "fmt"

func minWindowTypes(logs []string, need map[string]int) (int, int) {
	window := map[string]int{}
	required := len(need)
	formed := 0
	l := 0
	bestLen, bestL := 1<<30, 0

	for r, x := range logs {
		if _, ok := need[x]; ok {
			window[x]++
			if window[x] == need[x] {
				formed++
			}
		}
		for formed == required {
			if r-l+1 < bestLen {
				bestLen = r - l + 1
				bestL = l
			}
			left := logs[l]
			if _, ok := need[left]; ok {
				window[left]--
				if window[left] < need[left] {
					formed--
				}
			}
			l++
		}
	}
	if bestLen == 1<<30 {
		return -1, -1
	}
	return bestL, bestL + bestLen - 1
}

func main() {
	logs := []string{"A", "B", "C", "A", "B", "C"}
	need := map[string]int{"A": 1, "B": 1, "C": 1}
	fmt.Println(minWindowTypes(logs, need))
}
```

### 场景 3：前端最短覆盖片段高亮（JavaScript，前端）

**背景**：在文本中找到最短片段覆盖所有高亮字符。  
**为什么适用**：前端可直接在浏览器内完成计算。

```javascript
function minWindow(s, t) {
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c) || 0) + 1);
  const window = new Map();
  const required = need.size;
  let formed = 0;
  let l = 0;
  let bestLen = Infinity;
  let bestL = 0;

  for (let r = 0; r < s.length; r += 1) {
    const ch = s[r];
    if (need.has(ch)) {
      window.set(ch, (window.get(ch) || 0) + 1);
      if (window.get(ch) === need.get(ch)) formed += 1;
    }
    while (formed === required) {
      if (r - l + 1 < bestLen) {
        bestLen = r - l + 1;
        bestL = l;
      }
      const left = s[l];
      if (need.has(left)) {
        window.set(left, window.get(left) - 1);
        if (window.get(left) < need.get(left)) formed -= 1;
      }
      l += 1;
    }
  }
  return bestLen === Infinity ? "" : s.slice(bestL, bestL + bestLen);
}

console.log(minWindow("ADOBECODEBANC", "ABC"));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
- 空间复杂度：O(Σ)（字符集大小）

### 替代方案与对比

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力枚举子串 | O(n^2) | O(1) | 易超时 |
| 双层哈希检查 | O(n^2) | O(Σ) | 仍然慢 |
| 滑动窗口 | O(n) | O(Σ) | 当前方法 |

### 常见错误思路

- 忘记考虑 `t` 中的重复字符  
- 窗口满足后不收缩，无法得到最短  
- `formed` 更新条件写错导致漏解

### 为什么当前方法最优

每个指针最多移动 n 次，整体线性；  
同时可以保证找到最短合法窗口。

---

## 常见问题与注意事项

1. **为什么要 `formed == required`？**  
   这是对“所有字符的数量需求都被满足”的精确刻画。

2. **`t` 中有重复字符怎么办？**  
   必须按次数覆盖，不能只看字符集合。

3. **没有答案怎么办？**  
   返回空字符串 `""`。

---

## 最佳实践与建议

- 用哈希表记录需求与窗口计数  
- `formed` 只在频次达到需求时更新  
- 收缩窗口时先更新答案再移动左指针  
- 对 ASCII 可用数组加速

---

## S — Summary（总结）

- 最小覆盖子串是可变滑动窗口的典型题  
- 覆盖条件必须按“字符 + 频次”判断  
- `formed == required` 是关键判定  
- O(n) 解法可直接迁移到工程场景  
- 收缩窗口是获得“最短答案”的核心步骤

### 推荐延伸阅读

- LeetCode 76 — Minimum Window Substring  
- Two Pointers + Sliding Window 模式  
- Counter / HashMap 计数技巧

---

## 小结 / 结论

这道题把“覆盖约束 + 窗口收缩”结合到一起，  
是滑动窗口中最值得背下来的模板之一。

---

## 参考与延伸阅读

- https://leetcode.com/problems/minimum-window-substring/
- https://en.wikipedia.org/wiki/Sliding_window_protocol
- https://docs.python.org/3/library/collections.html#collections.Counter

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：滑动窗口、哈希表、字符串  
- **SEO 关键词**：Minimum Window Substring, 最小覆盖子串, 滑动窗口  
- **元描述**：最小覆盖子串的滑动窗口解法与工程迁移。

---

## 行动号召（CTA）

如果你已经掌握固定窗口，下一步就是掌握“可变窗口 + 覆盖约束”。  
欢迎评论区分享你的窗口类题目清单。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import Counter, defaultdict


def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""
    need = Counter(t)
    window = defaultdict(int)
    required = len(need)
    formed = 0
    l = 0
    best_len = 10 ** 18
    best_l = 0

    for r, ch in enumerate(s):
        window[ch] += 1
        if ch in need and window[ch] == need[ch]:
            formed += 1

        while formed == required:
            if r - l + 1 < best_len:
                best_len = r - l + 1
                best_l = l
            left_ch = s[l]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1
            l += 1

    return "" if best_len == 10 ** 18 else s[best_l:best_l + best_len]


if __name__ == "__main__":
    print(min_window("ADOBECODEBANC", "ABC"))
```

```c
#include <stdio.h>
#include <string.h>

int min_window(const char *s, const char *t, char *out) {
    int need[128] = {0};
    int window[128] = {0};
    int required = 0;
    for (int i = 0; t[i]; ++i) {
        if (need[(int)t[i]] == 0) required++;
        need[(int)t[i]]++;
    }

    int formed = 0, l = 0;
    int best_len = 1 << 30, best_l = 0;
    int n = (int)strlen(s);
    for (int r = 0; r < n; ++r) {
        unsigned char c = (unsigned char)s[r];
        window[c]++;
        if (need[c] > 0 && window[c] == need[c]) formed++;
        while (formed == required) {
            if (r - l + 1 < best_len) {
                best_len = r - l + 1;
                best_l = l;
            }
            unsigned char lc = (unsigned char)s[l];
            window[lc]--;
            if (need[lc] > 0 && window[lc] < need[lc]) formed--;
            l++;
        }
    }
    if (best_len == (1 << 30)) {
        out[0] = '\0';
        return 0;
    }
    strncpy(out, s + best_l, (size_t)best_len);
    out[best_len] = '\0';
    return 1;
}

int main(void) {
    char out[1000];
    if (min_window("ADOBECODEBANC", "ABC", out)) {
        printf("%s\n", out);
    } else {
        printf("\n");
    }
    return 0;
}
```

```cpp
#include <iostream>
#include <string>
#include <unordered_map>

std::string minWindow(const std::string &s, const std::string &t) {
    if (s.empty() || t.empty()) return "";
    std::unordered_map<char, int> need, window;
    for (char c : t) need[c]++;
    int required = (int)need.size();
    int formed = 0;
    int l = 0;
    int best_len = 1e9, best_l = 0;

    for (int r = 0; r < (int)s.size(); ++r) {
        char c = s[r];
        window[c]++;
        if (need.count(c) && window[c] == need[c]) formed++;
        while (formed == required) {
            if (r - l + 1 < best_len) {
                best_len = r - l + 1;
                best_l = l;
            }
            char lc = s[l];
            window[lc]--;
            if (need.count(lc) && window[lc] < need[lc]) formed--;
            l++;
        }
    }
    return best_len == (int)1e9 ? "" : s.substr(best_l, best_len);
}

int main() {
    std::cout << minWindow("ADOBECODEBANC", "ABC") << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func minWindow(s, t string) string {
	if len(s) == 0 || len(t) == 0 {
		return ""
	}
	need := map[byte]int{}
	for i := 0; i < len(t); i++ {
		need[t[i]]++
	}
	window := map[byte]int{}
	required := len(need)
	formed := 0
	l := 0
	bestLen := 1 << 30
	bestL := 0

	for r := 0; r < len(s); r++ {
		c := s[r]
		window[c]++
		if need[c] > 0 && window[c] == need[c] {
			formed++
		}
		for formed == required {
			if r-l+1 < bestLen {
				bestLen = r - l + 1
				bestL = l
			}
			lc := s[l]
			window[lc]--
			if need[lc] > 0 && window[lc] < need[lc] {
				formed--
			}
			l++
		}
	}
	if bestLen == 1<<30 {
		return ""
	}
	return s[bestL : bestL+bestLen]
}

func main() {
	fmt.Println(minWindow("ADOBECODEBANC", "ABC"))
}
```

```rust
use std::collections::HashMap;

fn min_window(s: &str, t: &str) -> String {
    if s.is_empty() || t.is_empty() {
        return String::new();
    }
    let mut need: HashMap<u8, i32> = HashMap::new();
    for &b in t.as_bytes() {
        *need.entry(b).or_insert(0) += 1;
    }
    let mut window: HashMap<u8, i32> = HashMap::new();
    let required = need.len() as i32;
    let mut formed = 0;
    let bytes = s.as_bytes();
    let mut l: usize = 0;
    let mut best_len = usize::MAX;
    let mut best_l = 0;

    for r in 0..bytes.len() {
        let c = bytes[r];
        *window.entry(c).or_insert(0) += 1;
        if let Some(&need_cnt) = need.get(&c) {
            if window.get(&c) == Some(&need_cnt) {
                formed += 1;
            }
        }
        while formed == required {
            if r - l + 1 < best_len {
                best_len = r - l + 1;
                best_l = l;
            }
            let lc = bytes[l];
            if let Some(v) = window.get_mut(&lc) {
                *v -= 1;
            }
            if let Some(&need_cnt) = need.get(&lc) {
                if window.get(&lc).unwrap_or(&0) < &need_cnt {
                    formed -= 1;
                }
            }
            l += 1;
        }
    }
    if best_len == usize::MAX {
        String::new()
    } else {
        s[best_l..best_l + best_len].to_string()
    }
}

fn main() {
    println!("{}", min_window("ADOBECODEBANC", "ABC"));
}
```

```javascript
function minWindow(s, t) {
  if (!s || !t) return "";
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c) || 0) + 1);
  const window = new Map();
  const required = need.size;
  let formed = 0;
  let l = 0;
  let bestLen = Infinity;
  let bestL = 0;

  for (let r = 0; r < s.length; r += 1) {
    const c = s[r];
    window.set(c, (window.get(c) || 0) + 1);
    if (need.has(c) && window.get(c) === need.get(c)) formed += 1;
    while (formed === required) {
      if (r - l + 1 < bestLen) {
        bestLen = r - l + 1;
        bestL = l;
      }
      const lc = s[l];
      window.set(lc, window.get(lc) - 1);
      if (need.has(lc) && window.get(lc) < need.get(lc)) formed -= 1;
      l += 1;
    }
  }
  return bestLen === Infinity ? "" : s.slice(bestL, bestL + bestLen);
}

console.log(minWindow("ADOBECODEBANC", "ABC"));
```
