---
title: "最少涂色次数拿到 k 个连续黑块：滑动窗口的极简解法"
date: 2025-12-04T10:30:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["滑动窗口", "字符串", "双指针", "面试高频题"]
description: "给定只包含 W/B 的字符串 blocks，每次可将一个 W 涂成 B，如何用一次线性扫描求出得到至少一次连续 k 个黑色块的最少操作次数，并给出多语言实现与工程应用示例。"
keywords: ["LeetCode 2379", "Minimum Recolors", "滑动窗口", "字符串算法", "双指针", "算法题解析"]
---

> **副标题 / 摘要**  
> 一道看似暴力 O(n·k) 的刷题小题，实际只需要一个固定长度滑动窗口就能在 O(n) 内秒杀。本文从题意还原、窗口建模，到多语言实现与工程场景，把这类「固定长度窗口 + 计数」问题一网打尽。

- **预计阅读时长**：8~10 分钟  
- **适用场景标签**：`滑动窗口`、`字符串处理`、`面试刷题`  
- **SEO 关键词**：LeetCode 2379, minimum recolors, sliding window, k consecutive black blocks  

---

## 目标读者与背景

**目标读者**

- 正在系统刷 LeetCode / 力扣、想提升滑动窗口题目通过率的开发者
- 面试中经常被「固定窗口 + 计数」卡住的同学
- 想把算法题思路迁移到业务代码中的后端 / 前端工程师

**为什么这个问题值得认真写一篇？**

- 它是**滑动窗口最基础的形态**：窗口长度固定，维护一个简单计数。
- 很多更难的题（如「最长连续 1」、「至少 k 个元素」等）都可以退化到这个模板。
- 工程里也经常遇到类似需求：连续 k 个时间片、连续 k 条日志、连续 k 个卡片槽位是否满足某种条件。

---

## A — Algorithm（题目与算法）

### 题目描述（用自己的话再说一遍）

给你一个只包含 `'W'`（白块）和 `'B'`（黑块）的字符串 `blocks`，还有一个整数 `k`。

你可以进行若干次操作，每次操作：

- 选择一个位置，如果那里是 `'W'`，就可以把它涂成 `'B'`。

目标是：

> 通过涂色，让字符串中出现**至少一次**长度为 `k` 的连续黑色块（`k` 个连续 `'B'`），并且总操作次数最少。问最少要涂几次？

**输入**

- `blocks: str`，只包含字符 `'W'` 和 `'B'`
- `k: int`，目标连续黑块长度，`1 ≤ k ≤ len(blocks)`

**输出**

- 一个整数：达到目标至少需要的最少操作（涂色）次数

### 基础示例 1

```text
blocks = "WBBWWBBWBW"
k      = 7
```

我们要找长度为 7 的连续子串：

- 子串 `[0..6]`: `"WBBWWBB"`，里面有 3 个 `'W'`
- 子串 `[1..7]`: `"BBWWBBW"`，里面有 3 个 `'W'`
- 子串 `[2..8]`: `"BWWBBWB"`，里面有 3 个 `'W'`
- 子串 `[3..9]`: `"WWBBWBW"`，里面有 4 个 `'W'`

其中白块最少的是 `3`，所以至少需要涂 `3` 次，把那段里的 3 个 `'W'` 全涂黑，就能得到一个长度为 7 的连续黑块。

**输出**：`3`

### 基础示例 2

```text
blocks = "BBBBB"
k      = 3
```

任意长度为 3 的子串都已经是 `"BBB"`：

- 白块个数都是 0，不需要涂色。

**输出**：`0`

---

## C — Concepts（核心思想）

### 1. 把问题抽象成“固定窗口 + 计数”

我们最终要得到的是一段**长度恰好为 `k` 的连续黑块**。不妨想象一下：

- 先随便选出一段长度为 `k` 的子串（窗口）；
- 把这段里的所有 `'W'` 全部涂成 `'B'`，这段就变成连续黑块了；
- 所以，对这段来说，**至少需要涂色的次数 = 这段里 `'W'` 的数量**。

那么只要：

> 枚举所有长度为 `k` 的子串，找到其中**白块数量最少**的那个，答案就是这个最小白块数。

这就是一个典型的：

- **固定窗口大小（k）**
- **窗口内计数（白块个数）**

问题，非常适合滑动窗口。

### 2. 滑动窗口如何省掉 O(n·k) 的重复计算？

暴力做法会：

- 对每个起点 `i` 计算子串 `blocks[i..i+k-1]` 里有多少个 `'W'` → O(k)
- 总共有大约 `n-k+1` 个起点 → 总复杂度 O(n·k)

但**相邻的两个窗口高度重叠**：

```text
窗口 1: [i,       ..., i+k-1]
窗口 2: [i+1,     ..., i+k]
```

它们的区别只有：

- **窗口 1 的第一个字符**从窗口中「滑出」
- **窗口 2 的最后一个字符**新「滑入」

所以我们只用维护：

- `window_white`：当前窗口中的 `'W'` 数量

当窗口滑动时：

1. 新进入的字符如果是 `'W'` → `window_white++`
2. 离开的字符如果是 `'W'` → `window_white--`

这样每次滑动成本是 O(1)，总时间从 O(n·k) 降到了 **O(n)**。

### 3. 算法属于哪一类？

- 方法论：**滑动窗口（Sliding Window）**
- 窗口类型：**固定窗口长度（fixed-size window）**
- 实现手段：**双指针 / 单指针 + 下标判断**
- 关键状态：窗口内 `'W'` 个数

### 4. 核心状态与公式

- 字符串长度记为 `n`。
- 遍历下标 `i` 从 0 到 `n-1`：

```text
如果 blocks[i] == 'W'：window_white += 1
如果 i >= k 且 blocks[i-k] == 'W'：window_white -= 1
如果 i >= k-1：min_white = min(min_white, window_white)
```

最终：

```text
答案 = min_white
```

如果字符串中本来已经有某个窗口白块为 0，那么答案自然就是 0。

---

## 实践指南：从思路到代码的 5 个步骤

1. **理清本质**：  
   把题目转化为「在所有长度为 `k` 的子串中，**白块数量的最小值**」。

2. **设计窗口状态**：  
   仅维护一个整数 `window_white`，表示当前窗口中 `'W'` 的数量，再加一个 `min_white` 记录全局最小。

3. **写出滑动逻辑**：  
   - 顺序遍历 `blocks` 的每个位置 `i`  
   - 先把 `blocks[i]` 加入窗口（如果是 `'W'` 就 ++）  
   - 如果窗口长度超过 `k`，移除 `blocks[i-k]`（如果是 `'W'` 就 --）  
   - 当 `i >= k-1` 时，更新 `min_white`

4. **检查边界条件**：  
   - `k == 1` 时窗口只有一个字符，逻辑依然成立  
   - 字符串已经全是 `'B'` 时，`min_white` 会变为 0

5. **运行并断言结果**：  
   - 用本文的示例跑一遍  
   - 再加一些极端情况（如全 `W`、全 `B`、`k == len(blocks)`）

---

## E — Engineering（工程应用）

这一类「固定窗口 + 计数」问题在工程里很常见，下面给出 3 个真实感很强的场景。

### 场景 1：前端 UI —— 连续可用槽位检测（JavaScript）

**背景**  
有一个水平滚动的卡片列表，`'B'` 表示槽位已有卡片，`'W'` 表示空槽。  
产品希望知道：**至少要补几张卡片，才能确保存在一段连续 `k` 个槽位都不为空**？

这就和题目一模一样。

**为什么适用**  
你完全可以把 UI 状态压缩成一个字符串 / 数组，用滑动窗口在前端直接计算，  
再把结果反馈给产品或用于配置「推荐卡片」数量上限。

**示例代码（可直接在浏览器控制台 / Node.js 跑）**

```js
function minRecolors(blocks, k) {
  let windowWhite = 0;
  let minWhite = Infinity;

  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i] === 'W') windowWhite++;
    if (i >= k && blocks[i - k] === 'W') windowWhite--;
    if (i >= k - 1) minWhite = Math.min(minWhite, windowWhite);
  }

  return minWhite === Infinity ? 0 : minWhite;
}

console.log(minRecolors("WBBWWBBWBW", 7)); // 3
```

---

### 场景 2：日志 / 安全审计 —— 连续高风险事件注入估算（Python）

**背景**  
有一串审计日志，`'B'` 表示高风险事件，`'W'` 表示普通事件。  
你在做内部攻防演练，希望知道**至少还要插入多少高风险事件**，才能在日志中制造出一段连续 `k` 个高风险事件的窗口，方便验证告警系统。

**为什么适用**  
安全日志就是一个事件流，把高风险标记出来后，就可以视作 W/B 字符串。

```python
from typing import *


def minimum_recolors(blocks: str, k: int) -> int:
    window_white = 0
    min_white = float("inf")

    for i, ch in enumerate(blocks):
        if ch == "W":
            window_white += 1
        if i >= k and blocks[i - k] == "W":
            window_white -= 1
        if i >= k - 1:
            min_white = min(min_white, window_white)

    return 0 if min_white == float("inf") else min_white


if __name__ == "__main__":
    print(minimum_recolors("WBBWWBBWBW", 7))  # 3
```

---

### 场景 3：后端风控 / 交易系统 —— 连续风险窗口（Go）

**背景**  
在交易风控中，你可能给每笔交易打一个「是否命中风险规则」标记：  
`'B'` 表示命中，`'W'` 表示未命中。  
为了评估风控规则的「连击性」，你希望知道：  
**至少还要构造多少次命中，才能在日志中出现一段连续 `k` 次命中？**

**为什么适用**  
这些风险命中标记在时间上是有序的，本质还是固定长度窗口上的计数问题。

```go
package main

import "fmt"

func minimumRecolors(blocks string, k int) int {
	windowWhite := 0
	minWhite := 1<<31 - 1

	for i := 0; i < len(blocks); i++ {
		if blocks[i] == 'W' {
			windowWhite++
		}
		if i >= k && blocks[i-k] == 'W' {
			windowWhite--
		}
		if i >= k-1 && windowWhite < minWhite {
			minWhite = windowWhite
		}
	}
	if minWhite == 1<<31-1 {
		return 0
	}
	return minWhite
}

func main() {
	fmt.Println(minimumRecolors("WBBWWBBWBW", 7)) // 3
}
```

---

## R — Reflection（反思与深入）

### 1. 复杂度分析

- **时间复杂度**：  
  整个字符串只扫描一遍，每个字符最多进窗口一次、出窗口一次 → O(n)。

- **空间复杂度**：  
  只用到常数个变量（`window_white`、`min_white` 等） → O(1)。

对于 n 在 1e5 甚至更大时，这种线性算法在任何主流语言里都能轻松通过。

---

### 2. 与暴力法 / 其他思路对比

**暴力法（Brute Force）**

- 对每个起点 i，统计子串 `blocks[i..i+k-1]` 的白块数  
- 每个窗口 O(k)，窗口数约为 `n-k+1` 个 → 总复杂度 O(n·k)
- 当 n 和 k 都在 1e5 级别时，完全不可接受

**滑动窗口法（当前方案）**

- 在相邻窗口之间做「增量更新」  
- 每次滑动只处理 2 个字符（一个进一个出） → O(1)
- 总复杂度 O(n)，在工程中也容易优化和调试

**为什么滑动窗口更工程可行？**

- 模式通用：几乎所有「固定窗口 + 计数」问题都能套同一框架
- 可读性高：核心逻辑只围绕两个操作——进窗口、出窗口
- 性能稳定：不依赖复杂数据结构，对 GC / 内存压力小

---

### 3. 常见错误与注意事项

1. **窗口边界 off-by-one**
   - 条件 `i >= k` 与 `i >= k-1` 容易写错
   - 建议在纸上写几个具体的 i 值对照一下

2. **忘记处理初始窗口**
   - 一种常见写法是先处理前 `k` 个字符，再从第 `k` 个开始滑动  
   - 本文用的是「统一写法」，通过下标判断自然覆盖了初始窗口

3. **误把窗口长度写成可变**
   - 本题窗口长度是固定的，不能随便改变左指针，只能保证 `right - left + 1 == k`
   - 可变窗口对应的是另一类滑动窗口题目

4. **特例**
   - 全部是 `'B'` → 应该返回 0  
   - `k == len(blocks)` → 只会有一个窗口，也能被当前写法覆盖

---

## S — Summary（总结）

- 本题的本质是：在所有长度为 `k` 的子串中，找到白块数最少的那个窗口。
- 利用滑动窗口，只维护一个「当前窗口白块数」就能把复杂度从 O(n·k) 降到 O(n)。
- 固定窗口长度 + 窗口内计数，是滑动窗口中最基础、最常见的模式。
- 在工程实践中，连续时间窗口、连续日志条目、连续 UI 槽位等问题，都可以用同样模式建模。
- 认真处理好下标与边界，可以减少 90% 的滑动窗口调试时间。

---

## 参考与延伸阅读

- LeetCode 2379. Minimum Recolors to Get K Consecutive Black Blocks（题目原始出处）
- LeetCode 1004. Max Consecutive Ones III（可变窗口版本，对比学习）
- 滑动窗口专题刷题列表（可以在力扣 / Codeforces 按标签筛选）
- 《算法导论》第 8 章附近关于线性扫描与双指针的内容

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

下面是同一个思路在多种语言中的参考代码，可以直接复制到你的代码仓库或笔记中使用。

### Python 实现

```python
from typing import *


def minimum_recolors(blocks: str, k: int) -> int:
    window_white = 0
    min_white = float("inf")

    for i, ch in enumerate(blocks):
        if ch == "W":
            window_white += 1
        if i >= k and blocks[i - k] == "W":
            window_white -= 1
        if i >= k - 1:
            min_white = min(min_white, window_white)

    return 0 if min_white == float("inf") else min_white


if __name__ == "__main__":
    print(minimum_recolors("WBBWWBBWBW", 7))  # 3
```

---

### C 实现

```c
#include <stdio.h>
#include <limits.h>

int minimumRecolors(const char *blocks, int k) {
    int windowWhite = 0;
    int minWhite = INT_MAX;

    for (int i = 0; blocks[i] != '\0'; ++i) {
        if (blocks[i] == 'W') {
            windowWhite++;
        }
        if (i >= k && blocks[i - k] == 'W') {
            windowWhite--;
        }
        if (i >= k - 1 && windowWhite < minWhite) {
            minWhite = windowWhite;
        }
    }

    if (minWhite == INT_MAX) return 0;
    return minWhite;
}

int main(void) {
    printf("%d\n", minimumRecolors("WBBWWBBWBW", 7)); // 3
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int minimumRecolors(const string &blocks, int k) {
    int windowWhite = 0;
    int minWhite = INT_MAX;

    for (int i = 0; i < (int)blocks.size(); ++i) {
        if (blocks[i] == 'W') {
            windowWhite++;
        }
        if (i >= k && blocks[i - k] == 'W') {
            windowWhite--;
        }
        if (i >= k - 1) {
            minWhite = min(minWhite, windowWhite);
        }
    }
    if (minWhite == INT_MAX) return 0;
    return minWhite;
}

int main() {
    cout << minimumRecolors("WBBWWBBWBW", 7) << endl; // 3
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func minimumRecolors(blocks string, k int) int {
	windowWhite := 0
	minWhite := 1<<31 - 1

	for i := 0; i < len(blocks); i++ {
		if blocks[i] == 'W' {
			windowWhite++
		}
		if i >= k && blocks[i-k] == 'W' {
			windowWhite--
		}
		if i >= k-1 && windowWhite < minWhite {
			minWhite = windowWhite
		}
	}
	if minWhite == 1<<31-1 {
		return 0
	}
	return minWhite
}

func main() {
	fmt.Println(minimumRecolors("WBBWWBBWBW", 7)) // 3
}
```

---

### Rust 实现

```rust
fn minimum_recolors(blocks: &str, k: usize) -> i32 {
    let chars: Vec<char> = blocks.chars().collect();
    let mut window_white: i32 = 0;
    let mut min_white: i32 = i32::MAX;

    for i in 0..chars.len() {
        if chars[i] == 'W' {
            window_white += 1;
        }
        if i >= k && chars[i - k] == 'W' {
            window_white -= 1;
        }
        if i + 1 >= k {
            min_white = min_white.min(window_white);
        }
    }

    if min_white == i32::MAX {
        0
    } else {
        min_white
    }
}

fn main() {
    println!("{}", minimum_recolors("WBBWWBBWBW", 7)); // 3
}
```

---

### JavaScript 实现

```js
function minimumRecolors(blocks, k) {
  let windowWhite = 0;
  let minWhite = Infinity;

  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i] === 'W') windowWhite++;
    if (i >= k && blocks[i - k] === 'W') windowWhite--;
    if (i >= k - 1) minWhite = Math.min(minWhite, windowWhite);
  }

  return minWhite === Infinity ? 0 : minWhite;
}

console.log(minimumRecolors("WBBWWBBWBW", 7)); // 3
```

---

## 行动号召（CTA）

- 把这道题的代码按你最熟悉的语言写一遍，并在 IDE 里打几个断点，亲自观察窗口变量如何变化。
- 找 3 道「固定窗口 + 计数」的题目（如「最大连续 1 数量」、「至少含 k 个 1」），尝试完全复用本文的滑动窗口模板。
- 如果你在项目里也有「连续 k 个时间窗 / 槽位」之类的业务逻辑，可以尝试用这个模板做一次重构，看看是否更简洁、好测。

