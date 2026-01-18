---
title: "滑动窗口最大值：单调队列（Monotonic Queue）一遍扫描 ACERS 解析"
date: 2026-01-18T18:32:08+08:00
draft: false
categories: ["LeetCode"]
tags: ["滑动窗口", "单调队列", "数组", "队列", "LeetCode 239"]
description: "用单调队列在 O(n) 时间求滑动窗口最大值，含工程场景、复杂度对比与多语言实现。"
keywords: ["Sliding Window Maximum", "滑动窗口最大值", "单调队列", "deque", "O(n)"]
---

> **副标题 / 摘要**  
> 滑动窗口最大值是“滑动窗口 + 单调队列”的经典组合题。本文按 ACERS 模板拆解思路，给出可复用的工程做法与多语言实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`滑动窗口`、`单调队列`、`数组`  
- **SEO 关键词**：Sliding Window Maximum, 滑动窗口最大值, 单调队列, deque, O(n)  
- **元描述**：滑动窗口最大值的单调队列解法与工程应用，含复杂度分析与多语言代码。  

---

## 目标读者

- 正在刷 LeetCode / Hot100 的同学  
- 想建立“滑动窗口 + 单调队列”模板的中级开发者  
- 做实时监控、日志分析、风控的工程师

## 背景 / 动机

连续窗口的最大值在工程里非常常见：  
延迟监控、价格波动、温度报警、在线指标平滑等都需要“窗口最大值”。  
暴力做法每次窗口重算最大值是 O(nk)，当 n 很大时会不可接受。  
单调队列能把复杂度降到 O(n)，是最工程可行的方案之一。

## 核心概念

- **滑动窗口**：固定长度 k 的连续区间  
- **单调队列**：队列中元素按值单调递减，队首永远是当前最大值  
- **索引维护**：用索引判断元素是否过期（离开窗口）

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组最左侧移动到最右侧。  
你只能看到窗口内的 k 个数字，窗口每次右移一位。  
返回每个窗口中的最大值。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 整数数组 |
| k | int | 窗口大小 |
| 返回 | int[] | 每个窗口的最大值 |

### 示例 1

```text
nums = [1,3,-1,-3,5,3,6,7], k = 3
输出 = [3,3,5,5,6,7]
```

### 示例 2

```text
nums = [1], k = 1
输出 = [1]
```

---

## C — Concepts（核心思想）

### 方法类型

**滑动窗口 + 单调队列（Monotonic Queue）**。

### 关键不变式

1. 队列中索引对应的值 **单调递减**  
2. 队首索引始终在当前窗口内  
3. 队首元素就是当前窗口最大值

### 模型示意

```text
窗口右移:
1) 先弹出队首过期索引
2) 再从队尾弹出小于新值的索引
3) 把新索引加入队尾
4) 队首即最大值
```

---

## 实践指南 / 步骤

1. 使用一个双端队列 `dq` 存索引  
2. 遍历 `nums`，对每个 `i` 做：
   - 如果 `dq[0]` 已经离开窗口（`dq[0] <= i - k`），弹出  
   - 从队尾弹出所有 `nums[dq[-1]] <= nums[i]` 的索引  
   - 把 `i` 入队  
   - 当 `i >= k - 1` 时，记录 `nums[dq[0]]`

---

## 可运行示例（Python）

```python
from collections import deque
from typing import List


def max_sliding_window(nums: List[int], k: int) -> List[int]:
    dq = deque()
    ans = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            ans.append(nums[dq[0]])
    return ans


if __name__ == "__main__":
    print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
```

运行方式示例：

```bash
python3 demo.py
```

---

## 解释与原理（为什么这么做）

单调队列的核心是：  
**每个元素最多进队一次、出队一次**，因此总复杂度是 O(n)。  
通过“索引过期 + 维护单调递减”，队首永远是最大值。

如果用暴力法，每个窗口都要扫描 k 个元素，复杂度是 O(nk)。  
当 n 很大、k 也不小的时候，性能差距会非常明显。

---

## E — Engineering（工程应用）

### 场景 1：价格监控中的滚动最高价（Python，数据分析）

**背景**：统计某商品过去 k 天内的最高价。  
**为什么适用**：价格序列长，O(n) 滚动最大值更省时。

```python
from collections import deque


def rolling_max(prices, k):
    dq = deque()
    ans = []
    for i, x in enumerate(prices):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and prices[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            ans.append(prices[dq[0]])
    return ans


print(rolling_max([10, 12, 9, 14, 11, 15], 3))
```

### 场景 2：服务延迟监控（Go，后台服务）

**背景**：实时观察最近 k 个请求的最高延迟，用于报警与限流。  
**为什么适用**：在线统计，单调队列能做到 O(1) 均摊更新。

```go
package main

import "fmt"

func rollingMax(nums []int, k int) []int {
	dq := make([]int, 0)
	ans := make([]int, 0)
	for i, x := range nums {
		if len(dq) > 0 && dq[0] <= i-k {
			dq = dq[1:]
		}
		for len(dq) > 0 && nums[dq[len(dq)-1]] <= x {
			dq = dq[:len(dq)-1]
		}
		dq = append(dq, i)
		if i >= k-1 {
			ans = append(ans, nums[dq[0]])
		}
	}
	return ans
}

func main() {
	fmt.Println(rollingMax([]int{120, 98, 110, 140, 105}, 2))
}
```

### 场景 3：前端走势图高亮（JavaScript，前端）

**背景**：在折线图上标出每个窗口中的最高点。  
**为什么适用**：前端可直接完成计算，无需后端接口。

```javascript
function rollingMax(nums, k) {
  const dq = [];
  const ans = [];
  for (let i = 0; i < nums.length; i += 1) {
    if (dq.length && dq[0] <= i - k) dq.shift();
    while (dq.length && nums[dq[dq.length - 1]] <= nums[i]) dq.pop();
    dq.push(i);
    if (i >= k - 1) ans.push(nums[dq[0]]);
  }
  return ans;
}

console.log(rollingMax([2, 5, 3, 6, 1, 4], 3));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
- 空间复杂度：O(k)

### 替代方案与取舍

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力扫描 | O(nk) | O(1) | 简单但性能差 |
| 堆（优先队列） | O(n log k) | O(k) | 需要清理过期元素 |
| 单调队列 | O(n) | O(k) | 当前方法，最优 |

### 常见错误思路

- 队列里存值而不是索引，导致无法判断过期元素  
- 忘记在入队前弹出小于当前值的元素  
- 滑动窗口边界 off-by-one（`i >= k - 1`）写错

### 为什么这是最优

每个元素最多入队、出队一次，  
因此总操作数是线性的，满足最优复杂度要求。

---

## 常见问题与注意事项

1. **k=1 怎么办？**  
   结果就是原数组，每个窗口只有一个元素。

2. **为什么要存索引而不是值？**  
   因为需要判断元素是否已经滑出窗口。

3. **窗口大小大于数组长度？**  
   题目一般保证合法；工程中可加边界判断。

---

## 最佳实践与建议

- 把“单调队列模板”记成可复用代码片段  
- 用索引维护窗口边界  
- 避免使用 `shift()` 的场景可用双指针模拟队列以提速  
- 对于实时流式数据，可以把队列做成持续结构

---

## S — Summary（总结）

- 滑动窗口最大值的最优解是单调队列  
- 队首始终是当前窗口最大值  
- 每个元素最多进出队一次，复杂度 O(n)  
- 工程中常用于监控、滚动统计、实时指标

### 推荐延伸阅读

- LeetCode 239 — Sliding Window Maximum  
- Monotonic Queue / Deque 经典模板  
- Rolling Aggregation / Streaming Analytics

---

## 小结 / 结论

滑动窗口最大值的价值在于“可复用的模板化实现”。  
掌握单调队列，就等于掌握了一类高频的滚动统计问题。

---

## 参考与延伸阅读

- https://leetcode.com/problems/sliding-window-maximum/
- https://en.cppreference.com/w/cpp/container/deque
- https://docs.python.org/3/library/collections.html#collections.deque
- https://doc.rust-lang.org/std/collections/struct.VecDeque.html

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：滑动窗口、单调队列、数组  
- **SEO 关键词**：Sliding Window Maximum, 滑动窗口最大值, 单调队列  
- **元描述**：滑动窗口最大值的单调队列解法与工程实践，含多语言实现。  

---

## 行动号召（CTA）

如果你在刷题或做实时指标统计，建议把单调队列当作“必备模板”。  
欢迎评论区分享你在工程中使用滑动窗口的场景。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import deque
from typing import List


def max_sliding_window(nums: List[int], k: int) -> List[int]:
    dq = deque()
    ans = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            ans.append(nums[dq[0]])
    return ans


if __name__ == "__main__":
    print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
```

```c
#include <stdio.h>
#include <stdlib.h>

int *max_sliding_window(const int *nums, int n, int k, int *out_len) {
    if (k <= 0 || n <= 0) {
        *out_len = 0;
        return NULL;
    }
    int *ans = (int *)malloc(sizeof(int) * (n - k + 1));
    int *dq = (int *)malloc(sizeof(int) * n);
    int head = 0, tail = 0;
    int idx = 0;

    for (int i = 0; i < n; ++i) {
        if (head < tail && dq[head] <= i - k) head++;
        while (head < tail && nums[dq[tail - 1]] <= nums[i]) tail--;
        dq[tail++] = i;
        if (i >= k - 1) {
            ans[idx++] = nums[dq[head]];
        }
    }
    *out_len = idx;
    free(dq);
    return ans;
}

int main(void) {
    int nums[] = {1, 3, -1, -3, 5, 3, 6, 7};
    int out_len = 0;
    int *res = max_sliding_window(nums, 8, 3, &out_len);
    for (int i = 0; i < out_len; ++i) {
        printf("%d ", res[i]);
    }
    printf("\n");
    free(res);
    return 0;
}
```

```cpp
#include <deque>
#include <iostream>
#include <vector>

std::vector<int> maxSlidingWindow(const std::vector<int> &nums, int k) {
    std::deque<int> dq;
    std::vector<int> ans;
    for (int i = 0; i < (int)nums.size(); ++i) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) ans.push_back(nums[dq.front()]);
    }
    return ans;
}

int main() {
    std::vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
    auto res = maxSlidingWindow(nums, 3);
    for (int x : res) std::cout << x << " ";
    std::cout << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func maxSlidingWindow(nums []int, k int) []int {
	dq := make([]int, 0)
	ans := make([]int, 0)
	for i, x := range nums {
		if len(dq) > 0 && dq[0] <= i-k {
			dq = dq[1:]
		}
		for len(dq) > 0 && nums[dq[len(dq)-1]] <= x {
			dq = dq[:len(dq)-1]
		}
		dq = append(dq, i)
		if i >= k-1 {
			ans = append(ans, nums[dq[0]])
		}
	}
	return ans
}

func main() {
	fmt.Println(maxSlidingWindow([]int{1, 3, -1, -3, 5, 3, 6, 7}, 3))
}
```

```rust
use std::collections::VecDeque;

fn max_sliding_window(nums: &[i32], k: usize) -> Vec<i32> {
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut ans: Vec<i32> = Vec::new();
    for (i, &x) in nums.iter().enumerate() {
        if let Some(&front) = dq.front() {
            if front + k <= i {
                dq.pop_front();
            }
        }
        while let Some(&back) = dq.back() {
            if nums[back] <= x {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(i);
        if i + 1 >= k {
            ans.push(nums[*dq.front().unwrap()]);
        }
    }
    ans
}

fn main() {
    let nums = vec![1, 3, -1, -3, 5, 3, 6, 7];
    println!("{:?}", max_sliding_window(&nums, 3));
}
```

```javascript
function maxSlidingWindow(nums, k) {
  const dq = [];
  const ans = [];
  for (let i = 0; i < nums.length; i += 1) {
    if (dq.length && dq[0] <= i - k) dq.shift();
    while (dq.length && nums[dq[dq.length - 1]] <= nums[i]) dq.pop();
    dq.push(i);
    if (i >= k - 1) ans.push(nums[dq[0]]);
  }
  return ans;
}

console.log(maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3));
```
