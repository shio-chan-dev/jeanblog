---
title: "Sliding Window Maximum: Monotonic Queue One-Pass ACERS Guide"
date: 2026-01-19T17:46:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["sliding window", "monotonic queue", "array", "deque", "LeetCode 239"]
description: "Solve Sliding Window Maximum in O(n) with a monotonic queue, including engineering scenarios, complexity comparisons, and multi-language implementations."
keywords: ["Sliding Window Maximum", "monotonic queue", "deque", "O(n)"]
---

### **Title**

Sliding Window Maximum: Monotonic Queue One-Pass ACERS Guide

---

### **Subtitle / Summary**

Sliding Window Maximum is the classic combo of sliding window + monotonic queue.
This article follows the ACERS template with reusable engineering patterns and multi-language implementations.

- **Estimated reading time**: 12–15 minutes  
- **Tags**: `sliding window`, `monotonic queue`, `array`  
- **SEO keywords**: Sliding Window Maximum, monotonic queue, deque, O(n)  
- **Meta description**: Monotonic-queue solution for Sliding Window Maximum with engineering practice and multi-language implementations.  

---

## Target Readers

- People practicing LeetCode / Hot100
- Mid-level developers who want a reusable “sliding window + monotonic queue” template
- Engineers working on real-time monitoring, log analytics, or risk control

## Background / Motivation

Rolling-window maximum appears everywhere: latency monitoring, price spikes, temperature alerts,
real-time smoothing, and many more. The brute-force approach recomputes max per window in O(nk),
which is unacceptable for large n. The monotonic queue reduces it to O(n), making it the most
practical engineering choice.

## Core Concepts

- **Sliding window**: a fixed-length window of size k
- **Monotonic queue**: values are kept in decreasing order; the front is always the max
- **Index maintenance**: indices let us evict out-of-window elements

---

## A — Algorithm (Problem & Algorithm)

### Problem Restatement

Given an integer array `nums` and a window size `k`, a sliding window moves from left to right.
Each move shifts the window by one. Return the maximum for each window.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | input array |
| k | int | window size |
| return | int[] | max of each window |

### Example 1

```text
nums = [1,3,-1,-3,5,3,6,7], k = 3
output = [3,3,5,5,6,7]
```

### Example 2

```text
nums = [1], k = 1
output = [1]
```

---

## C — Concepts (Core Ideas)

### Method Type

**Sliding window + monotonic queue**.

### Key Invariants

1. Values at indices in the queue are **monotonically decreasing**
2. The front index always lies within the current window
3. The front element is the window maximum

### Model Sketch

```text
Window moves right:
1) pop front if it is out of window
2) pop from back while value <= new value
3) push new index to back
4) front is the max
```

---

## Practical Steps / Walkthrough

1. Use a deque `dq` to store indices
2. For each index `i`:
   - Pop front if `dq[0] <= i - k`
   - Pop from back while `nums[dq[-1]] <= nums[i]`
   - Push `i`
   - If `i >= k - 1`, record `nums[dq[0]]`

---

## Runnable Example (Python)

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

Run:

```bash
python3 demo.py
```

---

## Explanation & Rationale

The monotonic queue guarantees:
- Each element enters and leaves the deque at most once.
- Total operations are O(n).

Brute force scans each window in O(k), yielding O(nk). For large n and k, the gap is huge.

---

## E — Engineering (Applications)

### Scenario 1: Rolling Highest Price (Python, data analytics)

**Background**: Compute the highest price within the last k days.  
**Why it fits**: Long price series need O(n) rolling max.  

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

### Scenario 2: Service Latency Monitoring (Go, backend)

**Background**: Track the max latency in the latest k requests for alerting.  
**Why it fits**: O(1) amortized updates in streaming mode.  

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

### Scenario 3: Frontend Chart Highlighting (JavaScript, frontend)

**Background**: Highlight the max point in each window on a chart.  
**Why it fits**: Pure frontend computation, no backend needed.  

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

## R — Reflection (Deeper Insight)

### Complexity

- Time: O(n)
- Space: O(k)

### Alternatives & Trade-offs

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| Brute force | O(nk) | O(1) | Simple but slow |
| Heap / PQ | O(n log k) | O(k) | Requires cleanup of expired elements |
| Monotonic queue | O(n) | O(k) | Optimal approach |

### Common Pitfalls

- Storing values instead of indices (can’t evict out-of-window elements)
- Forgetting to pop smaller elements before pushing the new value
- Off-by-one errors on window boundary (`i >= k - 1`)

### Why This Is Optimal

Each element is pushed and popped at most once, so total operations are linear.

---

## Common Questions & Notes

1. **What if k = 1?**  
   The result is the original array.

2. **Why store indices instead of values?**  
   You need indices to know when elements expire.

3. **What if k > len(nums)?**  
   LeetCode guarantees valid input; in production add boundary checks.

---

## Best Practices & Tips

- Keep a reusable monotonic-queue template
- Use indices to manage window boundaries
- For large JS arrays, replace `shift()` with a head pointer for performance
- For streaming data, keep the queue as a long-lived structure

---

## S — Summary

- The optimal solution uses a monotonic queue
- The front always holds the window maximum
- Each element enters and leaves once → O(n)
- Widely used for monitoring, rolling stats, and real-time metrics

### Recommended Reading

- LeetCode 239 — Sliding Window Maximum
- Monotonic Queue / Deque templates
- Rolling Aggregation / Streaming Analytics

---

## Conclusion

The value of Sliding Window Maximum lies in its reusable template.
Once you master the monotonic queue, you unlock a class of rolling-statistics problems.

---

## References

- https://leetcode.com/problems/sliding-window-maximum/
- https://en.cppreference.com/w/cpp/container/deque
- https://docs.python.org/3/library/collections.html#collections.deque
- https://doc.rust-lang.org/std/collections/struct.VecDeque.html

---

## Meta

- **Reading time**: 12–15 minutes  
- **Tags**: sliding window, monotonic queue, array  
- **SEO keywords**: Sliding Window Maximum, monotonic queue, deque  
- **Meta description**: Monotonic-queue solution for Sliding Window Maximum with engineering practice and multi-language implementations.  

---

## CTA

If you work on rolling metrics or real-time analytics, keep the monotonic queue as a core template.
Share your use cases in the comments.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

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
