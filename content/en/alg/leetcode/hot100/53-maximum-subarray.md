---
title: "Hot100: Maximum Subarray (Kadane O(n) ACERS Guide)"
date: 2026-01-23T13:21:02+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "greedy", "subarray", "LeetCode 53", "ACERS"]
description: "Use Kadane's algorithm to compute the maximum subarray sum in O(n). Includes engineering scenarios, pitfalls, and multi-language implementations."
keywords: ["Maximum Subarray", "Kadane", "dynamic programming", "O(n)", "Hot100"]
---

> **Subtitle / Summary**  
> Maximum Subarray is the classic 1D DP / greedy template. This ACERS guide explains Kadane's idea, engineering use cases, and runnable multi-language solutions.

- **Reading time**: 10–12 min  
- **Tags**: `Hot100`, `dynamic programming`, `greedy`  
- **SEO keywords**: Maximum Subarray, Kadane, dynamic programming, O(n), Hot100  
- **Meta description**: Kadane O(n) maximum subarray sum with engineering scenarios and multi-language code.

---

## Target Readers

- Hot100 learners building stable templates  
- Engineers analyzing peak segments in time series  
- Anyone who wants a clean O(n) solution

## Background / Motivation

Maximum subarray sum appears in P&L streaks, KPI lift windows, anomaly bursts, and throughput gains.  
The naive O(n^2) enumeration does not scale. Kadane's algorithm solves it in one pass.

## Core Concepts

- **Subarray**: contiguous, non-empty segment  
- **State**: `dp[i]` = best sum ending at index `i`  
- **Kadane**: if the running sum is negative, drop it and restart

---

## A — Algorithm

### Problem Restatement

Given an integer array `nums`, find the contiguous subarray with the largest sum (must contain at least one element) and return the sum.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | integer array |
| return | int | maximum subarray sum |

### Example 1 (official)

```text
nums = [-2,1,-3,4,-1,2,1,-5,4]
output = 6
explanation: subarray [4,-1,2,1] has sum 6
```

### Example 2 (official)

```text
nums = [1]
output = 1
```

---

## C — Concepts

### Key Formula

Let `dp[i]` be the maximum subarray sum ending at `i`:

```
dp[i] = max(nums[i], dp[i-1] + nums[i])
answer = max(dp[i])
```

### Method Type

- **1D DP**  
- **Greedy restart when prefix is negative**

### Intuition

If the best sum ending at `i-1` is negative, extending it only makes the sum worse.  
So we restart at `i`.

---

## Practical Steps

1. Initialize `cur = nums[0]`, `best = nums[0]`  
2. Scan from index 1:  
   - `cur = max(nums[i], cur + nums[i])`  
   - `best = max(best, cur)`  
3. Return `best`

Runnable Python example (save as `maximum_subarray.py`):

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

## E — Engineering

### Scenario 1: Profit streak detection (Python, data analysis)

**Background**: daily profit deltas, find the best contiguous streak.  
**Why it fits**: Kadane yields the peak gain window in O(n).

```python
def best_profit_streak(deltas):
    cur = best = deltas[0]
    for x in deltas[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

print(best_profit_streak([3, -2, 5, -1, 2, -4, 6]))
```

### Scenario 2: High-performance metric bursts (C++)

**Background**: find the strongest contiguous spike in CPU delta metrics.  
**Why it fits**: O(n) scan is cache-friendly and fast.

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

### Scenario 3: Backend throughput improvement window (Go)

**Background**: QPS deltas over time, find the best continuous improvement.  
**Why it fits**: Kadane works in a streaming update loop.

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

### Scenario 4: Frontend engagement lift (JavaScript)

**Background**: analyze consecutive engagement deltas to find best campaign window.  
**Why it fits**: linear scan in browser or Node.js.

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

console.log(maxSubArray([3, -2, 5, -1, 2]));
```

---

## R — Reflection

### Complexity

- **Time**: O(n)  
- **Space**: O(1)

### Alternatives

| Method | Idea | Complexity | Drawbacks |
| --- | --- | --- | --- |
| Brute force | enumerate all subarrays | O(n^2) | too slow |
| Prefix sums | compute each interval sum | O(n^2) | still slow |
| Divide & conquer | split and merge | O(n log n) | more complex |
| **Kadane** | 1D DP | **O(n)** | simplest and optimal |

### Why This Is Optimal

- One pass, constant memory  
- Clear correctness argument  
- Easy to embed in streaming pipelines

---

## Explanation & Rationale

Kadane's algorithm keeps the best sum ending at the current index.  
When the running sum turns negative, it cannot help any future subarray, so we restart.  
This gives the optimal sum with linear complexity.

---

## FAQs / Pitfalls

1. **What if all numbers are negative?**  
   Still works: the answer is the least negative single element.

2. **Are empty subarrays allowed?**  
   No, the problem requires at least one element.

3. **Do we need the interval indices?**  
   If needed, track start/end when updating `cur`.

---

## Best Practices

- Use two variables (`cur`, `best`) instead of full DP arrays  
- Convert complex signals into delta arrays before applying Kadane  
- If you need indices, store a candidate start pointer

---

## S — Summary

### Key Takeaways

- Maximum Subarray is a classic 1D DP template  
- Kadane drops negative prefixes for optimality  
- O(n) time and O(1) space scale well  
- Works for all-negative arrays without special cases  
- Common in profit, throughput, and metric burst analysis

### Conclusion

Kadane is a must-know pattern for contiguous optimum problems.  
Once mastered, you can apply it to many real-world sequences.

### References & Further Reading

- LeetCode 53. Maximum Subarray
- Standard DP textbooks (maximum subarray sum)
- CLRS discussion of divide-and-conquer vs Kadane

---

## Meta

- **Reading time**: 10–12 min  
- **Tags**: Hot100, dynamic programming, greedy, subarray  
- **SEO keywords**: Maximum Subarray, Kadane, O(n), Hot100  
- **Meta description**: Kadane O(n) maximum subarray sum with engineering scenarios.

---

## Call to Action

If you are working through Hot100, turn Kadane into a reusable template in your toolbox.  
Share your engineering adaptations in the comments.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

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
