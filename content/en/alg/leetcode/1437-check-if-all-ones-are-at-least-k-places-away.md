---
title: "LeetCode 1437: Check If All 1's Are at Least K Apart (ACERS Guide)"
date: 2026-01-22T10:49:42+08:00
draft: false
categories: ["LeetCode"]
tags: ["array", "two pointers", "greedy", "LeetCode 1437", "ACERS"]
description: "One-pass check to ensure all 1s are at least k apart. Includes engineering scenarios, pitfalls, and multi-language implementations."
keywords: ["Check If All 1's Are at Least Length K Places Away", "event spacing", "O(n)", "LeetCode 1437"]
---

> **Subtitle / Summary**  
> A classic event-spacing validation model. This ACERS guide explains the one-pass logic, engineering use cases, and runnable multi-language solutions.

- **Reading time**: 10–12 min  
- **Tags**: `array`, `two pointers`, `event spacing`  
- **SEO keywords**: LeetCode 1437, event spacing, O(n)  
- **Meta description**: One-pass validation for minimum spacing between 1s, with engineering use cases and multi-language code.

---

## Target Readers

- LeetCode learners building stable templates  
- Engineers working on monitoring / risk control / behavior analytics  
- Developers who need spacing or rate-limit validations

## Background / Motivation

Many systems require events to be spaced apart: login failures, alarms, sensitive actions, API calls, etc.  
This problem maps directly to **event spacing validation**.  
A one-pass, O(1)-memory solution is ideal for real-time systems.

## Core Concepts

- **Event spacing**: at least `k` zeros between two `1`s  
- **Online validation**: only the last event index is needed  
- **Boundary handling**: initialize `last = -k-1` to avoid special cases

---

## A — Algorithm

### Problem Restatement

Given an integer array `nums` and integer `k`, return `true` if every pair of `1`s is at least `k` apart; otherwise return `false`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | binary array with 0/1 |
| k | int | required minimum spacing |
| return | bool | whether the spacing rule holds |

### Example 1

```text
nums = [1,0,0,0,1,0,0,1], k = 2
output = true
```

### Example 2

```text
nums = [1,0,1], k = 2
output = false
```

---

## C — Concepts

### Key Observation

- Track the index of the **last seen 1** (`last`)  
- On each new `1`, if `i - last <= k` → spacing violated

### Method Type

- **One-pass scan**  
- **Event spacing validation**  
- **Greedy with a last pointer**

### Formula

Spacing requirement:

```
(j - i - 1) >= k  ⇔  (j - i) > k
```

So the violation check is:

```
if i - last <= k: return false
```

---

## Practical Steps

1. Set `last = -k - 1` (so the first 1 always passes)  
2. Scan from left to right  
3. When seeing `1`, check distance to `last`  
4. If too close, return `false`  
5. Otherwise update `last` and continue

Runnable Python example (save as `k_length_apart.py`):

```python
def k_length_apart(nums, k):
    last = -k - 1
    for i, x in enumerate(nums):
        if x == 1:
            if i - last <= k:
                return False
            last = i
    return True


if __name__ == "__main__":
    print(k_length_apart([1, 0, 0, 0, 1, 0, 0, 1], 2))  # True
    print(k_length_apart([1, 0, 1], 2))                  # False
```

---

## E — Engineering

### Scenario 1: Risk control for login failures (Python)

**Background**: repeated login failures too close together suggest brute force.  
**Why it fits**: only the last failure index is required.

```python
def check_login_spacing(events, k):
    last = -k - 1
    for i, x in enumerate(events):
        if x != 1:
            continue
        if i - last <= k:
            return False
        last = i
    return True
```

### Scenario 2: Monitoring error density (Go)

**Background**: errors shouldn’t occur too frequently in a time window.  
**Why it fits**: O(1) memory, stream-friendly.

```go
package main

import "fmt"

func okSpacing(log []int, k int) bool {
    last := -k - 1
    for i, x := range log {
        if x == 1 {
            if i-last <= k {
                return false
            }
            last = i
        }
    }
    return true
}

func main() {
    fmt.Println(okSpacing([]int{1, 0, 0, 1}, 2))
}
```

### Scenario 3: Debounce in embedded systems (C)

**Background**: sensor triggers must be spaced to avoid bouncing.  
**Why it fits**: minimal state, fast checks.

```c
#include <stdio.h>

int k_length_apart(const int *a, int n, int k) {
    int last = -k - 1;
    for (int i = 0; i < n; ++i) {
        if (a[i] == 1) {
            if (i - last <= k) return 0;
            last = i;
        }
    }
    return 1;
}

int main(void) {
    int a[] = {1,0,0,1};
    printf("%d\n", k_length_apart(a, 4, 2));
    return 0;
}
```

### Scenario 4: Frontend click throttling (JavaScript)

**Background**: avoid bursts of high-value actions.  
**Why it fits**: same spacing model on a click sequence.

```javascript
function okSpacing(events, k) {
  let last = -k - 1;
  for (let i = 0; i < events.length; i++) {
    if (events[i] === 1) {
      if (i - last <= k) return false;
      last = i;
    }
  }
  return true;
}

console.log(okSpacing([1, 0, 0, 1], 2));
```

---

## R — Reflection

### Complexity

- **Time**: O(n)  
- **Space**: O(1)

### Alternatives

| Method | Idea | Complexity | Drawbacks |
| --- | --- | --- | --- |
| Store all 1 indices | then validate gaps | O(n) | extra memory |
| Double loop | compare all pairs | O(n^2) | too slow |
| **One-pass** | keep last index | **O(n)** | simplest |

### Why This Is Best

- Minimal state  
- Works in streaming systems  
- Straightforward correctness

---

## Explanation & Rationale

Keeping only the last `1` is enough because the constraint is local to **consecutive** 1s.  
Initializing `last = -k-1` creates a “virtual 1” so the first real 1 always passes.  
Any time `i - last <= k`, the rule is violated.

---

## FAQs / Pitfalls

1. **Why `i - last <= k`?**  
   The requirement `(i - last - 1) >= k` rearranges to `i - last > k`.

2. **Is `k = 0` valid?**  
   Yes. It means adjacent 1s are allowed.

3. **Is the array required to be 0/1?**  
   The problem is binary, but the model can be generalized if you define “event” as 1.

---

## Best Practices

- Use `last = -k-1` to avoid special cases  
- Wrap the logic as a reusable spacing validator  
- Combine with rate-limit checks if needed

---

## S — Summary

### Key Takeaways

- The task is an event-spacing validation problem  
- Only the last event index is needed  
- Initialization trick simplifies boundary handling  
- One-pass scan gives O(n)/O(1)  
- Useful in risk control, monitoring, and throttling

### Conclusion

This is a simple but powerful template that maps directly to production systems.  
Turn it into a reusable utility and you’ll use it again and again.

### References & Further Reading

- LeetCode 1437. Check If All 1's Are at Least Length K Places Away
- Rate limiting / debounce / throttle docs
- Event stream processing basics

---

## Meta

- **Reading time**: 10–12 min  
- **Tags**: array, event spacing, monitoring, risk control  
- **SEO keywords**: LeetCode 1437, event spacing, O(n)  
- **Meta description**: One-pass minimum spacing validation with engineering use cases.

---

## Call to Action

If you’re building monitoring or risk-control systems, add this “event spacing” template to your toolkit.  
Share your real-world adaptations in the comments.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
def k_length_apart(nums, k):
    last = -k - 1
    for i, x in enumerate(nums):
        if x == 1:
            if i - last <= k:
                return False
            last = i
    return True


if __name__ == "__main__":
    print(k_length_apart([1, 0, 0, 0, 1, 0, 0, 1], 2))
```

```c
#include <stdio.h>

int k_length_apart(const int *a, int n, int k) {
    int last = -k - 1;
    for (int i = 0; i < n; ++i) {
        if (a[i] == 1) {
            if (i - last <= k) return 0;
            last = i;
        }
    }
    return 1;
}

int main(void) {
    int a[] = {1,0,0,1};
    printf("%d\n", k_length_apart(a, 4, 2));
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

bool kLengthApart(const std::vector<int>& nums, int k) {
    int last = -k - 1;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (nums[i] == 1) {
            if (i - last <= k) return false;
            last = i;
        }
    }
    return true;
}

int main() {
    std::cout << std::boolalpha << kLengthApart({1,0,0,1}, 2) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func kLengthApart(nums []int, k int) bool {
    last := -k - 1
    for i, x := range nums {
        if x == 1 {
            if i-last <= k {
                return false
            }
            last = i
        }
    }
    return true
}

func main() {
    fmt.Println(kLengthApart([]int{1, 0, 0, 1}, 2))
}
```

```rust
fn k_length_apart(nums: &[i32], k: i32) -> bool {
    let mut last = -k - 1;
    for (i, &x) in nums.iter().enumerate() {
        if x == 1 {
            let i = i as i32;
            if i - last <= k {
                return false;
            }
            last = i;
        }
    }
    true
}

fn main() {
    let nums = vec![1, 0, 0, 1];
    println!("{}", k_length_apart(&nums, 2));
}
```

```javascript
function kLengthApart(nums, k) {
  let last = -k - 1;
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] === 1) {
      if (i - last <= k) return false;
      last = i;
    }
  }
  return true;
}

console.log(kLengthApart([1, 0, 0, 1], 2));
```
