---
title: "Hot100: Trapping Rain Water (Two Pointers O(n) ACERS Guide)"
date: 2026-01-24T10:40:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "two pointers", "array", "prefix max", "LeetCode 42", "ACERS"]
description: "Compute trapped rain water in O(n) using two pointers and left/right maxima. Includes engineering scenarios, pitfalls, and multi-language implementations."
keywords: ["Trapping Rain Water", "two pointers", "left right max", "O(n)", "Hot100"]
---

> **Subtitle / Summary**  
> Trapping Rain Water is the classic boundary-constraint problem. This ACERS guide explains the two-pointer method, key formulas, and runnable multi-language solutions.

- **Reading time**: 12–15 min  
- **Tags**: `Hot100`, `two pointers`, `array`  
- **SEO keywords**: Trapping Rain Water, two pointers, left right max, O(n), Hot100  
- **Meta description**: Two-pointer O(n) trapped water solution with engineering scenarios and multi-language code.

---

## Target Readers

- Hot100 learners building core templates  
- Engineers handling capacity/volume constraints  
- Anyone who wants a clean O(n) solution

## Background / Motivation

Trapped water is a proxy for “capacity under boundary constraints.”  
It appears in cache headroom estimation, buffer overflow analysis, and terrain capacity modeling.  
The naive O(n^2) method is too slow; the two-pointer approach reduces it to O(n).

## Core Concepts

- **Local water level**: `water[i] = min(maxLeft[i], maxRight[i]) - height[i]`  
- **Boundary constraints**: the lower side limits water  
- **Two pointers**: maintain left/right maxima in one pass

---

## A — Algorithm

### Problem Restatement

Given an array of non-negative integers representing bar heights (each width 1), compute how much water can be trapped after raining.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| height | int[] | bar heights |
| return | int | total trapped water |

### Example 1 (official)

```text
height = [0,1,0,2,1,0,1,3,2,1,2,1]
output = 6
```

### Example 2 (official)

```text
height = [4,2,0,3,2,5]
output = 9
```

---

## C — Concepts

### Key Formula

For each index `i`:

```
water[i] = min(maxLeft[i], maxRight[i]) - height[i]
```

### Method Type

- **Two pointers**  
- **Left/right maxima boundary**

### Intuition

The lower of the two boundaries determines the water level.  
If `leftMax <= rightMax`, the left side is settled and can be computed safely.

---

## Practical Steps

1. Initialize `l=0`, `r=n-1`, `leftMax`, `rightMax`  
2. Update `leftMax` and `rightMax` each step  
3. If `leftMax <= rightMax`, accumulate `leftMax - height[l]` and move `l`  
4. Otherwise accumulate `rightMax - height[r]` and move `r`

Runnable Python example (save as `trapping_rain_water.py`):

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

## E — Engineering

### Scenario 1: Cache headroom estimation (Python)

**Background**: treat usage as heights and compute “empty capacity” between peaks.  
**Why it fits**: identical boundary-constrained volume calculation.

```python
def free_capacity(usage):
    return trap(usage)

print(free_capacity([2, 0, 2]))
```

### Scenario 2: Terrain cross-section volume (C++)

**Background**: approximate water volume on a 1D elevation slice.  
**Why it fits**: left/right maxima are the limiting walls.

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

### Scenario 3: Backend buffer overflow risk (Go)

**Background**: estimate how much extra load can fit between high-water marks.  
**Why it fits**: two-pointer O(n) is fast enough for online checks.

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

## R — Reflection

### Complexity

- **Time**: O(n)  
- **Space**: O(1)

### Alternatives

| Method | Idea | Complexity | Drawbacks |
| --- | --- | --- | --- |
| Brute force | scan left/right for each index | O(n^2) | too slow |
| Precompute arrays | store maxLeft/maxRight | O(n) | extra memory |
| Monotonic stack | compute basins | O(n) | more complex |
| **Two pointers** | online maxima | **O(n)** | simplest |

### Why This Is Best in Practice

- No extra arrays  
- Linear scan, easy to reason about  
- Great for streaming or large datasets

---

## Explanation & Rationale

Water is bounded by the lower of the two sides.  
By always processing the side with the smaller boundary, we ensure the water level is fixed.  
This allows a single pass without missing any contribution.

---

## FAQs / Pitfalls

1. **Why compare `leftMax <= rightMax`?**  
   The smaller boundary determines the water level on that side.

2. **Do zeros break anything?**  
   No, zeros are just low bars.

3. **Are negative heights allowed?**  
   The problem restricts to non-negative heights.

---

## Best Practices

- Use the two-pointer variant for O(1) space  
- Use the precompute variant if you want clearer intermediate arrays  
- Make sure indices don’t cross (`l < r`)

---

## S — Summary

### Key Takeaways

- Trapped water depends on left/right maxima  
- Two pointers compute it in one pass  
- O(n) time and O(1) space  
- Applicable to capacity and boundary-constrained volume problems  
- Hot100 essential template

### Conclusion

The two-pointer solution is both elegant and production-friendly.  
Mastering it gives you a reusable pattern for boundary-constrained volume problems.

### References & Further Reading

- LeetCode 42. Trapping Rain Water
- Monotonic stack techniques
- Boundary constraint modeling

---

## Meta

- **Reading time**: 12–15 min  
- **Tags**: Hot100, two pointers, array, prefix max  
- **SEO keywords**: Trapping Rain Water, two pointers, O(n), Hot100  
- **Meta description**: Two-pointer O(n) trapped water with engineering scenarios and code.

---

## Call to Action

If you are working through Hot100, turn this into a template for boundary-constrained problems.  
Share your real-world adaptations in the comments.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

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
