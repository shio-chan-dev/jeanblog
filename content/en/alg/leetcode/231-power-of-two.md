---
title: "LeetCode 231: Power of Two (Bit Trick O(1) ACERS Guide)"
date: 2026-01-21T14:06:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["bit manipulation", "binary", "math", "LeetCode 231", "ACERS"]
description: "Judge whether an integer is a power of two in O(1) time using bit tricks. Includes engineering scenarios, pitfalls, and multi-language solutions."
keywords: ["Power of Two", "bit manipulation", "binary", "O(1)", "LeetCode 231"]
---

> **Subtitle / Summary**  
> A classic bit-manipulation template: determine if a number is a power of two in O(1). This ACERS guide covers the core insight, practical uses, and runnable multi-language implementations.

- **Reading time**: 8–12 min  
- **Tags**: `bit manipulation`, `binary`, `math`  
- **SEO keywords**: Power of Two, bit manipulation, binary, O(1), LeetCode 231  
- **Meta description**: O(1) power-of-two check using bit tricks, with engineering scenarios and multi-language code.

---

## Target Readers

- LeetCode learners building a bit-manipulation toolkit  
- Backend / systems engineers who need alignment or capacity checks  
- Anyone who wants stable O(1) integer tests

## Background / Motivation

Power-of-two checks show up everywhere: hash table capacities, memory alignment, sharding, FFT window sizes.  
Looping or using floating-point logs is slower and prone to corner-case bugs.  
The bitwise method is fast, simple, and reliable.

## Core Concepts

- **Binary form**: a power of two has exactly one `1` in its binary representation  
- **Bitwise AND**: `n & (n - 1)` clears the lowest set bit  
- **Positive-only**: `n` must be greater than 0

---

## A — Algorithm

### Problem Restatement

Given an integer `n`, determine whether it is a power of two.  
Return `true` if it is; otherwise, return `false`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| n | int | input integer |
| return | bool | whether `n` is a power of two |

### Example 1

```text
Input: n = 1
Output: true
Explanation: 2^0 = 1
```

### Example 2

```text
Input: n = 12
Output: false
Explanation: 12 in binary is 1100, which has multiple 1s
```

---

## C — Concepts

### Core Insight

A power of two has a single `1` bit:

```text
1  = 0001
2  = 0010
4  = 0100
8  = 1000
```

If `n` has exactly one `1`, then:

```text
n     = 1000...000
n - 1 = 0111...111
n & (n - 1) = 0
```

Therefore:

```text
n is power of two  ⟺  n > 0 and (n & (n - 1)) == 0
```

### Method Type

- **Bit manipulation (bit hacks)**
- **Constant-time numeric test**

---

## Practical Steps

1. Reject non-positive values: `n <= 0` → `false`  
2. Compute `(n & (n - 1))`  
3. If the result is `0`, return `true`

Runnable Python example (save as `power_of_two.py` and run `python3 power_of_two.py`):

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    print(is_power_of_two(1))   # True
    print(is_power_of_two(12))  # False
```

---

## E — Engineering

### Scenario 1: Data analysis / signal processing window size (Python)

**Background**: FFT and some convolution routines require power-of-two sizes.  
**Why it fits**: one-line validation avoids runtime failures.

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

window = 1024
if not is_power_of_two(window):
    raise ValueError("window must be power of two")
print("ok")
```

### Scenario 2: Memory alignment in allocators (C)

**Background**: memory allocators often align blocks to powers of two.  
**Why it fits**: the check is constant-time and branch-light.

```c
#include <stdio.h>
#include <stdint.h>

int is_pow2(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int main(void) {
    printf("%d\n", is_pow2(64));
    printf("%d\n", is_pow2(48));
    return 0;
}
```

### Scenario 3: Backend sharding / worker count validation (Go)

**Background**: shard counts are often powers of two to enable `idx & (n-1)` routing.  
**Why it fits**: avoids modulo cost and keeps mapping uniform.

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

## R — Reflection

### Complexity

- **Time**: O(1)  
- **Space**: O(1)

### Alternative Approaches

| Method | Idea | Complexity | Drawbacks |
| --- | --- | --- | --- |
| Divide by 2 loop | keep dividing while even | O(log n) | slower, more code |
| Popcount == 1 | count 1-bits | O(1) | library/intrinsic dependency |
| log2 check | check integer log | varies | floating-point precision |
| **Bit trick** | `n & (n - 1)` | **O(1)** | simplest and robust |

### Why This Is Best in Practice

- No loops, no divisions, no floating point  
- Stable across languages and integer sizes  
- Matches real-world systems requirements

---

## Explanation & Rationale

A power of two has a single set bit. Subtracting 1 flips that bit to 0 and turns all lower bits to 1.  
So only a number with one set bit will satisfy `n & (n - 1) == 0`.  
We must additionally check `n > 0` to rule out 0 and negatives.

---

## FAQs / Pitfalls

1. **Is `n = 0` a power of two?**  
   No. Always guard with `n > 0`.

2. **What about negative numbers?**  
   They are not powers of two in this context. The sign bit in two's complement breaks the single-bit property.

3. **Is `log2(n)` safe?**  
   Not reliably—floating-point precision can misclassify large values.

---

## Best Practices

- Always include the `n > 0` check  
- Encapsulate the logic into a small utility function  
- If you need the nearest power of two, build a separate helper instead of overloading this function

---

## S — Summary

### Key Takeaways

- A power of two has exactly one `1` bit  
- `n & (n - 1)` removes the lowest set bit  
- `n > 0` is required to exclude 0 and negatives  
- The bit trick is O(1), concise, and reliable  
- Widely used in hashing, alignment, and sharding

### Conclusion

This is a core bit-manipulation template worth memorizing. It shows up repeatedly in systems code and performance-critical logic.

### References & Further Reading

- LeetCode 231. Power of Two
- LeetCode 191. Number of 1 Bits
- LeetCode 342. Power of Four
- *Hacker's Delight* (bit tricks)
- *Computer Systems: A Programmer's Perspective* (binary operations)

---

## Meta

- **Reading time**: 8–12 min  
- **Tags**: bit manipulation, binary, math, LeetCode 231  
- **SEO keywords**: Power of Two, bit manipulation, binary, O(1)  
- **Meta description**: O(1) power-of-two check with bit tricks and engineering applications.

---

## Call to Action

Try converting a few related problems (power of four, count of 1 bits) into your own ACERS templates.  
Share your variants or engineering use cases in the comments.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    print(is_power_of_two(1))   # True
    print(is_power_of_two(12))  # False
```

```c
#include <stdio.h>

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
