---
title: "Hot100: First Missing Positive In-Place Index Placement ACERS Guide"
date: 2026-02-10T10:25:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "array", "in-place hashing", "index placement", "swapping", "LeetCode 41"]
description: "Find the first missing positive in O(n) time and O(1) extra space using in-place index placement (value x should be placed at index x-1), with engineering mapping, pitfalls, and runnable multi-language implementations."
keywords: ["First Missing Positive", "in-place hashing", "index mapping", "O(n)", "Hot100", "LeetCode 41"]
---

> **Subtitle / Summary**  
> First Missing Positive is a classic in-place indexing problem. Place each valid value `x` into slot `x-1`, then scan for the first mismatch. This ACERS guide explains the derivation, invariant, pitfalls, and production-style transfer.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `array`, `in-place hashing`  
- **SEO keywords**: First Missing Positive, in-place hashing, index mapping, O(n), Hot100, LeetCode 41  
- **Meta description**: O(n)/O(1) solution for First Missing Positive using in-place index placement, with complexity analysis, engineering scenarios, and runnable multi-language code.

---

## Target Readers

- Hot100 learners building stable array templates
- Intermediate developers who want to master in-place indexing techniques
- Engineers who need linear-time, constant-space array normalization

## Background / Motivation

"Find the smallest missing positive" is fundamentally a **placement** problem.

If value `x` is present and `1 <= x <= n`, then in an ideal arrangement it should sit at index `x-1`.
Once this structure is built, the answer is just the first index where the rule breaks.

The challenge is constraint-driven:

- O(n) time
- O(1) extra space

That forces us to avoid sorting (O(n log n)) and extra hash sets (O(n) space), and use in-place swapping instead.

## Core Concepts

| Concept | Meaning | Why it matters |
| --- | --- | --- |
| In-place hashing | Use array indices as hash buckets | keeps extra space O(1) |
| Index placement | value `x` belongs to index `x-1` | transforms search into scan |
| Swap-to-place | repeatedly swap until stable | enables O(n) normalization |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given an unsorted integer array `nums`, return the smallest missing positive integer.
You must design an algorithm with O(n) time and O(1) extra space.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | unsorted integer array |
| return | int | smallest missing positive integer |

### Example 1

```text
input: nums = [1, 2, 0]
output: 3
```

### Example 2

```text
input: nums = [3, 4, -1, 1]
output: 2
```

### High-Level Procedure

1. For each index `i`, keep swapping until `nums[i]` is either invalid or already in correct slot.
2. After placement, scan from left to right.
3. First index `i` where `nums[i] != i + 1` gives answer `i + 1`.
4. If all match, answer is `n + 1`.

---

## Thought Process: From Sorting/Hashing to In-Place Placement

### Naive option 1: brute-force candidate test

Try `1, 2, 3, ...` and scan the array each time.
This is O(n^2), not acceptable for large inputs.

### Naive option 2: sort then scan

Sorting makes order visible, but time becomes O(n log n), violating O(n) requirement.

### Naive option 3: hash set

Hash set gives O(n) time, but costs O(n) extra memory, violating O(1) space.

### Key observation

Only values in range `[1, n]` can affect the answer within `[1, n+1]`.
So we can treat the array itself as buckets:

```text
value x should be placed at index x - 1
```

Once placement converges, the first broken bucket reveals the answer.

---

## C - Concepts (Core Ideas)

### Method Category

- In-place hashing (index-as-hash)
- Array normalization via swapping
- Linear validation scan

### Core Invariant

After placement phase:

- if positive integer `k` exists in array and `1 <= k <= n`, then ideally `nums[k-1] == k`
- first index `i` with `nums[i] != i+1` means value `i+1` is missing

### Why repeated swaps are still linear

Each successful swap places at least one value into its final position.
A value can be moved only a limited number of times before becoming stable.
Total swaps across whole array remain O(n).

---

## Practical Guide / Steps

1. Let `n = len(nums)`, set pointer `i = 0`
2. While `i < n`:
   - `v = nums[i]`
   - if `1 <= v <= n` and `nums[v-1] != v`, swap `nums[i]` with `nums[v-1]`
   - else `i += 1`
3. Run a second scan:
   - first `i` where `nums[i] != i+1`, return `i+1`
4. If no mismatch, return `n+1`

---

## Runnable Example (Python)

```python
from typing import List


def first_missing_positive(nums: List[int]) -> int:
    n = len(nums)
    i = 0
    while i < n:
        v = nums[i]
        if 1 <= v <= n and nums[v - 1] != v:
            nums[i], nums[v - 1] = nums[v - 1], nums[i]
        else:
            i += 1

    for i, v in enumerate(nums):
        if v != i + 1:
            return i + 1
    return n + 1


if __name__ == "__main__":
    print(first_missing_positive([1, 2, 0]))
    print(first_missing_positive([3, 4, -1, 1]))
```

Run:

```bash
python3 first_missing_positive.py
```

---

## Explanation / Why This Works

This algorithm converts a search problem into a placement problem.

- Values outside `[1, n]` are irrelevant for first missing positive in `[1, n+1]`
- Valid value `x` is forced toward slot `x-1`
- Placement phase builds a partially sorted-by-meaning layout
- Scan phase finds the first missing positive in one pass

So we satisfy both constraints simultaneously: O(n) time and O(1) extra memory.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: next available compact ID in analytics batch (Python)

**Background**: find smallest missing positive ID in a daily import batch.  
**Why it fits**: no extra structures needed for large in-memory batches.

```python
def next_missing_id(ids):
    n = len(ids)
    i = 0
    while i < n:
        v = ids[i]
        if 1 <= v <= n and ids[v - 1] != v:
            ids[i], ids[v - 1] = ids[v - 1], ids[i]
        else:
            i += 1
    for idx, val in enumerate(ids):
        if val != idx + 1:
            return idx + 1
    return n + 1

print(next_missing_id([2, 1, 4, 6, 3]))
```

### Scenario 2: shard-index gap detection in backend config (Go)

**Background**: detect smallest missing shard number in an unsorted config list.  
**Why it fits**: linear-time check during startup validation.

```go
package main

import "fmt"

func firstMissingPositive(nums []int) int {
	n := len(nums)
	i := 0
	for i < n {
		v := nums[i]
		if v >= 1 && v <= n && nums[v-1] != v {
			nums[i], nums[v-1] = nums[v-1], nums[i]
		} else {
			i++
		}
	}
	for i, v := range nums {
		if v != i+1 {
			return i + 1
		}
	}
	return n + 1
}

func main() {
	fmt.Println(firstMissingPositive([]int{3, 4, -1, 1}))
}
```

### Scenario 3: front-end task sequence gap finder (JavaScript)

**Background**: allocate the smallest missing positive sequence number on client side.  
**Why it fits**: no dependency on backend call or extra cache.

```javascript
function firstMissingPositive(nums) {
  const n = nums.length;
  let i = 0;
  while (i < n) {
    const v = nums[i];
    if (v >= 1 && v <= n && nums[v - 1] !== v) {
      const tmp = nums[v - 1];
      nums[v - 1] = nums[i];
      nums[i] = tmp;
    } else {
      i += 1;
    }
  }
  for (let j = 0; j < n; j += 1) {
    if (nums[j] !== j + 1) return j + 1;
  }
  return n + 1;
}

console.log(firstMissingPositive([1, 2, 0]));
```

---

## R - Reflection

### Complexity

- Time: O(n)
- Extra space: O(1)

### Alternative Comparison

| Method | Time | Space | Issue |
| --- | --- | --- | --- |
| brute-force candidate checks | O(n^2) | O(1) | too slow |
| sorting + scan | O(n log n) | O(1) / O(log n) | violates linear-time target |
| hash set | O(n) | O(n) | violates constant-space target |
| in-place index placement | O(n) | O(1) | best fit for constraints |

### Common Mistakes

- Forgetting duplicate guard `nums[v-1] != v`, causing endless swaps
- Advancing `i` after every swap (should re-check new value at same index)
- Trying to place values outside `[1, n]`
- Returning wrong fallback (should be `n + 1` when all slots match)

### Why this method is practically optimal

It converts strict constraints into structure:

- no extra memory allocations
- linear scan + bounded swapping
- deterministic and template-friendly for array normalization tasks

---

## FAQ and Notes

1. **Why ignore non-positive values and values > n?**  
   They cannot be the first missing positive in `[1, n+1]`.

2. **Why can answer be `n+1`?**  
   If all `1..n` exist, the smallest missing positive is next one.

3. **Will duplicates break correctness?**  
   No, as long as duplicate guard is present before swap.

4. **Can this be done without swaps?**  
   There are marking-based variants, but swap placement is the most direct O(1)-space template.

---

## Best Practices

- Memorize the placement condition as one line:
  - `1 <= v <= n && nums[v-1] != v`
- Keep two phases explicit: placement then scan
- Add tests for edge cases:
  - all negatives
  - already continuous `[1..n]`
  - duplicates
  - empty array
- Prefer stable variable names (`v`, `n`, `i`) for readability in pointer-style loops

---

## S - Summary

- First Missing Positive is an index-placement problem under strict constraints
- In-place hashing maps value `x` to slot `x-1`
- Placement + validation scan gives O(n)/O(1)
- Duplicate guard and loop control are the two bug hotspots
- This pattern is highly reusable for array normalization tasks

### Recommended Follow-up

- LeetCode 448 — Find All Numbers Disappeared in an Array
- LeetCode 442 — Find All Duplicates in an Array
- LeetCode 287 — Find the Duplicate Number
- Cyclic sort / index placement pattern notes

---

## Conclusion

Once you internalize "place each value to its index slot, then scan first mismatch",
LeetCode 41 becomes a reusable engineering pattern rather than an interview trick.

---

## References

- https://leetcode.com/problems/first-missing-positive/
- https://en.cppreference.com/w/cpp/algorithm/swap
- https://docs.python.org/3/library/stdtypes.html#list
- https://go.dev/doc/effective_go

---

## Meta Info

- **Reading time**: 12-15 min  
- **Tags**: Hot100, array, in-place hashing  
- **SEO keywords**: First Missing Positive, in-place hashing, index mapping, LeetCode 41  
- **Meta description**: O(n)/O(1) first missing positive with in-place index placement and linear validation.

---

## Call To Action (CTA)

Do this drill sequence now:

1. Re-implement 41 from memory with duplicate guard
2. Adapt same idea to 448 (missing numbers)
3. Compare swap-based placement and sign-marking variants

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import List


def first_missing_positive(nums: List[int]) -> int:
    n = len(nums)
    i = 0
    while i < n:
        v = nums[i]
        if 1 <= v <= n and nums[v - 1] != v:
            nums[i], nums[v - 1] = nums[v - 1], nums[i]
        else:
            i += 1
    for i, v in enumerate(nums):
        if v != i + 1:
            return i + 1
    return n + 1
```

```c
int firstMissingPositive(int* nums, int numsSize) {
    int i = 0;
    while (i < numsSize) {
        int v = nums[i];
        if (v >= 1 && v <= numsSize && nums[v - 1] != v) {
            int tmp = nums[v - 1];
            nums[v - 1] = nums[i];
            nums[i] = tmp;
        } else {
            i++;
        }
    }
    for (i = 0; i < numsSize; ++i) {
        if (nums[i] != i + 1) return i + 1;
    }
    return numsSize + 1;
}
```

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = (int)nums.size();
        int i = 0;
        while (i < n) {
            int v = nums[i];
            if (v >= 1 && v <= n && nums[v - 1] != v) {
                swap(nums[i], nums[v - 1]);
            } else {
                i++;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }
};
```

```go
func firstMissingPositive(nums []int) int {
    n := len(nums)
    i := 0
    for i < n {
        v := nums[i]
        if v >= 1 && v <= n && nums[v-1] != v {
            nums[i], nums[v-1] = nums[v-1], nums[i]
        } else {
            i++
        }
    }
    for i, v := range nums {
        if v != i+1 {
            return i + 1
        }
    }
    return n + 1
}
```

```rust
pub fn first_missing_positive(nums: &mut Vec<i32>) -> i32 {
    let n = nums.len();
    let mut i = 0usize;
    while i < n {
        let v = nums[i];
        if v >= 1 && (v as usize) <= n && nums[(v - 1) as usize] != v {
            nums.swap(i, (v - 1) as usize);
        } else {
            i += 1;
        }
    }
    for (i, v) in nums.iter().enumerate() {
        if *v != (i as i32) + 1 {
            return (i as i32) + 1;
        }
    }
    (n as i32) + 1
}
```

```javascript
function firstMissingPositive(nums) {
  const n = nums.length;
  let i = 0;
  while (i < n) {
    const v = nums[i];
    if (v >= 1 && v <= n && nums[v - 1] !== v) {
      [nums[i], nums[v - 1]] = [nums[v - 1], nums[i]];
    } else {
      i += 1;
    }
  }
  for (let i = 0; i < n; i += 1) {
    if (nums[i] !== i + 1) return i + 1;
  }
  return n + 1;
}
```
