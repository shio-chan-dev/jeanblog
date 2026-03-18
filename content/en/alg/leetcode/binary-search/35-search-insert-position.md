---
title: "LeetCode 35: Search Insert Position Lower-Bound Binary Search ACERS Guide"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "lower bound", "sorted array", "LeetCode 35", "ACERS"]
description: "Find the insertion index of a target in a sorted array with lower-bound binary search in O(log n), with boundary reasoning, engineering mappings, and runnable implementations in six languages."
keywords: ["Search Insert Position", "lower bound", "binary search", "sorted array", "LeetCode 35"]
---

> **Subtitle / Summary**  
> Search Insert Position is the cleanest lower-bound problem in LeetCode. If you can reliably find the first index where `nums[i] >= target`, you already have the core template for insert positions, range starts, and many boundary-search problems.

- **Reading time**: 10-12 min  
- **Tags**: `binary search`, `lower bound`, `sorted array`  
- **SEO keywords**: Search Insert Position, lower bound, binary search, LeetCode 35  
- **Meta description**: Lower-bound binary search for Search Insert Position, with boundary reasoning, pitfalls, engineering scenarios, and runnable implementations in Python, C, C++, Go, Rust, and JavaScript.

## Target Readers

- Learners who know basic binary search but still hesitate on boundary handling
- Engineers who insert or locate values in sorted tables
- Interview candidates who want one reusable lower-bound template

## Background / Motivation

This problem looks simple because the output is a single index.  
The real lesson is deeper:

- if the target exists, return its index
- if it does not exist, return the insertion position

Those two requirements can be unified into one question:

> What is the first index whose value is greater than or equal to `target`?

That question is exactly `lower_bound`.

Once that idea is stable, a large family of binary-search problems becomes easier:

- insert position
- first occurrence
- range start
- counting `< target` or `>= target`

## Core Concepts

- **Sorted array**: binary search only works because order gives a monotonic decision rule
- **Lower bound**: the first index `i` such that `nums[i] >= target`
- **Half-open interval**: using `[l, r)` keeps the code concise and avoids fencepost bugs
- **Monotonic predicate**: `nums[i] >= target` is `false ... false, true ... true`

## A - Algorithm

### Problem Restatement

Given a non-decreasing integer array `nums` and an integer `target`:

- return the index of `target` if it exists
- otherwise return the index where `target` should be inserted to keep the array sorted

The required time complexity is `O(log n)`.

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `nums` | `int[]` | sorted array in non-decreasing order |
| `target` | `int` | value to search or insert |
| return | `int` | existing index or insertion index |

### Example 1

```text
nums   = [1, 3, 5, 6]
target = 5
output = 2
```

### Example 2

```text
nums   = [1, 3, 5, 6]
target = 2
output = 1
```

### Example 3

```text
nums   = [1, 3, 5, 6]
target = 7
output = 4
```

### Example 4

```text
nums   = [1, 3, 5, 6]
target = 0
output = 0
```

## Thought Process: From Linear Scan to Lower Bound

The brute-force idea is simple:

1. scan from left to right
2. stop at the first value `>= target`
3. return that index
4. if nothing qualifies, return `n`

That works, but it costs `O(n)`.

Because the array is already sorted, the condition

```text
nums[i] >= target
```

forms a monotonic pattern:

```text
false false false true true true
```

Binary search is exactly the tool for finding the first `true`.

## C - Concepts

### Method Category

- Binary search
- Boundary search
- Lower-bound template

### Why the Same Index Solves Both Cases

If `target` exists, the first position with `nums[i] >= target` is the first position where `nums[i] == target`.

If `target` does not exist, the first position with `nums[i] >= target` is the place where `target` must be inserted.

So one return value handles both outcomes.

### Stable Template

Use a half-open interval `[l, r)`:

1. initialize `l = 0`, `r = len(nums)`
2. while `l < r`:
   - let `mid = l + (r - l) // 2`
   - if `nums[mid] >= target`, keep the answer on the left: `r = mid`
   - otherwise move right: `l = mid + 1`
3. return `l`

### Why `[l, r)` Is Convenient

- `r` can safely start at `len(nums)`
- the insertion-at-end case naturally returns `len(nums)`
- the loop invariant is easy: the answer always stays inside `[l, r)`

## E - Engineering

### Scenario 1: Threshold Tables in Backend Services (Python)

**Background**: a service stores sorted thresholds such as latency buckets or score cutoffs.  
**Why it fits**: you need the first threshold that is not smaller than the incoming value.

```python
def search_insert(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l


thresholds = [10, 30, 60, 100]
for x in [5, 10, 25, 70, 101]:
    print(x, "-> slot", search_insert(thresholds, x))
```

### Scenario 2: Pricing or Risk Tiers (Go)

**Background**: an order amount must be mapped into the correct sorted tier table.  
**Why it fits**: insert position is exactly the tier index.

```go
package main

import "fmt"

func searchInsert(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func main() {
	tiers := []int{1000, 5000, 10000, 50000}
	for _, amount := range []int{500, 1000, 2000, 20000} {
		fmt.Println(amount, "-> tier", searchInsert(tiers, amount))
	}
}
```

### Scenario 3: Frontend Timeline or Version Selection (JavaScript)

**Background**: a UI highlights the first version not smaller than the current version.  
**Why it fits**: the highlighted node is a lower-bound lookup.

```js
function searchInsert(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) r = mid;
    else l = mid + 1;
  }
  return l;
}

console.log(searchInsert([1, 3, 5, 6], 5)); // 2
console.log(searchInsert([1, 3, 5, 6], 4)); // 2
```

## R - Reflection

### Complexity

- Time: `O(log n)`
- Space: `O(1)`

### Alternatives

- **Linear scan**: simpler, but `O(n)` and misses the point of the sorted input
- **Library lower_bound / bisect_left**: excellent in production, but you still need the boundary model to use them correctly

### Common Mistakes

- Using `>` instead of `>=`, which turns the answer into an upper bound
- Returning immediately when `nums[mid] == target`, which loses the insertion-position interpretation
- Mixing interval styles, for example using `[l, r)` updates with a `[l, r]` initialization

### Why This Is the Best Practical Method

The array is sorted, the predicate is monotonic, and the return value is a boundary index.  
That is exactly the shape binary search is designed for, so this is both the optimal asymptotic solution and the cleanest engineering template.

## S - Summary

- Search Insert Position is a pure lower-bound problem.
- The answer is the first index where `nums[i] >= target`.
- A half-open interval `[l, r)` makes the boundary logic stable.
- This template directly extends to range queries, counts, and insertion-point lookups in real systems.

## Further Reading

- LeetCode 34: Find First and Last Position of Element in Sorted Array
- LeetCode 744: Find Smallest Letter Greater Than Target
- Python `bisect_left`, C++ `lower_bound`, and Go `sort.Search`

## Multi-language Implementations

### Python

```python
from typing import List


def search_insert(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l


if __name__ == "__main__":
    print(search_insert([1, 3, 5, 6], 5))  # 2
    print(search_insert([1, 3, 5, 6], 2))  # 1
    print(search_insert([1, 3, 5, 6], 7))  # 4
    print(search_insert([1, 3, 5, 6], 0))  # 0
```

### C

```c
#include <stdio.h>

int searchInsert(int *nums, int numsSize, int target) {
    int l = 0, r = numsSize;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

int main(void) {
    int nums[] = {1, 3, 5, 6};
    int n = sizeof(nums) / sizeof(nums[0]);
    printf("%d\n", searchInsert(nums, n, 5));
    printf("%d\n", searchInsert(nums, n, 2));
    printf("%d\n", searchInsert(nums, n, 7));
    printf("%d\n", searchInsert(nums, n, 0));
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <vector>
using namespace std;

int searchInsert(const vector<int>& nums, int target) {
    int l = 0, r = (int)nums.size();
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

int main() {
    vector<int> nums{1, 3, 5, 6};
    cout << searchInsert(nums, 5) << "\n";
    cout << searchInsert(nums, 2) << "\n";
    cout << searchInsert(nums, 7) << "\n";
    cout << searchInsert(nums, 0) << "\n";
    return 0;
}
```

### Go

```go
package main

import "fmt"

func searchInsert(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] >= target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func main() {
	nums := []int{1, 3, 5, 6}
	fmt.Println(searchInsert(nums, 5))
	fmt.Println(searchInsert(nums, 2))
	fmt.Println(searchInsert(nums, 7))
	fmt.Println(searchInsert(nums, 0))
}
```

### Rust

```rust
fn search_insert(nums: &[i32], target: i32) -> usize {
    let (mut l, mut r) = (0usize, nums.len());
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] >= target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    l
}

fn main() {
    let nums = vec![1, 3, 5, 6];
    println!("{}", search_insert(&nums, 5));
    println!("{}", search_insert(&nums, 2));
    println!("{}", search_insert(&nums, 7));
    println!("{}", search_insert(&nums, 0));
}
```

### JavaScript

```js
function searchInsert(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) {
      r = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

console.log(searchInsert([1, 3, 5, 6], 5));
console.log(searchInsert([1, 3, 5, 6], 2));
console.log(searchInsert([1, 3, 5, 6], 7));
console.log(searchInsert([1, 3, 5, 6], 0));
```
