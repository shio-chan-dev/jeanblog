---
title: "LeetCode 2529: Maximum Count of Positive Integer and Negative Integer ACERS Guide"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "lower bound", "upper bound", "sorted array", "counting", "LeetCode 2529", "ACERS"]
description: "Count negatives and positives in a sorted array by locating the boundaries around zero with lower-bound and upper-bound binary search, with engineering mappings and runnable implementations in six languages."
keywords: ["Maximum Count of Positive Integer and Negative Integer", "binary search", "lower bound", "upper bound", "LeetCode 2529"]
---

> **Subtitle / Summary**  
> This problem is a compact exercise in boundary counting. Because the array is already sorted, you do not count negatives and positives one by one; you find where zero starts and where zero ends, then compute both counts from those boundaries.

- **Reading time**: 10-12 min  
- **Tags**: `binary search`, `counting`, `sorted array`, `boundaries`  
- **SEO keywords**: Maximum Count of Positive Integer and Negative Integer, LeetCode 2529, boundary counting  
- **Meta description**: Use lower-bound and upper-bound binary search around zero to count negatives and positives in a sorted array, with correctness reasoning, engineering scenarios, and runnable implementations in six languages.

## Target Readers

- Learners practicing boundary search beyond exact-match lookup
- Engineers who count segments in sorted data
- Interview candidates learning how lower and upper bounds produce counts

## Background / Motivation

The input is already sorted. That changes the problem completely.

Instead of scanning the whole array and incrementing counters, we can ask:

- where do negative numbers stop?
- where do positive numbers start?

Zeros act as the separator in the middle.  
So this is really a boundary problem around the value `0`.

## Core Concepts

- **Negative count**: number of values `< 0`
- **Positive count**: number of values `> 0`
- **Lower bound of 0**: first index where `nums[i] >= 0`
- **Upper bound of 0**: first index where `nums[i] > 0`

From those:

- `neg = lower_bound(0)`
- `pos = n - upper_bound(0)`

## A - Algorithm

### Problem Restatement

Given a sorted integer array `nums` that may contain negative numbers, zeros, and positive numbers:

- let `countNeg` be the number of elements `< 0`
- let `countPos` be the number of elements `> 0`

Return `max(countNeg, countPos)`.

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `nums` | `int[]` | sorted integer array |
| return | `int` | larger of negative count and positive count |

### Example 1

```text
nums   = [-3, -2, -1, 0, 0, 1, 2]
output = 3
```

### Example 2

```text
nums   = [-2, -1, -1, 1, 2, 3]
output = 3
```

### Example 3

```text
nums   = [0, 0, 0]
output = 0
```

## Thought Process: From Counting to Boundaries

The direct idea is:

- scan the array
- count negatives
- count positives

That is `O(n)`, and for this problem it is actually acceptable.

But because the array is sorted, we can do better conceptually:

- negatives are on the left
- zeros are in the middle
- positives are on the right

So the counts are determined by two positions:

- the first index `>= 0`
- the first index `> 0`

## C - Concepts

### Method Category

- Binary search
- Boundary counting
- Sorted partition lookup

### Why Two Searches Are Enough

Let:

- `a = lower_bound(nums, 0)` => first non-negative index
- `b = upper_bound(nums, 0)` => first positive index

Then:

- all indices `[0, a)` are negative
- all indices `[b, n)` are positive

So:

- `countNeg = a`
- `countPos = n - b`

### Stable Algorithm

1. compute `neg = lower_bound(nums, 0)`
2. compute `pos = len(nums) - upper_bound(nums, 0)`
3. return `max(neg, pos)`

## E - Engineering

### Scenario 1: Signed Score Analysis (Python)

**Background**: a sorted score array contains losses, neutral events, and gains.  
**Why it fits**: you only need the boundaries around zero, not a full scan for every query.

```python
def lower_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l


def upper_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l


nums = [-3, -2, -1, 0, 0, 1, 2]
print(max(lower_bound(nums, 0), len(nums) - upper_bound(nums, 0)))
```

### Scenario 2: Sorted Risk Buckets (Go)

**Background**: a risk engine stores sorted signed deviations and needs the dominant side.  
**Why it fits**: zero is the natural split point.

```go
package main

import "fmt"

func lowerBound(nums []int, target int) int {
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

func upperBound(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func main() {
	nums := []int{-2, -1, -1, 1, 2, 3}
	neg := lowerBound(nums, 0)
	pos := len(nums) - upperBound(nums, 0)
	fmt.Println(max(neg, pos))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

### Scenario 3: Frontend Sorted Trend Display (JavaScript)

**Background**: a UI shows sorted changes and wants to summarize whether negative or positive values dominate.  
**Why it fits**: a boundary lookup is cheaper than repeated category checks when the list is reused.

```js
function lowerBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) r = mid;
    else l = mid + 1;
  }
  return l;
}

function upperBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l;
}

const nums = [-3, -2, -1, 0, 0, 1, 2];
console.log(Math.max(lowerBound(nums, 0), nums.length - upperBound(nums, 0)));
```

## R - Reflection

### Complexity

- Time: `O(log n)`
- Space: `O(1)`

### Alternatives

- **Linear scan**: valid and simple, but it ignores the structural value of the sorted input
- **Two pointers from both ends**: unnecessary and harder to reason about than boundary search

### Common Mistakes

- Counting zeros as positive or negative
- Using only one boundary and trying to infer both counts from it
- Using `>= 0` when you need strictly positive count

### Why This Method Is the Most Reusable

This problem is really about extracting counts from sorted partitions.  
That exact pattern appears in metric analysis, score bands, and threshold reporting, so lower and upper bounds are the right abstractions.

## S - Summary

- The sorted array splits naturally into negatives, zeros, and positives.
- `lower_bound(0)` gives the negative count.
- `len(nums) - upper_bound(0)` gives the positive count.
- Boundary search turns counting into a clean `O(log n)` partition lookup.

## Further Reading

- LeetCode 35: Search Insert Position
- LeetCode 34: Find First and Last Position of Element in Sorted Array
- Any standard `lower_bound` / `upper_bound` documentation

## Multi-language Implementations

### Python

```python
from typing import List


def lower_bound(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l


def upper_bound(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l


def maximum_count(nums: List[int]) -> int:
    neg = lower_bound(nums, 0)
    pos = len(nums) - upper_bound(nums, 0)
    return max(neg, pos)


if __name__ == "__main__":
    print(maximum_count([-3, -2, -1, 0, 0, 1, 2]))
    print(maximum_count([-2, -1, -1, 1, 2, 3]))
    print(maximum_count([0, 0, 0]))
```

### C

```c
#include <stdio.h>

int lowerBound(int *nums, int n, int target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target) r = mid;
        else l = mid + 1;
    }
    return l;
}

int upperBound(int *nums, int n, int target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > target) r = mid;
        else l = mid + 1;
    }
    return l;
}

int maximumCount(int *nums, int n) {
    int neg = lowerBound(nums, n, 0);
    int pos = n - upperBound(nums, n, 0);
    return neg > pos ? neg : pos;
}

int main(void) {
    int a[] = {-3, -2, -1, 0, 0, 1, 2};
    int b[] = {-2, -1, -1, 1, 2, 3};
    int c[] = {0, 0, 0};
    printf("%d\n", maximumCount(a, 7));
    printf("%d\n", maximumCount(b, 6));
    printf("%d\n", maximumCount(c, 3));
    return 0;
}
```

### C++

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int maximumCount(const vector<int>& nums) {
    int neg = lower_bound(nums.begin(), nums.end(), 0) - nums.begin();
    int pos = nums.end() - upper_bound(nums.begin(), nums.end(), 0);
    return max(neg, pos);
}

int main() {
    cout << maximumCount({-3, -2, -1, 0, 0, 1, 2}) << "\n";
    cout << maximumCount({-2, -1, -1, 1, 2, 3}) << "\n";
    cout << maximumCount({0, 0, 0}) << "\n";
    return 0;
}
```

### Go

```go
package main

import "fmt"

func lowerBound(nums []int, target int) int {
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

func upperBound(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

func maximumCount(nums []int) int {
	neg := lowerBound(nums, 0)
	pos := len(nums) - upperBound(nums, 0)
	if neg > pos {
		return neg
	}
	return pos
}

func main() {
	fmt.Println(maximumCount([]int{-3, -2, -1, 0, 0, 1, 2}))
	fmt.Println(maximumCount([]int{-2, -1, -1, 1, 2, 3}))
	fmt.Println(maximumCount([]int{0, 0, 0}))
}
```

### Rust

```rust
fn lower_bound(nums: &[i32], target: i32) -> usize {
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

fn upper_bound(nums: &[i32], target: i32) -> usize {
    let (mut l, mut r) = (0usize, nums.len());
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    l
}

fn maximum_count(nums: &[i32]) -> usize {
    let neg = lower_bound(nums, 0);
    let pos = nums.len() - upper_bound(nums, 0);
    neg.max(pos)
}

fn main() {
    println!("{}", maximum_count(&[-3, -2, -1, 0, 0, 1, 2]));
    println!("{}", maximum_count(&[-2, -1, -1, 1, 2, 3]));
    println!("{}", maximum_count(&[0, 0, 0]));
}
```

### JavaScript

```js
function lowerBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] >= target) r = mid;
    else l = mid + 1;
  }
  return l;
}

function upperBound(nums, target) {
  let l = 0, r = nums.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (nums[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l;
}

function maximumCount(nums) {
  const neg = lowerBound(nums, 0);
  const pos = nums.length - upperBound(nums, 0);
  return Math.max(neg, pos);
}

console.log(maximumCount([-3, -2, -1, 0, 0, 1, 2]));
console.log(maximumCount([-2, -1, -1, 1, 2, 3]));
console.log(maximumCount([0, 0, 0]));
```
