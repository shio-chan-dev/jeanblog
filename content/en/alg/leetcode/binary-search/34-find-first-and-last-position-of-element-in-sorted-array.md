---
title: "LeetCode 34: Find First and Last Position of Element in Sorted Array ACERS Guide"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "lower bound", "upper bound", "sorted array", "LeetCode 34", "ACERS"]
description: "Find the start and end positions of a target in a sorted array by combining lower-bound and upper-bound binary search, with boundary reasoning, engineering mappings, and runnable implementations in six languages."
keywords: ["Find First and Last Position of Element in Sorted Array", "Search Range", "binary search", "lower bound", "upper bound", "LeetCode 34"]
---

> **Subtitle / Summary**  
> This problem is the standard upgrade from “find one target” to “find the whole target block.” The clean solution is not a special-case binary search, but two boundary searches: lower bound for the start and upper bound for the end.

- **Reading time**: 12-14 min  
- **Tags**: `binary search`, `lower bound`, `upper bound`, `range query`  
- **SEO keywords**: Search Range, lower bound, upper bound, LeetCode 34  
- **Meta description**: Use lower-bound and upper-bound binary search to find the first and last positions of a target in a sorted array, with pitfalls, engineering scenarios, and runnable implementations in six languages.

## Target Readers

- Learners who already know basic binary search but struggle with boundary problems
- Engineers who query sorted logs, timestamps, or grouped IDs
- Interview candidates who want one reusable range-search template

## Background / Motivation

Finding a single target in a sorted array is the easy version.  
Real systems often need the full range:

- all log entries at the same timestamp
- all records with the same sorted key
- all metrics points equal to one threshold

So the real question becomes:

- where does the target block start?
- where does the target block end?

That is why this problem is best understood as a combination of:

- `lower_bound(target)`
- `upper_bound(target)`

## Core Concepts

- **Lower bound**: first index `i` such that `nums[i] >= target`
- **Upper bound**: first index `i` such that `nums[i] > target`
- **Target range**: if the target exists, the answer is
  - `start = lower_bound(target)`
  - `end = upper_bound(target) - 1`

## A - Algorithm

### Problem Restatement

Given a non-decreasing integer array `nums` and an integer `target`, return:

- `[start, end]` if `target` appears in the array
- `[-1, -1]` otherwise

The required time complexity is `O(log n)`.

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `nums` | `int[]` | sorted array in non-decreasing order |
| `target` | `int` | value to locate |
| return | `int[]` | `[start, end]` or `[-1, -1]` |

### Example 1

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 8
output = [3, 4]
```

### Example 2

```text
nums   = [5, 7, 7, 8, 8, 10]
target = 6
output = [-1, -1]
```

### Example 3

```text
nums   = []
target = 0
output = [-1, -1]
```

## Thought Process: From Scan to Two Boundaries

The naive idea is:

1. scan until you see `target`
2. keep moving until the target block ends

That works, but costs `O(n)`.

The array is sorted, so the target values form one continuous block if they exist.  
That means the answer can be described by two monotonic boundaries:

- first position with value `>= target`
- first position with value `> target`

So instead of trying to invent one tricky binary search, run two simple boundary searches.

## C - Concepts

### Method Category

- Binary search
- Boundary search
- Range query on sorted data

### Correctness Logic

Let:

- `l = lower_bound(target)`
- `r = upper_bound(target)`

Then:

- if `l == len(nums)` or `nums[l] != target`, the target does not exist
- otherwise the target occupies indices `[l, r - 1]`

### Stable Algorithm

1. compute `l = lower_bound(nums, target)`
2. if `l` is out of range or `nums[l] != target`, return `[-1, -1]`
3. compute `r = upper_bound(nums, target)`
4. return `[l, r - 1]`

### Why This Is Better Than “Find Any Equal First”

If you stop when `nums[mid] == target`, you still do not know:

- whether there is another target on the left
- whether there is another target on the right

Boundary search solves the actual problem directly.

## E - Engineering

### Scenario 1: Querying Timestamp Blocks (Python)

**Background**: logs are sorted by timestamp or event ID.  
**Why it fits**: all matching records form one contiguous interval.

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


times = [10, 10, 10, 13, 13, 20]
left = lower_bound(times, 10)
right = upper_bound(times, 10) - 1
print(left, right)
```

### Scenario 2: Sorted Order Buckets (Go)

**Background**: a service stores sorted group IDs and needs the full range of one ID.  
**Why it fits**: equal IDs appear in one block after sorting.

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
	nums := []int{5, 7, 7, 8, 8, 10}
	l := lowerBound(nums, 8)
	r := upperBound(nums, 8) - 1
	fmt.Println(l, r)
}
```

### Scenario 3: Highlighting Duplicate Segments in a UI (JavaScript)

**Background**: a frontend receives a sorted list and wants to highlight all matching values.  
**Why it fits**: the UI needs a start index and an end index, not one arbitrary match.

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

const nums = [5, 7, 7, 8, 8, 10];
console.log([lowerBound(nums, 8), upperBound(nums, 8) - 1]); // [3, 4]
```

## R - Reflection

### Complexity

- Time: `O(log n)` because we run two binary searches
- Space: `O(1)`

### Alternatives

- **Linear scan**: easy, but `O(n)`
- **Find one target and expand outward**: still degrades to `O(n)` when many duplicates exist

### Common Mistakes

- Using `>=` in both helpers, which makes lower and upper bounds identical
- Forgetting to verify `nums[l] == target` before returning a range
- Returning `[l, r]` instead of `[l, r - 1]`

### Why This Method Is the Most Practical

The target block is defined by two boundaries, so two boundary searches are the most direct, readable, and reusable solution.  
This is exactly the form engineers use in sorted logs, metrics series, and key-index tables.

## S - Summary

- Search Range is a two-boundary problem, not a “find one match” problem.
- `lower_bound` gives the start and `upper_bound - 1` gives the end.
- Verifying `nums[l] == target` is what separates “found” from “not found.”
- This template generalizes to many sorted-data range queries in real systems.

## Further Reading

- LeetCode 35: Search Insert Position
- LeetCode 744: Find Smallest Letter Greater Than Target
- Boundary-search utilities such as `bisect_left`, `bisect_right`, `lower_bound`, and `upper_bound`

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


def search_range(nums: List[int], target: int) -> List[int]:
    left = lower_bound(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    right = upper_bound(nums, target) - 1
    return [left, right]


if __name__ == "__main__":
    print(search_range([5, 7, 7, 8, 8, 10], 8))
    print(search_range([5, 7, 7, 8, 8, 10], 6))
    print(search_range([], 0))
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

void searchRange(int *nums, int n, int target, int ans[2]) {
    int left = lowerBound(nums, n, target);
    if (left == n || nums[left] != target) {
        ans[0] = -1;
        ans[1] = -1;
        return;
    }
    ans[0] = left;
    ans[1] = upperBound(nums, n, target) - 1;
}

int main(void) {
    int nums[] = {5, 7, 7, 8, 8, 10};
    int ans[2];
    searchRange(nums, 6, 8, ans);
    printf("[%d, %d]\n", ans[0], ans[1]);
    searchRange(nums, 6, 6, ans);
    printf("[%d, %d]\n", ans[0], ans[1]);
    return 0;
}
```

### C++

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

vector<int> searchRange(const vector<int>& nums, int target) {
    auto left = lower_bound(nums.begin(), nums.end(), target);
    if (left == nums.end() || *left != target) {
        return {-1, -1};
    }
    auto right = upper_bound(nums.begin(), nums.end(), target);
    return {(int)(left - nums.begin()), (int)(right - nums.begin() - 1)};
}

int main() {
    vector<int> nums{5, 7, 7, 8, 8, 10};
    auto a = searchRange(nums, 8);
    auto b = searchRange(nums, 6);
    cout << "[" << a[0] << ", " << a[1] << "]\n";
    cout << "[" << b[0] << ", " << b[1] << "]\n";
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

func searchRange(nums []int, target int) []int {
	left := lowerBound(nums, target)
	if left == len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	return []int{left, upperBound(nums, target) - 1}
}

func main() {
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 8))
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 6))
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

fn search_range(nums: &[i32], target: i32) -> [i32; 2] {
    let left = lower_bound(nums, target);
    if left == nums.len() || nums[left] != target {
        return [-1, -1];
    }
    [left as i32, (upper_bound(nums, target) - 1) as i32]
}

fn main() {
    let nums = vec![5, 7, 7, 8, 8, 10];
    println!("{:?}", search_range(&nums, 8));
    println!("{:?}", search_range(&nums, 6));
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

function searchRange(nums, target) {
  const left = lowerBound(nums, target);
  if (left === nums.length || nums[left] !== target) {
    return [-1, -1];
  }
  return [left, upperBound(nums, target) - 1];
}

console.log(searchRange([5, 7, 7, 8, 8, 10], 8));
console.log(searchRange([5, 7, 7, 8, 8, 10], 6));
```
