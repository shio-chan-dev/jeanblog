---
title: "LeetCode 2089: Find Target Indices After Sorting Array ACERS Guide"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "sorting", "lower bound", "upper bound", "LeetCode 2089", "ACERS"]
description: "Sort the array, locate the target block with lower and upper bounds, and return every target index, with tradeoff analysis, engineering mappings, and runnable implementations in six languages."
keywords: ["Find Target Indices After Sorting Array", "binary search", "sorting", "lower bound", "upper bound", "LeetCode 2089"]
---

> **Subtitle / Summary**  
> This problem is a useful bridge between sorting and binary search. After sorting the array, all copies of `target` become one contiguous block, and the answer is simply every index inside that block.

- **Reading time**: 10-12 min  
- **Tags**: `sorting`, `binary search`, `range location`  
- **SEO keywords**: Find Target Indices After Sorting Array, LeetCode 2089, lower bound, upper bound  
- **Meta description**: Sort the array, use lower and upper bounds to find the target block, and return every matching index, with tradeoffs, engineering scenarios, and runnable implementations in six languages.

## Target Readers

- Learners connecting sorting with lower/upper bound search
- Engineers who need all positions of one value after offline sorting
- Interview candidates reviewing how contiguous blocks form in sorted data

## Background / Motivation

The input array is not sorted, so we cannot apply binary search immediately.  
But once we sort it, every copy of the same value becomes one continuous segment.

That gives a clean workflow:

1. sort the data
2. find where the target block starts
3. find where the target block ends
4. output all indices in that block

This is the same “range of equal values” idea used in many sorted-data systems:

- leaderboard grouping
- equal-score buckets
- value-based offline analytics

## Core Concepts

- **Sorted target block**: equal values are contiguous after sorting
- **Lower bound**: first index `i` such that `nums[i] >= target`
- **Upper bound**: first index `i` such that `nums[i] > target`
- **Answer range**: all indices in `[lower_bound(target), upper_bound(target))`

## A - Algorithm

### Problem Restatement

Given an integer array `nums` and an integer `target`:

1. sort `nums` in non-decreasing order
2. return all indices where the sorted array equals `target`

If `target` does not exist, return an empty array.

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `nums` | `int[]` | unsorted integer array |
| `target` | `int` | value to locate after sorting |
| return | `int[]` | all indices of `target` in the sorted array |

### Example 1

```text
nums   = [1, 2, 5, 2, 3]
target = 2
sorted = [1, 2, 2, 3, 5]
output = [1, 2]
```

### Example 2

```text
nums   = [1, 2, 5, 2, 3]
target = 3
sorted = [1, 2, 2, 3, 5]
output = [3]
```

### Example 3

```text
nums   = [1, 2, 5, 2, 3]
target = 5
sorted = [1, 2, 2, 3, 5]
output = [4]
```

## Thought Process: From Sort-and-Scan to Sort-and-Bounds

The most direct solution is:

1. sort the array
2. scan the sorted result
3. collect every index whose value equals `target`

That is valid and easy to understand.

If you want to train the binary-search pattern, there is a cleaner post-sort observation:

- after sorting, all targets are adjacent
- so the answer is one contiguous interval

That means we can:

- find the first target with lower bound
- find the first value greater than target with upper bound
- generate the index list from that interval

## C - Concepts

### Method Category

- Sorting + binary search
- Range discovery in sorted data
- Boundary search

### Why the Target Indices Form a Continuous Range

After sorting:

- all values smaller than `target` appear first
- then all copies of `target`
- then all values greater than `target`

So if:

- `l = lower_bound(target)`
- `r = upper_bound(target)`

then the target indices are exactly:

```text
[l, l+1, ..., r-1]
```

### Stable Algorithm

1. sort `nums`
2. compute `l = lower_bound(nums, target)`
3. compute `r = upper_bound(nums, target)`
4. if `l == r`, return `[]`
5. otherwise return all integers from `l` to `r - 1`

## E - Engineering

### Scenario 1: Equal-Score Buckets After Offline Sort (Python)

**Background**: scores are collected unsorted, then sorted for reporting.  
**Why it fits**: equal scores become one contiguous block.

```python
from bisect import bisect_left, bisect_right

scores = sorted([1, 2, 5, 2, 3])
l = bisect_left(scores, 2)
r = bisect_right(scores, 2)
print(list(range(l, r)))
```

### Scenario 2: Batch Value Grouping in Services (Go)

**Background**: a batch job sorts values before producing grouped summaries.  
**Why it fits**: the exact index block of one value is found by two boundaries.

```go
package main

import (
	"fmt"
	"sort"
)

func main() {
	nums := []int{1, 2, 5, 2, 3}
	sort.Ints(nums)
	l := sort.Search(len(nums), func(i int) bool { return nums[i] >= 2 })
	r := sort.Search(len(nums), func(i int) bool { return nums[i] > 2 })
	var ans []int
	for i := l; i < r; i++ {
		ans = append(ans, i)
	}
	fmt.Println(ans)
}
```

### Scenario 3: Frontend Highlighting of Equal Rankings (JavaScript)

**Background**: a UI sorts scores and highlights every position tied with one value.  
**Why it fits**: ties become one adjacent segment after sorting.

```js
const nums = [1, 2, 5, 2, 3].slice().sort((a, b) => a - b);
const ans = [];
for (let i = 0; i < nums.length; i++) {
  if (nums[i] === 2) ans.push(i);
}
console.log(nums, ans); // [1,2,2,3,5] [1,2]
```

## R - Reflection

### Complexity

For the sort + bounds solution:

- Time: `O(n log n)` because sorting dominates
- Space:
  - `O(1)` extra if the language sort is in-place and we ignore implementation details
  - otherwise depends on the sorting implementation

### Alternative: Direct Counting

There is an important alternative:

- count how many elements are `< target`
- count how many elements are `== target`
- build the answer range directly

That approach is `O(n)` and is asymptotically better for this standalone problem.

### Why Keep the Sort + Bounds Version in a Binary-Search Series

Even though counting can be faster here, sort + lower/upper bound is still valuable because it teaches a reusable pattern:

- equal values form one block after sorting
- lower and upper bounds recover that block

That same reasoning appears in many other problems and real systems.

### Common Mistakes

- Sorting the array but still scanning the whole result after already knowing the boundaries
- Forgetting that `upper_bound` is exclusive
- Claiming this is strictly optimal without mentioning the linear counting alternative

## S - Summary

- After sorting, every copy of `target` becomes one contiguous segment.
- Lower and upper bounds recover the exact index interval of that segment.
- Returning `range(l, r)` is enough once the two boundaries are known.
- For this specific problem, direct counting is a valid faster alternative, but sort + bounds is the better teaching pattern for a binary-search series.

## Further Reading

- LeetCode 34: Find First and Last Position of Element in Sorted Array
- LeetCode 35: Search Insert Position
- Any standard documentation for `bisect_left`, `bisect_right`, `lower_bound`, and `upper_bound`

## Multi-language Implementations

### Python

```python
from bisect import bisect_left, bisect_right
from typing import List


def target_indices(nums: List[int], target: int) -> List[int]:
    nums = sorted(nums)
    l = bisect_left(nums, target)
    r = bisect_right(nums, target)
    return list(range(l, r))


if __name__ == "__main__":
    print(target_indices([1, 2, 5, 2, 3], 2))
    print(target_indices([1, 2, 5, 2, 3], 3))
    print(target_indices([1, 2, 5, 2, 3], 5))
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) {
    return (*(int *)a) - (*(int *)b);
}

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

int main(void) {
    int nums[] = {1, 2, 5, 2, 3};
    int n = sizeof(nums) / sizeof(nums[0]);
    qsort(nums, n, sizeof(int), cmp);
    int l = lowerBound(nums, n, 2);
    int r = upperBound(nums, n, 2);
    for (int i = l; i < r; i++) {
        printf("%d ", i);
    }
    printf("\n");
    return 0;
}
```

### C++

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

vector<int> targetIndices(vector<int> nums, int target) {
    sort(nums.begin(), nums.end());
    auto l = lower_bound(nums.begin(), nums.end(), target);
    auto r = upper_bound(nums.begin(), nums.end(), target);
    vector<int> ans;
    for (auto it = l; it != r; ++it) {
        ans.push_back((int)(it - nums.begin()));
    }
    return ans;
}

int main() {
    auto ans = targetIndices({1, 2, 5, 2, 3}, 2);
    for (int x : ans) cout << x << " ";
    cout << "\n";
    return 0;
}
```

### Go

```go
package main

import (
	"fmt"
	"sort"
)

func targetIndices(nums []int, target int) []int {
	sort.Ints(nums)
	l := sort.Search(len(nums), func(i int) bool { return nums[i] >= target })
	r := sort.Search(len(nums), func(i int) bool { return nums[i] > target })
	ans := make([]int, 0, r-l)
	for i := l; i < r; i++ {
		ans = append(ans, i)
	}
	return ans
}

func main() {
	fmt.Println(targetIndices([]int{1, 2, 5, 2, 3}, 2))
	fmt.Println(targetIndices([]int{1, 2, 5, 2, 3}, 3))
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

fn target_indices(mut nums: Vec<i32>, target: i32) -> Vec<usize> {
    nums.sort();
    let l = lower_bound(&nums, target);
    let r = upper_bound(&nums, target);
    (l..r).collect()
}

fn main() {
    println!("{:?}", target_indices(vec![1, 2, 5, 2, 3], 2));
    println!("{:?}", target_indices(vec![1, 2, 5, 2, 3], 3));
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

function targetIndices(nums, target) {
  nums = nums.slice().sort((a, b) => a - b);
  const l = lowerBound(nums, target);
  const r = upperBound(nums, target);
  const ans = [];
  for (let i = l; i < r; i++) ans.push(i);
  return ans;
}

console.log(targetIndices([1, 2, 5, 2, 3], 2));
console.log(targetIndices([1, 2, 5, 2, 3], 3));
console.log(targetIndices([1, 2, 5, 2, 3], 5));
```
