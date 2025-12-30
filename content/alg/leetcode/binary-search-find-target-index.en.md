---
title: "Find Target Indices After Sorting"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Find Target Indices After Sorting

## Summary
Sort the array and return all indices where the target appears. Use binary search to find the range efficiently.

## Approach
- Sort the array.
- Use `bisect_left` and `bisect_right` to get `[l, r)`.
- Return `list(range(l, r))`.

## Complexity
- Time: O(n log n)
- Space: O(n) if sorting a copy

## Python reference implementation
```python
from bisect import bisect_left, bisect_right

def target_indices(nums, target):
    nums = sorted(nums)
    l = bisect_left(nums, target)
    r = bisect_right(nums, target)
    return list(range(l, r))
```
