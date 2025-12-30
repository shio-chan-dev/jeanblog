---
title: "Find First and Last Position of Element in Sorted Array"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Find First and Last Position of Element in Sorted Array

## Summary
Given a sorted array, return the first and last index of a target value, or `[-1, -1]` if not found. Use lower_bound and upper_bound.

## Approach
- `l = lower_bound(target)`
- `r = upper_bound(target) - 1`
- If `l` is out of range or `nums[l] != target`, return `[-1, -1]`.

## Complexity
- Time: O(log n)
- Space: O(1)

## Python reference implementation
```python
from bisect import bisect_left, bisect_right

def search_range(nums, target):
    l = bisect_left(nums, target)
    r = bisect_right(nums, target) - 1
    if l >= len(nums) or nums[l] != target:
        return [-1, -1]
    return [l, r]
```
