---
title: "Maximum Count of Positive and Negative Numbers"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Maximum Count of Positive and Negative Numbers

## Summary
Given a sorted array, return the maximum of the number of positive and negative values. Binary search gives the split points in O(log n).

## Approach
- Use `bisect_left(nums, 0)` to get count of negatives.
- Use `bisect_right(nums, 0)` to get first positive index.
- Positive count = `n - right_zero`.

## Complexity
- Time: O(log n)
- Space: O(1)

## Python reference implementation
```python
from bisect import bisect_left, bisect_right

def maximum_count(nums):
    n = len(nums)
    neg = bisect_left(nums, 0)
    pos = n - bisect_right(nums, 0)
    return max(neg, pos)
```
