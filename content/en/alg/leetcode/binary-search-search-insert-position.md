---
title: "Search Insert Position: Binary Search for the Insert Index"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Search Insert Position

## Summary
Find the index where a target should be inserted into a sorted array. If it exists, return its index; otherwise return the insertion position. This is a classic lower_bound binary search.

## Approach
Use binary search to find the first index `i` such that `nums[i] >= target`.

## Complexity
- Time: O(log n)
- Space: O(1)

## Python reference implementation
```python
from bisect import bisect_left

def search_insert(nums, target):
    return bisect_left(nums, target)
```
