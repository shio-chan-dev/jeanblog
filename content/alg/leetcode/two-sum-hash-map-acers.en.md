---
title: "Two Sum with Hash Map (ACERS Summary)"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Two Sum

## Summary
Find two indices such that `nums[i] + nums[j] = target`. Use a hash map for O(n) time.

## Approach
Iterate and store `value -> index`. For each number `x`, check if `target - x` exists.

## Complexity
- Time: O(n)
- Space: O(n)

## Python reference implementation
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return [seen[y], i]
        seen[x] = i
```
