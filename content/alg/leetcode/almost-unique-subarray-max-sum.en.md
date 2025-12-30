---
title: "Maximum Sum of Almost Unique Subarray"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Maximum Sum of Almost Unique Subarray

## Summary
Given an array, window size `k`, and threshold `m`, find the maximum sum of any length-`k` subarray that contains at least `m` distinct elements.

## Approach
Use a sliding window with a frequency map, track window sum and number of distinct values.

## Complexity
- Time: O(n)
- Space: O(n) for frequency map

## Python reference implementation
```python
def max_sum_almost_unique(nums, m, k):
    from collections import defaultdict
    count = defaultdict(int)
    distinct = 0
    window_sum = 0
    ans = 0

    for i, x in enumerate(nums):
        window_sum += x
        if count[x] == 0:
            distinct += 1
        count[x] += 1

        if i >= k:
            y = nums[i - k]
            window_sum -= y
            count[y] -= 1
            if count[y] == 0:
                distinct -= 1

        if i >= k - 1 and distinct >= m:
            ans = max(ans, window_sum)

    return ans
```
