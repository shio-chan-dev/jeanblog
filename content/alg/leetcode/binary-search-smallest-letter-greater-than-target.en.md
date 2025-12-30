---
title: "Smallest Letter Greater Than Target"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Smallest Letter Greater Than Target

## Summary
Given a sorted list of letters with wrap-around, return the smallest letter strictly greater than the target.

## Approach
Use `bisect_right` to find the first letter greater than target. If index reaches the end, wrap to index 0.

## Complexity
- Time: O(log n)
- Space: O(1)

## Python reference implementation
```python
from bisect import bisect_right

def next_greatest_letter(letters, target):
    i = bisect_right(letters, target)
    return letters[i] if i < len(letters) else letters[0]
```
