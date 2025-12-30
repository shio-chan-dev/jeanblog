---
title: "Minimum Recolors to Get K Consecutive Black Blocks"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Minimum Recolors to Get K Consecutive Black Blocks

## Summary
Given a string of 'B' and 'W', find the minimum recolors to make a substring of length `k` all black.

## Approach
Use a sliding window of length `k` and count the number of whites in the window. The minimum whites across all windows is the answer.

## Complexity
- Time: O(n)
- Space: O(1)

## Python reference implementation
```python
def minimum_recolors(blocks, k):
    whites = sum(1 for c in blocks[:k] if c == 'W')
    ans = whites
    for i in range(k, len(blocks)):
        if blocks[i-k] == 'W':
            whites -= 1
        if blocks[i] == 'W':
            whites += 1
        ans = min(ans, whites)
    return ans
```
