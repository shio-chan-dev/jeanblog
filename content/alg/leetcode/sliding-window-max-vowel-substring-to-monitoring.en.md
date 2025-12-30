---
title: "High-Value Sliding Window Applications: From Max Vowels to Real Monitoring"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Sliding Window in Engineering: Max Vowels to Monitoring

## Summary
Sliding window is a simple but powerful pattern. The LeetCode problem "Maximum Number of Vowels in a Substring of Given Length" is a clean entry point and maps to real monitoring windows.

## Approach (max vowels)
Keep a window of length `k`, maintain a running count of vowels, and update the maximum.

## Complexity
- Time: O(n)
- Space: O(1)

## Python reference implementation
```python
def max_vowels(s, k):
    vowels = set("aeiou")
    cur = sum(1 for c in s[:k] if c in vowels)
    ans = cur
    for i in range(k, len(s)):
        if s[i-k] in vowels:
            cur -= 1
        if s[i] in vowels:
            cur += 1
        ans = max(ans, cur)
    return ans
```
