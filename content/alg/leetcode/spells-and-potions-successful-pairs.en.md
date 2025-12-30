---
title: "Spells and Potions: Count Successful Pairs"
date: 2025-12-04T11:10:00+08:00
draft: false
categories: ["LeetCode"]
---

# Spells and Potions

## Summary
For each spell, count how many potions make `spell * potion >= success`. Sort potions and binary search the threshold.

## Approach
- Sort potions.
- For each spell, compute `need = ceil(success / spell)`.
- Use binary search to find the first potion >= need.

## Complexity
- Time: O(n log n)
- Space: O(1) extra (or O(n) if sorting a copy)

## Python reference implementation
```python
import bisect
import math

def successful_pairs(spells, potions, success):
    potions = sorted(potions)
    n = len(potions)
    res = []
    for s in spells:
        need = (success + s - 1) // s
        idx = bisect.bisect_left(potions, need)
        res.append(n - idx)
    return res
```
