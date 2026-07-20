---
title: "LeetCode 42: Why Can Water Stay Above One Position?"
date: 2026-01-24T10:40:53+08:00
draft: true
categories: ["LeetCode"]
tags: ["Hot100", "array", "LeetCode 42"]
description: "Start with the water above one position, then build a runnable solution for LeetCode 42 step by step."
keywords: ["LeetCode 42", "Trapping Rain Water", "array", "Python"]
---

## Problem Requirement

You are given `n` non-negative integers in `height`. Each integer is the height of a bar with width `1`, and all bars are adjacent from left to right.

After rain, taller bars on both sides may hold water above shorter bars. Return the total amount of water trapped by the entire elevation map.

LeetCode expects this interface:

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ...
```

### Example 1

```text
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

### Example 2

```text
Input: height = [4,2,0,3,2,5]
Output: 9
```

### Constraints

```text
n == len(height)
1 <= n <= 2 * 10^4
0 <= height[i] <= 10^5
```

## Step 1: First Answer How Much Water One Position Holds

Do not calculate the whole elevation map yet. Focus on one position:

```text
height = [3,0,2]
            ^
           i = 1
```

The bar at index `1` has height `0`. There is a bar of height `3` on its left and a bar of height `2` on its right.

If we look only at the left side, it seems that the water could rise to height `3`. But the right wall has height `2`, so any water above `2` would spill over that side. The highest possible water level is therefore:

```text
min(highest bar on the left, highest bar on the right)
= min(3, 2)
= 2
```

The water above this position is:

```text
water level - current bar height
= 2 - 0
= 2
```

The current baseline is:

> Find walls on both sides of the current position, then determine how high the water can remain.

But "look at both walls" is not executable enough. If one side contains several bars, we need the highest boundary that side can provide. If we use only the taller side, water may still spill over the shorter side.

For one index `i`, add one executable rule:

1. Find `left_highest` in `0..i`.
2. Find `right_highest` in `i..n-1`.
3. Let the shorter boundary determine `water_level`.
4. Use `water_level - height[i]` as the water above this position.

Both ranges include `i`. This guarantees that neither highest value is lower than `height[i]`, so the result cannot become negative.

Write this local rule as the first runnable version:

```python
from typing import List


def trapped_at(height: List[int], i: int) -> int:
    left_highest = max(height[: i + 1])
    right_highest = max(height[i:])
    water_level = min(left_highest, right_highest)
    return water_level - height[i]
```

Check it against the valley above:

```python
assert trapped_at([3, 0, 2], 1) == 2
```

The two endpoints do not have complete boundaries on both sides, so neither traps water:

```python
assert trapped_at([3, 0, 2], 0) == 0
assert trapped_at([3, 0, 2], 2) == 0
```

Check another valley with equal-height boundaries:

```python
assert trapped_at([2, 1, 2], 1) == 1
```

Now this version can:

- calculate the trapped water above any one index
- explain why the shorter of the two highest boundaries determines the water level
- keep the result non-negative by including the current position in both ranges

It still lacks:

- the total trapped water for the entire elevation map
- reuse across positions that repeatedly scan the same left and right ranges
