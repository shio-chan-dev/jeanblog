---
title: "LeetCode 84: Which Bar Limits a Contiguous Rectangle?"
date: 2026-07-20T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "array", "stack", "monotonic stack", "LeetCode 84"]
description: "Start from the shortest bar in a contiguous interval, build an interval baseline, then derive a linear monotonic-stack solution that settles maximal widths."
keywords: ["LeetCode 84", "Largest Rectangle in Histogram", "monotonic stack", "histogram", "Hot100", "Python"]
---

## Problem Requirement

You are given a non-negative integer array `heights`. Each `heights[i]` is the height of a bar with width `1`, and all bars are adjacent.

Return the area of the largest rectangle that can be formed in the histogram.

A legal rectangle covers a contiguous interval of bars. Its width is the number of bars in that interval, and its height cannot exceed the shortest bar in the interval.

LeetCode provides this method contract:

```text
largestRectangleArea(heights: List[int]) -> int
```

### Example 1

```text
Input: heights = [2,1,5,6,2,3]
Output: 10
```

Bars at indices `2` and `3` have heights `5` and `6`. They form a rectangle with height `5`, width `2`, and area `10`.

### Example 2

```text
Input: heights = [2,4]
Output: 4
```

Height `4` with width `1` and height `2` with width `2` both produce area `4`.

### Constraints

- `1 <= heights.length <= 10^5`
- `0 <= heights[i] <= 10^4`

## Step 1: Rectangle Area Is Not the Sum of Bar Heights

Start with:

```text
heights = [2,1,2]
```

If one rectangle covers all three bars, its width is `3`, but its height can be at most `1`:

```text
height = 1
width = 3
area = 1 * 3 = 3
```

Adding the bar heights to get `5` is incorrect. A histogram rectangle must fill one complete rectangular region, so the middle height `1` limits the entire interval.

The current baseline is:

```text
Choose a contiguous interval and use its shortest bar as the rectangle height.
```

For any interval `[left, right]`:

```text
width = right - left + 1
height = the shortest bar in the interval
area = height * width
```

This baseline breaks because:

> We can evaluate one chosen interval, but we do not yet have an executable process for comparing every contiguous interval.

Now this version can:

- distinguish rectangle area from the sum of bar heights
- preserve the contiguous-interval requirement
- use the shortest bar as the limiting height
- explain area `3` for `[2,1,2]` and area `10` in the standard example

It still lacks:

- a correct algorithm that enumerates and compares all legal rectangles

## Step 2: Enumerate Intervals for a Correct Baseline

The current baseline knows how to evaluate an interval:

```text
shortest interval height * interval width
```

Turn "compare every interval" into runnable code.

After fixing `left`, extend `right` one bar at a time. Each extension only needs one update to the current shortest height:

```python
min_height = min(min_height, heights[right])
```

First correct implementation:

```python
from typing import List


def largest_rectangle_quadratic(heights: List[int]) -> int:
    n = len(heights)
    best = 0

    for left in range(n):
        min_height = heights[left]

        for right in range(left, n):
            min_height = min(min_height, heights[right])
            width = right - left + 1
            best = max(best, min_height * width)

    return best
```

Trace `left = 0` for `[2,1,2]`:

| `right` | Interval | `min_height` | Width | Area | Current best |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `[2]` | 2 | 1 | 2 | 2 |
| 1 | `[2,1]` | 1 | 2 | 2 | 2 |
| 2 | `[2,1,2]` | 1 | 3 | 3 | 3 |

Checks:

```python
assert largest_rectangle_quadratic([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle_quadratic([2, 4]) == 4
assert largest_rectangle_quadratic([2, 1, 2]) == 3
assert largest_rectangle_quadratic([1]) == 1
assert largest_rectangle_quadratic([0]) == 0
assert largest_rectangle_quadratic([1, 2, 3, 4]) == 6
```

Now this version can:

- enumerate every contiguous interval
- maintain its shortest height incrementally
- compare all legal rectangle areas correctly
- handle one bar and zero height

It still lacks:

- reuse across overlapping intervals; the same relative-height relationships are recomputed many times

There are O(n) left boundaries and up to O(n) right extensions for each one, so time is O(n^2). Extra space is O(1).

## Step 3: Let Each Bar Be the Limiting Height

The current baseline works in this order:

```text
choose an interval
-> find its shortest bar
-> calculate the area
```

The break is:

> One bar may be the limiting height for many overlapping intervals, but the baseline rediscovers that fact repeatedly.

Reverse the viewpoint. Fix index `i`, assume `heights[i]` is the limiting height, and ask how far that height can extend.

It extends until the first strictly shorter bar on each side:

- `left`: first index left of `i` where `heights[left] < heights[i]`
- `right`: first index right of `i` where `heights[right] < heights[i]`

The covered interval is:

```text
left + 1 through right - 1
```

Therefore:

```text
width = right - left - 1
area = heights[i] * width
```

Use the formula immediately on:

```text
heights = [2,1,5,6,2,3]
```

For index `i = 2`, height `5`:

- The first strictly shorter bar on the left is height `1` at index `1`.
- The first strictly shorter bar on the right is height `2` at index `4`.
- The valid interval is indices `2..3`.

```text
left = 1
right = 4
width = 4 - 1 - 1 = 2
area = 5 * 2 = 10
```

If no shorter bar exists on one side, use a virtual boundary outside the histogram:

```text
no shorter bar on the left: left = -1
no shorter bar on the right: right = n
```

For height `2` in `[2,4]`, the boundaries are `-1` and `2`:

```text
width = 2 - (-1) - 1 = 2
area = 2 * 2 = 4
```

The boundaries are strictly shorter. Equal-height bars do not stop expansion and may belong to the same rectangle.

Now this version can:

- replace interval enumeration with one candidate per limiting bar
- calculate maximal width from first-shorter boundaries
- handle virtual boundaries at the histogram edges
- produce area `10` in the standard example

It still lacks:

- efficient boundary discovery; scanning both directions for every bar is still O(n^2)
- shared state for unresolved boundaries

## Step 4: Settle the Stack Top When a Shorter Bar Arrives

The current baseline knows which boundaries every bar needs, but finding them independently still repeats work.

The scan itself provides the next pressure:

> When current index `i` is shorter than an earlier bar, `i` is that taller bar's first strictly shorter right boundary. Its area can now be settled.

Add an index stack named `stack`. Heights at its indices remain non-decreasing from bottom to top:

```text
heights[stack[0]] <= heights[stack[1]] <= ...
```

When `current_height` is lower than the stack top, keep popping taller bars:

```python
while stack and heights[stack[-1]] > current_height:
    height = heights[stack.pop()]
    left = stack[-1] if stack else -1
    width = i - left - 1
    best = max(best, height * width)
```

After a pop:

- Current `i` is the popped bar's first strictly shorter right boundary.
- The new stack top is the left position this candidate cannot cross.
- An empty stack means the rectangle can extend to the beginning, represented by `-1`.

Connect the mechanism to the real-bar scan:

```python
best = 0
stack = []

for i, current_height in enumerate(heights):
    while stack and heights[stack[-1]] > current_height:
        height = heights[stack.pop()]
        left = stack[-1] if stack else -1
        width = i - left - 1
        best = max(best, height * width)

    stack.append(i)
```

Trace `[2,1,5,6,2]`:

| `i` | Current height | Operation | `left` | Width | Area | Stack after |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 0 | 2 | Push 0 | - | - | - | `[0]` |
| 1 | 1 | Pop height 2; push 1 | -1 | 1 | 2 | `[1]` |
| 2 | 5 | Push 2 | - | - | - | `[1,2]` |
| 3 | 6 | Push 3 | - | - | - | `[1,2,3]` |
| 4 | 2 | Pop height 6 | 2 | 1 | 6 | `[1,2]` |
| 4 | 2 | Pop height 5 | 1 | 2 | 10 | `[1]` |
| 4 | 2 | Push 4 | - | - | - | `[1,4]` |

For height `5`, index `4` is the first shorter right boundary and index `1` blocks the left side:

```text
width = 4 - 1 - 1 = 2
area = 5 * 2 = 10
```

### Why Equal Heights May Stay in the Stack

The pop condition is strict `>`, so equal heights remain together. The new stack top after a pop may therefore be equal to the popped height rather than strictly shorter.

This does not lose the maximum area. A later equal-height bar may first settle with a narrower width, while an earlier equal-height bar remains in the stack and later covers the combined wider interval. At least one representative of that height receives the full useful width.

The accurate invariant is therefore non-decreasing heights, not strictly increasing heights.

Now this version can:

- settle taller bars when their first shorter right boundary arrives
- derive usable width from the stack after each pop
- share boundary work in one scan
- preserve equal heights without losing their widest candidate

It still lacks:

- settlement for bars left in the stack when an increasing suffix reaches the end
- the final wrapper, invariant, and complexity proof

## Step 5: Use a Trailing Zero to Settle the Final Bars

Consider a strictly increasing input:

```text
heights = [1,2,3,4]
```

The current baseline only pops when a shorter bar arrives. At the end, all four indices remain unresolved.

The break is:

> Remaining bars need a shorter right boundary at virtual index `n`.

Append a virtual zero to a copy:

```python
bars = heights + [0]
```

This does not mutate the caller's input. The virtual zero is not a real answer bar; it only triggers every final pop.

Complete LeetCode implementation:

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        bars = heights + [0]
        best = 0
        stack = []

        for i, current_height in enumerate(bars):
            while stack and bars[stack[-1]] > current_height:
                height = bars[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                best = max(best, height * width)

            stack.append(i)

        return best
```

The loop invariant is:

> Before each iteration, `stack` stores indices whose first strictly shorter right boundary has not appeared, in increasing index order and non-decreasing height order.

When a lower current height arrives:

- Current `i` is the popped bar's first strictly shorter right boundary.
- The new stack top is the left blocking position for this candidate.
- `i - left - 1` is the usable width.

After all taller bars are popped, the top height is less than or equal to the current height, so pushing `i` preserves the invariant.

The final virtual zero settles every remaining positive-height bar. Real zero-height bars also work correctly under the strict comparison.

### Checks

```python
solution = Solution()

assert solution.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
assert solution.largestRectangleArea([2, 4]) == 4
assert solution.largestRectangleArea([2, 1, 2]) == 3
assert solution.largestRectangleArea([1]) == 1
assert solution.largestRectangleArea([0]) == 0
assert solution.largestRectangleArea([1, 2, 3, 4]) == 6
assert solution.largestRectangleArea([4, 3, 2, 1]) == 6
assert solution.largestRectangleArea([2, 2, 2]) == 6
```

### Complexity

Every index is:

- pushed once
- popped at most once

All inner-loop iterations therefore total O(n):

- Time: O(n).
- Extra space: O(n), for the copied `bars` array and index stack.

## Common Mistakes

### 1. Forgetting bars left in the stack

A strictly increasing input never triggers a real pop. Use an explicit cleanup loop or a trailing zero.

### 2. Mutating the input with `heights.append(0)`

Use:

```python
bars = heights + [0]
```

This preserves the caller's array.

### 3. Using the wrong width

Both `left` and current `i` are blocking positions outside the rectangle:

```text
width = i - left - 1
```

### 4. Storing heights instead of indices

Width calculation requires boundary indices.

### 5. Replacing `>` with `>=` without changing the invariant

An `>=` solution can also be correct, but it uses a different equal-height policy. This tutorial keeps equal heights and a non-decreasing stack.

### 6. Declaring O(n^2) from the nested loops

Count total pushes and pops. One index cannot be popped twice.

## Summary

The derivation is:

```text
define shortest height and width for a contiguous interval
-> enumerate all intervals with min_height
-> let each bar be the limiting height
-> use first-shorter boundaries to determine maximal width
-> settle taller stack tops when a shorter bar arrives
-> use a trailing zero to settle the remaining bars
-> push and pop every index at most once
```

LeetCode 84 goes beyond 739 and 503: a pop does not merely write one next-greater answer. It combines the current right boundary and the new stack top to derive a full rectangle width.
