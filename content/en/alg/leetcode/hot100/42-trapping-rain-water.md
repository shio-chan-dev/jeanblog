---
title: "LeetCode 42: How Much Rain Water Can an Elevation Map Hold?"
date: 2026-01-24T10:40:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "array", "two pointers", "prefix max", "monotonic stack", "LeetCode 42"]
description: "Start with one position, derive boundary arrays and an O(1)-space two-pointer solution, then connect the same boundary model to a monotonic stack."
keywords: ["LeetCode 42", "Trapping Rain Water", "two pointers", "prefix max", "monotonic stack", "array", "Python"]
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

## Step 2: Add the Water From Every Position

The current version can answer:

```text
How much water is trapped above index i?
```

But the problem asks for the total across the entire elevation map. Calling `trapped_at` once misses every other position.

The current baseline is the local function from the previous step:

```python
def trapped_at(height: List[int], i: int) -> int:
    left_highest = max(height[: i + 1])
    right_highest = max(height[i:])
    water_level = min(left_highest, right_highest)
    return water_level - height[i]
```

Its local answer is correct. It only lacks a completion rule: process every index once and add each local contribution to `total`.

After the previous version, add:

```python
def trap_by_scanning(height: List[int]) -> int:
    total = 0

    for i in range(len(height)):
        total += trapped_at(height, i)

    return total
```

For `[3,0,2]`, the three contributions are:

| `i` | `trapped_at(height, i)` | Running `total` |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 2 | 2 |
| 2 | 0 | 2 |

Now check complete inputs:

```python
assert trap_by_scanning([3, 0, 2]) == 2
assert trap_by_scanning([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_by_scanning([4, 2, 0, 3, 2, 5]) == 9
assert trap_by_scanning([1]) == 0
```

Now this version can:

- calculate the total trapped water for the entire elevation map
- make every position contribute its local amount exactly once
- handle a single bar and inputs without a valley

It still lacks:

- reuse when different indices search for the same highest boundaries
- efficient execution at the largest input size

`trapped_at` examines O(n) elements for one position, and the outer loop processes n positions, so the worst-case time is O(n^2). Python slicing also creates O(n) temporary space. Replacing slices with explicit loops would remove that temporary allocation but would not remove the repeated O(n^2) scanning.

## Step 3: Store Boundaries That Were Already Found

Consider:

```text
height = [4,2,0,3,2,5]
```

To find the highest left boundary for index `1`, we inspect `[4,2]`. For index `2`, we inspect `[4,2,0]`, repeating all the work for `[4,2]`.

The right side has the same problem. The input may contain `2 * 10^4` positions, so the O(n^2) baseline repeatedly answers the same prefix and suffix questions.

The current baseline is:

```python
for i in range(len(height)):
    total += trapped_at(height, i)
```

Its local formula is correct. The break is that every call searches for the boundaries again.

Replace those searches with two reusable arrays:

- `left_highest[i]`: the highest bar in `0..i`
- `right_highest[i]`: the highest bar in `i..n-1`

Each new position only needs the result already stored beside it:

```text
left_highest[i] = max(left_highest[i - 1], height[i])
right_highest[i] = max(right_highest[i + 1], height[i])
```

These names now participate directly in updates and in the final water calculation:

```python
def trap_with_boundaries(height: List[int]) -> int:
    n = len(height)

    left_highest = [0] * n
    left_highest[0] = height[0]
    for i in range(1, n):
        left_highest[i] = max(left_highest[i - 1], height[i])

    right_highest = [0] * n
    right_highest[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_highest[i] = max(right_highest[i + 1], height[i])

    total = 0
    for i in range(n):
        water_level = min(left_highest[i], right_highest[i])
        total += water_level - height[i]

    return total
```

The two construction loops maintain these invariants:

> After writing `left_highest[i]`, it equals the maximum of `height[0..i]`.

> After writing `right_highest[i]`, it equals the maximum of `height[i..n-1]`.

The final loop therefore uses the same two real boundaries verified in Step 1, without searching for them again.

Check the middle version:

```python
assert trap_with_boundaries([3, 0, 2]) == 2
assert trap_with_boundaries([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_with_boundaries([4, 2, 0, 3, 2, 5]) == 9
assert trap_with_boundaries([5, 4, 3, 2, 1]) == 0
assert trap_with_boundaries([2, 2, 2]) == 0
```

Now this version can:

- calculate the total in O(n) time
- reuse every highest prefix and suffix boundary
- preserve the local water formula for every position

It still lacks:

- a way to avoid storing two arrays of length n
- O(1) extra space

The three linear loops take O(n) total time, and the two boundary arrays use O(n) extra space.

## Step 4: Settle Only the Side Whose Boundary Is Known

The boundary-array version has reduced the time to O(n), but it stores two values for every index:

```text
left_highest[0..n-1]
right_highest[0..n-1]
```

Once the final loop processes a position, it never uses that position again. The new pressure is:

> Can we keep only the two boundaries needed now instead of storing every boundary?

The current baseline depends on:

```text
water[i] = min(left_highest[i], right_highest[i]) - height[i]
```

Before removing the arrays, we must know which side already has a settled lower boundary.

Now introduce four pieces of state:

- `left` and `right`: the two ends of the unsettled interval
- `left_highest`: the highest bar seen from the beginning through `left`
- `right_highest`: the highest bar seen from `right` through the end

First update both highest values. If:

```text
left_highest <= right_highest
```

then the true highest value to the right of `left` is at least `right_highest`, so it is also at least `left_highest`. The lower boundary for `left` is therefore known to be `left_highest`, and its water is:

```text
left_highest - height[left]
```

Conversely, if `left_highest > right_highest`, the lower boundary for `right` is known to be `right_highest`, and its water is:

```text
right_highest - height[right]
```

Settle one side per iteration and move that pointer. Replace the boundary-array version with the final LeetCode implementation:

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        left_highest = 0
        right_highest = 0
        total = 0

        while left <= right:
            left_highest = max(left_highest, height[left])
            right_highest = max(right_highest, height[right])

            if left_highest <= right_highest:
                total += left_highest - height[left]
                left += 1
            else:
                total += right_highest - height[right]
                right -= 1

        return total
```

Use `[5,0,1,0,2]` to check the easy-to-miss right-side branch:

| `left` | `right` | `left_highest` | `right_highest` | Settled this round | Added water | `total` |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 4 | 5 | 2 | Right index 4 | 0 | 0 |
| 0 | 3 | 5 | 2 | Right index 3 | 2 | 2 |
| 0 | 2 | 5 | 2 | Right index 2 | 1 | 3 |
| 0 | 1 | 5 | 2 | Right index 1 | 2 | 5 |
| 0 | 0 | 5 | 5 | Left index 0 | 0 | 5 |

Final checks:

```python
solution = Solution()

assert solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert solution.trap([4, 2, 0, 3, 2, 5]) == 9
assert solution.trap([5, 0, 1, 0, 2]) == 5
assert solution.trap([3, 0, 2]) == 2
assert solution.trap([5, 4, 3, 2, 1]) == 0
assert solution.trap([2, 2, 2]) == 0
assert solution.trap([1]) == 0
```

The loop invariant is:

> At the start of each iteration, every index outside `left..right` has been settled exactly once.

After updating the two highest values:

```text
left_highest = max(height[0..left])
right_highest = max(height[right..n-1])
```

- If `left_highest <= right_highest`, the true right boundary cannot be lower than `left_highest`, so settling `left` is correct.
- Otherwise, the true left boundary cannot be lower than `right_highest`, so settling `right` is correct.
- Exactly one pointer moves each round, carrying the invariant into the next iteration.
- When `left > right`, every index has been settled once, so `total` is the complete answer.

This version now satisfies the problem requirements:

- Time: O(n). One pointer moves each round, and every index is settled once.
- Extra space: O(1). Only the pointers, highest boundaries, and total are stored.
- The input array is not modified.

## Two-Pointer Common Mistakes

### 1. Calculating Water Before Updating the Current Boundaries

The current bar is also a candidate for the highest boundary on its side. First execute:

```python
left_highest = max(left_highest, height[left])
right_highest = max(right_highest, height[right])
```

Only then is the local water calculation guaranteed to be non-negative.

### 2. Replacing the Comparison With the Current Bar Heights

Comparing `height[left]` and `height[right]` can support another correct implementation, but that version uses a different movement rule and proof. This tutorial settles a side from the maintained `left_highest` and `right_highest`; do not replace only the condition while keeping the original proof.

### 3. Moving Both Pointers in One Iteration

Only the side with the lower known boundary is safe to settle. Moving both pointers skips a position on the other side whose lower boundary may still be unknown.

### 4. Excluding the Current Position From a Boundary Range

If a highest value excludes the current bar, `water_level - height[i]` may become negative. All three versions include the current position in both highest values, so they do not need an extra `max(0, ...)`.

### 5. Changing `left <= right` to `left < right` Without Updating the Proof

Other loop boundaries can support correct implementations, but this tutorial's invariant says that every index is settled exactly once. `left <= right` explicitly processes the meeting position. Changing the loop condition also requires a different termination argument.

## Alternative: Settle Valleys With a Monotonic Stack

The two-pointer solution settles one position as soon as its lower boundary is
known. A monotonic stack uses a different unit of work:

> Keep valley bottoms unresolved until a higher right wall arrives, then settle
> the horizontal water layer closed by that wall.

This is not a further optimization of the two-pointer version. Both approaches
run in O(n) time. The stack version is useful because it connects this problem
to the same unresolved-candidate model used by Daily Temperatures and Largest
Rectangle in Histogram.

### Keep Unresolved Indices in Non-Increasing Height Order

Scan bars from left to right and store their indices in `stack`. The heights at
those indices remain non-increasing from bottom to top:

```text
height[stack[0]] >= height[stack[1]] >= ...
```

If the current bar is not taller than the stack top, it cannot close a valley
above that top. Push its index and continue.

When the current bar is taller than the stack top, the top can act as a valley
bottom whose right wall has just appeared:

```python
bottom = stack.pop()
```

After the pop:

- the current index `right` is the right wall
- the new stack top is the left wall
- `bottom` is the lower level between those walls

If the stack becomes empty, no left wall exists, so that bottom cannot trap
water.

Otherwise, calculate the newly closed horizontal layer:

```text
width = right - left - 1
bounded_height = min(height[left], height[right]) - height[bottom]
water = width * bounded_height
```

Continue popping while the current bar is taller than the new stack top. One
right wall may settle several layers of the same valley.

### Trace One Layer at a Time

Use:

```text
height = [4,2,0,3]
```

Before index `3`, the stack contains `[0,1,2]`, with heights `[4,2,0]`.
The current height is `3`.

First pop index `2`, whose height is `0`:

```text
left = 1
right = 3
width = 3 - 1 - 1 = 1
bounded_height = min(2, 3) - 0 = 2
added water = 1 * 2 = 2
```

The current bar is still taller than the new top at index `1`, so pop again:

```text
left = 0
right = 3
width = 3 - 0 - 1 = 2
bounded_height = min(4, 3) - 2 = 1
added water = 2 * 1 = 2
```

The two pops calculate different vertical layers, so they do not count the
same water twice. The total is `4`.

### Complete Monotonic-Stack Implementation

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        total = 0
        stack = []

        for right, right_height in enumerate(height):
            while stack and right_height > height[stack[-1]]:
                bottom = stack.pop()

                if not stack:
                    break

                left = stack[-1]
                width = right - left - 1
                bounded_height = min(height[left], right_height) - height[bottom]
                total += width * bounded_height

            stack.append(right)

        return total
```

### Why Each Pop Is Correct

Before processing `right`, every index in the stack is still missing a higher
right wall. Heights are non-increasing from bottom to top.

When `height[right] > height[bottom]`:

- `right` is the first processed position that can close water above
  `bottom`; otherwise `bottom` would have been popped earlier
- the new stack top is the nearest remaining left boundary
- both boundaries are higher than or equal to the level being added
- `right - left - 1` covers exactly the positions between the two walls

The pop calculates only the layer above `height[bottom]` and below the shorter
wall. If another pop follows, it starts from a higher bottom level, so the
layers remain disjoint.

After all lower tops are popped, the current height is less than or equal to
the stack-top height. Pushing `right` restores the non-increasing invariant.

### Checks

```python
solution = Solution()

assert solution.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert solution.trap([4, 2, 0, 3, 2, 5]) == 9
assert solution.trap([4, 2, 0, 3]) == 4
assert solution.trap([3, 0, 2]) == 2
assert solution.trap([5, 4, 3, 2, 1]) == 0
assert solution.trap([2, 2, 2]) == 0
assert solution.trap([1]) == 0
```

### Complexity

Every index is pushed once and popped at most once. All iterations of the
nested `while` therefore total O(n):

- Time: O(n).
- Extra space: O(n), for the index stack.

The two-pointer version remains preferable when O(1) extra space is the main
goal. The stack version is preferable when the learning goal is to recognize
unresolved candidates and settle a full interval when its future boundary
arrives.

### Connection to Other Monotonic-Stack Problems

The stack mechanism is shared, but each problem assigns a different meaning to
a pop:

| Problem | Stack order | What a pop settles |
| --- | --- | --- |
| 739 Daily Temperatures | Non-increasing temperatures | Waiting distance to the first warmer day |
| 503 Next Greater Element II | Non-increasing values | Next greater value in a circular array |
| 84 Largest Rectangle in Histogram | Non-decreasing heights | Maximal width for one limiting height |
| 42 Trapping Rain Water | Non-increasing heights | Water layer closed by left and right walls |

## Summary

The derivation is:

```text
calculate the water above one position
-> scan for every position to get an O(n^2) correct solution
-> store every highest prefix and suffix for O(n) time and O(n) space
-> prove that the side with the lower known boundary can be settled now
-> move one pointer per round for O(n) time and O(1) extra space
```

The alternative monotonic-stack branch is:

```text
keep unresolved valley indices in non-increasing height order
-> let a higher right wall pop one valley bottom
-> combine the new stack top and current index as two boundaries
-> calculate one horizontal water layer
-> push and pop every index at most once
```

The essential part of the two-pointer solution is explaining why the lower
known boundary settles one side. The essential part of the stack solution is
explaining why one pop has both boundaries needed to settle a disjoint water
layer. The two approaches share the same physical boundary model, but they
organize the computation differently.
