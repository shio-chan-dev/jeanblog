---
title: "LeetCode 452: Minimum Number of Arrows From Shared Intersections to Interval Greedy"
date: 2026-07-06T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "greedy", "intervals", "sorting", "LeetCode 452"]
description: "Solve LeetCode 452 in Python by deriving arrow coverage from shared interval intersections, then compressing the baseline into an earliest-ending arrow_pos greedy scan."
keywords: ["LeetCode 452", "Minimum Number of Arrows to Burst Balloons", "greedy", "interval greedy", "sorting", "Python"]
---

> **Subtitle / Summary**
> A single arrow does not belong to one balloon. It belongs to a group of balloon intervals that share at least one x-position.

- **Reading time**: 9-11 min
- **Tags**: `Hot100`, `greedy`, `intervals`, `sorting`
- **SEO keywords**: LeetCode 452, Minimum Number of Arrows to Burst Balloons, greedy, interval greedy
- **Meta description**: A pressure-first Python guide to LeetCode 452 that derives shared intersections, an intersection-scanning baseline, and the final arrow position greedy.

---

## Problem Requirement

You are given an array `points`.

Each balloon is a horizontal interval:

```text
[start, end]
```

If an arrow is shot at x-coordinate `x`, and:

```text
start <= x <= end
```

then that arrow bursts the balloon.

An arrow keeps traveling upward, so every balloon covering the same `x` is burst by that arrow.

Return:

```text
the minimum number of arrows needed to burst all balloons
```

### Input and Output

- Input: `points: List[List[int]]`
- Output: `int`
- Each balloon is an interval `[start, end]`.
- The arrow position `x` may be exactly on an endpoint.
- You only need to return the minimum number of arrows, not the arrow positions.

### Examples

```text
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
```

One valid shooting plan is:

```text
x = 6  bursts [2,8] and [1,6]
x = 11 bursts [10,16] and [7,12]
```

Two boundary examples:

```text
Input: points = [[1,2],[3,4],[5,6],[7,8]]
Output: 4
```

The balloons are disjoint, so each balloon needs its own arrow.

```text
Input: points = [[1,2],[2,3],[3,4],[4,5]]
Output: 2
```

Because arrows may land on endpoints:

```text
x = 2 bursts [1,2] and [2,3]
x = 4 bursts [3,4] and [4,5]
```

### Constraints

- `1 <= points.length <= 10^5`
- `points[i].length == 2`
- `-2^31 <= start < end <= 2^31 - 1`

## Step 1: What Can One Arrow Cover?

Start with a tiny question:

```text
[1,2] and [2,3]
```

Can one arrow burst both balloons?

Yes.

The arrow can be shot at:

```text
x = 2
```

That satisfies both:

```text
1 <= 2 <= 2
2 <= 2 <= 3
```

The current baseline is:

```text
One balloon needs one arrow.
```

This baseline breaks because:

> Example 1 has four balloons, but the answer is two. One arrow is not necessarily tied to one balloon.

The real question is:

```text
Which balloons share at least one x-position?
```

If a group of balloon intervals has a non-empty common intersection, one arrow can burst that whole group.

For example:

```text
[1,6] and [2,8]
```

Their common intersection is:

```text
[2,6]
```

Any `x` in that range bursts both balloons.

Now this version can:

- Explain that one arrow corresponds to a group of intervals with a common intersection.
- Explain why endpoint touching can still share one arrow.
- Explain why the answer can be smaller than the number of balloons.

It still lacks:

- A repeatable method for grouping all balloons.

## Step 2: First Write the Shared-Intersection Baseline

The current baseline is:

```text
One arrow can cover a group of balloons if their intervals have a common intersection.
```

This breaks because:

> We still cannot compute the minimum number of arrows for arbitrary `points`.

Start with a correct version that keeps a little more state.

Sort balloons by start:

```python
points.sort(key=lambda p: p[0])
```

During the scan, maintain the current group's common shooting range:

```text
[left, right]
```

For a new balloon `[start, end]`:

- If `start <= right`, it still intersects the current group.
- The common shooting range shrinks to `[max(left, start), min(right, end)]`.
- If `start > right`, the current group cannot cover this balloon, so we need a new arrow.

Code:

```python
from typing import List


def arrows_by_intersection(points: List[List[int]]) -> int:
    points.sort(key=lambda p: p[0])

    arrows = 1
    left, right = points[0]

    for start, end in points[1:]:
        if start <= right:
            left = max(left, start)
            right = min(right, end)
        else:
            arrows += 1
            left, right = start, end

    return arrows
```

Check the main example:

```text
points = [[10,16],[2,8],[1,6],[7,12]]
```

After sorting by start:

```text
[[1,6],[2,8],[7,12],[10,16]]
```

Scan:

```text
current group: [1,6]
read [2,8], still intersects, common range becomes [2,6]
read [7,12], 7 > 6, so start a new arrow group [7,12]
read [10,16], still intersects, common range becomes [10,12]
```

The answer is `2`.

Now this version can:

- Correctly count how many common-intersection groups exist.
- Explain what each arrow group covers.
- Pass the core example.

It still lacks:

- The state keeps both `left` and `right`, but deciding whether the next balloon needs a new arrow only depends on the right boundary.

## Step 3: Compress the Common Intersection Into `arrow_pos`

The current baseline is:

```text
Maintain the current group's common shooting range [left, right].
```

This breaks because:

> The baseline is correct, but it carries more state than the final greedy needs. To decide whether the next balloon can be covered, we only need the latest safe position for the current arrow.

Inside a group, the most restrictive balloon is the one that ends earliest.

If the earliest ending position in the current group is:

```text
right
```

then the current arrow cannot be placed to the right of `right`.

So place the arrow at that earliest end:

```text
arrow_pos = right
```

This is the same pressure as LeetCode 435's "keep the earlier-ending interval":

```text
Freeze the right boundary as early as needed, so the current required interval is not missed.
```

Now sort by end.

For each balloon `[start, end]`:

- If `start <= arrow_pos`, the current arrow bursts it.
- If `start > arrow_pos`, the current arrow cannot reach it, so we need a new arrow at `end`.

The new-arrow condition is:

```text
start > arrow_pos
```

not:

```text
start >= arrow_pos
```

because arrows can land on endpoints.

Use the boundary example:

```text
points = [[1,2],[2,3],[3,4],[4,5]]
```

If the first arrow is at:

```text
arrow_pos = 2
```

then `[2,3]` has `start = 2`, so it is still burst by that arrow.

Thus `[1,2]` and `[2,3]` share one arrow.

Now this version can:

- Compress the current common intersection's right boundary into `arrow_pos`.
- Explain why sorting by end is the natural final order.
- Decide exactly when a new arrow is required.

It still lacks:

- The complete LeetCode function and runnable checks.

## Step 4: Count Arrows With `arrow_pos`

The current baseline is:

```text
Sort by end and place the current arrow at the current group's earliest ending point.
```

This breaks because:

> The rule has not yet been assembled into LeetCode's `findMinArrowShots`.

Complete code:

```python
from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])

        arrows = 1
        arrow_pos = points[0][1]

        for start, end in points[1:]:
            if start > arrow_pos:
                arrows += 1
                arrow_pos = end

        return arrows
```

Trace the main example:

```text
points = [[10,16],[2,8],[1,6],[7,12]]
```

After sorting by end:

```text
[[1,6],[2,8],[7,12],[10,16]]
```

Scan:

```text
place the first arrow at 6
[2,8] has start = 2 <= 6, so it is covered
[7,12] has start = 7 > 6, so open a second arrow at 12
[10,16] has start = 10 <= 12, so it is covered
```

The final answer is `2`.

Now this version can:

- Count arrows with the minimum necessary state.
- Handle endpoint touching correctly.
- Satisfy the LeetCode method signature.

## Correctness Intuition

The current arrow must cover all balloons in the current group.

If one balloon in the group ends earliest at `arrow_pos`, then the arrow cannot be placed to the right of `arrow_pos`.

Putting the arrow exactly at `arrow_pos` is safe and flexible:

```text
It does not miss the current earliest-ending balloon,
and it gives later balloons as much chance as possible to still include the arrow.
```

After sorting by end, each arrow position is decided by the earliest-ending balloon not yet covered.

If a later balloon starts at or before `arrow_pos`, it is covered by the current arrow.

If a later balloon starts after `arrow_pos`, no position can cover both that balloon and the current group, so a new arrow is necessary.

## Complexity

Sorting costs:

```text
O(n log n)
```

The scan is linear:

```text
O(n)
```

Total time complexity:

```text
O(n log n)
```

The extra state is only `arrows` and `arrow_pos`:

```text
O(1)
```

## Common Mistakes

### 1. Treating `start == arrow_pos` as a New Arrow

The coverage rule is:

```text
start <= x <= end
```

So if:

```text
start == arrow_pos
```

the current arrow still bursts that balloon.

The new-arrow condition must be:

```python
start > arrow_pos
```

### 2. Sorting by Start and Using the Final Greedy Rule Directly

Sorting by start is useful for the shared-intersection baseline.

The final greedy version is cleaner with sorting by end, because `arrow_pos` comes from the current earliest-ending balloon.

If the sorting key and state meaning do not match, the code may pass some examples while remaining hard to prove.

### 3. Adding an Empty-Array Branch

The constraints say:

```text
1 <= points.length
```

So the LeetCode version can initialize directly:

```python
arrows = 1
arrow_pos = points[0][1]
```

No empty-array branch is needed.

## Runnable Checks

```python
from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])

        arrows = 1
        arrow_pos = points[0][1]

        for start, end in points[1:]:
            if start > arrow_pos:
                arrows += 1
                arrow_pos = end

        return arrows


def check() -> None:
    s = Solution()
    assert s.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert s.findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert s.findMinArrowShots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2
    assert s.findMinArrowShots([[1, 2]]) == 1


check()
```

## Summary

The greedy point in LeetCode 452 is:

```text
Place each arrow at the earliest ending point of the current uncovered group.
```

The scan invariant is:

> The current arrow is placed at `arrow_pos`; it covers all balloons already assigned to this group. If the next balloon has `start > arrow_pos`, the current group is closed and a new arrow is required.

Relation to LeetCode 435:

- 435: keep as many non-overlapping intervals as possible.
- 452: cover all intervals with as few points as possible.

Both train the same interval-greedy instinct:

```text
sort by end, and handle the earliest ending constraint first
```
