---
title: "LeetCode 435: Non-overlapping Intervals From Minimum Removals to Maximum Kept"
date: 2026-07-06T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "greedy", "intervals", "sorting", "LeetCode 435"]
description: "Solve LeetCode 435 in Python by turning minimum removals into maximum kept non-overlapping intervals, then deriving the earliest-ending interval greedy scan."
keywords: ["LeetCode 435", "Non-overlapping Intervals", "greedy", "interval greedy", "sorting", "Python"]
---

> **Subtitle / Summary**
> Non-overlapping Intervals is easier when you stop asking which interval to delete first. First ask how many intervals can remain.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `greedy`, `intervals`, `sorting`
- **SEO keywords**: LeetCode 435, Non-overlapping Intervals, greedy, interval greedy
- **Meta description**: A pressure-first Python guide to LeetCode 435 that derives minimum removals from maximum kept intervals and an earliest-ending greedy scan.

---

## Problem Requirement

You are given an array of intervals `intervals`.

Each interval is written as:

```text
[start, end]
```

The task is to remove as few intervals as possible so that the remaining intervals do not overlap.

Return:

```text
the minimum number of intervals to remove
```

### Input and Output

- Input: `intervals: List[List[int]]`
- Output: `int`
- Each interval satisfies `start < end`.
- If two intervals only touch at an endpoint, they are non-overlapping.

So:

```text
[1,2] and [2,3]
```

can both remain.

### Examples

```text
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
```

After removing `[1,3]`, the remaining intervals are:

```text
[[1,2],[2,3],[3,4]]
```

They do not overlap.

Two boundary examples:

```text
Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
```

Three identical intervals can keep at most one copy, so two must be removed.

```text
Input: intervals = [[1,2],[2,3]]
Output: 0
```

These two intervals only touch at endpoint `2`, so nothing must be removed.

### Constraints

- `1 <= intervals.length <= 10^5`
- `intervals[i].length == 2`
- `-5 * 10^4 <= start_i < end_i <= 5 * 10^4`

## Step 1: Do Not Ask Which One to Delete First

Start with:

```text
intervals = [[1,2],[2,3],[3,4],[1,3]]
```

The problem asks:

```text
How many intervals must be removed?
```

The current baseline is:

```text
Directly decide which interval to delete.
```

This baseline breaks because:

> When several intervals conflict, staring at "which one should I delete?" does not reveal the rule for making that choice.

Look at the problem from the other side.

If there are `n` intervals and we can keep at most `max_kept` non-overlapping intervals, then:

```text
minimum removals = n - max_kept
```

Or:

```text
min_removed = n - max_kept
```

In the example, we can keep:

```text
[[1,2],[2,3],[3,4]]
```

These three intervals do not overlap.

There are `4` intervals total, and at most `3` can remain, so the minimum number removed is:

```text
4 - 3 = 1
```

This step changes only the objective, not the answer:

```text
minimum removals
```

is equivalent to:

```text
maximum number of non-overlapping intervals kept
```

Now this version can:

- Avoid guessing which interval to delete first.
- Compute removals later from `n - max_kept`.
- Explain why `[[1,2],[2,3],[3,4],[1,3]]` returns `1`.

It still lacks:

- A rule for choosing which interval to keep when two intervals conflict.

## Step 2: When Intervals Conflict, Keep the Earlier-Ending One

The current baseline is:

```text
Maximize how many non-overlapping intervals remain.
```

This breaks because:

> We have transformed the objective, but we still do not know which interval to keep when two intervals overlap.

Use a smaller conflict:

```text
[1,2], [1,3], [2,3]
```

If we keep `[1,3]` first:

```text
kept: [1,3]
```

Then `[2,3]` overlaps with `[1,3]`, so it is hard to keep more intervals.

If we keep `[1,2]` first:

```text
kept: [1,2]
```

Because endpoint touching is allowed, `[2,3]` can still remain:

```text
[1,2], [2,3]
```

So when there is a conflict, prefer the interval with the earlier end:

```text
keep the interval that ends earlier
```

The reason is not that it merely looks shorter. The reason is that it leaves more room for future intervals.

In other words:

```text
If two intervals can both be candidates for the current kept interval,
the earlier-ending one never reduces future choices.
```

So the later scan should process intervals by increasing end.

Now this version can:

- Explain why conflicts should favor the earlier-ending interval.
- Derive the sorting key from "leave room for the future."
- Explain why `[1,2]` is better to keep than `[1,3]` in the tiny conflict.

It still lacks:

- A full scan that applies this rule to the entire array.

## Step 3: Sort by End and Count How Many Can Remain

The current baseline is:

```text
When intervals conflict, keep the one that ends earlier.
```

This breaks because:

> The local choice is not yet an executable algorithm.

Sort intervals by end:

```python
intervals.sort(key=lambda x: x[1])
```

During the scan, maintain:

```text
last_end: the end of the last kept interval
kept:     how many intervals have been kept so far
```

For the current interval:

```text
[start, end]
```

It can be kept if:

```text
start >= last_end
```

That means it does not overlap with the last interval we kept.

The condition is `>=`, not `>`.

Because:

```text
[1,2] and [2,3]
```

touch at an endpoint, and endpoint touching is non-overlapping in this problem.

First write a version that only computes `max_kept`:

```python
from typing import List


def max_non_overlapping(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[1])

    kept = 0
    last_end = float("-inf")

    for start, end in intervals:
        if start >= last_end:
            kept += 1
            last_end = end

    return kept
```

Check two key cases.

First:

```text
intervals = [[1,2],[2,3]]
```

The sorted order is still:

```text
[[1,2],[2,3]]
```

Scan:

```text
keep [1,2], last_end = 2
[2,3] has start = 2, so start >= last_end
keep [2,3]
```

So:

```text
max_kept = 2
```

Second:

```text
intervals = [[1,2],[1,2],[1,2]]
```

Scan:

```text
keep the first [1,2], last_end = 2
the second [1,2] has start = 1, so it cannot be kept
the third [1,2] has start = 1, so it cannot be kept
```

So:

```text
max_kept = 1
```

Now this version can:

- Scan intervals by increasing end.
- Use `last_end` to decide whether the current interval can remain.
- Compute the maximum number of non-overlapping intervals kept.

It still lacks:

- The original problem asks for minimum removals, not maximum kept count.

## Step 4: Convert Maximum Kept Back to Minimum Removed

The current baseline is:

```text
max_non_overlapping(intervals) computes the maximum number of intervals that can remain.
```

This breaks because:

> Returning `kept` answers the transformed problem, not the original problem.

The original answer is:

```text
minimum number of intervals removed
```

We already derived:

```text
min_removed = n - max_kept
```

So the final step is to convert `kept` back to removals.

Complete LeetCode code:

```python
from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])

        kept = 0
        last_end = float("-inf")

        for start, end in intervals:
            if start >= last_end:
                kept += 1
                last_end = end

        return len(intervals) - kept
```

Check the first example:

```text
intervals = [[1,2],[2,3],[3,4],[1,3]]
```

After sorting by end:

```text
[[1,2],[2,3],[1,3],[3,4]]
```

Scan:

```text
keep [1,2], last_end = 2
keep [2,3], last_end = 3
[1,3] overlaps, skip it
keep [3,4], last_end = 4
```

We keep `3` intervals.

There were `4` intervals total, so the minimum number removed is:

```text
4 - 3 = 1
```

This matches the expected output.

Now this version can:

- Sort by end and keep earlier-ending intervals.
- Correctly handle endpoint touching with `start >= last_end`.
- Return the minimum number of intervals removed.

## Correctness Intuition

The key is not "delete any interval when there is overlap."

The actual greedy choice is:

> Among intervals that could be kept next, keep the one that ends earliest.

The earlier an interval ends, the easier it is for later intervals to start after it.

Keeping a later-ending interval does not increase the current kept count, but it can block future intervals.

So after sorting by end, whenever the current interval starts after or at `last_end`, keep it immediately.

That scan computes the maximum number of intervals that can remain.

Then convert back:

```text
minimum removals = total intervals - maximum kept
```

## Complexity

Sorting costs:

```text
O(n log n)
```

The scan touches each interval once:

```text
O(n)
```

Total time complexity:

```text
O(n log n)
```

Aside from sorting, the algorithm only stores `kept` and `last_end`:

```text
O(1)
```

## Common Mistakes

### 1. Treating Endpoint Touching as Overlap

The problem says:

```text
[1,2] and [2,3]
```

are non-overlapping.

So the keep condition is:

```python
start >= last_end
```

not:

```python
start > last_end
```

### 2. Returning `kept`

`kept` is the maximum number of intervals that remain.

The original problem asks for the minimum number removed, so return:

```python
len(intervals) - kept
```

### 3. Sorting by Start and Then Deleting Casually

There are other valid interval implementations, but the most stable greedy explanation for this problem sorts by end.

"Ends earlier" directly means:

```text
leave more room for later intervals
```

That is easier to prove and less error-prone than repairing overlaps after sorting by start.

## Runnable Checks

```python
from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])

        kept = 0
        last_end = float("-inf")

        for start, end in intervals:
            if start >= last_end:
                kept += 1
                last_end = end

        return len(intervals) - kept


def check() -> None:
    s = Solution()
    assert s.eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert s.eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert s.eraseOverlapIntervals([[1, 2], [2, 3]]) == 0
    assert s.eraseOverlapIntervals([[1, 100], [11, 22], [1, 11], [2, 12]]) == 2


check()
```

## Summary

The greedy point in LeetCode 435 is:

```text
minimum removals = total intervals - maximum kept
```

The maximum-kept strategy is:

```text
sort by end, and keep an interval whenever it can start after the last kept interval
```

The scan invariant is:

> The kept intervals are non-overlapping, and `last_end` is as early as possible, leaving as much room as possible for later intervals.

After this problem, LeetCode 452 becomes more natural: there, we do not keep intervals; we cover intervals with the fewest arrow positions.
