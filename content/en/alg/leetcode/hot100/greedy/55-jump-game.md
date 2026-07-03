---
title: "LeetCode 55: Jump Game With Farthest Reach Greedy"
date: 2026-07-03T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "greedy", "array", "reachability", "LeetCode 55"]
description: "Solve LeetCode 55 Jump Game in Python by deriving reachability from a zero barrier, a reachable-array baseline, and the farthest reachable range greedy."
keywords: ["LeetCode 55", "Jump Game", "greedy", "farthest reach", "reachability", "array", "Python"]
---

> **Subtitle / Summary**
> Jump Game is not about guessing one path. It is about maintaining how far all reachable positions can cover.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `greedy`, `array`, `reachability`
- **SEO keywords**: LeetCode 55, Jump Game, greedy, farthest reach, reachability
- **Meta description**: A pressure-first Python guide to LeetCode 55 that derives a reachable-array baseline and compresses it into the farthest reachable range.

---

## Problem Requirement

You are given an integer array `nums`.

You start at index `0`. `nums[i]` is the maximum jump length from index `i`.

Return whether you can reach the last index.

### Input and Output

- Input: `nums: List[int]`
- Output: `bool`
- You start at index `0`.
- Each value is the maximum jump length, not the exact jump length.
- You only need to decide reachability; you do not need to return a path.

### Examples

```text
Input: nums = [2,3,1,1,4]
Output: true
```

One valid path is:

```text
0 -> 1 -> 4
```

From index `0`, jump to index `1`; from index `1`, jump to the last index.

```text
Input: nums = [3,2,1,0,4]
Output: false
```

No matter how you jump, you get stuck at index `3`, whose jump length is `0`, before reaching index `4`.

### Constraints

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 10^5`

## Step 1: Do Not Guess a Path First

Start with the failing example:

```text
nums = [3,2,1,0,4]
```

From index `0`, you can jump as far as index `3`.

There seem to be several choices:

```text
0 -> 1
0 -> 2
0 -> 3
```

The current baseline is:

```text
Try to pick a jump path and see whether it reaches the end.
```

This baseline breaks because:

> The problem does not ask for a path. Focusing on one path hides the more important question: how far can all reachable positions cover?

In `[3,2,1,0,4]`, every reachable route fails to cross index `3`:

```text
index:  0  1  2  3  4
nums:   3  2  1  0  4
cover:  ----------^
```

Index `4` is outside the covered range, so the answer is `False`.

The change in this step is the viewpoint:

> Do not choose a concrete path first. Maintain the farthest index currently covered by all reachable positions.

Now this version can:

- treat the problem as reachability, not path reconstruction
- see that failure happens when coverage stops before the last index
- prepare the state idea: farthest reachable index

It still lacks:

- a first correct runnable reachability check

## Step 2: Build a Reachability Array Baseline

The current baseline is:

```text
We need to know which indices are reachable.
```

The break is:

> The coverage idea is not executable yet. We need a correct version before compressing it.

Use an array:

```text
reachable[i] == True means index i can be reached from index 0
```

Initialize:

```python
reachable[0] = True
```

If index `i` is reachable, then it can mark:

```text
i + 1, i + 2, ..., i + nums[i]
```

Correct baseline:

```python
from typing import List


def can_jump_reachable(nums: List[int]) -> bool:
    n = len(nums)
    reachable = [False] * n
    reachable[0] = True

    for i in range(n):
        if not reachable[i]:
            continue

        right = min(n - 1, i + nums[i])
        for nxt in range(i + 1, right + 1):
            reachable[nxt] = True

    return reachable[-1]
```

Check:

```python
assert can_jump_reachable([2, 3, 1, 1, 4]) is True
assert can_jump_reachable([3, 2, 1, 0, 4]) is False
```

Now this version can:

- explicitly mark which indices are reachable
- expand reachability from each reachable index
- handle both official examples correctly

It still lacks:

- compression. The `reachable` array and inner marking loop are heavier than necessary.

## Step 3: Compress Reachability to farthest

The current baseline is:

```text
reachable[i] stores whether each index is reachable.
```

Look at the successful example:

```text
nums = [2,3,1,1,4]
```

From index `0`, we can cover up to index `2`:

```text
farthest = 2
```

As long as the scan index satisfies `i <= farthest`, index `i` is reachable and can extend the right boundary:

```text
farthest = max(farthest, i + nums[i])
```

The break is:

> If every index in `[0..farthest]` is reachable, we do not need to store every boolean. The right boundary is enough.

Replace the array with one variable:

```python
farthest = 0

for i, jump in enumerate(nums):
    if i <= farthest:
        farthest = max(farthest, i + jump)
```

Trace `[2,3,1,1,4]`:

```text
start: farthest = 0

i = 0, jump = 2: i <= 0, farthest = max(0, 2) = 2
i = 1, jump = 3: i <= 2, farthest = max(2, 4) = 4
```

Now `farthest` already reaches the last index, so the answer is `True`.

Now this version can:

- represent current coverage with one boundary
- extend coverage from each reachable index
- use local farthest reach to prove global reachability

It still lacks:

- the failure rule for when the scan reaches an unreachable index

## Step 4: Fail When the Scan Reaches a Gap

The current baseline is:

```text
Scan reachable indices and update farthest.
```

The break is:

> If `i > farthest`, index `i` is not reachable. Using it to expand coverage would be invalid.

Final rules:

- if `i > farthest`, return `False`
- otherwise update `farthest` with `i + nums[i]`
- if `farthest >= last`, return `True`

Complete code:

```python
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last = len(nums) - 1
        farthest = 0

        for i, jump in enumerate(nums):
            if i > farthest:
                return False

            farthest = max(farthest, i + jump)

            if farthest >= last:
                return True

        return True
```

The loop invariant is:

> At the start of each loop, `farthest` is the farthest index covered by previously reachable positions. If the current `i` is beyond it, `i` is unreachable and later indices cannot be saved by earlier positions.

Check:

```python
def check() -> None:
    s = Solution()

    assert s.canJump([2, 3, 1, 1, 4]) is True
    assert s.canJump([3, 2, 1, 0, 4]) is False
    assert s.canJump([0]) is True
    assert s.canJump([2, 0, 0]) is True
    assert s.canJump([1, 0, 1, 0]) is False


check()
```

Now this version can:

- avoid constructing a concrete jump path
- avoid storing a full reachability array
- use `farthest` as the current coverage range
- return `False` as soon as the scan reaches a gap

## Complexity

Let `n = len(nums)`.

- Time complexity: `O(n)`, because each index is scanned at most once.
- Space complexity: `O(1)`, because only `farthest` is stored.

## Summary

The greedy idea in Jump Game is:

```text
As long as the current index i is inside the covered range,
it can try to extend the right boundary with i + nums[i].
```

So we maintain:

```text
farthest = the farthest index covered by all reachable positions so far
```

If at any point:

```text
i > farthest
```

the scan has reached a gap. No earlier jump can cover `i`, so the answer is `False`.
