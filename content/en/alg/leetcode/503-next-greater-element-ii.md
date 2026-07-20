---
title: "LeetCode 503: Where Does the Right Side End in a Circular Array?"
date: 2026-07-20T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["array", "monotonic stack", "circular array", "LeetCode 503"]
description: "Start from the exact circular next-greater query, build a verifiable scan, then derive a linear monotonic-stack solution that pushes each original index once."
keywords: ["LeetCode 503", "Next Greater Element II", "circular array", "monotonic stack", "Python"]
---

## Problem Requirement

You are given a circular integer array `nums`. Return an array `answer`.

For every index `i`:

```text
answer[i] = the first value strictly greater than nums[i] when moving right
```

If one full trip around the array finds no greater value, `answer[i] = -1`.

"Circular" means that moving past the final position continues from index `0`. An index cannot travel one full circle and use itself as its own answer.

LeetCode provides this method contract:

```text
nextGreaterElements(nums: List[int]) -> List[int]
```

### Example 1

```text
Input: nums = [1,2,1]
Output: [2,-1,2]
```

- The first `1` meets the greater value `2` immediately.
- `2` has no strictly greater value anywhere in one full circle.
- The final `1` wraps to the beginning and then reaches `2`.

### Example 2

```text
Input: nums = [2,2,2]
Output: [-1,-1,-1]
```

Equal values are not greater values.

### Example 3

```text
Input: nums = [7]
Output: [-1]
```

A single element cannot use itself as its next greater value.

### Constraints

- `1 <= nums.length <= 10^4`
- `-10^9 <= nums[i] <= 10^9`

## Step 1: Where Does "To the Right" End in a Circle?

Start with:

```text
nums = [1,2,1]
```

In a normal array, the final `1` has no element to its right. In a circular array, its observation order continues with:

```text
the 1 at the beginning
-> the middle 2
```

The first strictly greater value is `2`.

The current baseline is:

```text
Move right from the current position; after the end, continue from the beginning.
```

This baseline also needs a stopping point. From index `i`, inspect at most the other `n - 1` positions:

- Return the first greater value if one appears.
- Return `-1` if all of them fail.
- Do not enter a second lap or treat the starting index as a candidate.

This baseline breaks because:

> The manual order is clear, but it is not an executable index rule. A normal suffix stops at the physical end of the array.

Now this version can:

- state the circular right-side order precisely
- distinguish the first strictly greater value from the maximum value in the circle
- stop after one lap and preserve `-1` when no answer exists

It still lacks:

- a runnable circular index operation

## Step 2: Build a Correct Modulo Baseline

The current baseline is:

```text
From index i, inspect the following n - 1 circular positions.
```

The ordinary index `i + step` may cross the array boundary. Add one rule that performs the wraparound:

```text
next_index = (i + step) % n
```

When `i + step < n`, the result is the normal right-side index. After the end, modulo maps it back to the beginning.

First correct implementation:

```python
from typing import List


def next_greater_elements_scan(nums: List[int]) -> List[int]:
    n = len(nums)
    answer = [-1] * n

    for i in range(n):
        for step in range(1, n):
            next_index = (i + step) % n

            if nums[next_index] > nums[i]:
                answer[i] = nums[next_index]
                break

    return answer
```

The inner loop starts at `1`, so it moves at least one position. It stops before `n`, so it checks exactly the other `n - 1` positions and never returns to the starting index.

Trace the final index `i = 2` in `[1,2,1]`:

| `step` | `(i + step) % n` | Value | Strictly greater? |
| ---: | ---: | ---: | --- |
| 1 | 0 | 1 | No |
| 2 | 1 | 2 | Yes; write the answer and stop |

Checks:

```python
assert next_greater_elements_scan([1, 2, 1]) == [2, -1, 2]
assert next_greater_elements_scan([5, 4, 3, 2, 1]) == [-1, 5, 5, 5, 5]
assert next_greater_elements_scan([1, 2, 3]) == [2, 3, -1]
assert next_greater_elements_scan([2, 2, 2]) == [-1, -1, -1]
assert next_greater_elements_scan([7]) == [-1]
```

Now this version can:

- inspect one complete circular future for every index
- stop at the first strictly greater value
- preserve `-1` when no answer exists
- handle equal and singleton inputs

It still lacks:

- reuse across starting indices; many of them inspect the same circular positions

In the worst case, every index checks O(n) candidates, so the time complexity is O(n^2). Extra space is O(1), excluding the output.

## Step 3: Expand the Circle Into Two Virtual Passes

The current baseline rebuilds this circular order independently for every start:

```text
i + 1, i + 2, ..., i + n - 1
```

The break is:

> Every start reconstructs the same circle. We first need one shared scan order that represents wraparound.

For an array of length `n`, scan these virtual positions:

```text
i = 0, 1, 2, ..., 2n - 1
```

Map every virtual position to the original array with:

```python
index = i % n
```

Trace `[1,2,1]`:

| Virtual `i` | Original `i % n` | Value |
| ---: | ---: | ---: |
| 0 | 0 | 1 |
| 1 | 1 | 2 |
| 2 | 2 | 1 |
| 3 | 0 | 1 |
| 4 | 1 | 2 |
| 5 | 2 | 1 |

The virtual sequence is:

```text
first pass:  1,2,1
second pass: 1,2,1
virtual:     1,2,1,1,2,1
```

Two passes are sufficient because:

- The first pass provides the ordinary suffix.
- The second pass provides the wrapped prefix.
- Together they expose one complete circular future for every original index.
- A third pass only repeats candidates already seen.

The only change in this step is the shared scan route:

```python
for i in range(2 * n):
    index = i % n
```

This is not a complete algorithm yet. It identifies the current original position but stores no unresolved work.

Now this version can:

- represent the circle as `2n` linear visits
- expose one complete circular future for every original index
- explain why an infinite loop or third pass is unnecessary

It still lacks:

- state that remembers which original indices still need answers

## Step 4: Push Each Unresolved Original Index Once

The current baseline has one two-pass order, but it still only visits values.

The break is:

> One current greater value may answer several earlier indices. Without storing those unresolved indices, they still need independent scans.

Add a stack named `stack` containing original indices whose answers are unknown.

When a virtual position maps to `index`, a strictly greater current value settles the stack top:

```python
while stack and nums[index] > nums[stack[-1]]:
    previous = stack.pop()
    answer[previous] = nums[index]
```

The stack stores indices rather than values because the algorithm must know which `answer` entry to write and must ensure each original index is pushed once.

The second pass only provides candidate values. Push original indices during the first pass only:

```python
if i < n:
    stack.append(index)
```

Connect these changes to the previous virtual loop:

```python
answer = [-1] * n
stack = []

for i in range(2 * n):
    index = i % n

    while stack and nums[index] > nums[stack[-1]]:
        previous = stack.pop()
        answer[previous] = nums[index]

    if i < n:
        stack.append(index)
```

Trace all six virtual positions for `[1,2,1]`:

| `i` | `index` | Value | Operation | Stack after | `answer` |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | 0 | 1 | First pass; push 0 | `[0]` | `[-1,-1,-1]` |
| 1 | 1 | 2 | Pop 0, write 2; push 1 | `[1]` | `[2,-1,-1]` |
| 2 | 2 | 1 | Cannot pop 1; push 2 | `[1,2]` | `[2,-1,-1]` |
| 3 | 0 | 1 | Equal to top; no pop or second-pass push | `[1,2]` | `[2,-1,-1]` |
| 4 | 1 | 2 | Pop 2, write 2; no second-pass push | `[1]` | `[2,-1,2]` |
| 5 | 2 | 1 | Cannot pop 1; no second-pass push | `[1]` | `[2,-1,2]` |

Values at stack indices remain non-increasing from bottom to top. Equal values do not satisfy the strict comparison and may remain together.

Indices still in the stack at the end have no greater value. Their initial answer `-1` is already correct.

Now this version can:

- let one current value settle multiple unresolved indices
- resolve wrapped answers during the second pass
- push every original index only during the first pass
- preserve strict comparison for equal values

It still lacks:

- the full first-greater invariant
- an O(n) proof for the nested loop
- the complete LeetCode wrapper and tests

## Step 5: Why Two Passes Still Take O(n) Time

The stack mechanism is operational. Two questions remain:

1. Why is the current value the first greater value for each popped index?
2. Why does a `while` inside a `2n` loop remain O(n)?

Use this loop invariant:

> Before each virtual position is processed, `stack` stores first-pass original indices whose answers have not appeared in the scanned circular order, and their values are non-increasing from bottom to top.

If `previous` remains in the stack, every already-scanned circular candidate failed to exceed `nums[previous]`. The first current value satisfying:

```text
nums[index] > nums[previous]
```

is therefore its first greater value.

Complete LeetCode implementation:

```python
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        answer = [-1] * n
        stack = []

        for i in range(2 * n):
            index = i % n

            while stack and nums[index] > nums[stack[-1]]:
                previous = stack.pop()
                answer[previous] = nums[index]

            if i < n:
                stack.append(index)

        return answer
```

Every original index is pushed exactly once during the first pass. The second pass never pushes. Once an index is popped, it has its answer and never enters the stack again.

Therefore:

- total pushes: `n`
- total pops: at most `n`
- outer iterations: `2n`
- total iterations of every inner `while`: at most `n`

The total time is O(n), not O(n^2).

### Checks

```python
solution = Solution()

assert solution.nextGreaterElements([1, 2, 1]) == [2, -1, 2]
assert solution.nextGreaterElements([5, 4, 3, 2, 1]) == [-1, 5, 5, 5, 5]
assert solution.nextGreaterElements([1, 2, 3]) == [2, 3, -1]
assert solution.nextGreaterElements([2, 2, 2]) == [-1, -1, -1]
assert solution.nextGreaterElements([7]) == [-1]
assert solution.nextGreaterElements([3, 1, 3]) == [-1, 3, -1]
```

### Complexity

- Time: O(n), because every original index is pushed once and popped at most once.
- Extra space: O(n), for the output and stack.

## Common Mistakes

### 1. Pushing during the second pass

That inserts duplicate copies of an original index and breaks the one-push/one-pop model.

### 2. Popping with `>=`

The problem asks for strictly greater values. Equal values cannot settle an answer, so the condition must be:

```python
nums[index] > nums[stack[-1]]
```

### 3. Scanning only once

One pass cannot resolve wrapped candidates such as the final `1` in `[1,2,1]`.

### 4. Scanning more than twice

Two passes already expose one full circle of future candidates for every index.

### 5. Declaring O(n^2) from the nested syntax

Count total pushes and pops. One original index cannot be popped twice.

## Summary

The derivation is:

```text
define the first strictly greater value in a circle
-> scan one lap per index with modulo
-> expand the circular order into two virtual passes
-> store unresolved original indices from the first pass
-> use the second pass only for wrapped candidates
-> push and pop every original index at most once
```

LeetCode 503 extends the 739 stack model by changing the candidate order for a circular array while preserving one stack entry per original index.

Continue with LeetCode 84 Largest Rectangle in Histogram to use a lower current value as the event that settles stored boundaries.
