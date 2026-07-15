---
title: "LeetCode 739: Daily Temperatures and the First Warmer Day to the Right"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "array", "stack", "monotonic stack", "LeetCode 739"]
description: "Start from an O(n^2) right-scan baseline, derive a monotonic stack of unresolved indices, and solve LeetCode 739 in O(n) time."
keywords: ["LeetCode 739", "Daily Temperatures", "monotonic stack", "next greater element", "array", "Hot100"]
---

## Problem Requirement

You are given an integer array `temperatures`, where `temperatures[i]` is the temperature on day `i`.

Return an array `answer` where:

```text
answer[i] = the number of days after day i until a warmer temperature
```

If no later day is warmer, `answer[i] = 0`.

"Warmer" means strictly greater. An equal temperature does not resolve a waiting day.

LeetCode provides this method contract:

```text
dailyTemperatures(temperatures: List[int]) -> List[int]
```

### Example

```text
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

### Constraints

- `1 <= temperatures.length <= 10^5`
- `30 <= temperatures[i] <= 100`

## Step 1: The Answer Is a Waiting Time, Not a Temperature

Start with a smaller input:

```text
temperatures = [73,71,72,76]
```

Answer each day directly:

- Day 0 is `73`. Its first warmer day is day 3 at `76`, so the wait is `3`.
- Day 1 is `71`. Day 2 is `72`, so the wait is `1`.
- Day 2 is `72`. Day 3 is `76`, so the wait is `1`.
- Day 3 has no later day, so its answer is `0`.

The result is:

```text
[3,1,1,0]
```

The current baseline is:

```text
For each day, look to the right for the first warmer temperature.
```

Fix three details before choosing an algorithm:

1. Find the first warmer day, not the maximum future temperature.
2. Return an index distance, not a temperature difference.
3. Equal is not warmer; `[70,70]` produces `[0,0]`.

Now this version can:

- state the exact meaning of each output entry
- preserve zero when no warmer future day exists
- distinguish strictly warmer from equal

It still lacks:

- a correct executable search for every day

## Step 2: Build the Right-Scan Baseline

The current baseline says to search right from each day. The direct implementation is a nested loop.

```python
from typing import List


def daily_temperatures_scan(temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    answer = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if temperatures[j] > temperatures[i]:
                answer[i] = j - i
                break

    return answer
```

Every line maps to a requirement:

- `answer` starts with zeros for unresolved days.
- `j` starts at `i + 1`, so only future days are checked.
- `>` preserves the strict warmer rule.
- `break` keeps the first warmer day.
- `j - i` is the waiting time.

Checks:

```python
assert daily_temperatures_scan([73, 71, 72, 76]) == [3, 1, 1, 0]
assert daily_temperatures_scan([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures_scan([60, 50, 40]) == [0, 0, 0]
assert daily_temperatures_scan([70, 70]) == [0, 0]
```

Now this version can:

- find the first strictly warmer future day correctly
- preserve zero when no answer exists
- follow the problem definition directly

It still lacks:

- acceptable worst-case performance; many days rescan the same suffix

For a decreasing array, every `i` scans to the end. The worst-case time is O(n^2), which is too expensive when `n` reaches `10^5`.

## Step 3: Keep the Days That Are Still Waiting

Revisit:

```text
[73,71,72,76]
```

When `71` arrives, its answer is unknown. When `72` arrives, that answer becomes known immediately because `72 > 71`.

Reverse the responsibility:

> Do not make every old day search forward. Let today's temperature resolve earlier days that are still waiting.

To calculate waiting time at resolution, store day indices rather than temperature values. Add a stack named `stack` containing indices whose warmer day has not been found.

For current index `i` and current `temperature`, inspect the stack top:

```python
while stack and temperature > temperatures[stack[-1]]:
    previous = stack.pop()
    answer[previous] = i - previous
```

The stack now performs real work: comparison, removal, and answer updates. After all resolvable days are popped, push the current index:

```python
stack.append(i)
```

Trace `[73,71,72,76]`:

| `i` | Temperature | Operation | Update | Stack after |
| ---: | ---: | --- | --- | --- |
| 0 | 73 | Push 0 | None | `[0]` |
| 1 | 71 | 71 is not above 73; push 1 | None | `[0,1]` |
| 2 | 72 | Pop 1, then push 2 | `answer[1] = 1` | `[0,2]` |
| 3 | 76 | Pop 2 and 0, then push 3 | `answer[2] = 1`, `answer[0] = 3` | `[3]` |

During the scan, the stack has three properties:

- indices increase from bottom to top
- every stored day is still unresolved
- corresponding temperatures are non-increasing from bottom to top

The last property is not "strictly decreasing." Equal temperatures do not satisfy `>`, so equal values may remain together.

Now this version can:

- remember unresolved days instead of rescanning their suffixes
- let one warmer day resolve several earlier days
- write the waiting time directly from index distance

It still lacks:

- a proof that the current day is the first warmer day for every popped index
- the complete LeetCode wrapper and complexity argument

## Step 4: Why a Pop Finds the First Warmer Day

Before processing day `i`, an index `previous` still in the stack means:

```text
No day from previous + 1 through i - 1 was strictly warmer than temperatures[previous].
```

Otherwise, `previous` would have been popped on that earlier day.

When the current day satisfies:

```text
temperatures[i] > temperatures[previous]
```

day `i` is therefore the first warmer day to the right, and the algorithm can safely write:

```text
answer[previous] = i - previous
```

Complete implementation:

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0] * len(temperatures)
        stack = []

        for i, temperature in enumerate(temperatures):
            while stack and temperature > temperatures[stack[-1]]:
                previous = stack.pop()
                answer[previous] = i - previous

            stack.append(i)

        return answer
```

The loop invariant is:

> Before each iteration, `stack` stores unresolved indices in increasing date order, and their temperatures are non-increasing from bottom to top.

Popping resolves every top temperature lower than the current temperature. When popping stops, the top temperature is greater than or equal to the current one, so pushing the current index preserves the invariant.

Indices left in the stack after the scan have no warmer future day. Their answers are already the required default `0`.

### Checks

```python
solution = Solution()

assert solution.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert solution.dailyTemperatures([73, 71, 72, 76]) == [3, 1, 1, 0]
assert solution.dailyTemperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert solution.dailyTemperatures([60, 50, 40]) == [0, 0, 0]
assert solution.dailyTemperatures([70, 70]) == [0, 0]
```

### Complexity

- Time: O(n), because every index is pushed once and popped at most once.
- Extra space: O(n), because all indices may remain unresolved in the worst case.

The nested `while` does not make the total O(n^2). One index cannot be popped twice, so all while-loop iterations across the entire scan total O(n).

## Common Mistakes

### 1. Storing only temperatures

The result is a day distance, so the algorithm needs indices to compute `i - previous`.

### 2. Popping with `>=`

Equal is not warmer. Using `>=` would incorrectly resolve day 0 in `[70,70]`.

### 3. Forgetting to push the current day

Today may resolve older days, but today itself may still wait for a warmer future day.

### 4. Calling the stack strictly decreasing

Equal temperatures remain, so "non-increasing" is the accurate invariant.

### 5. Declaring O(n^2) from the nested syntax

Count total pushes and pops, not merely the visual loop nesting.

## Summary

The derivation is:

```text
define the first strictly warmer day
-> scan right from each day
-> expose repeated suffix scans
-> store unresolved day indices
-> let the current temperature pop and resolve lower stack tops
-> push and pop each index at most once
```

The monotonic stack is not useful merely because it looks ordered. It preserves unresolved candidates and settles each one at the first moment its answer becomes known.
