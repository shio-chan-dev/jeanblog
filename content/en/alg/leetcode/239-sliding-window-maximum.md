---
title: "LeetCode 239: Sliding Window Maximum"
date: 2026-01-19T17:46:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["sliding window", "array", "deque", "LeetCode 239"]
description: "Solve LeetCode 239 by defining complete windows, building a correct scan, and deriving an O(n) index-deque solution."
keywords: ["Sliding Window Maximum", "sliding window", "deque", "LeetCode 239", "Python"]
---

## Problem Requirement

You are given an integer array `nums` and an integer `k`. A window of exactly
`k` contiguous elements starts at the left edge of `nums` and moves one
position to the right at a time. Return the maximum value from every window, in
the same left-to-right order as the windows.

The elements in a window must be contiguous. Their original order and
positions do not change, and adjacent windows can overlap. Values do not need
to be unique and may be negative.

### LeetCode Contract

LeetCode calls `maxSlidingWindow(nums, k)`. The method receives the integer
array and the valid window size, then returns an integer array containing one
maximum for each complete window. The input is guaranteed to satisfy the
constraints below.

### Examples

Example 1:

```text
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

Example 2:

```text
Input: nums = [1], k = 1
Output: [1]
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `1 <= k <= nums.length`

## Step 1: When Is a Window Complete?

For Example 1, which indices form the window when its right edge reaches index
`2`? What changes when that edge reaches index `3`?

The current baseline says only that a size-`k` window moves right. This breaks
when we try to enumerate the outputs: it does not yet specify the window's
exact left edge or the first position at which a full window exists.

Use zero-based indices and call the current ending index `right`. If the window
begins at `left` and contains exactly `k` elements, then:

```text
right - left + 1 = k
left = right - k + 1
```

The window is complete only when `left >= 0`, which is equivalent to:

```text
right >= k - 1
```

Before that condition is true, there are fewer than `k` elements and no output
position exists. Once it is true, the window is `nums[left:right + 1]`. During
a full left-to-right pass, that window contributes output position `left`.

Check this boundary rule against the entire first example, where `k = 3`:

| `right` | `left = right - k + 1` | Complete? | Window indices | Window values | Output position | Maximum |
| ---: | ---: | :---: | :---: | :--- | ---: | ---: |
| 0 | -2 | No | - | - | - | - |
| 1 | -1 | No | - | - | - | - |
| 2 | 0 | Yes | `0..2` | `[1,3,-1]` | 0 | 3 |
| 3 | 1 | Yes | `1..3` | `[3,-1,-3]` | 1 | 3 |
| 4 | 2 | Yes | `2..4` | `[-1,-3,5]` | 2 | 5 |
| 5 | 3 | Yes | `3..5` | `[-3,5,3]` | 3 | 5 |
| 6 | 4 | Yes | `4..6` | `[5,3,6]` | 4 | 6 |
| 7 | 5 | Yes | `5..7` | `[3,6,7]` | 5 | 7 |

The first output appears at `right = 2 = k - 1`. For an array of length `n`,
the legal left edges are `0` through `n - k`, so the output count is:

```text
(n - k) - 0 + 1 = n - k + 1
```

Here, `n = 8` and `k = 3`, so there are `8 - 3 + 1 = 6` outputs. That matches
both the six complete rows in the table and the six values in the expected
output.

### Checkpoint 1

**Freeze:** The reader can identify every complete window and its output
position.

**It still lacks:** No runnable algorithm calculates all maxima.

## Step 2: Scan Every Complete Window

Step 1 can identify every complete window, but it still leaves the maxima in
the example table as manual results. For instance, when `right = 2`, the
boundary rule gives `left = 0` and the slice `nums[0:3]`, but no executable
rule computes `3` or places it in the returned list.

The current baseline is the exact boundary model from Step 1. This breaks when
we need to produce the full answer: knowing every legal `left` and `right`
does not calculate or collect any maximum.

Make one change: turn that boundary model into a complete scan function. For
each legal `left`, derive the inclusive `right`, scan exactly
`nums[left:right + 1]`, and append its maximum to `answer`.

```python
def max_sliding_window_scan(nums: list[int], k: int) -> list[int]:
    answer = []

    for left in range(len(nums) - k + 1):
        right = left + k - 1
        answer.append(max(nums[left:right + 1]))

    return answer


# Official examples
assert max_sliding_window_scan([1, 3, -1, -3, 5, 3, 6, 7], 3) == [
    3, 3, 5, 5, 6, 7
]
assert max_sliding_window_scan([1], 1) == [1]

# Boundary cases
assert max_sliding_window_scan([4, -2, 7], 1) == [4, -2, 7]
assert max_sliding_window_scan([4, -2, 7], 3) == [7]
assert max_sliding_window_scan([9, 7, 5, 3, 1], 3) == [9, 7, 5]
```

Check this change by running the block. The assertions cover both official
examples, `k = 1`, `k = n`, and decreasing input. They finish without output
when every returned list is correct.

There are `n - k + 1` windows. Each slice contains `k` values, and both
building that slice and finding its maximum take `O(k)` time. The total time
is therefore `O((n - k + 1)k)`, commonly written as `O(nk)`. The returned list
holds `n - k + 1` values, and the temporary slice uses `O(k)` extra space.

### Checkpoint 2

**Freeze:** The reader can compute every window maximum correctly with the
complete `O(nk)` scan baseline.

**It still lacks:** Overlapping windows repeatedly scan up to `k` values from
scratch instead of reusing work from the previous window.

## Step 3: Which Stored Position Just Expired?

The scan baseline is correct, but it rebuilds a slice for every window and
then discards that window's membership. To carry the membership forward, we
must know exactly which old element leaves when `left` moves right.

This becomes ambiguous if we store only values. For `nums = [2,2,1]` and
`k = 2`, the first complete window contains a `2` from index `0` and another
`2` from index `1`. When the next window starts at index `1`, index `0` leaves
but index `1` stays. The value `2` alone does not identify those two
occurrences.

The current baseline is `max_sliding_window_scan`, which knows `left` and
`right` but keeps no state between windows. This breaks when we try to update
membership from one window to the next: expiration is about a position, and
the baseline has not retained any position to remove.

Make one change: use an ordinary `collections.deque` named `candidates` and
store indices in arrival order. Append each `right` index. Once `left` is
known, remove front indices while `candidates[0] < left`, because those
positions are outside the current window. An index equal to `left` must stay.

This version deliberately finds each maximum by scanning the values at all
current candidate indices:

```python
from collections import deque


def max_sliding_window_candidates(nums: list[int], k: int) -> list[int]:
    answer = []
    candidates = deque()

    for right in range(len(nums)):
        candidates.append(right)
        left = right - k + 1

        while candidates and candidates[0] < left:
            candidates.popleft()

        if left >= 0:
            answer.append(max(nums[index] for index in candidates))

    return answer


cases = [
    ([1, 3, -1, -3], 3),
    ([2, 2, 1], 2),
    ([1, 3, -1, -3, 5, 3, 6, 7], 3),
    ([1], 1),
    ([4, -2, 7], 1),
    ([4, -2, 7], 3),
    ([9, 7, 5, 3, 1], 3),
]

for nums, k in cases:
    assert max_sliding_window_candidates(nums, k) == max_sliding_window_scan(
        nums, k
    )
```

Check expiration with `nums = [1,3,-1,-3]` and `k = 3`:

| `right` | `left` | After append | Expired indices | `candidates` after eviction | Current values | Output |
| ---: | ---: | :---: | :---: | :---: | :--- | ---: |
| 0 | -2 | `[0]` | - | `[0]` | `[1]` | - |
| 1 | -1 | `[0,1]` | - | `[0,1]` | `[1,3]` | - |
| 2 | 0 | `[0,1,2]` | - | `[0,1,2]` | `[1,3,-1]` | 3 |
| 3 | 1 | `[0,1,2,3]` | `0` | `[1,2,3]` | `[3,-1,-3]` | 3 |

At `right = 3`, the new left edge is `1`. The test `0 < 1` removes index `0`
from the front, while indices `1`, `2`, and `3` remain as the exact current
window.

Now check duplicate identity with `nums = [2,2,1]` and `k = 2`:

| `right` | `left` | After append | Expired indices | `candidates` after eviction | Current values | Output |
| ---: | ---: | :---: | :---: | :---: | :--- | ---: |
| 0 | -1 | `[0]` | - | `[0]` | `[2]` | - |
| 1 | 0 | `[0,1]` | - | `[0,1]` | `[2,2]` | 2 |
| 2 | 1 | `[0,1,2]` | `0` | `[1,2]` | `[2,1]` | 2 |

The two equal values remain distinguishable at indices `0` and `1`. When the
window moves, the expiration rule removes index `0` specifically and keeps
the equal value at index `1`. The executable checks compare this membership
version with the complete Task 2 scan for both traces, the official input, and
the earlier boundary cases.

Each index is appended once and removed from the front at most once, with each
operation taking constant time. However, every complete window still scans all
`k` current candidate values to call `max`, so the total time is
`O((n - k + 1)k + n)`, commonly written as `O(nk)`. The deque uses `O(k)`
auxiliary space, apart from the returned answer.

### Checkpoint 3

**Freeze:** The reader can maintain exactly the indices in the current window
and distinguish equal values by position.

**It still lacks:** Finding each maximum still scans all current candidates,
which takes `O(k)` time per complete window.

## Step 4: Which Candidates Can Never Become Maximum?

The ordinary membership deque is correct, but a complete window can still
contain `k` candidate indices. Calling `max()` across those candidates repeats
an `O(k)` scan for every output.

Consider an earlier candidate index `i` and the new index `right`, where:

```text
i < right
nums[i] <= nums[right]
```

The later value makes the earlier value permanently useless:

- As long as a future window still contains `i`, it also contains `right`
  because `right` is later.
- `nums[right]` is at least as large as `nums[i]`, so `i` cannot provide a
  better maximum.
- Index `i` expires before `right`, so there is no later window in which `i`
  remains after `right` has left.

Call `i` a **dominated candidate**. The current baseline keeps dominated
indices and scans them again. This breaks the attempt to make maximum lookup
constant time.

The deque already stores indices in increasing arrival order, so compare the
new value with the tail. In the previous version, replace the per-window
`max()` scan by removing every smaller-or-equal tail before appending `right`:

```python
while candidates and nums[candidates[-1]] <= nums[right]:
    candidates.pop()
```

If the tail is removed, compare the same new value with the next tail. When
the loop stops, either the deque is empty or its tail value is greater than
`nums[right]`. Appending `right` therefore leaves candidate values in strictly
decreasing order from front to back. The maximum is now
`nums[candidates[0]]`.

This is the first point at which the deque is **monotonic**: its indices
increase, while their values strictly decrease.

### Final LeetCode Implementation

Connect the new rule to the membership version. For each `right`, expire old
front indices, remove dominated tail indices, append `right`, and emit the
front value once the first window is complete.

```python
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        answer = []
        candidates = deque()

        for right in range(len(nums)):
            left = right - k + 1

            while candidates and candidates[0] < left:
                candidates.popleft()

            while candidates and nums[candidates[-1]] <= nums[right]:
                candidates.pop()

            candidates.append(right)

            if right >= k - 1:
                answer.append(nums[candidates[0]])

        return answer
```

### Official-Example Trace

For `nums = [1,3,-1,-3,5,3,6,7]` and `k = 3`, each deque entry below is
written as `index:value`. Tail removals are listed in the order they occur.

| `right` | `left` | Expired front | Removed tails | Deque after append | Output |
| ---: | ---: | :---: | :--- | :--- | ---: |
| 0 | -2 | - | - | `[0:1]` | - |
| 1 | -1 | - | `0:1` | `[1:3]` | - |
| 2 | 0 | - | - | `[1:3, 2:-1]` | 3 |
| 3 | 1 | - | - | `[1:3, 2:-1, 3:-3]` | 3 |
| 4 | 2 | `1:3` | `3:-3`, `2:-1` | `[4:5]` | 5 |
| 5 | 3 | - | - | `[4:5, 5:3]` | 5 |
| 6 | 4 | - | `5:3`, `4:5` | `[6:6]` | 6 |
| 7 | 5 | - | `6:6` | `[7:7]` | 7 |

The outputs are `[3,3,5,5,6,7]`. At `right = 4`, front expiration first
removes index `1`, then the new value `5` removes both smaller tail values.
This row exercises both removal directions in one iteration.

### Equal-Value Policy

The `<=` comparison removes an earlier equal value and keeps the later equal
value. For `[2,2,1]` with `k = 2`, processing index `1` removes index `0`
before appending index `1`. Both values could produce the same maximum, but
index `1` expires later, so it is never worse for any current or future
window.

Using `<` instead would also produce correct maxima, but it would retain equal
values and make the deque values non-increasing rather than strictly
decreasing. This implementation deliberately uses `<=` so one representative
of an equal maximum remains: the latest one.

### Loop Invariant and Correctness

After each `right` is processed and appended:

1. Candidate indices are strictly increasing and no candidate is left of the
   current `left` boundary.
2. Candidate values are strictly decreasing from front to back.
3. Every processed, unexpired index omitted from the deque is dominated by a
   later deque index with a greater-or-equal value.

The expiration loop preserves the first property because expired indices can
only occur at the front. The tail loop preserves the third property because
each removed index is replaced by the later `right`, which has a
greater-or-equal value. Popping continues until appending `right` preserves the
second property.

For a complete window, the front candidate is at least as large as every
other stored candidate by the second property. Every omitted window index is
no larger than a later stored candidate by the third property. Therefore the
front value is the maximum of the entire current window, so every appended
output is correct.

### Complexity

Each of the `n` indices is appended exactly once. An index can then be removed
once: either from the tail when a later value dominates it or from the front
when it expires. Across the whole run, the two `while` loops therefore perform
at most `n` successful removals, not `k` removals per window. The outer loop
and all failed stopping comparisons add only constant work per index, giving
amortized `O(n)` time.

After expiration, the deque contains only indices from the current window, so
it holds at most `k` indices. Auxiliary space is `O(k)`, excluding the returned
array of `n - k + 1` maxima.

### Executable Checks

Fixed assertions cover the official examples, both window-size extremes,
decreasing input, repeated tail removal, and equal values:

```python
solution = Solution()

assert solution.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3) == [
    3, 3, 5, 5, 6, 7
]
assert solution.maxSlidingWindow([1], 1) == [1]
assert solution.maxSlidingWindow([4, -2, 7], 1) == [4, -2, 7]
assert solution.maxSlidingWindow([4, -2, 7], 3) == [7]
assert solution.maxSlidingWindow([9, 7, 5, 3, 1], 3) == [9, 7, 5]
assert solution.maxSlidingWindow([1, 2, 3, 4, 5], 3) == [3, 4, 5]
assert solution.maxSlidingWindow([2, 2, 1], 2) == [2, 2]
```

The optimized method can also be checked against the Task 2 scan on generated
valid inputs. A fixed seed makes a failure reproducible, and the copy check
verifies that the method does not mutate `nums`:

```python
from random import Random


rng = Random(239)

for _ in range(2000):
    n = rng.randint(1, 30)
    nums = [rng.randint(-20, 20) for _ in range(n)]
    k = rng.randint(1, n)
    original = nums.copy()

    actual = solution.maxSlidingWindow(nums, k)

    assert nums == original
    assert actual == max_sliding_window_scan(nums, k)
```

### Common Mistakes

- Storing values instead of indices makes front expiration ambiguous when
  values repeat.
- Expiring `candidates[0] <= left` incorrectly removes the index at the
  current left boundary. Only indices `< left` are outside the window.
- Appending `right` before the tail loop compares the new index with itself.
  Remove dominated old tails first.
- Using `<` in the tail loop while claiming values are strictly decreasing
  leaves equal values in the deque. That policy is correct, but the claimed
  invariant is not.
- Calling `max()` over the deque after maintaining decreasing order restores
  the `O(k)` lookup that this step removed. Read the maximum from the front.

## Derivation Summary

1. `left = right - k + 1` identifies the current window, and output begins at
   `right = k - 1`.
2. Scanning every complete window gives a correct `O(nk)` baseline.
3. Storing indices in a deque makes expiration exact, even for equal values.
4. A later greater-or-equal value permanently dominates an earlier one, so
   dominated tails can be removed.
5. The remaining values are strictly decreasing, making the unexpired front
   the window maximum in amortized `O(n)` time and `O(k)` auxiliary space.

### Checkpoint 4

**Freeze:** The reader can solve LeetCode 239 through the required
`maxSlidingWindow` contract in amortized `O(n)` time and `O(k)` auxiliary
space.

**It still lacks:** Nothing required by the problem. Only independent
full-draft review remains.
