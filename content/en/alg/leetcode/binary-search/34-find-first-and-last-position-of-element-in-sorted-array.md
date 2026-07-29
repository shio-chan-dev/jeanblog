---
title: "LeetCode 34: Find First and Last Position of Element in Sorted Array"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "sorted array", "boundary search", "LeetCode 34"]
description: "Start from a linear-scan baseline, derive two half-open binary-search boundaries, and return the complete target range in O(log n) time."
keywords: ["Find First and Last Position of Element in Sorted Array", "Search Range", "binary search", "boundary search", "sorted array", "LeetCode 34"]
---

## Problem Requirement

Start with the official input `nums = [5,7,7,8,8,10]` and `target = 8`. The answer must be the complete range `[3,4]`; returning only index `3` or `4` does not identify both the target's first and last occurrences.

Given an integer array `nums` sorted in non-decreasing order and an integer `target`:

- If `target` exists, return the indices of its first and last occurrences as `[first, last]`.
- If `target` does not exist, return `[-1, -1]`.
- The problem ultimately requires an algorithm with `O(log n)` runtime.

LeetCode uses this method contract:

```text
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
```

### Official Examples

```text
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Input: nums = [], target = 0
Output: [-1,-1]
```

### Constraints

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `-10^9 <= target <= 10^9`
- `nums` is sorted in non-decreasing order.

## Step 1: Get a Definitely Correct Range

When the target appears several times in a row, how can we guarantee that both the earliest and latest indices are recorded?

### Previous Baseline

The current baseline contains only the problem requirement, examples, constraints, and the `Solution.searchRange` method contract. It has no executable method yet.

### Break

Finding one element equal to `target` gives only one index. For `8` in `[5,7,7,8,8,10]`, that index could be `3` or `4`; neither one alone represents the complete range. The current baseline has no rule that guarantees both endpoints.

### Change

Add one complete scan to the previous baseline:

- `first` means the earliest match seen so far. It starts at `-1` and changes only on the first match.
- `last` means the latest match seen so far. It starts at `-1` and changes on every match.

With no match, both values remain `-1`. With duplicate targets, the first and final matches leave the two required endpoints.

```python
from typing import List


def search_range_scan(nums: List[int], target: int) -> List[int]:
    first = -1
    last = -1

    for index, value in enumerate(nums):
        if value == target:
            if first == -1:
                first = index
            last = index

    return [first, last]


assert search_range_scan([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert search_range_scan([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert search_range_scan([], 0) == [-1, -1]
assert search_range_scan([2, 2, 2, 2], 2) == [0, 3]
assert search_range_scan([7], 7) == [0, 0]
assert search_range_scan([7], 8) == [-1, -1]
assert search_range_scan([1, 2, 3], 1) == [0, 0]
assert search_range_scan([1, 2, 3], 3) == [2, 2]
```

### Check

The fixed assertions cover all three official examples, all-equal values, a singleton hit and miss, and targets at the first and last array positions. Running the whole block without an assertion failure shows that this change handles the two-endpoint pressure created by duplicate targets.

### Complexity

For an array of length `n`, this version checks every element, so its time complexity is `O(n)`. It stores only `first`, `last`, `index`, and `value`, so its auxiliary space complexity is `O(1)`.

### Step 1 Result

This version can now return the correct first and last positions with one scan, including the absent-target case.

It still lacks the required runtime: the `O(n)` scan does not satisfy the problem's `O(log n)` requirement.

## Step 2: Force the Search to Stop at the First Candidate

The linear scan is correct, but for an input of length `10^5`, it may inspect every element. The array is already sorted in non-decreasing order. How can we use that order to cut the search space in half each time?

### Previous Baseline

The previous version scans the whole array from left to right in `O(n)` time. It cannot miss a duplicate, but it does not use the fact that the array is sorted.

An ordinary binary search that checks only `nums[mid] == target` is not enough either. For `[5,7,7,8,8,10]` and `target = 8`, it may first hit index `4` and return immediately, even though index `3` also contains `8`. An equality hit finds some target occurrence; it does not force the search to stop at the leftmost candidate.

### Break

For now, reduce the problem to this smaller question:

> Find the first index satisfying `nums[i] >= target`. If every element is less than `target`, return `len(nums)`.

Because the array is sorted, the result of `nums[i] >= target` can change from `False` to `True` only once. For example:

```text
nums:       [5,    7,    7,    8,    8,    10]
>= 8:       F     F     F     T     T      T
                                 ^
                            first True
```

We are no longer looking for any element equal to `target`. We are looking for the boundary where this predicate first becomes `True`.

### Change: Maintain a Half-Open Search Interval

Use `[left, right)` for the actual array indices that have not been ruled out. The right side is excluded, so initialize it as:

```python
left = 0
right = len(nums)
```

Let `boundary` be the position we want. If no array element satisfies the predicate, `boundary` is `len(nums)`. At the start of every iteration, maintain this invariant:

- `left <= boundary <= right`.
- Every index less than `left` is known to fail the predicate, so its value is less than `target`.
- Every actual array index at least `right` is known to satisfy the predicate, so its value is not less than `target`.
- The unchecked actual array indices lie in the half-open interval `[left, right)`.

After choosing the midpoint, there are only two cases:

```python
mid = left + (right - left) // 2

if nums[mid] >= target:
    right = mid
else:
    left = mid + 1
```

When `nums[mid] >= target`, `mid` itself may be the first satisfying position. We cannot discard it, so set `right = mid`.

When `nums[mid] < target`, neither `mid` nor anything to its left can be the boundary, so set `left = mid + 1`.

Both updates preserve `left <= boundary <= right`. Also, whenever `left < right`, we have `left <= mid < right`, so either branch strictly decreases `right - left`. The loop must eventually stop with `left == right`; together with the invariant, this forces `left == boundary`.

Write that rule as the helper for this stage:

```python
from typing import List


def first_not_less(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] >= target:
            right = mid
        else:
            left = mid + 1

    return left


assert first_not_less([5, 7, 7, 8, 8, 10], 8) == 3
assert first_not_less([1, 3, 5], 4) == 2
assert first_not_less([], 0) == 0
assert first_not_less([2, 2, 2], 2) == 0
assert first_not_less([1, 3, 5], 0) == 0
assert first_not_less([1, 3, 5], 6) == 3
```

### Check 1: Duplicate Target

Trace `[5,7,7,8,8,10]` with `target = 8`:

| `left` | `right` | `mid` | `nums[mid] >= 8` | Interval after update |
| ---: | ---: | ---: | :---: | :--- |
| 0 | 6 | 3 | `True` | `[0, 3)` |
| 0 | 3 | 1 | `False` | `[2, 3)` |
| 2 | 3 | 2 | `False` | `[3, 3)` |

When the interval becomes empty, the helper returns `3`, the first index satisfying `nums[i] >= 8`. Even though the first checked index `3` already satisfies the predicate, the search keeps the left side and verifies that no earlier candidate exists.

### Check 2: Target Between Two Values

Trace `[1,3,5]` with `target = 4`:

| `left` | `right` | `mid` | `nums[mid] >= 4` | Interval after update |
| ---: | ---: | ---: | :---: | :--- |
| 0 | 3 | 1 | `False` | `[2, 3)` |
| 2 | 3 | 2 | `True` | `[2, 2)` |

The helper returns `2`, meaning that inserting `4` at index `2` would preserve sorted order. This result describes only an insertion boundary; it does not say that the value at index `2` equals `4`.

### Check 3: Empty Input

For an empty array, initialization gives `left = right = 0`. The loop never runs, and the helper returns `0` immediately. Here `0` also equals `len(nums)`, so it is still a valid insertion boundary.

### Step 2 Result

This version can now find the first position whose value is not less than `target` in `O(log n)` time. That position is also the insertion boundary that preserves sorted order.

It still lacks two things: the insertion position alone does not prove that `target` exists, and it does not locate the last position of a duplicate target block.

## Step 3: A Candidate Position Does Not Prove Existence

Consider `nums = [1,3,5]` and `target = 4`. Step 2's `first_not_less` returns `2` because index `2` is the first position where inserting `4` would preserve sorted order. But `nums[2]` is `5`; the array does not contain `4` at all.

How can we distinguish "the target could be inserted here" from "the target really starts here"?

### Previous Baseline

The current version has `first_not_less(nums, target)`. It returns the first index satisfying `nums[i] >= target`, or `len(nums)` if every value is less than `target`.

That is exactly an insertion boundary, but it is not yet a verified match.

### Break

Treating the insertion boundary as the start creates two kinds of failure:

1. For `[1,3,5]` and `target = 4`, `start = 2` is still inside the array, but `nums[2] != 4`.
2. For `[1,3,5]` and `target = 6`, `start = 3`, exactly equal to `len(nums)`; reading `nums[start]` would be out of bounds. An empty array has the same problem.

The validation order therefore cannot be reversed. We must first determine whether `start` equals the array length. Only when it is still an actual index may we read `nums[start]`.

### Change: Rule Out an Out-of-Bounds Position Before Reading

Reuse `first_not_less` from Step 2 and add this condition after it returns:

```python
if start == len(nums) or nums[start] != target:
    return [-1, -1]
```

Python evaluates `or` from left to right and short-circuits:

- If `start == len(nums)` is true, the whole condition is already true, so `nums[start]` is not evaluated.
- Only when `start < len(nums)` does Python continue to test `nums[start] != target`.
- When both conditions are false, `start` is an actual index and `nums[start] == target`.

Attach this check to the helper from the previous step:

```python
from typing import List


def verified_start_or_absent(nums: List[int], target: int):
    start = first_not_less(nums, target)

    if start == len(nums) or nums[start] != target:
        return [-1, -1]

    return start


assert verified_start_or_absent([1, 3, 5], 0) == [-1, -1]
assert verified_start_or_absent([1, 3, 5], 6) == [-1, -1]
assert verified_start_or_absent([1, 3, 5], 4) == [-1, -1]
assert verified_start_or_absent([], 0) == [-1, -1]
assert verified_start_or_absent([1, 2, 2, 2, 4], 2) == 1
```

This intermediate helper is not the complete LeetCode method. For an absent target, it follows the problem contract by returning `[-1, -1]` early. For a present target, it exposes only the verified `start`; it does not fabricate the other index that has not been derived yet.

### Check: What Each Part of the Guard Prevents

| Input | `first_not_less` result | Validation result |
| :--- | ---: | :--- |
| `[1,3,5]`, `target = 0` | `0` | `nums[0]` is `1`, so return `[-1, -1]` |
| `[1,3,5]`, `target = 6` | `3` | `start == len(nums)`, so short-circuit and return `[-1, -1]` |
| `[1,3,5]`, `target = 4` | `2` | `nums[2]` is `5`, so return `[-1, -1]` |
| `[]`, `target = 0` | `0` | `start == len(nums)`, so short-circuit and return `[-1, -1]` |
| `[1,2,2,2,4]`, `target = 2` | `1` | `nums[1] == 2`, so preserve the verified start index `1` |

The last row also shows why we continue to reuse `first_not_less`: after validation, its index not only contains the target but remains the target's first occurrence.

### Step 3 Result

This version can now safely distinguish an insertion position from a real match: it returns `[-1, -1]` when the target is absent and produces a verified first occurrence when the target is present.

It still lacks the last occurrence of a present target, so it cannot return the complete range yet.

## Step 4: Find the First Position After the Target Block

Step 3 verifies `start`, but it tells us only where the target block begins, not how far that block extends.

A direct approach is to scan right from `start` until the value changes:

```python
last = start
while last + 1 < len(nums) and nums[last + 1] == target:
    last += 1
```

If all `n` elements equal `target`, this scan still walks through almost the whole array, giving `O(n)` worst-case time. That would lose the `O(log n)` advantage already earned from binary search.

### Break

Instead of walking from inside the target block to its end, we need another boundary that can be found with binary search:

> Find the first index satisfying `nums[i] > target`. If there is no greater element, return `len(nums)`.

For `[5,7,7,8,8,10]` and `target = 8`, the predicate looks like this:

```text
nums:       [5,    7,    7,    8,    8,    10]
> 8:        F     F     F     F     F      T
                                             ^
                                         first True
```

This position is `5`. The target block occupies the consecutive indices immediately before it, so the target's last index is:

```text
end = first_greater - 1 = 5 - 1 = 4
```

### Change: Move Right When the Value Equals the Target

The second search still uses the half-open interval `[left, right)`, but its predicate must be the strict condition `nums[mid] > target`:

```python
mid = left + (right - left) // 2

if nums[mid] > target:
    right = mid
else:
    left = mid + 1
```

Let `greater_boundary` be the first position whose value is greater than `target`, or `len(nums)` if there is no such value. At the start of each iteration, maintain:

- `left <= greater_boundary <= right`.
- Every index less than `left` is known to satisfy `nums[i] <= target`.
- Every actual array index at least `right` is known to satisfy `nums[i] > target`.
- The unchecked actual array indices lie in `[left, right)`.

When `nums[mid] > target`, `mid` may be the first greater value, so keep it by setting `right = mid`. Otherwise `nums[mid] <= target`; this includes every position equal to `target`, and none of those positions can be the first greater value, so set `left = mid + 1`.

That is exactly how this search differs from `first_not_less`:

- The first search treats `nums[mid] == target` as a true predicate, applies `right = mid`, and keeps that position while looking for an earlier candidate.
- The second search treats `nums[mid] == target` as a false predicate, applies `left = mid + 1`, and discards that position while looking farther right.

Either branch strictly decreases `right - left`. The loop terminates with `left == right`, and the invariant then gives `left == greater_boundary`.

### Check: Both Boundaries on the Official Duplicate Example

First find the first position satisfying `nums[i] >= 8`:

| `left` | `right` | `mid` | Test | Interval after update |
| ---: | ---: | ---: | :--- | :--- |
| 0 | 6 | 3 | `8 >= 8` | `[0, 3)` |
| 0 | 3 | 1 | `7 < 8` | `[2, 3)` |
| 2 | 3 | 2 | `7 < 8` | `[3, 3)` |

The first search returns `start = 3`.

Then find the first position satisfying `nums[i] > 8`:

| `left` | `right` | `mid` | Test | Interval after update |
| ---: | ---: | ---: | :--- | :--- |
| 0 | 6 | 3 | `8 <= 8` | `[4, 6)` |
| 4 | 6 | 5 | `10 > 8` | `[4, 5)` |
| 4 | 5 | 4 | `8 <= 8` | `[5, 5)` |

The second search returns `first_greater = 5`, so `end = 5 - 1 = 4` and the final range is `[3,4]`.

### Assemble the Final LeetCode Method

The code below expands the already verified first-boundary search directly inside `searchRange`, keeps the safe existence guard, and then adds the second boundary search. The two loops are not merged into one configurable helper because their important distinction is the visible `>=` versus `>` predicate.

```python
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums)

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1

        start = left

        if start == len(nums) or nums[start] != target:
            return [-1, -1]

        left = 0
        right = len(nums)

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1

        first_greater = left
        end = first_greater - 1
        return [start, end]


solution = Solution()

assert solution.searchRange([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert solution.searchRange([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert solution.searchRange([], 0) == [-1, -1]
assert solution.searchRange([2, 2, 2, 2], 2) == [0, 3]
assert solution.searchRange([7], 7) == [0, 0]
assert solution.searchRange([7], 8) == [-1, -1]
assert solution.searchRange([1, 2, 3], 1) == [0, 0]
assert solution.searchRange([1, 2, 3], 3) == [2, 2]

unchanged = [5, 7, 7, 8, 8, 10]
snapshot = unchanged.copy()
assert solution.searchRange(unchanged, 8) == [3, 4]
assert unchanged == snapshot
```

### Randomized Differential Check

Fixed examples can miss combinations of boundaries. Reuse `search_range_scan` from Step 1 as a correct but slower baseline, generate sorted arrays with a fixed random seed, and compare the two versions. Also verify that every call leaves its input array unchanged.

In the same Python session, run the Step 1 baseline and the final implementation above, then run:

```python
import random


rng = random.Random(34)
solution = Solution()

for _ in range(1000):
    length = rng.randint(0, 50)
    nums = sorted(rng.randint(-10, 10) for _ in range(length))
    target = rng.randint(-12, 12)
    snapshot = nums.copy()

    assert solution.searchRange(nums, target) == search_range_scan(nums, target)
    assert nums == snapshot
```

### Correctness Proof

**The start is correct.** The first loop always keeps the first boundary satisfying `nums[i] >= target` between `left` and `right`. At termination, `start` is that boundary, so every value before `start` is less than `target`.

**The absence check is correct and safe.** If `start == len(nums)`, no array value is at least `target`, so the target is absent, and short-circuit evaluation avoids an out-of-bounds read. If `start < len(nums)` but `nums[start] != target`, the boundary definition gives `nums[start] > target`, while every earlier value is less than `target`, so the target is also absent. Conversely, when the guard does not return, `nums[start] == target`, and every earlier position is smaller, so `start` is the first occurrence.

**The end is correct.** The second loop returns `first_greater`, the first position satisfying `nums[i] > target`. We already know `nums[start] == target`. Because the array is sorted, every value from `start` through `first_greater - 1` is both at least and at most `target`, so all of them equal the target. Since `first_greater` is the first boundary beyond the target, `end = first_greater - 1` is exactly the last occurrence.

Together, these three facts make the method return `[start, end]` when the target exists and `[-1, -1]` otherwise.

### Complexity

Each half-open binary search takes `O(log n)` time. The constant number of checks and calculations takes `O(1)` time, so the total time complexity is `O(log n)`. The algorithm uses only a fixed number of integer variables and allocates no input-sized auxiliary structure, so its auxiliary space complexity is `O(1)`.

## Summary

- The linear scan first provides a correct baseline against which the optimized result can be checked, but it requires `O(n)` time in the worst case.
- The first position satisfying `nums[i] >= target` gives the candidate start; the safe length and equality guard determines whether the target exists.
- The first position satisfying `nums[i] > target` lies after the target block, so subtracting one gives the last occurrence.
- Both searches use `[left, right)`, but equality is handled differently: the first search keeps the current index and continues left, while the second discards it and continues right.
- The final `Solution.searchRange` returns the complete range in `O(log n)` time and `O(1)` auxiliary space without modifying the input array.
