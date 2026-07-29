---
title: "LeetCode 153: Find Minimum in Rotated Sorted Array"
date: 2026-07-28T16:13:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "rotated array", "LeetCode 153"]
description: "Start from a linear scan, derive a closed minimum-containing interval, and return the rotated array's minimum in O(log n) time and O(1) auxiliary space."
keywords: ["Find Minimum in Rotated Sorted Array", "binary search", "rotated array", "minimum value", "LeetCode 153"]
---

## Problem Requirement

The input is a non-empty integer array `nums`. Every element is unique, and before rotation the array was sorted in strictly increasing order.

The array is rotated between `1` and `nums.length` times. One rotation moves the last element to the front; therefore, `nums.length` rotations restore the original increasing order. Return the minimum value in the rotated array. The problem requires an algorithm with `O(log n)` runtime.

LeetCode uses this method contract:

```text
class Solution:
    def findMin(self, nums: List[int]) -> int:
```

### Official Examples

```text
Input: nums = [3,4,5,1,2]
Output: 1

Input: nums = [4,5,6,7,0,1,2]
Output: 0

Input: nums = [11,13,15,17]
Output: 11
```

### Constraints

- `1 <= nums.length <= 5000`
- `-5000 <= nums[i] <= 5000`
- All integers in `nums` are unique.
- Before rotation, `nums` is sorted in strictly increasing order.
- `nums` is rotated between `1` and `nums.length` times.

## Step 1: First Get a Definitely Correct Minimum

For `[3,4,5,1,2]`, why can we not simply return the first element?

### Pressure

The first element is `3`, but the minimum is `1`, which moved into the middle after rotation. Rotation preserves every value but does not guarantee that the minimum remains at index `0`, so returning `nums[0]` gives the wrong answer.

### Previous Baseline

The current baseline contains only the non-empty input, strict pre-rotation order, unique elements, required minimum, official examples, constraints, and the `Solution.findMin` method contract. It has no executable rule for finding the minimum.

### Break

The method contract says what to return, but it provides no executable process that handles every valid rotation, including the original order restored after exactly `nums.length` rotations.

### Change

Add one linear scan to the previous baseline. Because the problem guarantees that `nums` is non-empty, `find_min_scan` can initialize the running `minimum` from `nums[0]` and then visit every remaining value. When it sees a smaller value, it updates `minimum`; after the scan, it returns that value.

```python
from typing import List


def find_min_scan(nums: List[int]) -> int:
    minimum = nums[0]

    for index in range(1, len(nums)):
        if nums[index] < minimum:
            minimum = nums[index]

    return minimum


assert find_min_scan([3, 4, 5, 1, 2]) == 1
assert find_min_scan([4, 5, 6, 7, 0, 1, 2]) == 0
assert find_min_scan([11, 13, 15, 17]) == 11
assert find_min_scan([7]) == 7
assert find_min_scan([4, 1, 2, 3]) == 1
assert find_min_scan([1, 2, 3, 4]) == 1
```

### Check

The first three assertions cover all official examples. The final three check a singleton, one rotation that moves the last element to the front, and the original increasing order produced after `nums.length` rotations. Running the entire block without an assertion failure shows that the scan does not depend on where the minimum appears after rotation.

### Correctness

After initialization, `minimum` is the minimum of the first element. For each remaining element, the scan replaces `minimum` when the new value is smaller and otherwise keeps it. After every iteration, `minimum` is therefore the smallest value visited so far. Once every element has been visited, it is the minimum of the whole array.

### Complexity

For an array of length `n`, this version visits every element once, so its time complexity is `O(n)`. It stores only `minimum` and the loop index, so its auxiliary space complexity is `O(1)`.

### Step 1 Result

This version now returns the correct minimum for every valid non-empty input, including a singleton, one rotation, and exactly `nums.length` rotations.

It still lacks the required runtime: the `O(n)` linear scan does not satisfy the problem's `O(log n)` requirement.

## Step 2: Which Side Can Still Contain the Minimum?

If we stop inspecting every value, one decision must discard some positions while proving that the minimum remains among the positions we keep. How can we do both?

### Pressure

The linear scan takes `O(n)` time because it never rules out an unvisited value. To beat the scan, every step must discard some values. Shrinking alone is not enough, however: the smaller range must still contain the true minimum, or every later decision loses its correctness foundation.

### Previous Baseline

The previous version uses `find_min_scan` to visit every element and maintains the running `minimum` for correctness. It records neither which positions can still be the minimum nor any proved rule for discarding positions.

### Break

A tempting approach is to compare `nums[mid]` with the first array element, but that comparison alone does not give a stable update rule for the current range. For both `[4,5,6,7,0,1,2]` and `[11,13,15,17]`, the initial midpoint is greater than the first element, yet the former has its minimum to the midpoint's right while the latter has its minimum to the midpoint's left. Once the range's left endpoint moves, the original first element may also be outside the current range.

We therefore need a comparison reference that always belongs to the current range and a direct proof that each update keeps the minimum.

### Change: Maintain a Closed Minimum-Containing Interval

Use the closed interval `[left, right]` for the indices that can still contain the minimum. Both endpoints are included. Its meaning is:

> The true minimum's index is guaranteed to be inside `[left, right]`.

The complete array gives the initial range `[0, len(nums) - 1]`. A one-step shrink applies only to a range with at least two positions, so `left < right`. Its midpoint is:

```text
mid = left + (right - left) // 2
```

This gives `mid < right`. The problem guarantees that all values are unique, so `nums[mid]` and `nums[right]` cannot be equal. Only the following two cases remain.

#### `nums[mid] > nums[right]`

If the values from `mid` through `right` still followed ordinary increasing order, then `nums[mid] < nums[right]` would hold. The opposite relation shows that the drop created by rotation lies in this part of the range, and the array's minimum comes after that drop. The minimum must therefore be in `[mid + 1, right]`; neither `mid` nor any current position to its left can be the minimum. We can execute:

```text
left = mid + 1
```

#### `nums[mid] < nums[right]`

Now the values from `mid` through `right` do not cross the drop created by rotation, so the minimum is not in `[mid + 1, right]`. It remains in `[left, mid]`, and `mid` itself may be the minimum. For example, in the current range `[3,4]` of `[3,4,5,1,2]`, `mid = 3` and its value `1` is the minimum. We must keep `mid` and execute:

```text
right = mid
```

The helper below performs exactly one comparison and one range update. It returns the updated closed interval; it contains neither a loop nor a rule for returning the final answer.

```python
from typing import List, Tuple


def shrink_minimum_interval_once(
    nums: List[int], left: int, right: int
) -> Tuple[int, int]:
    mid = left + (right - left) // 2

    if nums[mid] > nums[right]:
        left = mid + 1
    else:
        right = mid

    return left, right


first = [4, 5, 6, 7, 0, 1, 2]
first_interval = shrink_minimum_interval_once(first, 0, 6)
assert first_interval == (4, 6)
assert min(first) in first[first_interval[0] : first_interval[1] + 1]

second = [3, 4, 5, 1, 2]
second_interval = shrink_minimum_interval_once(second, 0, 4)
assert second_interval == (3, 4)
assert min(second) in second[second_interval[0] : second_interval[1] + 1]

second_interval = shrink_minimum_interval_once(
    second, second_interval[0], second_interval[1]
)
assert second_interval == (3, 3)
assert min(second) in second[second_interval[0] : second_interval[1] + 1]

third = [11, 13, 15, 17]
third_interval = shrink_minimum_interval_once(third, 0, 3)
assert third_interval == (0, 1)
assert min(third) in third[third_interval[0] : third_interval[1] + 1]
```

### Check

Every call in the code matches one manual trace below. The final column directly checks that the next closed interval still contains the true minimum.

| Array | Current range | `mid` | Comparison | Next range | Evidence that the minimum remains |
| --- | --- | ---: | --- | --- | --- |
| `[4,5,6,7,0,1,2]` | `[0,6]` | `3` | `7 > 2` | `[4,6]` | `[0,1,2]` contains `0` |
| `[3,4,5,1,2]` | `[0,4]` | `2` | `5 > 2` | `[3,4]` | `[1,2]` contains `1` |
| `[3,4,5,1,2]` | `[3,4]` | `3` | `1 < 2` | `[3,3]` | Keeps minimum `1` at `mid` |
| `[11,13,15,17]` | `[0,3]` | `1` | `13 < 17` | `[0,1]` | `[11,13]` contains `11` |

The first two greater-than branches move `left` to `mid + 1`, excluding the midpoint and current positions to its left because none can be the minimum. The final two less-than branches move `right` to `mid`; the third row concretely proves that discarding `mid` would discard the answer itself.

### Step 2 Result

This version can now use the comparison between `nums[mid]` and the current `nums[right]` to shrink one closed interval safely while guaranteeing that the true minimum remains in the next range.

It still lacks a complete loop that assembles these updates, a termination condition, and a final return rule.

## Step 3: When Does the Candidate Interval Become the Answer?

The two interval updates can keep the minimum safely, but how do we repeat them and return the answer at the right time?

### Pressure

`shrink_minimum_interval_once` performs only one step. It does not say when to continue, when to stop, or how to turn the final retained index into the minimum value required by the problem. One safe update is not yet a submit-ready `Solution.findMin`.

### Previous Baseline

The previous version maintains a closed interval `[left, right]` and guarantees that the true minimum's index remains inside it. It has also proved both updates:

- Execute `left = mid + 1` when `nums[mid] > nums[right]`.
- Otherwise execute `right = mid`; uniqueness means that the actual relation is `nums[mid] < nums[right]`.

### Break

Without a loop condition, we do not know how many updates to perform. Without a strict-progress proof, a loop might not finish. Without a meaning for the final state, even stopped updates do not tell us which value to return.

### Change: Loop Until One Index Remains

In the previous version, place the two proved updates inside `while left < right`. Recompute `mid` from the current endpoints in every iteration. When the condition becomes false, the closed interval contains only the index where `left == right`. Because the interval always retains the minimum, return `nums[left]`.

The following is the article's only final `Solution.findMin` implementation. The fixed assertions and randomized differential checks reuse Step 1's `find_min_scan` as the correctness baseline, so running this article's Python blocks in order executes every check.

```python
from random import Random
from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        return nums[left]


solution = Solution()

assert solution.findMin([3, 4, 5, 1, 2]) == 1
assert solution.findMin([4, 5, 6, 7, 0, 1, 2]) == 0
assert solution.findMin([11, 13, 15, 17]) == 11
assert solution.findMin([7]) == 7
assert solution.findMin([4, 1, 2, 3]) == 1
assert solution.findMin([1, 2, 3, 4]) == 1

nums = [4, 5, 6, 7, 0, 1, 2]
snapshot = nums.copy()
assert solution.findMin(nums) == 0
assert nums == snapshot

rng = Random(153)

for length in range(1, 33):
    original = sorted(rng.sample(range(-5000, 5001), length))

    for rotations in range(1, length + 1):
        rotated = original[-rotations:] + original[:-rotations]
        snapshot = rotated.copy()

        assert solution.findMin(rotated) == find_min_scan(rotated)
        assert rotated == snapshot
```

### Strict Progress

At the start of an iteration, `left < right`, so the midpoint satisfies `left <= mid < right`.

- With `left = mid + 1`, the new `left` is strictly greater than the old `left` and does not exceed `right`.
- With `right = mid`, the new `right` is strictly less than the old `right` and is not less than `left`.

Both branches strictly shorten the closed interval while preserving `left <= right`. Its length is finite, so the loop must terminate.

### Correctness Proof

The loop maintains this invariant:

> At the start of every iteration, the true minimum's index is inside the closed interval `[left, right]`.

**Initialization:** The problem guarantees a non-empty array. The initial interval `[0, len(nums) - 1]` contains every index, so it contains the minimum's index.

**Preservation:** If `nums[mid] > nums[right]`, Step 2 proved that the minimum lies in `[mid + 1, right]`, so `left = mid + 1` preserves the invariant. Otherwise, uniqueness makes the actual relation `nums[mid] < nums[right]`; Step 2 proved that the minimum lies in `[left, mid]`, so `right = mid` also preserves the invariant.

**Termination and return:** Strict progress guarantees termination. At termination, `left == right`, while the invariant still places the minimum's index inside `[left, right]`. This interval has one index, so that index must hold the minimum, and returning `nums[left]` is correct.

### Slow-Branch Trace

For the unrotated array `[11,13,15,17]`, every iteration takes the `right = mid` branch that retains `mid`:

| `[left, right]` | `mid` | Comparison | Updated interval |
| --- | ---: | --- | --- |
| `[0,3]` | `1` | `13 < 17` | `right = mid = 1`, giving `[0,1]` |
| `[0,1]` | `0` | `11 < 13` | `right = mid = 0`, giving `[0,0]` |

Now `left == right == 0`. The invariant says that the sole retained index is the minimum's index, so the method returns `nums[0] == 11`.

### Boundary Cases and Checks

- **Singleton:** Initially `left == right == 0`, so the loop is skipped and the sole element is returned.
- **Unrotated or fully rotated:** The current range remains strictly increasing, so every iteration has `nums[mid] < nums[right]`. Moving `right` to `mid` eventually retains index `0`.
- **Fixed assertions:** Six assertions cover all official examples, a singleton, one rotation, and a full rotation.
- **Randomized differential:** Seed `153` generates 32 distinct increasing arrays and checks every rotation from `1` through `length`, for 528 inputs total. The final implementation must match the Step 1 linear scan on each input.
- **Input immutability:** The fixed case and every randomized case compare the array before and after the call, confirming that the method does not modify `nums`.

### Complexity

Let the current closed interval contain `m = right - left + 1` candidates. With the midpoint rounded down, either update retains at most `ceil(m / 2)` candidates for the next iteration. The candidate count therefore falls from `n` to `1` in `O(log n)` time. The algorithm stores only `left`, `right`, and `mid`, so its auxiliary space complexity is `O(1)`.

### Summary

The linear scan first supplies a correct baseline independent of the rotation position. The closed minimum-containing interval then turns the comparison between `nums[mid]` and the current `nums[right]` into two safe updates. Finally, strict progress reduces the interval to the unique minimum index, returning the answer in `O(log n)` time and `O(1)` auxiliary space without modifying the input array.
