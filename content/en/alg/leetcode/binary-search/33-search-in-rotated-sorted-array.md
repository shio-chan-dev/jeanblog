---
title: "LeetCode 33: Search in Rotated Sorted Array"
date: 2026-07-28T15:40:34+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "rotated array", "LeetCode 33"]
description: "Derive sorted-half recognition and target-range tests from a linear baseline to search a rotated array in O(log n) time and O(1) auxiliary space."
keywords: ["Search in Rotated Sorted Array", "rotated sorted array", "binary search", "LeetCode 33"]
---

## Problem Requirement

The input gives an integer array `nums` and an integer `target`. Every value in `nums` is distinct, and the array was strictly increasing before rotation. Before the method is called, it may be rotated at an unknown index `k` (`0 <= k < nums.length`) into:

```text
[nums[k], ..., nums[n-1], nums[0], ..., nums[k-1]]
```

Return the index of `target` in the rotated array when it exists; otherwise return `-1`. The problem requires an algorithm with `O(log n)` runtime.

LeetCode uses this method contract:

```text
class Solution:
    def search(self, nums: List[int], target: int) -> int:
```

### Official Examples

```text
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Input: nums = [1], target = 0
Output: -1
```

### Constraints

- `1 <= nums.length <= 5000`
- `-10^4 <= nums[i], target <= 10^4`
- Every value in `nums` is distinct.
- Before rotation, `nums` is sorted in strictly increasing order.

## Step 1: First Search Every Rotation Correctly

For `[4,5,6,7,0,1,2]`, how can we first obtain a correct answer that does not depend on the rotation position?

### Pressure

The array contains two increasing pieces, `[4,5,6,7]` and `[0,1,2]`, but the jump from `7` to `0` means the whole array is no longer in ordinary increasing order. The whole-array ordering premise used by ordinary binary search therefore does not survive the rotation, so that method cannot be applied unchanged.

### Previous Baseline

The current baseline contains only the problem input, output, official examples, constraints, and the `Solution.search` method contract. It has no executable search method yet.

### Break

The method contract says what to return, but it provides no executable process that remains correct for every valid rotation.

### Change

Add one linear scan to the previous baseline. `search_scan` checks each value in index order, returns the current index when it meets `target`, and returns `-1` if the scan ends without a match. Rotation changes where values appear, but it cannot make this element-by-element check skip the target.

```python
from typing import List


def search_scan(nums: List[int], target: int) -> int:
    for index, value in enumerate(nums):
        if value == target:
            return index
    return -1


assert search_scan([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_scan([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert search_scan([1], 0) == -1
assert search_scan([1, 3, 5, 7], 5) == 2
assert search_scan([9], 9) == 0
assert search_scan([9], 4) == -1
assert search_scan([6, 7, 1, 2, 3, 4, 5], 1) == 2
assert search_scan([6, 7, 1, 2, 3, 4, 5], 8) == -1
```

### Check

These fixed assertions cover all three official examples, an unrotated array, a singleton hit and miss, a target exactly at the rotation point, and a missing target. Running the whole block without an assertion failure shows that the scan handles the correctness problem created when rotation removes ordinary whole-array order.

### Complexity

For an array of length `n`, the worst case checks all `n` elements, so the time complexity is `O(n)`. The loop variables use no storage that grows with the input, so the auxiliary space complexity is `O(1)`.

### Step 1 Result

The scan returns the target index in every valid rotation and returns `-1` when the target is absent. Its answer is correct, but its `O(n)` runtime does not yet meet the requirement.

## Step 2: Which Half Is Still Sorted?

The linear scan is correct, but it does not use the local order that survives a rotation. To move toward logarithmic search, every current interval must contain a contiguous half that can be proved to retain ordinary increasing order.

### Pressure

In `[4,5,6,7,0,1,2]`, the whole array is not increasing, but the contiguous piece `[4,5,6,7]` still is. We now need evidence for an ordinarily sorted contiguous half inside the current interval instead of assuming that the entire array is sorted.

### Previous Baseline

The previous version is the correct `O(n)` linear scan. It checks each element, depends on no ordering, and therefore works for every valid rotation, but it may inspect the entire array.

### Break

The current interval may cross the rotation point, so the whole interval is not globally monotone. Knowing that the original array was strictly increasing is not enough to treat the entire current interval as an ordinary sorted array.

### Change: Classify the Sorted Half

Introduce a closed candidate interval `[left, right]`:

- `left` is the first included index in the current interval.
- `right` is the last included index in the current interval.
- `mid = left + (right - left) // 2` is the midpoint of the current interval.
- The left half is `[left, mid]`, and the right half is `[mid, right]`; both include `mid`.

For a valid rotated array, the left half is sorted if and only if `nums[left] <= nums[mid]`; otherwise, the right half is sorted.

This conclusion depends on two problem facts. A strictly increasing array rotated once has at most one break where a larger value is followed by a smaller value. Splitting one contiguous interval at `mid` cannot put that break inside both halves, so at least one half retains ordinary increasing order.

Distinct values make the endpoint comparison identify the break. If `[left, mid]` does not cross it, that half is strictly increasing and `nums[left] <= nums[mid]`. If it does cross the break, every value before the break is greater than every value after it, so `nums[left] > nums[mid]`. In the latter case, the only break is already in the left half, which guarantees that the right half is sorted.

The function below performs one classification and does not change `left` or `right`:

```python
from typing import List


def classify_sorted_half(nums: List[int], left: int, right: int) -> str:
    mid = left + (right - left) // 2
    if nums[left] <= nums[mid]:
        return "left"
    return "right"


assert classify_sorted_half([6, 7, 0, 1, 2, 4, 5], 0, 6) == "right"
assert classify_sorted_half([4, 5, 6, 7, 0, 1, 2], 0, 6) == "left"
assert classify_sorted_half([0, 1, 2, 4, 5, 6, 7], 0, 6) == "left"
```

### Check

The three assertions match the three shapes below. Each row computes the current midpoint and classifies a sorted half without changing the interval.

| Case | `nums` | `[left, right]` | `mid` | Endpoint comparison | Classification evidence and result |
| --- | --- | --- | ---: | --- | --- |
| Rotation point left of `mid` | `[6,7,0,1,2,4,5]` | `[0, 6]` | 3 | `6 <= 1` is false | Left half `[6,7,0,1]` crosses the rotation point; right half `[1,2,4,5]` is sorted |
| Rotation point right of `mid` | `[4,5,6,7,0,1,2]` | `[0, 6]` | 3 | `4 <= 7` is true | Left half `[4,5,6,7]` is sorted |
| Unrotated | `[0,1,2,4,5,6,7]` | `[0, 6]` | 3 | `0 <= 4` is true | Both halves are sorted; the rule classifies the left half |

In the first row, the rotation point is index `2`, left of `mid = 3`, so the left half crosses the break and the result is the right half. In the second row, the rotation point is index `4`, right of `mid = 3`, so the left half does not cross the break. The third row has no rotation break, and the endpoint comparison still identifies one sorted half consistently.

### Step 2 Result

For any current closed interval, the comparison can now determine whether the left half is sorted and, if not, establish that the right half is sorted. A sorted-half label alone, however, does not show which half can contain `target`.

## Step 3: Keep the Half That Can Contain the Target

A sorted-half label becomes useful only after it proves whether `target` lies inside that half's value range. Until that containment is established, no position can be discarded safely.

### Pressure

In `[4,5,6,7,0,1,2]`, the first midpoint is index `3`, and the left half `[4,5,6,7]` is sorted. Searching for `6` should keep the left side, while searching for `0` should keep the right side. The label "left half is sorted" alone supports neither interval reduction.

### Previous Baseline

The previous version uses the closed interval `[left, right]` for current candidate positions and recognizes the left half as sorted when `nums[left] <= nums[mid]`; otherwise it establishes that the right half is sorted. It does not yet compare `target` with the sorted half's endpoints.

### Break

No current rule proves which half contains the target, so no interval update yet guarantees that an existing `target` is preserved. Discarding either half before that proof may lose the answer.

### Change: Update the Closed Interval by Target Range

Maintain this candidate-interval invariant at the start of every iteration:

> If `target` exists in the array, its index is included in the current closed interval `[left, right]`.

Initialize the interval as `[0, len(nums) - 1]`, which includes every array index. After computing `mid`, first test `nums[mid] == target` and return immediately. The later range tests may exclude `mid` precisely because this equality check has already proved that the midpoint is not the target.

Now convert the Task 2 sorted-half classification into interval updates:

- If the left half is sorted, use `nums[left] <= target < nums[mid]` to test whether the target lies in its value range.
  - When true, keep `[left, mid - 1]` by setting `right = mid - 1`.
  - Otherwise, keep `[mid + 1, right]` by setting `left = mid + 1`.
- Otherwise the right half is sorted, so use `nums[mid] < target <= nums[right]` to test whether the target lies in its value range.
  - When true, keep `[mid + 1, right]` by setting `left = mid + 1`.
  - Otherwise, keep `[left, mid - 1]` by setting `right = mid - 1`.

Both value ranges exclude `nums[mid]` because the midpoint equality branch runs first. The closed interval endpoints remain included, so a target equal to `nums[left]` or `nums[right]` is not discarded.

### Final Implementation

Integrate only the earned midpoint equality, sorted-half classification, target-range tests, and boundary updates into the single LeetCode implementation:

```python
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid

            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1
```

### Why the Updates Preserve the Candidate Invariant

- **Initialization**: `[0, len(nums) - 1]` includes every valid index, so it includes an existing target.
- **Midpoint hit**: If `nums[mid] == target`, returning `mid` is correct and no further interval must be maintained.
- **Sorted left half**: Strict increase makes `nums[left] <= target < nums[mid]` an exact test for membership in `[left, mid - 1]`. When true, update `right`; otherwise, with the midpoint already excluded, an existing target must lie in `[mid + 1, right]`.
- **Sorted right half**: Similarly, `nums[mid] < target <= nums[right]` exactly tests membership in `[mid + 1, right]`. When true, update `left`; otherwise, an existing target must lie in `[left, mid - 1]`.
- **Absent target**: The invariant constrains only the case where the target exists. When it is absent, the interval still shrinks until the method returns `-1`.

This reasoning is scoped to the problem's distinct-value contract. Distinctness prevents ambiguity in Task 2's sorted-half comparison and makes each sorted-half range strict; no duplicate-value fallback outside the stated contract is added here.

Every non-matching iteration executes either `right = mid - 1` or `left = mid + 1`, so the new closed interval is strictly shorter than the old one. The loop must terminate when `left > right`.

### Check 1: Target in the Sorted Left Half

Search for `6` in `[4,5,6,7,0,1,2]`:

| `left` | `right` | `mid` | `nums[mid]` | Decision | Result |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | Left half sorted; `4 <= 6 < 7` | `right = 2` |
| 0 | 2 | 1 | 5 | Left half sorted, but `4 <= 6 < 5` is false | `left = 2` |
| 2 | 2 | 2 | 6 | Midpoint hit | Return `2` |

### Check 2: Target in the Sorted Right Half

Search for `4` in `[6,7,0,1,2,4,5]`:

| `left` | `right` | `mid` | `nums[mid]` | Decision | Result |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 1 | Right half sorted; `1 < 4 <= 5` | `left = 4` |
| 4 | 6 | 5 | 4 | Midpoint hit | Return `5` |

### Check 3: Target at the Rotation Point

Search for rotation-point value `0` in `[4,5,6,7,0,1,2]`:

| `left` | `right` | `mid` | `nums[mid]` | Decision | Result |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | Left half sorted, but `4 <= 0 < 7` is false | `left = 4` |
| 4 | 6 | 5 | 1 | Left half sorted; `0 <= 0 < 1` | `right = 4` |
| 4 | 4 | 4 | 0 | Midpoint hit | Return `4` |

### Check 4: Target Absent

Search for `3` in `[4,5,6,7,0,1,2]`:

| `left` | `right` | `mid` | `nums[mid]` | Decision | Result |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 6 | 3 | 7 | Left half sorted, but `4 <= 3 < 7` is false | `left = 4` |
| 4 | 6 | 5 | 1 | Left half sorted, but `0 <= 3 < 1` is false | `left = 6` |
| 6 | 6 | 6 | 2 | Singleton left half sorted, but `2 <= 3 < 2` is false | `left = 7` |

Now `left = 7 > right = 6`, so the candidate interval is empty and the method returns `-1`.

### Executable Verification

Run these checks after the earlier Task 1 `search_scan` block and the final `Solution` block. The fixed assertions cover the official examples, singleton cases, an unrotated array, the rotation point, two-element arrays, and a missing target. The seeded checks then generate strictly increasing arrays of lengths `1` through `40`, test every rotation, and confirm that search never mutates its input.

```python
from random import Random


solution = Solution()

assert solution.search([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert solution.search([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert solution.search([1], 0) == -1
assert solution.search([1], 1) == 0
assert solution.search([1, 3, 5, 7], 1) == 0
assert solution.search([1, 3, 5, 7], 7) == 3
assert solution.search([6, 7, 1, 2, 3, 4, 5], 1) == 2
assert solution.search([6, 7, 0, 1, 2, 4, 5], 4) == 5
assert solution.search([6, 7, 1, 2, 3, 4, 5], 8) == -1
assert solution.search([3, 1], 3) == 0
assert solution.search([3, 1], 1) == 1

nums = [4, 5, 6, 7, 0, 1, 2]
before = nums.copy()
assert solution.search(nums, 0) == 4
assert nums == before

rng = Random(33)
checked_cases = 0

for length in range(1, 41):
    original = sorted(rng.sample(range(-10_000, 10_001), length))
    missing = []

    while len(missing) < 3:
        candidate = rng.randint(-10_000, 10_000)
        if candidate not in original and candidate not in missing:
            missing.append(candidate)

    for rotation in range(length):
        rotated = original[rotation:] + original[:rotation]

        for target in original + missing:
            before = rotated.copy()
            assert solution.search(rotated, target) == search_scan(rotated, target)
            assert rotated == before
            checked_cases += 1

assert checked_cases == 24_600
```

### Complexity

Each iteration performs a constant number of comparisons and keeps only one side of the midpoint, reducing the candidate interval to at most about half its previous length. There are at most `O(log n)` iterations, so the time complexity is `O(log n)`. The algorithm stores only a constant number of variables such as `left`, `right`, and `mid`, does not modify the input, and therefore uses `O(1)` auxiliary space.

## Summary

Rotation removes ordinary order from the whole array, but every current interval still has at least one ordinarily sorted half around its midpoint. Check midpoint equality first, then use the sorted half's endpoints to decide whether `target` lies in its value range. Each boundary update preserves an existing target while strictly shrinking the candidate interval, producing the required `O(log n)` time and `O(1)` auxiliary-space search.
