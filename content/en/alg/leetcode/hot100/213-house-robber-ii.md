---
title: "LeetCode 213: House Robber II, Splitting the Circle into Two Linear Problems"
date: 2026-09-22T10:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "1D DP", "circular array", "house robber", "LeetCode 213"]
---

## Problem

### Input and Output

- Input: an integer array `nums`, where `nums[i]` is the money in the `i`-th house.
- The houses form a circle. Adjacent houses cannot both be robbed, so houses `0` and `n - 1` cannot both be robbed either.
- Output: the maximum amount of money that can be robbed without triggering the alarm.
- Constraints: `1 <= nums.length <= 100`, `0 <= nums[i] <= 1000`.

### Examples

```text
Input: nums = [2,3,2]
Output: 3
Explanation: Houses 0 and 2 are adjacent around the circle, so they cannot both be robbed; rob the house containing 3.
```

```text
Input: nums = [1,2,3,1]
Output: 4
Explanation: We can rob indices 0 and 2, or indices 1 and 3.
```

```text
Input: nums = [1,2,3]
Output: 3
```

This tutorial uses Python only and starts with the smallest first-last conflict.

## Step 1: Handle the Conflict Between the First and Last Houses

Start with one concrete question: for `nums = [2,3,2]`, can we reuse the straight-street approach and simply choose non-adjacent houses across the entire array?

The current baseline has only one rule: adjacent houses cannot both be robbed. If we treat these three houses as a line, we may consider indices `0` and `2` non-adjacent, choose both endpoints, and incorrectly get `4`.

That fails for a circular input because the first and last houses are also adjacent, so `0` and `2` must be mutually exclusive. To remove this conflict, add just one branching rule: every legal plan belongs to at least one of these candidate branches.

- **Exclude the last house**: consider only indices `0..n-2`, or `nums[0:n-1]`.
- **Exclude the first house**: consider only indices `1..n-1`, or `nums[1:n]`.

These ranges represent the "exclude the last house" and "exclude the first house" candidate branches. Each range is now a straight line. Together they cover every legal plan: because the two endpoints cannot both be robbed, every legal plan excludes at least one endpoint. A plan that excludes both endpoints may belong to both candidate ranges, but that does not affect taking the maximum at the end.

Check this split on the smallest example:

```text
[2,3,2]

Exclude the last house -> [2,3], best value in the range = 3
Exclude the first house -> [3,2], best value in the range = 3
Take the larger result -> 3
```

This version can:

- show why a circular street cannot be treated directly as a line
- rewrite the original problem as two candidate linear ranges that jointly cover every legal plan

It still needs:

- a way to compute the optimal value for any linear range
- boundary handling when a linear range contains only one or two houses

## Step 2: Let One Linear Range Handle One or Two Houses

Now consider just one candidate range produced by Step 1, such as `[2,3]`. We know it is a line, but we still do not have a function that can answer, "What is the most money we can rob from this range?"

The current baseline is only a description of two range boundaries: `[0..n-2]` or `[1..n-1]`. It tells us which portion to inspect, but it cannot store the best result already found for a prefix of that range.

We must first solve this for ranges containing only one or two houses:

- With one house, the only choice is that house.
- With two houses, we cannot rob both, so we take the larger amount.

Add only one linear helper to the previous two-range model, and first give it these two base cases:

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size

    # dp[k]: maximum money from nums[left..left+k]
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    return dp[1]
```

Here, `dp[k]` uses an offset within the range rather than an index in the original array: `dp[0]` corresponds to `nums[left]`, while `dp[1]` corresponds to `nums[left..left+1]`. The state is now operationally used by the assignments and returns in both base cases.

Check the smallest inputs:

```python
assert rob_line([5], 0, 0) == 5
assert rob_line([2, 3], 0, 1) == 3
```

This version can:

- compute the optimum for any linear range of length 1 or 2
- keep the meaning of `dp[k]` distinct from an index in the original array

It still needs:

- a way to derive the current state from the previous two states when there are three or more houses
- the "rob current or skip current" transition after these two base cases

## Step 3: Add the Rob-or-Skip Transition for a Linear Range

Now extend the range to `[2,7,9]`. The Step 2 version can initialize the first two houses, but it stops when the third house, `9`, arrives: it does not say which prefix result can be combined with `9`.

The current baseline is a `rob_line` that handles only lengths 1 and 2. It fails on three houses because there is no `dp[2]`, so it cannot answer whether to rob `9` or keep the best result from the first two houses. We continue to require `rob_line` to receive a non-empty range (`left <= right`); the one-house guard for the circular entry point will come later.

In the previous version, replace only the fixed `return dp[1]` at the end so that every later position compares two sources:

- Skip the current house and keep `dp[offset - 1]`.
- Rob the current house, which can only follow `dp[offset - 2]`, then add the current amount.

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size

    # dp[k]: maximum money from nums[left..left+k]
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])

    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)

    return dp[-1]
```

Check this transition:

```python
assert rob_line([2, 7, 9], 0, 2) == 11
assert rob_line([2, 7, 9, 3, 1], 0, 4) == 12
```

The states in the first assertion are `2 -> 7 -> max(7, 2 + 9) = 11`. The second assertion continues with `11 -> max(11, 7 + 3) = 11 -> max(11, 11 + 1) = 12`.

This version can:

- correctly compute the maximum amount for any non-empty linear range
- explicitly compare the two legal sources at each position: skip the current house or connect it to the state two positions back

It still needs:

- to connect this linear helper back to the circular array's two candidate branches
- special handling for `n == 1`, where we cannot construct the two exclusion ranges directly

## Step 4: Connect the Linear Helper Back to the Circular Array

Now return to the original circular input. Task 3 can solve one non-empty linear range, but there is still no entry point that evaluates both candidate ranges. Use `[2,3,2]` as the pressure: calling `rob_line` only once would omit either the "exclude the first house" or the "exclude the last house" possibility.

The current baseline is a correct `rob_line(nums, left, right)`. That is not yet enough for a circular array because the first-last conflict can only be removed through the two candidate branches. For now, require `n >= 2` at the entry point and leave the one-house input for the next step.

Add only one `rob` entry point around the previous helper. Compute the result that excludes the last house and the result that excludes the first house, then take the larger one. The linear transition inside `rob_line` remains unchanged.

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)
    return dp[-1]


def rob(nums: list[int]) -> int:
    n = len(nums)
    exclude_last = rob_line(nums, 0, n - 2)
    exclude_first = rob_line(nums, 1, n - 1)
    return max(exclude_last, exclude_first)
```

Check three circular examples with at least two houses:

```python
assert rob([2, 3, 2]) == 3
assert rob([1, 2, 3, 1]) == 4
assert rob([1, 2, 3]) == 3
```

This version can:

- fully compare both linear candidate branches for a circular array with `n >= 2`
- reuse the same linear helper instead of duplicating the transition logic

It still needs:

- an entry-point guard for `n == 1`, where both candidate ranges are not valid
- a space improvement, because the linear helper still stores the entire `dp` table and uses `O(n)` extra space

## Step 5: Add the One-House Boundary First

Now test Task 4 with the smallest input allowed by the constraints, `[5]`. The current `rob` immediately constructs two exclusion ranges, but when the length is `1`, there are not two valid non-empty ranges. The linear transition is not the problem; the entry point is missing a boundary check before the split.

The current baseline is correct for both candidate ranges when `n >= 2`. On `[5]`, it computes `right` as `-1`, so it cannot call `rob_line` safely.

Add only one early return to the previous complete code, and place it before both helper calls. The linear helper and circular split remain unchanged. The unchanged helper and updated entry point are shown together in one runnable block:

```python
def rob_line(nums: list[int], left: int, right: int) -> int:
    size = right - left + 1
    dp = [0] * size
    dp[0] = nums[left]
    if size == 1:
        return dp[0]

    dp[1] = max(nums[left], nums[left + 1])
    for offset in range(2, size):
        current = nums[left + offset]
        dp[offset] = max(dp[offset - 1], dp[offset - 2] + current)
    return dp[-1]


def rob(nums: list[int]) -> int:
    n = len(nums)
    if n == 1:
        return nums[0]

    exclude_last = rob_line(nums, 0, n - 2)
    exclude_first = rob_line(nums, 1, n - 1)
    return max(exclude_last, exclude_first)
```

Check the boundary and regular examples:

```python
assert rob([5]) == 5
assert rob([2, 3]) == 3
assert rob([2, 3, 2]) == 3
assert rob([1, 2, 3, 1]) == 4
assert rob([1, 2, 3]) == 3
```

This version can:

- cover every non-empty length in the constraints, including a circle with only one house
- keep comparing both linear candidate branches when `n >= 2`, giving the first complete correct solution
- run in `O(n)` time while still using `O(n)` extra space

It still needs:

- to avoid storing the entire `dp` array when the linear helper reads only the previous two states; the next step can reduce the extra space to `O(1)`

## Step 6: Keep Only the Two States Needed by the Linear Transition

Task 5 has produced the first correct solution, but each step in the linear helper reads only two old values: `dp[offset - 2]` and `dp[offset - 1]`. The rest of the `dp` table is never read again, so keeping it only consumes extra space.

The current baseline is the runnable array version. Its correctness already covers one-house, two-house, and circular examples, but its space complexity is `O(n)`. This step changes only the storage inside the linear helper. It does not change the two circular candidate ranges or the `n == 1` guard.

In the previous `rob_line`, use two rolling states to retain the two most recent prefix results:

- `prev2` is the previous round's `dp[offset - 2]`.
- `prev1` is the previous round's `dp[offset - 1]`.
- First compute `cur`, then move `prev1` into `prev2` and `cur` into `prev1`.

```python
class Solution:
    @staticmethod
    def rob_line(nums: list[int], left: int, right: int) -> int:
        size = right - left + 1
        prev2 = nums[left]
        if size == 1:
            return prev2

        prev1 = max(nums[left], nums[left + 1])
        for offset in range(2, size):
            current = nums[left + offset]
            cur = max(prev1, prev2 + current)
            prev2, prev1 = prev1, cur

        return prev1

    def rob(self, nums: list[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        exclude_last = self.rob_line(nums, 0, n - 2)
        exclude_first = self.rob_line(nums, 1, n - 1)
        return max(exclude_last, exclude_first)
```

Check the final version:

```python
solver = Solution()
assert solver.rob([2, 3, 2]) == 3
assert solver.rob([1, 2, 3, 1]) == 4
assert solver.rob([1, 2, 3]) == 3
assert solver.rob([5]) == 5
assert solver.rob([2, 7, 9, 3, 1]) == 11
```

The last example is circular: excluding the last house gives the linear range `[2,7,9,3]` with result `11`, while excluding the first house gives `[7,9,3,1]` with result `10`, so the answer is `11`.

The correctness invariant is: after processing an `offset`, `prev1` equals the optimum for that range prefix, while `prev2` equals the optimum for the preceding prefix. The next iteration can therefore still compute both "skip current" and "rob current" accurately. The two candidate ranges cover the first-last conflict, and the one-house guard covers the smallest input.

This version can:

- resolve the circular first-last conflict with two linear candidate ranges
- cover every non-empty input while preserving the same results as the array version
- run in `O(n)` time with `O(1)` extra space

This is the final incremental checkpoint for this problem. Only an independent review of the teaching chain and final code remains; there is no second reference answer that introduces new logic.
