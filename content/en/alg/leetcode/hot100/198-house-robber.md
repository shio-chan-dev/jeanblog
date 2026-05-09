---
title: "LeetCode 198: House Robber, Deriving 1D DP from Rob or Skip"
date: 2026-05-03T14:33:39+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "1D DP", "house robber", "LeetCode 198"]
---

## Problem

### Input and Output

- Input: an integer array `nums`
- `nums[i]` is the money in the `i`-th house
- Adjacent houses cannot both be robbed
- Output: return the maximum amount of money that can be robbed
- Constraints: `1 <= nums.length <= 100`, `0 <= nums[i] <= 400`

### Examples

```text
Input: nums = [1,2,3,1]
Output: 4
Explanation: rob indices 0 and 2, for 1 + 3 = 4
```

```text
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: rob indices 0, 2, and 4, for 2 + 9 + 1 = 12
```

This article uses Python only and derives 1D DP from the conflict between two choices.

## Start from the Adjacent Conflict in [1,2,3,1]

Look at the example:

```text
nums = [1,2,3,1]
```

If we rob the house at index `2`, whose money is `3`, then indices `1` and `3` cannot be robbed.
If we do not rob index `2`, the answer may come from the best result among indices `0..1`.

So when we reach a house, the core choice has only two cases:

- rob the current house: we must skip the previous house
- skip the current house: we reuse the best result before it

## Step 1: Define the Best Value Up to House i

Asking "how much can we rob from the whole street" is too large at first. Define:

```text
dp[i] = maximum money we can rob considering houses 0..i
```

This definition has one benefit: when processing house `i`, the previous prefix already has a stable optimal value.

Start with the smallest skeleton:

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
```

This version can:

- reserve one optimal value for each prefix `0..i`
- make it clear that `dp[i]` means "up to house i", not "must rob house i"

It still needs:

- the base cases for the first one or two houses
- the transition between robbing and skipping the current house

## Step 2: Handle the First and Second Houses

Build on the previous version and fill the base cases.

If we only look at house `0`, the most we can rob is that house:

```python
dp[0] = nums[0]
```

If we look at houses `0..1`, adjacent houses cannot both be robbed, so we take the larger one:

```python
if n == 1:
    return dp[0]

dp[1] = max(nums[0], nums[1])
```

The current code is:

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    if n == 1:
        return dp[0]

    dp[1] = max(nums[0], nums[1])
```

This version can:

- handle `n = 1` correctly
- handle the adjacent conflict when there are only two houses

It still needs:

- how to decide "rob or skip" when `i >= 2`

## Step 3: House i Has Only Two Choices

Now consider `i >= 2`.

If we skip house `i`, the best value is:

```text
dp[i - 1]
```

If we rob house `i`, then house `i - 1` cannot be robbed, so we can only add:

```text
dp[i - 2] + nums[i]
```

Add this transition to the previous version:

```python
for i in range(2, n):
    dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
```

The first complete correct version is:

```python
def rob(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    if n == 1:
        return dp[0]

    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[n - 1]
```

This version can:

- compute the maximum amount for every prefix
- make each step explicitly compare "skip current" and "rob current"

It still has room for:

- space optimization, because each state only depends on the previous two prefix states

## Step 4: Walk Through the Table Slowly

For `nums = [2,7,9,3,1]`:

| i | nums[i] | Skip current dp[i - 1] | Rob current dp[i - 2] + nums[i] | dp[i] |
| --- | --- | --- | --- | --- |
| 0 | 2 | - | 2 | 2 |
| 1 | 7 | - | - | 7 |
| 2 | 9 | 7 | 2 + 9 = 11 | 11 |
| 3 | 3 | 11 | 7 + 3 = 10 | 11 |
| 4 | 1 | 11 | 11 + 1 = 12 | 12 |

The answer is:

```text
dp[4] = 12
```

The important point in this table is:

> `dp[i]` does not require house i to be robbed. It means the global optimum up to house i.

## Step 5: Compress the dp Array into Two Variables

The previous transition only depends on:

- `dp[i - 2]`
- `dp[i - 1]`

So keep two variables:

- `prev2`: the previous round's `dp[i - 2]`
- `prev1`: the previous round's `dp[i - 1]`

Replace the array with:

```python
prev2 = nums[0]
prev1 = max(nums[0], nums[1])

for i in range(2, n):
    cur = max(prev1, prev2 + nums[i])
    prev2 = prev1
    prev1 = cur
```

The final complete code is:

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        n = len(nums)

        if n == 1:
            return nums[0]

        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])

        for i in range(2, n):
            cur = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = cur

        return prev1


if __name__ == "__main__":
    print(Solution().rob([1, 2, 3, 1]))
    print(Solution().rob([2, 7, 9, 3, 1]))
```

This version can:

- keep the same state meaning as the `dp` table version
- reduce extra space from `O(n)` to `O(1)`

It still has one boundary:

- if the houses form a circle, the problem becomes `213. House Robber II`, and we need to split it into two linear ranges; this problem is a straight line

## Correctness

Invariants:

- after processing house `i`, `prev1` equals `dp[i]`, the maximum money among indices `0..i`
- `prev2` equals the previous round's `dp[i - 1]`, used for the next transition

Why the transition is correct:

- Any optimal plan for house `i` has only two cases: rob it or skip it.
- If we skip house `i`, the best value is the prefix optimum `dp[i - 1]`.
- If we rob house `i`, house `i - 1` cannot be robbed, so the value is `dp[i - 2] + nums[i]`.
- These two cases cover all legal plans, so take the maximum.

Why return `prev1`:

- after the loop, `prev1` corresponds to the prefix optimum at the last index
- that is exactly the maximum money for the whole street

## Complexity

- Time complexity: `O(n)`
- Extra space: `O(1)`

## Common Mistakes

- Defining `dp[i]` as "must rob house i" but writing a transition like `max(dp[i - 1], ...)`, which conflicts with the state meaning.
- Forgetting to handle `n = 1`.
- When robbing the current house, adding `dp[i - 1]`, which violates the adjacent-house constraint.
- Mixing this problem with the circular House Robber problem; in this problem, the first and last houses are not adjacent.

## Summary

- `dp[i]` means the maximum money after considering houses up to index `i`.
- Each position has only two decisions: rob current or skip current.
- Robbing current must connect to `dp[i - 2]`; skipping current is `dp[i - 1]`.
- Space optimization only keeps the previous two DP states.

## References and Follow-up

- LeetCode 198: House Robber
- LeetCode 213: House Robber II
- LeetCode 70: Climbing Stairs
- LeetCode 746: Min Cost Climbing Stairs

## Notes

- The problem statement, examples, and constraints are based on the public LeetCode 198 summary.
- Python is used to match the current LeetCode tutorial style in this repository.
