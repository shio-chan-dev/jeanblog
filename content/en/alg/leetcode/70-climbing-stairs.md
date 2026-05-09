---
title: "LeetCode 70: Climbing Stairs, Deriving 1D DP from dp[i]"
date: 2026-05-03T14:33:39+08:00
draft: false
categories: ["LeetCode"]
tags: ["dynamic programming", "1D DP", "climbing stairs", "LeetCode 70"]
---

## Problem

### Input and Output

- Input: an integer `n`
- Meaning: climb to the top of the `n`-step staircase
- Each move can climb `1` or `2` steps
- Output: return the number of distinct ways to reach the top
- Constraints: `1 <= n <= 45`

### Examples

```text
Input: n = 2
Output: 2
Explanation: 1+1, 2
```

```text
Input: n = 3
Output: 3
Explanation: 1+1+1, 1+2, 2+1
```

This article uses Python only and derives the final solution step by step from the meaning of `dp[i]`.

## Start from the Last Move of n = 3

Look at the smallest example that exposes the transition:

```text
n = 3
```

The last move to step 3 can only come from:

- step 2, then climb 1 step
- step 1, then climb 2 steps

So the number of ways to reach step 3 is not created from nowhere. It comes from two smaller positions.

## Step 1: Define the Smaller Problem

Asking "how many ways are there to reach step `n`" is too large at first. Define:

```text
dp[i] = number of ways to reach step i
```

Here `i` means a staircase position, not merely an array index.

Start with the smallest skeleton:

```python
def climb_stairs(n: int) -> int:
    dp = [0] * (n + 1)
```

This version can:

- reserve a state for every position `0..n`
- make it clear that `dp[i]` means "number of ways to reach step i"

It still needs:

- the base cases for the start and step 1
- the transition for later positions

## Step 2: Fill the Start and Step 1

Build on the previous version and fill the base cases.

Standing at step `0` without moving can be understood as one way to already be at the start:

```python
dp[0] = 1
```

To reach step `1`, there is only one move:

```python
dp[1] = 1
```

The current code is:

```python
def climb_stairs(n: int) -> int:
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
```

This version can:

- handle `n = 1` correctly
- provide valid sources for the later transition

It still needs:

- how to reach `i >= 2` from previous positions

## Step 3: Step i Can Only Come from i - 1 or i - 2

Now consider `i >= 2`.

If the last move climbs `1` step, the previous position is `i - 1`.
If the last move climbs `2` steps, the previous position is `i - 2`.

These two groups do not overlap because their last move lengths are different.

Therefore the transition is `dp[i] = dp[i - 1] + dp[i - 2]`.[^why-not-plus-one]

Add this transition to the previous version:

```python
for i in range(2, n + 1):
    dp[i] = dp[i - 1] + dp[i - 2]
```

The first complete table-based solution is:

```python
def climb_stairs(n: int) -> int:
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
```

This version can:

- compute the number of ways for every staircase position
- return `dp[n]` directly

It still has room for:

- space optimization, because each state only depends on the previous two states

## Step 4: Walk Through the Table Slowly

For `n = 5`:

| i | Source | dp[i] |
| --- | --- | --- |
| 0 | start | 1 |
| 1 | only climb 1 step | 1 |
| 2 | dp[1] + dp[0] | 2 |
| 3 | dp[2] + dp[1] | 3 |
| 4 | dp[3] + dp[2] | 5 |
| 5 | dp[4] + dp[3] | 8 |

The important point in this table is:

> `dp[i]` always means "number of ways to reach step i", not "what the i-th move does".

## Step 5: Compress the dp Array into Two Variables

The previous transition is:

```text
dp[i] = dp[i - 1] + dp[i - 2]
```

It only depends on the previous two positions. Name them:

- `prev2`: `dp[i - 2]`
- `prev1`: `dp[i - 1]`

Replace the whole table with two variables:

```python
prev2 = 1
prev1 = 1

for _ in range(2, n + 1):
    cur = prev1 + prev2
    prev2 = prev1
    prev1 = cur
```

The final complete code is:

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1

        prev2 = 1
        prev1 = 1

        for _ in range(2, n + 1):
            cur = prev1 + prev2
            prev2 = prev1
            prev1 = cur

        return prev1


if __name__ == "__main__":
    print(Solution().climbStairs(2))
    print(Solution().climbStairs(3))
```

This version can:

- keep the same state transition as the `dp` table version
- reduce extra space from `O(n)` to `O(1)`

It still has one boundary:

- if the problem allowed more step sizes, there would be more transition sources; this problem only allows `1` or `2` steps

## Correctness

Invariant:

- when processing position `i`, `prev1` equals `dp[i]` and `prev2` equals `dp[i - 1]`

Why the transition is correct:

- Any way to reach step `i` must end with either a `1`-step move or a `2`-step move.
- Ways whose last move is `1` step correspond to `dp[i - 1]`.
- Ways whose last move is `2` steps correspond to `dp[i - 2]`.
- The two groups have different last move lengths, so they do not overlap. Add them.

## Complexity

- Time complexity: `O(n)`
- Extra space: `O(1)`

## Common Mistakes

- Setting `dp[0]` to `0`, which makes `n = 2` miss the direct `2`-step climb.
- Forgetting to handle `n = 1`; the space-optimized version may return an undefined `cur`.
- Understanding `dp[i]` as "what the i-th move chooses" instead of "number of ways to reach step i".

## Summary

- The first step of 1D DP is to define the meaning of `dp[i]`.
- In Climbing Stairs, `dp[i]` means the number of ways to reach step `i`.
- Step `i` can only come from `i - 1` or `i - 2`.
- Space optimization only keeps `dp[i - 2]` and `dp[i - 1]` as two variables.

## References and Follow-up

- LeetCode 70: Climbing Stairs
- LeetCode 746: Min Cost Climbing Stairs
- LeetCode 198: House Robber

## Notes

- The problem statement, examples, and constraints are based on the public LeetCode 70 summary.
- Python is used to match the current LeetCode tutorial style in this repository.

[^why-not-plus-one]: **Question: Why is it `dp[i] = dp[i - 1] + dp[i - 2]`, not `dp[i] = dp[i - 1] + 1`?**

    **Answer:** The key is that `dp[i]` stores a number of ways, not a step number or a move count. `dp[i - 1] + 1` roughly means "take the number of ways to reach step `i - 1`, then add one new way". That is not what happens. Every way to reach step `i - 1` can append one more `1`-step move and become one way to reach step `i`. So the contribution from `i - 1` is `dp[i - 1]` ways, not `1` way.

    Now look at the last move. To reach step `i`, the last move has only two possibilities: come from `i - 1` by climbing `1` step, or come from `i - 2` by climbing `2` steps. Therefore:

    ```text
    dp[i] = ways coming from i - 1 + ways coming from i - 2
          = dp[i - 1]             + dp[i - 2]
    ```

    For example, take `i = 4`. There are 3 ways to reach step 3: `1+1+1`, `1+2`, and `2+1`. Appending one `1`-step move to each of them gives 3 ways to reach step 4, so coming from step 3 contributes `dp[3] = 3` ways, not `+1`.

    There are 2 ways to reach step 2: `1+1` and `2`. Appending one `2`-step move to each of them also reaches step 4, so:

    ```text
    dp[4] = dp[3] + dp[2]
          = 3 + 2
          = 5
    ```

    If we wrote `dp[i] = dp[i - 1] + 1`, we would only consider the group that comes from the previous step, and we would incorrectly collapse that whole group into a single way.
