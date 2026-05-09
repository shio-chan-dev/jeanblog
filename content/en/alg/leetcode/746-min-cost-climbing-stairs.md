---
title: "LeetCode 746: Min Cost Climbing Stairs, Deriving DP from the Top Position"
date: 2026-05-03T14:33:39+08:00
draft: false
categories: ["LeetCode"]
tags: ["dynamic programming", "1D DP", "climbing stairs", "LeetCode 746"]
---

## Problem

### Input and Output

- Input: an integer array `cost`
- `cost[i]` is the cost of stepping on stair `i`
- Each move can climb `1` or `2` steps
- You may start from index `0` or index `1`
- Output: return the minimum cost to reach the top
- Constraints: `2 <= cost.length <= 1000`, `0 <= cost[i] <= 999`

### Examples

```text
Input: cost = [10,15,20]
Output: 15
Explanation: start from index 1, pay 15, then reach the top directly
```

```text
Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
```

## Start from the Top Position of [10,15,20]

Look at the small example:

```text
cost = [10,15,20]
```

The top is not index `2`. It is the position after the last stair, which can be called position `3`.

The last move to top position `3` can only come from:

- position `2`, paying `cost[2]`
- position `1`, paying `cost[1]`

This is the easiest place to make a mistake: the task asks for the cost to reach the top, not the cost to reach the last index.

## Step 1: Define Positions, Not Stair Indices

If we only stare at `cost[i]`, it is easy to put the answer on the last stair. First define positions:

```text
position 0: stair 0
position 1: stair 1
...
position n: top, after the last stair
```

Define the state:

```text
dp[i] = minimum cost to reach position i
```

Start with the smallest skeleton:

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
```

This version can:

- reserve states for positions `0..n`
- make it clear that `dp[n]` is the answer at the top

It still needs:

- how to define the starting cost
- how to transfer from the previous two positions

## Step 2: Starting from 0 or 1 Costs 0

The problem allows starting from index `0` or index `1`. Starting there itself is not counted in `dp`; the cost is paid when moving upward from a stair.

So the base cases are:

```python
dp[0] = 0
dp[1] = 0
```

The current code is:

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0
```

This version can:

- express that we can start from `0` or `1` for free
- provide the previous two sources for later positions

It still needs:

- where the payment comes from when reaching position `i`

## Step 3: Position i Can Only Be Reached from i - 1 or i - 2

Now consider `i >= 2`.

If the last move comes from `i - 1`, we first reach position `i - 1`, then pay `cost[i - 1]`.
If the last move comes from `i - 2`, we first reach position `i - 2`, then pay `cost[i - 2]`.

Add this transition to the previous version:

```python
for i in range(2, n + 1):
    dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
```

The two terms mean:

- `dp[i - 1] + cost[i - 1]`: pay from the previous stair and move one step to `i`
- `dp[i - 2] + cost[i - 2]`: pay from two stairs before and move two steps to `i`

The first complete correct version is:

```python
def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0

    for i in range(2, n + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

    return dp[n]
```

This version can:

- compute the minimum cost to reach every position
- put the answer at top position `dp[n]`

It still has room for:

- space optimization, because each state only depends on the previous two positions

## Step 4: Walk Through the Table Slowly

For `cost = [10,15,20]`:

| i | Position Meaning | From i - 1 | From i - 2 | dp[i] |
| --- | --- | --- | --- | --- |
| 0 | start 0 | - | - | 0 |
| 1 | start 1 | - | - | 0 |
| 2 | position 2 | 0 + 15 | 0 + 10 | 10 |
| 3 | top | 10 + 20 | 0 + 15 | 15 |

The answer is:

```text
dp[3] = 15
```

This table checks two things:

- the top is position `n`, not `n - 1`
- the paid cost belongs to the stair we leave from

## Step 5: Compress the dp Array into Two Variables

The previous transition only depends on:

- `dp[i - 2]`
- `dp[i - 1]`

So two variables are enough:

```python
prev2 = 0
prev1 = 0
```

Replace the table with two variables:

```python
for i in range(2, n + 1):
    cur = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
    prev2 = prev1
    prev1 = cur
```

The final complete code is:

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        n = len(cost)
        prev2 = 0
        prev1 = 0

        for i in range(2, n + 1):
            cur = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
            prev2 = prev1
            prev1 = cur

        return prev1


if __name__ == "__main__":
    print(Solution().minCostClimbingStairs([10, 15, 20]))
    print(Solution().minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))
```

This version can:

- preserve the same position meaning of `dp[i]`
- reduce extra space from `O(n)` to `O(1)`

It still has one boundary:

- if the problem allowed more jump lengths, we would need more sources; this problem only allows `1` or `2` steps

## Correctness

Invariant:

- after processing position `i`, `prev1` equals `dp[i]` and `prev2` equals `dp[i - 1]`

Why the transition is correct:

- The last move to position `i` can only come from position `i - 1` or `i - 2`.
- Coming from `i - 1` must pay `cost[i - 1]`.
- Coming from `i - 2` must pay `cost[i - 2]`.
- The problem asks for the minimum cost, so take the smaller of the two.

Why return `prev1`:

- after the loop, position `n` has been processed
- `prev1` is `dp[n]`, the minimum cost to reach the top

## Complexity

- Time complexity: `O(n)`
- Extra space: `O(1)`

## Common Mistakes

- Returning the cost to reach the last stair `n - 1` instead of the top position `n`.
- Initializing `dp[0]` and `dp[1]` as `cost[0]` and `cost[1]`, which breaks the rule that we may start from 0 or 1 for free.
- Writing `cost[i]` in the transition, even though reaching position `i` pays the cost of the source stair.

## Summary

- The key is to treat the top as position `n`.
- `dp[i]` means the minimum cost to reach position `i`.
- To reach `i`, we can only pay and jump from `i - 1` or `i - 2`.
- Space optimization only keeps `dp[i - 2]` and `dp[i - 1]` as two variables.

## References and Follow-up

- LeetCode 70: Climbing Stairs
- LeetCode 746: Min Cost Climbing Stairs
- LeetCode 198: House Robber

## Notes

- The problem statement, examples, and constraints are based on the public LeetCode 746 summary.
- Python is used to match the current LeetCode tutorial style in this repository.
