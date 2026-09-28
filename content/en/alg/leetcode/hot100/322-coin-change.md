---
title: "LeetCode 322: Coin Change, Deriving Unbounded Knapsack DP from Smaller Amounts"
date: 2026-09-22
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "unbounded knapsack", "coin change", "LeetCode 322"]
description: "Start from a greedy failure and smaller-amount subproblems, then build an unbounded knapsack DP that finds the minimum number of coins."
keywords: ["LeetCode 322", "Coin Change", "unbounded knapsack", "dynamic programming", "minimum coins"]
---

## Problem Requirements

You are given an integer array `coins`, where `coins[i]` is a coin denomination, and a non-negative integer `amount`. The goal is to make a total exactly equal to `amount`.

- You may use each denomination any number of times.
- Return the minimum number of coins needed to make `amount`.
- If the amount cannot be made, return `-1`.
- If `amount = 0`, no coins are needed, so the answer is `0`.

The constraints are `1 <= coins.length <= 12`, `1 <= coins[i] <= 2^31 - 1`, and `0 <= amount <= 10^4`. All coin denominations are distinct.

For example:

```text
Input: coins = [1, 2, 5], amount = 11
Output: 3
Explanation: 5 + 5 + 1 uses 3 coins.
```

```text
Input: coins = [2], amount = 3
Output: -1
Explanation: It is impossible to make 3 using only denomination 2.
```

```text
Input: coins = [1], amount = 0
Output: 0
Explanation: The target is already 0, so no coin needs to be chosen.
```

## Step 1: Break the Target Amount into a Smaller Problem

Start with a small example that exposes the structure of the problem:

```text
coins = [1, 3, 4]
amount = 6
```

If we follow the intuition of taking the largest denomination first, we might take `4`, leave `2`, and end with `4 + 1 + 1`, which uses 3 coins. But `3 + 3` is better and uses only 2 coins.

The current baseline is "choose a coin that looks large at every step." It fails on this example: the same target has several possible combinations, and a locally larger denomination does not guarantee the smallest total number of coins.

This failure forces a concrete question: if a particular coin is chosen last, how much remains? We must compare different choices for the last coin. Once one coin is fixed, the remaining amount becomes a smaller problem of exactly the same form.

For now, add only a definition for this smaller problem:

```text
best(x) = minimum number of coins needed to make amount x
```

This step only identifies the question that will be asked repeatedly. We have not yet defined the completion condition `best(0)`, nor have we written the transition from `best(x - coin)` to `best(x)`.

Check this step: `4 + 1 + 1` uses 3 coins, while `3 + 3` uses 2. Looking at only one coin choice cannot determine the answer; we must continue with the remaining amount.

This checkpoint can:

- state the input, output, and the rule that each denomination may be reused
- use `[1, 3, 4], 6` to prove that direct greedy selection is insufficient
- rewrite the large target as a question about smaller amounts

It still needs:

- a stopping rule when the remaining amount is 0
- a way to try every coin for a positive amount and compare the results

## Step 2: Handle Amount 0 First

Ask the smallest possible question: if the remaining amount is already `0`, how many more coins are needed? The answer is not "impossible" but **0 coins**. We need to fix this completion condition before any later coin-selection path can know when to stop.

The current baseline is the `best(x)` definition from the previous section, but it has no stopping rule. It loses its meaning at `x = 0`; if even the smallest amount has no answer, there is nothing for the positive-amount cases to build on.

Add only the zero-amount branch to the previous `best(x)` definition:

```python
def best(x: int) -> int:
    if x == 0:
        return 0

    raise NotImplementedError("Candidates for positive amounts arrive in the next step")
```

The `raise` deliberately preserves the current gap: this checkpoint verifies only the completion condition and does not pretend to solve positive amounts.

Check this step: substituting `x = 0` makes the function return `0` immediately. Substituting `x = 1` explicitly exposes the missing positive-amount transition instead of returning a temporary value that could be mistaken for an answer.

This checkpoint can:

- establish `best(0) = 0`, meaning that the target has been completed
- separate the positive-amount branch from the completion condition, leaving a clear place for the next coin-selection rule

It still needs:

- a way to try every coin no larger than `x` when `x > 0`
- a way for each candidate to connect to `best` for another, smaller amount

## Step 3: Enumerate the Last Coin to Get a First Correct Version

Now fill the remaining gap. For a positive amount `x`, the last coin could be any denomination in `coins`. After choosing `coin`, the preceding coins still need to make `x - coin`. If that smaller amount is reachable, add 1 for the current coin.

The current baseline is `best(x)` from the previous section: it returns 0 when `x = 0`, but it raises an exception for every `x > 0`. The failure is concrete: without a candidate transition, no positive amount can be answered.

In the previous `best` function, replace `NotImplementedError` with logic that tries every coin and keeps the minimum:

```python
def coin_change_recursive(coins: list[int], amount: int) -> int:
    def best(x: int) -> int:
        if x == 0:
            return 0

        minimum = float("inf")
        for coin in coins:
            if coin > x:
                continue

            rest = best(x - coin)
            if rest == -1:
                continue

            minimum = min(minimum, rest + 1)

        return -1 if minimum == float("inf") else minimum

    return best(amount)
```

Here, `minimum` is only the best candidate for the current `x`. `float("inf")` means that no reachable candidate has been found yet. The current coin is added only when the recursive result is not `-1`.

Check this step with `coins = [1, 3, 4]` and `amount = 6`. If the last coin is `4`, the remaining amount `2` needs at least 2 coins, so that branch uses 3 coins in total. If the last coin is `3`, the remaining amount `3` needs just 1 coin, so that branch uses 2 coins, and the function keeps it. Running two small examples should also produce:

```text
[1, 2, 5], 11 -> 3
[2], 3 -> -1
```

This checkpoint can:

- try every legal last coin for each positive amount
- handle a smaller amount that cannot be reached
- produce the first complete and correct answer for small inputs

It still needs:

- a way to avoid recomputing the same `best(x)` through different paths
- better performance for large amounts, because this correct baseline builds a rapidly growing recursion tree

## Step 4: Cache Amounts That Have Already Been Solved

First observe how the recursion from Step 3 repeats the same question. For `[1, 3, 4]` and `6`, one path contains `best(6) -> best(5) -> best(2)`, while another contains `best(6) -> best(3) -> best(2)`. Both paths need exactly the same answer for `best(2)`, but they recompute the entire subtree.

The current baseline is the recursive version with the correct transition. Its failure is not a wrong answer but duplicated work: as the amount grows, the number of repeated recursive branches grows as well.

Add only an amount-based cache decorator before the previous `best` definition. The candidate transition inside the function remains unchanged:

```python
from functools import cache


def coin_change_memo(coins: list[int], amount: int) -> int:
    @cache
    def best(x: int) -> int:
        if x == 0:
            return 0

        minimum = float("inf")
        for coin in coins:
            if coin > x:
                continue

            rest = best(x - coin)
            if rest == -1:
                continue

            minimum = min(minimum, rest + 1)

        return -1 if minimum == float("inf") else minimum

    return best(amount)
```

The cache key is only the amount `x`: the minimum number of coins for the same `x` does not depend on the path used to reach it. At most `amount + 1` distinct amount states are therefore solved. "Recompute on every path" becomes "solve each state once."

Check this step by running the cached version on the same inputs. The answers should remain `[1, 2, 5], 11 -> 3`, `[2], 3 -> -1`, and `[1, 3, 4], 6 -> 2`, and the number of states for each case should not exceed `amount + 1`. For `[1, 3, 4], 6`, which really does contain repeated subproblems, `cache_info()` should report more than zero hits. A single failing path such as `[2], 3` may correctly report zero hits.

This checkpoint can:

- preserve the transition and unreachable-case handling from Step 3
- keep one recursive result for each amount, with at most `amount + 1` states

It still needs:

- a way to remove the recursion stack, which still grows with the amount
- an iterative table that applies the same transition from smaller amounts to larger ones

## Step 5: Replace Recursion with a Bottom-Up Table

Step 4 established that each amount state only needs to be solved once. However, it still asks for smaller amounts through function calls. For a large amount, the call stack itself becomes a burden.

The current baseline is the memoized `best(x)`. Its failure is not an unclear state definition; the states are still hidden inside recursive calls. We need to lay out the answers for amounts `0` through `amount` explicitly in a table.

Keep the candidate transition from the previous version, but replace "recursively solve the smaller amount" with a lookup in an already-filled `dp` table:

```python
def coin_change_table(coins: list[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for x in range(1, amount + 1):
        for coin in coins:
            if coin > x or dp[x - coin] == float("inf"):
                continue

            dp[x] = min(dp[x], dp[x - coin] + 1)

    return -1 if dp[amount] == float("inf") else dp[amount]
```

The state meaning is: `dp[x]` is the minimum number of coins needed to make exactly amount `x`. The initialization `dp[0] = 0` carries forward the completion condition already verified in the previous section. Every other entry starts at infinity to mean "no solution has been found yet." When processing `x`, every smaller entry `dp[x - coin]` has already been filled.

Walk through the table slowly for `coins = [1, 2, 5]` and `amount = 11`:

| `x` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dp[x]` | 0 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 3 | 3 | 2 | 3 |

Therefore, `dp[11] = 3`. This table uses the same transition as the recursive version; it only replaces "call the smaller amount" with "read the completed table entry."

Check this step by running the table version. It should produce `[1, 2, 5], 11 -> 3`, `[2], 3 -> -1`, `[1], 0 -> 0`, and `[1, 3, 4], 6 -> 2`. The complete table for `[1, 2, 5], 11` should also match the manual table above.

This checkpoint can:

- store the optimal value for every amount in an explicit array
- produce the first complete iterative solution without relying on the recursion stack

It still needs:

- an explanation of why amounts are processed in ascending order and how that permits unlimited reuse of a denomination
- a connection between the name "unbounded knapsack" and the update order

## Step 6: Why This Is Unbounded Knapsack

Now consider an example that cannot avoid reusing the same denomination:

```text
coins = [2]
amount = 6
```

The answer must be `2 + 2 + 2`, or 3 copies of the same coin. The current `dp[x]` transition can already express this answer, but it has not explained why the update order permits the `2` that was just used to be used again. If we mistakenly treat this as a 0/1 problem in which each coin may be chosen only once, we lose the successive states for `4` and `6`.

This failure exposes the meaning of the traversal order. When a denomination `coin` is being processed, the inner amount loop must move from smaller to larger amounts. After computing `dp[2] = 1`, computing `dp[4]` may read that state, which already includes the current denomination. Computing `dp[6]` may then read `dp[4]`.

Reorganize the two loops in the previous table version so that unlimited reuse is explicit:

```python
def coin_change_complete(coins: list[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] == float("inf"):
                continue

            dp[x] = min(dp[x], dp[x - coin] + 1)

    return -1 if dp[amount] == float("inf") else dp[amount]
```

For `[2]` and `6`, the update trace is: at `x = 2`, set `dp[2] = 1`; at `x = 4`, read `dp[2]` and set `dp[4] = 2`; at `x = 6`, read `dp[4]` and set `dp[6] = 3`. This is unbounded knapsack: one item (here, one denomination) may be taken any number of times, so the inner amount loop runs in **ascending order**. A 0/1 knapsack uses descending order instead so the current item cannot be read again during the same outer iteration. The two directions are not interchangeable.

The objective here is to minimize the number of coins, not to count different orderings. Each update therefore uses only `min`; it does not count `2 + 5` and `5 + 2` as two separate answers.

Check this step by running the reorganized loops. They should produce `coins = [2], amount = 6 -> 3` and `coins = [2, 3, 5], amount = 7 -> 2`. The earlier official and pressure examples should keep the same answers.

This checkpoint can:

- express unlimited reuse as a coin-outer, amount-ascending unbounded-knapsack update
- explain why `dp[x - coin]` may come from a state updated earlier in the current outer iteration
- preserve the minimum-count objective instead of turning it into an ordering count

It still needs:

- the final LeetCode `class Solution` interface
- a unified unreachable return, correctness invariant, and complexity analysis

## Step 7: Freeze the Final Implementation

The previous section expressed reusable coins through a coin-outer loop and an amount-ascending inner loop. This checkpoint only prepares the solution for delivery: place the same logic in the `class Solution` interface required by LeetCode, and convert an unreachable sentinel to `-1`. No new state or transition is introduced.

The current baseline is the runnable `coin_change_complete` function. Its remaining failure is that it cannot yet be submitted through the required interface, and the return rules for `amount = 0`, unreachable amounts, and ordinary amounts need to live in one method.

Integrate the previous function into the single final implementation:

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        unreachable = amount + 1
        dp = [unreachable] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for current in range(coin, amount + 1):
                dp[current] = min(dp[current], dp[current - coin] + 1)

        return -1 if dp[amount] == unreachable else dp[amount]
```

`unreachable = amount + 1` is a sufficiently large finite sentinel. Any feasible solution uses at most `amount` coins because every denomination is at least 1, so the sentinel cannot be confused with a valid answer. The initialization `dp[0] = 0` remains the verified completion condition. Every update still reads `dp[current - coin]` after smaller amounts have been updated with the current denomination, so the unlimited-reuse semantics are unchanged.

Correctness follows from this invariant: after processing a denomination `coin`, for every `current`, `dp[current]` is the minimum number of coins needed to make `current` using only the denominations processed so far, with each denomination reusable any number of times. The ascending scan allows the current denomination to extend a smaller result produced earlier in the same outer iteration. After all denominations have been processed, `dp[amount]` is the minimum required by the problem. If it is still the sentinel, no combination can make the target, so the method returns `-1`.

There are `amount + 1` states, and every denomination may inspect each state once. The time complexity is therefore `O(amount * len(coins))`, and the extra space complexity is `O(amount)`.

Boundary regression cases:

```text
coins = [1, 2, 5], amount = 11 -> 3
coins = [2], amount = 3 -> -1
coins = [1], amount = 0 -> 0
coins = [2], amount = 6 -> 3
coins = [1, 3, 4], amount = 6 -> 2
```

This checkpoint can:

- satisfy LeetCode's `class Solution` method contract directly
- cover reachable, unreachable, zero-amount, and repeated-denomination cases
- connect the final code, correctness argument, and complexity analysis directly to the checkpoints already established

The incremental build is now frozen. There is no second `Reference Answer` that introduces new logic.
