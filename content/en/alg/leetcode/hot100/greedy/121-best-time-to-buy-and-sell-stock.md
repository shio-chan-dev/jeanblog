---
title: "LeetCode 121: Best Time to Buy and Sell Stock With Greedy"
date: 2026-07-03T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "greedy", "array", "stock", "LeetCode 121"]
description: "Solve LeetCode 121 in Python by deriving the one-transaction greedy idea from the buy-before-sell constraint, an O(n^2) baseline, and the historical minimum price."
keywords: ["LeetCode 121", "Best Time to Buy and Sell Stock", "greedy", "array", "Python"]
---

> **Subtitle / Summary**
> The greedy idea is not "buy low, sell high" as a slogan. The precise rule is: if today is the sell day, the best buy day must be the lowest price before today.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `greedy`, `array`, `stock`
- **SEO keywords**: LeetCode 121, Best Time to Buy and Sell Stock, greedy, historical minimum price
- **Meta description**: A pressure-first Python guide to LeetCode 121 that derives brute force, historical minimum price, and the final one-pass greedy solution.

---

## Problem Requirement

You are given an array `prices`, where `prices[i]` is the stock price on day `i`.

You may complete one transaction:

- choose one day to buy one stock
- choose a different future day to sell that stock

Return the maximum profit you can achieve. If no profitable transaction exists, return `0`.

### Input and Output

- Input: `prices: List[int]`
- Output: maximum profit as an `int`
- You can buy once and sell once.
- The buy day must be before the sell day.
- You may choose not to trade, giving profit `0`.

### Examples

```text
Input: prices = [7,1,5,3,6,4]
Output: 5
```

The best trade buys at price `1` and sells at price `6`, giving profit `6 - 1 = 5`.

```text
Input: prices = [7,6,4,3,1]
Output: 0
```

Prices keep decreasing, so every buy-before-sell trade loses money. Return `0`.

### Constraints

- `1 <= prices.length <= 10^5`
- `0 <= prices[i] <= 10^4`

## Step 1: Fix the Buy-Sell Order First

Start with:

```text
prices = [7,1,5,3,6,4]
```

If we only look at price difference, the best profit is:

```text
1 -> 6
profit = 5
```

This is legal because price `1` appears before price `6`.

The current baseline is:

```text
Find two prices that maximize sell price - buy price.
```

This baseline breaks on time order.

For example:

```text
prices = [6,1]
```

A raw difference mindset might try `1 -> 6`, but price `1` happens after price `6`. You cannot buy in the future and sell in the past.

The break is:

> The maximum difference must respect buy day before sell day. A future low price cannot be used as the buy price for an earlier high price.

So we should turn the problem into a scan:

> When we stand on a day and try to sell today, the buy price must come from an earlier day.

Trace `[7,1,5,3,6,4]`:

```text
day 0 price 7: no earlier buy day, cannot sell
day 1 price 1: earlier minimum is 7, profit 1 - 7 < 0
day 2 price 5: earlier minimum is 1, profit 5 - 1 = 4
day 3 price 3: earlier minimum is 1, profit 3 - 1 = 2
day 4 price 6: earlier minimum is 1, profit 6 - 1 = 5
```

Now this version can:

- avoid treating the task as arbitrary maximum price difference
- view each day as a possible sell day
- restrict today's buy candidate to earlier prices

It still lacks:

- a first correct runnable version

## Step 2: Write a Correct but Slow Baseline

The current baseline is:

```text
The buy day must be before the sell day.
```

To make correctness visible, enumerate every legal transaction.

The break is:

> We only have a verbal rule. We still need executable code that checks every legal pair.

Use two loops:

```python
from typing import List


def max_profit_bruteforce(prices: List[int]) -> int:
    best = 0

    for buy in range(len(prices)):
        for sell in range(buy + 1, len(prices)):
            best = max(best, prices[sell] - prices[buy])

    return best
```

The loop meaning is direct:

- `buy` is the buy day
- `sell` starts at `buy + 1`, so selling always happens in the future
- `best` stores the best profit among all legal trades

Check it:

```python
assert max_profit_bruteforce([7, 1, 5, 3, 6, 4]) == 5
assert max_profit_bruteforce([7, 6, 4, 3, 1]) == 0
```

Now this version can:

- enumerate every legal `buy < sell` trade
- avoid reverse-time trades
- return `0` when no profit is possible

It still lacks:

- speed. `prices.length` can be `10^5`, so `O(n^2)` will time out.

## Step 3: Selling Today Only Needs the Historical Minimum

The current baseline is:

```text
For each sell day, try every earlier buy day.
```

Look at day 4, price `6`:

```text
prices = [7,1,5,3,6,4]
                 ^
               sell
```

If we sell today, the possible buy prices are:

```text
7, 1, 5, 3
```

The break is:

> For the same sell price, only the lowest earlier buy price can be optimal. Every higher buy price gives a smaller profit.

So we do not need all earlier buy days. We need one state:

```text
min_price = the lowest price seen before or up to the current scan point
```

And one answer state:

```text
best_profit = the best profit found so far
```

When scanning `price`, do:

```python
best_profit = max(best_profit, price - min_price)
min_price = min(min_price, price)
```

Computing profit first treats today as a sell day. Updating `min_price` then lets today become a buy candidate for later days.

One-pass version:

```python
from typing import List


def max_profit_greedy(prices: List[int]) -> int:
    min_price = prices[0]
    best_profit = 0

    for price in prices[1:]:
        best_profit = max(best_profit, price - min_price)
        min_price = min(min_price, price)

    return best_profit
```

Trace `[7,1,5,3,6,4]`:

```text
start: min_price = 7, best_profit = 0

price = 1: profit = -6, best = 0, min_price = 1
price = 5: profit = 4,  best = 4, min_price = 1
price = 3: profit = 2,  best = 4, min_price = 1
price = 6: profit = 5,  best = 5, min_price = 1
price = 4: profit = 3,  best = 5, min_price = 1
```

Now this version can:

- try selling today with only one historical state
- keep the global best profit in `best_profit`
- reduce the double loop to one scan

It still lacks:

- the LeetCode `class Solution` wrapper, edge checks, and complexity section

## Step 4: Complete Code and Verification

The current baseline is:

```text
Maintain min_price and best_profit while scanning prices once.
```

The break is:

> The code is not yet in the LeetCode `Solution.maxProfit` shape, and boundary cases are not checked.

Complete code:

```python
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        best_profit = 0

        for price in prices[1:]:
            best_profit = max(best_profit, price - min_price)
            min_price = min(min_price, price)

        return best_profit
```

The loop invariant is:

> Before processing `price`, `min_price` is the lowest price among earlier days, and `best_profit` is the best legal transaction already checked.

For the current `price`, compute the profit if today is the sell day, then let today become a possible buy day for future prices.

Check:

```python
def check() -> None:
    s = Solution()

    assert s.maxProfit([7, 1, 5, 3, 6, 4]) == 5
    assert s.maxProfit([7, 6, 4, 3, 1]) == 0
    assert s.maxProfit([5]) == 0
    assert s.maxProfit([1, 2, 3, 4]) == 3
    assert s.maxProfit([4, 3, 2, 1]) == 0


check()
```

Now this version can:

- guarantee buy day is before sell day
- keep only the one historical state that matters: the lowest buy price
- return `0` when no profitable trade exists
- fit the required LeetCode interface

## Complexity

Let `n = len(prices)`.

- Time complexity: `O(n)`, because each price is scanned once.
- Space complexity: `O(1)`, because only `min_price` and `best_profit` are stored.

## Summary

The precise greedy proof for LeetCode 121 is:

```text
If today is the sell day,
the best buy day must be the lowest price before today.
```

So the scan only needs:

- `min_price`: the lowest price seen so far
- `best_profit`: the best profit found so far

That is the smallest provable greedy state for one stock transaction.
