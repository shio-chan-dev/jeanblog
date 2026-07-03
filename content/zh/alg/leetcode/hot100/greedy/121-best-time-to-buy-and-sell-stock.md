---
title: "LeetCode 121：买卖股票的最佳时机，从历史最低价推出一次交易贪心"
date: 2026-07-03T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "贪心", "数组", "股票", "LeetCode 121"]
description: "从买入必须早于卖出的顺序压力出发，推导 LeetCode 121 买卖股票的最佳时机：先写 O(n^2) 正确基线，再压缩成维护历史最低价的一次扫描贪心。"
keywords: ["LeetCode 121", "买卖股票的最佳时机", "Best Time to Buy and Sell Stock", "贪心", "数组", "Python"]
---

## 题目要求

给你一个数组 `prices`，其中 `prices[i]` 表示第 `i` 天的股票价格。

你只能完成一次交易：

- 选择某一天买入一支股票
- 选择未来某一天卖出这支股票

返回能获得的最大利润。如果无法盈利，返回 `0`。

### 输入输出

- 输入：`prices: List[int]`
- 输出：最大利润 `int`
- 只能买一次、卖一次。
- 买入日必须早于卖出日。
- 可以选择不交易，此时利润是 `0`。

### 示例

```text
输入：prices = [7,1,5,3,6,4]
输出：5
```

最优做法是在价格为 `1` 时买入，在价格为 `6` 时卖出，利润是 `6 - 1 = 5`。

```text
输入：prices = [7,6,4,3,1]
输出：0
```

价格一直下降，任何买入后再卖出都会亏钱，所以返回 `0`。

### 约束

- `1 <= prices.length <= 10^5`
- `0 <= prices[i] <= 10^4`

## Step 1：先固定买卖顺序

先看这个例子：

```text
prices = [7,1,5,3,6,4]
```

如果只看价格差，最大利润来自：

```text
1 -> 6
profit = 5
```

这是合法的，因为价格 `1` 出现在价格 `6` 之前。

当前 baseline 是：

```text
找两个价格，让卖出价 - 买入价 最大。
```

这个 baseline 会在顺序上出错。

比如如果数组是：

```text
prices = [6,1]
```

单纯看差值可能会想用 `1 -> 6`，但 `1` 在 `6` 后面，不能先在未来买入，再回到过去卖出。

break 是：

> 最大差值必须满足买入日在卖出日前。未来的低价不能拿来给过去的高价当买入价。

所以问题要改成一个扫描问题：

> 当我们站在某一天准备卖出时，只能从它之前的天里选买入价。

用 `[7,1,5,3,6,4]` 手推：

```text
第 0 天价格 7：没有更早的买入日，无法卖出
第 1 天价格 1：更早最低价是 7，利润 1 - 7 < 0
第 2 天价格 5：更早最低价是 1，利润 5 - 1 = 4
第 3 天价格 3：更早最低价是 1，利润 3 - 1 = 2
第 4 天价格 6：更早最低价是 1，利润 6 - 1 = 5
```

这一步之后，当前版本能做到：

- 知道目标不是任意两个价格的最大差值。
- 知道每一天都可以被看成“今天卖出”。
- 知道今天卖出时，只能使用今天之前的最低买入价。

它还缺：

- 一个先正确、可运行的版本。

## Step 2：先写一个正确但慢的版本

当前 baseline 是：

```text
买入日必须早于卖出日。
```

为了先保证正确，可以直接枚举所有合法交易。

break 是：

> 现在只有口头规则，还没有一个能验证所有合法交易的代码版本。

先写双循环：

```python
from typing import List


def max_profit_bruteforce(prices: List[int]) -> int:
    best = 0

    for buy in range(len(prices)):
        for sell in range(buy + 1, len(prices)):
            best = max(best, prices[sell] - prices[buy])

    return best
```

这个版本的循环含义很清楚：

- `buy` 是买入日
- `sell` 从 `buy + 1` 开始，保证卖出日在未来
- `best` 记录所有合法交易里的最大利润

检查：

```python
assert max_profit_bruteforce([7, 1, 5, 3, 6, 4]) == 5
assert max_profit_bruteforce([7, 6, 4, 3, 1]) == 0
```

现在这个版本能做到：

- 枚举所有 `buy < sell` 的合法交易。
- 不会使用未来价格当过去的买入价。
- 无法盈利时返回 `0`。

它还缺：

- 复杂度太高。`prices.length` 最大是 `10^5`，`O(n^2)` 会超时。

## Step 3：今天卖出时，只需要历史最低价

当前 baseline 是：

```text
对每个 sell，枚举所有 buy < sell。
```

看第 4 天价格 `6`：

```text
prices = [7,1,5,3,6,4]
                 ^
               sell
```

如果今天卖出，所有候选买入价是：

```text
7, 1, 5, 3
```

break 是：

> 对同一个卖出价来说，历史买入价里只有最低价可能最优。其他更高买入价都会得到更小利润。

所以不需要保存所有历史买入日，只需要保存一个状态：

```text
min_price = 扫描到当前天之前见过的最低价格
```

再保存当前最好利润：

```text
best_profit = 到目前为止能得到的最大利润
```

扫描到价格 `price` 时，做两件事：

```python
best_profit = max(best_profit, price - min_price)
min_price = min(min_price, price)
```

这里先算利润再更新最低价，意思是：

- 今天可以作为卖出日
- 今天也可以成为后面某天的买入候选

完整的一次扫描版本：

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

用 `[7,1,5,3,6,4]` trace：

```text
start: min_price = 7, best_profit = 0

price = 1: profit = -6, best = 0, min_price = 1
price = 5: profit = 4,  best = 4, min_price = 1
price = 3: profit = 2,  best = 4, min_price = 1
price = 6: profit = 5,  best = 5, min_price = 1
price = 4: profit = 3,  best = 5, min_price = 1
```

这一步之后，当前版本能做到：

- 每天只用一个历史最低价尝试“今天卖出”。
- 用 `best_profit` 保留全局最优答案。
- 把双循环压成一次扫描。

它还缺：

- LeetCode 要求的 `class Solution` 包装、边界检查和复杂度说明。

## Step 4：完整代码和验证

当前 baseline 是：

```text
维护 min_price 和 best_profit，一次扫描 prices。
```

break 是：

> 还没有整理成 LeetCode 的 `Solution.maxProfit`，也还没有覆盖边界场景。

完整代码：

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

循环 invariant 是：

> 每次处理 `price` 前，`min_price` 是之前所有天的最低价格；`best_profit` 是之前已经检查过的合法交易中的最大利润。

处理当前 `price` 时，先用它当卖出价计算利润，再把它纳入后续天的最低买入价候选。

检查：

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

现在这个版本能做到：

- 保证买入日在卖出日前。
- 每天只保留一个足够好的历史状态：最低买入价。
- 无法盈利时返回 `0`，不会返回负数。
- 满足 LeetCode 提交接口。

## 复杂度

设 `n = len(prices)`。

- 时间复杂度：`O(n)`，每个价格只扫描一次。
- 空间复杂度：`O(1)`，只维护 `min_price` 和 `best_profit`。

## 小结

121 的贪心点不是“看到低价就买，看到高价就卖”这么口语化。

更准确的说法是：

```text
当今天作为卖出日时，
最优买入日一定是今天之前价格最低的那一天。
```

因此扫描时只需要维护：

- `min_price`：到今天之前见过的最低价
- `best_profit`：到目前为止能得到的最大利润

这就是这题最小、可证明的一次交易贪心。
