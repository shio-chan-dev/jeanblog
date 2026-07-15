---
title: "LeetCode 338: Counting Bits by Reusing Smaller Results"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "bit manipulation", "dynamic programming", "binary", "LeetCode 338"]
description: "Start by counting every value in 0..n independently, then derive answer[i] = answer[i & (i - 1)] + 1 for an O(n) solution."
keywords: ["LeetCode 338", "Counting Bits", "dynamic programming", "n & (n - 1)", "bit manipulation", "Hot100"]
---

## Problem Requirement

Given a non-negative integer `n`, return an array `answer` of length `n + 1`.

For every index:

```text
answer[i] = the number of 1 bits in the binary representation of i
```

The requested range includes both `0` and `n`.

LeetCode provides this method contract:

```text
countBits(n: int) -> List[int]
```

### Example 1

```text
Input: n = 2
Output: [0,1,1]
```

```text
0 -> 0   -> 0 set bits
1 -> 1   -> 1 set bit
2 -> 10  -> 1 set bit
```

### Example 2

```text
Input: n = 5
Output: [0,1,1,2,1,2]
```

### Constraints

- `0 <= n <= 10^5`

## Step 1: This Time We Need Every Answer From 0 to n

LeetCode 191 asks for one integer's set-bit count. For `n = 5`, this problem asks for all six related answers:

| `i` | Binary | Set bits |
| ---: | --- | ---: |
| 0 | `0` | 0 |
| 1 | `1` | 1 |
| 2 | `10` | 1 |
| 3 | `11` | 2 |
| 4 | `100` | 1 |
| 5 | `101` | 2 |

The output is:

```text
[0,1,1,2,1,2]
```

The current baseline is:

```text
We already know how to count the set bits of one integer.
```

This baseline breaks because:

> One scalar count does not produce the ordered results for every value in `0..n`.

Now this version can:

- define exactly what `answer[i]` means
- establish that the output length is `n + 1`
- preserve `answer[0] = 0` as the base result

It still lacks:

- a correct algorithm that builds the entire output

## Step 2: Repeat the LeetCode 191 Process for Every Value

The current baseline can count one value with:

```text
value & (value - 1)
```

Each operation removes one lowest set bit. The most direct batch solution is to run that complete process independently for every value from `0` through `n`.

```python
from typing import List


def count_bits_repeated(n: int) -> List[int]:
    answer = []

    for value in range(n + 1):
        current = value
        count = 0

        while current:
            current &= current - 1
            count += 1

        answer.append(count)

    return answer
```

Checks:

```python
assert count_bits_repeated(0) == [0]
assert count_bits_repeated(2) == [0, 1, 1]
assert count_bits_repeated(5) == [0, 1, 1, 2, 1, 2]
```

The control flow is explicit:

- the outer loop visits every value in order
- the inner loop counts that value independently
- `answer` stores each completed result

Now this version can:

- generate the full output correctly
- reuse the already-proved clearing operation from 191
- handle the boundary `n = 0`

It still lacks:

- reuse across different values; every count starts from scratch

Because a value has O(log n) binary positions, this baseline has an O(n log n) upper bound.

## Step 3: After Clearing One Set Bit, the Result Already Exists

The current baseline repeatedly executes:

```text
value = value & (value - 1)
```

It ignores an important fact. For every `i > 0`:

```text
i & (i - 1) < i
```

The operation removes one set bit, so when values are processed in increasing order, the smaller result already exists in `answer`.

Use `i = 12`:

```text
i             = 1100₂
i & (i - 1)   = 1000₂ = 8
```

`12` has exactly one more set bit than `8`, so:

```text
answer[12] = answer[8] + 1
```

The general recurrence is:

```text
answer[i] = answer[i & (i - 1)] + 1
```

Use it immediately for `1..5`:

| `i` | `i & (i - 1)` | Existing result | `answer[i]` |
| ---: | ---: | ---: | ---: |
| 1 | 0 | `answer[0] = 0` | 1 |
| 2 | 0 | `answer[0] = 0` | 1 |
| 3 | 2 | `answer[2] = 1` | 2 |
| 4 | 0 | `answer[0] = 0` | 1 |
| 5 | 4 | `answer[4] = 1` | 2 |

Only now does the dynamic-programming classification earn its place:

- current state: `answer[i]`
- smaller subproblem: `answer[i & (i - 1)]`
- transition: add one to that smaller result

Now this version can:

- compute each new count with one lookup and one addition
- prove that the dependency has already been computed
- turn the 191 clearing operation into cross-value reuse

It still lacks:

- a complete table-filling implementation with an explicit invariant

## Step 4: Fill the Table in Increasing Order

Initialize the output:

```python
answer = [0] * (n + 1)
```

`answer[0] = 0` is already the correct base case. Applying the recurrence at `i = 0` would create a self-dependency, so the loop starts at `1`.

Complete LeetCode implementation:

```python
from typing import List


class Solution:
    def countBits(self, n: int) -> List[int]:
        answer = [0] * (n + 1)

        for i in range(1, n + 1):
            answer[i] = answer[i & (i - 1)] + 1

        return answer
```

The loop invariant is:

> Before iteration `i`, every entry in `answer[0..i-1]` is correct.

For `i > 0`, `i & (i - 1)` is strictly smaller than `i`, so the transition reads an earlier correct entry. Writing `answer[i]` extends the correct prefix by one position.

### Checks

```python
solution = Solution()

assert solution.countBits(0) == [0]
assert solution.countBits(2) == [0, 1, 1]
assert solution.countBits(5) == [0, 1, 1, 2, 1, 2]
```

### Complexity

- Time: O(n), because each `i` uses one transition.
- Output space: O(n), required by the problem.
- Extra space: O(1), excluding the returned array.

## Common Mistakes

### 1. Starting the recurrence at zero

In Python, `0 & (0 - 1)` is still `0`, so applying the transition would incorrectly set `answer[0] = answer[0] + 1`. Preserve the base case and start from `1`.

### 2. Writing the formula without proving the lookup is ready

The key relation is `i & (i - 1) < i`. Without it, the fill order is not justified.

### 3. Calling a complete counting function for every value

That is the correct baseline, but it does not achieve O(n) batch reuse.

### 4. Counting the output array as extra space

`answer` is the required return value. Distinguish O(n) output space from O(1) extra space.

## Summary

The derivation is:

```text
define answer[i]
-> repeat LeetCode 191 for every value
-> notice that the cleared predecessor is already known
-> answer[i] = answer[i & (i - 1)] + 1
-> fill the table in increasing order
```

LeetCode 191 teaches how to remove one set bit from one integer. LeetCode 338 reuses the smaller result across an entire range.
