---
title: "LeetCode 191: Number of 1 Bits and How to Skip Irrelevant Zeros"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["bit manipulation", "binary", "Hamming Weight", "LeetCode 191"]
description: "Start from a right-shift baseline, derive how n & (n - 1) clears one lowest set bit, and prove LeetCode 191 with a loop invariant."
keywords: ["LeetCode 191", "Number of 1 Bits", "Hamming Weight", "n & (n - 1)", "bit manipulation"]
---

## Problem Requirement

Given a positive integer `n`, return the number of `1` bits in its binary representation. This count is also called the Hamming weight.

LeetCode provides this method contract:

```text
hammingWeight(n: int) -> int
```

### Example 1

```text
Input: n = 11
Binary: 1011
Output: 3
```

### Example 2

```text
Input: n = 128
Binary: 10000000
Output: 1
```

### Constraints

- `1 <= n <= 2^31 - 1`
- The input stays in the problem's non-negative integer domain.

Although the current constraints start at `1`, the implementation also handles `n = 0` naturally and returns `0`.

## Step 1: State Exactly What We Are Counting

Start with a small task that does not use decimal notation:

```text
n = 101100₂
```

There are three visible `1` bits:

```text
1 0 1 1 0 0
^   ^ ^
```

The answer is `3`.

The current baseline is:

```text
Expand the binary representation and count its 1 bits by eye.
```

This baseline breaks because:

> The input is an integer, not an already-expanded binary string. Visual counting is not a function for arbitrary input.

Now this version can:

- distinguish binary `1` bits from decimal digits equal to `1`
- explain why `11` has answer `3` and `128` has answer `1`
- preserve the requirement to operate on integer state directly

It still lacks:

- a runnable integer bit-inspection process

## Step 2: Inspect Bits From Right to Left

The current baseline needs to read an integer's binary bits.

The lowest bit has only two possibilities:

- `n & 1 == 1`: the lowest bit is `1`
- `n & 1 == 0`: the lowest bit is `0`

After inspecting it, remove that position with:

```python
n >>= 1
```

This right shift makes the next bit the new lowest bit.

First correct implementation:

```python
def hamming_weight_by_shift(n: int) -> int:
    count = 0

    while n:
        count += n & 1
        n >>= 1

    return count
```

Trace `101100₂`:

| Current `n` | `n & 1` | Updated `count` | Shifted `n` |
| --- | ---: | ---: | --- |
| `101100` | 0 | 0 | `10110` |
| `10110` | 0 | 0 | `1011` |
| `1011` | 1 | 1 | `101` |
| `101` | 1 | 2 | `10` |
| `10` | 0 | 2 | `1` |
| `1` | 1 | 3 | `0` |

The loop invariant is:

> `count` is the number of `1` bits among the positions already removed from the right, while `n` stores the unprocessed higher positions.

Checks:

```python
assert hamming_weight_by_shift(11) == 3
assert hamming_weight_by_shift(128) == 1
assert hamming_weight_by_shift(0) == 0
```

Now this version can:

- inspect every binary position directly on the integer
- count all set bits correctly
- terminate when no unprocessed positions remain

It still lacks:

- a way to avoid spending iterations on zero bits

## Step 3: Delete One Set Bit Instead of Crossing Zeros

Return to the second example:

```text
n = 10000000₂
```

It contains one set bit, but the shift baseline runs eight iterations.

The break is:

> Work follows the total bit length rather than the number of `1` bits we actually need to count.

Observe what subtraction does to `101100₂`:

```text
n     = 101100
n - 1 = 101011
```

The lowest `1` becomes `0`, and all lower zeros become `1`.

Apply bitwise AND:

```text
  101100
& 101011
--------
  101000
```

The result removes exactly the lowest set bit from `n` while preserving all higher bits. Therefore:

```text
n & (n - 1)
```

removes one `1` per operation.

Use it repeatedly on `101100₂`:

```text
101100 -> 101000
101000 -> 100000
100000 -> 000000
```

There are three operations, exactly matching the original set-bit count.

Replace the previous bit read and shift with:

```python
n &= n - 1
count += 1
```

Now this version can:

- remove one set bit in every iteration
- skip all zeros between set bits
- finish `128` in one iteration

It still lacks:

- a final invariant tying the number of removals to the return value
- the complete LeetCode method

## Step 4: The Number of Deletions Is the Answer

Complete implementation:

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0

        while n:
            n &= n - 1
            count += 1

        return count
```

The loop invariant is:

> After each iteration, `count` equals the number of `1` bits already removed, and `n` contains the set bits not yet removed.

Initially, no bit has been removed, so `count = 0`. Each `n &= n - 1` removes exactly one set bit, and the following increment records that removal.

When `n` becomes `0`, no set bits remain. `count` is therefore the number of set bits in the original input.

### Checks

```python
solution = Solution()

assert solution.hammingWeight(0) == 0
assert solution.hammingWeight(11) == 3
assert solution.hammingWeight(128) == 1
assert solution.hammingWeight(2**31 - 1) == 31
```

### Complexity

Let `k` be the number of set bits in `n`:

- Time: O(k), because every iteration removes one set bit.
- Extra space: O(1).

## Why the Input Domain Matters

Python models negative integers with conceptual infinite sign extension. Repeatedly applying this loop to a negative value does not naturally reach `0` as a fixed-width unsigned integer would.

The problem supplies positive integers, so the final code needs no width mask. If an engineering boundary accepts negative input, define the intended 32-bit or 64-bit representation and normalize it there before running this loop.

## Common Mistakes

### 1. Memorizing the expression without knowing what it removes

The important fact about `n & (n - 1)` is not merely that the result is smaller. It removes exactly the lowest set bit.

### 2. Forgetting to increment `count`

Changing `n` deletes a bit; the algorithm must also record how many deletions occurred.

### 3. Applying the Python loop directly to negative input

That is outside the problem domain and may not terminate as expected.

### 4. Calling the shift baseline incorrect

The shift version is correct. The optimization only changes how many irrelevant zero positions consume iterations.

## Summary

The derivation is:

```text
visually count binary 1 bits
-> inspect n & 1 and shift right
-> expose iterations spent on zeros
-> use n & (n - 1) to remove one set bit
-> count the removals
```

LeetCode 338 Counting Bits extends this operation from one integer to every value in `0..n`, then removes repeated work across those values.
