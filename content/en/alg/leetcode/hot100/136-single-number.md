---
title: "LeetCode 136: Single Number Without Growing Extra Storage"
date: 2026-07-15T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "array", "bit manipulation", "XOR", "LeetCode 136"]
description: "Start from a set-based pairing baseline, derive XOR cancellation, and solve LeetCode 136 in O(n) time with O(1) extra space."
keywords: ["LeetCode 136", "Single Number", "XOR", "bit manipulation", "array", "Hot100"]
---

## Problem Requirement

You are given a non-empty integer array `nums`.

Exactly one element appears once. Every other element appears exactly twice. Return the element that appears once.

LeetCode provides this method contract:

```text
singleNumber(nums: List[int]) -> int
```

The solution must also satisfy two resource requirements:

- O(n) time
- O(1) extra space

### Example 1

```text
Input: nums = [2,2,1]
Output: 1
```

`2` appears twice. Only `1` appears once.

### Example 2

```text
Input: nums = [4,1,2,1,2]
Output: 4
```

Both `1` and `2` have matching copies. Only `4` remains unpaired.

### Example 3

```text
Input: nums = [1]
Output: 1
```

If the array has one element, that element is the answer.

### Constraints

- `1 <= nums.length <= 3 * 10^4`
- `-3 * 10^4 <= nums[i] <= 3 * 10^4`
- Every element except one appears exactly twice.

## Step 1: State "Appears Once" Precisely

Start with:

```text
nums = [4,1,2,1,2]
```

For this tiny array, we can pair equal values by eye:

```text
1 pairs with 1
2 pairs with 2
4 has no partner
```

The answer is `4`.

The current baseline is:

```text
Find matching elements one by one and remove every completed pair.
```

This baseline breaks because:

> Visual pairing only works for tiny inputs. When the array grows and equal values are far apart, we still lack an executable process for tracking which values have found partners.

Do not choose an algorithm yet. First fix the two questions that every later version must answer:

- How do we remove every value that appears twice?
- How do we leave the unique value in O(n) time and O(1) extra space?

Now this version can:

- state the exact duplicate guarantee
- identify the result as the unpaired value
- preserve the final linear-time and constant-space target

It still lacks:

- a runnable pairing process

## Step 2: Build a Correct Set Baseline

The current baseline is:

```text
Find matching values and remove completed pairs.
```

The idea is correct, but not executable. Two equal values may be far apart, so we need to remember which values have not found partners yet.

Add a set named `seen`:

```python
from typing import List


def single_number_with_set(nums: List[int]) -> int:
    seen = set()

    for num in nums:
        if num in seen:
            seen.remove(num)
        else:
            seen.add(num)

    return next(iter(seen))
```

The loop invariant is:

> After any prefix has been processed, `seen` contains the values in that prefix that have not completed a pair.

Trace `[4,1,2,1,2]`:

| Value | Operation | `seen` |
| --- | --- | --- |
| `4` | Not present, add it | `{4}` |
| `1` | Not present, add it | `{4, 1}` |
| `2` | Not present, add it | `{4, 1, 2}` |
| `1` | Present, remove it | `{4, 2}` |
| `2` | Present, remove it | `{4}` |

The problem guarantees exactly one unpaired value, so the set ends with exactly the answer.

Check the baseline:

```python
assert single_number_with_set([2, 2, 1]) == 1
assert single_number_with_set([4, 1, 2, 1, 2]) == 4
assert single_number_with_set([1]) == 1
assert single_number_with_set([-1, 2, 2]) == -1
```

Now this version can:

- execute the pairing process for arbitrary order
- find the unique value in O(n) expected time
- explain why one value remains

It still lacks:

- O(1) extra space; `seen` may hold O(n) values

## Step 3: Can Equal Values Cancel Without Being Stored?

The current set baseline needs extra space because every value waiting for a partner must be stored.

The break is now concrete:

> Preserve the effect that equal values disappear, but do not store those values.

This pressure finally introduces XOR, written as `^` in Python.

Four properties matter here:

```text
x ^ x = 0
x ^ 0 = x
a ^ b = b ^ a
(a ^ b) ^ c = a ^ (b ^ c)
```

The first two properties cancel equal values. The last two allow matching values to cancel even when they are far apart.

Use the properties immediately on `[4,1,2,1,2]`:

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

This performs the same work as the set baseline:

- the two `1` values disappear
- the two `2` values disappear
- the unpaired `4` remains

The difference is that XOR compresses the entire cancellation state into one integer.

The algebraic properties also hold for negative values:

```text
-1 ^ 2 ^ 2
= -1 ^ 0
= -1
```

Now this version can:

- cancel duplicate pairs regardless of position
- explain why input order does not affect the result
- replace growing unmatched-value storage with one integer state

It still lacks:

- a one-pass state update
- a loop invariant for that update

## Step 4: Turn Cancellation Into an Accumulator

The current baseline has established that the array result is:

```text
nums[0] ^ nums[1] ^ ... ^ nums[n - 1]
```

Turn that expression into one scan by initializing:

```python
answer = 0
```

Merge each value into the current result:

```python
answer ^= num
```

Complete LeetCode implementation:

```python
from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        answer = 0

        for num in nums:
            answer ^= num

        return answer
```

The loop invariant is:

> After every iteration, `answer` is the XOR of all values in the processed prefix.

Before the loop, the processed prefix is empty and its XOR identity is `0`. Each `answer ^= num` extends the represented prefix by one value.

At the end, the prefix is the full array. Every duplicate pair cancels, leaving the unique value.

### Checks

```python
solution = Solution()

assert solution.singleNumber([2, 2, 1]) == 1
assert solution.singleNumber([4, 1, 2, 1, 2]) == 4
assert solution.singleNumber([1]) == 1
assert solution.singleNumber([-1, 2, 2]) == -1
```

### Complexity

- Time: O(n), because each value participates in one XOR operation.
- Extra space: O(1), because only `answer` is stored.

## Common Mistakes

### 1. Hiding the set's space cost

The set version is a correct baseline, but its worst-case extra space is O(n).

### 2. Confusing XOR with logical OR

Python's XOR operator is `^`, not `or` and not `|`.

### 3. Forgetting the occurrence guarantee

This solution depends on every non-unique value appearing exactly twice. Values that appear three times require a different model.

### 4. Claiming the arithmetic formula uses O(1) space

`2 * sum(set(nums)) - sum(nums)` still creates a set, so its extra space is not O(1).

## Summary

The derivation is:

```text
visual pairing
-> store unmatched values in a set
-> expose the set's O(n) space cost
-> use XOR to cancel duplicate pairs
-> apply cancellation with one accumulator
```

The reusable idea is not merely one line of code. XOR compresses the unmatched state while forcing every duplicate pair to zero.

Continue with LeetCode 191 Number of 1 Bits to learn how bit operations can modify binary state directly.
