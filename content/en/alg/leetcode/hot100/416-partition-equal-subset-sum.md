---
title: "LeetCode 416: Partition Equal Subset Sum, from the Half-Sum Target to Backward 0/1 Knapsack"
date: 2026-09-22T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "0/1 knapsack", "subset sum", "LeetCode 416"]
description: "Start from the constraints of Partition Equal Subset Sum, build the 0/1 knapsack state step by step, and see why one-dimensional compression requires backward iteration."
keywords: ["Partition Equal Subset Sum", "LeetCode 416", "0/1 knapsack", "subset sum", "backward iteration"]
---

## Problem Requirements

Given an array `nums` containing only positive integers, determine whether all its elements can be split into two subsets whose sums are equal.

### Input and Output

- Input: an integer array `nums`
- Output: return `True` if the array can be split equally; otherwise, return `False`
- Every element must belong to exactly one of the two subsets. No element may be discarded or used twice

### Examples

```text
Input: nums = [1,5,11,5]
Output: True
Explanation: [1,5,5] and [11] both have a sum of 11
```

```text
Input: nums = [1,2,3,5]
Output: False
Explanation: the total is 11, and an odd total cannot be split equally
```

```text
Input: nums = [1,1]
Output: True
Explanation: put one 1 in each subset
```

### Constraints

According to the LeetCode 416 statement:

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`
- `sum(nums) <= 20000`

## Step 1: Reject Arrays with an Odd Total First

Start with a question that does not involve choosing any elements yet: **if two integer subsets have the same sum, what must be true about the total sum of the array?**

### Problem Pressure

Consider `nums = [1,2,3,5]`. Its total is `11`. If both subsets have sum `x`, then we must have:

```text
2 * x = 11
```

No integer `x` can satisfy this equation, so we can return `False` for this input before trying to select any elements.

### Current Baseline

At this point, we only have the problem contract: every element must eventually belong to one subset, but we have not recorded which elements are selected and cannot yet handle cases with an even total.

### Break

If the total is odd, continuing the search only performs work that must fail. If the total is even, however, parity alone does not tell us whether some subset actually reaches half the total.

### Add a Parity Gate to the Previous Version

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)

    if total % 2 == 1:
        return False

    target = total // 2

    # For now, only compute the target. The next step checks reachability.
    # This placeholder return is not the final answer for even-total cases.
    return False
```

### Check This Change

```python
assert can_partition([1, 2, 3, 5]) is False
```

For `[1,2,3,5]`, `total % 2 == 1`, so the function returns `False` before entering any element-selection logic. For `[1,1]`, `target` is computed as `1`, but this version still cannot decide whether the elements can form that `1`.

### Current Checkpoint

This version can now:

- rule out impossible inputs by checking whether the total is odd
- turn an even total into a target that still needs to be verified: `target = total // 2`

It still needs to:

- determine whether some subset of elements sums to exactly `target`
- handle every real even-total input; the `return False` in the even branch is still a placeholder

The next step reduces the large problem to a smaller one: after processing a prefix of the array, which subset sums are reachable?

## Step 2: Record Which Sums Are Reachable After Each Prefix

### Problem Pressure

The total of `[1,1]` is `2`, so the target is `1`. The parity check only tells us that the input is worth examining. It does not answer these questions: after seeing the first `1`, can we form `1`? After seeing the second `1`, which sums can we form?

Write out the small process:

| Elements processed | Currently reachable sums |
| --- | --- |
| None | `{0}` |
| First `1` | `{0, 1}` |
| Both `1`s | `{0, 1, 2}` |

The smaller problem is: **after processing a prefix of the array, what set contains every sum that can be formed using only that prefix, with each element used at most once?**

### Current Baseline

Step 1 can already compute `target` for an even total, but the even branch still ends with the placeholder `return False`. The current version does not store any sum that can already be formed.

### Break

If we stare only at `target`, we lose the intermediate states. For example, the target for `[1,5,11,5]` is `11`. We first need to know which smaller sums each prefix can form before we can decide whether the later `11`, or the two `5`s, can reach it.

### Replace the Placeholder Return in the Previous Version

Start with a set of reachable sums. When processing a value `num`, there are two paths: skip it, or add it to every sum from the previous round. To ensure that the current element is processed only once, copy the old set into a new set and pass that new set to the next round.

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)

    if total % 2 == 1:
        return False

    target = total // 2
    reachable = {0}

    for num in nums:
        next_reachable = set(reachable)

        for current in reachable:
            new_sum = current + num
            if new_sum <= target:
                next_reachable.add(new_sum)

        reachable = next_reachable

    return target in reachable
```

### Check This Change

First, trace `[1,1]`:

```text
reachable = {0}
Read the first 1  -> {0,1}
Read the second 1 -> {0,1,2}
target = 1, so return True
```

Then run three executable checks:

```python
assert can_partition([1, 5, 11, 5]) is True
assert can_partition([1, 2, 3, 5]) is False
assert can_partition([1, 1]) is True
```

### Current Checkpoint

This version can now:

- rewrite "can the array be split equally?" as "does the set of reachable sums contain `target`?"
- make it explicit that each array element is processed only once by copying the old set
- pass three representative examples with a runnable baseline

It still needs:

- a state table with fixed bounds and a more stable meaning for the relationship between a prefix and a sum
- control over the cost of copying sets, and a way to preserve the one-use rule when the state is compressed later

The next step writes the set state as a two-dimensional table, producing a complete correct transition before compressing any space.

## Step 3: Express "Prefix + Sum" as a Two-Dimensional State

### Problem Pressure

The set baseline already returns the right answer, but it creates a new problem: every input value requires copying and expanding an entire set. As `target` grows, the set's contents and size both change dynamically. It is also difficult to inspect directly why a particular sum is reachable after a particular number of elements.

For `[1,5,11,5]`, the target is `11`. The set version generates many intermediate sums, but what we really want to ask is a fixed family of questions: **using only the first `i` elements, can we form sum `s`?**

### Current Baseline

The `reachable` set from Step 2 stores every sum reachable from the current prefix. Each round copies the old set into `next_reachable` and then adds the new sums produced by choosing the current element.

### Break

The set version is correct, but it does not give us a fixed coordinate for the relationship between "the first `i` elements" and "target sum `s`." To retain every prefix state, we need to make those two dimensions explicit.

### Replace the State Representation in the Previous Version

Only now do we name this fixed state for "choose or skip each element once": two-dimensional `dp[i][s]`. It means: **using only `nums[0]` through `nums[i-1]`, can we form sum `s`?**

Start with the base case that selects no elements: `dp[0][0] = True`. When processing the `i`-th element, `num = nums[i - 1]`, there are only two sources:

- Skip `num`: carry forward `dp[i - 1][s]`.
- Choose `num`: only when `s >= num` can the state come from `dp[i - 1][s - num]` in the previous row.

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)

    if total % 2 == 1:
        return False

    target = total // 2
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        num = nums[i - 1]

        for s in range(target + 1):
            dp[i][s] = dp[i - 1][s]
            if s >= num:
                dp[i][s] = dp[i][s] or dp[i - 1][s - num]

    return dp[n][target]
```

### Check This Change

For `[1,1]` with `target = 1`, inspect only the target columns:

| `i` (number processed) | `dp[i][0]` | `dp[i][1]` |
| --- | --- | --- |
| `0` | `True` | `False` |
| `1` | `True` | `True` |
| `2` | `True` | `True` |

Every "choose" transition reads from the previous row, so the second `1` cannot produce another new `1` within the same row. Now run the representative cases:

```python
assert can_partition([1, 5, 11, 5]) is True
assert can_partition([1, 2, 3, 5]) is False
assert can_partition([1, 1]) is True
assert can_partition([2]) is False
assert can_partition([2, 2]) is True
```

### Current Checkpoint

This version can now:

- use `dp[i][s]` to represent the reachable sums for every prefix explicitly
- express both skipping and choosing the current element with the previous row, giving us the first complete correct solution
- pass the main examples and the one-element and two-element edge cases

It still needs:

- to reduce the `O(n * target)` extra space used by the two-dimensional table, since each row depends only on the previous row
- an update direction that preserves the "use each element at most once" rule after the rows are compressed into one dimension

The next step attempts one-dimensional compression and uses a counterexample to find out why a forward update reuses the current element.

## Step 4: A Forward Update Reuses the Same Element After 1D Compression

### Problem Pressure

In the two-dimensional table from Step 3, row `i` depends only on row `i - 1`, so it appears that we can retain just one row. This can reduce the space below `O(n * target)`, but once new results are written back into the same array, we must decide in which direction to process the sums associated with `num`.

### Current Baseline

The current baseline is the correct two-dimensional version: choosing `num` reads `dp[i - 1][s - num]` from the previous row. Once the table is compressed into one dimension, there is no physical copy of the previous row. The most direct attempt is to process `s` from small to large.

### Break

When we iterate forward, `dp[s - num]` may already have been set to `True` **during the current round for this same `num`**. That reuses one element and violates the requirement that each element may be used at most once.

### Replace the Previous Version with a Forward 1D Version

For now, change only the state representation and use the intuitive forward loop:

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)

    if total % 2 == 1:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for s in range(num, target + 1):
            dp[s] = dp[s] or dp[s - num]

    return dp[target]
```

### Check This Change

Use `nums = [1,1,4]` as a counterexample. The total is `6`, so the target is `3`. The correct answer is `False`: the two `1`s can form only `2`, while `4` is larger than the target.

But when the first `1` is processed in forward order, the array changes like this:

```text
Initial: [True, False, False, False]
s = 1:  [True, True,  False, False]  # Used one 1
s = 2:  [True, True,  True,  False]  # Read dp[1], just written this round
s = 3:  [True, True,  True,  True]   # Read dp[2], also written this round
```

This failed version therefore uses one `1` as if it were three separate `1`s and returns `True`. The two-dimensional version returns `False` for the same input. The `True` here is evidence of the bug, not the answer to the problem.

### Current Checkpoint

This version can now:

- show why the two-dimensional table can be compressed into a one-dimensional array
- use a concrete trace to locate the same-round reuse caused by forward updates

It still needs:

- `dp[s - num]` to continue representing the state before the current element was added during this round
- an update direction that truly preserves 0/1 semantics and passes `[1,1,4]`

The next step changes only the direction in which `s` is traversed and checks whether backward iteration restores the previous-row semantics.

## Step 5: Iterate Backward to Restore One-Use-Per-Element Semantics

### Problem Pressure

The counterexample in Step 4 has reduced the problem to one place: a one-dimensional array does not retain a previous row, so a forward loop can read a state that was written in the current round. We do not need a new state definition. We only need the smaller sums to remain unchanged until later in the round.

### Current Baseline

The current baseline is the forward version:

```python
for num in nums:
    for s in range(num, target + 1):
        dp[s] = dp[s] or dp[s - num]
```

### Break

When `s` moves from small to large, `s - num` is smaller than `s`, so it may already include the current `num`. That is exactly how the first `1` in `[1,1,4]` gets reused and incorrectly makes target `3` reachable.

### Replace Only the Traversal Direction in the Previous Version

Move `s` downward from `target` to `num`:

```python
for num in nums:
    for s in range(target, num - 1, -1):
        dp[s] = dp[s] or dp[s - num]
```

Why can this version not reuse the current element? When processing `s`, the position `s - num` is smaller. A backward loop has not visited that smaller position yet, so `dp[s - num]` still contains its value from the start of the round, exactly like the previous row in the two-dimensional table.

Put this one change back into the complete function:

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)

    if total % 2 == 1:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]

    return dp[target]
```

### Check This Change

First rerun the counterexample from Step 4:

```python
assert can_partition([1, 1, 4]) is False
```

Then cover the main examples and edge cases:

```python
assert can_partition([1, 5, 11, 5]) is True
assert can_partition([1, 2, 3, 5]) is False
assert can_partition([1, 1]) is True
assert can_partition([2]) is False
assert can_partition([2, 2]) is True
```

At the end of each round for a value `num`, `dp[s]` means: "among the elements processed so far, does some subset sum to `s`?" Backward iteration changes only the write order, not the meaning of the state. It ensures that the smaller sum read during the current round does not yet include the current element.

### Complexity

Let `target = sum(nums) // 2`. Each element scans at most `target` sums, so the time complexity is `O(n * target)`. Keeping only one target-sum array uses `O(target)` extra space.

### Current Checkpoint

This version can now:

- complete the 0/1 choice correctly with one-dimensional state, using each element at most once
- pass the forward version's counterexample, the main examples, and the edge-case tests
- return the final answer in `O(n * target)` time and `O(target)` extra space

It still needs:

- a consolidated review of the invariant, edge cases, and the reason forward iteration is invalid, so the method is easier to transfer to other knapsack problems

The next section only adds correctness and edge-case explanations. It does not change the algorithm.

## Correctness and Edge Cases

### Loop Invariant

After the first `k` elements have been processed, `dp[s]` means: using only those `k` elements, with each element chosen at most once, can we form sum `s`?

- Initially, no elements have been processed. Only the empty subset's sum `0` is reachable, so `dp[0] = True` and every other position is `False`.
- When a new element `num` is processed, `dp[s]` can keep its old value, which means skipping `num`; or it can become `True` from `dp[s - num]`, which means choosing `num`.
- Because `s` is updated from large to small, `dp[s - num]` has not yet been modified by the current `num` in this round. It therefore comes only from the first `k` elements, exactly matching the meaning of "read the previous row" in the two-dimensional transition.

The invariant remains true after every round. When all elements have been processed, `dp[target]` answers exactly whether a subset sums to half of the total.

### Edge-Case Checks

The following cases cover the branches most likely to cause mistakes:

| Input | Result | What it checks |
| --- | --- | --- |
| `[1]` | `False` | Odd total, so return early |
| `[2]` | `False` | Even total, but the only element cannot form target `1` |
| `[1,1]` | `True` | Use each of two equal elements once |
| `[2,2]` | `True` | One element reaches target `2` |
| `[1,1,4]` | `False` | One `1` cannot be reused within the same round |
| `[1,2,3,5]` | `False` | Odd-total gate |
| `[1,5,11,5]` | `True` | Standard partitionable example |

The problem constraints guarantee that `nums` is nonempty and that every element is a positive integer. We therefore do not add branches for an empty array or zero-valued elements; the edge-case discussion stays within the stated input domain.

### Common Mistakes

- Writing the inner loop as `range(num, target + 1)`: this forward update allows the same element to be reused within one round. It matches unbounded-knapsack semantics, not the 0/1 semantics required here.
- Writing the backward range as `range(target, num, -1)`: this skips `s == num`. Use `range(target, num - 1, -1)` instead.
- Checking only whether the total is even: an even total is necessary, but `target` still has to be reachable.

## Summary

The path through this problem can be compressed into four connected questions:

1. If the total is odd, reject the input immediately.
2. If the total is even, rewrite the problem as "can we form `target`?"
3. Each element can be chosen or skipped exactly once, so the state has 0/1 semantics.
4. After one-dimensional compression, iterate backward so every state read in the current round still comes from the previous round.

The key idea is not a line of template code but this chain of cause and effect: **each element can be used only once -> one-dimensional state must not read a position just written in the same round -> the sum loop must run backward.**
