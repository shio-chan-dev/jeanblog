---
title: "LeetCode 45: Jump Game II From Reachability to Minimum Jumps"
date: 2026-07-08T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "greedy", "array", "reachability boundary", "LeetCode 45"]
description: "Solve LeetCode 45 Jump Game II in Python by upgrading the farthest reach idea from LeetCode 55 into a minimum-jump layer boundary with current_end and farthest."
keywords: ["LeetCode 45", "Jump Game II", "greedy", "minimum jumps", "current_end", "farthest", "Python"]
---

> **Subtitle / Summary**
> Jump Game II is not just reachability. It asks how many coverage layers are needed to reach the last index.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `greedy`, `array`, `reachability boundary`
- **SEO keywords**: LeetCode 45, Jump Game II, greedy, current_end, farthest, minimum jumps
- **Meta description**: A pressure-first Python guide to LeetCode 45 that derives the minimum-jump greedy scan from coverage layers.

---

## Problem Requirement

You are given an integer array `nums`.

You start at index `0`.

`nums[i]` means:

```text
from index i, you may jump at most nums[i] steps forward
```

Return:

```text
the minimum number of jumps needed to reach the last index
```

### Input and Output

- Input: `nums: List[int]`
- Output: `int`
- You start at index `0`.
- Each number is the maximum jump length, not the exact jump length.
- The problem guarantees that the last index is reachable.
- You only need the minimum jump count; you do not need to return a path.

### Examples

```text
Input: nums = [2,3,1,1,4]
Output: 2
```

One minimum path is:

```text
0 -> 1 -> 4
```

Jump from index `0` to index `1`, then from index `1` to the last index.

```text
Input: nums = [2,3,0,1,4]
Output: 2
```

Even though there is a `0` in the middle, this still works:

```text
0 -> 1 -> 4
```

### Constraints

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 1000`
- `nums[n - 1]` is guaranteed to be reachable

## Step 1: 45 Does Not Ask Whether We Can Reach

After LeetCode 55, we already know how to maintain:

```text
farthest = the farthest index covered by all reachable positions so far
```

That is enough to answer:

```text
Can we reach the last index?
```

But LeetCode 45 asks a different question:

```text
What is the minimum number of jumps?
```

The current baseline is:

```text
Maintain one farthest boundary and decide reachability.
```

This baseline breaks because:

> `farthest` can tell us whether the covered range touches the end, but it does not tell us when the jump count should increase.

Look at:

```text
nums = [2,3,1,1,4]
```

The answer is `2`:

```text
0 -> 1 -> 4
```

This is not asking for any path, and it is not asking whether the end is reachable.

The real question is:

```text
How many coverage layers are needed to cover the last index?
```

Now this version can:

- separate LeetCode 45 from LeetCode 55
- explain why a single `farthest` boundary is not enough for counting jumps
- prepare the next question: where does one jump's coverage end?

It still lacks:

- a layer model that explains one jump and the next jump.

## Step 2: Treat One Jump as One Coverage Layer

The current baseline is:

```text
We need the minimum number of jumps, which means the minimum number of coverage layers.
```

This breaks when:

> Guessing one path, such as `0 -> 1 -> 4`, can produce an answer, but it does not prove that the answer is minimum.

Use the same example:

```text
nums = [2,3,1,1,4]
```

Start at index `0`.

Layer `0` is the start:

```text
step 0: [0]
```

From index `0`, `nums[0] = 2`, so one jump can cover:

```text
step 1: [1..2]
```

Do not pick one concrete path yet.

Scan every index inside layer `1`:

```text
i = 1, nums[1] = 3, farthest reach is 4
i = 2, nums[2] = 1, farthest reach is 3
```

The best expansion inside this layer comes from index `1`, so the next layer can reach index `4`.

Therefore:

```text
step 2 can reach the last index
```

This is a BFS idea, but we do not need to build a graph.

Each layer is a contiguous range in the array:

```text
step 0: [0]
step 1: [1..2]
step 2: [3..4] or farther
```

Now this version can:

- avoid guessing one path first
- treat the current jump count as one whole coverage layer
- explain why `[2,3,1,1,4]` needs two jumps

It still lacks:

- compact state for the layer boundaries.

## Step 3: Use current_end and farthest as Two Boundaries

The current baseline is:

```text
Each jump covers one contiguous layer of indices.
```

This breaks because:

> The layer idea is correct, but explicitly storing each layer is heavier than necessary. During a left-to-right scan, boundaries are enough.

We need two boundaries:

```text
current_end = the right boundary covered by the current jump count
farthest    = the farthest right boundary reachable by one more jump while scanning this layer
```

Their jobs are different.

`current_end` answers:

```text
When has the current layer been fully scanned?
```

`farthest` answers:

```text
After scanning this layer, how far can the next layer reach?
```

First only focus on the `farthest` update.

When scanning index `i`, if `i` belongs to the current layer, then one more jump from `i` can reach:

```text
i + nums[i]
```

So update:

```python
farthest = max(farthest, i + nums[i])
```

Trace `[2,3,1,1,4]`:

```text
start:
current_end = 0
farthest = 0

i = 0:
farthest = max(0, 0 + nums[0]) = 2
```

This means:

```text
from the current layer, the next jump can reach index 2
```

When scanning the next layer:

```text
i = 1:
farthest = max(2, 1 + nums[1]) = 4

i = 2:
farthest = max(4, 2 + nums[2]) = 4
```

This means:

```text
after this layer, the next jump can reach index 4
```

Now this version can:

- use `current_end` as the boundary of the current jump layer
- use `farthest` as the boundary of the next jump layer
- derive the `farthest` update from the indices inside the current layer

It still lacks:

- the exact moment when the layer is finished and `steps` should increase.

## Step 4: When i == current_end, This Layer Is Finished

The current baseline is:

```text
While scanning, maintain current_end and farthest.
```

This breaks because:

> `farthest` may keep expanding, but `steps` should not increase at every index. A jump is counted only when the current coverage layer has been fully scanned.

The right boundary of the current layer is:

```text
current_end
```

So when:

```text
i == current_end
```

it means:

```text
all positions covered by the current jump count have been scanned
```

Now we must use one more jump:

```python
steps += 1
current_end = farthest
```

The scan is now:

```python
steps = 0
current_end = 0
farthest = 0

for i in range(len(nums) - 1):
    farthest = max(farthest, i + nums[i])

    if i == current_end:
        steps += 1
        current_end = farthest
```

Why stop at:

```python
range(len(nums) - 1)
```

Because once we are at the last index, no more jump is needed.

If the loop also scans the last index, it may add one extra jump at the destination.

Trace `[2,3,1,1,4]`:

```text
start:
steps = 0, current_end = 0, farthest = 0

i = 0:
farthest = max(0, 0 + 2) = 2
i == current_end, layer 0 is finished
steps = 1
current_end = 2

i = 1:
farthest = max(2, 1 + 3) = 4
i != current_end, do not increment steps

i = 2:
farthest = max(4, 2 + 1) = 4
i == current_end, layer 1 is finished
steps = 2
current_end = 4
```

Now two jumps can cover the last index.

Now this version can:

- increment `steps` only at a layer boundary
- move into the next layer with `current_end = farthest`
- avoid counting an extra jump at the destination

It still lacks:

- the complete LeetCode method, runnable checks, complexity, and common mistakes.

## Step 5: Complete LeetCode Solution

The current baseline is:

```text
Use current_end and farthest to scan layers, and increment steps when i == current_end.
```

This breaks because:

> The rule still needs to be assembled into LeetCode's `jump(nums)` method and checked against boundary cases.

Complete code:

```python
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                steps += 1
                current_end = farthest

        return steps
```

The problem guarantees reachability, so the final code does not need an unreachable branch.

Check the second example:

```text
nums = [2,3,0,1,4]
```

Trace:

```text
i = 0:
farthest = 2
reach current_end, steps = 1, current_end = 2

i = 1:
farthest = max(2, 1 + 3) = 4

i = 2:
farthest = max(4, 2 + 0) = 4
reach current_end, steps = 2, current_end = 4
```

Index `2` is `0`, but index `1` in the same layer has already expanded the next layer to the end.

Now this version can:

- return the minimum number of jumps
- handle `len(nums) == 1`
- handle a zero inside the current layer when another index extends farther

## Correctness Intuition

This problem is a compressed BFS.

Each jump corresponds to one layer:

```text
all indices reachable with the current steps count
```

While scanning one layer, do not commit to one concrete next index.

Only record:

```text
how far the next layer can reach
```

That is `farthest`.

When `i == current_end`, the current layer has been fully scanned.

If we still need to move forward, we must spend one more jump and enter the next layer:

```text
steps += 1
current_end = farthest
```

Because every layer represents all positions reachable with one more jump, the first layer boundary that covers the last index gives the minimum jump count.

## Complexity

The array is scanned once:

```text
O(n)
```

Only three variables are stored:

```text
steps
current_end
farthest
```

Space complexity:

```text
O(1)
```

## Common Mistakes

### 1. Scanning the Last Index

Do not write:

```python
for i in range(len(nums)):
```

Once the last index is reached, no more jump is needed.

Use:

```python
for i in range(len(nums) - 1):
```

### 2. Incrementing steps Whenever farthest Changes

`farthest` is the next layer's farthest boundary.

It becoming larger does not mean one jump has finished.

Only when:

```text
i == current_end
```

has the current layer finished, so only then should `steps` increase.

### 3. Mixing current_end and farthest

`current_end` means:

```text
the right boundary covered by the current jump count
```

`farthest` means:

```text
the right boundary reachable by the next jump
```

They should not be synchronized at every index.

Only when the current layer ends do we run:

```python
current_end = farthest
```

## Runnable Checks

```python
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                steps += 1
                current_end = farthest

        return steps


def check() -> None:
    s = Solution()
    assert s.jump([2, 3, 1, 1, 4]) == 2
    assert s.jump([2, 3, 0, 1, 4]) == 2
    assert s.jump([0]) == 0
    assert s.jump([1, 2]) == 1
    assert s.jump([1, 1, 1, 1]) == 3


check()
```

## Takeaway

The precise greedy idea in Jump Game II is not:

```text
always jump to the farthest immediate index
```

It is:

```text
scan the entire layer covered by the current jump count,
use farthest to record how far the next jump can cover,
and increment steps when i == current_end
```

The three variables mean:

```text
steps       = how many jumps have been used
current_end = the right boundary covered by the current jump count
farthest    = the right boundary reachable by the next jump
```

That is the upgrade from LeetCode 55's reachability boundary to LeetCode 45's minimum-jump boundary.
