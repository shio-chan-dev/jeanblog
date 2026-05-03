---
title: "LeetCode 90: Subsets II, Derive Layer-Level Deduplication"
date: 2026-05-03T14:25:07+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "subsets", "deduplication", "LeetCode 90"]
---

## Start From Duplicate Branches in `[1,2,2]`

The problem gives an integer array `nums` that may contain duplicates and asks for all unique subsets. The output order does not matter.

The smallest example that exposes the issue is:

```text
nums = [1,2,2]
```

If we copy the `78. Subsets` template directly, two different branches collide:

```text
choose the 2 at index 1 -> [2]
choose the 2 at index 2 -> [2]
```

The branches use different indices, but the value sequence is the same.

So the new problem is not basic backtracking. The real question is:

> How do we skip duplicate branches without deleting valid answers such as `[2,2]`?

This tutorial starts with a correct but wasteful version, then derives sorting plus layer-level deduplication.

## Problem Facts

- Input: `nums`, with `1 <= nums.length <= 10`
- Value range: `-10 <= nums[i] <= 10`
- `nums` may contain duplicates
- Output: all unique subsets

Example:

```text
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```

## Step 1: Reuse the 78 State and Watch What Breaks

The partial answer still needs `path`; each layer still needs `start` so it only chooses to the right.

Start with the core version from `78. Subsets`:

```python
def dfs(start: int) -> None:
    res.append(path.copy())

    for i in range(start, len(nums)):
        path.append(nums[i])
        dfs(i + 1)
        path.pop()
```

This version is correct when all values are distinct.

Now this version can:

- Enumerate every index combination.
- Keep `[2,2]`, because the two `2`s come from different indices.

It still lacks:

- A way to distinguish a valid repeated value from a duplicate branch at the same layer.

## Step 2: First Build a Correct but Wasteful Version With `seen`

In the previous version, do not optimize yet. To make the answer correct, convert each `path` into a tuple and store it in `seen`.

Add `seen`:

```python
seen: set[tuple[int, ...]] = set()
```

Replace the collection rule with:

```python
state = tuple(path)
if state not in seen:
    seen.add(state)
    res.append(path.copy())
```

The first complete correct version is:

```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    seen: set[tuple[int, ...]] = set()

    def dfs(start: int) -> None:
        state = tuple(path)
        if state not in seen:
            seen.add(state)
            res.append(path.copy())

        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res
```

Now this version can:

- Return unique subsets.
- Preserve valid results such as `[2,2]`.

It still lacks:

- It still generates duplicate branches before throwing duplicates away.
- If input is `[2,1,2]`, equal values are not adjacent, so branch-level deduplication is hard.

## Step 3: Sort First So Equal Values Become Adjacent

The bottleneck in the previous version is that duplicate branches are already built. To skip them before entering recursion, equal values must be next to each other.

In the previous version, add sorting before DFS:

```python
nums.sort()
```

For `[2,1,2]`, sorting gives:

```text
[1,2,2]
```

Once equal values are adjacent, if the current layer already tried the first `2`, it can recognize the second `2` as an equivalent same-layer branch.

Now this version can:

- Group equal values together.
- Keep correctness through `seen`.

It still lacks:

- The actual rule that skips duplicate branches before they are built.

## Step 4: Skip Only Duplicate Candidates in the Same Layer

Now replace “generate then deduplicate” with “skip before generating”.

The key condition is:

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

Why `i > start`?

- `i > start` means this candidate is not the first candidate in the current layer.
- `nums[i] == nums[i - 1]` means an equal value has already opened a branch in this layer.

Why not `i > 0`?

Because deeper layers must still be allowed to choose the second `2` to form `[2,2]`.

For `[1,2,2]`:

```text
First layer:
  i = 1 chooses the first 2 -> valid, produces [2]
  i = 2 sees another 2 and i > start -> skip duplicate [2]

Inside the [2] branch:
  start = 2
  i = 2 chooses the second 2 -> i == start, do not skip, produce [2,2]
```

Now `seen` is no longer needed.

The final complete solution is:

```python
class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res: list[list[int]] = []
        path: list[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())

            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue

                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res


if __name__ == "__main__":
    ans = Solution().subsetsWithDup([1, 2, 2])
    print(ans)
```

Now this version can:

- Skip duplicate branches before entering recursion.
- Keep `[2,2]`.
- Avoid extra `seen` storage for answer deduplication.

It still lacks:

- A guaranteed output order; the problem does not require one.

## Slow Branch Trace

After sorting, `nums = [1,2,2]`.

At the first layer `dfs(0)`:

```text
collect []
i = 0, choose 1 -> path = [1]
i = 1, choose the first 2 -> path = [2]
i = 2, second 2 equals previous 2 and i > start, skip
```

Inside the `[2]` branch, at `dfs(2)`:

```text
start = 2
i = 2, here i == start, do not skip
choose the second 2 -> path = [2,2]
```

So the rule is not “duplicate values cannot be used”. The real rule is:

> In the same layer, equal values open only one branch; deeper layers may still use later duplicates.

## Correctness

Invariant:

- Sorting makes equal values adjacent.
- In the same recursion layer, if an equal value has already opened a branch, a later equal value would generate duplicate value sequences.
- `i > start` restricts the skip to the current layer only.

Why nothing is missed:

- Each value group still keeps the first candidate branch in every layer.
- If an answer needs multiple equal values such as `[2,2]`, the later equal value appears in a deeper layer where `i == start`, so it is not skipped.

Why nothing is duplicated:

- Duplicate subsets come from same-layer equal-value branches.
- The skip rule removes all but the first such branch.

## Complexity

- Sorting costs `O(n log n)`.
- There can still be up to `2^n` subsets.
- Copying collected paths costs `O(n * 2^n)` overall.
- The recursion stack and `path` use `O(n)` extra space, excluding output.

## Common Mistakes

- Writing `i > 0`, which can incorrectly remove `[2,2]`.
- Forgetting to sort, so equal values are not adjacent.
- Thinking duplicate values are forbidden; only duplicate same-layer branches are forbidden.
- Keeping both `seen` and layer-level deduplication, which leaves two competing dedup mechanisms.

## Summary

- `90. Subsets II` is `78. Subsets` plus duplicate handling.
- A `seen` version is correct but wastes search.
- Sorting is the precondition for branch-level deduplication.
- `if i > start and nums[i] == nums[i - 1]` means layer-level deduplication, not value-level deletion.

## References

- LeetCode 78: Subsets
- LeetCode 90: Subsets II
- LeetCode 40: Combination Sum II

## Notes

- Problem facts, examples, and constraints were taken from the existing repository draft for LeetCode 90.
- Python is used as the implementation language for the guided build.
