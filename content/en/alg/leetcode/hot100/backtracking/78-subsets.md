---
title: "LeetCode 78: Subsets, Derive the startIndex Backtracking Template"
date: 2026-05-03T14:25:07+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "subsets", "DFS", "LeetCode 78"]
---

## Start From the Search Tree for `[1,2]`

The problem gives an integer array `nums` with distinct elements and asks for all possible subsets. The output order does not matter, and subset element order is not the point.

The smallest branching example is:

```text
nums = [1,2]
```

The answer should contain:

```text
[], [1], [1,2], [2]
```

It should not contain `[2,1]`. That means we are not generating all permutations. We are choosing elements into a set-like result where each combination appears once.

The construction tree looks like this:

```text
[]
|- [1]
|  |- [1,2]
|- [2]
```

Two facts fall out of this tree:

- Every node is already a valid subset.
- After choosing `1`, the next layer may only look to the right, at `2`.

This tutorial builds one minimal Python solution.

## Problem Facts

- Input: `nums`, with `1 <= nums.length <= 10`
- Value range: `-10 <= nums[i] <= 10`
- All elements in `nums` are distinct
- Output: all possible subsets

Example:

```text
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

## Step 1: Define What One Recursion Layer Means

If the current partial answer is `path`, the smaller problem is:

> Starting from some index, choose zero or more remaining elements to extend `path`.

That starting index must be part of the state. Without it, `[1,2]` and `[2,1]` can be generated from different branches.

Start with the smallest skeleton. It keeps two pieces of state:

- `path`: the elements chosen on the current branch
- `res`: all subsets collected so far

```python
def subsets(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []

    def dfs(start: int) -> None:
        pass

    dfs(0)
    return res
```

Now this version can:

- Define the outer function and recursive function.
- Reserve `start` as the boundary for the current layer.

It still lacks:

- A rule for collecting answers.
- A way to enumerate choices in the current layer.

## Step 2: Every Node Is an Answer, So Collect `path` First

In the previous version, add the first rule inside `dfs`:

```python
def dfs(start: int) -> None:
    res.append(path.copy())
```

Why collect immediately?

Subsets do not require a fixed length or target sum. If `path` was formed by valid choices, it is already a valid subset.

Why `copy()`?

Because `path` will keep changing during recursion. Appending `path` itself would leave all saved answers pointing to the same mutable list.

Now this version can:

- Collect the empty subset `[]`.
- Collect the current partial subset whenever a recursion layer starts.

It still lacks:

- Downward expansion; it can only collect the empty subset.

## Step 3: Enumerate Only From `start` to the Right

In the previous version, add the candidate range for the current layer:

```python
def dfs(start: int) -> None:
    res.append(path.copy())

    for i in range(start, len(nums)):
        pass
```

The range `range(start, len(nums))` means:

- This layer may only choose elements at index `start` or later.
- Earlier elements have already been handled by the current path.

For `nums = [1,2,3]`:

- `dfs(0)` may try `1,2,3`
- after choosing `1`, `dfs(1)` may only try `2,3`
- after choosing `2`, `dfs(2)` may only try `3`

Now this version can:

- Define the available candidates for each layer.
- Use `start` to prevent order-based duplicates.

It still lacks:

- Actually adding a candidate into the current path.

## Step 4: Choose One Element and Recurse on the Suffix

In the previous version, replace the empty loop body with choosing and recursing:

```python
for i in range(start, len(nums)):
    path.append(nums[i])
    dfs(i + 1)
```

The call `dfs(i + 1)` is the key rule:

> After choosing `nums[i]`, the next layer may only choose elements to its right.

Writing `dfs(start + 1)` would be wrong when `i != start`; the next boundary must follow the chosen index, not the old layer boundary.

Now this version can:

- Generate downward paths such as `[] -> [1] -> [1,2]`.
- Keep subset elements in increasing original-index order.

It still lacks:

- State restoration after returning from recursion.

## Step 5: Undo the Choice and Finish the Solution

In the previous version, add the undo operation after the recursive call:

```python
for i in range(start, len(nums)):
    path.append(nums[i])
    dfs(i + 1)
    path.pop()
```

This solves the branch-isolation problem.

For `nums = [1,2,3]`:

```text
path = [1,2,3] is collected
pop 3 -> path = [1,2]
pop 2 -> path = [1]
same layer now tries 3 -> path = [1,3]
```

Without `path.pop()`, later branches would inherit choices that belong to earlier branches.

This version is the complete runnable solution:

```python
class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res: list[list[int]] = []
        path: list[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())

            for i in range(start, len(nums)):
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res


if __name__ == "__main__":
    ans = Solution().subsets([1, 2, 3])
    print(ans)
```

Now this version can:

- Enumerate every subset.
- Avoid order-based duplicates such as `[2,1]`.
- Restore `path` after each branch.

It still lacks:

- Duplicate-value handling. That belongs to `90. Subsets II`.

## Slow Branch Trace

For `nums = [1,2,3]`, follow one branch:

```text
dfs(0), path = []
collect []

i = 0, choose 1
path = [1]
dfs(1), collect [1]

i = 1, choose 2
path = [1,2]
dfs(2), collect [1,2]

i = 2, choose 3
path = [1,2,3]
dfs(3), collect [1,2,3]

return, pop 3 -> [1,2]
return, pop 2 -> [1]
same layer tries i = 2, choose 3 -> [1,3]
```

The important invariants are:

- `path` is always the current branch.
- `start` always points to the next allowed suffix.

## Correctness

Invariant:

- At the start of `dfs(start)`, `path` is a valid subset whose original indices are strictly increasing.
- The current layer only chooses indices from `start` onward, so no branch can create a reversed duplicate.

Why nothing is missed:

- Any subset can be represented by its chosen original indices in increasing order.
- DFS can follow exactly that increasing sequence, so the subset will be visited.

Why nothing is duplicated:

- Every branch has strictly increasing indices.
- The same set of indices has only one increasing order.

## Complexity

- There are `2^n` subsets.
- Copying paths across all collected answers costs `O(n * 2^n)`.
- The recursion stack and `path` use `O(n)` extra space, excluding output.
- The output itself uses `O(n * 2^n)` space.

## Common Mistakes

- Collecting only at leaves, which misses `[]`, `[1]`, `[1,2]`, and other internal nodes.
- Appending `path` without `copy()`, which lets later backtracking mutate saved answers.
- Calling `dfs(start + 1)` instead of `dfs(i + 1)`.
- Forgetting `path.pop()`, which leaks state across branches.

## Summary

- In subsets, every search-tree node is an answer.
- `path` stores the current branch; `start` stores the next allowed index.
- `dfs(i + 1)` removes order duplicates.
- `path.pop()` restores branch-local state.

## References

- LeetCode 78: Subsets
- LeetCode 90: Subsets II
- LeetCode 46: Permutations

## Notes

- Problem facts, examples, and constraints were taken from the existing repository draft for LeetCode 78.
- Python is used as the implementation language for the guided build.
