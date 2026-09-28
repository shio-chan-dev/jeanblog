---
title: "LeetCode 300: Longest Increasing Subsequence, Deriving Sequence DP from Ending at i"
date: 2026-09-22T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "sequence DP", "longest increasing subsequence", "LeetCode 300"]
---

## Problem

### Input and Output

- Input: an integer array `nums`
- Output: return the length of the longest strictly increasing subsequence in `nums`
- A subsequence preserves the relative order of the original array, but it does not need to be contiguous, so elements may be skipped
- If the selected indices are `i_1 < i_2 < ... < i_k`, they must also satisfy `nums[i_1] < nums[i_2] < ... < nums[i_k]`
- The subsequence is non-empty, so at least one element must be selected
- Constraints: `1 <= nums.length <= 2500`, `-10^4 <= nums[i] <= 10^4`

### Examples

```text
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: [2,5,7,101] is a strictly increasing subsequence of length 4; its elements do not need to be contiguous in the original array.
```

```text
Input: nums = [0,1,0,3,2,3]
Output: 4
Explanation: [0,1,2,3] has length 4; [0,1,3] is also valid, but shorter.
```

This article uses Python only. We start with the local question "how long can a subsequence ending at each position be?" and gradually turn it into a runnable solution.

## Step 1: First Ask Which Position the Subsequence Ends At

Start with the first example:

```text
Index:  0   1  2  3  4  5    6   7
Value: 10   9  2  5  3  7  101  18
```

The answer `[2,5,7,101]` skips `9` and `3`. When we reach the position whose value is `7`, it is not enough to ask, "Which value is immediately to its left?" Instead, we need to ask:

> Which earlier, smaller values can be valid predecessors, and how long can the subsequence become after appending 7?

**The current baseline is**: we know that the problem asks for the longest strictly increasing subsequence in the entire array, but we do not yet have a state that describes the result for one particular position.

**Where does this break?** If we enumerate whole candidates directly, it is easy to confuse a subsequence with a contiguous subarray. We also cannot reuse the results ending at `2`, `5`, or `3` to determine the result ending at `7`.

**Make one change to the current problem**: shrink it into a question indexed by the ending position.

```text
end_at_i = the length of the longest strictly increasing subsequence
           that must select nums[i] and use nums[i] as its final element
```

For example, at index `5`, whose value is `7`:

- `2 -> 5 -> 7` is valid;
- `2 -> 3 -> 7` is also valid;
- `10 -> 7` is invalid because `10 < 7` is false;
- `7` alone can always form a subsequence of length `1`.

At index `6`, whose value is `101`, we can continue from `7`. Instead of asking for the global longest length all at once, we first answer the local question for every possible ending position.

**Check this change**: return to `[10,9,2,5,3,7,101,18]`. The `end_at_i` question always requires the current element to be the final element and preserves the original index order. It allows elements to be skipped, but it does not allow indices to run backward or non-increasing values to be appended.

**This version can now**:

- distinguish a subsequence from a contiguous subarray;
- associate each position with a smaller problem that must end there;
- show that the result ending at `7` needs to reuse results from earlier positions.

**It still lacks**:

- a runnable state for every position;
- the minimum value of `end_at_i` when there is no valid predecessor;
- an array representation of this state and a base case for a single element.

## Step 2: With No Predecessor, One Element Has Length 1

We now know that each position needs its own `end_at_i`. Start with the simplest case: no earlier element can be appended before the current one.

**The current baseline is**: Step 1 provides only a verbal state definition; there is no array that stores these states yet.

**Where does this break?** Without a valid predecessor, the state has no starting value. Even if we later find a smaller element, we would not know what value to increment.

**Add one base case to the previous version**: any single element is a strictly increasing subsequence of length `1`. Initialize every position to `1`. To inspect this state table first, the current function temporarily returns the array; this is not yet the final integer required by the problem.

```python
def length_of_lis(nums: list[int]) -> list[int]:
    n = len(nums)
    dp = [1] * n
    return dp
```

Here, `dp[i]` is the code representation of `end_at_i` from Step 1:

```text
dp[i] = the length of the longest strictly increasing subsequence
        that must end at nums[i]
```

**Check this change**: run only the initialization and inspect several cases with no usable predecessor.

```python
assert length_of_lis([7]) == [1]
assert length_of_lis([7, 7]) == [1, 1]
assert length_of_lis([10, 9, 2]) == [1, 1, 1]
```

The repeated values `[7, 7]` still produce two `1`s because strict increase requires the previous value to be smaller than the next value. No connection between positions has been made yet.

**This version can now**:

- give every position a valid single-element starting state;
- turn `end_at_i` into the runnable state `dp[i]` for the first time;
- assign the correct local length `1` when no predecessor exists.

**It still lacks**:

- a check for whether an earlier position can precede the current one;
- a way to extend a length-`1` state into a longer subsequence;
- an update that considers every earlier and smaller value.

## Step 3: Append the Current Element Only After an Earlier, Smaller Value

Now look at the position whose value is `7`. Before it are the larger values `10` and `9`, as well as the smaller values `2`, `5`, and `3`. Looking only at the adjacent position, or ignoring the value comparison, would create invalid sequences.

**The current baseline is**: Step 2 places `1` at every position, so every element can form a subsequence by itself, but every `dp[i]` is still `1`.

**Where does this break?** Any subsequence longer than `1` must append the current element after an earlier valid subsequence. We have not enumerated those candidate predecessors or checked whether they can be followed by the current value.

**Add one transition to the previous version**: when processing position `i`, inspect every earlier position `j`. Position `j` can be a predecessor only when `nums[j] < nums[i]`. After appending `nums[i]`, the candidate length is `dp[j] + 1`.

```python
def length_of_lis(nums: list[int]) -> list[int]:
    n = len(nums)
    dp = [1] * n

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return dp
```

The `max` here updates only the local state ending at `nums[i]`; it does not yet compute the final answer for the entire array. The condition `j < i` preserves the original order, while `nums[j] < nums[i]` enforces strict increase rather than allowing equal values.

**Check this change**: when the example reaches `i = 5`, whose value is `7`, inspect each predecessor:

| `j` | `nums[j]` | Can it precede `7`? | Candidate length |
| ---: | ---: | --- | ---: |
| 0 | 10 | No, because `10 < 7` is false | - |
| 1 | 9 | No, because `9 < 7` is false | - |
| 2 | 2 | Yes | `dp[2] + 1 = 2` |
| 3 | 5 | Yes | `dp[3] + 1 = 3` |
| 4 | 3 | Yes | `dp[4] + 1 = 3` |

Therefore, `dp[5] = 3`, for example from `2 -> 5 -> 7`. Also check repeated values:

```python
assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == [1, 1, 1, 2, 2, 3, 4, 4]
assert length_of_lis([7, 7, 7]) == [1, 1, 1]
```

**This version can now**:

- scan all candidate predecessors before each position;
- append the current element only after an earlier, smaller value;
- compute the longest length ending at every position rather than considering only adjacent elements.

**It still lacks**:

- a single answer, because `dp` currently stores one local result for every ending position;
- a way to account for the longest subsequence ending at any position, not necessarily the last one;
- an aggregation of all `dp[i]` values into the integer required by the problem.

## Step 4: The Answer Is the Maximum over All Ending States

Step 3 computes the length ending at every position, but the problem asks for the longest length anywhere in the array. The longest subsequence does not have to end at the last element.

**The current baseline is**: the function returns the entire `dp` array, and each position stores a correct local result.

**Where does this break?** Returning `dp[-1]` assumes that an optimal subsequence must use the final element. For example:

```text
nums = [1, 2, 3, 0]
dp   = [1, 2, 3, 1]
```

The longest subsequence `[1,2,3]` ends at index `2`, so the answer is `3`; however, `dp[-1]` is only `1`.

**Change only the answer aggregation in the previous version**: replace the temporary `return dp` with `return max(dp)`. This `max` compares all possible ending positions, while the initialization and predecessor transition already established above remain unchanged. To make the code directly submit-ready, place the same logic inside LeetCode's required `Solution` method; the wrapper adds no new algorithmic logic.

```python
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```

**Check this change**:

```python
assert Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert Solution().lengthOfLIS([0, 1, 0, 3, 2, 3]) == 4
assert Solution().lengthOfLIS([7, 7, 7, 7, 7, 7, 7]) == 1
assert Solution().lengthOfLIS([1, 2, 3, 0]) == 3
```

The final assertion specifically checks a case in which the optimal subsequence ends before the last position. It ensures that the function returns the maximum across all `dp[i]`, not merely the final state.

**This version can now**:

- compute the longest strictly increasing subsequence ending at every position;
- select the global answer from all local states;
- run as a complete LeetCode Python submission.

**It still lacks**:

- a complete table that recomputes every state update step by step;
- a proof that enumerating every earlier predecessor covers all valid subsequences;
- a check of the `O(n^2)` time, `O(n)` space, and boundary cases.

## Step 5: Walk Through the State Table Slowly

**The current baseline is**: Step 4 provides a submit-ready `O(n^2)` implementation that returns the correct answer for the standard example.

**Where does this break?** A passing implementation alone does not confirm the meaning of every `dp[i]`, the coverage of the predecessor scan, why `max(dp)` is necessary, or whether `O(n^2)` fits the problem constraints.

**Add only verification evidence to the previous version**: do not change any state or transition. Add a complete state table, an invariant proof, a complexity check, and boundary checks. No new algorithm is introduced here.

**Check this change**: compute every state for `nums = [10,9,2,5,3,7,101,18]`:

| `i` | `nums[i]` | Earlier indices that can precede `i` | Computation of `dp[i]` | `dp[i]` |
| ---: | ---: | --- | --- | ---: |
| 0 | 10 | None | Single-element subsequence | 1 |
| 1 | 9 | None | Single-element subsequence | 1 |
| 2 | 2 | None | Single-element subsequence | 1 |
| 3 | 5 | 2 | `dp[2] + 1 = 2` | 2 |
| 4 | 3 | 2 | `dp[2] + 1 = 2` | 2 |
| 5 | 7 | 2, 3, 4 | `max(2, 3, 3) = 3` | 3 |
| 6 | 101 | 0, 1, 2, 3, 4, 5 | `max(2, 2, 2, 3, 3, 4) = 4` | 4 |
| 7 | 18 | 0, 1, 2, 3, 4, 5 | `max(2, 2, 2, 3, 3, 4) = 4` | 4 |

The final state is:

```text
dp = [1, 1, 1, 2, 2, 3, 4, 4]
```

Therefore, `max(dp) = 4`, matching the example output.

## Correctness

Express the meaning of `dp[i]` as an invariant:

> After processing index `i`, `dp[i]` is the length of the longest strictly increasing subsequence that must select `nums[i]` and end there.

**Base case**: every element alone is a subsequence of length `1`, so initializing `dp[i] = 1` is correct.

**Transition**: if a valid subsequence ends at `nums[i]`, it either contains only `nums[i]`, or it has a final predecessor `j` satisfying `j < i` and `nums[j] < nums[i]`. In the second case, its length is `dp[j] + 1`. The code enumerates every such `j` and takes the maximum, so it misses no valid predecessor and never appends an element in reverse index order or after an equal value.

**Answer aggregation**: the problem does not require the longest subsequence to end at the final position, so any valid ending position may contain the answer. `max(dp)` selects the longest among them.

## Complexity

- Time complexity: the nested loops over `i` and `j` inspect at most `n(n - 1) / 2` index pairs, so the time complexity is `O(n^2)`.
- Extra space complexity: the algorithm stores only the length-`n` `dp` array, so the extra space complexity is `O(n)`.
- The constraint is `n <= 2500`, so this sequence DP solution is sufficient for the problem limits.

## Boundary Cases and Common Mistakes

- A single-element array has answer `1`, which the base case already covers.
- Equal values cannot be connected. The condition must be the strict `nums[j] < nums[i]`, not `<=`.
- In a decreasing array, every `dp[i]` remains `1`.
- A subsequence does not need to be contiguous; only the indices must remain increasing. Comparing adjacent elements alone misses valid predecessors.
- `dp[i]` means "must end at `i`", not "the global answer among the first `i` elements". These meanings cannot be mixed.
- Return `max(dp)` rather than assuming that the final position is the optimal ending position.

**This version can now**:

- recompute every predecessor choice with the state table;
- explain why `dp[i]`, the strict `<` condition, and `max(dp)` are correct;
- determine from the constraints when the `O(n^2)` version is appropriate.

**It still lacks**:

- a derivation of an `O(n log n)` optimization using `tails` and binary search;
- a solution for input sizes beyond this problem's constraints, which should be derived in a separate tutorial from new performance pressure rather than inserted into this checkpoint.

## Summary

- Break the global problem into one local problem per ending position: `dp[i]`.
- Every position has at least a single-element subsequence, so initialize it to `1`.
- Transition only from an earlier, smaller predecessor: `dp[i] = max(dp[i], dp[j] + 1)`.
- The longest subsequence can end anywhere, so the answer is `max(dp)`.
- Under the constraint `n <= 2500`, the nested-loop `O(n^2)` sequence DP is sufficient.

## References and Further Practice

- [LeetCode 300: Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- The "state by ending position" idea in this article transfers to other sequence DP problems. An optimization for larger inputs should be derived in a separate checkpoint so that the state meaning is not skipped.
