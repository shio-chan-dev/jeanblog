---
title: "LeetCode 139: Word Break, Starting from Prefix Reachability"
date: 2026-09-22T12:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "dynamic programming", "string", "prefix reachability", "LeetCode 139"]
---

## Problem

### Input and Output

- Input: a string `s` and a list of strings `wordDict`
- Every element in `wordDict` is an available word, and the same word may be reused
- Output: return `True` if `s` can be split into one or more dictionary words; otherwise, return `False`
- The split must cover the entire string without skipping or reordering characters

### Examples

```text
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: True
Explanation: "leetcode" can be split into "leet" + "code"
```

```text
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: True
Explanation: "apple" + "pen" + "apple"; the same word may be reused
```

```text
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: False
Explanation: some prefixes can be formed, but the split cannot cover the entire string
```

### Constraints

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` and `wordDict[i]` contain only lowercase English letters
- All strings in `wordDict` are unique

## Step 1: Make the Completed Prefix Visible

Start with one concrete question. For `s = "leetcode"`, before deciding whether the whole string can be split, can we record how much of it has already been completed?

### Pressure: The Answer Depends on an Intermediate Boundary

The valid split in this example is:

```text
"leetcode" = "leet" + "code"
```

Once we have confirmed `"leet"`, the next position to process is boundary `4`. But `"code"` has not been processed yet, so "this prefix works" and "the whole string works" cannot be collapsed into one Boolean value.

### Current Baseline

At this point, we only have the problem requirement. We can imagine trying different split points, but we have nowhere to store the result that a particular prefix has already been completed.

### Where Does This Baseline Break?

Without recording intermediate boundaries, we cannot express this state:

```text
First 4 characters, "leet": already splittable
Remaining substring, "code": not processed yet
```

We need to store prefix feasibility on its own. A prefix is represented by a boundary: boundary `i` corresponds to `s[:i]`. Therefore, `i = 4` means the first four characters, not the character at index `4`.

### Add One State Table to the Current Version

For now, create only the state. Do not match dictionary words yet:

```python
s = "leetcode"

# reachable[i] means that s[:i] can be split completely
reachable = [False] * (len(s) + 1)

# The empty prefix needs no words, so it is a valid starting point
reachable[0] = True
```

That extra slot matters. Although `s` has `8` characters, we must record `9` boundaries from `0` through `8`. `reachable[0]` represents the empty prefix `s[:0]`, while `reachable[8]` represents the entire string `s[:8]`.

### Check This Change

This version has not attached any dictionary word, so boundary `0` should be the only boundary known to be reachable:

```python
assert len(reachable) == len(s) + 1
assert reachable[0] is True
assert all(flag is False for flag in reachable[1:])
print(reachable)
# [True, False, False, False, False, False, False, False, False]
```

### Freeze This Checkpoint

This version can now:

- use `reachable[i]` to mean that prefix `s[:i]` can be split completely
- treat the empty prefix as a valid starting point for later extensions
- distinguish "a prefix is reachable" from "the whole string has an answer"

It still lacks:

- how to match a dictionary word from an already reachable boundary
- how to mark the new ending boundary as reachable after a match

The next step will extend only one starting boundary by one word. It will not jump directly to scanning every boundary.

## Step 2: Take Only the First Word

### Pressure: The Empty Prefix Is Reachable, but No New Boundary Is

Step 1 tells us only that `reachable[0]` is true. Return to `"leetcode"`: if we try different endings from boundary `0`, when may we mark the next boundary as true?

### Current Baseline

The previous version has this boundary table:

```text
reachable[0] = True
reachable[1..8] = False
```

It can represent the starting point, but it has not used the dictionary supplied by the problem.

### Where Does This Baseline Break?

Without checking whether a candidate substring is a dictionary word, we cannot advance from `0` to `4`. This step deliberately studies only one starting boundary. If we scanned every start now, it would be harder to see which exact rule performs the extension.

### Add One Single-Layer Extension to the Previous Version

Store the dictionary as a set, and for now try endings only from `start = 0`:

```python
s = "leetcode"
word_set = {"leet", "code"}

# reachable[i] means that s[:i] can be split completely
reachable = [False] * (len(s) + 1)
reachable[0] = True

start = 0
for end in range(start + 1, len(s) + 1):
    piece = s[start:end]
    if reachable[start] and piece in word_set:
        reachable[end] = True
```

Only one thing happens here: if `s[start:end]` is a dictionary word and `start` is already reachable, mark `end` as reachable. In this example, when `end = 4`, the piece is `"leet"`, so we get `reachable[4] = True`.

### Check This Change

```python
assert reachable[0] is True
assert reachable[4] is True      # 0 -> 4: "leet"
assert reachable[8] is False     # 4 -> 8 has not been scanned yet
print(reachable)
# [True, False, False, False, True, False, False, False, False]
```

`reachable[8]` is still `False`, but that does not mean the answer is false. This version intentionally processes only the fixed starting boundary `0`; it has not continued from the newly reached boundary `4` to try `"code"`.

### Freeze This Checkpoint

This version can now:

- enumerate candidate endings from one already reachable boundary
- use a dictionary match to advance from `0` to `4`
- keep "a prefix is reachable" separate from "the whole string is reachable"

It still lacks:

- extending from later boundaries such as `4`
- applying the same rule to every possible `start` so that multiword splits are covered

The next step will add only the outer rule that scans all reachable starts. That will produce the first complete correct version.

## Step 3: Give the Same Rule to Every Reachable Boundary

### Pressure: `reachable[4]` Is True, but the Answer Is Still Stuck Halfway

Step 2 finds `"leet"`, but because the code fixes `start = 0`, it never gets a chance to try `"code"`. For a multiword split, every newly reached boundary must be allowed to become the next starting point.

### Current Baseline

The previous version performs only this layer:

```text
start = 0
enumerate end from 0
```

The inner substring-matching rule is already correct. What is missing is a scan over later values of `start`.

### Where Does This Baseline Break?

If we always start from `0`, `reachable[4]` may be marked but will never participate in another decision. The second word `"code"` in `"leetcode"` is therefore missed, and `reachable[8]` stays `False`.

### Add Only the Outer Boundary Scan to the Previous Version

Replace the fixed `start = 0` with a loop over all boundaries. Keep the inner rule, "a dictionary match marks its ending boundary," unchanged:

```python
def can_break(s: str, word_set: set[str]) -> bool:
    # reachable[i] means that s[:i] can be split completely
    reachable = [False] * (len(s) + 1)
    reachable[0] = True

    for start in range(len(s)):
        if not reachable[start]:
            continue

        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if piece in word_set:
                reachable[end] = True

    return reachable[len(s)]
```

The only new algorithmic rule is the outer `start` scan: continue only from boundaries that are already reachable. The `end` loop and the `piece in word_set` check are carried over unchanged from Step 2.

### Trace the Boundaries for `"leetcode"`

```text
start = 0  -> match "leet"  -> reachable[4] = True
start = 1,2,3 -> unreachable, skip
start = 4  -> match "code"  -> reachable[8] = True
```

The function therefore returns `reachable[8]`, not merely the result of the first extension.

### Check This Change

```python
assert can_break("leetcode", {"leet", "code"}) is True
assert can_break("applepenapple", {"apple", "pen"}) is True
assert can_break(
    "catsandog",
    {"cats", "dog", "sand", "and", "cat"},
) is False
print("three examples passed")
```

These three assertions cover a two-word connection, reuse of a word, and a case where promising prefixes still cannot cover the final boundary. This is the first version that returns the correct Boolean result for a complete input. For now, it receives an already prepared `word_set`; the final wrapper for list input will come later.

### Freeze This Checkpoint

This version can now:

- continue from every reachable boundary and try subsequent dictionary words
- cover multiword splits with the same transition rule
- return the correct Boolean result for all three problem examples

It still lacks:

- avoiding candidates longer than the longest dictionary word, because the current inner loop tries every ending through the end of the string
- bringing together the `wordDict` list, LeetCode `Solution` wrapper, and complexity explanation

The next step will optimize only the range of candidate endings. It will not change the meaning of `reachable` or its transition rule.

## Step 4: Scan Only Possible Word Lengths

### Pressure: The Correct Version Still Checks Impossible Substrings

Step 3 is correct, but its inner loop runs to the end of the string from every reachable `start`. For example, the longest word in `word_set = {"leet", "code"}` has length `4`. From `start = 0`, candidate substrings of lengths `5`, `6`, through `8` cannot possibly match the dictionary.

### Current Baseline

For every reachable `start`, the previous version runs:

```python
for end in range(start + 1, len(s) + 1):
    piece = s[start:end]
```

The reachability transition is already correct. The waste exists only in the candidate range.

### Where Does This Baseline Break?

The problem guarantees that dictionary words have bounded lengths. A candidate longer than the longest dictionary word can never match `word_set`. Checking it adds work but can never make another boundary reachable.

### Narrow Only the Upper Bound of `end`

First obtain the longest dictionary word length, then look no farther than that distance from each starting boundary:

```python
def can_break(s: str, word_set: set[str]) -> bool:
    # reachable[i] means that s[:i] can be split completely
    reachable = [False] * (len(s) + 1)
    reachable[0] = True
    max_word_len = max(len(word) for word in word_set)

    for start in range(len(s)):
        if not reachable[start]:
            continue

        end_limit = min(len(s), start + max_word_len)
        for end in range(start + 1, end_limit + 1):
            piece = s[start:end]
            if piece in word_set:
                reachable[end] = True

    return reachable[len(s)]
```

For `word_set = {"leet", "code"}`, `max_word_len` is `4`. When `start = 0`, `end` takes only the values `1..4`, which still includes the longest candidate that could match. The rule "a matching word marks its ending boundary" has not changed; only impossible candidates have been removed.

### Check This Change

```python
assert can_break("leetcode", {"leet", "code"}) is True
assert can_break("applepenapple", {"apple", "pen"}) is True
assert can_break(
    "catsandog",
    {"cats", "dog", "sand", "and", "cat"},
) is False

# A valid word whose length equals the maximum must remain inside the range
assert can_break("abcdef", {"abc", "def"}) is True
print("bounded scan examples passed")
```

During the self-check, the same cases were run through both the unbounded Step 3 version and this version, and every result matched. A range trace with `start = 0` and a maximum word length of `4` also confirmed that substrings of lengths `5..8` were no longer checked.

### Freeze This Checkpoint

This version can now:

- preserve every reachability result from Step 3
- check only candidates no longer than the longest dictionary word from each start
- reduce pointless scanning without changing the state transition

It still lacks:

- converting the `wordDict` list and arranging the LeetCode `Solution.wordBreak` method and final tests into a directly submittable form
- a complete invariant, correctness argument, and complexity explanation

The next step will add only the final platform wrapper and proof. It will introduce no new algorithmic logic.

## Step 5: Deliver the Final Submittable Version

### Pressure: The Core Logic Does Not Yet Match the LeetCode Interface

The `can_break` function from Step 4 already returns correct results, but it accepts a prepared `word_set`. The problem interface supplies a `wordDict` list, and LeetCode requires the code inside `Solution.wordBreak`.

### Current Baseline

The previous version has already established every algorithmic part:

- `reachable[0] = True`
- scan each reachable `start`
- check only `end` values within the longest word length
- mark `reachable[end]` when a substring matches a dictionary word

### Where Does This Baseline Break?

It cannot yet be pasted directly into the LeetCode method signature, and it does not state its correctness as a checkable invariant. We will not search for a new strategy here; we will only finish the delivery form and proof.

### Add Only the Platform Wrapper, Without Changing the Loop

The loop below is identical to Step 4. The only additions are `class Solution`, the `wordBreak` method, and conversion of the list to a set:

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        word_set = set(wordDict)
        reachable = [False] * (len(s) + 1)
        reachable[0] = True
        max_word_len = max(len(word) for word in word_set)

        for start in range(len(s)):
            if not reachable[start]:
                continue

            end_limit = min(len(s), start + max_word_len)
            for end in range(start + 1, end_limit + 1):
                piece = s[start:end]
                if piece in word_set:
                    reachable[end] = True

        return reachable[len(s)]
```

### Correctness: What Invariant Does `reachable` Maintain?

At every boundary `i`, `reachable[i]` is true if and only if the prefix `s[:i]` can be formed completely from dictionary words.

- Initially, `reachable[0] = True` because the empty prefix needs no words.
- If `reachable[start]` is true and `s[start:end]` is in the dictionary, appending that valid word forms a valid `s[:end]`, so setting `reachable[end] = True` is sound.
- Conversely, the last word in any valid split corresponds to some interval `start:end`. Its length is no greater than `max_word_len`, so the loop will examine it. That interval can be appended only when the preceding prefix `s[:start]` is already reachable.

Therefore, after the loops finish, `reachable[len(s)]` is true if and only if the whole string can be split completely.

### Complexity

Let `n = len(s)`, let `L` be the length of the longest dictionary word, and let `C` be the total number of characters across all dictionary words.

- The algorithm checks at most `O(nL)` candidate intervals.
- This Python implementation creates a slice and computes a string hash for each candidate. Including those costs, the time complexity is `O(nL^2 + C)`. If intervals are treated as constant-time string views, the number of transitions is `O(nL)`.
- `reachable` uses `O(n)` extra space, and `word_set` uses `O(C)` storage.

### Regression Check

```python
solver = Solution()
assert solver.wordBreak("leetcode", ["leet", "code"]) is True
assert solver.wordBreak("applepenapple", ["apple", "pen"]) is True
assert solver.wordBreak(
    "catsandog",
    ["cats", "dog", "sand", "and", "cat"],
) is False
assert solver.wordBreak("abcdef", ["abc", "def"]) is True
assert solver.wordBreak("a", ["b"]) is False
assert solver.wordBreak("", ["a"]) is True
print("Solution.wordBreak regression tests passed")
```

### Freeze the Final Checkpoint

This version can now:

- use the LeetCode 139 `Solution.wordBreak` interface directly
- reuse the prefix-reachability state and longest-word bound verified in the earlier steps
- explain every state update with an invariant and pass regression assertions for splittable, repeated-word, unsplittable, and boundary cases

The tutorial's code-growth chain is now complete: the final wrapper adds no algorithmic logic that did not appear earlier.

## Summary

- Represent "this prefix can be split completely" as the boundary state `reachable[i]`.
- The empty prefix, `reachable[0]`, is the starting point of every valid split.
- Appending a dictionary word to a reachable boundary marks that word's ending boundary as reachable.
- Scanning every reachable boundary is what covers multiword splits.
- The longest word length only narrows the candidate range; it does not change the state transition.
