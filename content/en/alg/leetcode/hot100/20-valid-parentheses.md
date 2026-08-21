---
title: "LeetCode 20: Valid Parentheses, Why Equal Counts Are Not Enough"
date: 2026-08-21
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "stack", "string", "parentheses matching", "LeetCode 20"]
description: "Start from the counterexample ([)], separate equal bracket counts from correct types and closing order, and derive an O(n) stack solution."
keywords: ["LeetCode 20", "Valid Parentheses", "stack", "parentheses matching", "Python", "Hot100"]
---

## Problem Requirement

You are given a string `s` containing only these six characters:

```text
( ) { } [ ]
```

Determine whether the string is valid. A valid string must satisfy all three conditions:

1. Every opening bracket is closed by the same type of closing bracket.
2. Opening brackets are closed in the correct order.
3. Every closing bracket has a corresponding opening bracket of the same type.

Return `True` when all conditions hold; otherwise, return `False`.

### Examples

| Input | Output |
| --- | --- |
| `"()"` | `True` |
| `"()[]{}"` | `True` |
| `"(]"` | `False` |
| `"([])"` | `True` |
| `"([)]"` | `False` |

### Constraints

- `1 <= s.length <= 10^4`
- `s` contains only characters from `()[]{}`

LeetCode provides this method signature:

```python
class Solution:
    def isValid(self, s: str) -> bool:
        pass
```

## Step 1: Why Equal Counts Can Still Be Invalid

Compare these two strings:

```text
()[]{}
([)]
```

Both have one `(` and one `)`, plus one `[` and one `]`. A check that only counts each type of opening and closing bracket accepts both strings.

The current baseline is:

```text
Count each opening and closing bracket. If all counts match, the string is valid.
```

This baseline gives the wrong answer for `([)]`. The problem is not the number of brackets but their closing order: a closing bracket must close the most recently seen opening bracket that has not yet been closed.

Apply that rule to `([)]`:

| Index | Character | Unclosed opening brackets | Decision |
| ---: | --- | --- | --- |
| 0 | `(` | `(` | Wait for the matching `)` |
| 1 | `[` | `([` | `[` is the most recent unclosed bracket |
| 2 | `)` | `([` | `)` cannot close the recent `[`, so the string is invalid |

The result is already known at index `2`; the final `]` cannot repair it. Although every bracket count eventually matches, `(` and `)` cross an unclosed `[`, producing an invalid order.

Use the same rule on several representative inputs:

| Input | Result | First decisive reason |
| --- | --- | --- |
| `()[]{}` | Valid | Every closing bracket closes the most recent matching opening bracket |
| `(]` | Invalid | `]` has a different type from the most recent `(` |
| `([)]` | Invalid | When `)` arrives, `[` is the most recent unclosed bracket |
| `([])` | Valid | `]` closes `[` first, then `)` closes `(` |

This checkpoint can now:

- distinguish equal bracket counts from correct closing order
- explain examples by bracket type and the most recent unclosed position

It still lacks:

- an algorithm that executes this decision for any input string
- runnable checks that compare the implementation with the manual reasoning

## Step 2: Repeatedly Remove Complete Pairs First

The current baseline can judge closing order by hand, but it does not yet define an executable process.

Consider a valid nested string:

```text
{[()]}
```

The innermost `()` is already an adjacent complete pair, so remove it first:

```text
{[()]}
  -> remove ()
{[]}
  -> remove []
{}
  -> remove {}
empty string
```

Every non-empty valid parentheses string contains at least one adjacent complete pair, regardless of its nesting depth. Removing that pair preserves the relative order of the remaining characters, so the same operation can continue.

After the manual rule from Step 1, add a runnable baseline: on each round, remove all three adjacent pair types. Stop when a round no longer shortens the string. The input is valid exactly when nothing remains.

```python
def is_valid_by_elimination(s: str) -> bool:
    while True:
        previous = s
        s = s.replace("()", "").replace("[]", "").replace("{}", "")

        if len(s) == len(previous):
            return s == ""


assert is_valid_by_elimination("()") is True
assert is_valid_by_elimination("()[]{}") is True
assert is_valid_by_elimination("(]") is False
assert is_valid_by_elimination("([])") is True
assert is_valid_by_elimination("([)]") is False
assert is_valid_by_elimination("{[()]}") is True
assert is_valid_by_elimination("(") is False
assert is_valid_by_elimination("]") is False
```

This process must terminate:

- If a round removes a pair, the string becomes at least `2` characters shorter.
- If its length does not change, no adjacent complete pair remains and the function returns immediately.

It also rejects an invalid order correctly. For example, `([)]` contains none of `()`, `[]`, or `{}`, so the first round makes no progress and returns `False`.

The baseline still processes characters repeatedly. A string nested to depth `n / 2` may expose only one new pair per round and require `O(n)` rounds. Python string replacement also scans and copies the current string, so the worst-case time complexity is `O(n^2)` and the extra string space is `O(n)`.

This checkpoint can now:

- execute a complete decision for any input that satisfies the character constraint
- express correct nesting through the order in which adjacent complete pairs disappear
- pass the official examples, deep nesting, and one-sided missing-bracket checks

It still lacks:

- a way to avoid scanning and copying already inspected characters repeatedly
- matching in one left-to-right pass

## Step 3: Keep Only the Opening Brackets That Are Still Unclosed

The elimination baseline is correct, but the same character may be scanned and copied in many replacement rounds.

Return to the rule from Step 1: when a closing bracket arrives, we only need the most recent opening bracket that has not yet been closed. Earlier opening brackets must wait until that one closes. This is exactly last-in, first-out order.

Use a stack named `stack` to store opening brackets that have been read but not closed:

- Push an opening bracket when it appears.
- A closing bracket must match the opening bracket at the top; pop after a match.
- If a closing bracket arrives while the stack is empty, or its type differs, return `False` immediately.
- After all characters are read, return `True` only when the stack is empty.

Trace `([)]`:

| Character | Stack before | Decision or operation | Stack after |
| --- | --- | --- | --- |
| `(` | `[]` | Push opening bracket | `['(']` |
| `[` | `['(']` | Push opening bracket | `['(', '[']` |
| `)` | `['(', '[']` | `)` needs `(` but the top is `[`, return `False` | Stop |

Now trace the valid nesting `([])`:

| Character | Stack before | Decision or operation | Stack after |
| --- | --- | --- | --- |
| `(` | `[]` | Push opening bracket | `['(']` |
| `[` | `['(']` | Push opening bracket | `['(', '[']` |
| `]` | `['(', '[']` | Match `[`, then pop | `['(']` |
| `)` | `['(']` | Match `(`, then pop | `[]` |

Replace repeated elimination with one scan:

```python
import random


class Solution:
    def isValid(self, s: str) -> bool:
        closing_to_opening = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        stack = []

        for char in s:
            if char in "([{":
                stack.append(char)
                continue

            if not stack or stack[-1] != closing_to_opening[char]:
                return False
            stack.pop()

        return not stack


solution = Solution()

# Official examples.
assert solution.isValid("()") is True
assert solution.isValid("()[]{}") is True
assert solution.isValid("(]") is False
assert solution.isValid("([])") is True
assert solution.isValid("([)]") is False

# Deep nesting, a missing opening bracket, and an opening bracket left over.
assert solution.isValid("{[()]}") is True
assert solution.isValid("(") is False
assert solution.isValid("]") is False
assert solution.isValid("(()") is False

# Compare 2,000 fixed-seed short strings with the correct Step 2 baseline.
rng = random.Random(20)
brackets = "()[]{}"

for _ in range(2_000):
    length = rng.randint(1, 12)
    candidate = "".join(rng.choice(brackets) for _ in range(length))
    expected = is_valid_by_elimination(candidate)
    actual = solution.isValid(candidate)
    assert actual is expected
```

### Why Checking Only the Stack Top Is Enough

During the scan, the stack always contains the opening brackets that have been read but not closed, in their original order.

When the current character is a closing bracket, correct order requires it to close the most recent unclosed opening bracket, which is the top of the stack:

- If the types match, popping the top leaves exactly the unclosed outer brackets.
- If the types differ, the closing bracket crosses the top opening bracket and no later character can repair the order.
- If the stack is empty, the closing bracket has no corresponding opening bracket.

A non-empty stack after the scan means that some opening brackets were never closed. Therefore, `return not stack` covers the final invalid case.

### Complexity

Each character is read once. Every opening bracket is pushed at most once and popped at most once, so the time complexity is `O(n)`. If every character is an opening bracket, the stack stores all `n` characters, giving `O(n)` extra space.

The final version now:

- checks bracket type and closing order in one left-to-right pass
- fails immediately on an unmatched closing bracket or type conflict
- rejects opening brackets left over after the scan
- runs in `O(n)` time and passes fixed examples plus randomized differential tests
