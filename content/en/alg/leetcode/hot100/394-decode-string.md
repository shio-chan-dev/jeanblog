---
title: "LeetCode 394: Decode String by Saving and Restoring Nested Context"
date: 2026-08-21
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "stack", "string", "recursion", "LeetCode 394"]
description: "Start from the context lost by nested encoding, build a correct recursive parser, and make its saved frames explicit with a stack."
keywords: ["LeetCode 394", "Decode String", "context stack", "nested string", "recursion", "Python", "Hot100"]
---

## Problem Requirement

Given an encoded string `s`, return its decoded string.

The encoding rule is:

```text
k[encoded_string]
```

The `encoded_string` inside the brackets is repeated exactly `k` times, where `k` is a positive integer. Encodings may be nested or adjacent to ordinary lowercase letters.

The problem guarantees that:

- The input is always valid, with matching brackets and no extra spaces.
- Original text contains no digits; digits only represent repeat counts.
- Inputs such as `3a` or `2[4]` do not occur.
- The decoded string length does not exceed `10^5`.

### Examples

| Input | Output |
| --- | --- |
| `"3[a]2[bc]"` | `"aaabcbc"` |
| `"3[a2[c]]"` | `"accaccacc"` |
| `"2[abc]3[cd]ef"` | `"abcabccdcdcdef"` |

### Constraints

- `1 <= s.length <= 30`
- `s` contains only lowercase English letters, digits, and `[]`
- Every repeat count is in `[1, 300]`

LeetCode provides this method signature:

```python
class Solution:
    def decodeString(self, s: str) -> str:
        pass
```

## Step 1: Where Does the Outer Information Go When We Enter a Nested Layer?

Start with an input that has no nesting:

```text
3[a]
```

After reading `3`, we know that the next segment must be repeated three times. After reading `a` inside the brackets, the result is:

```text
a * 3 = aaa
```

The current baseline is:

```text
Read one repeat count, collect the text in the following brackets, and repeat it when ] appears.
```

This baseline handles one layer such as `3[a]`, but it breaks on `3[a2[c]]`. The outer layer has already read count `3` and ordinary letter `a` when the nested encoding `2[c]` begins. If the current count is overwritten with `2` and the current text with `c`, the outer `3` and `a` are lost.

Expand the input one layer at a time:

| Stage | Layer being processed | What this layer knows | Information that must be retained |
| --- | --- | --- | --- |
| Read outer `3[` | Outer | Repeat `3` times | None |
| Read outer `a` | Outer | Text prefix is `a` | None |
| Read inner `2[` | Inner | Prepare to repeat `2` times | Outer text `a`, outer count `3` |
| Read inner `c]` | Inner complete | `c * 2 = cc` | Restore outer text `a`, outer count `3` |
| Return to outer | Outer | `a + cc = acc` | Outer count is still `3` |
| Read outer `]` | Outer complete | `acc * 3` | None |

The result is:

```text
3[a2[c]]
-> 3[a + cc]
-> 3[acc]
-> accaccacc
```

Before entering an inner layer, we therefore cannot overwrite unfinished outer information. At least two pieces of context must be retained:

- The text already decoded in the outer layer.
- The number of times the completed outer layer must eventually repeat.

Adjacent but non-nested input such as `3[a]2[bc]` does not keep two layers at once. It first completes `aaa`, independently decodes `2[bc]` as `bcbc`, and concatenates them into `aaabcbc`. The need to save context appears only when a new encoding starts before the current encoding is complete.

This checkpoint can now:

- distinguish single-layer, adjacent, and nested encodings
- explain layer by layer why `3[a2[c]]` becomes `accaccacc`
- identify the outer text and repeat count that must survive entry into a nested layer

It still lacks:

- an executable rule for entering an inner layer, returning to its parent, and continuing the scan
- runtime checks for multi-digit counts, ordinary suffixes, and deeper nesting

## Step 2: Let Each Recursive Call Finish One Layer

The current baseline knows that outer context must survive entry into an inner layer, but it has not defined when the inner layer finishes or where the outer layer resumes.

Reduce the problem to one bracket layer: read from a given position until the `]` that belongs to this layer. This process must return two results:

```text
the decoded text of the current layer
the next position after its closing bracket
```

When an outer layer sees `[`, it asks the next layer to start immediately after that bracket. When the child reaches its own `]`, it returns. The caller repeats the child text by the current count and resumes at the returned position.

Trace the positions in `3[a2[c]]`:

| Position | Character | Current-layer operation | Next position or return value |
| ---: | --- | --- | --- |
| 0 | `3` | Top level accumulates count `3` | 1 |
| 1 | `[` | Enter outer encoding at position 2 | Wait for return |
| 2 | `a` | Append `a` to outer text | 3 |
| 3 | `2` | Outer layer accumulates inner count `2` | 4 |
| 4 | `[` | Enter inner encoding at position 5 | Wait for return |
| 5 | `c` | Append `c` to inner text | 6 |
| 6 | `]` | Inner layer returns `("c", 7)` | Outer gets `c * 2` |
| 7 | `]` | Outer layer returns `("acc", 8)` | Top level gets `acc * 3` |
| 8 | End of string | Top level returns `"accaccacc"` | Complete |

Turn this layer contract into a recursive baseline:

```python
class RecursiveDecoder:
    def decodeString(self, s: str) -> str:
        decoded, _ = self._decode_layer(s, 0)
        return decoded

    def _decode_layer(self, s: str, index: int) -> tuple[str, int]:
        parts = []
        repeat = 0

        while index < len(s):
            char = s[index]

            if char.isdigit():
                repeat = repeat * 10 + int(char)
                index += 1
            elif char == "[":
                nested, index = self._decode_layer(s, index + 1)
                parts.append(nested * repeat)
                repeat = 0
            elif char == "]":
                return "".join(parts), index + 1
            else:
                parts.append(char)
                index += 1

        return "".join(parts), index


decoder = RecursiveDecoder()

# Official examples.
assert decoder.decodeString("3[a]2[bc]") == "aaabcbc"
assert decoder.decodeString("3[a2[c]]") == "accaccacc"
assert decoder.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"

# A multi-digit count, ordinary text, and deeper nesting.
assert decoder.decodeString("12[a]") == "a" * 12
assert decoder.decodeString("abc3[cd]xyz") == "abccdcdcdxyz"
assert decoder.decodeString("2[a2[b2[c]]]") == "abccbccabccbcc"
```

The variable `repeat` belongs only to the current layer. Consecutive digits form one count through `repeat * 10 + int(char)`. When the current layer enters a child, that recursive call has its own `parts` and `repeat`; the caller's local state still exists when the child returns.

At `]`, the parser must return `index + 1`, because the caller should next read the character after the closing bracket rather than process the same `]` again. The top level has no closing bracket, so it returns the final result when `index == len(s)`.

### Correctness and Complexity

Each call consumes only its own ordinary characters, counts, and direct child layers. A child returns fully decoded text, so its parent only needs to repeat that text and append it. The input guarantees valid brackets, so every child returns at its matching `]`, and the top-level call eventually reaches the end of the string.

Let `n` be the encoded input length, `L` the final output length, and `d` the maximum nesting depth. Each input character causes one branch decision, but repetition and concatenation must construct actual output. More precisely, the time complexity is `O(n + W)`, where `W` is the total number of string characters constructed across all layers during the run. Its worst-case bound is `O(n + dL)`.

Space cannot be measured by cumulative construction cost `W`, because earlier temporary strings may already have been released. Recursive frames use `O(d)` space, while per-layer lists, references, and indices use `O(n + d)`. Let `P` be the number of decoded and temporary string characters that remain live at one time. Peak space is `O(n + d + P)`. Since each layer's live text is no longer than the final output, `P` is at most `O(dL)`, giving a safe `O(n + dL)` bound including the returned result.

This checkpoint can now:

- correctly decode single-layer, adjacent, and nested encodings
- accumulate multi-digit repeat counts
- return the position where the caller should continue after a child layer
- pass the official examples, ordinary text, and deeper nesting checks

It still lacks:

- a visible representation of the outer text and count saved at each nested entry
- explicit state in place of context stored automatically by recursive calls

## Step 3: Move the Context Saved by Recursion Into an Explicit Stack

The recursive baseline is correct, but the language call stack saves outer state for us. Each call frame retains the text decoded so far, the repeat count, and the continuation position until its child returns.

Now scan the string once from left to right. The loop itself owns the continuation position, so at `[` we only need to save:

```text
(outer text decoded so far, current repeat count)
```

An inner layer must finish before its parent can resume, which is last-in, first-out order. Store unfinished outer contexts in `contexts`:

- On a digit, accumulate `repeat_count` one decimal digit at a time.
- On `[`, push the current text and count, then reset the current-layer state.
- On `]`, pop the most recent outer context, repeat the current-layer text, and append it to the outer text.
- On a letter, append it to the current-layer text.

Python strings are immutable, so repeated character-by-character concatenation can copy the existing text. Instead, `current_parts` stores the text fragments of the current layer and joins them only when a layer is completed. It has the same meaning as recursive `parts`, but the loop now saves and restores it directly.

Trace the stack for `3[a2[c]]`:

| Character | `repeat_count` | `contexts` (bottom to top) | Current-layer text | Operation |
| --- | ---: | --- | --- | --- |
| `3` | `3` | `[]` | `""` | Accumulate outer count |
| `[` | `0` | `[("", 3)]` | `""` | Save outer context and reset |
| `a` | `0` | `[("", 3)]` | `"a"` | Append ordinary character |
| `2` | `2` | `[("", 3)]` | `"a"` | Accumulate inner count |
| `[` | `0` | `[("", 3), ("a", 2)]` | `""` | Save current outer context and reset |
| `c` | `0` | `[("", 3), ("a", 2)]` | `"c"` | Append ordinary character |
| `]` | `0` | `[("", 3)]` | `"acc"` | Restore `a`, then append `c * 2` |
| `]` | `0` | `[]` | `"accaccacc"` | Restore empty prefix, then append `acc * 3` |

Maintain three conditions throughout the scan:

1. `current_parts` stores the decoded text of the unfinished current layer in order.
2. `repeat_count` stores the integer formed by the most recent consecutive digits and resets immediately after `[`.
3. `contexts`, from bottom to top, stores unfinished contexts from outermost to innermost; its top is always the layer restored by the current `]`.

Implement the four character branches:

```python
class Solution:
    def decodeString(self, s: str) -> str:
        contexts: list[tuple[list[str], int]] = []
        current_parts: list[str] = []
        repeat_count = 0

        for char in s:
            if char.isdigit():
                repeat_count = repeat_count * 10 + int(char)
            elif char == "[":
                contexts.append((current_parts, repeat_count))
                current_parts = []
                repeat_count = 0
            elif char == "]":
                nested_text = "".join(current_parts)
                outer_parts, count = contexts.pop()
                outer_parts.append(nested_text * count)
                current_parts = outer_parts
            else:
                current_parts.append(char)

        return "".join(current_parts)
```

First check fixed boundaries:

```python
solution = Solution()

# Official examples.
assert solution.decodeString("3[a]2[bc]") == "aaabcbc"
assert solution.decodeString("3[a2[c]]") == "accaccacc"
assert solution.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"

# A multi-digit count, ordinary text, and deeper nesting.
assert solution.decodeString("12[a]") == "a" * 12
assert solution.decodeString("abc3[cd]xyz") == "abccdcdcdxyz"
assert solution.decodeString("2[a2[b2[c]]]") == "abccbccabccbcc"
```

Next, generate both a valid encoding and its expected result, then compare the explicit-stack solution with the recursive baseline from Step 2. Run this block after the previous two complete code blocks:

```python
import random


random_generator = random.Random(394)


def generate_valid_case(depth: int = 0) -> tuple[str, str]:
    encoded_parts = []
    decoded_parts = []

    for _ in range(random_generator.randint(1, 3)):
        if depth < 3 and random_generator.random() < 0.5:
            nested_encoded, nested_decoded = generate_valid_case(depth + 1)
            repeat = random_generator.randint(1, 12)
            encoded_parts.append(f"{repeat}[{nested_encoded}]")
            decoded_parts.append(nested_decoded * repeat)
        else:
            text = "".join(
                random_generator.choice("abc")
                for _ in range(random_generator.randint(1, 3))
            )
            encoded_parts.append(text)
            decoded_parts.append(text)

    return "".join(encoded_parts), "".join(decoded_parts)


baseline = RecursiveDecoder()
checked = 0

while checked < 2_000:
    encoded, expected = generate_valid_case()
    if len(encoded) > 30 or len(expected) > 100_000:
        continue

    assert baseline.decodeString(encoded) == expected
    assert solution.decodeString(encoded) == expected
    checked += 1
```

The random check does not ask two implementations to prove each other correct. The generator directly constructs `expected`, which provides an independent expected value. Both the recursive and explicit-stack implementations must match it; comparing the two also helps expose differences in layer entry or restoration.

### Correctness and Complexity

At `[`, the algorithm saves the complete unfinished outer state. At the matching `]`, last-in, first-out order restores the nearest outer layer. The current layer is already fully decoded, so `nested_text * count` is exactly this encoded segment's result. Appending it to the outer layer re-establishes all three conditions. Because the input brackets are valid, every saved context has been restored when the scan ends, and `current_parts` represents the entire decoded string.

Let `n` be the input length, `L` the final output length, and `d` the maximum nesting depth. The scan itself is `O(n)`, but joining and repeating must construct decoded text. If `W` is the total number of characters constructed across all layers during the run, time is `O(n + W)` and at worst `O(n + dL)`. The stack depth is `O(d)`, active fragment lists and references use `O(n)`, and live decoded text plus temporary copies use `O(L)`. Peak space including the returned result is therefore `O(n + d + L)`.

The final version now:

- saves and restores outer context at any nesting depth with an explicit stack
- handles adjacent encodings, multi-digit counts, ordinary prefixes and suffixes, and deep nesting
- passes 6 fixed assertions and 2,000 valid randomized encodings
- completes the basic stack loop across LeetCode 20, 155, and 394: match the nearest state, synchronize historical state, and preserve nested context
