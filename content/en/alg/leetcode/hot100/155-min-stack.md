---
title: "LeetCode 155: Min Stack, Keeping the Minimum in Sync With Stack State"
date: 2026-08-21
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "stack", "design", "auxiliary stack", "LeetCode 155"]
description: "Start from restoring the previous minimum after pop, build a correct scan baseline, and derive an O(1) Min Stack with synchronized prefix minima."
keywords: ["LeetCode 155", "Min Stack", "auxiliary stack", "prefix minimum", "Python", "Hot100"]
---

## Problem Requirement

Design a stack named `MinStack` that supports these operations:

- `MinStack()`: initialize the stack.
- `push(value)`: push `value` onto the stack.
- `pop()`: remove the top element.
- `top()`: return the top element.
- `getMin()`: return the minimum element in the stack.

Every operation must run in `O(1)` time.

`pop`, `top`, and `getMin` are called only when the stack is non-empty, so no additional empty-stack return value is needed.

### Example

```text
Operations:
["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"]

Arguments:
[[], [-2], [0], [-3], [], [], [], []]

Output:
[null, null, null, null, -3, null, 0, -2]
```

The sequence corresponds to:

```python
min_stack = MinStack()
min_stack.push(-2)
min_stack.push(0)
min_stack.push(-3)
min_stack.getMin()  # -3
min_stack.pop()
min_stack.top()     # 0
min_stack.getMin()  # -2
```

### Constraints

- `-2^31 <= value <= 2^31 - 1`
- At most `3 * 10^4` calls are made to `push`, `pop`, `top`, and `getMin`
- The stack is non-empty whenever `pop`, `top`, or `getMin` is called

## Step 1: After Popping the Minimum, Where Does the Previous Minimum Come From?

A regular stack already stores values and implements `push`, `pop`, and `top` in last-in, first-out order. The current baseline is:

```text
Store only stack elements. The top is the most recently pushed element that has not been popped.
```

This baseline answers `top()` directly but not `getMin()`. The minimum may be at the bottom or in the middle rather than at the top.

The more important problem appears after `pop()`. Trace the official example:

| Operation | Stack afterward | Current minimum | Return value |
| --- | --- | ---: | ---: |
| Initialize | `[]` | None | `null` |
| `push(-2)` | `[-2]` | `-2` | `null` |
| `push(0)` | `[-2, 0]` | `-2` | `null` |
| `push(-3)` | `[-2, 0, -3]` | `-3` | `null` |
| `getMin()` | `[-2, 0, -3]` | `-3` | `-3` |
| `pop()` | `[-2, 0]` | `-2` | `null` |
| `top()` | `[-2, 0]` | `-2` | `0` |
| `getMin()` | `[-2, 0]` | `-2` | `-2` |

After `-3` is popped, the current minimum must immediately return to `-2`. A single variable holding the smallest value seen so far remains stuck at `-3`, which is no longer in the stack, and cannot tell us what to restore.

Duplicate minima create another pressure:

| Operation | Stack afterward | Current minimum |
| --- | --- | ---: |
| `push(2)` | `[2]` | `2` |
| `push(1)` | `[2, 1]` | `1` |
| `push(1)` | `[2, 1, 1]` | `1` |
| `pop()` | `[2, 1]` | `1` |

After one `1` is popped, another remains, so the minimum must not return to `2`.

Together these traces show that every stack state has its own current minimum. `push` enters a new state and `pop` returns to the previous state; the minimum must change in sync with that state transition.

This checkpoint can now:

- explain why `top()` and `getMin()` return different kinds of information
- explain why popping a unique minimum must restore a historical minimum
- explain why a duplicate minimum survives one pop

It still lacks:

- a runnable class implementing all five interfaces
- a concrete way to calculate the correct minimum after the stack changes

## Step 2: Scan the Whole Stack When the Minimum Is Requested

The current baseline knows that each stack state has a minimum but does not yet implement the class. First make the behavior correct: store every value in one list, and use the end of the list for `push`, `pop`, and `top`.

When `getMin()` is called, scan the current list and return its minimum. Even if the unique minimum was just popped, this recomputes the correct answer from the remaining elements.

This does not contradict Step 1:

- A single current-minimum variable cannot restore history in `O(1)` after that value is popped.
- Scanning every remaining element can recompute the minimum, but takes `O(n)` time.

Write this baseline as a complete class:

```python
class MinStackScan:
    def __init__(self):
        self.values = []

    def push(self, value: int) -> None:
        self.values.append(value)

    def pop(self) -> None:
        self.values.pop()

    def top(self) -> int:
        return self.values[-1]

    def getMin(self) -> int:
        return min(self.values)


# Official operation sequence.
stack = MinStackScan()
stack.push(-2)
stack.push(0)
stack.push(-3)
assert stack.getMin() == -3
stack.pop()
assert stack.top() == 0
assert stack.getMin() == -2

# A duplicate minimum remains valid after one copy is popped.
duplicates = MinStackScan()
duplicates.push(2)
duplicates.push(1)
duplicates.push(1)
assert duplicates.getMin() == 1
duplicates.pop()
assert duplicates.getMin() == 1

# The minimum is at the bottom of an increasing stack and at the top of a decreasing stack.
increasing = MinStackScan()
for value in [1, 2, 3]:
    increasing.push(value)
assert increasing.top() == 3
assert increasing.getMin() == 1

decreasing = MinStackScan()
for value in [3, 2, 1]:
    decreasing.push(value)
assert decreasing.top() == 1
assert decreasing.getMin() == 1
```

All five operations now behave correctly, but they do not all meet the required complexity:

| Operation | Time complexity | Reason |
| --- | --- | --- |
| `MinStackScan()` | `O(1)` | Create one empty list |
| `push(value)` | Amortized `O(1)` | Append at the list end |
| `pop()` | `O(1)` | Remove from the list end |
| `top()` | `O(1)` | Read the list end |
| `getMin()` | `O(n)` | Scan every current element |

Repeated `getMin()` calls compare the same elements again and again. The problem allows up to `3 * 10^4` operations and explicitly requires every method to be `O(1)`, so this cannot be the final answer.

This checkpoint can now:

- implement all five required interfaces
- return the right result after popping a unique minimum or one duplicate minimum
- serve as a correct baseline for the optimization

It still lacks:

- a `getMin()` that does not scan the whole stack
- state saved during `push` and `pop` that can restore a previous minimum

## Step 3: Synchronize the Minimum With Every Stack Depth

The current baseline is correct, but every `getMin()` scans `self.values`. Those comparisons can instead happen during `push`, because after a new value is pushed there are only two candidates:

```text
minimum of the new state = min(minimum of the previous state, new value)
```

Add a list named `min_values` with the same length as the value stack. `min_values[i]` stores the minimum of `values[0:i+1]`, which is the current minimum when the value stack has depth `i + 1`.

The two lists change together in the official sequence:

| Operation | `values` | `min_values` |
| --- | --- | --- |
| `push(-2)` | `[-2]` | `[-2]` |
| `push(0)` | `[-2, 0]` | `[-2, -2]` |
| `push(-3)` | `[-2, 0, -3]` | `[-2, -2, -3]` |
| `pop()` | `[-2, 0]` | `[-2, -2]` |

The ends of both lists always describe the same stack state. Therefore:

- `push` appends both the value and the new current minimum.
- `pop` removes the end of both lists.
- `top` reads `values[-1]`.
- `getMin` reads `min_values[-1]` directly.

Duplicate minima are also stored once per depth. After pushing `2, 1, 1`, `min_values` is `[2, 1, 1]`. One pop removes only the final `1`; the earlier `1` remains at the top.

Replace the linear scan with synchronized state:

```python
import random


class MinStack:
    def __init__(self):
        self.values = []
        self.min_values = []

    def push(self, value: int) -> None:
        self.values.append(value)

        if self.min_values:
            value = min(value, self.min_values[-1])
        self.min_values.append(value)

    def pop(self) -> None:
        self.values.pop()
        self.min_values.pop()

    def top(self) -> int:
        return self.values[-1]

    def getMin(self) -> int:
        return self.min_values[-1]


# Official operation sequence.
stack = MinStack()
stack.push(-2)
stack.push(0)
stack.push(-3)
assert stack.getMin() == -3
stack.pop()
assert stack.top() == 0
assert stack.getMin() == -2

# Duplicate minima must be retained at each depth.
duplicates = MinStack()
duplicates.push(2)
duplicates.push(1)
duplicates.push(1)
assert duplicates.min_values == [2, 1, 1]
duplicates.pop()
assert duplicates.getMin() == 1

# Compare 5,000 fixed-seed valid operations with the Step 2 baseline.
rng = random.Random(155)
expected = MinStackScan()
actual = MinStack()

for _ in range(5_000):
    if not expected.values or rng.random() < 0.55:
        value = rng.randint(-100, 100)
        expected.push(value)
        actual.push(value)
    else:
        operation = rng.choice(["pop", "top", "getMin"])

        if operation == "pop":
            expected.pop()
            actual.pop()
        elif operation == "top":
            assert actual.top() == expected.top()
        else:
            assert actual.getMin() == expected.getMin()

    assert len(actual.values) == len(actual.min_values)
    if actual.values:
        assert actual.min_values[-1] == min(actual.values)
        assert actual.top() == expected.top()
        assert actual.getMin() == expected.getMin()
```

### Why the Synchronized List Restores Historical Minima

The following invariant holds before and after every operation:

```text
len(values) == len(min_values)

When values is non-empty:
min_values[-1] == min(values)
```

When `values` is empty, `min_values` must also be empty. The problem guarantees that `top()` and `getMin()` are not called then, so neither method needs to read a missing final element.

On the first `push`, the only value is naturally also the minimum. For each later `push(value)`, the new prefix differs from the previous one by only `value`, so comparing the previous minimum with the new value gives the new minimum.

`pop()` removes both list ends and returns both structures to the previous depth. The minimum for that depth was already saved at the new `min_values[-1]`, so `values` does not need to be scanned again.

### Complexity

With Python lists, `push` and list-end `pop` are amortized `O(1)`, while `top` and `getMin` are worst-case `O(1)`. Under the stack-operation model used by the problem, all four interfaces meet the constant-time requirement. Both lists store one item per stack depth, so the space complexity is `O(n)`.

The final version now:

- implements every required interface correctly
- returns the current minimum in `O(1)` time
- restores the correct historical state after popping a unique or duplicate minimum
- passes the fixed cases and 5,000 randomized valid operations
