---
title: "Hot100: Linked List Cycle Floyd Fast/Slow Pointer ACERS Guide"
date: 2026-02-10T08:55:58+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "two pointers", "fast slow pointers", "Floyd", "LeetCode 141"]
description: "Detect whether a singly linked list has a cycle using Floyd fast/slow pointers in O(n) time and O(1) extra space, with proof intuition, pitfalls, engineering mapping, and runnable multi-language implementations."
keywords: ["Linked List Cycle", "Floyd cycle detection", "fast slow pointers", "O(1) space", "LeetCode 141", "Hot100"]
---

> **Subtitle / Summary**  
> Detecting a cycle in a linked list is a pointer chasing problem, not a value comparison problem. This ACERS guide explains why Floyd’s fast/slow pointers must meet if a cycle exists, how to avoid null-pointer bugs, and how the same pattern maps to engineering checks.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `linked list`, `fast slow pointers`, `Floyd`  
- **SEO keywords**: Linked List Cycle, Floyd, fast slow pointers, LeetCode 141, Hot100  
- **Meta description**: O(n)/O(1) cycle detection in singly linked lists using Floyd fast/slow pointers, with alternatives, common mistakes, and runnable multi-language code.

---

## Target Readers

- Hot100 learners and interview candidates
- Developers building reusable linked-list two-pointer templates
- Engineers who need to detect loops in chain-like structures

## Background / Motivation

Cycle bugs are common in pointer-linked structures:

- traversal never ends (infinite loop)
- cleanup/free logic hangs
- system looks "randomly stuck" while root cause is structural

So we need a detection method that is:

- online (single pass style)
- memory-light (no large side structure)
- robust under large lists

Floyd fast/slow pointer detection is the standard solution for this profile.

## Core Concepts

- **Cycle**: from some node, following `next` can eventually return to itself
- **`pos` in problem statement**: only used by test data construction, not a function parameter
- **Fast/slow pointers**:
  - `slow` moves 1 step
  - `fast` moves 2 steps
- **Node identity vs node value**: compare pointer/reference identity, not `val`

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given head node `head` of a singly linked list, determine whether there is a cycle in the list.
Return `true` if a cycle exists, else `false`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list (can be null) |
| return | bool | whether cycle exists |

### Example 1

```text
head: 3 -> 2 -> 0 -> -4
               ^     |
               |_____|
output: true
```

### Example 2

```text
head: 1 -> 2 -> null
output: false
```

---

## Thought Process: From Hash Set to Floyd

### Naive approach: record visited nodes

Traverse nodes and store each node reference in a set:

- seen again => cycle
- reaches null => no cycle

Pros: easy to reason about.
Cons: `O(n)` extra memory.

### Key observation

If a cycle exists, once both pointers enter the cycle:

- fast gains 1 node per step over slow
- relative gap changes modulo cycle length
- gap must become 0 eventually

So they must meet inside the cycle.

### Final choice

Floyd cycle detection:

- time `O(n)`
- extra space `O(1)`

---

## C - Concepts (Core Ideas)

### Method Category

- Two pointers
- Floyd cycle detection (tortoise-hare)
- Online structural integrity check

### Why meeting is guaranteed (intuition)

Let cycle length be `L`.
After both pointers are in cycle, each round:

- slow +1
- fast +2

Relative movement is +1 modulo `L`.
So relative distance cycles through all residues and eventually becomes 0, meaning they meet.

### Safe loop condition

Before `fast = fast.next.next`, we must ensure:

- `fast != null`
- `fast.next != null`

Otherwise null dereference happens.

---

## Practical Guide / Steps

1. Initialize `slow = head`, `fast = head`
2. While `fast != null && fast.next != null`:
   - `slow = slow.next`
   - `fast = fast.next.next`
   - if `slow == fast`, return `true`
3. Return `false`

Runnable Python example (`linked_list_cycle.py`):

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def build(values: List[int], pos: int) -> Optional[ListNode]:
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


if __name__ == "__main__":
    print(has_cycle(build([3, 2, 0, -4], 1)))  # True
    print(has_cycle(build([1, 2], -1)))        # False
```

---

## Explanation / Why This Works

The algorithm has two outcomes:

1. **No cycle**: fast pointer reaches null first => return `false`
2. **Has cycle**: fast and slow eventually meet in cycle => return `true`

No extra memory is needed because we do not store history.
We rely on relative speed and finite cycle length.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: free-list integrity check in memory pools (C)

**Background**: low-level allocators often keep free blocks as a singly linked list.  
**Why it fits**: if free list becomes cyclic, allocation/release may hang; Floyd check is O(1) memory.

```c
int hasCycle(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1;
    }
    return 0;
}
```

### Scenario 2: backend workflow next-pointer validation (Go)

**Background**: lightweight workflow nodes may be chained by `Next`.  
**Why it fits**: config bugs can create loops and stall execution.

```go
func hasCycle(head *Node) bool {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true
        }
    }
    return false
}
```

### Scenario 3: linked object debug check in browser tools (JavaScript)

**Background**: front-end tooling may use chain objects with `next` links.  
**Why it fits**: quickly detect accidental circular links before traversing.

```javascript
function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}
```

---

## R - Reflection

### Complexity

- Time: `O(n)`
- Space: `O(1)` (Floyd), versus `O(n)` for hash-set approach

### Alternatives and Tradeoffs

| Method | Time | Space | Tradeoff |
| --- | --- | --- | --- |
| Hash set visited nodes | O(n) | O(n) | easy but memory-heavy |
| Marker on node | O(n) | O(1) | mutates structure; usually forbidden |
| Floyd fast/slow | O(n) | O(1) | best practical baseline |

### Common Mistakes

1. Compare values instead of node identity
2. Forget null checks before double-step fast move
3. Check `slow == fast` before first movement (trivial true at start)
4. Assume `pos` is passed to function

### Why this is engineering-optimal in most cases

It balances runtime, memory, and non-intrusiveness:

- linear scan
- constant memory
- no mutation

---

## FAQ and Notes

1. **Can this also find cycle entry?**  
   Yes, that is LeetCode 142 (extra phase after first meeting).

2. **What if list is very large?**  
   Still linear and memory-safe versus visited set growth.

3. **When is hash set preferable?**  
   If you also need to record traversal path or list all repeated nodes.

---

## Best Practices

- Always use `while fast && fast.next`
- Compare identity (`is`, `===`, pointer equality), not value
- Keep this template as your default cycle check

---

## S - Summary

- Cycle detection is about pointer identity and relative speed, not values.
- Floyd detects cycles in `O(n)` time with `O(1)` extra space.
- Null-check ordering is critical for safety.
- This template maps directly to chain integrity checks in real systems.

### Recommended Further Reading

- LeetCode 141. Linked List Cycle
- LeetCode 142. Linked List Cycle II
- Floyd’s cycle detection variants and proofs

---

## Conclusion

LeetCode 141 is a foundational two-pointer template.
Once internalized, it becomes a reusable structural safety check across many chain-based systems.

---

## References

- https://leetcode.com/problems/linked-list-cycle/
- https://leetcode.com/problems/linked-list-cycle-ii/
- https://en.wikipedia.org/wiki/Cycle_detection
- https://en.cppreference.com/w/cpp/container/forward_list

---

## Meta Info

- **Reading time**: 10-12 min
- **Tags**: Hot100, linked list, Floyd, fast slow pointers, LeetCode 141
- **SEO keywords**: Linked List Cycle, Floyd, fast slow pointers, O(1), LeetCode 141
- **Meta description**: Floyd fast/slow pointers detect linked-list cycle in O(n)/O(1), with proof intuition and multi-language implementations.

---

## Call To Action (CTA)

After this article, solve `142` immediately:

1. detect cycle (this problem)
2. find cycle entry (next step)

Treat them as one template family.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def hasCycle(head: Optional[ListNode]) -> bool:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

```c
struct ListNode {
    int val;
    struct ListNode* next;
};

int hasCycle(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1;
    }
    return 0;
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    val: i32,
    next: Option<Rc<RefCell<Node>>>,
}

fn next(node: &Option<Rc<RefCell<Node>>>) -> Option<Rc<RefCell<Node>>> {
    node.as_ref().and_then(|rc| rc.borrow().next.clone())
}

fn has_cycle(head: Option<Rc<RefCell<Node>>>) -> bool {
    let mut slow = head.clone();
    let mut fast = head;

    loop {
        slow = next(&slow);
        fast = next(&fast);
        if fast.is_none() || slow.is_none() {
            return false;
        }
        fast = next(&fast);
        if fast.is_none() {
            return false;
        }
        if let (Some(ref s), Some(ref f)) = (&slow, &fast) {
            if Rc::ptr_eq(s, f) {
                return true;
            }
        } else {
            return false;
        }
    }
}
```

```javascript
function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}
```
