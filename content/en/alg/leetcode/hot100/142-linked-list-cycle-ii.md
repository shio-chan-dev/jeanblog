---
title: "Hot100: Linked List Cycle II Floyd Detection + Entry Node Localization ACERS Guide"
date: 2026-02-10T09:31:37+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "fast slow pointers", "Floyd", "two pointers", "LeetCode 142"]
description: "Return the first node where a singly linked list enters a cycle without modifying the list: use Floyd fast/slow pointers to detect a meeting point, then reset one pointer to head and walk together to find cycle entry in O(n) time and O(1) extra space."
keywords: ["Linked List Cycle II", "cycle entry", "Floyd", "fast slow pointers", "O(1) space", "LeetCode 142", "Hot100"]
---

> **Subtitle / Summary**  
> This problem upgrades "cycle detection" into "cycle entry localization". The production-grade template is Floyd: detect a meeting inside the cycle first, then reset one pointer to head and move both one step at a time; their next meeting is exactly the cycle entry.

- **Reading time**: 12-16 min  
- **Tags**: `Hot100`, `linked list`, `fast/slow pointers`, `Floyd`  
- **SEO keywords**: Linked List Cycle II, cycle entry, Floyd, fast/slow pointers, O(1) space, LeetCode 142  
- **Meta description**: Floyd cycle detection + entry localization with proof intuition and runnable multi-language implementations in O(n) time and O(1) extra space.

---

## Target Readers

- Hot100 learners who want a robust cycle-detection template family (`141 + 142`)
- Engineers who need to locate the exact broken node in chain-like structures
- Interview prep readers who want the derivation behind "reset one pointer to head"

## Background / Motivation

A cycle in linked structures is a practical production issue, not just a textbook case:

- traversal never reaches `null` and loops forever
- cleanup/release logic can hang
- logs and metrics look "stuck" but root cause is structural corruption

Knowing whether a cycle exists is good, but operations usually need more:

- **where does the cycle start?** (entry node)

That is exactly what LeetCode 142 asks for.

## Core Concepts

| Concept | Meaning | Why it matters |
| --- | --- | --- |
| Cycle | Following `next` eventually revisits a node | causes infinite traversal |
| Entry node | First node where linear prefix enters the loop | required output |
| Floyd algorithm | `slow` moves 1, `fast` moves 2 | O(1) extra memory |
| Meeting point | First collision of `slow` and `fast` inside cycle | bridge to entry localization |
| Identity equality | compare node object/reference, not value | values can repeat |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given `head` of a singly linked list, return the node where the cycle begins.  
If there is no cycle, return `null`.

Notes:

- `pos` in problem statement is only used by the judge to build test data
- `pos` is **not** a function argument
- you must not modify the list

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list |
| return | ListNode / null | cycle entry node reference, or null |

### Example 1

```text
head: 3 -> 2 -> 0 -> -4
           ^         |
           |_________|

output: node(2)
```

### Example 2

```text
head: 1 -> 2 -> 3 -> null
output: null
```

---

## Thought Process: From Hashing to Floyd + Reset

### Naive approach: visited set

Walk through list and store each node reference in a hash set:

- if current node already exists in set, that node is the entry
- if we reach `null`, no cycle

Pros: easy to reason.
Cons: `O(n)` extra space.

### Goal upgrade

Need `O(1)` extra space and still return entry node.

### Key observation

Floyd gives two phases:

1. **Detection**: if `slow` and `fast` meet, there is a cycle
2. **Localization**: after meeting, move one pointer to `head`; then move both one step per round, and they meet at cycle entry

The second phase is where most people memorize but do not fully understand.

---

## C - Concepts (Core Ideas)

### Method Category

- Two pointers (`fast/slow`)
- Floyd cycle detection
- Distance alignment after pointer reset

### Why reset-to-head works

Let:

- `a`: distance from head to cycle entry
- `b`: distance from cycle entry to first meeting point (inside cycle)
- `c`: cycle length

At first meeting:

- slow traveled `a + b`
- fast traveled `2(a + b)`

Difference is:

```text
2(a + b) - (a + b) = a + b
```

This difference must be an integer number of full cycles:

```text
a + b = k * c
```

So:

```text
a = k*c - b = (k-1)*c + (c-b)
```

Interpretation:

- from head, walking `a` steps reaches cycle entry
- from meeting point, walking `a` steps equals walking `(k-1)` full cycles plus `(c-b)` steps, also reaches entry

Therefore:

- pointer `p` from `head`
- pointer `q` from meeting point

move both one step each round, they meet at entry.

---

## Practical Guide / Steps

1. Initialize `slow=head`, `fast=head`
2. Detection loop:
   - if `fast == null` or `fast.next == null`, return `null`
   - `slow = slow.next`, `fast = fast.next.next`
   - if `slow == fast`, go to phase 2
3. Localization:
   - set `p = head`, `q = slow`
   - while `p != q`: `p = p.next`, `q = q.next`
4. Return `p` (entry node)

Runnable Python example (`cycle_entry.py`):

```python
from __future__ import annotations


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: ListNode | None = None


def detect_cycle(head: ListNode | None) -> ListNode | None:
    slow = head
    fast = head

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            q = slow
            while p is not q:
                p = p.next  # type: ignore[assignment]
                q = q.next  # type: ignore[assignment]
            return p
    return None


if __name__ == "__main__":
    # 3 -> 2 -> 0 -> -4 -> back to 2
    n3 = ListNode(3)
    n2 = ListNode(2)
    n0 = ListNode(0)
    n4 = ListNode(-4)
    n3.next = n2
    n2.next = n0
    n0.next = n4
    n4.next = n2

    ans = detect_cycle(n3)
    print(ans.val if ans else None)  # 2
```

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: workflow next-pointer misconfiguration localization (Python)

**Background**: lightweight workflow engines may store step order as next pointers.  
**Why it fits**: you need exact faulty step (entry), not just yes/no cycle.

```python
class Task:
    def __init__(self, name):
        self.name = name
        self.next = None


def entry(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p, q = head, slow
            while p is not q:
                p = p.next
                q = q.next
            return p
    return None
```

### Scenario 2: free-list / object-pool integrity check (C)

**Background**: allocator free-lists are chain structures where accidental loops can freeze allocation paths.  
**Why it fits**: O(1) extra memory, no structural mutation.

```c
struct Node { struct Node* next; };

struct Node* detectCycle(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            struct Node* p = head;
            struct Node* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return 0;
}
```

### Scenario 3: front-end step graph loop debug (JavaScript)

**Background**: some UI flows serialize steps via `next` references.  
**Why it fits**: when infinite navigation appears, entry node pinpoints broken route config.

```javascript
function detectCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p = head, q = slow;
      while (p !== q) {
        p = p.next;
        q = q.next;
      }
      return p;
    }
  }
  return null;
}
```

---

## R - Reflection

### Complexity

- Time: `O(n)` (detection + localization are linear)
- Space: `O(1)`

### Alternatives and Tradeoffs

| Method | Time | Extra Space | Tradeoff |
| --- | --- | --- | --- |
| Hash set of visited nodes | O(n) | O(n) | easiest to reason, memory-heavy |
| Floyd + reset (current) | O(n) | O(1) | best space profile, requires proof understanding |

### Common Mistakes

1. Compare node values instead of node identity
2. Forget null checks before `fast.next.next`
3. Return first meeting point directly (meeting point is not always entry)
4. Assume `pos` is a runtime argument

### Why this is practical optimum

Under no-modification and low-memory constraints, Floyd + reset provides the best balance:
linear runtime, constant extra memory, deterministic behavior.

---

## FAQ and Notes

1. **Why not return meeting point directly?**  
   Meeting happens somewhere in cycle, but not necessarily at entry.

2. **Can this handle single-node self-cycle?**  
   Yes. `head.next = head` is detected, and entry returned as `head`.

3. **What if list may be concurrently mutated?**  
   Any pointer-based traversal can become unreliable; ensure structural immutability during check.

4. **Can this also return cycle length?**  
   Yes. After first meeting, loop one pointer around cycle to count length.

---

## Best Practices

- Keep a strict loop guard: `while fast != null && fast.next != null`
- Use identity comparison (`is`, pointer equality, `===`) not value equality
- Treat "detect + locate + optional length" as one reusable template family
- In diagnostics, print entry node identifier to accelerate root-cause analysis

---

## S - Summary

- Cycle entry localization is a direct extension of Floyd cycle detection.
- Meeting inside cycle gives enough information to align distances.
- Reset one pointer to head; same-speed walk converges at entry.
- Algorithm is O(n) time, O(1) extra space, and does not mutate list.
- This pattern is broadly reusable for chain-like graph integrity checks.

### Recommended Further Reading

- LeetCode 142. Linked List Cycle II
- LeetCode 141. Linked List Cycle
- Floyd cycle detection proofs and variants
- LeetCode 160 / 234 for linked-list pointer template family

---

## Conclusion

LeetCode 142 is not only a linked-list trick.
It is a compact lesson in invariant-driven reasoning:
detect, align, and localize - all without extra memory.

---

## References

- https://leetcode.com/problems/linked-list-cycle-ii/
- https://leetcode.com/problems/linked-list-cycle/
- https://en.wikipedia.org/wiki/Cycle_detection
- https://cp-algorithms.com/others/tortoise_and_hare.html

---

## Meta Info

- **Reading time**: 12-16 min
- **Tags**: Hot100, linked list, Floyd, cycle entry, two pointers
- **SEO keywords**: Linked List Cycle II, cycle entry, Floyd, O(1) space, LeetCode 142
- **Meta description**: Find cycle entry with Floyd meeting point + pointer reset in O(n) time and O(1) extra space.

---

## Call To Action (CTA)

Try this progression immediately:

1. solve `141` with pure cycle detection
2. upgrade to `142` with entry localization
3. implement cycle length extraction as an extension

Once these three are stable, most linked-list cycle questions become template applications.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from __future__ import annotations


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: ListNode | None = None


def detect_cycle(head: ListNode | None) -> ListNode | None:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            q = slow
            while p is not q:
                p = p.next  # type: ignore[assignment]
                q = q.next  # type: ignore[assignment]
            return p
    return None
```

```c
struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* detectCycle(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            struct ListNode* p = head;
            struct ListNode* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return 0;
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* detectCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            ListNode* p = head;
            ListNode* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return nullptr;
}
```

```go
package main

type ListNode struct {
    Val  int
    Next *ListNode
}

func detectCycle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            p, q := head, slow
            for p != q {
                p = p.Next
                q = q.Next
            }
            return p
        }
    }
    return nil
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct ListNode {
    val: i32,
    next: Option<Rc<RefCell<ListNode>>>,
}

fn next_of(n: &Option<Rc<RefCell<ListNode>>>) -> Option<Rc<RefCell<ListNode>>> {
    n.as_ref().and_then(|x| x.borrow().next.clone())
}

fn ptr_eq(a: &Option<Rc<RefCell<ListNode>>>, b: &Option<Rc<RefCell<ListNode>>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn detect_cycle(head: Option<Rc<RefCell<ListNode>>>) -> Option<Rc<RefCell<ListNode>>> {
    let mut slow = head.clone();
    let mut fast = head.clone();

    while fast.is_some() && next_of(&fast).is_some() {
        slow = next_of(&slow);
        fast = next_of(&next_of(&fast));
        if ptr_eq(&slow, &fast) {
            let mut p = head.clone();
            let mut q = slow.clone();
            while !ptr_eq(&p, &q) {
                p = next_of(&p);
                q = next_of(&q);
            }
            return p;
        }
    }
    None
}
```

```javascript
function detectCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p = head, q = slow;
      while (p !== q) {
        p = p.next;
        q = q.next;
      }
      return p;
    }
  }
  return null;
}
```
