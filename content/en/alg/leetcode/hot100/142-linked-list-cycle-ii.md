---
title: "Hot100: Linked List Cycle II Floyd Detection + Entry Localization ACERS Guide"
date: 2026-02-10T10:47:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "fast slow pointers", "Floyd", "two pointers", "LeetCode 142"]
description: "Return the first node where a singly linked list enters a cycle without modifying the list: use Floyd fast/slow pointers to detect a meeting point, then reset one pointer to head and move both one step to localize the entry in O(n) time and O(1) space."
keywords: ["Linked List Cycle II", "cycle entry", "Floyd", "fast slow pointers", "O(1) space", "LeetCode 142", "Hot100"]
---

> **Subtitle / Summary**  
> LeetCode 142 upgrades cycle detection into cycle entry localization. The robust template is Floyd: first detect a meeting inside the cycle, then reset one pointer to `head` and move both by one step; the next meeting node is the cycle entry.

- **Reading time**: 12-16 min  
- **Tags**: `Hot100`, `linked list`, `fast slow pointers`, `Floyd`  
- **SEO keywords**: Linked List Cycle II, cycle entry, Floyd, fast slow pointers, O(1) space, LeetCode 142, Hot100  
- **Meta description**: Floyd cycle detection + entry localization with proof intuition, engineering mapping, and runnable multi-language implementations in O(n) time and O(1) extra space.

---

## Target Readers

- Hot100 learners who want to fully internalize the `141 -> 142` linked-list template family
- Developers who need to locate where a pointer chain becomes cyclic
- Interview candidates who want to explain why "reset to head" works

## Background / Motivation

In real systems, cycle corruption in chain structures can cause:

- endless traversal loops
- stuck cleanup tasks
- misleadingly stable but non-progressing runtime behavior

Detecting whether a cycle exists is helpful, but operations/debugging usually require more:

- **where does the cycle begin?**

That exact requirement is modeled by LeetCode 142.

## Core Concepts

| Concept | Meaning | Why it matters |
| --- | --- | --- |
| Cycle | Following `next` eventually revisits a node | causes non-terminating traversals |
| Entry node | First node where linear prefix enters the loop | required return value |
| Floyd algorithm | `slow` moves 1 step, `fast` moves 2 steps | O(1) extra memory |
| Meeting point | First collision inside cycle | bridge to entry localization |
| Identity equality | compare node reference, not node value | value duplicates are common |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given `head` of a singly linked list, return the node where the cycle begins.
If there is no cycle, return `null`.

Notes:

- `pos` in the statement is only for test-data construction
- `pos` is not a function argument
- list structure must not be modified

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list |
| return | ListNode / null | entry node reference, or null |

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

Traverse nodes and store references in a hash set:

- if current node already exists in set, it is the entry
- if traversal reaches `null`, there is no cycle

Pros: straightforward.
Cons: O(n) extra space.

### Constraint-driven upgrade

Need O(1) extra memory while still returning the entry node.

### Key observation

Floyd gives two phases:

1. **Detection**: if `slow` and `fast` meet, cycle exists
2. **Localization**: move one pointer to `head`, keep the other at meeting point; step both by 1 until they meet again

Second meeting point is exactly the cycle entry.

---

## C - Concepts (Core Ideas)

### Method Category

- Two pointers (fast/slow)
- Floyd cycle detection
- Distance alignment after reset

### Why reset-to-head works

Define:

- `a`: distance from `head` to cycle entry
- `b`: distance from entry to first meeting point
- `c`: cycle length

At first meeting:

- `slow` traveled `a + b`
- `fast` traveled `2(a + b)`

Difference is one whole number of cycle lengths:

```text
2(a + b) - (a + b) = k * c
=> a + b = k * c
=> a = k * c - b
```

Meaning:

- from head to entry: `a` steps
- from meeting point to entry: also `a` steps modulo cycle length

So moving one pointer from `head` and one from `meeting`, both one step each round, guarantees meeting at the entry.

### Invariant for localization phase

During phase 2, both pointers have equal remaining distance (modulo cycle) to entry.
Equal-speed synchronous movement preserves this equality, so collision point is entry.

---

## Practical Guide / Steps

1. Initialize `slow = head`, `fast = head`
2. Detection loop:
   - `slow = slow.next`
   - `fast = fast.next.next`
   - if pointers meet, break
3. If `fast == null` or `fast.next == null`, no cycle -> return `null`
4. Set `p1 = head`, `p2 = meeting`
5. Move both by one step until `p1 == p2`
6. Return `p1` (entry)

---

## Runnable Example (Python)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p1 = head
            p2 = slow
            while p1 is not p2:
                p1 = p1.next
                p2 = p2.next
            return p1
    return None


def build_cycle_list(values, pos):
    dummy = ListNode()
    tail = dummy
    entry = None
    for idx, x in enumerate(values):
        tail.next = ListNode(x)
        tail = tail.next
        if idx == pos:
            entry = tail
    if tail and pos >= 0:
        tail.next = entry
    return dummy.next


if __name__ == "__main__":
    h = build_cycle_list([3, 2, 0, -4], 1)
    e = detect_cycle(h)
    print(e.val if e else None)  # 2
```

---

## Explanation / Why This Works

The algorithm is split by responsibility:

- phase 1 answers: cycle or no cycle
- phase 2 answers: where cycle starts

This separation is important for correctness and debugging.

Setting one pointer to `head` is not a trick; it is distance alignment derived from meeting-point equations.
That is why this method is both memory-optimal and proof-friendly.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: asynchronous callback chain corruption check (Go)

**Background**: callback chain unexpectedly loops in production.  
**Why it fits**: no extra map allocation, works on large chains.

```go
package main

type Node struct {
	Val  int
	Next *Node
}

func detectCycle(head *Node) *Node {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			p1, p2 := head, slow
			for p1 != p2 {
				p1 = p1.Next
				p2 = p2.Next
			}
			return p1
		}
	}
	return nil
}
```

### Scenario 2: ETL pointer-chain anomaly localization (Python)

**Background**: transformed record chain can accidentally self-link.  
**Why it fits**: allows locating the first bad join node directly.

```python
# Reuse detect_cycle(head) above and log entry node identity/value.
```

### Scenario 3: front-end linked-state graph guard (JavaScript)

**Background**: linked state nodes in memory may form unintended cycles.
**Why it fits**: fast runtime check in debug tooling.

```javascript
function detectCycle(head) {
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p1 = head;
      let p2 = slow;
      while (p1 !== p2) {
        p1 = p1.next;
        p2 = p2.next;
      }
      return p1;
    }
  }
  return null;
}
```

---

## R - Reflection

### Complexity

- Time: O(n)
- Extra space: O(1)

### Alternatives and Tradeoffs

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| visited hash set | O(n) | O(n) | easy, but extra memory |
| Floyd + reset | O(n) | O(1) | optimal for constraints |

### Common Mistakes

- forgetting `fast && fast.next` null checks
- comparing node values (`val`) instead of references
- returning meeting point directly (meeting is not always entry)
- modifying list to mark visited nodes (forbidden by problem)

### Why this method is optimal in practice

- no structural mutation
- no extra memory pressure
- deterministic behavior under large lists
- strong proof story for interviews and code review

---

## FAQ and Notes

1. **Why is the first meeting point not necessarily the entry?**  
   Because fast and slow can first collide anywhere inside the cycle.

2. **Can this fail with duplicate values?**  
   No, if you compare references (`is`, `== pointer`) rather than values.

3. **What if list length is very small?**  
   Null checks naturally handle `0` or `1` node lists.

4. **Can we stop once cycle is detected?**  
   For LeetCode 141 yes; for 142 you must run localization phase.

---

## Best Practices

- Implement `141` and `142` as a pair to reuse mental model
- Keep phase separation explicit in code (`detect` then `locate`)
- Add tests for:
  - no cycle
  - single-node self-cycle
  - cycle entry at head
  - cycle entry in middle
- In production debug tools, print entry node identity and predecessor info when possible

---

## S - Summary

- LeetCode 142 extends cycle detection to entry localization
- Floyd detects cycle with O(1) memory
- Reset-to-head works by distance alignment, not memorized magic
- Correct implementation depends on reference comparison and null-safe traversal
- This pattern is reusable for many chain-integrity diagnostics

### Recommended Follow-up

- LeetCode 141 — Linked List Cycle
- LeetCode 160 — Intersection of Two Linked Lists
- Floyd cycle-finding notes in pointer-heavy systems
- General linked-list invariants and mutation safety patterns

---

## Conclusion

Once you understand "detect first, then reset and align distances",
Linked List Cycle II becomes a stable template rather than a memorized trick.

---

## References

- https://leetcode.com/problems/linked-list-cycle-ii/
- https://en.wikipedia.org/wiki/Cycle_detection
- https://en.cppreference.com/w/cpp/language/pointer
- https://go.dev/doc/effective_go

---

## Meta Info

- **Reading time**: 12-16 min  
- **Tags**: Hot100, linked list, fast/slow pointers, Floyd  
- **SEO keywords**: Linked List Cycle II, cycle entry, Floyd, LeetCode 142  
- **Meta description**: O(n)/O(1) cycle entry localization with Floyd fast/slow pointers and reset alignment proof.

---

## Call To Action (CTA)

Run this mini drill:

1. Re-implement 141 and 142 back-to-back from memory
2. Write the distance equation once (`a+b = k*c`) and explain it aloud
3. Build four edge-case tests and verify pointer identity-based assertions

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p1 = head
            p2 = slow
            while p1 is not p2:
                p1 = p1.next
                p2 = p2.next
            return p1
    return None
```

```c
struct ListNode {
    int val;
    struct ListNode *next;
};

struct ListNode *detectCycle(struct ListNode *head) {
    struct ListNode *slow = head;
    struct ListNode *fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            struct ListNode *p1 = head;
            struct ListNode *p2 = slow;
            while (p1 != p2) {
                p1 = p1->next;
                p2 = p2->next;
            }
            return p1;
        }
    }
    return NULL;
}
```

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head;
        ListNode *fast = head;

        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                ListNode *p1 = head;
                ListNode *p2 = slow;
                while (p1 != p2) {
                    p1 = p1->next;
                    p2 = p2->next;
                }
                return p1;
            }
        }
        return nullptr;
    }
};
```

```go
func detectCycle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            p1, p2 := head, slow
            for p1 != p2 {
                p1 = p1.Next
                p2 = p2.Next
            }
            return p1
        }
    }
    return nil
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Link = Option<Rc<RefCell<ListNode>>>;

#[derive(Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Link,
}

pub fn detect_cycle(head: Link) -> Link {
    let mut slow = head.clone();
    let mut fast = head.clone();

    loop {
        slow = match slow.clone() {
            Some(node) => node.borrow().next.clone(),
            None => return None,
        };

        fast = match fast.clone() {
            Some(node) => match node.borrow().next.clone() {
                Some(next1) => next1.borrow().next.clone(),
                None => return None,
            },
            None => return None,
        };

        match (slow.clone(), fast.clone()) {
            (Some(s), Some(f)) if Rc::ptr_eq(&s, &f) => break,
            (Some(_), Some(_)) => {}
            _ => return None,
        }
    }

    let mut p1 = head;
    let mut p2 = slow;
    loop {
        match (p1.clone(), p2.clone()) {
            (Some(a), Some(b)) => {
                if Rc::ptr_eq(&a, &b) {
                    return Some(a);
                }
                p1 = a.borrow().next.clone();
                p2 = b.borrow().next.clone();
            }
            _ => return None,
        }
    }
}
```

```javascript
function detectCycle(head) {
  let slow = head;
  let fast = head;

  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p1 = head;
      let p2 = slow;
      while (p1 !== p2) {
        p1 = p1.next;
        p2 = p2.next;
      }
      return p1;
    }
  }
  return null;
}
```
