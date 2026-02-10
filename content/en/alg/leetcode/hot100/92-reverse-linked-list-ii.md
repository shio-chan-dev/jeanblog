---
title: "Hot100: Reverse Linked List II Dummy Node + Head-Insertion ACERS Guide"
date: 2026-02-10T09:56:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "sublist reversal", "dummy node", "head insertion", "LeetCode 92"]
description: "Reverse only the interval [left, right] in a singly linked list using a dummy node and in-place head insertion. O(n) time, O(1) extra space, with derivation, pitfalls, and runnable multi-language implementations."
keywords: ["Reverse Linked List II", "sublist reversal", "dummy node", "head insertion", "LeetCode 92", "Hot100", "O(1) space"]
---

> **Subtitle / Summary**  
> Reverse Linked List II is not about full-list reversal; it is about reversing a strict middle interval while preserving both outer connections. This ACERS guide explains the dummy-node anchor, head-insertion loop, and boundary-safe implementation.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `linked list`, `sublist reversal`, `dummy node`  
- **SEO keywords**: Reverse Linked List II, sublist reversal, dummy node, head insertion, LeetCode 92, Hot100  
- **Meta description**: In-place sublist reversal with dummy node + head insertion in O(n)/O(1), with correctness intuition, pitfalls, and runnable multi-language code.

---

## Target Readers

- Hot100 learners who already know 206 and want the interval version
- Developers who often fail at linked-list boundary handling (`left = 1`, `right = n`)
- Engineers building reusable pointer-rewiring templates

## Background / Motivation

LeetCode 206 reverses the whole list. LeetCode 92 asks for a stricter operation:

- reverse only nodes from position `left` to `right`
- keep prefix and suffix connected correctly

This pattern appears in engineering-style chain structures:

- replaying a partial compensation chain in reverse order
- local reorder in an event sequence
- in-place transformation without allocating new nodes

The hard part is not algorithmic complexity; it is pointer safety and boundary consistency.

## Core Concepts

- **Dummy node**: unifies the `left = 1` case with all other cases
- **Anchor predecessor `prev`**: ends at node `left - 1` (or dummy)
- **Current tail `cur`**: starts at `prev.next` and remains tail of reversed block during loop
- **Head insertion**: repeatedly detach `cur.next` and insert it right after `prev`

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a singly linked list and two integers `left` and `right` (`1 <= left <= right <= n`),
reverse the nodes from position `left` to `right`, and return the new head.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of the singly linked list |
| left | int | left boundary (1-based) |
| right | int | right boundary (1-based) |
| return | ListNode | head after sublist reversal |

### Example 1

```text
input:  head = 1 -> 2 -> 3 -> 4 -> 5, left = 2, right = 4
output: 1 -> 4 -> 3 -> 2 -> 5
```

### Example 2

```text
input:  head = 5, left = 1, right = 1
output: 5
```

---

## Thought Process: From Naive to In-Place

### Naive idea: array conversion

- Convert list to array
- Reverse `array[left-1:right]`
- Rebuild list (or rewrite values)

Problems:

- `O(n)` extra memory
- may violate node-identity expectations in real systems

### Key observation

You only need to rewire pointers inside the interval.

After locating `prev` (node before `left`), run this repeatedly:

1. `nxt = cur.next`
2. detach `nxt`: `cur.next = nxt.next`
3. insert `nxt` after `prev`:
   - `nxt.next = prev.next`
   - `prev.next = nxt`

Repeat `right - left` times.

---

## C - Concepts (Core Ideas)

### Method Category

- In-place linked-list rewiring
- Sublist transformation
- Head-insertion loop

### Invariant (the correctness handle)

After each iteration `i` (`0 <= i <= right - left`):

- `prev` still points to the predecessor of the reversing block
- `prev.next` is the head of the already reversed prefix of target block
- `cur` remains the tail of the reversed part and head of remaining unreversed part

At loop end:

- sublist is reversed
- prefix and suffix are still connected

### Pointer Trace (`left=2, right=4`)

```text
start:
1 -> 2 -> 3 -> 4 -> 5
     ^
    cur (prev=1)

round 1 (move 3 after 1):
1 -> 3 -> 2 -> 4 -> 5
          ^
         cur

round 2 (move 4 after 1):
1 -> 4 -> 3 -> 2 -> 5
             ^
            cur
```

---

## Practical Guide / Steps

1. Create `dummy`, set `dummy.next = head`
2. Move `prev` forward `left - 1` steps
3. Set `cur = prev.next`
4. Run `right - left` rounds of head insertion
5. Return `dummy.next`

---

## Runnable Example (Python)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    cur = prev.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next


def build(nums):
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = build([1, 2, 3, 4, 5])
    h = reverse_between(h, 2, 4)
    print(to_list(h))  # [1, 4, 3, 2, 5]
```

---

## Explanation (Why This Works)

Treat `prev` as a fixed anchor before the target interval.
Each loop extracts one node right after `cur` and places it right after `prev`.
That operation grows reversed prefix at the front while keeping the rest linked.

Benefits:

- No segment split + re-join ceremony
- O(1) extra memory
- Uniform behavior on `left = 1` due to dummy node

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Partial compensation-chain replay (Go)

**Background**: reverse execution order for one segment of compensation tasks.  
**Why it fits**: local reorder, identity-preserving nodes, constant memory.

```go
package main

type Node struct {
	Val  int
	Next *Node
}

func reverseBetween(head *Node, left, right int) *Node {
	if head == nil || left == right {
		return head
	}
	dummy := &Node{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
	}
	cur := prev.Next
	for i := 0; i < right-left; i++ {
		nxt := cur.Next
		cur.Next = nxt.Next
		nxt.Next = prev.Next
		prev.Next = nxt
	}
	return dummy.Next
}
```

### Scenario 2: Event-chain local rollback window (Python)

**Background**: only a middle event window needs reverse replay.  
**Why it fits**: precise interval operation without global rebuild.

```python
# Reuse reverse_between(head, left, right) from above.
```

### Scenario 3: Frontend node-flow local reorder (JavaScript)

**Background**: workflow editor supports "reverse selected range".  
**Why it fits**: fast in-memory adjustment, predictable pointer operations.

```javascript
function reverseBetween(head, left, right) {
  if (!head || left === right) return head;
  const dummy = { val: 0, next: head };
  let prev = dummy;
  for (let i = 0; i < left - 1; i += 1) prev = prev.next;
  const cur = prev.next;
  for (let i = 0; i < right - left; i += 1) {
    const nxt = cur.next;
    cur.next = nxt.next;
    nxt.next = prev.next;
    prev.next = nxt;
  }
  return dummy.next;
}
```

---

## R - Reflection

### Complexity

- Time: O(n)
- Space: O(1)

Work split:

- find predecessor in `left - 1` steps
- run `right - left` rewiring rounds

### Alternatives and Tradeoffs

| Approach | Time | Space | Notes |
| --- | --- | --- | --- |
| array conversion | O(n) | O(n) | easy but not in-place |
| cut/reverse/reconnect | O(n) | O(1) | valid, more connection points |
| dummy + head insertion | O(n) | O(1) | concise, boundary-safe, reusable |

### Common Mistakes

- skipping dummy node and breaking `left=1`
- moving `prev` wrong number of steps
- wrong pointer update order causing chain loss
- comparing values instead of node references in linked-list logic

### Why this is the practical template

- single anchor (`prev`)
- single loop (`right-left` rounds)
- single return (`dummy.next`)

Fewer branches, fewer failure points.

---

## FAQ and Notes

1. **What if `left == right`?**  
   Return `head` directly.

2. **Do we need to validate `right` bounds?**  
   LeetCode guarantees valid input; production code should still validate.

3. **Can recursion solve this elegantly?**  
   Yes, but recursion adds stack risk and usually increases boundary complexity.

4. **How is this related to 206?**  
   206 is whole-list reversal; 92 is interval-scoped pointer rewiring built on the same reversal mindset.

---

## Best Practices

- Memorize the 4-line head-insertion block
- Always start from `dummy`
- Dry-run with a 5-node list before coding
- Test 4 boundary cases:
  - `left = 1`
  - `right = n`
  - `left = right`
  - `n = 1`

---

## S - Summary

- LeetCode 92 is an interval rewiring problem, not a value swap problem
- Dummy node removes head-special branching
- Head insertion gives O(1) extra-space reversal
- Invariants are the fastest way to reason about correctness
- This template transfers directly to advanced list reorder problems

### Recommended Follow-up

- LeetCode 206 — Reverse Linked List
- LeetCode 25 — Reverse Nodes in k-Group
- LeetCode 24 — Swap Nodes in Pairs
- LeetCode 143 — Reorder List

---

## Conclusion

Once `dummy + predecessor + head insertion` becomes muscle memory,
sublist reversal stops being pointer chaos and becomes predictable engineering work.

---

## References

- https://leetcode.com/problems/reverse-linked-list-ii/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/book/ch15-01-box.html
- https://go.dev/doc/effective_go

---

## Meta Info

- **Reading time**: 12-15 min  
- **Tags**: Hot100, linked list, sublist reversal, dummy node  
- **SEO keywords**: Reverse Linked List II, sublist reversal, head insertion, LeetCode 92  
- **Meta description**: O(n)/O(1) in-place sublist reversal with dummy node and head insertion.

---

## Call To Action (CTA)

Do two drills now:

1. Reimplement 92 from memory using the 4-line insertion block
2. Move to 25 (k-group reversal) and compare the control flow

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    cur = prev.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

ListNode* reverseBetween(ListNode* head, int left, int right) {
    if (!head || left == right) return head;

    ListNode dummy;
    dummy.val = 0;
    dummy.next = head;

    ListNode* prev = &dummy;
    for (int i = 0; i < left - 1; ++i) prev = prev->next;

    ListNode* cur = prev->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode* nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = prev->next;
        prev->next = nxt;
    }

    return dummy.next;
}
```

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseBetween(ListNode* head, int left, int right) {
    if (!head || left == right) return head;

    ListNode dummy(0);
    dummy.next = head;

    ListNode* prev = &dummy;
    for (int i = 0; i < left - 1; ++i) prev = prev->next;

    ListNode* cur = prev->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode* nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = prev->next;
        prev->next = nxt;
    }

    return dummy.next;
}
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head == nil || left == right {
		return head
	}
	dummy := &ListNode{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
	}
	cur := prev.Next
	for i := 0; i < right-left; i++ {
		nxt := cur.Next
		cur.Next = nxt.Next
		nxt.Next = prev.Next
		prev.Next = nxt
	}
	return dummy.Next
}
```

```rust
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn reverse_between(head: Option<Box<ListNode>>, left: i32, right: i32) -> Option<Box<ListNode>> {
    if left == right {
        return head;
    }

    let mut vals = Vec::new();
    let mut cursor = head.as_ref();
    while let Some(node) = cursor {
        vals.push(node.val);
        cursor = node.next.as_ref();
    }

    let l = (left - 1) as usize;
    let r = (right - 1) as usize;
    vals[l..=r].reverse();

    let mut dummy = Box::new(ListNode::new(0));
    let mut tail = &mut dummy;
    for v in vals {
        tail.next = Some(Box::new(ListNode::new(v)));
        tail = tail.next.as_mut().unwrap();
    }

    dummy.next
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseBetween(head, left, right) {
  if (!head || left === right) return head;

  const dummy = new ListNode(0, head);
  let prev = dummy;
  for (let i = 0; i < left - 1; i += 1) prev = prev.next;

  const cur = prev.next;
  for (let i = 0; i < right - left; i += 1) {
    const nxt = cur.next;
    cur.next = nxt.next;
    nxt.next = prev.next;
    prev.next = nxt;
  }

  return dummy.next;
}
```
