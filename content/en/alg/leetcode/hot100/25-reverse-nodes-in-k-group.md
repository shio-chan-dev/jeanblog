---
title: "Hot100: Reverse Nodes in k-Group Group-Wise In-Place ACERS Guide"
date: 2026-02-10T10:25:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "group reversal", "dummy node", "two pointers", "LeetCode 25"]
description: "Reverse a linked list in groups of k nodes in-place, leaving the last group unchanged if its size is smaller than k. This guide presents a dummy-node + kth-scan + in-place reversal template in O(n) time and O(1) extra space."
keywords: ["Reverse Nodes in k-Group", "group reversal", "linked list", "LeetCode 25", "Hot100", "O(1) space"]
---

> **Subtitle / Summary**  
> LeetCode 25 is the upgrade path from 206 (full reversal) and 92 (interval reversal): split by groups, reverse inside each full group, reconnect safely, and keep the last incomplete group unchanged.

- **Reading time**: 14-18 min  
- **Tags**: `Hot100`, `linked list`, `group reversal`, `dummy node`  
- **SEO keywords**: Reverse Nodes in k-Group, group reversal, LeetCode 25, Hot100  
- **Meta description**: In-place k-group linked-list reversal with dummy-node anchoring and safe reconnection, including pitfalls, complexity, and runnable multi-language implementations.

---

## Target Readers

- Hot100 learners who already finished 206/92 and want the next linked-list jump
- Developers who often fail at boundary handling and reconnection in pointer-heavy tasks
- Engineers building reusable templates for chunk-based list transformation

## Background / Motivation

In real systems, chain structures are often processed in batches:

- replay compensation tasks per fixed batch size
- local reorder of pipeline nodes by chunk
- in-place structure rewrite without reallocating all nodes

These tasks require:

- transformation **inside each batch**
- stable order **between batches**
- clear rule for incomplete tail batches (keep unchanged)

LeetCode 25 models exactly this requirement.

## Core Concepts

- **Dummy node**: removes head-special branches
- **`groupPrev`**: predecessor of the current group
- **`kth` probe**: checks whether a full group of size `k` exists
- **`groupNext`**: first node after the current group
- **In-place group reversal**: reverse only `[groupStart, kth]`

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a linked list and an integer `k`, reverse nodes in groups of size `k` and return the modified head.  
If the number of remaining nodes is less than `k`, keep them in original order.  
You must modify pointers, not just node values.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list |
| k | int | group size (`k >= 1`) |
| return | ListNode | new head after group-wise reversal |

### Example 1

```text
input:  head = 1 -> 2 -> 3 -> 4 -> 5, k = 2
output: 2 -> 1 -> 4 -> 3 -> 5
```

### Example 2

```text
input:  head = 1 -> 2 -> 3 -> 4 -> 5, k = 3
output: 3 -> 2 -> 1 -> 4 -> 5
```

---

## Thought Process: From Naive to Optimal

### Naive approach: array conversion and rebuild

- Convert linked list to array
- Reverse every full k-sized segment
- Rebuild list from array

Problems:

- O(n) extra memory
- violates pointer-modification intent in many interview/engineering contexts

### Key observation

The process can be decomposed into a repeating loop:

1. verify current group has `k` nodes
2. reverse this group in-place
3. reconnect and move to next group

This is "interval reversal" repeated under group control.

### Method selection

Use:

- `dummy + groupPrev` to anchor global structure
- `kth` scanning to validate group completeness
- in-place pointer reversal per full group

Result:

- O(n) time
- O(1) extra space

---

## C - Concepts (Core Ideas)

### Method Category

- In-place linked-list rewiring
- Chunk/batch processing
- Boundary locating with predecessor + kth probe

### Loop Invariant

At the start of each iteration:

- `groupPrev.next` points to the first node of the next unprocessed group
- all nodes before `groupPrev` are already in final form

After one successful group operation:

- current group is reversed and reconnected correctly
- `groupPrev` moves to the new tail of this reversed group (old group head)

This guarantees stable forward progress without breaking previous groups.

### Structure Sketch (`k=3`)

```text
dummy -> a -> b -> c -> d -> e -> f -> g
          ^         ^
      groupStart   kth

reverse [a,b,c] =>
dummy -> c -> b -> a -> d -> e -> f -> g
                     ^
                 new groupPrev
```

---

## Practical Guide / Steps

1. Initialize `dummy.next = head`, set `groupPrev = dummy`
2. Scan `k` steps from `groupPrev` to find `kth`
   - if missing, stop (tail group is incomplete)
3. Save `groupNext = kth.next`
4. Reverse `[groupPrev.next, kth]` with `prev = groupNext`
5. Reconnect:
   - `groupPrev.next` -> new group head
   - old group head becomes new group tail
6. Move `groupPrev` to new group tail and continue

---

## Runnable Example (Python)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or k <= 1:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next

        group_next = kth.next
        prev = group_next
        cur = group_prev.next

        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        new_group_tail = group_prev.next
        group_prev.next = prev
        group_prev = new_group_tail


def build(nums):
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head):
    ans = []
    while head:
        ans.append(head.val)
        head = head.next
    return ans


if __name__ == "__main__":
    h = build([1, 2, 3, 4, 5])
    print(to_list(reverse_k_group(h, 2)))  # [2, 1, 4, 3, 5]
```

---

## Explanation (Why This Works)

`groupPrev` acts as a stable anchor before the current group.  
Inside each group, set `prev = groupNext` before reversal, so the reversed tail automatically reconnects to the next segment.

This removes extra branch logic and keeps each group operation structurally identical.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Batch compensation-chain replay (Go)

**Background**: reverse order inside each fixed-size task batch.  
**Why it fits**: in-place, bounded memory, clear tail behavior.

```go
package main

type Node struct {
	Val  int
	Next *Node
}

func reverseKGroup(head *Node, k int) *Node {
	if head == nil || k <= 1 {
		return head
	}
	dummy := &Node{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k; i++ {
			kth = kth.Next
			if kth == nil {
				return dummy.Next
			}
		}

		groupNext := kth.Next
		prev, cur := groupNext, groupPrev.Next
		for cur != groupNext {
			nxt := cur.Next
			cur.Next = prev
			prev = cur
			cur = nxt
		}

		tail := groupPrev.Next
		groupPrev.Next = prev
		groupPrev = tail
	}
}
```

### Scenario 2: Segment rollback in event chains (Python)

**Background**: replay/rollback events by fixed group windows.  
**Why it fits**: exact group control and deterministic pointer behavior.

```python
# Reuse reverse_k_group(head, k) from above.
```

### Scenario 3: Workflow editor chunk reversal (JavaScript)

**Background**: UI supports "reverse every k selected nodes".  
**Why it fits**: efficient in-memory operation without rebuilding the whole chain.

```javascript
function reverseKGroup(head, k) {
  if (!head || k <= 1) return head;
  const dummy = { val: 0, next: head };
  let groupPrev = dummy;

  while (true) {
    let kth = groupPrev;
    for (let i = 0; i < k; i += 1) {
      kth = kth.next;
      if (!kth) return dummy.next;
    }

    const groupNext = kth.next;
    let prev = groupNext;
    let cur = groupPrev.next;

    while (cur !== groupNext) {
      const nxt = cur.next;
      cur.next = prev;
      prev = cur;
      cur = nxt;
    }

    const newGroupTail = groupPrev.next;
    groupPrev.next = prev;
    groupPrev = newGroupTail;
  }
}
```

---

## R - Reflection

### Complexity Analysis

- Time: O(n)  
  Each node is visited and rewired a constant number of times.
- Space: O(1)  
  Only constant pointer variables are used.

### Alternatives and Tradeoffs

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| array conversion | O(n) | O(n) | simpler, but not in-place |
| recursive group reversal | O(n) | O(n/k)~O(n) stack | elegant but stack-risky |
| iterative in-place group reversal | O(n) | O(1) | production-friendly default |

### Common Mistakes

- not checking whether `kth` exists before reversing
- forgetting to move `groupPrev` after each group (can cause infinite loops)
- reversing incomplete tail group (violates problem requirement)
- swapping values instead of rewiring nodes

### Why this approach is optimal in practice

- linear time
- constant memory
- explicit and testable boundary control

It is the most stable template for interview and production-style pointer work.

---

## FAQ and Notes

1. **What if `k = 1`?**  
   Return original list.

2. **What if list length is not divisible by `k`?**  
   Keep the final incomplete group unchanged.

3. **Can recursion solve this problem?**  
   Yes, but iterative form is safer for stack depth and usually easier to debug.

4. **How is this related to 92?**  
   LeetCode 25 is repeated interval reversal (92) under fixed-size grouping.

---

## Best Practices

- Memorize the pattern: `dummy -> find kth -> reverse group -> reconnect -> move groupPrev`
- Log `groupPrev`, `kth`, `groupNext` during debugging
- Validate with edge sets: `k=1`, `k=2`, `k=len`, `k>len`
- Review together with 206 and 92 as a linked-list reversal trilogy

---

## S - Summary

- LeetCode 25 is group-driven interval reversal
- Dummy node unifies head-boundary handling
- `kth` probing is the correctness gate before each reversal
- In-place rewiring achieves O(n)/O(1)
- This template generalizes to many advanced list reordering tasks

### Recommended Follow-up

- LeetCode 206 — Reverse Linked List
- LeetCode 92 — Reverse Linked List II
- LeetCode 24 — Swap Nodes in Pairs
- LeetCode 143 — Reorder List

---

## Conclusion

Once "k-group scan + local reversal + safe reconnection" becomes your default pattern,
LeetCode 25 becomes predictable pointer engineering instead of fragile pointer gymnastics.

---

## References

- https://leetcode.com/problems/reverse-nodes-in-k-group/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/book/ch15-01-box.html
- https://go.dev/doc/effective_go

---

## Meta Info

- **Reading time**: 14-18 min  
- **Tags**: Hot100, linked list, group reversal, dummy node  
- **SEO keywords**: Reverse Nodes in k-Group, LeetCode 25, Hot100  
- **Meta description**: O(n)/O(1) in-place k-group linked-list reversal with robust boundary handling.

---

## Call To Action (CTA)

Do this practice loop now:

1. Re-implement 25 from memory without looking
2. Adapt same structure to 24 (pair swap)
3. Compare with 92 to internalize single-interval vs repeated-group control

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or k <= 1:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next

        group_next = kth.next
        prev = group_next
        cur = group_prev.next

        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        new_group_tail = group_prev.next
        group_prev.next = prev
        group_prev = new_group_tail
```

```c
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

ListNode* reverseKGroup(ListNode* head, int k) {
    if (!head || k <= 1) return head;

    ListNode dummy;
    dummy.val = 0;
    dummy.next = head;

    ListNode* groupPrev = &dummy;

    while (1) {
        ListNode* kth = groupPrev;
        for (int i = 0; i < k; ++i) {
            kth = kth->next;
            if (!kth) return dummy.next;
        }

        ListNode* groupNext = kth->next;
        ListNode* prev = groupNext;
        ListNode* cur = groupPrev->next;

        while (cur != groupNext) {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }

        ListNode* newGroupTail = groupPrev->next;
        groupPrev->next = prev;
        groupPrev = newGroupTail;
    }
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head || k <= 1) return head;

        ListNode dummy(0);
        dummy.next = head;
        ListNode* groupPrev = &dummy;

        while (true) {
            ListNode* kth = groupPrev;
            for (int i = 0; i < k; ++i) {
                kth = kth->next;
                if (!kth) return dummy.next;
            }

            ListNode* groupNext = kth->next;
            ListNode* prev = groupNext;
            ListNode* cur = groupPrev->next;

            while (cur != groupNext) {
                ListNode* nxt = cur->next;
                cur->next = prev;
                prev = cur;
                cur = nxt;
            }

            ListNode* newGroupTail = groupPrev->next;
            groupPrev->next = prev;
            groupPrev = newGroupTail;
        }
    }
};
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil || k <= 1 {
		return head
	}
	dummy := &ListNode{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k; i++ {
			kth = kth.Next
			if kth == nil {
				return dummy.Next
			}
		}

		groupNext := kth.Next
		prev, cur := groupNext, groupPrev.Next
		for cur != groupNext {
			nxt := cur.Next
			cur.Next = prev
			prev = cur
			cur = nxt
		}

		newGroupTail := groupPrev.Next
		groupPrev.Next = prev
		groupPrev = newGroupTail
	}
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

pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let k = k as usize;
    if k <= 1 {
        return head;
    }

    let mut dummy = Box::new(ListNode { val: 0, next: head });
    let mut group_prev: &mut Box<ListNode> = &mut dummy;

    loop {
        let mut check = group_prev.next.as_ref();
        for _ in 0..k {
            match check {
                Some(node) => check = node.next.as_ref(),
                None => return dummy.next,
            }
        }

        let mut cur = group_prev.next.take();
        let mut rev: Option<Box<ListNode>> = None;
        for _ in 0..k {
            let mut node = cur.unwrap();
            cur = node.next.take();
            node.next = rev;
            rev = Some(node);
        }

        group_prev.next = rev;
        for _ in 0..k {
            group_prev = group_prev.next.as_mut().unwrap();
        }
        group_prev.next = cur;
    }
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseKGroup(head, k) {
  if (!head || k <= 1) return head;
  const dummy = new ListNode(0, head);
  let groupPrev = dummy;

  while (true) {
    let kth = groupPrev;
    for (let i = 0; i < k; i += 1) {
      kth = kth.next;
      if (!kth) return dummy.next;
    }

    const groupNext = kth.next;
    let prev = groupNext;
    let cur = groupPrev.next;

    while (cur !== groupNext) {
      const nxt = cur.next;
      cur.next = prev;
      prev = cur;
      cur = nxt;
    }

    const newGroupTail = groupPrev.next;
    groupPrev.next = prev;
    groupPrev = newGroupTail;
  }
}
```
