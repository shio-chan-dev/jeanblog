---
title: "Hot100: Reorder List In-Place Split-Reverse-Merge ACERS Guide"
date: 2026-02-10T09:57:31+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "two pointers", "in-place", "LeetCode 143"]
description: "Reorder a linked list to L0->Ln->L1->Ln-1... in O(n) time and O(1) extra space using split, reverse, and alternating merge, with derivation, pitfalls, and runnable multi-language implementations."
keywords: ["Reorder List", "split reverse merge", "linked list", "LeetCode 143", "Hot100", "O(1) space"]
---

> **Subtitle / Summary**  
> Reorder List is a classic pointer choreography problem: find middle, reverse second half, then merge alternately. This guide derives the in-place O(n)/O(1) method from naive ideas and turns it into a reusable Hot100 template.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `linked list`, `in-place`  
- **SEO keywords**: Reorder List, split reverse merge, LeetCode 143, O(1) space  
- **Meta description**: A full ACERS explanation of Reorder List with correctness intuition, boundary handling, engineering mapping, and runnable code in Python/C/C++/Go/Rust/JS.

---

## Target Readers

- Hot100 learners who want a stable linked-list rewire template
- Developers who can reverse lists but still fail on alternating merge details
- Engineers preparing for interviews where O(1) extra space is required

## Background / Motivation

At first glance, this looks like simple reordering.
In reality, it tests whether you can safely perform **three dependent pointer operations** in one workflow:

1. Split one list into two valid lists
2. Reverse one half in-place
3. Alternate-merge without cycles or node loss

Most bugs come from boundary handling and pointer update order, not from "algorithmic complexity" itself.

## Core Concepts

- **Target order**: `L0 -> Ln -> L1 -> Ln-1 -> L2 -> ...`
- **In-place constraint**: do not allocate a new list
- **Three-phase pipeline**:
  1. middle split (`slow/fast`)
  2. reverse second half
  3. alternating merge
- **Critical safety rule**: cut first half tail (`slow.next = null`) before merge

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a singly linked list `head`, reorder it to:

`L0 -> Ln -> L1 -> Ln-1 -> L2 -> Ln-2 -> ...`

Constraints:

- Node values must remain unchanged
- You can only rewire `next` pointers

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | Head of the list |
| return | void | Reorder in-place (head remains first node) |

### Example 1

```text
input:  1 -> 2 -> 3 -> 4
output: 1 -> 4 -> 2 -> 3
```

### Example 2

```text
input:  1 -> 2 -> 3 -> 4 -> 5
output: 1 -> 5 -> 2 -> 4 -> 3
```

---

## Thought Process: From Naive to In-Place Optimal

### Naive idea 1: copy to array, rebuild by two pointers

- Put all nodes into an array
- Use `i` from left and `j` from right
- Reconnect in target order

This is straightforward, but costs O(n) extra memory.

### Naive idea 2: repeatedly find tail and splice

- Keep taking tail and inserting after current head-side node

This becomes O(n^2), too slow for larger inputs.

### Key observation

The target sequence always alternates:

- first half in natural order
- second half in reverse order

So the right decomposition is:

1. split at middle
2. reverse right half once
3. weave two lists alternately

This gives O(n) time and O(1) extra space.

---

## C - Concepts (Core Ideas)

### Method category

- Linked-list pointer manipulation
- Two-pointer middle search
- In-place reversal + zipper merge

### Phase invariants

1. **Split phase**: after cut, left and right are independent lists
2. **Reverse phase**: `prev` is always head of reversed prefix
3. **Merge phase**: each iteration appends exactly one node from right into left chain

### Why the order is correct

- Left half: `L0, L1, L2, ...`
- Reversed right half: `Ln, Ln-1, ...`
- Alternating merge produces exactly:
  `L0, Ln, L1, Ln-1, ...`

No node is duplicated because each node moves from one source list once.

---

## Practice Guide / Steps

1. Handle trivial lists (`0/1/2` nodes): already ordered
2. Use `slow/fast` to find middle
3. Let `second = slow.next`, then `slow.next = null`
4. Reverse `second`
5. Merge `first` and `second` alternately

Runnable Python example (`reorder_list.py`):

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reorder_list(head):
    if head is None or head.next is None:
        return

    # 1) find middle
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # 2) split and reverse second half
    second = slow.next
    slow.next = None
    prev = None
    cur = second
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    second = prev

    # 3) merge two lists alternately
    first = head
    while second:
        n1 = first.next
        n2 = second.next
        first.next = second
        second.next = n1
        first = n1 if n1 else second
        second = n2


def from_list(arr):
    dummy = ListNode()
    tail = dummy
    for x in arr:
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
    h = from_list([1, 2, 3, 4, 5])
    reorder_list(h)
    print(to_list(h))  # [1, 5, 2, 4, 3]
```

---

## Explanation / Why This Works

The whole method depends on one strict ordering:

1. find split point
2. cut list
3. reverse right half
4. weave

If you skip step 2 (cut), merge often creates cycles because old links remain active.

If merge updates pointers in wrong order, nodes are lost.
Always save both `n1` and `n2` before rewiring.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Feed interleaving by freshness and baseline priority (Python)

**Background**: a timeline combines old baseline-ranked items and newest items.
**Why it fits**: alternating merge after reversing one side approximates "front-back interleaving" cheaply.

```python
def interleave_ids(ids):
    left = ids[: (len(ids) + 1) // 2]
    right = list(reversed(ids[(len(ids) + 1) // 2 :]))
    out = []
    i = j = 0
    while i < len(left) or j < len(right):
        if i < len(left):
            out.append(left[i]); i += 1
        if j < len(right):
            out.append(right[j]); j += 1
    return out

print(interleave_ids([1, 2, 3, 4, 5]))  # [1, 5, 2, 4, 3]
```

### Scenario 2: Linked-list task queue reshaping in backend service (Go)

**Background**: a queue represented as linked nodes needs deterministic reshaping without allocating new nodes.
**Why it fits**: split-reverse-merge is O(1) extra memory and predictable.

```go
package main

import "fmt"

type Node struct {
	Val  int
	Next *Node
}

func build(a []int) *Node {
	dummy := &Node{}
	p := dummy
	for _, x := range a {
		p.Next = &Node{Val: x}
		p = p.Next
	}
	return dummy.Next
}

func toSlice(h *Node) []int {
	ans := []int{}
	for h != nil {
		ans = append(ans, h.Val)
		h = h.Next
	}
	return ans
}

func main() {
	head := build([]int{1, 2, 3, 4})
	// production code would call reorderList(head)
	fmt.Println(toSlice(head))
}
```

### Scenario 3: Frontend card stream alternating recent and legacy blocks (JavaScript)

**Background**: a client-side list needs quick visual alternation for A/B display experiments.
**Why it fits**: same structural pattern can be reused on arrays.

```javascript
function reorderArray(arr) {
  const left = arr.slice(0, Math.ceil(arr.length / 2));
  const right = arr.slice(Math.ceil(arr.length / 2)).reverse();
  const out = [];
  let i = 0;
  let j = 0;
  while (i < left.length || j < right.length) {
    if (i < left.length) out.push(left[i++]);
    if (j < right.length) out.push(right[j++]);
  }
  return out;
}

console.log(reorderArray([1, 2, 3, 4, 5])); // [1, 5, 2, 4, 3]
```

---

## R - Reflection (Complexity, Alternatives, Pitfalls)

### Complexity

- Time: O(n)
- Extra space: O(1)

### Alternatives and tradeoffs

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Array rebuild | O(n) | O(n) | Easy but violates strict in-place goal |
| Repeated tail extraction | O(n^2) | O(1) | Too slow |
| Split + reverse + merge | O(n) | O(1) | Best practical template |

### Common mistakes

1. Forgetting to cut at middle (`slow.next = null`) causing cycles
2. Wrong middle condition, especially for even length
3. Merge order bug: rewiring before saving next pointers
4. Not handling short lists (null or single node)

### Why this is the optimal practical method

It matches constraints exactly:

- in-place
- linear time
- no extra container

And each sub-step is reusable across multiple linked-list problems.

---

## FAQ and Notes

1. **Why is this not a palindrome-like compare problem?**  
   Because we must physically reorder links, not just compare values.

2. **Can recursion solve it cleanly?**  
   Yes in theory, but stack usage becomes O(n) and implementation is trickier.

3. **Do node values matter?**  
   No. This is pointer topology, not value sorting.

---

## Best Practices

- Treat split/reverse/merge as three isolated templates, then compose
- Write and reuse helper functions in production code
- Validate with odd/even lengths and minimal lists
- Use pointer diagrams when debugging merge order

---

## S - Summary

- Reorder List is solved by split -> reverse -> alternating merge
- The key optimization is recognizing "second half reversed" as required shape
- Correctness depends on strict pointer update order and middle cut
- This template is foundational for many advanced linked-list problems

### Further Reading

- LeetCode 143. Reorder List
- LeetCode 206. Reverse Linked List
- LeetCode 234. Palindrome Linked List
- LeetCode 25. Reverse Nodes in k-Group

---

## Conclusion

If you can implement this problem without pointer bugs under interview pressure, your linked-list manipulation skill is already at a solid intermediate level.
The same split/reverse/merge workflow appears repeatedly in production-grade list transformations.

---

## References

- https://leetcode.com/problems/reorder-list/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/std/option/

---

## Meta Info

- **Reading time**: 12-15 min
- **Tags**: Hot100, linked list, in-place, two pointers
- **SEO keywords**: Reorder List, LeetCode 143, split reverse merge, O(1) space
- **Meta description**: In-place O(n)/O(1) linked-list reordering with derivation, pitfalls, and multi-language implementations.

---

## CTA

Try coding this from scratch in 15 minutes without looking at notes.
Then extend it to: reverse k-group and palindrome linked list to lock in the pointer templates.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reorder_list(head):
    if head is None or head.next is None:
        return

    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    second = slow.next
    slow.next = None

    prev = None
    cur = second
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    second = prev

    first = head
    while second:
        n1 = first.next
        n2 = second.next
        first.next = second
        second.next = n1
        first = n1 if n1 else second
        second = n2
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

void reorderList(ListNode* head) {
    if (!head || !head->next) return;

    ListNode *slow = head, *fast = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    ListNode* second = slow->next;
    slow->next = NULL;

    ListNode *prev = NULL, *cur = second;
    while (cur) {
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    second = prev;

    ListNode* first = head;
    while (second) {
        ListNode* n1 = first->next;
        ListNode* n2 = second->next;
        first->next = second;
        second->next = n1;
        first = n1 ? n1 : second;
        second = n2;
    }
}
```

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v) : val(v), next(nullptr) {}
};

class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next) return;

        ListNode *slow = head, *fast = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* second = slow->next;
        slow->next = nullptr;

        ListNode* prev = nullptr;
        while (second) {
            ListNode* nxt = second->next;
            second->next = prev;
            prev = second;
            second = nxt;
        }
        second = prev;

        ListNode* first = head;
        while (second) {
            ListNode* n1 = first->next;
            ListNode* n2 = second->next;
            first->next = second;
            second->next = n1;
            first = n1 ? n1 : second;
            second = n2;
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

func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}

	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	second := slow.Next
	slow.Next = nil

	var prev *ListNode
	cur := second
	for cur != nil {
		nxt := cur.Next
		cur.Next = prev
		prev = cur
		cur = nxt
	}
	second = prev

	first := head
	for second != nil {
		n1 := first.Next
		n2 := second.Next
		first.Next = second
		second.Next = n1
		if n1 != nil {
			first = n1
		} else {
			first = second
		}
		second = n2
	}
}
```

```rust
use std::collections::VecDeque;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

// Safe Rust variant: uses O(n) extra deque to avoid unsafe pointer rewiring.
pub fn reorder_list(head: &mut Option<Box<ListNode>>) {
    let mut cur = head.take();
    let mut dq: VecDeque<Box<ListNode>> = VecDeque::new();

    while let Some(mut node) = cur {
        cur = node.next.take();
        dq.push_back(node);
    }

    let mut reordered: Vec<Box<ListNode>> = Vec::with_capacity(dq.len());
    let mut pick_front = true;
    while !dq.is_empty() {
        if pick_front {
            reordered.push(dq.pop_front().unwrap());
        } else {
            reordered.push(dq.pop_back().unwrap());
        }
        pick_front = !pick_front;
    }

    let mut new_head: Option<Box<ListNode>> = None;
    for mut node in reordered.into_iter().rev() {
        node.next = new_head;
        new_head = Some(node);
    }
    *head = new_head;
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reorderList(head) {
  if (!head || !head.next) return;

  let slow = head;
  let fast = head;
  while (fast.next && fast.next.next) {
    slow = slow.next;
    fast = fast.next.next;
  }

  let second = slow.next;
  slow.next = null;

  let prev = null;
  while (second) {
    const nxt = second.next;
    second.next = prev;
    prev = second;
    second = nxt;
  }
  second = prev;

  let first = head;
  while (second) {
    const n1 = first.next;
    const n2 = second.next;
    first.next = second;
    second.next = n1;
    first = n1 || second;
    second = n2;
  }
}
```
