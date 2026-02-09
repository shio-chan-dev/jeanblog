---
title: "Hot100: Intersection of Two Linked Lists Two-Pointer Switch-Head O(1) Space ACERS Guide"
date: 2026-02-09T09:26:37+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "two pointers", "hash", "LeetCode 160"]
description: "Find the intersection node of two singly linked lists without modifying structure: two pointers switching heads synchronize in O(m+n) time and O(1) extra space, with derivation, engineering mapping, and multi-language implementations."
keywords: ["Intersection of Two Linked Lists", "two pointers", "switch heads", "O(1) space", "LeetCode 160", "Hot100"]
---

> **Subtitle / Summary**  
> The key is not comparing values, but comparing node identity (same object / same address). This ACERS guide explains the naive hash approach, the length-alignment approach, and the most practical switch-head two-pointer template, with runnable multi-language implementations under the no-modification and no-cycle constraints.

- **Reading time**: 10-14 min  
- **Tags**: `Hot100`, `linked list`, `two pointers`  
- **SEO keywords**: Intersection of Two Linked Lists, switch heads, O(1) space, LeetCode 160  
- **Meta description**: Two pointers walk A then B and B then A, guaranteeing meeting at the intersection or both reaching null within m+n steps, with O(m+n) time and O(1) space.

---

## Target Readers

- Hot100 learners who want a reusable linked-list two-pointer template
- Developers who often confuse "same value" with "same node"
- Engineers working with shared tail structures in chain-like data

## Background / Motivation

This problem looks simple, but it forces you to separate three concepts:

1. **Intersection means sharing the exact same node object**, not equal values
2. You cannot modify structure (no rewriting `next`, no marking nodes)
3. You still need linear performance

The most practical solution is the switch-head two-pointer method.
It needs no hash set and no precomputed lengths, yet synchronizes both pointers in at most `m+n` steps.

## Core Concepts

| Concept | Meaning | Note |
| --- | --- | --- |
| Same node | Two pointers reference the same memory object | Pointer/reference equality |
| Shared suffix | Two lists share all nodes from some node onward | After intersection, tails are identical |
| Switch-head two pointers | At list end, jump to the other list head | Equalizes total traveled distance |
| No-cycle assumption | Problem guarantees no cycle in the structure | Otherwise cycle handling is required |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given heads `headA` and `headB` of two singly linked lists, return the node where they intersect.
If they do not intersect, return `null`.

Constraints:

- The linked structure has no cycle
- The original list structure must remain unchanged

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| headA | ListNode | Head of list A |
| headB | ListNode | Head of list B |
| return | ListNode / null | Intersection start node (same object), or null |

### Example 1 (Intersecting)

```text
A: a1 -> a2 -> c1 -> c2 -> c3
B: b1 -> b2 -> b3 -> c1 -> c2 -> c3

output: c1 (return node reference, not value)
```

### Example 2 (No intersection)

```text
A: 1 -> 2 -> 3
B: 4 -> 5

output: null
```

---

## Thought Process: From Hash to O(1) Template

### Naive approach: hash all nodes in A

1. Traverse A and put each node address into a hash set
2. Traverse B and return the first node found in the set

Pros: direct and easy to implement.
Cons: O(m) extra space.

### O(1) approach #1: length alignment

1. Compute lengths `m` and `n`
2. Advance the longer list by `abs(m-n)`
3. Move both pointers together until they meet

This is O(1) space, but requires separate length passes.

### O(1) approach #2 (most practical): switch-head two pointers

Initialize `pA=headA`, `pB=headB`:

- Move one step each round
- If a pointer reaches `null`, redirect it to the other list head

Intuition:
`pA` walks path `A + B`, `pB` walks path `B + A`.
Both paths have equal total length, so they synchronize at the intersection, or both become `null`.

---

## C - Concepts (Core Ideas)

### Method category

- **Two pointers on linked list**
- **Implicit length alignment** through "walk full A then B"
- **Identity equality without structural mutation**

### Why switch-head pointers must meet

Let:

- `a`: unique prefix length of A
- `b`: unique prefix length of B
- `c`: shared suffix length

Then:

- Length of A is `a + c`
- Length of B is `b + c`

Pointer travel:

- `pA` walks `a+c`, then `b`
- `pB` walks `b+c`, then `a`

Both walk `a+b+c` before entering the same alignment point.
So within `m+n` steps, they either meet at the intersection or both reach `null`.

---

## Practice Guide / Steps

1. Initialize `pA=headA`, `pB=headB`
2. Loop while `pA != pB`:
   - `pA = pA.next` else `headB`
   - `pB = pB.next` else `headA`
3. Return `pA` (intersection node or null)

Runnable Python example (`intersection.py`):

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def get_intersection_node(head_a: ListNode | None, head_b: ListNode | None) -> ListNode | None:
    p, q = head_a, head_b
    while p is not q:
        p = p.next if p else head_b
        q = q.next if q else head_a
    return p


if __name__ == "__main__":
    # Build shared tail: c1 -> c2 -> c3
    c1 = ListNode(8)
    c2 = ListNode(4)
    c3 = ListNode(5)
    c1.next = c2
    c2.next = c3

    # A: a1 -> a2 -> c1
    a1 = ListNode(4)
    a2 = ListNode(1)
    a1.next = a2
    a2.next = c1

    # B: b1 -> b2 -> b3 -> c1
    b1 = ListNode(5)
    b2 = ListNode(6)
    b3 = ListNode(1)
    b1.next = b2
    b2.next = b3
    b3.next = c1

    ans = get_intersection_node(a1, b1)
    print(ans.val if ans else None)  # 8
```

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Shared suffix dedup in versioned pipelines (Python)

**Background**: Some experiment/task pipelines are chain nodes, and multiple pipelines can share a common tail.
**Why it fits**: locating the intersection lets you execute shared tail once or cache it.

```python
class Step:
    def __init__(self, name):
        self.name = name
        self.next = None


def intersection(a, b):
    p, q = a, b
    while p is not q:
        p = p.next if p else b
        q = q.next if q else a
    return p


if __name__ == "__main__":
    common = Step("train")
    common.next = Step("evaluate")

    a = Step("clean")
    a.next = Step("fe")
    a.next.next = common

    b = Step("clean_v2")
    b.next = common

    hit = intersection(a, b)
    print(hit.name if hit else "none")  # train
```

### Scenario 2: Safety check to avoid double free (C)

**Background**: In C projects, accidentally shared list tails can cause double free if both lists are freed independently.
**Why it fits**: detect intersection first, then free shared suffix only once.

```c
struct Node { int v; struct Node* next; };

struct Node* intersection(struct Node* a, struct Node* b) {
    struct Node* p = a;
    struct Node* q = b;
    while (p != q) {
        p = p ? p->next : b;
        q = q ? q->next : a;
    }
    return p; // may be NULL
}
```

### Scenario 3: Merge point detection in frontend history branches (JavaScript)

**Background**: Some editors represent operation history as linked nodes; branches can share a common tail after merge/replay.
**Why it fits**: finding intersection gives "where shared history starts" for UI highlight and merge strategy.

```javascript
function intersection(headA, headB) {
  let p = headA;
  let q = headB;
  while (p !== q) {
    p = p ? p.next : headB;
    q = q ? q.next : headA;
  }
  return p;
}
```

---

## R - Reflection (Tradeoffs and Deepening)

### Complexity

- **Time**: O(m+n)
- **Space**: O(1)

### Alternatives

| Method | Idea | Time | Extra space | Note |
| --- | --- | --- | --- | --- |
| Hash set | Store nodes of A, scan B | O(m+n) | O(m) | Most direct |
| Length alignment | Align by length difference | O(m+n) | O(1) | Needs separate length pass |
| **Switch-head two pointers** | A->B and B->A traversal | **O(m+n)** | **O(1)** | Cleanest template |

### Common pitfalls

1. **Using value equality as intersection**: intersection requires node identity equality.
2. **Null handling mistakes**: loop should end by pointer identity; result can be null.
3. **Using template on cyclic lists without checks**: this problem guarantees acyclic lists; otherwise loop risk exists.

---

## FAQs and Notes

1. **Why no infinite loop?**
   Under no-cycle assumption, each pointer walks at most `m+n` steps before meeting at intersection or both becoming null.

2. **What if `headA == headB`?**
   They are already equal at start; return immediately.

3. **Can I mark nodes or modify values?**
   No. Problem requires preserving original structure, and structural mutation is unsafe in shared data.

---

## Best Practices

- Memorize the switch-head pattern: `p = p ? p->next : headB`, `q = q ? q->next : headA`
- For any "shared tail" question, first verify whether equality means identity or value
- If cycles are possible in production data, run cycle detection first

---

## S - Summary

### Key Takeaways

- Intersection in this problem means same node object, not same value
- Hash is easy but costs memory; length alignment is O(1) but needs explicit length pass
- **Switch-head two pointers** implicitly aligns path lengths, giving O(m+n) time and O(1) space without mutations
- The no-cycle guarantee is a core precondition for termination and correctness
- This template transfers to shared-tail / merge-point / common-suffix structures

### References and Further Reading

- LeetCode 160. Intersection of Two Linked Lists
- Classic linked-list pointer patterns: cycle detection, middle node, remove Nth from end
- Pointer identity concepts in shared mutable structures

---

## Meta

- **Reading time**: 10-14 min
- **Tags**: Hot100, linked list, two pointers, space optimization
- **SEO keywords**: Intersection of Two Linked Lists, switch heads, O(1) space, LeetCode 160
- **Meta description**: Two pointers walk A then B and B then A, meeting at intersection or null within m+n steps, with O(m+n) time and O(1) space.

---

## Call to Action

Use the same thinking on two follow-up problems:
1. Linked List Cycle (Floyd)
2. Remove Nth Node From End (fast/slow pointers)

If you want, I can also add an advanced follow-up post: how to reason about intersection when cycles may exist.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from __future__ import annotations


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: ListNode | None = None


def get_intersection_node(head_a: ListNode | None, head_b: ListNode | None) -> ListNode | None:
    p, q = head_a, head_b
    while p is not q:
        p = p.next if p else head_b
        q = q.next if q else head_a
    return p
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* getIntersectionNode(struct ListNode* headA, struct ListNode* headB) {
    struct ListNode* p = headA;
    struct ListNode* q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}

static struct ListNode* node(int v) {
    struct ListNode* n = (struct ListNode*)malloc(sizeof(struct ListNode));
    n->val = v;
    n->next = NULL;
    return n;
}

int main(void) {
    // shared: c1(8) -> c2(4) -> c3(5)
    struct ListNode* c1 = node(8);
    struct ListNode* c2 = node(4);
    struct ListNode* c3 = node(5);
    c1->next = c2;
    c2->next = c3;

    // A: 4 -> 1 -> c1
    struct ListNode* a1 = node(4);
    struct ListNode* a2 = node(1);
    a1->next = a2;
    a2->next = c1;

    // B: 5 -> 6 -> 1 -> c1
    struct ListNode* b1 = node(5);
    struct ListNode* b2 = node(6);
    struct ListNode* b3 = node(1);
    b1->next = b2;
    b2->next = b3;
    b3->next = c1;

    struct ListNode* ans = getIntersectionNode(a1, b1);
    if (ans) printf("%d\n", ans->val); else printf("null\n");

    // In real code, free nodes carefully: shared suffix should be freed once.
    return 0;
}
```

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    ListNode* p = headA;
    ListNode* q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}

int main() {
    // shared: c1 -> c2 -> c3
    auto* c1 = new ListNode(8);
    auto* c2 = new ListNode(4);
    auto* c3 = new ListNode(5);
    c1->next = c2;
    c2->next = c3;

    // A: 4 -> 1 -> c1
    auto* a1 = new ListNode(4);
    auto* a2 = new ListNode(1);
    a1->next = a2;
    a2->next = c1;

    // B: 5 -> 6 -> 1 -> c1
    auto* b1 = new ListNode(5);
    auto* b2 = new ListNode(6);
    auto* b3 = new ListNode(1);
    b1->next = b2;
    b2->next = b3;
    b3->next = c1;

    ListNode* ans = getIntersectionNode(a1, b1);
    std::cout << (ans ? std::to_string(ans->val) : std::string("null")) << "\n";

    // Demo only: free omitted.
    return 0;
}
```

```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
    p, q := headA, headB
    for p != q {
        if p == nil {
            p = headB
        } else {
            p = p.Next
        }
        if q == nil {
            q = headA
        } else {
            q = q.Next
        }
    }
    return p
}

func main() {
    // shared: c1(8) -> c2(4) -> c3(5)
    c3 := &ListNode{Val: 5}
    c2 := &ListNode{Val: 4, Next: c3}
    c1 := &ListNode{Val: 8, Next: c2}

    // A: 4 -> 1 -> c1
    a := &ListNode{Val: 4, Next: &ListNode{Val: 1, Next: c1}}

    // B: 5 -> 6 -> 1 -> c1
    b := &ListNode{Val: 5, Next: &ListNode{Val: 6, Next: &ListNode{Val: 1, Next: c1}}}

    ans := getIntersectionNode(a, b)
    if ans != nil {
        fmt.Println(ans.Val)
    } else {
        fmt.Println("null")
    }
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

fn node(val: i32) -> Rc<RefCell<ListNode>> {
    Rc::new(RefCell::new(ListNode { val, next: None }))
}

fn same(a: &Option<Rc<RefCell<ListNode>>>, b: &Option<Rc<RefCell<ListNode>>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn get_intersection_node(
    head_a: Option<Rc<RefCell<ListNode>>>,
    head_b: Option<Rc<RefCell<ListNode>>>,
) -> Option<Rc<RefCell<ListNode>>> {
    let mut p = head_a.clone();
    let mut q = head_b.clone();
    while !same(&p, &q) {
        p = if let Some(n) = p {
            n.borrow().next.clone()
        } else {
            head_b.clone()
        };
        q = if let Some(n) = q {
            n.borrow().next.clone()
        } else {
            head_a.clone()
        };
    }
    p
}

fn main() {
    // shared: c1(8) -> c2(4) -> c3(5)
    let c1 = node(8);
    let c2 = node(4);
    let c3 = node(5);
    c1.borrow_mut().next = Some(c2.clone());
    c2.borrow_mut().next = Some(c3.clone());

    // A: 4 -> 1 -> c1
    let a1 = node(4);
    let a2 = node(1);
    a1.borrow_mut().next = Some(a2.clone());
    a2.borrow_mut().next = Some(c1.clone());

    // B: 5 -> 6 -> 1 -> c1
    let b1 = node(5);
    let b2 = node(6);
    let b3 = node(1);
    b1.borrow_mut().next = Some(b2.clone());
    b2.borrow_mut().next = Some(b3.clone());
    b3.borrow_mut().next = Some(c1.clone());

    let ans = get_intersection_node(Some(a1), Some(b1));
    match ans {
        Some(n) => println!("{}", n.borrow().val),
        None => println!("null"),
    }
}
```

```javascript
class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

function getIntersectionNode(headA, headB) {
  let p = headA;
  let q = headB;
  while (p !== q) {
    p = p ? p.next : headB;
    q = q ? q.next : headA;
  }
  return p;
}

// demo
const c1 = new ListNode(8);
const c2 = new ListNode(4);
const c3 = new ListNode(5);
c1.next = c2;
c2.next = c3;

const a1 = new ListNode(4);
const a2 = new ListNode(1);
a1.next = a2;
a2.next = c1;

const b1 = new ListNode(5);
const b2 = new ListNode(6);
const b3 = new ListNode(1);
b1.next = b2;
b2.next = b3;
b3.next = c1;

const ans = getIntersectionNode(a1, b1);
console.log(ans ? ans.val : null);
```
