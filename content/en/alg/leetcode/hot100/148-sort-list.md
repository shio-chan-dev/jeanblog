---
title: "Hot100: Sort List Linked-List Merge Sort ACERS Guide"
date: 2026-02-10T17:07:38+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "merge sort", "fast slow pointers", "divide and conquer", "LeetCode 148"]
description: "Sort a singly linked list in O(n log n) time using split + recursive merge sort. This guide explains derivation, invariants, engineering mappings, and runnable multi-language implementations."
keywords: ["Sort List", "linked list merge sort", "divide and conquer", "LeetCode 148", "Hot100", "O(n log n)"]
---

> **Subtitle / Summary**  
> LeetCode 148 is not about whether you can sort; it is about choosing the right sorting strategy for linked-list constraints. For singly linked lists, merge sort fits naturally: split by middle, sort recursively, merge linearly.

- **Reading time**: 12-16 min  
- **Tags**: `Hot100`, `linked list`, `merge sort`, `divide and conquer`  
- **SEO keywords**: Sort List, linked list merge sort, LeetCode 148, Hot100  
- **Meta description**: A practical ACERS guide for LeetCode 148 with derivation, complexity analysis, engineering mappings, and runnable code in multiple languages.

---

## Target Readers

- Hot100 learners building reusable linked-list templates
- Developers who struggle with split-and-reconnect pointer safety
- Engineers who want a clear answer to "why merge sort for linked lists"

## Background / Motivation

Sorting linked structures appears in real systems:

- post-processing chained tasks by priority
- offline reordering of append-only linked logs
- memory-conscious restructuring with minimal copying

If you directly copy array-sorting intuition to linked lists, you usually hit:

- no O(1) random access
- expensive and error-prone pointer shuffling for quicksort-style partitioning

So this problem is fundamentally about **algorithm-data-structure fit**.

## Core Concepts

- **Divide and Conquer**: split list to subproblems, then merge upward
- **Fast/slow middle finding**: `slow` moves 1 step, `fast` moves 2
- **Linked-list merge**: linear splice of two sorted sublists
- **Stable sorting**: equal keys keep relative order

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a linked list `head`, sort it in ascending order and return the sorted list.
Required time complexity: `O(n log n)`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of a singly linked list (nullable) |
| return | ListNode | head of sorted list |

### Example 1

```text
input: 4 -> 2 -> 1 -> 3
output: 1 -> 2 -> 3 -> 4
```

### Example 2

```text
input: -1 -> 5 -> 3 -> 4 -> 0
output: -1 -> 0 -> 3 -> 4 -> 5
```

---

## Thought Process: From Naive to Optimal

### Naive approach: copy to array and sort

- Read values into array
- Use built-in sort
- Rebuild list

Tradeoff:

- O(n) extra memory
- misses the core linked-list manipulation skill target

### Key observation

Linked lists are good at:

- cutting (`next = null`)
- linear traversal
- splicing (`next` rewiring)

This exactly matches merge sort:

1. split around middle
2. sort each half
3. merge two sorted halves in linear time

### Method selection

Use top-down merge sort on list:

- Time: `O(n log n)`
- Extra: recursion stack `O(log n)`
- clean, stable, and interview-practical

---

## C - Concepts (Core Ideas)

### Method Category

- linked-list divide-and-conquer sorting
- fast/slow split
- merge-template reuse (same pattern as LeetCode 21)

### Correctness intuition

1. Base case: empty or single-node list is already sorted
2. Induction: recursively sorted left/right halves are each sorted
3. Merge: linear merge of two sorted lists remains sorted

Therefore, the final result is sorted.

### Complexity recurrence

`T(n) = 2T(n/2) + O(n)`

By Master theorem:

- `T(n) = O(n log n)`

---

## Practice Guide / Steps

1. Return directly for `0/1` node list
2. Find middle with fast/slow pointers and cut list into two halves
3. Recursively sort left and right halves
4. Merge two sorted halves with sentinel-node merge
5. Return merged head

Runnable Python example (`sort_list.py`):

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    if head is None or head.next is None:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)
    return merge(left, right)


def merge(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next
```

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Task-chain reordering by priority (Go)

**Background**: tasks are chained in insertion order but must run by priority.  
**Why it fits**: linked-list split+merge avoids repeated array conversion.

```go
type Task struct {
	Priority int
	Next     *Task
}

func merge(a, b *Task) *Task {
	d := &Task{}
	t := d
	for a != nil && b != nil {
		if a.Priority <= b.Priority {
			t.Next, a = a, a.Next
		} else {
			t.Next, b = b, b.Next
		}
		t = t.Next
	}
	if a != nil { t.Next = a } else { t.Next = b }
	return d.Next
}
```

### Scenario 2: Offline chained log normalization (Python)

**Background**: append-order logs need timestamp-order output for auditing.  
**Why it fits**: merge-friendly linear passes scale predictably.

```python
def merge_sorted_logs(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
```

### Scenario 3: Frontend incremental feed merge (JavaScript)

**Background**: cached and remote pages are already sorted and need merged rendering.  
**Why it fits**: merge gives deterministic linear behavior and stable ordering.

```javascript
function mergeByScore(a, b) {
  let i = 0, j = 0;
  const out = [];
  while (i < a.length && j < b.length) {
    if (a[i].score <= b[j].score) out.push(a[i++]);
    else out.push(b[j++]);
  }
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}
```

---

## R - Reflection (Complexity, Alternatives, Tradeoffs)

### Complexity

- Time: `O(n log n)`
- Extra space: `O(log n)` recursion stack

### Alternatives

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| Array copy + sort | O(n log n) | O(n) | easy but loses list advantages |
| List quicksort | avg O(n log n), worst O(n²) | O(log n) | partitioning is awkward on list |
| List merge sort (this) | O(n log n) | O(log n) | stable and structure-friendly |

### Common mistakes

1. Forgetting `slow.next = None`, causing infinite recursion
2. Wrong fast/slow initialization for even lengths
3. Missing tail attachment in merge
4. Pointer loss during recursive split

### Why this is the practical default

It matches linked-list properties directly:

- no random access dependency
- linear work per recursion level
- reusable merge logic across multiple linked-list problems

---

## FAQ and Notes

1. **Can this be strict O(1) extra space?**  
   Top-down recursion uses O(log n) stack. Strict O(1) needs bottom-up iterative merge sort.

2. **Why not quicksort?**  
   Linked-list partitioning is less natural and worst-case risk is higher.

3. **Does stability matter?**  
   Yes when equal keys carry extra business metadata.

---

## Best Practices

- Treat split+merge as a reusable helper template
- Test `null`, one node, even/odd lengths, and duplicate values
- Prioritize pointer correctness before micro-optimization
- Learn bottom-up merge sort after mastering this version

---

## S - Summary

- Merge sort is the best default for linked-list sorting
- Core workflow is split -> sort halves -> merge
- Complexity satisfies the target: `O(n log n)`
- This template generalizes to many list-based merge tasks

### Further Reading

- LeetCode 21. Merge Two Sorted Lists
- LeetCode 23. Merge k Sorted Lists
- LeetCode 147. Insertion Sort List
- LeetCode 25. Reverse Nodes in k-Group

---

## References

- https://leetcode.com/problems/sort-list/
- https://en.cppreference.com/w/cpp/algorithm/stable_sort
- https://docs.python.org/3/howto/sorting.html
- https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html

---

## Meta Info

- **Reading time**: 12-16 min
- **Tags**: Hot100, linked list, merge sort, divide and conquer
- **SEO keywords**: Sort List, linked list merge sort, LeetCode 148, Hot100
- **Meta description**: A complete linked-list merge-sort guide for LeetCode 148 with derivation, complexity, and multi-language code.

---

## CTA

Two practical next steps:

1. Re-implement recursive list merge sort from scratch without notes
2. Build the bottom-up iterative variant and compare space tradeoffs

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sortList(head):
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sortList(head)
    right = sortList(mid)
    return merge(left, right)


def merge(a, b):
    dummy = ListNode()
    t = dummy
    while a and b:
        if a.val <= b.val:
            t.next, a = a, a.next
        else:
            t.next, b = b, b.next
        t = t.next
    t.next = a if a else b
    return dummy.next
```

```c
typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

static ListNode* merge(ListNode* a, ListNode* b) {
    ListNode dummy = {0, NULL};
    ListNode* t = &dummy;
    while (a && b) {
        if (a->val <= b->val) {
            t->next = a; a = a->next;
        } else {
            t->next = b; b = b->next;
        }
        t = t->next;
    }
    t->next = a ? a : b;
    return dummy.next;
}

ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* slow = head;
    ListNode* fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode* mid = slow->next;
    slow->next = NULL;
    return merge(sortList(head), sortList(mid));
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x=0, ListNode* n=nullptr): val(x), next(n) {}
};

class Solution {
    ListNode* merge(ListNode* a, ListNode* b) {
        ListNode dummy;
        ListNode* t = &dummy;
        while (a && b) {
            if (a->val <= b->val) t->next = a, a = a->next;
            else t->next = b, b = b->next;
            t = t->next;
        }
        t->next = a ? a : b;
        return dummy.next;
    }
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *slow = head, *fast = head->next;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* mid = slow->next;
        slow->next = nullptr;
        return merge(sortList(head), sortList(mid));
    }
};
```

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	mid := slow.Next
	slow.Next = nil
	left := sortList(head)
	right := sortList(mid)
	return merge(left, right)
}

func merge(a, b *ListNode) *ListNode {
	dummy := &ListNode{}
	t := dummy
	for a != nil && b != nil {
		if a.Val <= b.Val {
			t.Next = a
			a = a.Next
		} else {
			t.Next = b
			b = b.Next
		}
		t = t.Next
	}
	if a != nil { t.Next = a } else { t.Next = b }
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
    fn new(val: i32) -> Self {
        Self { val, next: None }
    }
}

pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut vals = Vec::new();
    let mut cur = head;
    let mut p = cur;
    while let Some(mut node) = p {
        vals.push(node.val);
        p = node.next.take();
    }
    vals.sort_unstable();
    let mut ans = None;
    for v in vals.into_iter().rev() {
        let mut node = Box::new(ListNode::new(v));
        node.next = ans;
        ans = Some(node);
    }
    ans
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function sortList(head) {
  if (!head || !head.next) return head;

  let slow = head;
  let fast = head.next;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }

  const mid = slow.next;
  slow.next = null;

  return merge(sortList(head), sortList(mid));
}

function merge(a, b) {
  const dummy = new ListNode(0);
  let t = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      t.next = a;
      a = a.next;
    } else {
      t.next = b;
      b = b.next;
    }
    t = t.next;
  }
  t.next = a || b;
  return dummy.next;
}
```
