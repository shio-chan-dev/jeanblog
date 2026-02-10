---
title: "Hot100: Merge K Sorted Lists Divide-and-Conquer O(N log k) ACERS Guide"
date: 2026-02-10T17:05:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "divide and conquer", "merge", "priority queue", "LeetCode 23"]
description: "Upgrade from LeetCode 21 to k-way merge: derive why sequential merge degrades, then use divide-and-conquer to reach O(N log k), with heap comparison and runnable multi-language implementations."
keywords: ["Merge K Sorted Lists", "divide and conquer", "k-way merge", "LeetCode 23", "Hot100", "O(N log k)"]
---

> **Subtitle / Summary**  
> LeetCode 23 is a k-way merge problem, not just repeating LeetCode 21 in a loop. This ACERS guide derives the optimal structure, explains tradeoffs between divide-and-conquer and min-heap, and provides runnable implementations in multiple languages.

- **Reading time**: 12-16 min  
- **Tags**: `Hot100`, `linked list`, `divide and conquer`, `merge`  
- **SEO keywords**: Merge K Sorted Lists, LeetCode 23, divide and conquer, O(N log k), Hot100  
- **Meta description**: A full ACERS explanation of Merge K Sorted Lists from naive ideas to O(N log k) divide-and-conquer, with engineering mapping and multi-language code.

---

## Target Readers

- Hot100 learners who have finished LeetCode 21 and want the next-level merge template
- Developers who need predictable performance for k-way ordered data merge
- Engineers preparing for linked-list and divide-and-conquer interview rounds

## Background / Motivation

This problem appears in many production forms:

- merge sorted outputs from multiple shards
- combine sorted event streams from multiple services
- aggregate sorted pagination slices

The hard part is not correctness alone; it is controlling cost when `k` grows.

## Core Concepts

- **N**: total number of nodes across all lists
- **k**: number of lists
- **Sequential merge**: merge one list into the current result repeatedly
- **Divide-and-conquer merge**: merge in balanced rounds
- **Min-heap k-way merge**: keep current head from each list in a heap

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given an array `lists` of `k` sorted linked lists, merge them into one sorted linked list and return it.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| lists | ListNode[] | k sorted lists, each can be null |
| return | ListNode | merged sorted linked list head |

### Example 1

```text
input:  lists = [[1,4,5],[1,3,4],[2,6]]
output: [1,1,2,3,4,4,5,6]
```

### Example 2

```text
input:  lists = []
output: []
```

---

## C - Concepts (Core Ideas)

### Thought Process: Naive -> Bottleneck -> Better Structure

1. **Flatten + sort**
   - O(N log N), O(N) extra space
   - ignores that each list is already sorted

2. **Sequential merge**
   - `(((l1 merge l2) merge l3) ...)`
   - repeated scanning of early nodes can degrade toward O(Nk)

3. **Key observation**
   - Balanced pairwise merges form a merge tree
   - each level processes about N nodes
   - number of levels is about log k

4. **Method choice**
   - Divide-and-conquer merge
   - Time: O(N log k)
   - Extra space: O(log k) recursion stack

### Method Category

- Divide and conquer
- Linked-list in-place splicing
- Same complexity class as heap-based k-way merge

### Correctness Invariant

For interval `[l, r]`:

- `solve(l, r)` returns a fully sorted merge of all lists in that interval
- if left and right halves are correctly merged, `mergeTwo(left, right)` preserves sorted order and includes every node exactly once

---

## Practice Guide / Steps

1. Reuse a stable `mergeTwo` helper (LeetCode 21 template)
2. Build recursive `solve(l, r)`:
   - base: `l == r` return `lists[l]`
   - split at `mid`, solve both halves
3. Merge results from both halves
4. Start from full range `[0, k-1]`

Runnable Python example (`merge_k_lists.py`):

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(l: int, r: int) -> Optional[ListNode]:
        if l == r:
            return lists[l]
        m = (l + r) // 2
        return merge_two(solve(l, m), solve(m + 1, r))

    return solve(0, len(lists) - 1)
```

---

## Explanation / Why This Works

Balanced merging avoids repeatedly merging a huge partial result with small lists.

- work per level: about N node operations
- number of levels: about log k

So total time is O(N log k), while preserving in-place node reuse.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Merge sorted shard timelines (Go)

**Background**: each shard emits sorted events by timestamp.  
**Why it fits**: divide-and-conquer merge is easy to parallelize by levels.

```go
package main

import "fmt"

type Node struct {
	Ts   int
	Next *Node
}

func mergeTwo(a, b *Node) *Node {
	dummy := &Node{}
	tail := dummy
	for a != nil && b != nil {
		if a.Ts <= b.Ts {
			nxt := a.Next
			tail.Next = a
			a.Next = nil
			a = nxt
		} else {
			nxt := b.Next
			tail.Next = b
			b.Next = nil
			b = nxt
		}
		tail = tail.Next
	}
	if a != nil {
		tail.Next = a
	} else {
		tail.Next = b
	}
	return dummy.Next
}

func main() {
	a := &Node{1, &Node{4, &Node{9, nil}}}
	b := &Node{2, &Node{5, nil}}
	for p := mergeTwo(a, b); p != nil; p = p.Next {
		fmt.Print(p.Ts, " ")
	}
	fmt.Println()
}
```

### Scenario 2: Offline merge of sorted rule outputs (Python)

**Background**: multiple ranking pipelines output sorted IDs.
**Why it fits**: divide-and-conquer gives stable performance for large k.

```python
def merge_two(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_k(arrays):
    if not arrays:
        return []
    cur = arrays
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                nxt.append(merge_two(cur[i], cur[i + 1]))
            else:
                nxt.append(cur[i])
        cur = nxt
    return cur[0]
```

### Scenario 3: Frontend unified feed from multiple sorted sources (JavaScript)

**Background**: web app receives sorted cards from multiple APIs.
**Why it fits**: deterministic merge order with no global re-sort.

```javascript
function mergeTwo(a, b) {
  let i = 0;
  let j = 0;
  const out = [];
  while (i < a.length && j < b.length) {
    if (a[i].ts <= b[j].ts) out.push(a[i++]);
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

- Time: `O(N log k)`
- Space: `O(log k)` recursion stack

### Alternative Methods

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| Flatten + sort | O(N log N) | O(N) | easiest, wastes structure |
| Sequential merge | near O(Nk) worst | O(1) | degrades as k grows |
| Min-heap | O(N log k) | O(k) | great for streaming inputs |
| Divide-and-conquer | O(N log k) | O(log k) | clean and reusable |

### Common Mistakes

1. Forgetting empty input handling (`lists=[]`)
2. Assuming sequential merge is always efficient
3. Losing nodes in `mergeTwo` pointer rewiring
4. Incorrect base case in recursion

### Why this method is practical

It balances performance and implementation simplicity.
You can directly reuse LeetCode 21 helper logic and scale it to k-way merge.

---

## FAQ and Notes

1. **Divide-and-conquer or heap?**  
   Batch merge: divide-and-conquer is often cleaner. Streaming merge: heap is often better.

2. **Can this be fully in-place?**  
   Yes, by rewiring `next` pointers (except helper/sentinel nodes).

3. **What if values repeat?**  
   No issue. Use `<=` to keep stable tie behavior.

---

## Best Practices

- Treat `mergeTwo` as a shared utility function
- Never use sequential k-way merge for large k
- Validate with edge cases: empty lists, many null lists, uneven lengths
- Track `k` and `N` in production metrics for strategy switching

---

## S - Summary

- LeetCode 23 is a k-way merge scaling problem
- Divide-and-conquer reduces complexity to O(N log k)
- Heap and divide-and-conquer are both optimal in asymptotic time
- Mastering this pattern helps with many merge-based interview and production tasks

### Further Reading

- LeetCode 21. Merge Two Sorted Lists
- LeetCode 23. Merge K Sorted Lists
- LeetCode 148. Sort List
- LeetCode 632. Smallest Range Covering Elements from K Lists

---

## Conclusion

The key upgrade from LeetCode 21 to 23 is structural thinking:  
move from sequential accumulation to balanced reduction.
This is a general engineering pattern for multi-source ordered data.

---

## References

- https://leetcode.com/problems/merge-k-sorted-lists/
- https://docs.python.org/3/library/heapq.html
- https://en.cppreference.com/w/cpp/container/priority_queue
- https://pkg.go.dev/container/heap

---

## Meta Info

- **Reading time**: 12-16 min
- **Tags**: Hot100, linked list, divide and conquer, merge
- **SEO keywords**: Merge K Sorted Lists, LeetCode 23, O(N log k), Hot100
- **Meta description**: End-to-end ACERS guide for LeetCode 23 with complexity derivation and multi-language implementations.

---

## CTA

Implement `mergeTwo` + `mergeK` in your strongest language first, then re-implement in a second language.
That cross-language pass is the fastest way to internalize pointer invariants.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(l: int, r: int) -> Optional[ListNode]:
        if l == r:
            return lists[l]
        m = (l + r) // 2
        return merge_two(solve(l, m), solve(m + 1, r))

    return solve(0, len(lists) - 1)
```

```c
#include <stddef.h>

typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

ListNode* mergeTwo(ListNode* a, ListNode* b) {
    ListNode dummy;
    dummy.next = NULL;
    ListNode* tail = &dummy;

    while (a && b) {
        if (a->val <= b->val) {
            ListNode* nxt = a->next;
            tail->next = a;
            a->next = NULL;
            a = nxt;
        } else {
            ListNode* nxt = b->next;
            tail->next = b;
            b->next = NULL;
            b = nxt;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

ListNode* solve(ListNode** lists, int l, int r) {
    if (l > r) return NULL;
    if (l == r) return lists[l];
    int m = l + (r - l) / 2;
    ListNode* left = solve(lists, l, m);
    ListNode* right = solve(lists, m + 1, r);
    return mergeTwo(left, right);
}

ListNode* mergeKLists(ListNode** lists, int listsSize) {
    if (listsSize == 0) return NULL;
    return solve(lists, 0, listsSize - 1);
}
```

```cpp
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};

class Solution {
    ListNode* mergeTwo(ListNode* a, ListNode* b) {
        ListNode dummy;
        ListNode* tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) {
                ListNode* nxt = a->next;
                tail->next = a;
                a->next = nullptr;
                a = nxt;
            } else {
                ListNode* nxt = b->next;
                tail->next = b;
                b->next = nullptr;
                b = nxt;
            }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }

    ListNode* solve(vector<ListNode*>& lists, int l, int r) {
        if (l > r) return nullptr;
        if (l == r) return lists[l];
        int m = l + (r - l) / 2;
        return mergeTwo(solve(lists, l, m), solve(lists, m + 1, r));
    }

public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        return solve(lists, 0, (int)lists.size() - 1);
    }
};
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwo(a, b *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for a != nil && b != nil {
		if a.Val <= b.Val {
			nxt := a.Next
			tail.Next = a
			a.Next = nil
			a = nxt
		} else {
			nxt := b.Next
			tail.Next = b
			b.Next = nil
			b = nxt
		}
		tail = tail.Next
	}
	if a != nil {
		tail.Next = a
	} else {
		tail.Next = b
	}
	return dummy.Next
}

func solve(lists []*ListNode, l, r int) *ListNode {
	if l > r {
		return nil
	}
	if l == r {
		return lists[l]
	}
	m := l + (r-l)/2
	left := solve(lists, l, m)
	right := solve(lists, m+1, r)
	return mergeTwo(left, right)
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	return solve(lists, 0, len(lists)-1)
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
        ListNode { val, next: None }
    }
}

fn merge_two(a: Option<Box<ListNode>>, b: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    match (a, b) {
        (None, x) => x,
        (x, None) => x,
        (Some(mut na), Some(mut nb)) => {
            if na.val <= nb.val {
                let next = na.next.take();
                na.next = merge_two(next, Some(nb));
                Some(na)
            } else {
                let next = nb.next.take();
                nb.next = merge_two(Some(na), next);
                Some(nb)
            }
        }
    }
}

fn solve(lists: &mut [Option<Box<ListNode>>], l: usize, r: usize) -> Option<Box<ListNode>> {
    if l == r {
        return lists[l].take();
    }
    let m = (l + r) / 2;
    let left = solve(lists, l, m);
    let right = solve(lists, m + 1, r);
    merge_two(left, right)
}

pub fn merge_k_lists(mut lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    if lists.is_empty() {
        return None;
    }
    let n = lists.len();
    solve(&mut lists, 0, n - 1)
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function mergeTwo(a, b) {
  const dummy = new ListNode(0);
  let tail = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      const nxt = a.next;
      tail.next = a;
      a.next = null;
      a = nxt;
    } else {
      const nxt = b.next;
      tail.next = b;
      b.next = null;
      b = nxt;
    }
    tail = tail.next;
  }
  tail.next = a || b;
  return dummy.next;
}

function solve(lists, l, r) {
  if (l > r) return null;
  if (l === r) return lists[l];
  const m = (l + r) >> 1;
  return mergeTwo(solve(lists, l, m), solve(lists, m + 1, r));
}

function mergeKLists(lists) {
  if (!lists || lists.length === 0) return null;
  return solve(lists, 0, lists.length - 1);
}
```
