---
title: "Hot100: Merge Two Sorted Lists Sentinel Two-Pointer Merge ACERS Guide"
date: 2026-02-10T10:47:02+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "two pointers", "merge", "sentinel", "LeetCode 21"]
description: "Merge two sorted linked lists in O(m+n) with sentinel node and two pointers, with iterative and recursive comparison, engineering mapping, and runnable multi-language implementations."
keywords: ["Merge Two Sorted Lists", "sentinel node", "two pointers", "linked list merge", "LeetCode 21", "Hot100", "O(m+n)"]
---

> **Subtitle / Summary**  
> This problem is the linked-list version of merge-sort's merge step. Use a sentinel node plus two pointers to splice nodes in ascending order in O(m+n), without rebuilding the list.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `linked list`, `merge`, `two pointers`  
- **SEO keywords**: Merge Two Sorted Lists, sentinel node, linked list merge, LeetCode 21, Hot100  
- **Meta description**: A complete ACERS guide for LeetCode 21 with derivation, correctness invariants, pitfalls, and runnable multi-language code.

---

## Target Readers

- Hot100 learners preparing linked-list interview templates
- Developers who often lose nodes while rewiring `next`
- Engineers who need stable O(1)-extra-space merge patterns

## Background / Motivation

This is a small problem with large transfer value:

- It is a direct building block of `merge k sorted lists`
- It reinforces pointer safety under in-place rewiring
- It mirrors real-world merging of two already sorted streams

If this template is stable in your hands, many linked-list and divide-and-conquer problems become easier.

## Core Concepts

- **Sorted linked list**: non-decreasing values along `next`
- **Splicing merge**: reuse original nodes by rewiring pointers
- **Sentinel (dummy) node**: avoids special handling for the head of result
- **Tail pointer**: always points to the last node in merged list

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given heads `list1` and `list2` of two sorted linked lists, merge them into one sorted linked list and return its head.
The merged list should be formed by **splicing together** nodes from the original lists.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| list1 | ListNode | Head of sorted list 1 (nullable) |
| list2 | ListNode | Head of sorted list 2 (nullable) |
| return | ListNode | Head of merged sorted list |

### Example 1

```text
list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4
output: 1 -> 1 -> 2 -> 3 -> 4 -> 4
```

### Example 2

```text
list1: null
list2: 0 -> 5
output: 0 -> 5
```

---

## Thought Process: From Naive to Optimal

### Naive approach: flatten + sort + rebuild

1. Read values from both lists into array
2. Sort the array
3. Recreate a new linked list

Problems:

- O(m+n) extra space
- violates the spirit of "splice original nodes"

### Key observation

Both lists are already sorted.
At each step, the next smallest node must be one of the two current heads.

So we can:

- compare current nodes
- append smaller one to result tail
- move that list pointer forward

### Method choice

Use **sentinel + two pointers**:

- O(m+n) time
- O(1) extra space (excluding sentinel node)
- stable and interview-friendly

---

## C - Concepts (Core Ideas)

### Method Category

- Two-pointer merge
- In-place linked-list splicing
- Sentinel node pattern

### Loop Invariant

Before each iteration:

1. `dummy.next ... tail` is already sorted
2. `p1` and `p2` point to the first unmerged nodes in each list
3. All nodes before `p1` and `p2` have been merged exactly once

After appending the smaller head, invariants still hold.
When one list ends, append the rest of the other list directly.

### Why appending the remainder is safe

If `p1` is null, all remaining nodes in `p2` are already in sorted order and all are >= `tail.val`.
So attaching `tail.next = p2` preserves sorted order.

---

## Practice Guide / Steps

1. Create `dummy` and set `tail = dummy`
2. Initialize `p1 = list1`, `p2 = list2`
3. While both are non-null:
   - if `p1.val <= p2.val`, append `p1`
   - else append `p2`
   - move `tail`
4. Append the non-null remainder
5. Return `dummy.next`

Runnable Python example (`merge_two_lists.py`):

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    p1, p2 = list1, list2

    while p1 is not None and p2 is not None:
        if p1.val <= p2.val:
            nxt = p1.next
            tail.next = p1
            p1.next = None
            p1 = nxt
        else:
            nxt = p2.next
            tail.next = p2
            p2.next = None
            p2 = nxt
        tail = tail.next

    tail.next = p1 if p1 is not None else p2
    return dummy.next


def from_list(arr: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    l1 = from_list([1, 2, 4])
    l2 = from_list([1, 3, 4])
    print(to_list(merge_two_lists(l1, l2)))
    print(to_list(merge_two_lists(None, from_list([0, 5]))))
```

---

## Explanation / Why This Works

Each step picks the globally smallest remaining node between the two list heads.
That is exactly the merge-sort merge principle.

Because every node is moved once, and no node is revisited:

- time is linear in total node count
- space is constant (pointer variables + sentinel)

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Merge two ordered event streams (Go)

**Background**: backend services often merge logs/events by timestamp.
**Why it fits**: both inputs are already sorted; linear merge minimizes latency.

```go
package main

import "fmt"

type Node struct {
	Ts   int
	Next *Node
}

func merge(a, b *Node) *Node {
	dummy := &Node{}
	tail := dummy
	for a != nil && b != nil {
		if a.Ts <= b.Ts {
			tail.Next = a
			a = a.Next
		} else {
			tail.Next = b
			b = b.Next
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
	a := &Node{1, &Node{3, &Node{7, nil}}}
	b := &Node{2, &Node{4, &Node{8, nil}}}
	for p := merge(a, b); p != nil; p = p.Next {
		fmt.Print(p.Ts, " ")
	}
	fmt.Println()
}
```

### Scenario 2: Offline merge of sorted ID sets (Python)

**Background**: analytics jobs frequently merge sorted outputs from two rules.
**Why it fits**: pointer-based merge avoids expensive global sort for pre-sorted sources.

```python
def merge_sorted(a, b):
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

print(merge_sorted([1, 3, 7], [2, 4, 8]))
```

### Scenario 3: Frontend timeline composition from two sorted feeds (JavaScript)

**Background**: one feed from local cache and one from server response.
**Why it fits**: stable merge keeps timeline sorted with predictable complexity.

```javascript
function mergeSortedFeeds(a, b) {
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

console.log(mergeSortedFeeds([{ ts: 1 }, { ts: 5 }], [{ ts: 2 }, { ts: 4 }]));
```

---

## R - Reflection (Complexity, Alternatives, Tradeoffs)

### Complexity

- Time: `O(m+n)`
- Space: `O(1)` extra (iterative pointer solution)

### Alternatives

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Flatten + sort + rebuild | O((m+n)log(m+n)) | O(m+n) | easy but not in-place |
| Recursive merge | O(m+n) | O(m+n) stack worst-case | concise but stack risk |
| Sentinel iterative merge | O(m+n) | O(1) | most practical in interviews and systems code |

### Common mistakes

1. Forgetting to move `tail` after attachment
2. Losing list remainder by not attaching final non-null list
3. Mishandling `null` inputs
4. Creating new nodes unnecessarily

### Why this is the best practical method

It matches constraints and is robust:

- linear
- no extra container
- simple invariant-based correctness

---

## FAQ and Notes

1. **Do we need to allocate new nodes?**  
   No, splicing existing nodes is enough.

2. **Is recursion acceptable?**  
   Functionally yes, but iterative is safer for long lists.

3. **What if equal values appear?**  
   Use `<=` for stable preference from `list1`.

---

## Best Practices

- Always use `dummy + tail` for list-construction tasks
- Keep pointer updates in one consistent order
- Test edge cases: empty lists, one empty, all equal values
- Reuse this merge as a helper for `merge k lists` and list sorting

---

## S - Summary

- Merge Two Sorted Lists is exactly merge-sort's merge step on linked lists
- Sentinel node removes head special-case complexity
- Two-pointer iterative merge is O(m+n) time and O(1) extra space
- This template is foundational for many advanced linked-list problems

### Further Reading

- LeetCode 21. Merge Two Sorted Lists
- LeetCode 23. Merge k Sorted Lists
- LeetCode 148. Sort List
- LeetCode 206. Reverse Linked List

---

## Conclusion

If you can write this merge in one pass without pointer mistakes, your linked-list fundamentals are in good shape.
Practice this with `merge k lists` next to make the pattern production-ready.

---

## References

- https://leetcode.com/problems/merge-two-sorted-lists/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://docs.python.org/3/tutorial/datastructures.html
- https://doc.rust-lang.org/std/option/

---

## Meta Info

- **Reading time**: 10-12 min
- **Tags**: Hot100, linked list, merge, two pointers
- **SEO keywords**: Merge Two Sorted Lists, LeetCode 21, sentinel node, O(m+n)
- **Meta description**: A practical ACERS guide to sentinel-node linked-list merge with multi-language runnable code.

---

## CTA

Try implementing this in under 10 minutes in your primary language.
Then extend it to `merge k sorted lists` to lock in the divide-and-conquer upgrade path.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(list1, list2):
    dummy = ListNode(0)
    tail = dummy
    p1, p2 = list1, list2

    while p1 and p2:
        if p1.val <= p2.val:
            nxt = p1.next
            tail.next = p1
            p1.next = None
            p1 = nxt
        else:
            nxt = p2.next
            tail.next = p2
            p2.next = None
            p2 = nxt
        tail = tail.next

    tail.next = p1 if p1 else p2
    return dummy.next
```

```c
#include <stddef.h>

typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode dummy;
    dummy.next = NULL;
    ListNode* tail = &dummy;

    while (list1 && list2) {
        if (list1->val <= list2->val) {
            ListNode* nxt = list1->next;
            tail->next = list1;
            list1->next = NULL;
            list1 = nxt;
        } else {
            ListNode* nxt = list2->next;
            tail->next = list2;
            list2->next = NULL;
            list2 = nxt;
        }
        tail = tail->next;
    }
    tail->next = list1 ? list1 : list2;
    return dummy.next;
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy;
        ListNode* tail = &dummy;
        while (list1 && list2) {
            if (list1->val <= list2->val) {
                ListNode* nxt = list1->next;
                tail->next = list1;
                list1->next = nullptr;
                list1 = nxt;
            } else {
                ListNode* nxt = list2->next;
                tail->next = list2;
                list2->next = nullptr;
                list2 = nxt;
            }
            tail = tail->next;
        }
        tail->next = list1 ? list1 : list2;
        return dummy.next;
    }
};
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for list1 != nil && list2 != nil {
		if list1.Val <= list2.Val {
			nxt := list1.Next
			tail.Next = list1
			list1.Next = nil
			list1 = nxt
		} else {
			nxt := list2.Next
			tail.Next = list2
			list2.Next = nil
			list2 = nxt
		}
		tail = tail.Next
	}
	if list1 != nil {
		tail.Next = list1
	} else {
		tail.Next = list2
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
        ListNode { val, next: None }
    }
}

pub fn merge_two_lists(
    mut list1: Option<Box<ListNode>>,
    mut list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut dummy = Box::new(ListNode::new(0));
    let mut tail = &mut dummy;

    while list1.is_some() && list2.is_some() {
        let take_left = list1.as_ref().unwrap().val <= list2.as_ref().unwrap().val;
        if take_left {
            let mut node = list1.take().unwrap();
            list1 = node.next.take();
            tail.next = Some(node);
        } else {
            let mut node = list2.take().unwrap();
            list2 = node.next.take();
            tail.next = Some(node);
        }
        tail = tail.next.as_mut().unwrap();
    }

    tail.next = if list1.is_some() { list1 } else { list2 };
    dummy.next
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function mergeTwoLists(list1, list2) {
  const dummy = new ListNode(0);
  let tail = dummy;
  let p1 = list1;
  let p2 = list2;

  while (p1 && p2) {
    if (p1.val <= p2.val) {
      const nxt = p1.next;
      tail.next = p1;
      p1.next = null;
      p1 = nxt;
    } else {
      const nxt = p2.next;
      tail.next = p2;
      p2.next = null;
      p2 = nxt;
    }
    tail = tail.next;
  }

  tail.next = p1 || p2;
  return dummy.next;
}
```
