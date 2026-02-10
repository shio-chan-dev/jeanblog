---
title: "Hot100: Palindrome Linked List Fast/Slow + Reverse Second Half O(1) Space ACERS Guide"
date: 2026-02-09T17:38:32+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "fast slow pointers", "reverse list", "palindrome", "LeetCode 234"]
description: "Check whether a singly linked list is a palindrome in O(n) time and O(1) extra space: find middle with fast/slow pointers, reverse second half, compare, then restore list structure."
keywords: ["Palindrome Linked List", "fast slow pointers", "reverse second half", "O(1) space", "restore list", "LeetCode 234", "Hot100"]
---

> **Subtitle / Summary**  
> The core of palindrome validation is symmetric comparison, but a singly linked list cannot move backward. The most stable engineering template is: **find middle -> reverse second half in-place -> compare -> reverse back to restore**.

- **Reading time**: 10-14 min  
- **Tags**: `Hot100`, `linked list`, `fast slow pointers`, `in-place reverse`  
- **SEO keywords**: Palindrome Linked List, fast slow pointers, reverse second half, O(1) space, LeetCode 234  
- **Meta description**: O(n)/O(1) palindrome check for singly linked list with middle detection, second-half reversal, comparison, and full structure restoration.

---

## Target Readers

- Hot100 learners who want to master the "middle + reverse" linked-list combo
- Developers who frequently solve palindrome/symmetry interview questions
- Engineers who care about low extra memory and non-destructive checks

## Background / Motivation

For arrays, palindrome check is easy with two pointers from both ends.
For singly linked lists, you can only move forward via `next`, so symmetric comparison is not direct.

Real engineering constraints are often similar to this problem:

- avoid O(n) extra containers if possible
- do not permanently mutate the structure
- keep linear time

So we need a template that is:

- `O(n)` time
- `O(1)` extra space
- restorable (no side effects after checking)

## Core Concepts

| Concept | Meaning | Purpose |
| --- | --- | --- |
| Palindrome | same sequence forward and backward | needs symmetric comparison |
| Fast/slow pointers | fast moves 2, slow moves 1 | find middle in O(n) |
| In-place reverse | reverse pointer direction on second half | make backward side comparable forward |
| Restore step | reverse second half again and reconnect | preserve original structure |

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a singly linked list `head`, return `true` if it is a palindrome; otherwise return `false`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list |
| return | bool | whether list is palindrome |

### Example 1

```text
input:  1 -> 2 -> 2 -> 1
output: true
```

### Example 2

```text
input:  1 -> 2
output: false
```

---

## Thought Process: From Array Copy to In-Place Reversal

### Naive approach: copy to array

1. Traverse list and copy values to an array
2. Use two pointers on array to check palindrome

Pros: simple and robust.
Cons: needs `O(n)` extra memory.

### Better observation

If we can reverse only the second half of the list, then both halves become forward-comparable.

Example:

```text
1 -> 2 -> 3 -> 2 -> 1
```

Reverse second half around middle:

```text
left side forward:  1 -> 2 -> 3
right side forward: 1 -> 2
```

Now compare node by node.

### Final method

1. Find end of first half (fast/slow)
2. Reverse second half
3. Compare first half and reversed second half
4. Reverse back and reconnect (restore)

---

## C - Concepts (Core Ideas)

### Method Category

- Fast/slow pointer middle finding
- In-place linked-list reversal
- Temporary mutation + restoration

### Stable handling for odd/even lengths

A robust implementation uses `end_of_first_half`:

- odd length: first-half end is exact middle (middle value can be skipped in comparison)
- even length: first-half end is left-middle

Then reverse `first_half_end.next`, and compare only while second-half pointer is not null.
This removes many odd/even branch bugs.

### Key invariant

After reversing second half:

- `p1` starts from `head`
- `p2` starts from `reversed_second_half_head`

For palindrome lists, `p1.val == p2.val` for all nodes in second half.

After check:

- reverse second half again
- reconnect via `first_half_end.next`

So external observers see the original structure.

---

## Practical Guide / Steps

1. Return `true` for empty list or single node
2. Find `first_half_end` by fast/slow pointers
3. Reverse `first_half_end.next` to get `second_half_start`
4. Compare `head` and `second_half_start` node values
5. Restore: `first_half_end.next = reverse(second_half_start)`
6. Return comparison result

Runnable Python example (`palindrome_list.py`):

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def end_of_first_half(head: ListNode) -> ListNode:
    fast = head
    slow = head
    while fast.next and fast.next.next:
        fast = fast.next.next
        slow = slow.next  # type: ignore[assignment]
    return slow


def is_palindrome(head: ListNode | None) -> bool:
    if head is None or head.next is None:
        return True

    first_half_end = end_of_first_half(head)
    second_half_start = reverse_list(first_half_end.next)

    p1 = head
    p2 = second_half_start
    ok = True
    while ok and p2 is not None:
        if p1.val != p2.val:
            ok = False
        p1 = p1.next  # type: ignore[assignment]
        p2 = p2.next

    first_half_end.next = reverse_list(second_half_start)  # restore
    return ok
```

---

## Explanation / Why This Works

A singly linked list cannot directly read from tail to head.
Reversing the second half transforms the "backward side" into a forward list.
So palindrome checking becomes simple forward pair comparison.

The important engineering detail is restoration:

- temporary mutation is acceptable
- permanent mutation is usually not

By reversing the second half again, we restore exact original topology.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: symmetric event-chain validation (Python)

**Background**: a rule engine stores a session as a linked event chain and needs to detect mirrored behavior patterns.  
**Why it fits**: O(1) extra memory check without allocating a full copy.

```python
def is_symmetric_chain(head):
    return is_palindrome(head)
```

### Scenario 2: embedded frame-sequence symmetry check (C)

**Background**: in memory-limited systems, sampled frames may be chained via `next` pointers.  
**Why it fits**: avoids O(n) buffer allocation and preserves structure after check.

```c
struct ListNode { int val; struct ListNode* next; };

static struct ListNode* reverse(struct ListNode* head) {
    struct ListNode* prev = 0;
    struct ListNode* cur = head;
    while (cur) {
        struct ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}
```

### Scenario 3: browser operation-history mirror detection (JavaScript)

**Background**: editor actions are represented as a linked list in a demo tool.  
**Why it fits**: pointer-based structure allows direct reuse of middle+reverse template.

```javascript
function reverse(head) {
  let prev = null;
  let cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}
```

---

## R - Reflection

### Complexity

- Time: `O(n)` (middle find + reverse + compare + restore are all linear)
- Space: `O(1)`

### Alternatives and Tradeoffs

| Method | Time | Extra Space | Tradeoff |
| --- | --- | --- | --- |
| Copy to array | O(n) | O(n) | easy but memory-heavy |
| Stack half values | O(n) | O(n) | less copy than full array, still linear extra space |
| Recursion compare | O(n) | O(n) | stack risk on long lists |
| Reverse second half (current) | O(n) | O(1) | most practical, but needs careful restore |

### Common Mistakes

1. Forget restore step, leaving list mutated
2. Middle handling bug for odd/even length
3. Wrong compare range (comparing beyond second half)
4. Missing cycle assumption in non-LeetCode environments

### Why this method is practical optimum

It achieves linear time with constant extra memory, and can preserve original list by restoration.
This is usually the best tradeoff under production-style constraints.

---

## FAQ and Notes

1. **Why compare only while `p2` is not null?**  
   Second half length is less than or equal to first half length; this covers all mirrored pairs.

2. **What if list has a cycle?**  
   LeetCode input has no cycle. In production, detect cycle first (Floyd) before this template.

3. **Will reversal damage original structure?**  
   Temporarily yes; final reverse+reconnect restores it.

4. **Can I skip restore in interview?**  
   Depends on interviewer constraints. In engineering code, restoration is strongly recommended.

---

## Best Practices

- Use one stable template: `first_half_end` + `reverse(first_half_end.next)`
- Keep restore step mandatory unless explicitly allowed to mutate
- Test both odd and even lengths, plus edge cases (`[]`, `[x]`, non-palindrome near center)

---

## S - Summary

- Singly linked lists cannot directly do backward traversal for symmetry checks.
- Fast/slow pointers locate split point in O(n).
- Reversing second half converts backward comparison into forward comparison.
- Restore step preserves original structure and avoids side effects.
- This template transfers to many list problems using "middle + half processing".

### Recommended Further Reading

- LeetCode 234. Palindrome Linked List
- LeetCode 206. Reverse Linked List
- LeetCode 143. Reorder List
- LeetCode 876. Middle of the Linked List

---

## Conclusion

The value of this problem is not only palindrome checking.
It is a reusable engineering pattern:
**locate middle, temporarily transform half, compare, restore**.

---

## References

- https://leetcode.com/problems/palindrome-linked-list/
- https://leetcode.com/problems/reverse-linked-list/
- https://leetcode.com/problems/middle-of-the-linked-list/
- https://en.cppreference.com/w/cpp/container/forward_list

---

## Meta Info

- **Reading time**: 10-14 min
- **Tags**: Hot100, linked list, palindrome, fast slow pointers, in-place reverse
- **SEO keywords**: Palindrome Linked List, reverse second half, O(1) space, LeetCode 234, Hot100
- **Meta description**: O(n)/O(1) palindrome check by fast/slow split, reverse second half, compare, and restore.

---

## Call To Action (CTA)

After this one, solve these in order with the same skill set:

1. `206` Reverse Linked List
2. `143` Reorder List
3. `92` Reverse Linked List II

Treat them as one template family, not isolated questions.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def end_of_first_half(head: ListNode) -> ListNode:
    fast, slow = head, head
    while fast.next and fast.next.next:
        fast = fast.next.next
        slow = slow.next  # type: ignore[assignment]
    return slow


def is_palindrome(head: ListNode | None) -> bool:
    if head is None or head.next is None:
        return True
    first_half_end = end_of_first_half(head)
    second_half_start = reverse_list(first_half_end.next)
    p1, p2 = head, second_half_start
    ok = True
    while ok and p2:
        if p1.val != p2.val:
            ok = False
        p1 = p1.next  # type: ignore[assignment]
        p2 = p2.next
    first_half_end.next = reverse_list(second_half_start)
    return ok
```

```c
struct ListNode {
    int val;
    struct ListNode *next;
};

static struct ListNode* reverse(struct ListNode* head) {
    struct ListNode* prev = 0;
    struct ListNode* cur = head;
    while (cur) {
        struct ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

static struct ListNode* endFirstHalf(struct ListNode* head) {
    struct ListNode* fast = head;
    struct ListNode* slow = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

int isPalindrome(struct ListNode* head) {
    if (!head || !head->next) return 1;
    struct ListNode* firstEnd = endFirstHalf(head);
    struct ListNode* second = reverse(firstEnd->next);
    int ok = 1;
    struct ListNode *p1 = head, *p2 = second;
    while (ok && p2) {
        if (p1->val != p2->val) ok = 0;
        p1 = p1->next;
        p2 = p2->next;
    }
    firstEnd->next = reverse(second);
    return ok;
}
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

static ListNode* reverse(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* cur = head;
    while (cur) {
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

static ListNode* endFirstHalf(ListNode* head) {
    ListNode* fast = head;
    ListNode* slow = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

bool isPalindrome(ListNode* head) {
    if (!head || !head->next) return true;
    ListNode* firstEnd = endFirstHalf(head);
    ListNode* second = reverse(firstEnd->next);
    bool ok = true;
    ListNode *p1 = head, *p2 = second;
    while (ok && p2) {
        if (p1->val != p2->val) ok = false;
        p1 = p1->next;
        p2 = p2->next;
    }
    firstEnd->next = reverse(second);
    return ok;
}
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverse(head *ListNode) *ListNode {
	var prev *ListNode
	cur := head
	for cur != nil {
		nxt := cur.Next
		cur.Next = prev
		prev = cur
		cur = nxt
	}
	return prev
}

func endFirstHalf(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

func isPalindrome(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	firstEnd := endFirstHalf(head)
	second := reverse(firstEnd.Next)
	ok := true
	p1, p2 := head, second
	for ok && p2 != nil {
		if p1.Val != p2.Val {
			ok = false
		}
		p1 = p1.Next
		p2 = p2.Next
	}
	firstEnd.Next = reverse(second)
	return ok
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
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    let mut vals: Vec<i32> = Vec::new();
    let mut cur = head.as_ref();
    while let Some(node) = cur {
        vals.push(node.val);
        cur = node.next.as_ref();
    }
    let mut i = 0usize;
    let mut j = vals.len().saturating_sub(1);
    while i < j {
        if vals[i] != vals[j] {
            return false;
        }
        i += 1;
        j = j.saturating_sub(1);
    }
    true
}
```

```javascript
function reverse(head) {
  let prev = null;
  let cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}

function endFirstHalf(head) {
  let fast = head, slow = head;
  while (fast.next && fast.next.next) {
    fast = fast.next.next;
    slow = slow.next;
  }
  return slow;
}

function isPalindrome(head) {
  if (!head || !head.next) return true;
  const firstEnd = endFirstHalf(head);
  const second = reverse(firstEnd.next);
  let ok = true;
  let p1 = head, p2 = second;
  while (ok && p2) {
    if (p1.val !== p2.val) ok = false;
    p1 = p1.next;
    p2 = p2.next;
  }
  firstEnd.next = reverse(second);
  return ok;
}
```
