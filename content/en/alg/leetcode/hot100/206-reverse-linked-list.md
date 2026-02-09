---
title: "Hot100: Reverse Linked List Three-Pointer Iterative/Recursive ACERS Guide"
date: 2026-02-09T17:27:59+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "pointer", "iteration", "recursion", "LeetCode 206"]
description: "Reverse a singly linked list in O(n) time and O(1) extra space with the three-pointer iterative template, with recursive comparison, engineering mapping, and runnable multi-language implementations."
keywords: ["Reverse Linked List", "three pointers", "iterative", "recursive", "LeetCode 206", "Hot100", "O(1) space"]
---

> **Subtitle / Summary**  
> Reverse Linked List is the first serious pointer-rewiring exercise in Hot100. It looks simple, but most bugs come from broken links and wrong operation order. This ACERS guide explains the three-pointer iterative template thoroughly and compares it with recursion.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `linked list`, `pointer`, `iteration`  
- **SEO keywords**: Hot100, Reverse Linked List, three pointers, iterative, recursive, LeetCode 206  
- **Meta description**: Three-pointer iterative reversal in O(n)/O(1), with recursive contrast, common pitfalls, engineering mapping, and runnable multi-language implementations.

---

## Target Readers

- Hot100 learners and interview candidates
- Developers who often hit null-pointer or broken-chain bugs in list problems
- Engineers who want stable pointer manipulation patterns in C/C++/Rust/Go

## Background / Motivation

In production code, "reverse linked list" may not appear as a LeetCode function, but the skill is highly transferable:

- Reorder nodes in-place with **O(1)** extra memory
- Keep link integrity while changing direction
- Handle `head = null` and single-node lists without special-case chaos

If this template is truly internalized, many list problems become straightforward: reverse sublist, reverse k-group, palindrome list, and so on.

## Core Concepts

- **Singly linked list**: each node has one `next` pointer
- **Broken-link risk**: if you overwrite `cur.next` before saving old next, you lose the remaining chain
- **Three pointers (`prev`, `cur`, `next`)**: save successor, reverse link, then advance
- **Loop invariant**:
  - `prev` is always the head of the already reversed part
  - `cur` is always the head of the not-yet-processed part

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a singly linked list, reverse the list and return the new head.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| head | ListNode | head of singly linked list (can be null) |
| return | ListNode | new head after reversal |

### Example 1

```text
input:  1 -> 2 -> 3 -> 4 -> 5 -> null
output: 5 -> 4 -> 3 -> 2 -> 1 -> null
```

### Example 2

```text
input:  1 -> 2 -> null
output: 2 -> 1 -> null
```

---

## Thought Process: From Naive to In-Place

### Naive idea: copy values and rebuild

- Traverse list, collect values, rebuild in reverse order
- Works, but uses O(n) extra memory and creates new nodes

For interviews and systems code, this is usually not what is asked.

### Key observation

You do not need new nodes.  
You only need to rewire `next`.

For current node `cur`:

1. Save old successor: `next = cur.next`
2. Reverse pointer: `cur.next = prev`
3. Move forward: `prev = cur`, `cur = next`

### Method choice

Use iterative three-pointer template:

- Time: O(n)
- Extra space: O(1)
- Stable and stack-safe

---

## C - Concepts (Core Ideas)

### Method Category

- In-place linked-list manipulation
- Iterative simulation
- Recursion as equivalent reference solution

### Loop Invariant (why it is correct)

At each loop start:

- `prev` points to a valid reversed list
- `cur` points to the first node not yet reversed
- Original nodes are partitioned into:
  - reversed prefix
  - untouched suffix

Each iteration moves exactly one node from untouched suffix to reversed prefix.
When `cur == null`, all nodes are in reversed prefix, and `prev` is the new head.

### Recursive counterpart

Recursive idea:

1. Reverse `head.next` onward and get `new_head`
2. Let `head.next.next = head`
3. Set `head.next = null` to cut old forward link

Readable, but stack usage is O(n).

---

## Practical Guide / Steps

1. Initialize `prev = null`, `cur = head`
2. While `cur != null`:
   - `next = cur.next`
   - `cur.next = prev`
   - `prev = cur`
   - `cur = next`
3. Return `prev`

Runnable Python example (`reverse_list.py`):

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    prev = None
    cur = head
    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


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
    print(to_list(reverse_list(h)))
```

---

## Explanation / Why This Works

The order of operations is the whole point:

1. Save `next` first (avoid losing remaining chain)
2. Reverse current edge (`cur.next = prev`)
3. Advance both pointers

If you swap step 1 and 2, the rest of list may become unreachable.
That is the most common bug in this problem.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: free-list reorder in memory-oriented systems (C)

**Background**: some allocators keep free blocks in a singly linked free-list.  
**Why it fits**: reversing list order is an in-place strategy to alter reuse order without allocating memory.

```c
#include <stdio.h>

typedef struct Node {
    int id;
    struct Node* next;
} Node;

Node* reverse(Node* head) {
    Node* prev = NULL;
    Node* cur = head;
    while (cur) {
        Node* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

int main(void) {
    Node c = {3, NULL};
    Node b = {2, &c};
    Node a = {1, &b};
    Node* head = reverse(&a);
    for (Node* p = head; p; p = p->next) printf("%d ", p->id);
    printf("\n");
    return 0;
}
```

### Scenario 2: server-side operation stack replay direction switch (Go)

**Background**: a lightweight task chain is stored as a singly linked stack.  
**Why it fits**: reversing in-place switches replay direction without extra containers.

```go
package main

import "fmt"

type Node struct {
	Val  int
	Next *Node
}

func reverse(head *Node) *Node {
	var prev *Node
	cur := head
	for cur != nil {
		nxt := cur.Next
		cur.Next = prev
		prev = cur
		cur = nxt
	}
	return prev
}

func main() {
	head := &Node{1, &Node{2, &Node{3, nil}}}
	head = reverse(head)
	for p := head; p != nil; p = p.Next {
		fmt.Print(p.Val, " ")
	}
	fmt.Println()
}
```

### Scenario 3: pointer-animation teaching in browser (JavaScript)

**Background**: in algorithm visualization, object references simulate list nodes.  
**Why it fits**: the three-pointer state transition is easy to animate frame-by-frame.

```javascript
function Node(val, next = null) {
  this.val = val;
  this.next = next;
}

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

let head = new Node(1, new Node(2, new Node(3)));
head = reverse(head);
const out = [];
for (let p = head; p; p = p.next) out.push(p.val);
console.log(out.join(" "));
```

---

## R - Reflection

### Complexity

- Iterative:
  - Time: `O(n)`
  - Space: `O(1)`
- Recursive:
  - Time: `O(n)`
  - Space: `O(n)` (call stack)

### Alternatives and Tradeoffs

| Method | Time | Extra Space | Tradeoff |
| --- | --- | --- | --- |
| Rebuild new list | O(n) | O(n) | Easy but not in-place |
| Recursive reversal | O(n) | O(n) | Elegant, but stack-risk on long lists |
| Three-pointer iterative | O(n) | O(1) | Best engineering baseline |

### Common Mistakes

1. Rewire before saving `next` (break chain)
2. Forget advancing `cur` (infinite loop)
3. Recursive version forgetting `head.next = null` (cycle risk)
4. Reversing values instead of links (misses structural requirement)

### Why iterative is most practical

- Stack-safe for very long lists
- Local, inspectable pointer transitions
- Better fit for systems programming and production safety constraints

---

## FAQ and Notes

1. **What about empty list or single node?**  
   The same loop handles both naturally.

2. **Is three pointers mandatory?**  
   You must preserve successor somehow; variable names can differ, state is equivalent.

3. **Why not prefer recursion since it's shorter?**  
   Recursive depth can overflow stack on long input; iterative is safer as default.

4. **How to self-check quickly?**  
   Use this rule: **save next first, reverse next, then advance**.

---

## Best Practices

- Memorize the operation order as a fixed template
- Draw pointer state for at least 2-3 iterations before coding
- Use iterative as default in production-quality code

---

## S - Summary

- Reverse Linked List is fundamentally pointer rewiring, not value swapping.
- Three-pointer iterative template achieves `O(n)` time and `O(1)` extra space.
- Correct order is the core: save successor -> reverse link -> advance.
- Recursive form is useful for understanding, but iterative is usually safer for engineering.

### Recommended Further Reading

- LeetCode 206. Reverse Linked List
- LeetCode 92. Reverse Linked List II
- LeetCode 25. Reverse Nodes in k-Group
- LeetCode 234. Palindrome Linked List

---

## Conclusion

Once the three-pointer template is stable in your muscle memory, most linked-list reversal variants become local modifications rather than new problems.

---

## References

- https://leetcode.com/problems/reverse-linked-list/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/std/option/enum.Option.html
- https://go.dev/tour/moretypes/6

---

## Meta Info

- **Reading time**: 10-12 min
- **Tags**: Hot100, linked list, pointer, iteration, LeetCode 206
- **SEO keywords**: Reverse Linked List, three pointers, iterative, recursive, LeetCode 206, Hot100
- **Meta description**: O(n)/O(1) linked-list reversal with three pointers, with recursive comparison and runnable multi-language implementations.

---

## Call To Action (CTA)

Do two drills to lock this in:

1. Manually trace `prev/cur/next` for at least three steps without code
2. Solve LeetCode 92 right after this one to reuse the same rewiring pattern

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    cur = head
    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* reverseList(struct ListNode* head) {
    struct ListNode* prev = NULL;
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

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
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
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
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
```

```rust
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    pub fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

pub fn reverse_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut prev: Option<Box<ListNode>> = None;
    while let Some(mut node) = head {
        head = node.next.take();
        node.next = prev;
        prev = Some(node);
    }
    prev
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseList(head) {
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
