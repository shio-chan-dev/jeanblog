---
title: "LeetCode 19: Remove Nth Node From End of List (One-pass Two Pointers) ACERS Guide"
date: 2026-02-12T13:50:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["linked list", "two pointers", "fast and slow pointers", "dummy node", "LeetCode 19"]
description: "From naive two-pass traversal to one-pass fast/slow pointers, this ACERS guide explains Remove Nth Node From End of List with correctness intuition, edge cases, engineering mappings, and runnable multi-language code."
keywords: ["LeetCode 19", "Remove Nth Node From End of List", "linked list", "two pointers", "dummy node", "one pass"]
---

> **Subtitle / Summary**  
> The hard part is not deletion itself, but locating the predecessor of the nth node from the end in a singly linked list. This article derives the one-pass two-pointer solution from simpler baselines and explains correctness, boundaries, and engineering transfer.

- **Reading time**: 12-15 min  
- **Tags**: `linked list`, `two pointers`, `interview high frequency`  
- **SEO keywords**: LeetCode 19, Remove Nth Node From End of List, linked list, fast/slow pointers, dummy node  
- **Meta description**: A complete ACERS walkthrough for removing the nth node from the end: from brute force to one-pass two pointers, with complexity, pitfalls, engineering scenarios, and Python/C/C++/Go/Rust/JS implementations.

---

## Target Readers

- Beginners building a stable template for linked-list interview problems
- Developers who know fast/slow pointers but still make boundary mistakes
- Backend/system engineers who want to transfer problem-solving templates to chain-structured data in production

## Background / Motivation

"Remove the nth node from the end" is a classic medium-level linked-list problem. The challenge is usually not the delete operation itself, but:

- Singly linked lists cannot traverse backward from tail;
- Deleting the head node complicates return handling;
- Incorrect `next` rewiring can easily break the list.

Once you master this problem, you get a reusable pattern: **dummy node + fixed pointer gap**, which is useful in many list operations (split, reverse by group, merge variants).

## Core Concepts

- **Singly linked list**: each node has only `next`, so traversal is one-directional.
- **Dummy node**: add a virtual node before `head` to unify head deletion and middle deletion.
- **Fast/slow fixed gap**: move `fast` ahead by `n` steps first; when `fast` reaches the end, `slow` lands at the predecessor of the target node.

---

## A - Algorithm (Problem & Algorithm)

### Problem Restatement

Given the head of a linked list, remove the nth node from the end of the list and return its head.

### Input / Output

| Item | Type | Meaning |
| --- | --- | --- |
| `head` | `ListNode` | head of a singly linked list |
| `n` | `int` | nth position from the end |
| return | `ListNode` | head after deletion |

### Example 1

```text
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

Explanation: the 2nd node from the end is `4`, so remove it.

### Example 2

```text
Input: head = [1], n = 1
Output: []
```

Explanation: removing the only node leaves an empty list.

### Example 3

```text
Input: head = [1,2], n = 2
Output: [2]
```

Explanation: the 2nd node from the end is the head node `1`.

### Pointer-gap diagram

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null

After moving fast by n=2:
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow
             fast

Move both together until fast reaches the tail:
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
             slow           fast

Now slow.next is the target node (4)
```

---

## C - Concepts (Core Ideas)

### Derivation: from naive to optimal

1. **Naive array conversion**  
   Convert to array, remove by index, then rebuild list.
   - Works, but uses `O(L)` extra space.
   - Avoids linked-list strengths instead of using them.

2. **Two-pass traversal**  
   First pass gets length `L`; second pass stops at index `L - n - 1`.
   - Time `O(L)`, space `O(1)`.
   - Still needs two scans; head deletion handling is awkward without dummy.

3. **Best approach: one-pass two pointers + dummy**  
   - Move `fast` forward `n` steps first.
   - Move `fast` and `slow` together until `fast.next == null`.
   - `slow.next` is exactly the node to remove.

### Method category

- Two pointers
- Gap maintenance
- In-place pointer rewiring

### Correctness intuition

Let list length be `L`. If `fast` and `slow` maintain a fixed gap of `n` nodes:

- When `fast` is at index `L - 1` (tail),
- `slow` is at index `L - n - 1` (predecessor of target).

So removing `slow.next` is exactly removing the nth node from the end.

---

## Practice Guide / Steps

1. Create `dummy` and point `dummy.next = head`.
2. Initialize `fast = slow = dummy`.
3. Move `fast` ahead by `n` steps.
4. While `fast.next != null`, move both pointers forward.
5. Delete node by rewiring: `slow.next = slow.next.next`.
6. Return `dummy.next`.

Runnable Python example:

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy

    for _ in range(n):
        fast = fast.next

    while fast.next is not None:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next


def from_list(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    print(to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)))  # [1,2,3,5]
    print(to_list(remove_nth_from_end(from_list([1]), 1)))              # []
    print(to_list(remove_nth_from_end(from_list([1, 2]), 2)))           # [2]
```

---

## E - Engineering (Real-world Applications)

> The transferable idea is: **remove the kth element from tail in a single-direction chain**.

### Scenario 1: retry-trace chain trimming in backend jobs (Go)

**Background**: microservices often keep a singly linked retry trace for task failures.  
**Why it fits**: deleting the nth record from tail can reuse the exact fast/slow template.

```go
package main

import "fmt"

type Node struct {
	ID   int
	Next *Node
}

func removeNthFromEnd(head *Node, n int) *Node {
	dummy := &Node{Next: head}
	fast, slow := dummy, dummy

	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

func printList(head *Node) {
	for p := head; p != nil; p = p.Next {
		fmt.Printf("%d ", p.ID)
	}
	fmt.Println()
}

func main() {
	head := &Node{1, &Node{2, &Node{3, &Node{4, nil}}}}
	head = removeNthFromEnd(head, 2)
	printList(head) // 1 2 4
}
```

### Scenario 2: free-block chain cleanup in systems code (C)

**Background**: simplified memory managers may keep free blocks in a singly linked list.  
**Why it fits**: removing the nth node from the end can be done in one scan with deterministic pointer rewiring.

```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int addr;
    struct Node* next;
};

struct Node* remove_nth_from_end(struct Node* head, int n) {
    struct Node dummy = {0, head};
    struct Node *fast = &dummy, *slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }

    struct Node* del = slow->next;
    slow->next = del->next;
    free(del);
    return dummy.next;
}

int main() {
    struct Node* n4 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n3 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n2 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n1 = (struct Node*)malloc(sizeof(struct Node));
    n1->addr = 10; n1->next = n2;
    n2->addr = 20; n2->next = n3;
    n3->addr = 30; n3->next = n4;
    n4->addr = 40; n4->next = NULL;

    struct Node* head = remove_nth_from_end(n1, 3);
    for (struct Node* p = head; p; p = p->next) printf("%d ", p->addr);
    printf("\n");

    while (head) {
        struct Node* t = head;
        head = head->next;
        free(t);
    }
    return 0;
}
```

### Scenario 3: undo-chain compaction in frontend editor state (JavaScript)

**Background**: an editor can model undo history as a singly linked chain.  
**Why it fits**: deleting the nth snapshot from tail uses the same list primitive as this problem.

```javascript
class Node {
  constructor(v, next = null) {
    this.v = v;
    this.next = next;
  }
}

function removeNthFromEnd(head, n) {
  const dummy = new Node(0, head);
  let fast = dummy;
  let slow = dummy;

  for (let i = 0; i < n; i++) fast = fast.next;
  while (fast.next !== null) {
    fast = fast.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;
  return dummy.next;
}

function print(head) {
  const arr = [];
  for (let p = head; p; p = p.next) arr.push(p.v);
  console.log(arr);
}

const head = new Node(1, new Node(2, new Node(3, new Node(4))));
print(removeNthFromEnd(head, 1)); // [1,2,3]
```

---

## R - Reflection (Deep Dive)

### Complexity

- Time: `O(L)`, where `L` is list length
- Space: `O(1)` extra space (in-place pointer update)

### Approach comparison

| Approach | Time | Space | Pros | Cons |
| --- | --- | --- | --- | --- |
| Array conversion | O(L) | O(L) | intuitive | high extra space, less list-native |
| Two-pass traversal | O(L) | O(1) | stable and simple | two scans |
| One-pass + dummy | O(L) | O(1) | single scan, unified boundaries | requires gap invariant discipline |

### Common mistakes

- Forgetting dummy node, which explodes branches for head deletion
- Off-by-one errors by moving `fast` `n+1` or `n-1` steps
- Forgetting to free removed node in C/C++ contexts

### Why this method is production-friendly

- Template-like and reusable across many linked-list variants
- Stable boundary behavior, especially for removing the head
- Efficient in performance-sensitive systems (`O(1)` extra memory)

---

## FAQ

### Q1: Why use `while fast.next != null` instead of `while fast != null`?

Because we need `slow` to stop at the predecessor of the target node. Stopping when `fast` is exactly at the tail gives the correct predecessor position.

### Q2: What if `n` equals the list length?

It still works. With dummy, `slow` stays at dummy and we remove the original head safely.

### Q3: Can this be written recursively?

Yes, but recursion adds `O(L)` call-stack space. Iteration is generally more stable for long lists.

---

## Best Practices

- Always create dummy first, then implement deletion logic
- Standardize on "move `fast` by `n` first" to reduce off-by-one bugs
- Teach with two-pass baseline first, then optimize to one-pass
- In C/C++, explicitly release the removed node

---

## S - Summary

### Key takeaways

1. "nth from end" can be transformed into a fixed-gap two-pointer problem.
2. Dummy node is the safest boundary tool for linked-list deletion.
3. One-pass fast/slow gives a strong engineering balance: `O(L)` time, `O(1)` extra space.
4. This template transfers directly to many chain-structure rewiring tasks.
5. Many medium problems are robust combinations of small stable templates.

### Recommended reading

- LeetCode 19 (official): <https://leetcode.com/problems/remove-nth-node-from-end-of-list/>
- LeetCode CN: <https://leetcode.cn/problems/remove-nth-node-from-end-of-list/>
- Related: LeetCode 21, LeetCode 206, LeetCode 25

---

## CTA

Rewrite this from memory now:

1. Implement the two-pass version.
2. Refactor it into one-pass fast/slow.
3. Verify with three edge cases: `n=1`, `n=len`, and single-node list.

You will noticeably improve linked-list reliability in interviews and real code.

---

## Runnable Multi-language Implementations

### Python

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next is not None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next


def from_list(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for x in nums:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = from_list([1, 2, 3, 4, 5])
    print(to_list(remove_nth_from_end(h, 2)))  # [1, 2, 3, 5]
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* new_node(int v) {
    struct ListNode* p = (struct ListNode*)malloc(sizeof(struct ListNode));
    p->val = v;
    p->next = NULL;
    return p;
}

struct ListNode* removeNthFromEnd(struct ListNode* head, int n) {
    struct ListNode dummy = {0, head};
    struct ListNode *fast = &dummy, *slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }

    struct ListNode* del = slow->next;
    slow->next = del->next;
    free(del);
    return dummy.next;
}

void print_list(struct ListNode* head) {
    for (struct ListNode* p = head; p; p = p->next) printf("%d ", p->val);
    printf("\n");
}

void free_list(struct ListNode* head) {
    while (head) {
        struct ListNode* t = head;
        head = head->next;
        free(t);
    }
}

int main() {
    struct ListNode* h1 = new_node(1);
    h1->next = new_node(2);
    h1->next->next = new_node(3);
    h1->next->next->next = new_node(4);
    h1->next->next->next->next = new_node(5);

    h1 = removeNthFromEnd(h1, 2);
    print_list(h1); // 1 2 3 5
    free_list(h1);
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0);
    dummy.next = head;
    ListNode* fast = &dummy;
    ListNode* slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next != nullptr) {
        fast = fast->next;
        slow = slow->next;
    }
    ListNode* del = slow->next;
    slow->next = del->next;
    delete del;
    return dummy.next;
}

ListNode* build(const vector<int>& a) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int x : a) {
        tail->next = new ListNode(x);
        tail = tail->next;
    }
    return dummy.next;
}

void print(ListNode* head) {
    for (ListNode* p = head; p; p = p->next) cout << p->val << " ";
    cout << "\n";
}

void destroy(ListNode* head) {
    while (head) {
        ListNode* t = head;
        head = head->next;
        delete t;
    }
}

int main() {
    ListNode* h = build({1, 2, 3, 4, 5});
    h = removeNthFromEnd(h, 2);
    print(h); // 1 2 3 5
    destroy(h);
    return 0;
}
```

### Go

```go
package main

import "fmt"

type ListNode struct {
	Val  int
	Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	fast, slow := dummy, dummy

	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

func build(nums []int) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for _, x := range nums {
		tail.Next = &ListNode{Val: x}
		tail = tail.Next
	}
	return dummy.Next
}

func printList(head *ListNode) {
	for p := head; p != nil; p = p.Next {
		fmt.Printf("%d ", p.Val)
	}
	fmt.Println()
}

func main() {
	head := build([]int{1, 2, 3, 4, 5})
	head = removeNthFromEnd(head, 2)
	printList(head) // 1 2 3 5
}
```

### Rust (safe runnable two-pass variant)

> To keep ownership handling concise and safe in Rust, this implementation uses a two-pass traversal while preserving `O(L)` time and `O(1)` extra space.

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

fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut len = 0usize;
    let mut p = head.as_ref();
    while let Some(node) = p {
        len += 1;
        p = node.next.as_ref();
    }

    let idx = len - n as usize;
    let mut dummy = Box::new(ListNode { val: 0, next: head });
    let mut cur = &mut dummy;
    for _ in 0..idx {
        cur = cur.next.as_mut().unwrap();
    }
    let next = cur.next.as_mut().and_then(|node| node.next.take());
    cur.next = next;
    dummy.next
}

fn from_vec(a: Vec<i32>) -> Option<Box<ListNode>> {
    let mut head = None;
    for &x in a.iter().rev() {
        let mut node = Box::new(ListNode::new(x));
        node.next = head;
        head = Some(node);
    }
    head
}

fn to_vec(mut head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut out = Vec::new();
    while let Some(mut node) = head {
        out.push(node.val);
        head = node.next.take();
    }
    out
}

fn main() {
    let head = from_vec(vec![1, 2, 3, 4, 5]);
    let ans = remove_nth_from_end(head, 2);
    println!("{:?}", to_vec(ans)); // [1, 2, 3, 5]
}
```

### JavaScript

```javascript
class ListNode {
  constructor(val = 0, next = null) {
    this.val = val;
    this.next = next;
  }
}

function removeNthFromEnd(head, n) {
  const dummy = new ListNode(0, head);
  let fast = dummy;
  let slow = dummy;

  for (let i = 0; i < n; i++) {
    fast = fast.next;
  }
  while (fast.next !== null) {
    fast = fast.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;
  return dummy.next;
}

function fromArray(arr) {
  const dummy = new ListNode();
  let tail = dummy;
  for (const x of arr) {
    tail.next = new ListNode(x);
    tail = tail.next;
  }
  return dummy.next;
}

function toArray(head) {
  const out = [];
  for (let p = head; p; p = p.next) out.push(p.val);
  return out;
}

const head = fromArray([1, 2, 3, 4, 5]);
console.log(toArray(removeNthFromEnd(head, 2))); // [1,2,3,5]
```
