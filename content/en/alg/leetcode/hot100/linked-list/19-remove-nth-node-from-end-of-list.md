---
title: "Hot100: Remove Nth Node From End of List One-Pass Two-Pointer ACERS Guide"
date: 2026-04-20T09:36:56+08:00
draft: false
url: "/alg/leetcode/hot100/19-remove-nth-node-from-end-of-list/"
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "two pointers", "fast slow pointers", "dummy node", "LeetCode 19"]
description: "Remove the n-th node from the end of a singly linked list by turning backward counting into a fixed-gap two-pointer problem. This ACERS guide derives the one-pass solution and includes runnable multi-language code."
keywords: ["Remove Nth Node From End of List", "one-pass two pointers", "dummy node", "fast slow pointers", "LeetCode 19", "Hot100"]
---

> **Subtitle / Summary**  
> The hard part of LeetCode 19 is not deleting a node. It is locating the predecessor of the n-th node from the end in a singly linked list without walking backward.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `linked list`, `two pointers`, `dummy node`  
- **SEO keywords**: Remove Nth Node From End of List, one-pass two pointers, dummy node, LeetCode 19, Hot100  
- **Meta description**: A derivation-first ACERS explanation of LeetCode 19 with fixed-gap two pointers, dummy node boundary handling, engineering mapping, and runnable multi-language implementations.

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the head of a linked list, remove the n-th node from the end of the list and return the head of the modified list.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| `head` | `ListNode` | head of a singly linked list |
| `n` | `int` | position counted from the end |
| return | `ListNode` | head after deletion |

### Example 1

```text
input:  head = [1,2,3,4,5], n = 2
output: [1,2,3,5]
```

### Example 2

```text
input:  head = [1], n = 1
output: []
```

### Example 3

```text
input:  head = [1,2], n = 2
output: [2]
```

### Visual Intuition (Fixed Gap)

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
fast goes first by n = 2 nodes:
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow
             fast

Then move both together until fast reaches the tail:
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
             slow           fast

Now slow.next is exactly the node to delete.
```

---

## Target Readers

- Hot100 learners who want a reliable one-pass linked-list deletion template
- Developers who know two pointers but still get boundary cases wrong
- Engineers interested in turning "count from the end" into a forward-only traversal

## Background / Motivation

In a singly linked list, deleting a node is easy **if you already have its predecessor**.
The real problem is finding that predecessor when the target is described from the end:

- singly linked lists cannot move backward
- deleting the head requires special handling
- careless pointer updates can break the chain

This is why LeetCode 19 is a classic template problem.
It teaches how to convert a backward-position description into a forward traversal rule.

## Core Concepts

- **Singly linked list**: only forward traversal through `next`
- **Dummy node**: creates a stable predecessor even when the head is deleted
- **Fixed-gap two pointers**: let `fast` stay `n` nodes ahead of `slow`
- **Predecessor deletion rule**: `prev.next = prev.next.next`

---

## C - Concepts (Core Ideas)

### How To Build The Solution From Scratch

#### Step 1: Ask what node we really need to find

The problem says "delete the n-th node from the end".
But the actual linked-list operation is never:

```python
delete(target)
```

It is:

```python
prev.next = prev.next.next
```

So the useful target is not the node itself.
It is the node **before** the one we want to delete.

#### Step 2: Acknowledge the easy solutions first

Two straightforward solutions exist:

- copy nodes into an array, compute the position, and reconnect
- run two passes: first compute length, then walk to `length - n`

Both can work.
But neither captures the best one-pass linked-list idea.

#### Step 3: Turn "from the end" into a fixed-gap rule

Suppose `fast` first moves `n` steps ahead of `slow`.
Then the distance between them is fixed:

```text
slow ... n nodes ... fast
```

If both pointers move one step at a time afterward, that gap stays unchanged.
So when `fast` reaches the last node, `slow` must be standing right before the target node.

That is the whole conversion:

> "count from the end" becomes "maintain a gap of n".

#### Step 4: Why do we still need a dummy node?

If the node to delete is the original head, then its real predecessor does not exist.
Without a dummy node, that case becomes a special branch.

So we normalize every case with:

```python
dummy = ListNode(0, head)
fast = slow = dummy
```

Now even the original head has a valid predecessor: `dummy`.

#### Step 5: Define the movement phases precisely

Phase 1:

```python
for _ in range(n):
    fast = fast.next
```

Phase 2:

```python
while fast.next is not None:
    fast = fast.next
    slow = slow.next
```

When phase 2 ends, `slow.next` is exactly the node to remove.

#### Step 6: Define the deletion step

After localization, the deletion itself is small:

```python
slow.next = slow.next.next
```

That single line works for:

- deleting a middle node
- deleting the last node
- deleting the original head

because `dummy` unified all three cases.

#### Step 7: Walk one trace slowly

For:

```text
1 -> 2 -> 3 -> 4 -> 5, n = 2
```

After `fast` moves first by 2:

- `slow = dummy`
- `fast = 2`

Move both together:

- `slow = 1`, `fast = 3`
- `slow = 2`, `fast = 4`
- `slow = 3`, `fast = 5`

Now `fast.next` is `null`, so stop.
`slow.next` is `4`, which is exactly the node to delete.

#### Step 8: Reduce the method to one sentence

LeetCode 19 is "add a dummy node, let `fast` lead by `n`, then move both pointers together until `slow` reaches the predecessor of the target."

### Assemble the Full Code

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
    print(to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)))  # [1, 2, 3, 5]
    print(to_list(remove_nth_from_end(from_list([1]), 1)))              # []
    print(to_list(remove_nth_from_end(from_list([1, 2]), 2)))           # [2]
```

### Reference Answer

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy

    for _ in range(n):
        fast = fast.next

    while fast.next is not None:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next
```

### Method Category

- Two pointers
- Fixed-gap traversal
- Dummy-node boundary normalization

### Correctness Intuition

The invariant after phase 1 is:

- `fast` is `n` nodes ahead of `slow`

Phase 2 keeps that distance unchanged.
So when `fast` reaches the final node, `slow` must be right before the node counted as the n-th from the end.

### Why the dummy node matters

Without a dummy node, deleting the original head would require a separate branch.
With `dummy`, every deletion is expressed by the same predecessor-based line:

```python
slow.next = slow.next.next
```

## Practice Guide / Steps

1. Create `dummy = ListNode(0, head)`.
2. Set `fast = slow = dummy`.
3. Move `fast` forward by `n` nodes.
4. Move `fast` and `slow` together until `fast.next` becomes `None`.
5. Delete `slow.next`.
6. Return `dummy.next`.

Runnable Python example (`remove_nth_from_end.py`):

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
    cur = dummy
    for x in nums:
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
    print(to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)))
```

---

## Explanation / Why This Works

The algorithm never needs to know the full length of the list.
Instead, it converts a backward distance into a forward invariant:

- keep `fast` exactly `n` nodes ahead of `slow`

That is why the method stays one-pass after the initial lead-in movement.
The deletion itself is tiny; the real insight is how the gap encodes the target position.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: retry-chain trimming in backend jobs (Go)

**Background**: a retry queue stores attempts as a linked chain.  
**Why it fits**: operators often remove the k-th attempt counted backward from the newest record.

```go
package main

import "fmt"

type Node struct {
	Val  int
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

func main() {
	head := &Node{1, &Node{2, &Node{3, &Node{4, &Node{5, nil}}}}}
	head = removeNthFromEnd(head, 2)
	for p := head; p != nil; p = p.Next {
		fmt.Print(p.Val, " ")
	}
	fmt.Println()
}
```

### Scenario 2: free-block chain maintenance (C)

**Background**: a low-level allocator maintains a singly linked block chain.  
**Why it fits**: removing a block relative to the chain tail is naturally a predecessor-localization problem.

```c
#include <stdio.h>

struct Node {
    int id;
    struct Node* next;
};

struct Node* removeNthFromEnd(struct Node* head, int n) {
    struct Node dummy = {0, head};
    struct Node *fast = &dummy, *slow = &dummy;
    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }
    slow->next = slow->next->next;
    return dummy.next;
}
```

### Scenario 3: frontend undo-chain compaction (JavaScript)

**Background**: an editor stores undo snapshots as a singly linked chain.  
**Why it fits**: removing a node relative to the newest snapshot maps directly to a tail-relative deletion rule.

```javascript
function removeNthFromEnd(arr, n) {
  const idx = arr.length - n;
  return arr.filter((_, i) => i !== idx);
}

console.log(removeNthFromEnd([1, 2, 3, 4, 5], 2));
```

---

## R - Reflection (Complexity, Alternatives, Tradeoffs)

### Complexity

- Time: `O(L)`
- Auxiliary space: `O(1)`

### Alternatives

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Array conversion | O(L) | O(L) | simple but not list-native |
| Two-pass length method | O(L) | O(1) | correct, but less elegant |
| One-pass fixed-gap method | O(L) | O(1) | best reusable template |

### Common mistakes

1. Forgetting the dummy node and then mishandling head deletion.
2. Using `while fast is not None` instead of `while fast.next is not None`.
3. Deleting the target node directly without locating its predecessor.
4. Advancing `slow` too early and breaking the gap invariant.

### Why this is the best practical method

It gives you:

- one-pass traversal
- no extra container
- one uniform deletion rule for all positions

More importantly, it teaches a transferable pattern: convert tail-relative indexing into a forward-moving invariant.

---

## FAQ and Notes

1. **Why is the loop `while fast.next is not None` instead of `while fast is not None`?**  
   Because we want `slow` to stop at the predecessor of the target, not on the target itself.

2. **What if `n` equals the list length?**  
   Then the original head is deleted, and `dummy` handles that safely.

3. **Can recursion solve this too?**  
   Yes, but it is less practical and uses call-stack space.

---

## Best Practices

- Normalize deletion problems with a dummy node whenever the head might change.
- Decide first whether you need the target node or its predecessor.
- For two-pointer gaps, state the invariant in words before coding.
- Test at least these cases: delete head, delete tail, delete only node.

---

## S - Summary

- The useful node to locate is the predecessor of the target.
- A dummy node removes special-case logic for head deletion.
- The fast/slow gap converts "count from the end" into a forward-only traversal.
- This is one of the most reusable one-pass linked-list templates in Hot100.

### Further Reading

- LeetCode 19. Remove Nth Node From End of List
- LeetCode 21. Merge Two Sorted Lists
- LeetCode 92. Reverse Linked List II
- LeetCode 143. Reorder List

---

## Conclusion

Once you can express this problem as a fixed-gap invariant instead of a backward count, many linked-list problems stop feeling ad hoc and start feeling structural.

---

## References

- https://leetcode.com/problems/remove-nth-node-from-end-of-list/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://docs.python.org/3/tutorial/datastructures.html
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Indexed_collections

---

## Meta Info

- **Reading time**: 12-15 min
- **Tags**: Hot100, linked list, two pointers, dummy node, LeetCode 19
- **SEO keywords**: Remove Nth Node From End of List, fixed-gap two pointers, dummy node, LeetCode 19, Hot100
- **Meta description**: A derivation-first ACERS guide to LeetCode 19 using fixed-gap two pointers and a dummy node, with engineering mapping and runnable multi-language code.

---

## CTA

Do two drills right after this:

1. Re-implement the gap method from memory without checking the article.
2. Then solve a related list problem where the head may change again, such as LeetCode 92 or LeetCode 25.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

### Python

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
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
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = from_list([1, 2, 3, 4, 5])
    print(to_list(removeNthFromEnd(h, 2)))  # [1, 2, 3, 5]
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

int main(void) {
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

### Rust

> For a compact runnable version that stays ownership-safe, this Rust implementation uses a two-pass approach while preserving `O(L)` time and `O(1)` auxiliary space.

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
console.log(toArray(removeNthFromEnd(head, 2))); // [1, 2, 3, 5]
```
