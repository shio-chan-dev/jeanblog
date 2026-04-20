---
title: "Hot100: Add Two Numbers Linked-List Carry Simulation ACERS Guide"
date: 2026-04-20T09:36:56+08:00
draft: false
url: "/alg/leetcode/hot100/2-add-two-numbers/"
categories: ["LeetCode"]
tags: ["Hot100", "linked list", "carry", "simulation", "dummy node", "LeetCode 2"]
description: "Add two non-negative integers stored in reverse-order linked lists by simulating grade-school addition with carry propagation. This ACERS guide derives the one-pass solution and includes runnable multi-language implementations."
keywords: ["Add Two Numbers", "linked-list carry", "reverse order digits", "dummy node", "LeetCode 2", "Hot100"]
---

> **Subtitle / Summary**  
> LeetCode 2 is grade-school addition translated into a linked-list workflow. The only persistent state is the carry, and the only structural work is appending one new digit per round.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `linked list`, `carry`, `simulation`  
- **SEO keywords**: Add Two Numbers, linked-list carry, reverse order digits, LeetCode 2, Hot100  
- **Meta description**: A derivation-first ACERS guide to LeetCode 2 with carry propagation, dummy node construction, engineering mapping, and runnable multi-language code.

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

You are given two non-empty linked lists `l1` and `l2`.
Each list stores a non-negative integer in **reverse order**, and each node contains one digit.
Add the two integers and return the sum as a linked list in the same reverse order.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| `l1` | `ListNode` | first reverse-order digit list |
| `l2` | `ListNode` | second reverse-order digit list |
| return | `ListNode` | reverse-order digit list for the sum |

### Example 1

```text
input:  l1 = [2,4,3], l2 = [5,6,4]
meaning: 342 + 465 = 807
output: [7,0,8]
```

### Example 2

```text
input:  l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
output: [8,9,9,9,0,0,0,1]
```

---

## Target Readers

- Hot100 learners building reliable linked-list templates
- Developers who often make mistakes around carry propagation
- Engineers who want to map stream-wise state updates to list problems

## Background / Motivation

This looks like an entry-level linked-list problem, but it exercises three reusable skills:

- moving through two input chains in lockstep
- carrying state across rounds (`carry`)
- preserving correctness when the two inputs have different lengths

The same pattern appears in engineering-style data processing:

- digit-wise or chunk-wise aggregation
- stream alignment with missing values treated as zero
- incremental output construction with one persistent state variable

## Core Concepts

- **Reverse-order storage**: the least significant digit is at the list head
- **Carry propagation**: `carry = total // 10`, current digit = `total % 10`
- **Sentinel node (`dummy`)**: avoids special handling for the first output node
- **Tail pointer**: keeps appending the result in O(1) per digit

---

## C - Concepts (Core Ideas)

### How To Build The Solution From Scratch

#### Step 1: Start from the direction of digits

Take:

```text
l1 = 2 -> 4 -> 3
l2 = 5 -> 6 -> 4
```

These lists represent:

```text
342 + 465
```

The important structural clue is that the head already stores the ones digit.
So the list order is exactly the same order we use in grade-school addition:

- ones
- tens
- hundreds

That means we do not need to reverse anything first.

#### Step 2: Ask what state one round of addition really needs

In ordinary column addition, each round depends on only three values:

- current digit from `l1`
- current digit from `l2`
- previous carry

That is why the whole state can be compressed to:

```python
carry = 0
```

plus the two input pointers and one output tail pointer.

#### Step 3: Reject the unnecessary detours

It is tempting to:

- rebuild the full integers first
- or copy digits to arrays and then add

But both are worse fits for the problem:

- large integers may overflow in some languages
- arrays introduce extra space without giving us new information
- the lists are already aligned from low digit to high digit

So the structure of the input is already optimized for one-pass simulation.

#### Step 4: Define one addition round precisely

One round does exactly this:

```python
x = l1.val if l1 else 0
y = l2.val if l2 else 0
total = x + y + carry
carry = total // 10
digit = total % 10
```

Then append `digit` to the result list.

This is just vertical addition written in pointer form.

#### Step 5: Ask why we need `dummy` and `tail`

Every round creates one new output digit.
So we need a stable way to append a node without special-casing the first append.

That is why we start with:

```python
dummy = ListNode(0)
tail = dummy
```

and keep doing:

```python
tail.next = ListNode(digit)
tail = tail.next
```

#### Step 6: Decide when the loop is truly finished

The work is not done until all three sources are exhausted:

- `l1` is fully consumed
- `l2` is fully consumed
- `carry` becomes zero

So the real loop condition is:

```text
while l1 != null or l2 != null or carry != 0
```

That final `carry` check is what handles cases like `5 + 5 = 10`.

#### Step 7: Walk one full trace slowly

Using:

```text
2 -> 4 -> 3
5 -> 6 -> 4
```

Round 1:

- `2 + 5 + 0 = 7`
- write `7`
- `carry = 0`

Round 2:

- `4 + 6 + 0 = 10`
- write `0`
- `carry = 1`

Round 3:

- `3 + 4 + 1 = 8`
- write `8`
- `carry = 0`

So the answer becomes:

```text
7 -> 0 -> 8
```

#### Step 8: Reduce the method to one sentence

LeetCode 2 is "walk both lists from low digit to high digit, keep one carry, and append one new digit node per round."

### Assemble the Full Code

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    carry = 0

    while l1 is not None or l2 is not None or carry:
        x = l1.val if l1 is not None else 0
        y = l2.val if l2 is not None else 0
        total = x + y + carry
        carry = total // 10

        tail.next = ListNode(total % 10)
        tail = tail.next

        if l1 is not None:
            l1 = l1.next
        if l2 is not None:
            l2 = l2.next

    return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for n in nums:
        tail.next = ListNode(n)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head is not None:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    a = build([2, 4, 3])
    b = build([5, 6, 4])
    print(dump(add_two_numbers(a, b)))  # [7, 0, 8]
```

### Reference Answer

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    carry = 0

    while l1 is not None or l2 is not None or carry:
        x = l1.val if l1 is not None else 0
        y = l2.val if l2 is not None else 0
        total = x + y + carry
        carry = total // 10
        tail.next = ListNode(total % 10)
        tail = tail.next

        if l1 is not None:
            l1 = l1.next
        if l2 is not None:
            l2 = l2.next

    return dummy.next
```

### Method Category

- Linked-list simulation
- Carry propagation
- Sentinel-node result construction

### State Model

At each round we only need:

- `l1`: current node of the first number
- `l2`: current node of the second number
- `carry`: carry from previous digit
- `tail`: current end of the output list

This is why the algorithm stays simple and linear.

### Correctness Intuition

Each round writes exactly one output digit:

- the node value is the correct digit for the current place
- `carry` stores the only information that must move into the next place

Because the input order is already low-digit first, processing from head to tail is exactly the correct arithmetic order.

## Practice Guide / Steps

1. Create `dummy` and `tail` for the result list.
2. Initialize `carry = 0`.
3. While either list still has nodes or `carry` is non-zero:
   - read current digits, using `0` for missing nodes
   - compute `total = x + y + carry`
   - append `total % 10`
   - update `carry = total // 10`
4. Advance `l1` and `l2` when available.
5. Return `dummy.next`.

Runnable Python example (`add_two_numbers.py`):

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    carry = 0
    while l1 is not None or l2 is not None or carry:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0
        total = x + y + carry
        carry = total // 10
        tail.next = ListNode(total % 10)
        tail = tail.next
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    print(dump(add_two_numbers(build([2, 4, 3]), build([5, 6, 4]))))
```

---

## Explanation / Why This Works

The problem is already aligned with the natural digit-processing order.
So there is no hidden template beyond:

- read current digits
- add carry
- emit one new digit
- move forward

Because each input node is visited once and each output node is created once, the whole solution is linear in the input size.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: shard-wise amount accumulation (Python)

**Background**: a finance job stores low-order chunks first for streaming accumulation.  
**Why it fits**: carry propagation is exactly the same as digit-wise chunk addition.

```python
def add_chunks(a, b, base=10):
    i = j = 0
    carry = 0
    out = []
    while i < len(a) or j < len(b) or carry:
        x = a[i] if i < len(a) else 0
        y = b[j] if j < len(b) else 0
        total = x + y + carry
        carry = total // base
        out.append(total % base)
        i += 1 if i < len(a) else 0
        j += 1 if j < len(b) else 0
    return out

print(add_chunks([2, 4, 3], [5, 6, 4]))
```

### Scenario 2: backend counter-stream merge (Go)

**Background**: two services emit bucketed counts in the same low-to-high order.  
**Why it fits**: missing buckets are treated as zero and one state variable carries overflow.

```go
package main

import "fmt"

func addBuckets(a, b []int, base int) []int {
	i, j, carry := 0, 0, 0
	out := []int{}
	for i < len(a) || j < len(b) || carry != 0 {
		x, y := 0, 0
		if i < len(a) {
			x = a[i]
			i++
		}
		if j < len(b) {
			y = b[j]
			j++
		}
		total := x + y + carry
		carry = total / base
		out = append(out, total%base)
	}
	return out
}

func main() {
	fmt.Println(addBuckets([]int{2, 4, 3}, []int{5, 6, 4}, 10))
}
```

### Scenario 3: frontend offline draft version merge (JavaScript)

**Background**: a client stores version digits in little-endian order for compact reconciliation.  
**Why it fits**: the result can be built progressively with a single carry variable.

```javascript
function addVersionDigits(a, b) {
  let i = 0;
  let j = 0;
  let carry = 0;
  const out = [];
  while (i < a.length || j < b.length || carry !== 0) {
    const x = i < a.length ? a[i++] : 0;
    const y = j < b.length ? b[j++] : 0;
    const total = x + y + carry;
    carry = Math.floor(total / 10);
    out.push(total % 10);
  }
  return out;
}

console.log(addVersionDigits([2, 4, 3], [5, 6, 4]));
```

---

## R - Reflection (Complexity, Alternatives, Tradeoffs)

### Complexity

- Time: `O(max(m, n))`
- Space: `O(max(m, n))` for the output list itself
- Auxiliary space: `O(1)`

### Alternatives

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Rebuild integers first | depends on language integer size | O(1) or more | overflow risk |
| Copy to arrays then add | O(m+n) | O(m+n) | works, but unnecessary |
| One-pass linked-list simulation | O(max(m,n)) | O(1) auxiliary | best fit for problem structure |

### Common mistakes

1. Forgetting to include `carry` in the loop condition.
2. Stopping when one list ends even though the other still has nodes.
3. Treating missing digits as an error instead of as zero.
4. Trying to reverse the lists even though the problem already stores low digits first.

### Why this is the best practical method

It matches the arithmetic model exactly:

- same processing order as column addition
- no unnecessary data conversion
- minimal state
- direct handling of uneven lengths and final carry

---

## FAQ and Notes

1. **Why must the loop include `carry`?**  
   Because the final digit may exist only because of a leftover carry, such as `5 + 5 = 10`.

2. **Can we modify `l1` or `l2` in place?**  
   You can in some variants, but creating a clean result list is simpler and less error-prone.

3. **What changes if digits are stored in forward order?**  
   Then you need stacks, recursion, or list reversal first. That is a different problem shape.

---

## Best Practices

- Always write the loop as `while l1 or l2 or carry`.
- Treat absent nodes as zero digits, not as separate branches.
- Use `dummy + tail` for result construction.
- Trace one uneven-length example before coding.

---

## S - Summary

- LeetCode 2 is just grade-school addition expressed as pointer logic.
- The only cross-round state is `carry`.
- Reverse-order storage lets us solve the problem in one forward pass.
- `dummy + tail` makes output construction uniform and safe.

### Further Reading

- LeetCode 2. Add Two Numbers
- LeetCode 445. Add Two Numbers II
- LeetCode 21. Merge Two Sorted Lists
- LeetCode 206. Reverse Linked List

---

## Conclusion

If the carry update and loop condition feel natural to you after this problem, you have already internalized one of the most reusable linked-list simulation patterns in Hot100.

---

## References

- https://leetcode.com/problems/add-two-numbers/
- https://docs.python.org/3/tutorial/datastructures.html
- https://go.dev/tour/basics/11
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array

---

## Meta Info

- **Reading time**: 12-15 min
- **Tags**: Hot100, linked list, carry, simulation, LeetCode 2
- **SEO keywords**: Add Two Numbers, reverse-order digits, carry propagation, LeetCode 2, Hot100
- **Meta description**: A derivation-first ACERS guide to Add Two Numbers with carry handling, dummy node construction, and runnable multi-language code.

---

## CTA

Do two quick follow-ups:

1. Re-implement the `while l1 or l2 or carry` template from memory.
2. Then solve LeetCode 445 to see how the strategy changes when digits are stored in forward order.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

### Python

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        carry = 0

        while l1 is not None or l2 is not None or carry:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            total = x + y + carry
            carry = total // 10
            tail.next = ListNode(total % 10)
            tail = tail.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for v in nums:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    ans = Solution().addTwoNumbers(build([2, 4, 3]), build([5, 6, 4]))
    print(dump(ans))  # [7, 0, 8]
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
    struct ListNode* n = (struct ListNode*)malloc(sizeof(struct ListNode));
    n->val = v;
    n->next = NULL;
    return n;
}

struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    struct ListNode dummy = {0, NULL};
    struct ListNode* tail = &dummy;
    int carry = 0;

    while (l1 != NULL || l2 != NULL || carry != 0) {
        int x = l1 ? l1->val : 0;
        int y = l2 ? l2->val : 0;
        int total = x + y + carry;
        carry = total / 10;
        tail->next = new_node(total % 10);
        tail = tail->next;
        if (l1) l1 = l1->next;
        if (l2) l2 = l2->next;
    }

    return dummy.next;
}

struct ListNode* build(const int* a, int n) {
    struct ListNode dummy = {0, NULL};
    struct ListNode* tail = &dummy;
    for (int i = 0; i < n; ++i) {
        tail->next = new_node(a[i]);
        tail = tail->next;
    }
    return dummy.next;
}

void print_list(struct ListNode* h) {
    while (h) {
        printf("%d", h->val);
        if (h->next) printf(" -> ");
        h = h->next;
    }
    printf("\n");
}

void free_list(struct ListNode* h) {
    while (h) {
        struct ListNode* nxt = h->next;
        free(h);
        h = nxt;
    }
}

int main(void) {
    int a[] = {2, 4, 3};
    int b[] = {5, 6, 4};
    struct ListNode* l1 = build(a, 3);
    struct ListNode* l2 = build(b, 3);
    struct ListNode* ans = addTwoNumbers(l1, l2);
    print_list(ans); // 7 -> 0 -> 8
    free_list(l1);
    free_list(l2);
    free_list(ans);
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
    ListNode(int x = 0) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        int carry = 0;

        while (l1 || l2 || carry) {
            int x = l1 ? l1->val : 0;
            int y = l2 ? l2->val : 0;
            int total = x + y + carry;
            carry = total / 10;
            tail->next = new ListNode(total % 10);
            tail = tail->next;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        return dummy.next;
    }
};

ListNode* build(const vector<int>& a) {
    ListNode dummy;
    ListNode* tail = &dummy;
    for (int v : a) {
        tail->next = new ListNode(v);
        tail = tail->next;
    }
    return dummy.next;
}

void printList(ListNode* h) {
    while (h) {
        cout << h->val;
        if (h->next) cout << " -> ";
        h = h->next;
    }
    cout << '\n';
}

void freeList(ListNode* h) {
    while (h) {
        ListNode* nxt = h->next;
        delete h;
        h = nxt;
    }
}

int main() {
    ListNode* l1 = build({2, 4, 3});
    ListNode* l2 = build({5, 6, 4});
    ListNode* ans = Solution().addTwoNumbers(l1, l2);
    printList(ans); // 7 -> 0 -> 8
    freeList(l1);
    freeList(l2);
    freeList(ans);
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

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	carry := 0

	for l1 != nil || l2 != nil || carry != 0 {
		x, y := 0, 0
		if l1 != nil {
			x = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			y = l2.Val
			l2 = l2.Next
		}
		total := x + y + carry
		carry = total / 10
		tail.Next = &ListNode{Val: total % 10}
		tail = tail.Next
	}
	return dummy.Next
}

func build(a []int) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for _, v := range a {
		tail.Next = &ListNode{Val: v}
		tail = tail.Next
	}
	return dummy.Next
}

func printList(h *ListNode) {
	for h != nil {
		fmt.Print(h.Val)
		if h.Next != nil {
			fmt.Print(" -> ")
		}
		h = h.Next
	}
	fmt.Println()
}

func main() {
	l1 := build([]int{2, 4, 3})
	l2 := build([]int{5, 6, 4})
	ans := addTwoNumbers(l1, l2)
	printList(ans) // 7 -> 0 -> 8
}
```

### Rust

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

pub fn add_two_numbers(
    mut l1: Option<Box<ListNode>>,
    mut l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut digits: Vec<i32> = Vec::new();
    let mut carry = 0;

    while l1.is_some() || l2.is_some() || carry > 0 {
        let mut x = 0;
        let mut y = 0;

        if let Some(mut node) = l1 {
            x = node.val;
            l1 = node.next.take();
        } else {
            l1 = None;
        }

        if let Some(mut node) = l2 {
            y = node.val;
            l2 = node.next.take();
        } else {
            l2 = None;
        }

        let total = x + y + carry;
        carry = total / 10;
        digits.push(total % 10);
    }

    let mut head: Option<Box<ListNode>> = None;
    let mut tail = &mut head;
    for d in digits {
        *tail = Some(Box::new(ListNode::new(d)));
        if let Some(node) = tail {
            tail = &mut node.next;
        }
    }
    head
}

fn build(nums: &[i32]) -> Option<Box<ListNode>> {
    let mut head: Option<Box<ListNode>> = None;
    let mut tail = &mut head;
    for &n in nums {
        *tail = Some(Box::new(ListNode::new(n)));
        if let Some(node) = tail {
            tail = &mut node.next;
        }
    }
    head
}

fn dump(mut head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut out = Vec::new();
    while let Some(mut node) = head {
        out.push(node.val);
        head = node.next.take();
    }
    out
}

fn main() {
    let l1 = build(&[2, 4, 3]);
    let l2 = build(&[5, 6, 4]);
    let ans = add_two_numbers(l1, l2);
    println!("{:?}", dump(ans)); // [7, 0, 8]
}
```

### JavaScript

```javascript
function ListNode(val = 0, next = null) {
  this.val = val;
  this.next = next;
}

function addTwoNumbers(l1, l2) {
  const dummy = new ListNode(0);
  let tail = dummy;
  let carry = 0;

  while (l1 !== null || l2 !== null || carry !== 0) {
    const x = l1 ? l1.val : 0;
    const y = l2 ? l2.val : 0;
    const total = x + y + carry;
    carry = Math.floor(total / 10);
    tail.next = new ListNode(total % 10);
    tail = tail.next;
    if (l1) l1 = l1.next;
    if (l2) l2 = l2.next;
  }

  return dummy.next;
}

function build(arr) {
  const dummy = new ListNode();
  let tail = dummy;
  for (const v of arr) {
    tail.next = new ListNode(v);
    tail = tail.next;
  }
  return dummy.next;
}

function dump(head) {
  const out = [];
  while (head) {
    out.push(head.val);
    head = head.next;
  }
  return out;
}

const ans = addTwoNumbers(build([2, 4, 3]), build([5, 6, 4]));
console.log(dump(ans)); // [7, 0, 8]
```
