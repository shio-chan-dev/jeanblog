---
title: "LeetCode 2: Add Two Numbers from Naive to Optimal Carry Simulation"
date: 2026-02-11T07:54:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["linked list", "carry", "simulation", "LeetCode 2", "algorithm walkthrough"]
description: "Add two non-negative integers represented by reverse-order linked lists. This ACERS-style guide explains digit-by-digit carry propagation with engineering mappings and runnable multi-language code."
keywords: ["Add Two Numbers", "LeetCode 2", "linked list carry", "dummy node", "reverse-order list", "carry propagation"]
---

> **Subtitle / Summary**  
> This problem is just grade-school addition on a linked list: add one digit at a time, propagate carry, and append one final node if carry remains after both lists end. We move from naive ideas to the optimal one-pass solution, then map it to real engineering scenarios.

- **Reading time**: 12-15 min  
- **Tags**: `linked list`, `carry`, `simulation`, `LeetCode 2`  
- **SEO keywords**: Add Two Numbers, LeetCode 2, reverse-order list, carry, dummy node  
- **Meta description**: Use `dummy + tail + carry` to sum two reverse-order linked lists in `O(max(m,n))` time, with common pitfalls, engineering analogies, and six-language runnable implementations.

---

## Target Readers

- Beginners building a stable template for linked-list problems
- Intermediate developers who often miss carry or boundary cases
- Engineers who want to transfer algorithmic thinking to stream-style data processing

## Background / Motivation

This looks like an entry-level LeetCode problem, but it trains practical skills you will reuse:

- Synchronous progression across multiple input streams (`l1`, `l2`)
- Cross-iteration state propagation (`carry`)
- Boundary completeness (different lengths, final carry node)

These three appear frequently in production systems: chunked amount accumulation, multi-source counter merge, and streaming aggregation with backfill.

## Core Concepts

- **Reverse-order storage**: ones digit at the head, then tens, then hundreds...
- **Digit-wise addition**: each round handles only `x + y + carry`
- **Carry propagation**: `carry = sum // 10`, current digit `sum % 10`
- **Dummy node**: avoids special handling when creating the result head

---

## A — Algorithm (Problem & Algorithm)

### Problem Restatement

You are given two **non-empty** linked lists representing two non-negative integers.  
Digits are stored in **reverse order**, and each node stores one digit.  
Return their sum as a linked list in the same reverse order.  
Except for number `0`, the input numbers do not have leading zeros.

### Input / Output

| Item | Meaning |
| --- | --- |
| Input | Two linked lists `l1`, `l2`, each node value in `0~9` |
| Output | A new linked list representing `l1 + l2` in reverse order |

### Example 1

```text
Input: l1 = [2,4,3], l2 = [5,6,4]
Explanation: 342 + 465 = 807
Output: [7,0,8]
```

### Example 2

```text
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Explanation: 9999999 + 9999 = 10009998
Output: [8,9,9,9,0,0,0,1]
```

---

## Derivation: from Naive to Optimal

### Naive idea 1: convert to integers, add, then rebuild list

- Convert lists to integers `n1`, `n2`
- Compute `n1 + n2`
- Split the result back to digits and build a new list

Problems:

- May overflow in many languages for long inputs
- Extra conversion in both directions
- Misses the essence of linked-list digit simulation

### Naive idea 2: convert to arrays first, then add by index

- Convert both lists to arrays
- Add digit by digit

Problems:

- Needs `O(m+n)` extra space
- Inputs are already low-digit-first, so the array layer is unnecessary

### Key Observation

- The lists already start from the ones digit, exactly what column addition needs
- Each round depends only on current digits and carry
- One linear pass is enough

### Method Choice

Use `dummy + tail` to build the output. Loop while:

```text
while l1 != null or l2 != null or carry != 0
```

Per iteration:

1. Read current digits `x`, `y` (treat missing node as 0)
2. `sum = x + y + carry`
3. Append node `sum % 10`
4. Update `carry = sum // 10`

---

## C — Concepts (Core Ideas)

### Method Category

- **Linked-list simulation**
- **Carry state machine**
- **Dual-pointer synchronous traversal**

### State Model

Let `x_k`, `y_k` be the digits at round `k`, and `c_k` be carry-in:

```text
s_k = x_k + y_k + c_k
digit_k = s_k mod 10
c_(k+1) = floor(s_k / 10)
```

where `c_k ∈ {0,1}`.  
This is the exact mathematical form of decimal column addition.

### Correctness Intuition

- `digit_k` is exactly the `k`-th digit of the result
- `carry` passes the overflow (>9) to the next round
- If both lists end but `carry=1`, append one final node

---

## Practice Guide / Steps

1. Initialize `dummy`, `tail`, and `carry = 0`
2. Loop while either list remains or carry is non-zero
3. Read current values: `x = l1.val if l1 else 0`, `y = l2.val if l2 else 0`
4. Compute `sum`, append `sum % 10`
5. Update `carry = sum // 10`, move `tail` and input pointers
6. Return `dummy.next`

Minimal runnable Python example:

```python
from typing import Optional, List


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
        s = x + y + carry
        carry = s // 10
        tail.next = ListNode(s % 10)
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

---

## E — Engineering (Real-world Mapping)

### Scenario 1: chunked amount merge in finance (Python)

**Background**: some systems store or transmit very large amounts in chunks.  
**Why this maps well**: each chunk behaves like one digit group; the core is still same-position add + carry propagation.

```python
def add_digits(a, b):
    i = j = 0
    carry = 0
    out = []
    while i < len(a) or j < len(b) or carry:
        x = a[i] if i < len(a) else 0
        y = b[j] if j < len(b) else 0
        s = x + y + carry
        out.append(s % 10)
        carry = s // 10
        i += 1
        j += 1
    return out


print(add_digits([2, 4, 3], [5, 6, 4]))  # [7,0,8]
```

### Scenario 2: merge low-digit-first counters from multiple services (Go)

**Background**: two backend services report low-digit-first counter blocks.  
**Why this maps well**: digit-wise merge is stream-friendly and memory-stable.

```go
package main

import "fmt"

func addDigits(a, b []int) []int {
	i, j, carry := 0, 0, 0
	out := make([]int, 0)
	for i < len(a) || j < len(b) || carry > 0 {
		x, y := 0, 0
		if i < len(a) {
			x = a[i]
			i++
		}
		if j < len(b) {
			y = b[j]
			j++
		}
		s := x + y + carry
		out = append(out, s%10)
		carry = s / 10
	}
	return out
}

func main() {
	fmt.Println(addDigits([]int{9, 9, 9}, []int{1})) // [0 0 0 1]
}
```

### Scenario 3: offline draft version increment in frontend (JavaScript)

**Background**: offline editors may split very long version numbers into digits/chunks.  
**Why this maps well**: browser-side processing avoids dependency on big-integer libraries.

```javascript
function addDigits(a, b) {
  let i = 0;
  let j = 0;
  let carry = 0;
  const out = [];

  while (i < a.length || j < b.length || carry) {
    const x = i < a.length ? a[i++] : 0;
    const y = j < b.length ? b[j++] : 0;
    const s = x + y + carry;
    out.push(s % 10);
    carry = Math.floor(s / 10);
  }
  return out;
}

console.log(addDigits([2, 4, 3], [5, 6, 4])); // [7,0,8]
```

---

## R — Reflection (Deep Dive)

### Complexity

- Time: `O(max(m, n))`
- Space: `O(max(m, n))` for result list; auxiliary space is `O(1)`

### Alternative Comparison

| Approach | Time | Extra Space | Problem |
| --- | --- | --- | --- |
| Convert to integers then add | O(m+n) | depends on big-int | overflow risk or big-int dependency |
| Convert to arrays then add | O(m+n) | O(m+n) | unnecessary middle layer |
| One-pass list simulation (this) | O(max(m,n)) | O(1) auxiliary | clear boundaries, production-friendly |

### Common Mistakes

- Forgetting `carry != 0` in the loop condition, so `999 + 1` loses the last digit
- Dereferencing null when list lengths differ
- Over-optimizing in-place reuse of input lists and making code branches hard to reason about

### Why this method is optimal and practical

- Single pass and direct mapping to decimal addition
- No dependency on language big-integer support
- Unified boundary handling, easy to test and port

---

## FAQ

### Q1: Why must the loop condition include `carry`?

Because both lists may end while a carry is still pending. Example: `5 + 5 = 10` still needs one more node `1`.

### Q2: Can we modify `l1` or `l2` in place?

Possible, but usually not worth it: branch complexity increases and you may break caller expectations about input reuse. In interviews and production code, building a new result list is cleaner.

### Q3: What if digits are stored in forward order?

That is a different problem (LeetCode 445). Typical solutions use stack or recursion from high digit to low digit, unlike this low-digit-first model.

---

## Best Practices

- Keep the stable template: `dummy + tail + carry`
- Use `while l1 or l2 or carry` to unify boundaries
- Always test three cases: same length, different lengths, all-carry chain
- Keep “missing node means 0” in one place to avoid scattered null checks

---

## S — Summary

Key takeaways:

1. Reverse-order list addition is a decimal digit state machine.
2. `carry` is cross-round state and must be included in loop condition.
3. A dummy node removes fragile head-special-case logic.
4. This is a foundational template for linked-list simulation and boundary management.
5. The same model transfers well to chunked numeric merge and streaming counters.

Recommended follow-ups:

- LeetCode 445 `Add Two Numbers II` (forward-order digits)
- LeetCode 21 `Merge Two Sorted Lists` (dual-pointer list template)
- LeetCode 206 `Reverse Linked List` (core linked-list operations)
- CLRS chapters on linked lists and basic data structures

---

## Runnable Multi-language Implementations

### Python

```python
from typing import Optional, List


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
            s = x + y + carry
            carry = s // 10
            tail.next = ListNode(s % 10)
            tail = tail.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    d = ListNode()
    t = d
    for v in nums:
        t.next = ListNode(v)
        t = t.next
    return d.next


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
    struct ListNode dummy;
    dummy.val = 0;
    dummy.next = NULL;
    struct ListNode* tail = &dummy;
    int carry = 0;

    while (l1 != NULL || l2 != NULL || carry != 0) {
        int x = (l1 != NULL) ? l1->val : 0;
        int y = (l2 != NULL) ? l2->val : 0;
        int s = x + y + carry;
        carry = s / 10;

        tail->next = new_node(s % 10);
        tail = tail->next;

        if (l1 != NULL) l1 = l1->next;
        if (l2 != NULL) l2 = l2->next;
    }

    return dummy.next;
}

struct ListNode* build(const int* a, int n) {
    struct ListNode dummy;
    dummy.next = NULL;
    struct ListNode* tail = &dummy;
    for (int i = 0; i < n; i++) {
        tail->next = new_node(a[i]);
        tail = tail->next;
    }
    return dummy.next;
}

void print_list(struct ListNode* h) {
    while (h != NULL) {
        printf("%d", h->val);
        if (h->next != NULL) printf(" -> ");
        h = h->next;
    }
    printf("\n");
}

void free_list(struct ListNode* h) {
    while (h != NULL) {
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
            int s = x + y + carry;
            carry = s / 10;
            tail->next = new ListNode(s % 10);
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
		s := x + y + carry
		carry = s / 10
		tail.Next = &ListNode{Val: s % 10}
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

        let s = x + y + carry;
        carry = s / 10;
        digits.push(s % 10);
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
    const s = x + y + carry;
    carry = Math.floor(s / 10);

    tail.next = new ListNode(s % 10);
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

---

## CTA

If you often get stuck on boundary conditions in this problem, do these two drills right now:

1. Re-implement `while l1 or l2 or carry` from memory without looking.
2. Then solve LeetCode 445 and compare forward-order vs reverse-order addition.

You can also continue with LeetCode 25 or LeetCode 142 to strengthen linked-list pointer fundamentals.
