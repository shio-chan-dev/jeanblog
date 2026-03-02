---
title: "LeetCode 138: Copy List with Random Pointer — A Complete Deep-Copy Breakdown"
date: 2026-02-11T07:59:32+08:00
draft: false
categories: ["LeetCode"]
tags: ["Linked List", "Hash Table", "Deep Copy", "Random Pointer", "LeetCode 138"]
description: "The core of copying a random-pointer list is mapping original node identity to copied node identity, then rebuilding next/random pointers. This article uses the ACERS structure to cover intuition, engineering analogies, pitfalls, and runnable multi-language implementations."
keywords: ["Copy List with Random Pointer", "Random Pointer List Copy", "Deep Copy", "Hash Mapping", "LeetCode 138"]
---

> **Subtitle / Abstract**  
> The real challenge in this problem is not traversing the list, but correctly cloning the cross-node reference relationships created by `random` pointers. This article moves from naive intuition to a hash-mapping solution, and explains why it is stable, maintainable, and practical in real engineering.

- **Estimated reading time**: 12–16 minutes  
- **Tags**: `Linked List`, `Deep Copy`, `Hash Table`, `Random Pointer`  
- **SEO keywords**: LeetCode 138, Copy List with Random Pointer, random list copy, deep copy, hash mapping  
- **Meta description**: Perform deep copy of a random-pointer linked list via two passes plus a mapping table, with correctness, complexity, engineering practice, and six-language implementations.  

---

## Target Readers

- Developers who feel shaky on `random` pointer problems while practicing LeetCode
- Learners who want to clearly understand "shallow copy vs deep copy"
- Engineers who want to transfer algorithmic thinking to real object-copy scenarios

## Background / Motivation

For a normal linked list, copying `val` and `next` is straightforward.  
A random-pointer list adds one more pointer, `random`, which can:

- Point to any node (earlier node, later node, or itself)
- Or be `null`

That turns the problem from "linear copy" into "structure copy with extra references."  
Common engineering equivalents include:

- Copying workflow node objects while preserving cross-step jump relationships
- Copying cached object graphs while keeping internal references consistent
- Copying session chains while preserving backtracking / shortcut references

## Core Concepts

- **Shallow Copy**: copies only the node shell; internal references still point to old objects
- **Deep Copy**: rebuilds a full object graph; all references point to new objects
- **Node Identity Mapping**: `old_node -> new_node`, the key to rebuilding `random`
- **Structural Equivalence**: the new list is isomorphic to the old one in values and pointer relations, while sharing no nodes

---

## A — Algorithm (Problem and Algorithm)

### Problem Restatement

Given a linked list of length `n`, each node has:

- `val`
- `next`
- `random` (can point to any node or `null`)

Construct a **deep copy** of this list and return the new head node.  
No pointer in the new list may point to any node in the original list.

### Input / Output Representation

The problem statement often uses `[val, random_index]` to represent each node:

- `val`: node value
- `random_index`: index of the node pointed to by `random`; `null` if empty

Your function input is only `head`, and your output is the copied list head.

### Example 1

```text
Input: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Explanation: The output has the same value/reference structure as the input, but all nodes are newly created objects.
```

### Example 2

```text
Input: [[1,1],[2,1]]
Output: [[1,1],[2,1]]
Explanation: The first node's random points to the second node, and the second node's random points to itself.
```

---

## Thought Process: From Naive to Maintainable Solution

### Naive Pitfall: handling `random` immediately during traversal

If you try to set `new.random` when first visiting a node, you hit this issue:

- The target node of `random` may not have been copied yet
- You need repeated backfilling, which increases branching complexity and risks missing edge cases

### Key Observation

`random` cannot be rebuilt correctly without a node-identity mapping.  
Once `old -> new` mapping exists, all pointer reconstruction becomes simple lookup operations.

### Method Selection: two passes + hash mapping

1. First pass: copy node values and build mapping `map[old] = new`
2. Second pass: rebuild `next` and `random` from that mapping

Advantages of this approach:

- Intuitive and easy to debug
- Easy to prove correctness
- Maintainable in both interviews and production code

---

## C — Concepts (Core Ideas)

### Algorithm Classification

- **Linked-list traversal**
- **Hash mapping (object identity mapping)**
- **Graph copy (special graph: each node has at most two outgoing edges)**

### Conceptual Model

Treat the list as a directed graph:

- Node set: `V`
- Edge set: `E = {next edges, random edges}`

The copy target is an isomorphic graph `G'`, satisfying:

- `val(v') = val(v)`
- `f(next(v)) = next(f(v))`
- `f(random(v)) = random(f(v))`

where `f` is the mapping `old -> new`.

### Correctness Highlights (Brief)

- After pass one, each old node `u` has a unique copied node `f(u)`
- In pass two, for each edge `u -> v`, set `f(u).ptr = f(v)` (`v` may be null)
- Because each `next/random` edge is rewired via `f`, the copied structure is fully equivalent and contains no leaked references to old nodes

---

## Practical Guide / Steps

1. Handle empty list first: if `head == null`, return `null`
2. First pass: create a copied node for every old node and store it in mapping
3. Second pass: set `next` and `random` for each copied node
4. Return `map[head]`

Runnable Python example:

```python
from typing import Optional, List


class Node:
    def __init__(self, x: int, next: Optional["Node"] = None, random: Optional["Node"] = None):
        self.val = x
        self.next = next
        self.random = random


def copy_random_list(head: Optional[Node]) -> Optional[Node]:
    if head is None:
        return None

    mp = {}
    cur = head
    while cur is not None:
        mp[cur] = Node(cur.val)
        cur = cur.next

    cur = head
    while cur is not None:
        mp[cur].next = mp.get(cur.next)
        mp[cur].random = mp.get(cur.random)
        cur = cur.next

    return mp[head]


def build(arr: List[List[Optional[int]]]) -> Optional[Node]:
    if not arr:
        return None
    nodes = [Node(v) for v, _ in arr]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, (_, r) in enumerate(arr):
        nodes[i].random = nodes[r] if r is not None else None
    return nodes[0]


def dump(head: Optional[Node]) -> List[List[Optional[int]]]:
    out = []
    idx = {}
    cur, i = head, 0
    while cur is not None:
        idx[cur] = i
        cur = cur.next
        i += 1
    cur = head
    while cur is not None:
        out.append([cur.val, idx.get(cur.random)])
        cur = cur.next
    return out


if __name__ == "__main__":
    data = [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]
    src = build(data)
    cp = copy_random_list(src)
    print(dump(cp))
```

---

## Code / Test Cases / Test Results

### Code Highlights

- Two passes: create nodes in pass one, connect edges in pass two
- `map.get(None) == None` (Python) reduces explicit null-check branches

### Test Cases

```text
Case 1: []
Expected: []

Case 2: [[1,null]]
Expected: [[1,null]]

Case 3: [[1,0]]
Expected: [[1,0]]  (self-pointing random)

Case 4: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Expected: same structure after copy
```

### Test Results (Sample)

```text
All tests passed: structure is equivalent, and node addresses in the copied list are completely different from the original list.
```

---

## E — Engineering (Engineering Applications)

### Scenario 1: Deep copy of workflow definitions (Python)

**Background**: workflow nodes have sequential `next`, and may also include jump references (similar to `random`).  
**Why it fits**: when cloning a template into a new workflow, jump relationships must be preserved without contaminating the original template.

```python
class Step:
    def __init__(self, name):
        self.name = name
        self.next = None
        self.jump = None


def copy_steps(head):
    if not head:
        return None
    mp = {}
    cur = head
    while cur:
        mp[cur] = Step(cur.name)
        cur = cur.next
    cur = head
    while cur:
        mp[cur].next = mp.get(cur.next)
        mp[cur].jump = mp.get(cur.jump)
        cur = cur.next
    return mp[head]
```

### Scenario 2: Backend task-chain copy (Go)

**Background**: task nodes execute linearly, but can jump back to a compensation node on failure.  
**Why it fits**: failure-jump relationships are fundamentally `random` references and must be reconstructed during copying.

```go
package main

import "fmt"

type Task struct {
	Name   string
	Next   *Task
	Backup *Task
}

func copyTasks(head *Task) *Task {
	if head == nil {
		return nil
	}
	mp := map[*Task]*Task{}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur] = &Task{Name: cur.Name}
	}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur].Next = mp[cur.Next]
		mp[cur].Backup = mp[cur.Backup]
	}
	return mp[head]
}

func main() {
	a := &Task{Name: "A"}
	b := &Task{Name: "B"}
	a.Next = b
	b.Backup = b
	cp := copyTasks(a)
	fmt.Println(cp.Name, cp.Next.Name, cp.Next.Backup == cp.Next) // A B true
}
```

### Scenario 3: Frontend editor-history chain copy (JavaScript)

**Background**: editor history usually has a linear chain plus references for quick-jump key versions.  
**Why it fits**: when switching user sessions, copying history chains avoids cross-session object-reference contamination.

```javascript
class Version {
  constructor(id) {
    this.id = id;
    this.next = null;
    this.jump = null;
  }
}

function copyVersions(head) {
  if (!head) return null;
  const mp = new Map();
  for (let cur = head; cur; cur = cur.next) mp.set(cur, new Version(cur.id));
  for (let cur = head; cur; cur = cur.next) {
    mp.get(cur).next = mp.get(cur.next) || null;
    mp.get(cur).jump = mp.get(cur.jump) || null;
  }
  return mp.get(head);
}
```

---

## R — Reflection (Reflection and Depth)

### Complexity Analysis

- Time complexity: `O(n)` (two linear passes)
- Space complexity: `O(n)` (mapping table)

### Alternative Approach Comparison

| Approach | Time | Extra Space | Evaluation |
| --- | --- | --- | --- |
| Two-pass hash mapping (this article) | O(n) | O(n) | Easiest to write, most stable, high maintainability |
| Interleaving-list method (insert copies in-place, then split) | O(n) | O(1) | Better space usage, but more implementation details |
| Serialize + deserialize | Usually > O(n) | Depends on format | Possible in engineering, but not ideal for core interview evaluation |

### Common Incorrect Approaches

- Copying only `val/next` and forgetting `random`
- Accidentally pointing copied `random` back to original nodes
- Using old-node pointers directly in second-pass rewiring instead of mapped new nodes
- Forgetting to handle `head == null`

### Why this method is more practical in engineering

- Clear logical layering (node creation and edge rewiring are separated)
- Easy to debug (check mapping scale first, then pointer connections)
- Team-friendly and easier for newcomers to maintain quickly

---

## Frequently Asked Questions and Notes (FAQ)

### Q1: Why can this be viewed as a graph-copy problem?

Because each node has two edge types, `next` and `random`; what we copy is the full node-edge relationship, not just linear list order.

### Q2: Can it be done in one pass?

It is possible in theory, but code complexity and bug risk rise significantly. In interviews and engineering, the two-pass hash-mapping version is recommended.

### Q3: Is a hash table required?

Not strictly required. If you pursue `O(1)` extra space, you can use the interleaving-list method, but readability is usually worse than hash mapping.

---

## Best Practices and Recommendations

- Separate "copy nodes" and "rebuild pointers" into two phases to avoid state confusion
- Use node object identity as mapping key, not node values
- Cover these regression cases: empty list, self-pointing `random`, cross-pointing `random`, tail node with `random = null`
- For debugging output, `[val, random_index]` is usually more intuitive than raw addresses

---

## S — Summary (Summary)

Key takeaways:

1. This problem is essentially "object identity mapping + pointer rewiring," not ordinary linear list copy.
2. The two-pass approach splits the problem into node creation and edge rewiring, improving both correctness and maintainability.
3. Correct `random` reconstruction depends on a complete `old -> new` mapping.
4. Hash mapping is an extremely stable engineering baseline and the clearest way to explain the solution in interviews.
5. Once understood, this pattern transfers naturally to graph copy, workflow clone, and object-graph duplication scenarios.

Recommended follow-up reading:

- LeetCode 133 `Clone Graph`
- LeetCode 146 `LRU Cache` (hash mapping + linked-list coordination)
- LeetCode 21 / 206 (fundamental linked-list drills)
- *Designing Data-Intensive Applications* sections on object relationships and data copying

---

## Multi-language Runnable Implementations

### Python

```python
from typing import Optional


class Node:
    def __init__(self, x: int, next: Optional["Node"] = None, random: Optional["Node"] = None):
        self.val = x
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if head is None:
            return None

        mp = {}
        cur = head
        while cur is not None:
            mp[cur] = Node(cur.val)
            cur = cur.next

        cur = head
        while cur is not None:
            mp[cur].next = mp.get(cur.next)
            mp[cur].random = mp.get(cur.random)
            cur = cur.next

        return mp[head]
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int val;
    struct Node* next;
    struct Node* random;
};

struct Node* new_node(int v) {
    struct Node* n = (struct Node*)malloc(sizeof(struct Node));
    n->val = v;
    n->next = NULL;
    n->random = NULL;
    return n;
}

// Interleaving-list method: O(n) time, O(1) extra space
struct Node* copyRandomList(struct Node* head) {
    if (head == NULL) return NULL;

    struct Node* cur = head;
    while (cur != NULL) {
        struct Node* cp = new_node(cur->val);
        cp->next = cur->next;
        cur->next = cp;
        cur = cp->next;
    }

    cur = head;
    while (cur != NULL) {
        struct Node* cp = cur->next;
        cp->random = (cur->random != NULL) ? cur->random->next : NULL;
        cur = cp->next;
    }

    struct Node* new_head = head->next;
    cur = head;
    while (cur != NULL) {
        struct Node* cp = cur->next;
        cur->next = cp->next;
        cp->next = (cp->next != NULL) ? cp->next->next : NULL;
        cur = cur->next;
    }
    return new_head;
}

void print_list(struct Node* head) {
    struct Node* arr[128];
    int n = 0;
    for (struct Node* p = head; p != NULL; p = p->next) arr[n++] = p;
    for (int i = 0; i < n; i++) {
        int r = -1;
        for (int j = 0; j < n; j++) {
            if (arr[i]->random == arr[j]) {
                r = j;
                break;
            }
        }
        if (r >= 0) printf("[%d,%d] ", arr[i]->val, r);
        else printf("[%d,null] ", arr[i]->val);
    }
    printf("\n");
}

int main(void) {
    struct Node* a = new_node(1);
    struct Node* b = new_node(2);
    a->next = b;
    a->random = b;
    b->random = b;
    struct Node* cp = copyRandomList(a);
    print_list(cp); // [1,1] [2,1]
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <unordered_map>

using namespace std;

class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) : val(_val), next(nullptr), random(nullptr) {}
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        unordered_map<Node*, Node*> mp;
        for (Node* cur = head; cur; cur = cur->next) {
            mp[cur] = new Node(cur->val);
        }
        for (Node* cur = head; cur; cur = cur->next) {
            mp[cur]->next = cur->next ? mp[cur->next] : nullptr;
            mp[cur]->random = cur->random ? mp[cur->random] : nullptr;
        }
        return mp[head];
    }
};
```

### Go

```go
package main

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	mp := map[*Node]*Node{}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur] = &Node{Val: cur.Val}
	}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur].Next = mp[cur.Next]
		mp[cur].Random = mp[cur.Random]
	}
	return mp[head]
}
```

### Rust

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    val: i32,
    next: Option<Rc<RefCell<Node>>>,
    random: Option<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(val: i32) -> Self {
        Self { val, next: None, random: None }
    }
}

fn copy_random_list(head: Option<Rc<RefCell<Node>>>) -> Option<Rc<RefCell<Node>>> {
    let start = head.clone()?;
    let mut mp: HashMap<*const RefCell<Node>, Rc<RefCell<Node>>> = HashMap::new();

    let mut cur = head.clone();
    while let Some(node_rc) = cur {
        let ptr = Rc::as_ptr(&node_rc);
        let val = node_rc.borrow().val;
        mp.insert(ptr, Rc::new(RefCell::new(Node::new(val))));
        cur = node_rc.borrow().next.clone();
    }

    cur = head;
    while let Some(node_rc) = cur {
        let old_ptr = Rc::as_ptr(&node_rc);
        let new_node = mp.get(&old_ptr).unwrap().clone();

        let next_old = node_rc.borrow().next.clone();
        let random_old = node_rc.borrow().random.clone();

        {
            let mut nm = new_node.borrow_mut();
            nm.next = next_old
                .as_ref()
                .and_then(|x| mp.get(&Rc::as_ptr(x)).cloned());
            nm.random = random_old
                .as_ref()
                .and_then(|x| mp.get(&Rc::as_ptr(x)).cloned());
        }

        cur = next_old;
    }

    mp.get(&Rc::as_ptr(&start)).cloned()
}

fn main() {
    let n1 = Rc::new(RefCell::new(Node::new(1)));
    let n2 = Rc::new(RefCell::new(Node::new(2)));
    n1.borrow_mut().next = Some(n2.clone());
    n1.borrow_mut().random = Some(n2.clone());
    n2.borrow_mut().random = Some(n2.clone());

    let cp = copy_random_list(Some(n1)).unwrap();
    println!("{}", cp.borrow().val); // 1
}
```

### JavaScript

```javascript
function Node(val, next = null, random = null) {
  this.val = val;
  this.next = next;
  this.random = random;
}

function copyRandomList(head) {
  if (head === null) return null;

  const mp = new Map();
  for (let cur = head; cur !== null; cur = cur.next) {
    mp.set(cur, new Node(cur.val));
  }
  for (let cur = head; cur !== null; cur = cur.next) {
    mp.get(cur).next = cur.next ? mp.get(cur.next) : null;
    mp.get(cur).random = cur.random ? mp.get(cur.random) : null;
  }
  return mp.get(head);
}
```

---

## Call to Action (CTA)

I recommend doing these two reinforcement steps right now:

1. Write the two-pass hash-mapping solution once from memory and pass your own tests.
2. Then tackle `LeetCode 133 Clone Graph` to transfer identity-mapping copy logic to a more general graph structure.

If you want, I can write the next article on `LeetCode 146 LRU Cache`, extending from "hash + linked list" in copy problems to cache-eviction design.
