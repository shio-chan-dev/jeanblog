---
title: "Hot100: Same Tree (Synchronous Recursion / BFS ACERS Guide)"
date: 2026-03-16T13:00:54+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "DFS", "BFS", "tree comparison", "LeetCode 100"]
description: "A practical guide to LeetCode 100 covering synchronous recursion, pairwise BFS validation, structural equivalence, and runnable multi-language implementations."
keywords: ["Same Tree", "binary tree comparison", "synchronous recursion", "BFS", "LeetCode 100", "Hot100"]
---

> **Subtitle / Summary**  
> The real challenge in LeetCode 100 is not "can you traverse a tree", but "can you compare two trees node by node in lockstep". This ACERS guide explains the synchronous-recursion contract, the queue-of-pairs BFS variant, and why the pattern matters in real engineering work.

- **Reading time**: 9-11 min  
- **Tags**: `Hot100`, `binary tree`, `DFS`, `BFS`, `tree comparison`  
- **SEO keywords**: Hot100, Same Tree, binary tree comparison, synchronous recursion, BFS, LeetCode 100  
- **Meta description**: A systematic guide to LeetCode 100 from synchronous recursion to pairwise BFS comparison, with engineering scenarios and runnable multi-language implementations.  

---

## Target Readers

- Hot100 learners who want to build a stable "compare two trees together" template
- Developers who can write DFS on one tree but get confused once two trees must be checked in parallel
- Engineers who need structural-equivalence checks for config trees, component trees, or syntax trees

## Background / Motivation

When many people first see LeetCode 100, the instinct is:

- traverse tree `p`
- traverse tree `q`
- compare the two traversal results afterward

That can work only if you serialize very carefully, but it is not the core idea of the problem.

The real training value is:

- can you pull out matching nodes from `p` and `q` at the same time
- can you turn "same tree" into a precise decision contract
- can you handle null cases before touching node values

This matters far beyond one easy problem. The same pattern reappears in:

- subtree checking
- mirror and symmetry checking
- structural comparison between two tree-shaped configurations

So although LeetCode 100 is simple, it is the starting point of the **two-tree synchronous comparison template**.

## Core Concepts

- **Synchronous recursion**: the recursive function takes a pair of nodes `(p, q)`, not just one node
- **Structural equality**: matching positions must either both be null or both exist
- **Value equality**: if both nodes exist, their values must match
- **Pairwise traversal**: whether you use DFS or BFS, the unit of work is always a pair of nodes

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root nodes `p` and `q` of two binary trees, return `true` if the trees are the same.

Two trees are considered the same if:

- they have exactly the same structure
- corresponding nodes have the same values

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| p | TreeNode | root of the first binary tree |
| q | TreeNode | root of the second binary tree |
| return | `bool` | whether the two trees are exactly the same |

### Example 1

```text
input: p = [1,2,3], q = [1,2,3]
output: true
explanation:
The structures match, and every corresponding node value is equal.
```

### Example 2

```text
input: p = [1,2], q = [1,null,2]
output: false
explanation:
At the second level, one tree has a left child while the other has a right child.
The structure is different.
```

### Example 3

```text
input: p = [1,2,1], q = [1,1,2]
output: false
explanation:
The structure matches, but the values at corresponding positions do not.
```

### Constraints

- The number of nodes in both trees is in the range `[0, 100]`
- `-10^4 <= Node.val <= 10^4`

---

## C - Concepts (Core Ideas)

### Thought Process: Break "same" into four decision rules

For any pair of nodes `(p, q)`, the answer is determined by four stable checks:

1. **Both are null**: this position matches, so return `true`
2. **Exactly one is null**: the structure already differs, so return `false`
3. **Values differ**: corresponding nodes are not equal, so return `false`
4. **Both exist and values match**: continue checking
   - `p.left` with `q.left`
   - `p.right` with `q.right`

Written as a formula:

```text
same(p, q) =
    true, if p == null and q == null
    false, if exactly one is null
    false, if p.val != q.val
    same(p.left, q.left) and same(p.right, q.right), otherwise
```

### Why this is already the complete answer

The definition of "same tree" has only two pieces:

- same structure
- same values

Every recursive call checks whether the current paired position satisfies those two requirements, then reduces the problem to the left children and right children. That is the classic pattern of **local contract + same-shaped subproblem**.

### Method Category

- **Tree DFS**
- **Synchronous recursion**
- **Pairwise BFS validation**
- **Structural equivalence checking**

### Why BFS also works

If you do not want recursion, store pairs in a queue:

1. pop one pair of nodes
2. apply the same four rules as above
3. if the current pair matches, push `(left, left)` and `(right, right)` back into the queue

Nothing changes conceptually. Only the execution model changes from the call stack to an explicit queue.

---

## Practice Guide / Steps

### Recommended Approach: Synchronous recursion

1. Define a function `is_same_tree(p, q)`
2. Handle the null combinations first
3. Compare the current node values
4. Recursively compare left-left and right-right

Runnable Python example:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


if __name__ == "__main__":
    a = TreeNode(1, TreeNode(2), TreeNode(3))
    b = TreeNode(1, TreeNode(2), TreeNode(3))
    print(is_same_tree(a, b))
```

### BFS Alternative

If you prefer an explicit control flow or want to avoid recursion depth concerns:

1. keep `(p, q)` pairs in a queue
2. pop one pair at a time and compare it
3. if the pair is valid, push their children in matching directions

This style is also convenient when you want to log exactly which pair first fails.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Detect config-tree drift before release (Python)

**Background**: feature flags, permission inheritance, and routing rules are often stored as tree-shaped nested configs.  
**Why it fits**: before a rollout, teams may need to confirm that the staging config tree is exactly the same as production.

```python
def same_config(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a["name"] != b["name"]:
        return False
    return same_config(a.get("left"), b.get("left")) and same_config(a.get("right"), b.get("right"))


cfg1 = {"name": "root", "left": {"name": "A"}, "right": {"name": "B"}}
cfg2 = {"name": "root", "left": {"name": "A"}, "right": {"name": "B"}}
print(same_config(cfg1, cfg2))
```

### Scenario 2: Check component-tree snapshot equivalence (JavaScript)

**Background**: low-code editors and page builders often save UI layouts as component trees.  
**Why it fits**: regression checks need to confirm that two snapshots match in both node type and parent-child structure.

```javascript
function sameTree(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (a.type !== b.type) return false;
  return sameTree(a.left, b.left) && sameTree(a.right, b.right);
}

const oldTree = { type: "Split", left: { type: "List" }, right: { type: "Form" } };
const newTree = { type: "Split", left: { type: "List" }, right: { type: "Form" } };
console.log(sameTree(oldTree, newTree));
```

### Scenario 3: Validate AST structure after a rewrite pass (Go)

**Background**: compilers, linters, and rule engines often rewrite syntax trees.  
**Why it fits**: after a rewrite, you may want to verify that the resulting tree shape and labels match the expected output exactly.

```go
package main

import "fmt"

type Node struct {
	Label string
	Left  *Node
	Right *Node
}

func same(a, b *Node) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Label != b.Label {
		return false
	}
	return same(a.Left, b.Left) && same(a.Right, b.Right)
}

func main() {
	x := &Node{Label: "Add", Left: &Node{Label: "A"}, Right: &Node{Label: "B"}}
	y := &Node{Label: "Add", Left: &Node{Label: "A"}, Right: &Node{Label: "B"}}
	fmt.Println(same(x, y))
}
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, where `n` is the number of compared paired positions; in the worst case, every corresponding node is visited
- **Space complexity**:
  - Recursive DFS: `O(h)`, where `h` is tree height
  - BFS queue: worst-case `O(w)`, where `w` is the maximum width of a level

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Synchronous recursion | `O(n)` | `O(h)` | Most aligned with the definition and usually the best answer |
| BFS with node pairs | `O(n)` | `O(w)` | Non-recursive and easy to debug |
| Serialize then compare | `O(n)` | `O(n)` | Must encode null positions or it can give false positives |
| Hash-signature comparison | Depends on design | Extra hash storage | Useful as a quick filter in some systems, but less direct than plain comparison |

### Common Mistakes and Pitfalls

- Comparing only preorder or inorder values without encoding null positions
- Accidentally writing `same(p.left, q.right)`, which is the mirror template for LeetCode 101, not this problem
- Touching `p.val` before checking whether `p` or `q` is null
- Confusing "same value" with "same object identity in memory"

## Common Questions and Notes

### 1. Why can't we just compare traversal results?

Because different structures can produce the same value sequence.  
If you truly serialize, you must include null markers as well.

### 2. Does the problem ask whether `p` and `q` are the same object?

No. The question is about **same structure + same values**, not whether the two roots point to the same memory.

### 3. Which is more recommended, DFS or BFS?

For interviews and learning, recursion is shorter and closer to the definition. In engineering work, BFS can be more convenient if you want to log the failing comparison path or avoid deep recursion.

## Best Practices and Suggestions

- For two-tree problems, ask yourself first whether the recursive function should take two nodes
- Put null checks at the very top; it removes a large class of bugs
- Keep the concepts separate: same value, same structure, and same reference are different things
- When you see 100, 101, or 572, think "paired comparison template" immediately

## S - Summary

- The essence of LeetCode 100 is paired comparison, not separate traversal
- Once you keep the four decision rules stable, synchronous recursion becomes very reliable
- The BFS version is the same idea with an explicit queue of pairs
- Structural equivalence checks show up directly in config trees, component trees, and syntax trees
- If you can write 100 smoothly, LeetCode 101 becomes much easier next

## References and Further Reading

- [LeetCode 100: Same Tree](https://leetcode.com/problems/same-tree/)
- LeetCode 101: Symmetric Tree
- LeetCode 572: Subtree of Another Tree
- LeetCode 226: Invert Binary Tree
- LeetCode 102: Binary Tree Level Order Traversal

## CTA

Practice 100 and 101 back to back.  
100 trains **same-direction comparison**, while 101 trains **mirror-direction comparison**. Once those two templates are clear, binary-tree judgment problems become much easier to reason about.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


if __name__ == "__main__":
    p = TreeNode(1, TreeNode(2), TreeNode(3))
    q = TreeNode(1, TreeNode(2), TreeNode(3))
    print(is_same_tree(p, q))
```

```c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
};

struct TreeNode* new_node(int val) {
    struct TreeNode* node = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    node->val = val;
    node->left = NULL;
    node->right = NULL;
    return node;
}

bool isSameTree(struct TreeNode* p, struct TreeNode* q) {
    if (p == NULL && q == NULL) return true;
    if (p == NULL || q == NULL) return false;
    if (p->val != q->val) return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* p = new_node(1);
    p->left = new_node(2);
    p->right = new_node(3);

    struct TreeNode* q = new_node(1);
    q->left = new_node(2);
    q->right = new_node(3);

    printf("%s\n", isSameTree(p, q) ? "true" : "false");
    free_tree(p);
    free_tree(q);
    return 0;
}
```

```cpp
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    if (p->val != q->val) return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* p = new TreeNode(1);
    p->left = new TreeNode(2);
    p->right = new TreeNode(3);

    TreeNode* q = new TreeNode(1);
    q->left = new TreeNode(2);
    q->right = new TreeNode(3);

    std::cout << (isSameTree(p, q) ? "true" : "false") << '\n';
    freeTree(p);
    freeTree(q);
    return 0;
}
```

```go
package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

func main() {
	p := &TreeNode{Val: 1, Left: &TreeNode{Val: 2}, Right: &TreeNode{Val: 3}}
	q := &TreeNode{Val: 1, Left: &TreeNode{Val: 2}, Right: &TreeNode{Val: 3}}
	fmt.Println(isSameTree(p, q))
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

#[derive(Debug, Clone)]
struct TreeNode {
    val: i32,
    left: Node,
    right: Node,
}

impl TreeNode {
    fn new(val: i32) -> Rc<RefCell<TreeNode>> {
        Rc::new(RefCell::new(TreeNode {
            val,
            left: None,
            right: None,
        }))
    }
}

fn is_same_tree(p: &Node, q: &Node) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let a_ref = a.borrow();
            let b_ref = b.borrow();
            a_ref.val == b_ref.val
                && is_same_tree(&a_ref.left, &b_ref.left)
                && is_same_tree(&a_ref.right, &b_ref.right)
        }
        _ => false,
    }
}

fn main() {
    let p = Some(TreeNode::new(1));
    let q = Some(TreeNode::new(1));

    if let Some(root) = &p {
        root.borrow_mut().left = Some(TreeNode::new(2));
        root.borrow_mut().right = Some(TreeNode::new(3));
    }
    if let Some(root) = &q {
        root.borrow_mut().left = Some(TreeNode::new(2));
        root.borrow_mut().right = Some(TreeNode::new(3));
    }

    println!("{}", is_same_tree(&p, &q));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function isSameTree(p, q) {
  if (p === null && q === null) return true;
  if (p === null || q === null) return false;
  if (p.val !== q.val) return false;
  return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
}

const p = new TreeNode(1, new TreeNode(2), new TreeNode(3));
const q = new TreeNode(1, new TreeNode(2), new TreeNode(3));
console.log(isSameTree(p, q));
```
