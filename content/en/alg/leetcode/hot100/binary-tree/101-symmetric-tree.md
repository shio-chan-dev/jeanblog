---
title: "Hot100: Symmetric Tree (Mirror Recursion / BFS ACERS Guide)"
date: 2026-03-16T13:00:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "DFS", "BFS", "symmetry", "LeetCode 101"]
description: "A practical guide to LeetCode 101 covering mirror recursion, pairwise BFS checks, symmetry contracts, and runnable multi-language implementations."
keywords: ["Symmetric Tree", "mirror recursion", "binary tree symmetry", "BFS", "LeetCode 101", "Hot100"]
---

> **Subtitle / Summary**  
> The hard part of Symmetric Tree is not traversal itself, but comparison direction. You are not comparing left to left and right to right. You are comparing mirrored positions. This ACERS guide explains the mirror-recursion contract, the BFS queue-of-pairs variant, and real engineering cases where symmetry checking matters.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `binary tree`, `DFS`, `BFS`, `symmetry`  
- **SEO keywords**: Hot100, Symmetric Tree, mirror recursion, binary tree symmetry, BFS, LeetCode 101  
- **Meta description**: A systematic guide to LeetCode 101 from mirror recursion to pairwise BFS symmetry checks, with engineering scenarios and runnable multi-language implementations.  

---

## Target Readers

- Hot100 learners moving from Same Tree to mirror comparison
- Developers who can write ordinary tree recursion but still mix up outside and inside pairs
- Engineers who need left-right symmetry validation for layouts, topology templates, or mirrored structures

## Background / Motivation

LeetCode 101 is excellent training for directional thinking in tree problems:

- symmetry does **not** mean the left and right subtrees are identical in the same direction
- it means the left side should match the right side after mirroring
- the comparison direction changes from "same direction" to "cross direction"

Most mistakes fall into three groups:

- reusing the Same Tree logic and comparing `left.left` with `right.left`
- checking only node values while ignoring null positions
- flipping one subtree first, which adds an unnecessary transformation and makes reasoning harder

What this problem really trains is the **mirror-recursion template**. Once that clicks, symmetry, mirror, and structure-matching questions become much easier.

## Core Concepts

- **Mirror relation**: `left.left` should match `right.right`, and `left.right` should match `right.left`
- **Outside / inside pairing**: compare the outer children together and the inner children together
- **Pairwise recursion**: the helper function answers whether two positions are mirror images
- **Pairwise queueing**: in BFS, the queue stores node pairs that must be checked together

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root node `root` of a binary tree, return `true` if the tree is symmetric around its center.

A tree is symmetric if its left subtree and right subtree are mirror images of each other.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | `bool` | whether the tree is symmetric |

### Example 1

```text
input: root = [1,2,2,3,4,4,3]
output: true
explanation:
The left subtree and the right subtree match exactly after mirroring.
```

### Example 2

```text
input: root = [1,2,2,null,3,null,3]
output: false
explanation:
The right child of the left subtree and the right child of the right subtree appear on the same side.
That is not a mirror relation.
```

### Constraints

- The number of nodes is in the range `[1, 1000]`
- `-100 <= Node.val <= 100`

---

## C - Concepts (Core Ideas)

### Thought Process: Symmetry means comparing mirrored positions

Suppose you are comparing two nodes `a` and `b`. For them to be mirrors, the following must all hold:

1. **Both are null**: this mirrored position matches, so return `true`
2. **Exactly one is null**: the structure is broken, so return `false`
3. **Values differ**: the mirrored nodes do not match, so return `false`
4. **Both exist and values match**:
   - compare `a.left` with `b.right`
   - compare `a.right` with `b.left`

Written as a formula:

```text
mirror(a, b) =
    true, if a == null and b == null
    false, if exactly one is null
    false, if a.val != b.val
    mirror(a.left, b.right) and mirror(a.right, b.left), otherwise
```

### Why "left with left, right with right" is wrong here

That pattern checks equality, not symmetry.

The core difference between 100 and 101 is exactly this:

- **LeetCode 100 Same Tree**: compare same-direction positions
- **LeetCode 101 Symmetric Tree**: compare mirrored positions

If you do not switch the direction, you are solving a different problem.

### Method Category

- **Tree DFS**
- **Mirror recursion**
- **BFS with paired nodes**
- **Structural symmetry checking**

### Why BFS also fits well

You can also put mirror pairs into a queue:

1. start with `root.left` and `root.right`
2. pop a pair and apply the mirror contract
3. if the pair matches, push:
   - `left.left` with `right.right`
   - `left.right` with `right.left`

This is the same logic as recursion, only written as an explicit process.

---

## Practice Guide / Steps

### Recommended Approach: Mirror recursion

1. If `root` is null, return `true`
2. Define a helper `is_mirror(a, b)`
3. Inside the helper, keep the order: both null, one null, value mismatch, recursive mirror checks
4. Return `is_mirror(root.left, root.right)`

Runnable Python example:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return is_mirror(a.left, b.right) and is_mirror(a.right, b.left)

    return True if root is None else is_mirror(root.left, root.right)


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(2, TreeNode(4), TreeNode(3)),
    )
    print(is_symmetric(root))
```

### BFS Alternative

The non-recursive version is straightforward:

1. use a queue of mirror pairs
2. pop two nodes and compare them together
3. if they match, push the outside pair and the inside pair

This style is often easier to debug when you want to print the first non-symmetric pair explicitly.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Validate mirrored two-column layouts (JavaScript)

**Background**: visual editors often ship left-right mirrored page templates.  
**Why it fits**: before publishing a template, you may want to ensure the two sides are strict mirror images to avoid broken placements.

```javascript
function isMirror(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (a.type !== b.type) return false;
  return isMirror(a.left, b.right) && isMirror(a.right, b.left);
}

const left = { type: "Split", left: { type: "Menu" }, right: { type: "Detail" } };
const right = { type: "Split", left: { type: "Detail" }, right: { type: "Menu" } };
console.log(isMirror(left, right));
```

### Scenario 2: Check active-active topology symmetry (Python)

**Background**: some dual-site deployments require the left and right data-center templates to mirror each other in role and hierarchy.  
**Why it fits**: before rollout, symmetry checks can catch missing nodes or role drift on one side.

```python
def mirror_role(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a["role"] != b["role"]:
        return False
    return mirror_role(a.get("left"), b.get("right")) and mirror_role(a.get("right"), b.get("left"))


left_dc = {"role": "gateway", "left": {"role": "api"}, "right": {"role": "db"}}
right_dc = {"role": "gateway", "left": {"role": "db"}, "right": {"role": "api"}}
print(mirror_role(left_dc, right_dc))
```

### Scenario 3: Mirror-tree grading in an education tool (Go)

**Background**: algorithm teaching systems sometimes ask students to build a tree that mirrors a target structure.  
**Why it fits**: grading must check both node values and mirrored positions, not just a flat traversal result.

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func mirror(a, b *Node) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Val != b.Val {
		return false
	}
	return mirror(a.Left, b.Right) && mirror(a.Right, b.Left)
}

func main() {
	left := &Node{Val: 2, Left: &Node{Val: 3}, Right: &Node{Val: 4}}
	right := &Node{Val: 2, Left: &Node{Val: 4}, Right: &Node{Val: 3}}
	fmt.Println(mirror(left, right))
}
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is compared at most once
- **Space complexity**:
  - Recursive DFS: `O(h)`, where `h` is the tree height
  - BFS queue: worst-case `O(w)`, where `w` is the maximum width of a level

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Mirror recursion | `O(n)` | `O(h)` | Most aligned with the definition and usually the best answer |
| BFS with paired nodes | `O(n)` | `O(w)` | Explicit flow, easy to instrument for debugging |
| Invert one side then compare | `O(n)` | `O(h)` or `O(w)` | Adds an extra transformation and may mutate the tree |
| Serialize and compare mirror order | `O(n)` | `O(n)` | More cumbersome and still needs null markers |

### Common Mistakes and Pitfalls

- Writing LeetCode 101 as if it were LeetCode 100, still comparing `left.left` with `right.left`
- Comparing only values and ignoring null positions
- Flipping a subtree first, which is unnecessary and can introduce side effects
- Storing single nodes in the BFS queue instead of storing comparison pairs

## Common Questions and Notes

### 1. Is a single-node tree symmetric?

Yes. Its left and right subtrees are both null, so they are mirror images by definition.

### 2. Why not invert the left subtree and then compare it with the right subtree?

You can, but it is not recommended here. It adds an extra transformation step, makes the reasoning longer, and can modify the original tree.

### 3. How should I choose between DFS and BFS?

For learning and interviews, recursion is usually the clearest form. If you want explicit logging of failing pairs or want to avoid deep recursion, BFS is a good alternative.

## Best Practices and Suggestions

- Before coding, draw the outside-pair and inside-pair relation on paper once
- Memorize the template: `left.left` with `right.right`, `left.right` with `right.left`
- Practice 100 and 101 as a pair to build directional intuition quickly
- If mirror recursion still feels slippery, write the BFS pair queue once to make the pairing explicit

## S - Summary

- The core of Symmetric Tree is mirror-position comparison, not traversal by itself
- Once you keep the outside-inside mirror contract stable, the recursion becomes reliable
- The BFS version is conceptually identical; it just makes the node pairs explicit
- This problem pairs naturally with 100 and 226 when building tree-structure intuition
- In engineering work, the same idea applies to mirrored layouts, mirrored templates, and symmetric topologies

## References and Further Reading

- [LeetCode 101: Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- LeetCode 100: Same Tree
- LeetCode 226: Invert Binary Tree
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 102: Binary Tree Level Order Traversal

## CTA

Practice 100, 101, and 226 as a small bundle.  
100 trains **same-direction comparison**, 101 trains **mirror-direction comparison**, and 226 trains **structural transformation**. Together, they build strong binary-tree intuition quickly.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return is_mirror(a.left, b.right) and is_mirror(a.right, b.left)

    return True if root is None else is_mirror(root.left, root.right)


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(2, TreeNode(4), TreeNode(3)),
    )
    print(is_symmetric(root))
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

bool isMirror(struct TreeNode* a, struct TreeNode* b) {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    if (a->val != b->val) return false;
    return isMirror(a->left, b->right) && isMirror(a->right, b->left);
}

bool isSymmetric(struct TreeNode* root) {
    if (root == NULL) return true;
    return isMirror(root->left, root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->left = new_node(2);
    root->right = new_node(2);
    root->left->left = new_node(3);
    root->left->right = new_node(4);
    root->right->left = new_node(4);
    root->right->right = new_node(3);

    printf("%s\n", isSymmetric(root) ? "true" : "false");
    free_tree(root);
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

bool isMirror(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    if (a->val != b->val) return false;
    return isMirror(a->left, b->right) && isMirror(a->right, b->left);
}

bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return isMirror(root->left, root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(2);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(3);

    std::cout << (isSymmetric(root) ? "true" : "false") << '\n';
    freeTree(root);
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

func isMirror(a *TreeNode, b *TreeNode) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Val != b.Val {
		return false
	}
	return isMirror(a.Left, b.Right) && isMirror(a.Right, b.Left)
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isMirror(root.Left, root.Right)
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 3},
			Right: &TreeNode{Val: 4},
		},
		Right: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 3},
		},
	}
	fmt.Println(isSymmetric(root))
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

fn is_mirror(a: &Node, b: &Node) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            let xr = x.borrow();
            let yr = y.borrow();
            xr.val == yr.val
                && is_mirror(&xr.left, &yr.right)
                && is_mirror(&xr.right, &yr.left)
        }
        _ => false,
    }
}

fn is_symmetric(root: &Node) -> bool {
    match root {
        None => true,
        Some(node) => {
            let node_ref = node.borrow();
            is_mirror(&node_ref.left, &node_ref.right)
        }
    }
}

fn main() {
    let root = Some(TreeNode::new(1));
    if let Some(node) = &root {
        let left = Some(TreeNode::new(2));
        let right = Some(TreeNode::new(2));
        node.borrow_mut().left = left.clone();
        node.borrow_mut().right = right.clone();

        if let Some(l) = left {
            l.borrow_mut().left = Some(TreeNode::new(3));
            l.borrow_mut().right = Some(TreeNode::new(4));
        }
        if let Some(r) = right {
            r.borrow_mut().left = Some(TreeNode::new(4));
            r.borrow_mut().right = Some(TreeNode::new(3));
        }
    }
    println!("{}", is_symmetric(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function isMirror(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (a.val !== b.val) return false;
  return isMirror(a.left, b.right) && isMirror(a.right, b.left);
}

function isSymmetric(root) {
  if (root === null) return true;
  return isMirror(root.left, root.right);
}

const root = new TreeNode(
  1,
  new TreeNode(2, new TreeNode(3), new TreeNode(4)),
  new TreeNode(2, new TreeNode(4), new TreeNode(3)),
);
console.log(isSymmetric(root));
```
