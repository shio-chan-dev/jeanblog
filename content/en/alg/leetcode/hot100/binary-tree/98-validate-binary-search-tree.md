---
title: "Hot100: Validate Binary Search Tree (Range Constraints / Inorder ACERS Guide)"
date: 2026-04-19T15:13:19+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "BST", "DFS", "inorder traversal", "LeetCode 98"]
description: "A practical guide to LeetCode 98 covering ancestor range constraints, why parent-only checks fail, and two stable solutions: recursive bounds and inorder validation."
keywords: ["Validate Binary Search Tree", "BST", "range constraints", "inorder traversal", "DFS", "LeetCode 98", "Hot100"]
---

> **Subtitle / Summary**
> The hardest part of LeetCode 98 is usually not recursion itself. It is realizing that checking a node against only its parent is not enough. A valid BST is constrained by all of its ancestors, so the stable solution is to pass a valid range downward. This guide explains that idea from scratch, then connects it to the equivalent inorder-sorted view.

- **Reading time**: 11-13 min
- **Tags**: `Hot100`, `binary tree`, `BST`, `DFS`, `inorder traversal`
- **SEO keywords**: Validate Binary Search Tree, BST, range constraints, inorder traversal, DFS, LeetCode 98
- **Meta description**: Learn LeetCode 98 from the core invariant `low < val < high`, with step-by-step derivation, engineering mappings, and runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given the root `root` of a binary tree, determine whether it is a valid binary search tree (BST).

A valid BST must satisfy all of the following:

- all values in the left subtree are strictly smaller than the current node
- all values in the right subtree are strictly greater than the current node
- both subtrees must also be valid BSTs

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | bool | whether the tree is a valid BST |

### Example 1

```text
input: root = [2,1,3]
output: true
```

### Example 2

```text
input: root = [5,1,4,null,null,3,6]
output: false
explanation: the root value is 5, but the right child value is 4.
```

### Constraints

- The number of nodes in the tree is in the range `[1, 10^4]`
- `-2^31 <= Node.val <= 2^31 - 1`

---

## Target Readers

- Hot100 learners who want a stable BST validation template
- Developers who can write tree recursion but still confuse local checks with global constraints
- Engineers working with ordered tree-like structures, rule hierarchies, or range-constrained node systems

## Background / Motivation

This problem looks simple at first:

- compare the current node with its left child
- compare the current node with its right child
- recurse

But that approach is wrong.

The real issue is that BST validity is not a local parent-child rule.
It is a **subtree-wide constraint inherited from ancestors**.

That is why this problem is valuable:

- it teaches when local checks are insufficient
- it teaches how to carry ancestor constraints downward
- it gives you a clean mental model that later helps with BST search, k-th smallest, recovery, and tree invariants in general

## Core Concepts

- **Ancestor range constraint**: the current node must fit inside a range determined by all ancestors, not only its parent
- **Open interval `(low, high)`**: the current value must satisfy `low < val < high`
- **Range propagation**: the left subtree tightens the upper bound, and the right subtree tightens the lower bound
- **Inorder monotonicity**: an inorder traversal of a valid BST must be strictly increasing

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the counterexample that breaks parent-only checks

Look at the official counterexample:

```text
root = [5,1,4,null,null,3,6]
```

Once the root is `5`, the entire right subtree must be strictly greater than `5`.
So the node `4` is already illegal.

The key observation is:

- after you move into the right subtree
- you still have to remember the lower bound imposed by the ancestor `5`

That means comparing only with the direct parent is not enough.

#### Step 2: Decide what information a node must inherit

If recursion reaches some node, what minimum information do we need?

Two values:

- `low`: the smallest allowed value
- `high`: the largest allowed value

So the recursive shape becomes:

```python
def dfs(node, low, high):
    ...
```

The meaning is simple:

> check whether the subtree rooted at `node` is fully valid inside the open interval `(low, high)`.

#### Step 3: Define the recursive subproblem

Now the original problem becomes:

> Is this subtree valid, given the range it is allowed to occupy?

That is a stable recursive definition, because every node in a BST must satisfy a valid range, and that range narrows as we go down.

#### Step 4: Define the base case

If the current node is null, there is no violation inside this subtree.

```python
if node is None:
    return True
```

An empty tree is a valid BST.

#### Step 5: Check the current node against the inherited range

Before checking children, verify whether the current value fits:

```python
if node.val <= low or node.val >= high:
    return False
```

The inequality must be strict.
BSTs in this problem do not allow duplicates on the boundary.

#### Step 6: Pass tighter bounds to the children

Once the current node value is fixed:

- the left subtree must fit inside `(low, node.val)`
- the right subtree must fit inside `(node.val, high)`

So the recursive step is:

```python
return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
```

This is the core invariant of the whole problem.

#### Step 7: Why there is no backtracking state to undo

Unlike combination or permutation problems, we are not maintaining a mutable path.

There is no:

- `path`
- `used`
- state restoration

Each recursive call simply receives a narrower valid range as a parameter.
The state is functional and flows downward.

#### Step 8: Walk the bad example slowly

Again use:

```text
root = [5,1,4,null,null,3,6]
```

At the root:

- node = `5`
- valid range is `(-inf, +inf)`
- valid

Move to the left child `1`:

- valid range is `(-inf, 5)`
- valid

Move to the right child `4`:

- valid range is `(5, +inf)`

Now the failure is immediate:

```text
4 <= 5
```

So the tree is invalid without needing any deeper recursion.

#### Step 9: Connect this to inorder traversal

There is another equivalent way to validate a BST:

> its inorder traversal must be strictly increasing.

So you can also:

1. do an inorder traversal
2. keep the previously visited value `prev`
3. return `False` once the current value is `<= prev`

That is also correct.
But the range-constraint recursion is the better derivation-first explanation because it matches the actual definition of a BST.

### Assemble the Full Code

Now combine the fragments above into the first complete working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root):
    def dfs(node, low, high):
        if node is None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root, float("-inf"), float("inf"))


if __name__ == "__main__":
    root = TreeNode(2, TreeNode(1), TreeNode(3))
    print(is_valid_bst(root))
```

### Reference Answer

For LeetCode submission style, the same logic becomes:

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], low: float, high: float) -> bool:
            if node is None:
                return True
            if node.val <= low or node.val >= high:
                return False
            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

        return dfs(root, float("-inf"), float("inf"))
```

### What mental model should stick?

Do not remember this as "that BST template problem".
Remember the invariant:

> Every node must satisfy the range imposed by all of its ancestors, not only by its parent.

Once that sentence is clear, the code follows naturally.

---

## E — Engineering

### Scenario 1: Validate range-constrained rule trees (Python)

**Background**: some risk or quota systems organize thresholds as trees, where every child rule must stay inside an ancestor-defined range.  
**Why it fits**: this is structurally the same as BST validation with inherited bounds.

```python
class RuleNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def validate_rule_tree(root, low=float("-inf"), high=float("inf")):
    if root is None:
        return True
    if root.value <= low or root.value >= high:
        return False
    return validate_rule_tree(root.left, low, root.value) and validate_rule_tree(root.right, root.value, high)


tree = RuleNode(50, RuleNode(20), RuleNode(80))
print(validate_rule_tree(tree))
```

### Scenario 2: Sanity-check deserialized index trees (Go)

**Background**: a service may restore an ordered tree structure from disk or remote storage during startup.  
**Why it fits**: checking only direct children is not enough; you need full ancestor-bound validation.

```go
package main

import "fmt"

type Node struct {
	Key   int
	Left  *Node
	Right *Node
}

func validate(root *Node, low, high int64) bool {
	if root == nil {
		return true
	}
	v := int64(root.Key)
	if v <= low || v >= high {
		return false
	}
	return validate(root.Left, low, v) && validate(root.Right, v, high)
}

func main() {
	root := &Node{Key: 20, Left: &Node{Key: 10}, Right: &Node{Key: 30}}
	fmt.Println(validate(root, -1<<63, 1<<63-1))
}
```

### Scenario 3: Check ordering constraints in frontend rule trees (JavaScript)

**Background**: a frontend config system may represent priority or routing rules as a tree.  
**Why it fits**: if one child crosses an ancestor bound, the configuration order is inconsistent.

```javascript
function Node(priority, left = null, right = null) {
  this.priority = priority;
  this.left = left;
  this.right = right;
}

function validate(node, low = -Infinity, high = Infinity) {
  if (!node) return true;
  if (node.priority <= low || node.priority >= high) return false;
  return validate(node.left, low, node.priority) && validate(node.right, node.priority, high);
}

const root = new Node(10, new Node(5), new Node(15));
console.log(validate(root));
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)`, because every node is visited once
- **Space complexity**: `O(h)`, where `h` is the tree height, due to recursion stack depth

### Alternative Approaches

| Method | Time | Extra space | Notes |
| --- | --- | --- | --- |
| Range-constraint recursion | `O(n)` | `O(h)` | most direct and definition-first |
| Inorder strictly increasing check | `O(n)` | `O(h)` | equally valid and very stable |
| Parent-only comparison | looks like `O(n)` | `O(h)` | wrong, because it misses ancestor constraints |

### Common Mistakes

1. **Only comparing each node with its parent**: this misses violations created by higher ancestors.  
2. **Using non-strict inequalities incorrectly**: BST validity here requires strict ordering.  
3. **Using an `int` boundary sentinel carelessly**: the node value range already reaches 32-bit extremes, so wider bounds are safer.  
4. **Losing `prev` in the inorder method**: if `prev` is not maintained correctly across recursion, the inorder check becomes invalid.

## FAQ

### 1. Why is checking `node.left.val < node.val < node.right.val` not enough?

Because BST validity is a subtree-wide property.
A node deep in the right subtree must still be greater than all ancestors that forced that branch.

### 2. Why does inorder traversal work?

Because a valid BST produces a strictly increasing inorder sequence.
If the inorder sequence is not strictly increasing, the tree cannot be a BST.

### 3. Which version should I remember first?

Remember the range-constraint recursion first.
It matches the problem definition more directly and transfers better to other ancestor-constrained tree problems.

## Best Practices

- Anchor the solution around the invariant `low < val < high`
- Use a wider boundary type than the node value type when needed
- Hand-simulate one valid case and one invalid case to lock in the range propagation
- Practice this together with inorder traversal problems, because BST and inorder are deeply connected

## S — Summary

- BST validation is not a local parent-child comparison problem; it is a full ancestor-range problem
- The clean recursive form is `dfs(node, low, high)`
- The boundary check must be strict: `low < val < high`
- Inorder monotonicity gives an equivalent validation view, but range propagation is the clearer derivation
- Once this invariant is stable, many BST and tree-constraint problems become easier

## Further Reading

- [LeetCode 98: Validate Binary Search Tree](https://leetcode.cn/problems/validate-binary-search-tree/)
- LeetCode 94: Binary Tree Inorder Traversal
- LeetCode 230: Kth Smallest Element in a BST
- LeetCode 700: Search in a Binary Search Tree

## CTA

A good follow-up set is `98 + 94 + 230`.
That combination locks in BST validation, inorder traversal, and one classic ordered-tree query pattern together.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root):
    def dfs(node, low, high):
        if node is None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root, float("-inf"), float("inf"))


if __name__ == "__main__":
    root = TreeNode(2, TreeNode(1), TreeNode(3))
    print(is_valid_bst(root))
```

```c
#include <limits.h>
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

bool dfs(struct TreeNode* node, long long low, long long high) {
    if (node == NULL) return true;
    long long v = node->val;
    if (v <= low || v >= high) return false;
    return dfs(node->left, low, v) && dfs(node->right, v, high);
}

bool isValidBST(struct TreeNode* root) {
    return dfs(root, LLONG_MIN, LLONG_MAX);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(2);
    root->left = new_node(1);
    root->right = new_node(3);
    printf("%s\n", isValidBST(root) ? "true" : "false");
    free_tree(root);
    return 0;
}
```

```cpp
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

bool dfs(TreeNode* node, long long low, long long high) {
    if (!node) return true;
    long long v = node->val;
    if (v <= low || v >= high) return false;
    return dfs(node->left, low, v) && dfs(node->right, v, high);
}

bool isValidBST(TreeNode* root) {
    return dfs(root, LLONG_MIN, LLONG_MAX);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(2);
    root->left = new TreeNode(1);
    root->right = new TreeNode(3);
    std::cout << (isValidBST(root) ? "true" : "false") << '\n';
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

func dfs(node *TreeNode, low, high int64) bool {
	if node == nil {
		return true
	}
	v := int64(node.Val)
	if v <= low || v >= high {
		return false
	}
	return dfs(node.Left, low, v) && dfs(node.Right, v, high)
}

func isValidBST(root *TreeNode) bool {
	return dfs(root, -1<<63, 1<<63-1)
}

func main() {
	root := &TreeNode{
		Val:   2,
		Left:  &TreeNode{Val: 1},
		Right: &TreeNode{Val: 3},
	}
	fmt.Println(isValidBST(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn dfs(node: &Option<Box<TreeNode>>, low: i64, high: i64) -> bool {
    match node {
        None => true,
        Some(n) => {
            let v = n.val as i64;
            if v <= low || v >= high {
                return false;
            }
            dfs(&n.left, low, v) && dfs(&n.right, v, high)
        }
    }
}

fn is_valid_bst(root: &Option<Box<TreeNode>>) -> bool {
    dfs(root, i64::MIN, i64::MAX)
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 2,
        left: Some(Box::new(TreeNode {
            val: 1,
            left: None,
            right: None,
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: None,
        })),
    }));

    println!("{}", is_valid_bst(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function dfs(node, low, high) {
  if (!node) return true;
  if (node.val <= low || node.val >= high) return false;
  return dfs(node.left, low, node.val) && dfs(node.right, node.val, high);
}

function isValidBST(root) {
  return dfs(root, -Infinity, Infinity);
}

const root = new TreeNode(2, new TreeNode(1), new TreeNode(3));
console.log(isValidBST(root));
```
