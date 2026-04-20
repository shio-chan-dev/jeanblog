---
title: "Hot100: Flatten Binary Tree to Linked List (Reverse Preorder Rewiring ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "preorder", "in-place", "recursion", "LeetCode 114"]
description: "A practical guide to LeetCode 114 covering why the flattened list must follow preorder order and how reverse preorder with a prev pointer gives a stable in-place rewiring strategy."
keywords: ["Flatten Binary Tree to Linked List", "preorder", "in-place", "reverse preorder", "LeetCode 114", "Hot100"]
---

> **Subtitle / Summary**
> The real difficulty of LeetCode 114 is not "flattening" a tree. It is rewiring pointers without destroying the structure you still need. Once you notice that the final linked list must follow preorder order, reverse preorder with a `prev` pointer becomes a very stable in-place solution.

- **Reading time**: 12-15 min
- **Tags**: `Hot100`, `binary tree`, `preorder`, `in-place`, `recursion`
- **SEO keywords**: Flatten Binary Tree to Linked List, preorder, in-place, reverse preorder, LeetCode 114
- **Meta description**: Learn LeetCode 114 from preorder order and reverse-preorder rewiring, with step-by-step derivation, engineering mappings, and runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given the root `root` of a binary tree, flatten the tree into a linked list in place.

The flattened structure must satisfy:

- every `left` pointer becomes `null`
- the `right` pointers form a single chain
- the chain order must be exactly the preorder traversal order of the original tree

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | void | modify the tree in place |

### Example 1

```text
input: root = [1,2,5,3,4,null,6]
output: [1,null,2,null,3,null,4,null,5,null,6]
```

### Example 2

```text
input: root = []
output: []
```

### Example 3

```text
input: root = [0]
output: [0]
```

### Constraints

- The number of nodes in the tree is in the range `[0, 2000]`
- `-100 <= Node.val <= 100`

### Follow-up

Can you flatten the tree with `O(1)` extra space?

---

## Target Readers

- Hot100 learners who already know preorder traversal but still struggle with pointer rewiring
- Developers who often lose subtrees when doing in-place tree transformations
- Readers who want to connect preorder order with structural rewriting

## Background / Motivation

This problem is not mainly about traversal.
It is about how to **modify pointers safely** while keeping the required final order.

A first solution might be:

- collect nodes in preorder into an array
- reconnect them afterward

That works, but it uses `O(n)` extra space and misses the deeper skill:

- how to transform a tree into a chain in place

So the real question is:

> If the final chain must follow preorder order, what processing direction lets me rewire safely without losing future nodes?

## Core Concepts

- **Preorder traversal**: root, left, right
- **Reverse preorder**: right, left, root
- **`prev` pointer**: the next node in the final flattened chain
- **In-place rewiring**: reuse the existing nodes and modify their pointers directly

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Clarify what the final chain must look like

For:

```text
    1
   / \
  2   5
 / \   \
3   4   6
```

The preorder traversal is:

```text
[1,2,3,4,5,6]
```

So the flattened list must become:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

That means the problem is really:

> Rewire the tree so the `right` chain follows preorder order.

#### Step 2: Why is forward rewiring risky?

Suppose you stand at node `1` and immediately try:

```text
1.right = 2
1.left = null
```

You can easily lose the original right subtree rooted at `5` unless you save and reconnect it carefully.

So a direct forward rewrite is fragile.

#### Step 3: Reverse the direction of thought

If it is risky to build the chain from front to back, do the opposite:

> build the tail first, then connect the current node to it

The final order is:

```text
root -> left -> right
```

So the reverse processing order is:

```text
right -> left -> root
```

That is reverse preorder.

#### Step 4: Define the base case

An empty node needs no rewiring:

```python
if node is None:
    return
```

#### Step 5: Why must recursion go right first?

Because when we return to the current node, we want `prev` to already point to the correct chain that should follow it.

So the recursive order is:

```python
dfs(node.right)
dfs(node.left)
```

#### Step 6: Define what `prev` means

`prev` should mean:

> the node that should come immediately after the current node in the final flattened chain

That makes the rewiring step very small:

```python
node.right = prev
node.left = None
prev = node
```

#### Step 7: Why do those three lines work?

At the moment the recursion comes back to `node`:

- the right side has already been flattened
- the left side has already been flattened
- `prev` already points to the chain that should follow `node`

So connecting `node.right` to `prev` is exactly the needed link.

#### Step 8: Walk the official example slowly

Use:

```text
[1,2,5,3,4,null,6]
```

Reverse preorder visits nodes in this order:

```text
6 -> 5 -> 4 -> 3 -> 2 -> 1
```

Rewiring happens during the return:

1. `6.right = None`, `prev = 6`
2. `5.right = 6`, `prev = 5`
3. `4.right = 5`, `prev = 4`
4. `3.right = 4`, `prev = 3`
5. `2.right = 3`, `prev = 2`
6. `1.right = 2`, `prev = 1`

The final chain is exactly the preorder sequence.

#### Step 9: Connect this to the follow-up

The recursive `prev` solution is excellent for understanding the problem.

The `O(1)` extra-space follow-up can be solved with iterative predecessor rewiring, but that method is more subtle.
For derivation-first learning, reverse preorder is the cleaner starting point.

### Assemble the Full Code

Now combine the reverse-preorder steps into the first full working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


def collect_chain(root):
    ans = []
    cur = root
    while cur:
        ans.append(cur.val)
        cur = cur.right
    return ans


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(5, None, TreeNode(6)),
    )
    flatten(root)
    print(collect_chain(root))
```

### Reference Answer

The LeetCode-style submission version is:

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        prev = None

        def dfs(node: Optional[TreeNode]) -> None:
            nonlocal prev
            if node is None:
                return
            dfs(node.right)
            dfs(node.left)
            node.right = prev
            node.left = None
            prev = node

        dfs(root)
```

### What method did we just build?

You could call it:

- reverse preorder recursion
- in-place tree rewiring
- successor-link construction with `prev`

But the core insight is:

> If forward rewiring is dangerous, prepare the tail first and reconnect backward.

---

## E — Engineering

### Scenario 1: Flatten a module tree into startup order (Python)

**Background**: a system stores modules in a tree, but startup needs a linear execution chain.  
**Why it fits**: once the target order is fixed, in-place rewiring can avoid extra node copies.

```python
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


root = Node("core", Node("auth"), Node("feed"))
flatten(root)
print(root.name, root.right.name)
```

### Scenario 2: Turn a task tree into a linear execution chain (Go)

**Background**: a task tree needs to be flattened into one executable order.  
**Why it fits**: reverse preorder lets each task reconnect to the correct successor safely.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func flatten(root *Node) {
	var prev *Node
	var dfs func(*Node)
	dfs = func(node *Node) {
		if node == nil {
			return
		}
		dfs(node.Right)
		dfs(node.Left)
		node.Right = prev
		node.Left = nil
		prev = node
	}
	dfs(root)
}

func main() {
	root := &Node{Name: "A", Left: &Node{Name: "B"}, Right: &Node{Name: "C"}}
	flatten(root)
	fmt.Println(root.Name, root.Right.Name)
}
```

### Scenario 3: Flatten a menu tree into a tab order chain (JavaScript)

**Background**: a compact UI mode may want a tree-like menu represented as a linear focus order.  
**Why it fits**: the same rewiring idea converts a tree into one right-linked chain.

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function flatten(root) {
  let prev = null;
  function dfs(node) {
    if (!node) return;
    dfs(node.right);
    dfs(node.left);
    node.right = prev;
    node.left = null;
    prev = node;
  }
  dfs(root);
}

const root = new Node("Home", new Node("Docs"), new Node("Blog"));
flatten(root);
console.log(root.name, root.right.name);
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)` because each node is rewired once
- **Space complexity**: `O(h)` from recursion depth

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Reverse preorder + `prev` | `O(n)` | `O(h)` | Best learning-first solution |
| Preorder array + reconnect | `O(n)` | `O(n)` | Easy, but uses extra storage |
| Stack-based preorder rewrite | `O(n)` | `O(n)` | Also valid, but heavier |
| Iterative predecessor rewiring | `O(n)` | `O(1)` | Matches the follow-up, but is trickier to derive |

### Common Mistakes

1. **Rewiring forward too early**: that can disconnect the original right subtree before it is processed.  
2. **Forgetting to clear `left`**: the final linked list must use only `right` pointers.  
3. **Misdefining `prev`**: it is not just the previously visited node; it is the next node in the final chain.  
4. **Using left-right-root instead of right-left-root**: then the rewiring order breaks.

## FAQ and Notes

### 1. Why must recursion go right first and then left?

Because the final chain is preorder:

```text
root -> left -> right
```

So the safe reverse processing order is:

```text
right -> left -> root
```

### 2. Why can `prev` become `node.right` directly?

Because at that moment `prev` already points to the chain that must come after the current node in the final preorder list.

### 3. What about the `O(1)` follow-up?

You can solve it iteratively by:

- finding the rightmost node of the left subtree
- attaching the original right subtree there
- moving the left subtree to the right side

That is a great follow-up, but reverse preorder is easier to understand first.

## Best Practices

- Write down the target order before rewiring pointers
- If forward construction is fragile, try reversing the processing order
- Give `prev` one precise sentence of meaning before coding
- Pair this problem with preorder traversal and tree reconstruction problems

## S — Summary

- The flattened linked list must follow preorder order
- Reverse preorder with a `prev` pointer is a stable way to build that list in place
- `prev` means "the next node in the final chain", which is the key invariant
- This problem is really about safe in-place tree transformation, not just traversal
- After understanding this version, the `O(1)` predecessor-based solution becomes much easier to appreciate

## Further Reading

- [LeetCode 114: Flatten Binary Tree to Linked List](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)
- LeetCode 144: Binary Tree Preorder Traversal
- LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal
- LeetCode 173: Binary Search Tree Iterator

## CTA

Practice `144 + 114 + 105` together.
They connect preorder order, structural transformation, and reconstruction into one very useful tree pattern set.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


def collect(root):
    ans = []
    cur = root
    while cur:
        ans.append(cur.val)
        cur = cur.right
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5, None, TreeNode(6)))
    flatten(root)
    print(collect(root))
```

```c
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

void dfs(struct TreeNode* node, struct TreeNode** prev) {
    if (!node) return;
    dfs(node->right, prev);
    dfs(node->left, prev);
    node->right = *prev;
    node->left = NULL;
    *prev = node;
}

void flatten(struct TreeNode* root) {
    struct TreeNode* prev = NULL;
    dfs(root, &prev);
}

void print_chain(struct TreeNode* root) {
    while (root) {
        printf("%d ", root->val);
        root = root->right;
    }
    printf("\n");
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->left = new_node(2);
    root->right = new_node(5);
    root->left->left = new_node(3);
    root->left->right = new_node(4);
    root->right->right = new_node(6);
    flatten(root);
    print_chain(root);
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

void dfs(TreeNode* node, TreeNode*& prev) {
    if (!node) return;
    dfs(node->right, prev);
    dfs(node->left, prev);
    node->right = prev;
    node->left = nullptr;
    prev = node;
}

void flatten(TreeNode* root) {
    TreeNode* prev = nullptr;
    dfs(root, prev);
}

void printChain(TreeNode* root) {
    while (root) {
        std::cout << root->val << ' ';
        root = root->right;
    }
    std::cout << '\n';
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(5);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->right = new TreeNode(6);
    flatten(root);
    printChain(root);
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

func flatten(root *TreeNode) {
	var prev *TreeNode
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		dfs(node.Left)
		node.Right = prev
		node.Left = nil
		prev = node
	}
	dfs(root)
}

func printChain(root *TreeNode) {
	for root != nil {
		fmt.Print(root.Val, " ")
		root = root.Right
	}
	fmt.Println()
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
			Val:   5,
			Right: &TreeNode{Val: 6},
		},
	}
	flatten(root)
	printChain(root)
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn flatten(root: &mut Option<Box<TreeNode>>) {
    fn dfs(node: &mut Option<Box<TreeNode>>, prev: &mut Option<Box<TreeNode>>) {
        if let Some(mut cur) = node.take() {
            dfs(&mut cur.right, prev);
            dfs(&mut cur.left, prev);
            cur.right = prev.take();
            cur.left = None;
            *prev = Some(cur);
        }
    }

    let mut prev = None;
    dfs(root, &mut prev);
    *root = prev;
}

fn collect(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    let mut ans = Vec::new();
    let mut cur = root.as_deref();
    while let Some(node) = cur {
        ans.push(node.val);
        cur = node.right.as_deref();
    }
    ans
}

fn main() {
    let mut root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 3,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 5,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 6,
                left: None,
                right: None,
            })),
        })),
    }));

    flatten(&mut root);
    println!("{:?}", collect(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function flatten(root) {
  let prev = null;
  function dfs(node) {
    if (!node) return;
    dfs(node.right);
    dfs(node.left);
    node.right = prev;
    node.left = null;
    prev = node;
  }
  dfs(root);
}

function collect(root) {
  const ans = [];
  let cur = root;
  while (cur) {
    ans.push(cur.val);
    cur = cur.right;
  }
  return ans;
}

const root = new TreeNode(
  1,
  new TreeNode(2, new TreeNode(3), new TreeNode(4)),
  new TreeNode(5, null, new TreeNode(6))
);
flatten(root);
console.log(collect(root));
```
