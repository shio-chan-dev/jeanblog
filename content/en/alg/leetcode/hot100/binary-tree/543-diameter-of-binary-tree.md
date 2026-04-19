---
title: "Hot100: Diameter of Binary Tree (Tree DP / Height Return ACERS Guide)"
date: 2026-04-19T15:51:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "tree DP", "DFS", "postorder", "LeetCode 543"]
description: "A practical guide to LeetCode 543 covering the key split between returned height and global diameter updates, plus engineering mappings and runnable multi-language implementations."
keywords: ["Diameter of Binary Tree", "tree DP", "height", "DFS", "LeetCode 543", "Hot100"]
---

> **Subtitle / Summary**
> The most common confusion in LeetCode 543 is deciding what the recursive function should return. It should return height, not diameter. The diameter is updated globally at each node using `leftHeight + rightHeight`. Once that separation is clear, the problem becomes a clean introduction to tree DP.

- **Reading time**: 10-13 min
- **Tags**: `Hot100`, `binary tree`, `tree DP`, `DFS`, `postorder`
- **SEO keywords**: Diameter of Binary Tree, tree DP, height return, DFS, LeetCode 543
- **Meta description**: Learn LeetCode 543 from the postorder height-return pattern, with step-by-step derivation, engineering analogies, and runnable multi-language solutions.

---

## A — Algorithm

### Problem Restatement

Given the root `root` of a binary tree, return the diameter of the tree.

The diameter of a binary tree is:

- the length of the longest path between any two nodes in the tree
- a path that may or may not pass through the root
- measured by the number of edges

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | int | the diameter of the tree |

### Example 1

```text
input: root = [1,2,3,4,5]
output: 3
explanation: one longest path is [4,2,1,3], whose length is 3 edges.
```

### Example 2

```text
input: root = [1,2]
output: 1
```

### Constraints

- The number of nodes is in the range `[1, 10^4]`
- `-100 <= Node.val <= 100`

---

## Target Readers

- Hot100 learners moving from plain tree recursion into tree DP thinking
- Developers who often mix up recursive return values with the final global answer
- Engineers dealing with longest chains in hierarchical structures

## Background / Motivation

At first glance, this problem looks close to maximum depth.
But it is not the same.

Maximum depth asks:

- how far can I go downward from one node?

Diameter asks:

- what is the longest path between any two nodes in the entire tree?

That difference matters because the longest path may:

- pass through the root
- stay entirely inside one subtree

This is exactly why LeetCode 543 is such a good first tree-DP problem.
It forces you to separate:

- what a subtree returns upward
- how the global best answer is updated

## Core Concepts

- **Height**: the longest downward one-side path starting from the current node
- **Diameter candidate through the current node**: `leftHeight + rightHeight`
- **Postorder traversal**: the current node can only compute its candidate after it knows both subtree heights
- **Global answer**: the diameter is a whole-tree maximum, so it should not be confused with the returned height

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the phrase "between any two nodes"

The problem does not ask for a root-to-leaf path.
It asks for the longest path between any pair of nodes.

That means the answer is not limited to one downward chain.
A good candidate path often looks like:

```text
deep leaf in left subtree -> ... -> current node -> ... -> deep leaf in right subtree
```

So one node may connect two downward branches into a longer path.

#### Step 2: Ask what the current node needs in order to build such a path

If a path passes through the current node, the current node needs to know:

- how deep the left subtree can extend downward
- how deep the right subtree can extend downward

So the most useful recursive return value is:

> the height of the current subtree

#### Step 3: Define the recursive subproblem

For each node:

1. get the height of the left subtree
2. get the height of the right subtree
3. use them to update the best diameter candidate through this node
4. return the current height upward

That gives the postorder structure:

```python
left = dfs(node.left)
right = dfs(node.right)
```

#### Step 4: Define the base case

The cleanest definition is:

```python
if node is None:
    return 0
```

So an empty subtree has height `0`.

That makes leaf nodes convenient:

- left height = `0`
- right height = `0`
- leaf height = `1`

#### Step 5: Why is the diameter candidate `left + right`?

In this setup:

- `left` is the longest downward path length contributed by the left side
- `right` is the longest downward path length contributed by the right side

If the longest path goes through the current node, those two sides connect here.
So the number of edges in that candidate path is:

```python
left + right
```

This matches the problem statement, which measures path length by edges.

#### Step 6: Why should the diameter be updated in a separate variable?

Because the recursive return value and the final answer are not the same thing.

The parent node only needs to know:

> how far this subtree can extend upward as one branch

That is height.
The diameter, however, is the maximum over the whole tree.

So the clean split is:

- `dfs(node)` returns height
- `ans` stores the best diameter seen so far

```python
ans = max(ans, left + right)
```

#### Step 7: What should the current node return upward?

The parent can only continue one side upward, not both.
So the height returned by the current node is:

```python
return 1 + max(left, right)
```

That is the key separation:

- global answer uses both sides
- recursive return value uses only the longer side

#### Step 8: Walk the main example slowly

Use the official example:

```text
root = [1,2,3,4,5]
```

Bottom-up:

- node `4` has height `1`
- node `5` has height `1`
- at node `2`, `left = 1`, `right = 1`
- so the candidate diameter through `2` is `2`
- node `2` returns height `2`

At root `1`:

- left height = `2`
- right height = `1`

So the candidate diameter through the root is `3`, which is the final answer.

#### Step 9: Reduce the whole method to one reusable sentence

The whole idea can be compressed to:

> During postorder traversal, return height upward, and at every node update the global diameter with `leftHeight + rightHeight`.

Once that sentence is stable, the code is stable.

### Assemble the Full Code

Now combine the rules above into the first complete working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_of_binary_tree(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0

        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    print(diameter_of_binary_tree(root))
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
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0

        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return 0

            left = dfs(node.left)
            right = dfs(node.right)
            ans = max(ans, left + right)
            return 1 + max(left, right)

        dfs(root)
        return ans
```

### What mental model should stick?

Do not remember this as "the diameter trick".
Remember the separation of responsibilities:

- the recursive return value is height
- the global variable is the best diameter seen so far

That design is what makes the solution clean.

---

## E — Engineering

### Scenario 1: Measure the longest communication chain in an org tree (Python)

**Background**: in a hierarchy, you may want the longest possible chain between two members.  
**Why it fits**: that is the tree-diameter problem in an org-chart form.

```python
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def longest_chain(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


root = Node("CEO", Node("VP1", Node("M1"), Node("M2")), Node("VP2"))
print(longest_chain(root))
```

### Scenario 2: Measure the farthest propagation span in a service call tree (Go)

**Background**: a request trace often forms a tree of service calls.  
**Why it fits**: the farthest pair of nodes in that call tree is its diameter.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func diameter(root *Node) int {
	ans := 0
	var dfs func(*Node) int
	dfs = func(node *Node) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left+right > ans {
			ans = left + right
		}
		if left > right {
			return 1 + left
		}
		return 1 + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &Node{Name: "A", Left: &Node{Name: "B"}, Right: &Node{Name: "C"}}
	fmt.Println(diameter(root))
}
```

### Scenario 3: Estimate the longest event propagation chain in a component tree (JavaScript)

**Background**: deeply nested frontend component trees can create long state or event propagation paths.  
**Why it fits**: the maximum span across the tree is structurally the diameter.

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function diameter(root) {
  let ans = 0;
  function dfs(node) {
    if (!node) return 0;
    const left = dfs(node.left);
    const right = dfs(node.right);
    ans = Math.max(ans, left + right);
    return 1 + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new Node("App", new Node("Sidebar"), new Node("Content"));
console.log(diameter(root));
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)`, because every node is visited once
- **Space complexity**: `O(h)`, where `h` is the tree height, due to recursion stack depth

### Alternative Approaches

| Method | Time | Extra space | Notes |
| --- | --- | --- | --- |
| Postorder height return | `O(n)` | `O(h)` | cleanest and recommended |
| Recompute subtree heights at every node | worst-case `O(n^2)` | `O(h)` | too much repeated work |
| Convert to an undirected graph and run two BFS/DFS passes | `O(n)` | `O(n)` | valid, but heavier than needed here |

### Common Mistakes

1. **Trying to return diameter from the recursive function**: the parent really needs height, not subtree diameter.  
2. **Forgetting that the problem counts edges**: this creates off-by-one errors.  
3. **Only checking paths through the root**: the true longest path may lie entirely inside one subtree.  
4. **Updating `ans` at the wrong time**: the update must happen after both subtree heights are known.

## FAQ

### 1. Why is `left + right` the right candidate?

Because the current node can connect the deepest downward branch from the left and the deepest downward branch from the right into one full path.

### 2. Why not return `max(leftDiameter, rightDiameter, left + right)` directly?

You could return a larger structure with multiple fields, but that is unnecessary here.
Returning height and keeping a separate global best is simpler and clearer.

### 3. How is this related to Maximum Depth of Binary Tree?

Maximum depth only cares about one downward chain.
Diameter uses both sides together at each node.
So LeetCode 104 is a strong prerequisite for 543, but 543 adds a global-best layer on top.

## Best Practices

- Decide before coding that `dfs` returns height, not the final answer
- Keep the roles of `dfs` and `ans` strictly separate
- Hand-simulate one small tree and check edges versus nodes carefully
- Practice this together with LeetCode 104 and 110 to solidify postorder tree recurrence

## S — Summary

- Diameter is not the same as maximum depth
- The clean solution returns height upward and updates diameter globally
- Every node contributes one candidate path `left + right`
- The longest path may pass through the root or stay entirely inside a subtree
- LeetCode 543 is one of the best entry points into tree DP and postorder aggregation

## Further Reading

- [LeetCode 543: Diameter of Binary Tree](https://leetcode.cn/problems/diameter-of-binary-tree/)
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 110: Balanced Binary Tree
- LeetCode 236: Lowest Common Ancestor of a Binary Tree

## CTA

A strong practice group is `104 + 543 + 110`.
Those three problems together train the core tree skill of "what does the subtree return, and what does the current node aggregate?"

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_of_binary_tree(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    print(diameter_of_binary_tree(root))
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

int max_int(int a, int b) {
    return a > b ? a : b;
}

int dfs(struct TreeNode* node, int* ans) {
    if (node == NULL) return 0;

    int left = dfs(node->left, ans);
    int right = dfs(node->right, ans);
    if (left + right > *ans) *ans = left + right;
    return 1 + max_int(left, right);
}

int diameterOfBinaryTree(struct TreeNode* root) {
    int ans = 0;
    dfs(root, &ans);
    return ans;
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
    root->right = new_node(3);
    root->left->left = new_node(4);
    root->left->right = new_node(5);
    printf("%d\n", diameterOfBinaryTree(root));
    free_tree(root);
    return 0;
}
```

```cpp
#include <algorithm>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int dfs(TreeNode* node, int& ans) {
    if (!node) return 0;
    int left = dfs(node->left, ans);
    int right = dfs(node->right, ans);
    ans = std::max(ans, left + right);
    return 1 + std::max(left, right);
}

int diameterOfBinaryTree(TreeNode* root) {
    int ans = 0;
    dfs(root, ans);
    return ans;
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
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    std::cout << diameterOfBinaryTree(root) << '\n';
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

func diameterOfBinaryTree(root *TreeNode) int {
	ans := 0

	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		left := dfs(node.Left)
		right := dfs(node.Right)
		if left+right > ans {
			ans = left + right
		}
		if left > right {
			return 1 + left
		}
		return 1 + right
	}

	dfs(root)
	return ans
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 5},
		},
		Right: &TreeNode{Val: 3},
	}
	fmt.Println(diameterOfBinaryTree(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn dfs(node: &Option<Box<TreeNode>>, ans: &mut i32) -> i32 {
    match node {
        None => 0,
        Some(n) => {
            let left = dfs(&n.left, ans);
            let right = dfs(&n.right, ans);
            *ans = (*ans).max(left + right);
            1 + left.max(right)
        }
    }
}

fn diameter_of_binary_tree(root: &Option<Box<TreeNode>>) -> i32 {
    let mut ans = 0;
    dfs(root, &mut ans);
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 5,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: None,
        })),
    }));

    println!("{}", diameter_of_binary_tree(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function diameterOfBinaryTree(root) {
  let ans = 0;

  function dfs(node) {
    if (!node) return 0;
    const left = dfs(node.left);
    const right = dfs(node.right);
    ans = Math.max(ans, left + right);
    return 1 + Math.max(left, right);
  }

  dfs(root);
  return ans;
}

const root = new TreeNode(1, new TreeNode(2, new TreeNode(4), new TreeNode(5)), new TreeNode(3));
console.log(diameterOfBinaryTree(root));
```
