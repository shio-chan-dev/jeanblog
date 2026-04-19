---
title: "Hot100: Lowest Common Ancestor of a Binary Tree (Postorder Return Semantics ACERS Guide)"
date: 2026-04-19T15:51:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "LCA", "DFS", "postorder", "LeetCode 236"]
description: "A practical guide to LeetCode 236 covering the return-value meaning in postorder recursion, when the current node becomes the answer, and why a node can be its own ancestor."
keywords: ["Lowest Common Ancestor of a Binary Tree", "LCA", "postorder", "DFS", "LeetCode 236", "Hot100"]
---

> **Subtitle / Summary**
> The real difficulty of LeetCode 236 is not memorizing an LCA template. It is defining what each recursive call should return. Once that return value becomes stable, the whole problem collapses into a short and very clean postorder recursion.

- **Reading time**: 11-14 min
- **Tags**: `Hot100`, `binary tree`, `LCA`, `DFS`, `postorder`
- **SEO keywords**: Lowest Common Ancestor of a Binary Tree, LCA, postorder, DFS, LeetCode 236
- **Meta description**: Learn LeetCode 236 from the recursive return-value semantics, with step-by-step derivation, engineering mappings, and runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes `p` and `q`.

The LCA is defined as the lowest node in the tree that:

- is an ancestor of both `p` and `q`
- has the greatest possible depth among such common ancestors
- may be one of the nodes itself

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| p, q | TreeNode | the two target nodes; examples identify them by unique values |
| return | TreeNode | the lowest common ancestor of `p` and `q` |

### Example 1

```text
input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
output: 3
explanation: the LCA of nodes 5 and 1 is 3.
```

### Example 2

```text
input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
output: 5
explanation: node 5 is an ancestor of node 4, and a node may be its own ancestor.
```

### Example 3

```text
input: root = [1,2], p = 1, q = 2
output: 1
```

### Constraints

- The number of nodes is in the range `[2, 10^5]`
- `-10^9 <= Node.val <= 10^9`
- All `Node.val` values are unique
- `p != q`
- Both `p` and `q` exist in the tree

---

## Target Readers

- Hot100 learners who can write basic tree recursion but still get stuck on LCA return values
- Developers who want one stable postorder divide-and-conquer pattern for binary trees
- Engineers who work with org charts, directory trees, or component trees and need nearest shared ancestors

## Background / Motivation

When many people first see this problem, they instinctively think about:

- building parent pointers
- finding the path from the root to `p`
- finding the path from the root to `q`
- comparing the two paths

Those methods work.
But the most valuable version of this problem is the recursive one:

- no extra parent map
- no extra graph conversion
- one postorder traversal

The core question is:

> What should a subtree report back to its parent?

Once that answer is clear, the implementation becomes short and reliable.

## Core Concepts

- **Postorder recursion**: first solve the left subtree, then the right subtree, then interpret the current node
- **Return-value semantics**: return the target node found in this subtree, or the confirmed LCA if it is already determined
- **A node can be its own ancestor**: if `p` is above `q`, the answer may be `p`
- **Bubble up**: once a subtree already holds the right answer, parents simply keep returning it upward

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the rule that a node can be its own ancestor

Look at the official second example:

```text
p = 5, q = 4
answer = 5
```

That means:

- the answer does not have to be strictly above both nodes
- if one target is already an ancestor of the other, that node is the LCA

So when recursion reaches `p` or `q`, that event is already important enough to preserve.

#### Step 2: Decide what each subtree should return

The most stable return-value definition is:

> `dfs(node)` returns the key node already identified inside this subtree.

That return value has three possibilities:

- `None`: this subtree contains neither target
- `p` or `q`: this subtree contains exactly one target so far
- the LCA node: this subtree has already found the final answer

That single definition makes the whole recursion consistent.

#### Step 3: Define the recursive subproblem

Once the left subtree and right subtree have already reported what they found, how should the current node interpret those two results?

That leads directly to postorder structure:

```python
left = dfs(node.left)
right = dfs(node.right)
```

#### Step 4: Define the base cases

Two base cases matter most:

```python
if node is None:
    return None
if node.val == p or node.val == q:
    return node
```

The second one is the key.
If the current node is one of the targets, that information must be kept and returned upward immediately.

#### Step 5: List the possible outcomes after left and right return

Once `left` and `right` come back, only three cases remain:

- both are `None`: neither target is in this subtree
- exactly one is non-null: one target, or an already found answer, is somewhere below
- both are non-null: one target is on each side, so the current node is the first merge point

That last case is the heart of the problem.

#### Step 6: When does the current node become the answer?

If both sides are non-null:

```python
if left and right:
    return node
```

That means:

- one target was found in the left subtree
- the other target was found in the right subtree

So the current node is exactly their lowest common ancestor.

#### Step 7: Why do we return the non-null side when only one side is non-null?

Because the current node is not yet proven to be the LCA.
It only knows that one important node exists somewhere below one side.

So the correct move is:

```python
return left if left else right
```

This keeps bubbling the useful result upward until a higher node has enough information to decide.

#### Step 8: Walk one example slowly

Use the official first example:

```text
root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
```

At node `5`, recursion immediately returns node `5`.
At node `1`, recursion immediately returns node `1`.

Back at the root `3`:

- `left = 5`
- `right = 1`

Both are non-null, so `3` becomes the LCA.

#### Step 9: Why do runnable examples often pass values instead of node references?

The actual LeetCode interface passes node objects `p` and `q`.
But this problem guarantees all node values are unique.

So for blog examples, it is perfectly fine to pass `pVal` and `qVal`:

- the example stays self-contained
- the recursion logic stays identical
- multi-language snippets become easier to run locally

### Assemble the Full Code

Now combine the pieces above into the first complete working solution.
This version passes target values for easier local execution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root, pVal, qVal):
    def dfs(node):
        if node is None:
            return None
        if node.val == pVal or node.val == qVal:
            return node

        left = dfs(node.left)
        right = dfs(node.right)

        if left and right:
            return node
        return left if left else right

    return dfs(root)


if __name__ == "__main__":
    root = TreeNode(
        3,
        TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4))),
        TreeNode(1, TreeNode(0), TreeNode(8)),
    )
    ans = lowest_common_ancestor(root, 5, 1)
    print(ans.val if ans else None)
```

### Reference Answer

For LeetCode submission style, the same logic becomes:

```python
from typing import Optional


class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class Solution:
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        if root is None or root is p or root is q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        return left if left else right
```

### What mental model should stick?

Do not memorize this as just "the LCA recursion".
Remember the design order:

1. define what a subtree returns
2. recurse left and right
3. let the current node interpret those two returned values

That sequence is the reusable idea.

---

## E — Engineering

### Scenario 1: Find the nearest shared manager in an org tree (Python)

**Background**: in an org chart, you often need the closest shared manager of two employees.  
**Why it fits**: that is the direct real-world shape of the LCA problem.

```python
class Staff:
    def __init__(self, id_, left=None, right=None):
        self.id = id_
        self.left = left
        self.right = right


def lca(root, a, b):
    if root is None or root.id == a or root.id == b:
        return root
    left = lca(root.left, a, b)
    right = lca(root.right, a, b)
    if left and right:
        return root
    return left or right


root = Staff(3, Staff(5), Staff(1))
print(lca(root, 5, 1).id)
```

### Scenario 2: Find the nearest shared container in a component tree (JavaScript)

**Background**: while debugging layout or event boundaries, you may need the nearest shared container of two component nodes.  
**Why it fits**: the LCA is exactly the first place where the two component paths merge.

```javascript
function Node(id, left = null, right = null) {
  this.id = id;
  this.left = left;
  this.right = right;
}

function lca(root, a, b) {
  if (!root || root.id === a || root.id === b) return root;
  const left = lca(root.left, a, b);
  const right = lca(root.right, a, b);
  if (left && right) return root;
  return left || right;
}

const root = new Node("page", new Node("sidebar"), new Node("content"));
console.log(lca(root, "sidebar", "content").id);
```

### Scenario 3: Find the nearest shared parent directory in a tree model (Go)

**Background**: when two files or folders need a batch operation, systems often first locate their nearest shared parent.  
**Why it fits**: this is the directory-tree version of LCA.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func lca(root *Node, a, b string) *Node {
	if root == nil || root.Name == a || root.Name == b {
		return root
	}
	left := lca(root.Left, a, b)
	right := lca(root.Right, a, b)
	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	}
	return right
}

func main() {
	root := &Node{Name: "root", Left: &Node{Name: "docs"}, Right: &Node{Name: "blog"}}
	fmt.Println(lca(root, "docs", "blog").Name)
}
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is visited at most once
- **Space complexity**: `O(h)`, where `h` is the tree height, due to recursion stack depth

### Alternative Approaches

| Method | Time | Extra space | Notes |
| --- | --- | --- | --- |
| Postorder return-value recursion | `O(n)` | `O(h)` | cleanest and most direct |
| Parent map + ancestor set | `O(n)` | `O(n)` | iterative-friendly, but needs extra structure |
| Root-to-node path comparison | `O(n)` | `O(h)` to `O(n)` | intuitive, but requires two path searches |

### Common Mistakes

1. **Forgetting that a node can be its own ancestor**: this breaks examples like `p = 5, q = 4`.  
2. **Not treating `root == p` or `root == q` as a base case**: then the key signal never bubbles upward correctly.  
3. **Treating "found one target" as "already found the answer"**: the current node is only the LCA when both sides return non-null.  
4. **Comparing by value in a tree without uniqueness**: the runnable examples do it only because this problem guarantees unique values.

## FAQ

### 1. What if `p` is an ancestor of `q`?

Then the answer is `p`.
That is exactly what "a node can be its own ancestor" means in this problem.

### 2. Why is postorder the natural fit?

Because the current node must first know what the left subtree and right subtree found.
Only then can it decide whether it is the first merge point.

### 3. What if the problem did not guarantee that both nodes exist?

Then the return-value design would need extra information, usually a count of how many targets were actually found.
This problem is simpler because both are guaranteed to exist.

## Best Practices

- Write the return-value meaning in plain English before writing code
- Handle `None`, `p`, and `q` immediately at function entry
- Whenever both recursive sides are non-null, think "first merge point"
- Practice this together with LeetCode 235 to compare ordinary binary trees with BST-specific LCA logic

## S — Summary

- The heart of LeetCode 236 is the meaning of the recursive return value
- Once a subtree returns "the key node found below", the whole solution becomes short and stable
- Postorder fits because the current node must interpret results from both children
- A node can be its own ancestor, and that rule is essential
- This problem is excellent practice for tree divide-and-conquer and upward result propagation

## Further Reading

- [LeetCode 236: Lowest Common Ancestor of a Binary Tree](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)
- LeetCode 235: Lowest Common Ancestor of a Binary Search Tree
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 543: Diameter of Binary Tree

## CTA

A good pair is `236 + 235`.
The first trains postorder return-value semantics; the second trains BST-specific pruning. Together they make the LCA topic much more stable.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root, pVal, qVal):
    if root is None or root.val == pVal or root.val == qVal:
        return root

    left = lowest_common_ancestor(root.left, pVal, qVal)
    right = lowest_common_ancestor(root.right, pVal, qVal)

    if left and right:
        return root
    return left if left else right


if __name__ == "__main__":
    root = TreeNode(
        3,
        TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4))),
        TreeNode(1, TreeNode(0), TreeNode(8)),
    )
    ans = lowest_common_ancestor(root, 5, 1)
    print(ans.val if ans else None)
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

struct TreeNode* lowestCommonAncestor(struct TreeNode* root, int p, int q) {
    if (root == NULL || root->val == p || root->val == q) return root;

    struct TreeNode* left = lowestCommonAncestor(root->left, p, q);
    struct TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(3);
    root->left = new_node(5);
    root->right = new_node(1);
    root->left->left = new_node(6);
    root->left->right = new_node(2);
    root->left->right->left = new_node(7);
    root->left->right->right = new_node(4);
    root->right->left = new_node(0);
    root->right->right = new_node(8);

    struct TreeNode* ans = lowestCommonAncestor(root, 5, 1);
    printf("%d\n", ans ? ans->val : -1);
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

TreeNode* lowestCommonAncestor(TreeNode* root, int p, int q) {
    if (!root || root->val == p || root->val == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(5);
    root->right = new TreeNode(1);
    root->left->left = new TreeNode(6);
    root->left->right = new TreeNode(2);
    root->left->right->left = new TreeNode(7);
    root->left->right->right = new TreeNode(4);
    root->right->left = new TreeNode(0);
    root->right->right = new TreeNode(8);

    TreeNode* ans = lowestCommonAncestor(root, 5, 1);
    std::cout << (ans ? ans->val : -1) << '\n';
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

func lowestCommonAncestor(root *TreeNode, p, q int) *TreeNode {
	if root == nil || root.Val == p || root.Val == q {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)

	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	}
	return right
}

func main() {
	root := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 5,
			Left: &TreeNode{Val: 6},
			Right: &TreeNode{
				Val:   2,
				Left:  &TreeNode{Val: 7},
				Right: &TreeNode{Val: 4},
			},
		},
		Right: &TreeNode{
			Val:   1,
			Left:  &TreeNode{Val: 0},
			Right: &TreeNode{Val: 8},
		},
	}

	ans := lowestCommonAncestor(root, 5, 1)
	fmt.Println(ans.Val)
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn lowest_common_ancestor(root: &Option<Box<TreeNode>>, p: i32, q: i32) -> Option<i32> {
    match root {
        None => None,
        Some(node) => {
            if node.val == p || node.val == q {
                return Some(node.val);
            }

            let left = lowest_common_ancestor(&node.left, p, q);
            let right = lowest_common_ancestor(&node.right, p, q);

            if left.is_some() && right.is_some() {
                return Some(node.val);
            }
            if left.is_some() {
                return left;
            }
            right
        }
    }
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 3,
        left: Some(Box::new(TreeNode {
            val: 5,
            left: Some(Box::new(TreeNode {
                val: 6,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 2,
                left: Some(Box::new(TreeNode {
                    val: 7,
                    left: None,
                    right: None,
                })),
                right: Some(Box::new(TreeNode {
                    val: 4,
                    left: None,
                    right: None,
                })),
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 1,
            left: Some(Box::new(TreeNode {
                val: 0,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 8,
                left: None,
                right: None,
            })),
        })),
    }));

    println!("{:?}", lowest_common_ancestor(&root, 5, 1));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function lowestCommonAncestor(root, p, q) {
  if (!root || root.val === p || root.val === q) return root;

  const left = lowestCommonAncestor(root.left, p, q);
  const right = lowestCommonAncestor(root.right, p, q);

  if (left && right) return root;
  return left || right;
}

const root = new TreeNode(
  3,
  new TreeNode(5, new TreeNode(6), new TreeNode(2, new TreeNode(7), new TreeNode(4))),
  new TreeNode(1, new TreeNode(0), new TreeNode(8))
);

const ans = lowestCommonAncestor(root, 5, 1);
console.log(ans ? ans.val : null);
```
