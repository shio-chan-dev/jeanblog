---
title: "Hot100: Binary Tree Maximum Path Sum (Tree DP / Single-Side Gain ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "tree DP", "DFS", "postorder", "LeetCode 124"]
description: "A practical guide to LeetCode 124 covering the key design split: return only the best single-side gain upward, while using the full path through the current node to update a global answer."
keywords: ["Binary Tree Maximum Path Sum", "tree DP", "single-side gain", "postorder", "DFS", "LeetCode 124", "Hot100"]
---

> **Subtitle / Summary**
> The easiest way to get lost in LeetCode 124 is to make one recursive return value carry too much meaning. The stable design is to separate two roles: the recursion returns only the best single-side gain to the parent, while the full path through the current node is used to update the global maximum.

- **Reading time**: 12-15 min
- **Tags**: `Hot100`, `binary tree`, `tree DP`, `DFS`, `postorder`
- **SEO keywords**: Binary Tree Maximum Path Sum, tree DP, single-side gain, postorder, DFS, LeetCode 124
- **Meta description**: Learn LeetCode 124 from the single-side gain invariant, global path update, and negative-branch pruning, with runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

A path in a binary tree is a sequence of nodes such that:

- each pair of adjacent nodes is connected by an edge
- a node appears at most once in the path
- the path contains at least one node
- the path does not need to pass through the root

The path sum is the sum of all node values in that path.
Return the maximum path sum in the entire tree.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | int | maximum path sum |

### Example 1

```text
input: root = [1,2,3]
output: 6
explanation: the best path is 2 -> 1 -> 3
```

### Example 2

```text
input: root = [-10,9,20,null,null,15,7]
output: 42
explanation: the best path is 15 -> 20 -> 7
```

### Constraints

- The number of nodes in the tree is in the range `[1, 3 * 10^4]`
- `-1000 <= Node.val <= 1000`

---

## Target Readers

- Hot100 learners who already solved 543 but still mix up return values in tree-path problems
- Developers who keep confusing "path through the current node" with "value returned to the parent"
- Readers who want one stable tree-DP interpretation for weighted path problems

## Background / Motivation

This problem is valuable because it teaches a very general lesson:

- a recursive function does not always return the final answer
- sometimes it returns only the information the parent truly needs

Two questions usually cause confusion:

- if the best path through a node uses both left and right children, what can the recursion return upward?
- if a child subtree contributes a negative sum, should it still be included?

The clean solution appears once you separate:

- **global answer**: the best full path seen anywhere
- **recursive return value**: the best single-side gain the current node can offer upward

## Core Concepts

- **Postorder recursion**: collect left and right information before processing the current node
- **Single-side gain**: the best downward contribution the current node can extend to its parent
- **Full-path candidate**: a complete path that goes through the current node and may use both sides
- **Negative pruning**: if a child contribution is negative, it should be dropped

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Clarify what the answer can look like

Take:

```text
    1
   / \
  2   3
```

The best path is not just one downward chain.
It is:

```text
2 -> 1 -> 3
```

So some node can act as a turning point and combine both sides.
That is the first important distinction from simpler depth problems.

#### Step 2: Decide what the parent actually needs back

Even if the best path through the current node uses both left and right children, the parent cannot keep both branches when continuing upward.

A path extended to the parent can only choose one side.

So the recursive return value should mean:

> the maximum single-side gain that starts at this node and can continue upward through the parent

That is the stable invariant.

#### Step 3: Define the smaller subproblem

If the left and right subtrees have already reported their best single-side gains, then the current node can compute:

- the best full path that passes through itself
- the single-side gain it returns upward

That is exactly a postorder situation.

#### Step 4: Define the base case

An empty node contributes no gain:

```python
if node is None:
    return 0
```

Notice that this is a gain value, not the final answer.

#### Step 5: Build the full-path candidate at the current node

First get both gains:

```python
left = dfs(node.left)
right = dfs(node.right)
```

But if a side is negative, keeping it only hurts the total.
So clamp each one at zero:

```python
left = max(left, 0)
right = max(right, 0)
```

Now the full path through the current node is:

```python
candidate = node.val + left + right
```

#### Step 6: Update the global answer

Since `candidate` is a complete valid path, use it to update the global maximum:

```python
ans = max(ans, candidate)
```

This path may be the best in the whole tree.

#### Step 7: Return only one side upward

The parent cannot continue through both sides.
So the recursive return value must be:

```python
return node.val + max(left, right)
```

This is the best single-side chain starting at the current node.

#### Step 8: Walk the official example slowly

Use:

```text
root = [-10,9,20,null,null,15,7]
```

At node `20`:

- left gain = `15`
- right gain = `7`

So the full-path candidate through `20` is:

```text
20 + 15 + 7 = 42
```

That updates the global answer to `42`.

But the value returned upward to `-10` is only:

```text
20 + max(15, 7) = 35
```

This is why the recursive return value and the global answer must remain separate.

#### Step 9: Connect this to the pruning rule

If one side returns `-5`, the current node should not keep it.
Using `max(gain, 0)` means:

- positive contributions are kept
- harmful branches are cut off

That is the crucial weighted-tree twist in this problem.

### Assemble the Full Code

Now combine the postorder rules into the first full working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0

        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)

        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_path_sum(root))
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
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = float("-inf")

        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return 0

            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)

            ans = max(ans, node.val + left + right)
            return node.val + max(left, right)

        dfs(root)
        return ans
```

### What method did we just build?

You could call it:

- tree DP
- postorder gain propagation
- weighted path divide and conquer

But the main takeaway is:

> A full path through the current node may use both sides, but the value returned upward can use only one side.

---

## E — Engineering

### Scenario 1: Find the highest-value dependency path (Python)

**Background**: each node in a dependency tree has a positive or negative score, and the system wants the most valuable connected path.  
**Why it fits**: negative branches should be dropped, while a local merge point may create the global optimum.

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def best_path(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


root = Node(-5, Node(10), Node(20))
print(best_path(root))
```

### Scenario 2: Find the highest-value propagation path in a service tree (Go)

**Background**: a call tree assigns net value or net cost to each node, and monitoring wants the most valuable connected route.  
**Why it fits**: harmful branches should be pruned, while a local turning point may become the global best.

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func maxPath(root *Node) int {
	if root == nil {
		return 0
	}
	ans := root.Val
	var dfs func(*Node) int
	dfs = func(node *Node) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		if left < 0 {
			left = 0
		}
		right := dfs(node.Right)
		if right < 0 {
			right = 0
		}
		sum := node.Val + left + right
		if sum > ans {
			ans = sum
		}
		if left > right {
			return node.Val + left
		}
		return node.Val + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &Node{Val: -10, Left: &Node{Val: 9}, Right: &Node{Val: 20, Left: &Node{Val: 15}, Right: &Node{Val: 7}}}
	fmt.Println(maxPath(root))
}
```

### Scenario 3: Find the highest-score skill path in a game tree (JavaScript)

**Background**: a game or visualization tree stores positive and negative weights on nodes, and the UI wants the best connected route.  
**Why it fits**: the same "single-side gain + global full path" pattern applies directly.

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function maxPath(root) {
  let ans = -Infinity;
  function dfs(node) {
    if (!node) return 0;
    const left = Math.max(dfs(node.left), 0);
    const right = Math.max(dfs(node.right), 0);
    ans = Math.max(ans, node.val + left + right);
    return node.val + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new Node(-10, new Node(9), new Node(20, new Node(15), new Node(7)));
console.log(maxPath(root));
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)` because each node is processed once
- **Space complexity**: `O(h)` from recursion depth

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Postorder single-side gain + global update | `O(n)` | `O(h)` | Best standard solution |
| Enumerate all paths | very large | very large | Not practical |
| Return full-path sum upward | incorrect | - | The parent cannot extend both sides simultaneously |

### Common Mistakes

1. **Returning the full path through the current node**: that cannot be extended safely by the parent.  
2. **Keeping negative child gains**: that only makes the path worse.  
3. **Initializing the answer with `0`**: that fails on all-negative trees.  
4. **Mixing this with problem 543**: both use postorder, but 124 works with weighted gains and negative pruning.

## FAQ and Notes

### 1. Why is returning `0` for null nodes correct?

Because the recursion returns a gain value, not the final answer.
An empty child contributes no useful gain.

### 2. Why can the global answer not start at `0`?

Because the entire tree may be negative.
For example, a single node `-3` should return `-3`, not `0`.

### 3. How is this different from problem 543?

Both use postorder recursion, but:

- 543 returns height and updates a diameter in edges
- 124 returns gain and updates a path sum in node weights

Problem 124 also needs negative pruning.

## Best Practices

- Ask first what the parent truly needs back from the child
- Keep the full-path candidate and upward gain as separate concepts
- Use `max(gain, 0)` whenever negative branches should be discarded
- Pair this with 543 to sharpen your tree-DP return-value intuition

## S — Summary

- The stable design of problem 124 is: return single-side gain upward, update the global answer with the full path through the node
- A full path can use both sides, but an upward return can use only one
- Negative contributions should be cut off with `max(gain, 0)`
- This problem looks similar to 543, but weighted paths and negative values make the semantics different
- Once you understand this split, many tree-path DP problems become much clearer

## Further Reading

- [LeetCode 124: Binary Tree Maximum Path Sum](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)
- LeetCode 543: Diameter of Binary Tree
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 337: House Robber III

## CTA

Practice `104 + 543 + 124` together.
They form a strong tree-DP ladder: single-side depth, structural path, and weighted path.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_path_sum(root))
```

```c
#include <limits.h>
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
    if (!node) return 0;
    int left = max_int(dfs(node->left, ans), 0);
    int right = max_int(dfs(node->right, ans), 0);
    int candidate = node->val + left + right;
    if (candidate > *ans) *ans = candidate;
    return node->val + max_int(left, right);
}

int maxPathSum(struct TreeNode* root) {
    int ans = INT_MIN;
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
    struct TreeNode* root = new_node(-10);
    root->left = new_node(9);
    root->right = new_node(20);
    root->right->left = new_node(15);
    root->right->right = new_node(7);
    printf("%d\n", maxPathSum(root));
    free_tree(root);
    return 0;
}
```

```cpp
#include <algorithm>
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int dfs(TreeNode* node, int& ans) {
    if (!node) return 0;
    int left = std::max(dfs(node->left, ans), 0);
    int right = std::max(dfs(node->right, ans), 0);
    ans = std::max(ans, node->val + left + right);
    return node->val + std::max(left, right);
}

int maxPathSum(TreeNode* root) {
    int ans = INT_MIN;
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
    TreeNode* root = new TreeNode(-10);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);
    std::cout << maxPathSum(root) << '\n';
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

func maxPathSum(root *TreeNode) int {
	if root == nil {
		return 0
	}
	ans := root.Val
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		if left < 0 {
			left = 0
		}
		right := dfs(node.Right)
		if right < 0 {
			right = 0
		}
		if node.Val+left+right > ans {
			ans = node.Val + left + right
		}
		if left > right {
			return node.Val + left
		}
		return node.Val + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &TreeNode{
		Val:  -10,
		Left: &TreeNode{Val: 9},
		Right: &TreeNode{
			Val:   20,
			Left:  &TreeNode{Val: 15},
			Right: &TreeNode{Val: 7},
		},
	}
	fmt.Println(maxPathSum(root))
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
            let left = dfs(&n.left, ans).max(0);
            let right = dfs(&n.right, ans).max(0);
            *ans = (*ans).max(n.val + left + right);
            n.val + left.max(right)
        }
    }
}

fn max_path_sum(root: &Option<Box<TreeNode>>) -> i32 {
    let mut ans = i32::MIN;
    dfs(root, &mut ans);
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: -10,
        left: Some(Box::new(TreeNode {
            val: 9,
            left: None,
            right: None,
        })),
        right: Some(Box::new(TreeNode {
            val: 20,
            left: Some(Box::new(TreeNode {
                val: 15,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 7,
                left: None,
                right: None,
            })),
        })),
    }));

    println!("{}", max_path_sum(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function maxPathSum(root) {
  let ans = -Infinity;
  function dfs(node) {
    if (!node) return 0;
    const left = Math.max(dfs(node.left), 0);
    const right = Math.max(dfs(node.right), 0);
    ans = Math.max(ans, node.val + left + right);
    return node.val + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new TreeNode(-10, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
console.log(maxPathSum(root));
```
