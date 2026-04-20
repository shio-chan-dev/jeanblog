---
title: "Hot100: Binary Tree Right Side View (Level Order Last-Node Rule ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "BFS", "level order", "queue", "LeetCode 199"]
description: "A practical guide to LeetCode 199 covering why the right side view is just the last node of each level, plus two stable solutions: level-order BFS and right-first DFS."
keywords: ["Binary Tree Right Side View", "BFS", "level order", "queue", "right-first DFS", "LeetCode 199", "Hot100"]
---

> **Subtitle / Summary**
> LeetCode 199 is not really about visual imagination. It is about translating a viewpoint problem into a level problem. Once you realize that the right side view is simply the last node of each level, the problem becomes a standard breadth-first traversal.

- **Reading time**: 10-13 min
- **Tags**: `Hot100`, `binary tree`, `BFS`, `level order`, `queue`
- **SEO keywords**: Binary Tree Right Side View, BFS, level order, queue, right-first DFS, LeetCode 199
- **Meta description**: Learn LeetCode 199 from the core equivalence "right side view = last node of each level", with step-by-step derivation, engineering mappings, and runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given the root `root` of a binary tree, imagine standing on its right side and return the values of the nodes you can see from top to bottom.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| return | `int[]` | node values visible from the right side, top to bottom |

### Example 1

```text
input: root = [1,2,3,null,5,null,4]
output: [1,3,4]
```

### Example 2

```text
input: root = [1,2,3,4,null,null,null,5]
output: [1,3,4,5]
```

### Example 3

```text
input: root = [1,null,3]
output: [1,3]
```

### Example 4

```text
input: root = []
output: []
```

### Constraints

- The number of nodes in the tree is in the range `[0, 100]`
- `-100 <= Node.val <= 100`

---

## Target Readers

- Hot100 learners who already know level order traversal but want to handle "view" variations cleanly
- Developers who get distracted by the visual wording of the problem
- Readers who want to connect problem 199 directly to problem 102

## Background / Motivation

This problem is useful because it trains one important move:

- convert a visual description into a structural rule

When people first read "right side view", they often wonder:

- do I need coordinates?
- do I need actual geometry?
- do I need to simulate visibility?

None of that is necessary.
The phrase "from top to bottom" already tells you that the answer is organized by levels.

So the real question is:

> Which single node should survive from each level?

The answer is:

> the rightmost one

And with left-to-right level processing, that becomes:

> the last processed node of the level

## Core Concepts

- **Level order traversal**: visit the tree layer by layer
- **Queue**: the natural container for BFS
- **Level width control**: fix the current level size before processing that level
- **Right side view**: either the last node of each level (BFS) or the first node visited at each depth in right-first DFS

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the visible meaning of one level

Take:

```text
    1
   / \
  2   3
   \   \
    5   4
```

From the right side:

- level 1 shows `1`
- level 2 shows `3`
- level 3 shows `4`

So the answer is:

```text
[1,3,4]
```

This shows the problem is not asking for one literal right-edge chain.
It is asking for the visible representative of each level.

#### Step 2: Decide what state the algorithm needs

Since the answer is level-based, BFS is the most direct fit.

So the first state variables are:

- `queue`
- `ans`

#### Step 3: Define the smaller subproblem

Instead of "solve the whole tree at once", the repeated task is:

> Process one level, then decide which node from this level should be kept.

To do that, the algorithm must know how many nodes belong to the current level:

```python
level_size = len(queue)
```

#### Step 4: Define when one level is complete

Once exactly `level_size` nodes have been popped, the level is done.

The last one popped is the rightmost node of that level, so:

```python
if i == level_size - 1:
    ans.append(node.val)
```

#### Step 5: Expand the next frontier normally

Nothing special happens here.
For each node in the current level:

```python
if node.left:
    queue.append(node.left)
if node.right:
    queue.append(node.right)
```

This is just regular level order traversal.

#### Step 6: Why does the last node of the level equal the right side view?

Because within a level, BFS processes nodes from left to right.

So:

- earlier processed nodes are more leftward
- the last processed node is the rightmost one

That is exactly the node that remains visible from the right side.

#### Step 7: Connect this to an alternative DFS view

There is another way to think about the same answer:

- traverse right child before left child
- at each depth, record the first node you see

That first node is also the rightmost visible node at that depth.

So the two stable views are:

- BFS: keep the last node of each level
- right-first DFS: keep the first node of each depth

#### Step 8: Walk the official example slowly

Use:

```text
root = [1,2,3,null,5,null,4]
```

Level 1:

- nodes: `1`
- last node: `1`

Level 2:

- nodes: `2, 3`
- last node: `3`

Level 3:

- nodes: `5, 4`
- last node: `4`

So the answer is `[1,3,4]`.

#### Step 9: Keep the derivation simple

There is no need for coordinates or geometry.
The problem becomes easy as soon as the right-side view is rewritten as:

> one kept node per level

### Assemble the Full Code

Now combine the level-order rules into the first full working solution.

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def right_side_view(root):
    if root is None:
        return []

    ans = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if i == level_size - 1:
                ans.append(node.val)

    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
    print(right_side_view(root))
```

### Reference Answer

The LeetCode-style submission version is:

```python
from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []

        ans = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == level_size - 1:
                    ans.append(node.val)

        return ans
```

### What method did we just build?

You could call it:

- level order traversal
- one-node-per-level selection
- BFS with level representatives

But the core insight is:

> The right side view is not a geometry problem. It is a per-level selection problem.

---

## E — Engineering

### Scenario 1: Show the outermost owner of each org layer (Python)

**Background**: a compressed org chart may show just one representative from each level.  
**Why it fits**: this is structurally the same as choosing one visible node per level.

```python
from collections import deque


class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def rightmost_each_level(root):
    if root is None:
        return []
    q = deque([root])
    ans = []
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if i == size - 1:
                ans.append(node.name)
    return ans


root = Node("CEO", Node("Eng"), Node("Sales"))
print(rightmost_each_level(root))
```

### Scenario 2: Pick one exposed service per topology layer (Go)

**Background**: a simplified service dependency view may keep only one representative service from each layer.  
**Why it fits**: the "last node per layer" rule works on any layered tree traversal.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func rightView(root *Node) []string {
	if root == nil {
		return nil
	}
	q := []*Node{root}
	ans := []string{}
	for len(q) > 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
			if i == size-1 {
				ans = append(ans, node.Name)
			}
		}
	}
	return ans
}

func main() {
	root := &Node{Name: "api", Left: &Node{Name: "auth"}, Right: &Node{Name: "feed"}}
	fmt.Println(rightView(root))
}
```

### Scenario 3: Keep one endpoint from each menu depth (JavaScript)

**Background**: a compact navigation preview may display only one endpoint from each tree depth.  
**Why it fits**: the same level-order selection logic applies.

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function rightView(root) {
  if (!root) return [];
  const q = [root];
  const ans = [];
  while (q.length) {
    const size = q.length;
    for (let i = 0; i < size; i++) {
      const node = q.shift();
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
      if (i === size - 1) ans.push(node.name);
    }
  }
  return ans;
}

const root = new Node("Home", new Node("Docs"), new Node("Blog"));
console.log(rightView(root));
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)` because each node is visited once
- **Space complexity**: `O(w)` where `w` is the maximum width of the tree

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| BFS by level | `O(n)` | `O(w)` | Most direct solution |
| Right-first DFS | `O(n)` | `O(h)` | Elegant recursive alternative |
| Collect full levels then take last element | `O(n)` | higher | Works, but stores more than needed |

### Common Mistakes

1. **Mistaking the answer for one literal right-edge chain**: that fails on trees where the visible nodes come from different branches.  
2. **Not fixing the level size first**: then levels mix together and "last node" loses its meaning.  
3. **Thinking right-first traversal is required for BFS**: it is not; left-to-right BFS still makes the last node the rightmost one.  
4. **Forgetting the empty-tree case**: `[]` should return an empty list immediately.

## FAQ and Notes

### 1. Why does left-to-right BFS still produce the rightmost node last?

Because nodes within one level are processed from left to right.
So the final node of that level is the rightmost one.

### 2. How is this related to problem 102?

Problem 199 is almost a direct variant of problem 102:

- 102 keeps the whole level
- 199 keeps only the last node of the level

### 3. Why does right-first DFS also work?

Because if you always go right before left, the first node you see at a new depth is the rightmost visible one.

## Best Practices

- If a tree problem says "from top to bottom" or "each level", think BFS early
- Always capture `level_size` before processing the level
- Keep the derivation simple: one kept node per level
- Pair this with problems 102 and 637 to stabilize level-order patterns

## S — Summary

- The right side view is best understood as one selected node per level
- In BFS, that node is simply the last node of the level
- Right-first DFS gives the same answer from a depth-first perspective
- Problem 199 is very close to problem 102 conceptually
- Once you see the level equivalence, many "tree view" problems become much simpler

## Further Reading

- [LeetCode 199: Binary Tree Right Side View](https://leetcode.cn/problems/binary-tree-right-side-view/)
- LeetCode 102: Binary Tree Level Order Traversal
- LeetCode 637: Average of Levels in Binary Tree
- LeetCode 103: Binary Tree Zigzag Level Order Traversal

## CTA

Try `102 + 199 + 637` together.
They all train the same skill: process one level at a time, then decide what summary to keep.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def right_side_view(root):
    if root is None:
        return []
    ans = []
    q = deque([root])
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if i == size - 1:
                ans.append(node.val)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
    print(right_side_view(root))
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

int* rightSideView(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    if (!root) return NULL;

    struct TreeNode* queue[256];
    int front = 0, back = 0;
    int* ans = (int*)malloc(sizeof(int) * 256);
    queue[back++] = root;

    while (front < back) {
        int size = back - front;
        for (int i = 0; i < size; ++i) {
            struct TreeNode* node = queue[front++];
            if (node->left) queue[back++] = node->left;
            if (node->right) queue[back++] = node->right;
            if (i == size - 1) ans[(*returnSize)++] = node->val;
        }
    }
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
    root->left->right = new_node(5);
    root->right->right = new_node(4);

    int size = 0;
    int* ans = rightSideView(root, &size);
    for (int i = 0; i < size; ++i) printf("%d ", ans[i]);
    printf("\n");

    free(ans);
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <queue>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

std::vector<int> rightSideView(TreeNode* root) {
    if (!root) return {};
    std::queue<TreeNode*> q;
    std::vector<int> ans;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
            if (i == size - 1) ans.push_back(node->val);
        }
    }
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
    root->left->right = new TreeNode(5);
    root->right->right = new TreeNode(4);
    for (int x : rightSideView(root)) std::cout << x << ' ';
    std::cout << '\n';
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

func rightSideView(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	q := []*TreeNode{root}
	ans := []int{}
	for len(q) > 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
			if i == size-1 {
				ans = append(ans, node.Val)
			}
		}
	}
	return ans
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Right: &TreeNode{Val: 5},
		},
		Right: &TreeNode{
			Val:   3,
			Right: &TreeNode{Val: 4},
		},
	}
	fmt.Println(rightSideView(root))
}
```

```rust
use std::collections::VecDeque;

#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn right_side_view(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    let mut ans = Vec::new();
    let mut q: VecDeque<&TreeNode> = VecDeque::new();
    if let Some(node) = root.as_deref() {
        q.push_back(node);
    } else {
        return ans;
    }

    while !q.is_empty() {
        let size = q.len();
        for i in 0..size {
            let node = q.pop_front().unwrap();
            if let Some(left) = node.left.as_deref() {
                q.push_back(left);
            }
            if let Some(right) = node.right.as_deref() {
                q.push_back(right);
            }
            if i + 1 == size {
                ans.push(node.val);
            }
        }
    }
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 5,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
    }));

    println!("{:?}", right_side_view(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function rightSideView(root) {
  if (!root) return [];
  const q = [root];
  const ans = [];
  while (q.length) {
    const size = q.length;
    for (let i = 0; i < size; i++) {
      const node = q.shift();
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
      if (i === size - 1) ans.push(node.val);
    }
  }
  return ans;
}

const root = new TreeNode(1, new TreeNode(2, null, new TreeNode(5)), new TreeNode(3, null, new TreeNode(4)));
console.log(rightSideView(root));
```
