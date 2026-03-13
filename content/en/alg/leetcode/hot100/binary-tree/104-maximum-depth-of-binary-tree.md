---
title: "Hot100: Maximum Depth of Binary Tree (DFS / BFS ACERS Guide)"
date: 2026-03-06T17:58:22+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "DFS", "BFS", "recursion", "LeetCode 104"]
description: "A practical guide to LeetCode 104 covering the depth definition, recursive DFS, level-order BFS, engineering mappings, and runnable multi-language implementations."
keywords: ["Maximum Depth of Binary Tree", "DFS", "BFS", "recursion", "LeetCode 104", "Hot100"]
---

> **Subtitle / Summary**  
> "Maximum depth" is one of the cleanest starting points for tree recursion. Once you truly understand that the answer for the current tree depends on the answers from its left and right subtrees, a whole family of tree DP and DFS problems becomes easier. This guide uses LeetCode 104 to explain recursive DFS, level-order BFS, and the engineering value of the same pattern.

- **Reading time**: 9-11 min  
- **Tags**: `Hot100`, `binary tree`, `DFS`, `BFS`, `recursion`  
- **SEO keywords**: Hot100, Maximum Depth of Binary Tree, DFS, BFS, LeetCode 104  
- **Meta description**: Learn the DFS and BFS solutions for LeetCode 104 from the definition of depth, with engineering mappings and runnable multi-language code.  

---

## Target Readers

- Learners who are just starting tree problems and want to truly internalize "tree recursion return values"
- Developers who can write traversals but get confused once the task becomes "compute height", "compute path", or "compute an answer"
- Engineers who need depth analysis on hierarchical data such as menus, org charts, or nested JSON

## Background / Motivation

LeetCode 104 looks like an easy problem, but it is almost the parent problem of tree recursion:

- you first need to answer "**what is the depth of an empty tree?**"
- then answer "**who determines the answer for the current node?**"
- and finally write the relation as `1 + max(left, right)`

Once this recursive definition is built correctly, later problems such as balanced binary tree, tree diameter, path sums, and lowest common ancestor become much easier to enter.

## Core Concepts

- **Depth / height**: in this problem, the number of nodes on the longest path from root to the farthest leaf
- **Postorder-style thinking**: to know the answer for the current node, you must first know the answers of the left and right subtrees
- **DFS**: recurse downward and combine answers while backtracking
- **BFS**: traverse level by level; the last level number is the tree depth

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root node `root` of a binary tree, return its **maximum depth**.

Maximum depth means the number of nodes along the longest path from the root down to the farthest leaf node.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree, may be null |
| return | int | maximum depth of the tree |

### Example 1

```text
input: root = [3,9,20,null,null,15,7]
output: 3
explanation:
level 1: 3
level 2: 9, 20
level 3: 15, 7
so the maximum depth is 3.
```

### Example 2

```text
input: root = [1,null,2]
output: 2
```

### Constraints

- The number of nodes is in the range `[0, 10^4]`
- `-100 <= Node.val <= 100`

---

## C - Concepts (Core Ideas)

### Thought Process: Why the recursive formula is `1 + max(left, right)`

For any node `node`:

- if it is null, the depth is `0`
- if it is not null, then the maximum depth from that node is:
  - `1` for the current level
  - plus the deeper side between the left subtree and the right subtree

So the state transition is direct:

```text
depth(node) = 1 + max(depth(node.left), depth(node.right))
```

### Method Category

- **Tree recursion / DFS**
- **Level-order traversal / BFS**
- **Bottom-up answer combination in tree problems**

### When DFS and BFS are each a good fit

1. **Recursive DFS**
   - shortest code
   - matches the definition best
   - ideal for most interviews and explanations

2. **Level-order BFS**
   - very convenient for problems that are naturally layer-based
   - if you also want the node distribution by level, BFS is more direct

### Why DFS is the recommended template here

This problem does not ask you to print each level; it only asks for one final number.  
DFS writes the definition directly, gives the clearest expression, and has the lowest error rate.

---

## Practice Guide / Steps

### Recommended Approach: Recursive DFS

1. If the node is null, return `0`
2. Recursively compute the maximum depth of the left subtree
3. Recursively compute the maximum depth of the right subtree
4. Return `1 + max(leftDepth, rightDepth)`

Runnable Python example:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_depth(root))
```

### BFS Alternative

If you prefer level-order traversal, you can also:

1. Use a queue to hold the current level
2. Increase depth after processing one full level
3. Stop when the queue becomes empty

This method is also common, especially when the problem also asks for level-by-level output.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Maximum nesting depth of frontend menu config (JavaScript)

**Background**: backend systems often allow menus to be configured as trees.  
**Why it fits**: before release, you can check whether the menu exceeds the maximum nesting level allowed by the design.

```javascript
const menu = {
  name: "root",
  children: [
    { name: "dashboard", children: [] },
    { name: "settings", children: [{ name: "profile", children: [] }] },
  ],
};

function depth(node) {
  if (!node) return 0;
  if (!node.children || node.children.length === 0) return 1;
  return 1 + Math.max(...node.children.map(depth));
}

console.log(depth(menu));
```

### Scenario 2: Longest reporting chain in an org chart (Go)

**Background**: org charts and approval flows are often represented as trees.  
**Why it fits**: maximum depth measures hierarchy complexity and helps with workflow optimization and permission design.

```go
package main

import "fmt"

type Node struct {
	Name     string
	Children []*Node
}

func depth(node *Node) int {
	if node == nil {
		return 0
	}
	best := 0
	for _, child := range node.Children {
		if d := depth(child); d > best {
			best = d
		}
	}
	return 1 + best
}

func main() {
	root := &Node{
		Name: "CEO",
		Children: []*Node{
			{
				Name: "VP",
				Children: []*Node{
					{Name: "Manager"},
				},
			},
		},
	}
	fmt.Println(depth(root))
}
```

### Scenario 3: Maximum nesting validation for JSON (Python)

**Background**: logs, configs, and ETL payloads often contain deeply nested JSON.  
**Why it fits**: overly deep data hurts readability and downstream processing, so it is useful to enforce a depth limit at the input boundary.

```python
def json_depth(x):
    if isinstance(x, dict):
        if not x:
            return 1
        return 1 + max(json_depth(v) for v in x.values())
    if isinstance(x, list):
        if not x:
            return 1
        return 1 + max(json_depth(v) for v in x)
    return 1


data = {"a": {"b": {"c": [1, {"d": 2}]}}}
print(json_depth(data))
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is visited once
- **Space complexity**:
  - DFS recursion: `O(h)`
  - BFS queue: worst case `O(n)`, or more precisely `O(w)` where `w` is the maximum tree width

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| DFS recursion | `O(n)` | `O(h)` | Matches the definition best and is recommended |
| BFS level order | `O(n)` | `O(w)` | Very convenient for level-based problems |
| Explicit-stack DFS | `O(n)` | `O(h)` | Useful if you do not want recursion |

### Common Mistakes and Pitfalls

- Writing the depth of a null node as `1`, which adds an extra level to the whole tree
- Mixing up "edge count" and "node count"; this problem counts **nodes**
- Recursing into only one side and forgetting `max(left, right)`
- Incrementing depth on every popped BFS node, which counts nodes instead of levels

## Common Questions and Notes

### 1. Is this preorder, inorder, or postorder?

More accurately, it follows a **postorder-style merge** pattern, because the answer for the current node depends on the answers of its left and right subtrees.

### 2. Which is better, DFS or BFS?

If you only need one depth value, DFS is simpler. If you also want the nodes grouped by level, BFS is more natural.

### 3. Can recursion overflow the stack?

Yes, for extremely degenerate trees. In engineering situations where tree depth is unbounded, explicit stacks or BFS are safer.

## Best Practices and Suggestions

- Write the base case clearly first: what should the function return when `node == null`?
- For tree problems where the current answer depends on the left and right subtree answers, think of recursive return values first
- When writing complexity, distinguish `O(h)` from `O(w)` for a more accurate statement
- Being able to explain when DFS or BFS is appropriate matters more than simply memorizing code

## S - Summary

- The core of LeetCode 104 is not the code, but the definition of depth itself
- Once `depth(node) = 1 + max(left, right)` is clear, the recursive solution almost writes itself
- DFS is the most recommended template for this problem, while BFS is an excellent level-based alternative
- This problem is foundational for balanced-tree, diameter, and path-sum questions
- In engineering, any hierarchical structure with "maximum nesting depth" can reuse the same idea

## References and Further Reading

- [LeetCode 104: Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- LeetCode 111: Minimum Depth of Binary Tree
- LeetCode 110: Balanced Binary Tree
- LeetCode 543: Diameter of Binary Tree
- LeetCode 102: Binary Tree Level Order Traversal

## CTA

It is worth practicing 104 together with 111.  
One is about `max`, while the other often exposes tricky null-subtree handling; together they make your tree-recursion base cases much more stable.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_depth(root))
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

int maxDepth(struct TreeNode* root) {
    if (root == NULL) return 0;
    int left = maxDepth(root->left);
    int right = maxDepth(root->right);
    return 1 + (left > right ? left : right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(3);
    root->left = new_node(9);
    root->right = new_node(20);
    root->right->left = new_node(15);
    root->right->right = new_node(7);
    printf("%d\n", maxDepth(root));
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

int maxDepth(TreeNode* root) {
    if (!root) return 0;
    int left = maxDepth(root->left);
    int right = maxDepth(root->right);
    return 1 + std::max(left, right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);
    std::cout << maxDepth(root) << '\n';
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

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := maxDepth(root.Left)
	right := maxDepth(root.Right)
	if left > right {
		return 1 + left
	}
	return 1 + right
}

func main() {
	root := &TreeNode{
		Val: 3,
		Left: &TreeNode{Val: 9},
		Right: &TreeNode{
			Val:   20,
			Left:  &TreeNode{Val: 15},
			Right: &TreeNode{Val: 7},
		},
	}
	fmt.Println(maxDepth(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn max_depth(root: &Option<Box<TreeNode>>) -> i32 {
    match root {
        None => 0,
        Some(node) => 1 + max_depth(&node.left).max(max_depth(&node.right)),
    }
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 3,
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

    println!("{}", max_depth(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function maxDepth(root) {
  if (!root) return 0;
  return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}

const root = new TreeNode(
  3,
  new TreeNode(9),
  new TreeNode(20, new TreeNode(15), new TreeNode(7))
);

console.log(maxDepth(root));
```
