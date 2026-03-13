---
title: "Hot100: Invert Binary Tree (Recursion / BFS ACERS Guide)"
date: 2026-03-06T17:58:23+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "recursion", "BFS", "tree transformation", "LeetCode 226"]
description: "A practical guide to LeetCode 226 covering tree mirroring, recursive left-right swaps, BFS traversal, and engineering mappings."
keywords: ["Invert Binary Tree", "binary tree inversion", "tree mirror", "recursion", "BFS", "LeetCode 226", "Hot100"]
---

> **Subtitle / Summary**  
> Invert Binary Tree looks tiny, but it is one of the fastest ways to test whether you really understand recursive structure on trees. This guide uses LeetCode 226 to break down the essence of "swap left and right subtrees", covers both recursion and BFS, and shows how the same idea transfers to engineering scenarios.

- **Reading time**: 8-10 min  
- **Tags**: `Hot100`, `binary tree`, `recursion`, `BFS`, `tree transformation`  
- **SEO keywords**: Hot100, Invert Binary Tree, tree mirror, recursion, BFS, LeetCode 226  
- **Meta description**: Learn the recursive and BFS solutions for LeetCode 226, then extend the idea to layout mirroring and structural transformations.  

---

## Target Readers

- Hot100 learners who want to verify whether they truly understand "apply recursion to every node in the whole tree"
- Developers who instinctively start traversing any tree problem, but are unsure when to process the current node
- Engineers who need tree mirroring, layout inversion, or symmetric structural transforms

## Background / Motivation

The code for LeetCode 226 is usually very short, but the thinking pattern is extremely typical:

- What should the current node do?  
  Swap `left` and `right`.

- What is the subproblem?  
  The left and right subtrees must also be inverted.

This is a very pure example of **current operation + recursive handling of identical subproblems**.

If you do not fully internalize this problem, you often end up with mistakes like:

- swapping only the root and forgetting the subtrees
- getting the recursive direction mixed up after the swap
- rebuilding a brand new tree for something that can be done in place

## Core Concepts

- **Tree mirror**: swap the left and right subtree of every node
- **In-place transform**: do not rebuild the whole tree; only swap pointers or references
- **Recursive divide-and-conquer**: after handling the current node, each subtree is still the same kind of problem
- **BFS level-order transform**: you can also swap each node's children level by level

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root node `root` of a binary tree, invert the tree and return the root of the inverted tree.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree, may be null |
| return | TreeNode | root of the inverted tree |

### Example 1

```text
input: root = [4,2,7,1,3,6,9]
output: [4,7,2,9,6,3,1]
explanation:
after swapping the left and right subtrees across the whole tree,
every node is mirrored.
```

### Example 2

```text
input: root = [2,1,3]
output: [2,3,1]
```

### Example 3

```text
input: root = []
output: []
```

### Constraints

- The number of nodes is in the range `[0, 100]`
- `-100 <= Node.val <= 100`

---

## C - Concepts (Core Ideas)

### Thought Process: Why "swap + recursion" is enough

Suppose the current node is `node`. We only need two steps:

1. Swap `node.left` and `node.right`
2. Recursively invert the new left subtree and the new right subtree

The pseudocode is very short:

```text
invert(node):
    if node is null:
        return null
    swap node.left and node.right
    invert(node.left)
    invert(node.right)
    return node
```

### Why this is the complete answer

Because inverting the whole tree is essentially "perform one left-right swap on every node".  
And every local subtree is still a tree, so recursion fits naturally.

### Method Category

- **Tree recursion**
- **In-place structural transform**
- **BFS / queue traversal**

### Recursion vs BFS

1. **Recursion**
   - shortest code
   - matches the recursive definition of a tree
   - recommended as the main solution

2. **BFS**
   - swap nodes level by level
   - useful if you also want to do level statistics or visualization

---

## Practice Guide / Steps

### Recommended Approach: Recursion

1. Handle the null case
2. Swap the left and right child
3. Recursively invert the left subtree
4. Recursively invert the right subtree
5. Return the current node

Runnable Python example:

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root


def level_order(root):
    if root is None:
        return []
    q = deque([root])
    res = []
    while q:
        node = q.popleft()
        res.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res


if __name__ == "__main__":
    root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)),
                    TreeNode(7, TreeNode(6), TreeNode(9)))
    invert_tree(root)
    print(level_order(root))
```

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Mirrored split-pane layout preview (JavaScript)

**Background**: visual editors often store split-pane layouts as binary trees.  
**Why it fits**: a "mirror preview" is essentially swapping the left and right regions at every split node.

```javascript
function Pane(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function invert(node) {
  if (!node) return null;
  [node.left, node.right] = [node.right, node.left];
  invert(node.left);
  invert(node.right);
  return node;
}

const root = new Pane("root", new Pane("left"), new Pane("right"));
console.log(invert(root));
```

### Scenario 2: Tree-mirroring demos in teaching tools (Python)

**Background**: algorithm teaching platforms often need dynamic demonstrations of the "mirror" concept.  
**Why it fits**: the solution to LeetCode 226 is the standard tree-mirroring transform.

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert(node):
    if node is None:
        return None
    node.left, node.right = invert(node.right), invert(node.left)
    return node


root = Node("A", Node("B"), Node("C"))
print(invert(root).left.val)
```

### Scenario 3: Left/right branch inversion tests for rule trees (Go)

**Background**: some rule engines organize "match / no-match" branches as binary trees.  
**Why it fits**: when doing mirror-based tests, this quickly verifies whether the left/right branch logic is symmetric.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func invert(node *Node) *Node {
	if node == nil {
		return nil
	}
	node.Left, node.Right = invert(node.Right), invert(node.Left)
	return node
}

func main() {
	root := &Node{"root", &Node{"allow", nil, nil}, &Node{"deny", nil, nil}}
	root = invert(root)
	fmt.Println(root.Left.Name, root.Right.Name)
}
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is swapped once
- **Space complexity**:
  - Recursion: `O(h)`
  - BFS: `O(w)`, where `w` is the maximum width of the tree

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Recursion | `O(n)` | `O(h)` | Simplest and recommended |
| BFS queue | `O(n)` | `O(w)` | Convenient for level-by-level processing |
| Build a new mirror tree | `O(n)` | `O(n)` | Unnecessary extra memory allocation |

### Common Mistakes and Pitfalls

- Swapping only the root once and forgetting to recurse into the subtrees
- Swapping first, then recursing through stale references and confusing the logic
- Rebuilding a whole new tree even though the job can be done in place
- Confusing "invert binary tree" with "reverse linked list" and mistakenly thinking a linear reconnection order is needed

## Common Questions and Notes

### 1. Can I recurse first and swap later?

Yes. As long as every node eventually completes the left-right swap, the result is correct.  
But "swap first, recurse later" is usually the most intuitive version.

### 2. Which is better in interviews, recursion or BFS?

Recursion is the default choice for this problem. BFS is better treated as an alternative implementation that shows you understand traversal variations.

### 3. Is this preorder or postorder?

It is closer to a preorder-style action: the current node swaps immediately, and only then do we process the subtrees.

## Best Practices and Suggestions

- For tree-transform problems, first ask "what changes at the current node?", then ask "is each subtree the same type of subproblem?"
- If you can swap in place, do that and avoid unnecessary object allocation
- Keep the recursive function semantic simple: input a tree, return the inverted version of that same tree
- Do not just memorize the code; be able to explain verbally why recursion is such a natural fit here

## S - Summary

- The essence of LeetCode 226 is one left-right swap at every node
- Recursion works because every subtree is still the same problem
- This is a classic template of "process current node + recurse on subproblems"
- BFS also works, but recursion expresses the idea more directly
- In engineering, the same pattern applies to layout mirroring, visualization mirroring, and symmetry tests on rule trees

## References and Further Reading

- [LeetCode 226: Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- LeetCode 101: Symmetric Tree
- LeetCode 100: Same Tree
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 102: Binary Tree Level Order Traversal

## CTA

Try solving 226, 101, and 100 together.  
One trains structural transformation, one trains structural comparison, and together they make tree recursion feel much more solid.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root


def level_order(root):
    if root is None:
        return []
    q = deque([root])
    res = []
    while q:
        node = q.popleft()
        res.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res


if __name__ == "__main__":
    root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)),
                    TreeNode(7, TreeNode(6), TreeNode(9)))
    invert_tree(root)
    print(level_order(root))
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

struct TreeNode* invertTree(struct TreeNode* root) {
    if (root == NULL) return NULL;
    struct TreeNode* tmp = root->left;
    root->left = invertTree(root->right);
    root->right = invertTree(tmp);
    return root;
}

void preorder(struct TreeNode* root) {
    if (!root) return;
    printf("%d ", root->val);
    preorder(root->left);
    preorder(root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(4);
    root->left = new_node(2);
    root->right = new_node(7);
    root->left->left = new_node(1);
    root->left->right = new_node(3);
    root->right->left = new_node(6);
    root->right->right = new_node(9);

    invertTree(root);
    preorder(root);
    printf("\n");
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <utility>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;
    std::swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    return root;
}

void preorder(TreeNode* root) {
    if (!root) return;
    std::cout << root->val << ' ';
    preorder(root->left);
    preorder(root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(7);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(9);

    invertTree(root);
    preorder(root);
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

func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
	return root
}

func preorder(root *TreeNode) {
	if root == nil {
		return
	}
	fmt.Print(root.Val, " ")
	preorder(root.Left)
	preorder(root.Right)
}

func main() {
	root := &TreeNode{
		Val: 4,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 1},
			Right: &TreeNode{Val: 3},
		},
		Right: &TreeNode{
			Val:   7,
			Left:  &TreeNode{Val: 6},
			Right: &TreeNode{Val: 9},
		},
	}

	invertTree(root)
	preorder(root)
	fmt.Println()
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn invert_tree(root: &mut Option<Box<TreeNode>>) {
    if let Some(node) = root {
        std::mem::swap(&mut node.left, &mut node.right);
        invert_tree(&mut node.left);
        invert_tree(&mut node.right);
    }
}

fn preorder(root: &Option<Box<TreeNode>>) {
    if let Some(node) = root {
        print!("{} ", node.val);
        preorder(&node.left);
        preorder(&node.right);
    }
}

fn main() {
    let mut root = Some(Box::new(TreeNode {
        val: 4,
        left: Some(Box::new(TreeNode {
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
        })),
        right: Some(Box::new(TreeNode {
            val: 7,
            left: Some(Box::new(TreeNode {
                val: 6,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 9,
                left: None,
                right: None,
            })),
        })),
    }));

    invert_tree(&mut root);
    preorder(&root);
    println!();
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function invertTree(root) {
  if (!root) return null;
  [root.left, root.right] = [invertTree(root.right), invertTree(root.left)];
  return root;
}

function preorder(root, out = []) {
  if (!root) return out;
  out.push(root.val);
  preorder(root.left, out);
  preorder(root.right, out);
  return out;
}

const root = new TreeNode(
  4,
  new TreeNode(2, new TreeNode(1), new TreeNode(3)),
  new TreeNode(7, new TreeNode(6), new TreeNode(9))
);

invertTree(root);
console.log(preorder(root));
```
