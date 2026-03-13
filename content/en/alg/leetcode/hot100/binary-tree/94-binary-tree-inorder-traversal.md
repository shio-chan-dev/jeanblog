---
title: "Hot100: Binary Tree Inorder Traversal (Recursion / Stack ACERS Guide)"
date: 2026-03-06T17:58:21+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "DFS", "stack", "inorder traversal", "LeetCode 94"]
description: "A practical guide to LeetCode 94 covering left-root-right traversal, recursion, explicit stacks, engineering mappings, and runnable multi-language implementations."
keywords: ["Binary Tree Inorder Traversal", "inorder traversal", "explicit stack", "DFS", "LeetCode 94", "Hot100"]
---

> **Subtitle / Summary**  
> Binary tree traversal is the starting point of most tree templates, and inorder traversal is one of the cleanest problems for understanding both recursive thinking and explicit stack simulation. This ACERS guide uses LeetCode 94 to explain the left-root-right order, the iterative stack template, and why the pattern matters in real engineering work.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `binary tree`, `DFS`, `stack`, `inorder traversal`  
- **SEO keywords**: Hot100, Binary Tree Inorder Traversal, inorder traversal, explicit stack, LeetCode 94  
- **Meta description**: A systematic guide to LeetCode 94 from recursion to explicit stacks, with engineering scenarios and runnable multi-language implementations.  

---

## Target Readers

- Hot100 learners who want to lock in a stable tree-traversal template
- Developers moving from arrays and linked lists to trees, and still mixing up preorder, inorder, and postorder
- Engineers who want to reuse the left-root-right idea in BSTs, expression trees, or syntax trees

## Background / Motivation

Inorder traversal is not hard by itself, but its training value is high:

- it is one of the easiest problems for building intuition that **recursion = implicit stack**, while iteration = **explicit stack**
- it helps you internalize the process of "go left all the way, backtrack to visit the root, then move into the right subtree"
- in a **binary search tree (BST)**, inorder traversal naturally produces a sorted sequence, so the engineering value is very real

When many people first solve tree problems, the issue is not the logic itself, but:

- not being sure which node gets visited first
- not knowing exactly when to push and pop in the iterative version
- getting the code tangled when the tree is empty or degenerates into a one-sided chain

If you master this template, later problems like validating a BST, finding the k-th smallest element, or recovering a BST become much smoother.

## Core Concepts

- **Inorder traversal**: visit in the order `left subtree -> root node -> right subtree`
- **DFS (depth-first search)**: the most common organization pattern for tree traversal; inorder is one specific visitation order
- **Explicit stack**: manually simulate the recursion call stack by storing nodes you still need to come back to
- **Tree height h**: space complexity is usually written as `O(h)`; for balanced trees this is about `O(log n)`, and for a degenerate chain it becomes `O(n)`

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root node `root` of a binary tree, return its **inorder traversal** result.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree, may be null |
| return | `int[]` / `List[int]` | node values in inorder sequence |

### Example 1

```text
input: root = [1,null,2,3]
output: [1,3,2]
explanation:
    1
     \
      2
     /
    3

The inorder order is left -> root -> right, so the answer is [1,3,2].
```

### Example 2

```text
input: root = []
output: []
```

### Example 3

```text
input: root = [1]
output: [1]
```

### Constraints

- The number of nodes is in the range `[0, 100]`
- `-100 <= Node.val <= 100`

---

## C - Concepts (Core Ideas)

### Thought Process: From recursive definition to explicit stack template

1. **The most natural form is recursion**  
   For each node `node`:
   - traverse the left subtree first
   - visit the current node
   - traverse the right subtree last

   That matches the definition of inorder exactly, so the code is very short.

2. **But interviews often ask: can you do it without recursion?**  
   Since recursion relies on the function call stack, interviewers often want you to write that process out explicitly.

3. **Why do we keep pushing nodes while going left?**  
   Because inorder requires the left subtree to be processed first. So as long as the current node is not null, we push it and continue to `left`.  
   Once we hit null, the leftmost chain is exhausted, and the top of the stack is exactly the next root we should visit.

### Method Category

- **Tree DFS**
- **Recursive traversal**
- **Stack-based recursion simulation**

### Explicit Stack Template

The iterative version can be remembered in four stable steps:

1. `cur = root`
2. While `cur != null`, keep pushing and move left
3. After the left side ends, pop the stack top and record its value
4. Move `cur` to the right subtree of the popped node, then repeat

Pseudo flow:

```text
while cur is not null or stack is not empty:
    while cur is not null:
        stack.push(cur)
        cur = cur.left

    cur = stack.pop()
    record cur.val
    cur = cur.right
```

### Why this order is always correct

- Each node is visited exactly once during left-chain backtracking
- The left subtree always finishes before the node itself
- The right subtree only starts after the root node has been visited

That is exactly equivalent to the definition of inorder traversal, so the result is correct.

---

## Practice Guide / Steps

### Recommended Approach: Iterative explicit stack

1. Prepare the result array `res` and a stack `stack`
2. Start `cur` from the root
3. Keep pushing the left chain
4. Pop and visit the root
5. Move into the right subtree
6. Stop when both the stack is empty and `cur` is null

Runnable Python example:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    res = []
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res


if __name__ == "__main__":
    root = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
    print(inorder_traversal(root))
```

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Export sorted primary keys from a BST (Python)

**Background**: many in-memory indexes, cache dictionaries, and teaching-oriented search trees store data in BST form.  
**Why it fits**: inorder traversal of a BST naturally yields ascending order, so it is useful for audit exports or debug snapshots.

```python
class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


def inorder(node, out):
    if node is None:
        return
    inorder(node.left, out)
    out.append(node.key)
    inorder(node.right, out)


root = Node(5, Node(3, Node(2), Node(4)), Node(7))
result = []
inorder(root, result)
print(result)
```

### Scenario 2: Convert an expression tree to infix notation (JavaScript)

**Background**: compilers, formula editors, and rule engines often organize expressions as binary trees.  
**Why it fits**: inorder traversal naturally matches the reading order of infix expressions, which makes the result more human-friendly.

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function inorder(node) {
  if (!node) return "";
  if (!node.left && !node.right) return String(node.val);
  return `(${inorder(node.left)} ${node.val} ${inorder(node.right)})`;
}

const tree = new Node("*", new Node("+", new Node(1), new Node(2)), new Node(3));
console.log(inorder(tree));
```

### Scenario 3: Inspect local ordering in a tree-based config (Go)

**Background**: some rule systems use "left branch / current node / right branch" as a stable manual inspection order.  
**Why it fits**: inorder traversal lets developers inspect nodes in a fixed local order, which helps with diffs and manual verification.

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func inorder(node *Node, out *[]string) {
	if node == nil {
		return
	}
	inorder(node.Left, out)
	*out = append(*out, node.Name)
	inorder(node.Right, out)
}

func main() {
	root := &Node{"root", &Node{"L", nil, nil}, &Node{"R", nil, nil}}
	order := []string{}
	inorder(root, &order)
	fmt.Println(order)
}
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is processed exactly once
- **Space complexity**:
  - Recursive version: `O(h)` call stack
  - Explicit-stack version: `O(h)` auxiliary stack

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Recursion | `O(n)` | `O(h)` | Most intuitive and shortest |
| Explicit stack | `O(n)` | `O(h)` | Most common interview template and highly reusable |
| Morris traversal | `O(n)` | `O(1)` | Temporarily modifies tree structure and is harder to reason about |

### Common Mistakes and Pitfalls

- Mixing up the visitation points of preorder, inorder, and postorder
- Forgetting to move to `cur.right` after popping in the iterative version
- Writing only `while cur != null` and missing the case where nodes are still waiting in the stack
- Accessing `node.left` directly in recursion without a null check first

## Common Questions and Notes

### 1. Is inorder traversal always sorted?

No. It is sorted only when the tree satisfies the **BST property**.

### 2. Which is more recommended, recursion or iteration?

In interviews, you should know both. Early on, recursion is the best way to build the definition in your head; after that, the explicit stack template is the most stable pattern to memorize.

### 3. Is Morris traversal worth memorizing?

It is worth understanding, but it should not be your first priority at the fundamentals stage. Get recursion and explicit stacks stable first.

## Best Practices and Suggestions

- Memorize the definition in one sentence: **left, root, right**
- For the iterative template, remember: "push left chain -> pop and visit -> move right"
- Whenever you see BST, think of "inorder = sorted"
- For tree problems, writing space as `O(h)` is often more accurate than writing only `O(n)`

## S - Summary

- The core of inorder traversal is the fixed visitation order: `left -> root -> right`
- Recursion matches the definition best, while the explicit stack version is the best interview template
- This problem trains two core abilities: tree recursion and manual simulation of the call stack
- In BSTs, expression trees, and configuration trees, inorder thinking has practical engineering value
- Once you can write 94 smoothly, BST validation and k-th-smallest problems become much easier

## References and Further Reading

- [LeetCode 94: Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- LeetCode 144: Binary Tree Preorder Traversal
- LeetCode 145: Binary Tree Postorder Traversal
- LeetCode 98: Validate Binary Search Tree
- LeetCode 230: Kth Smallest Element in a BST

## CTA

First handwrite the recursive version, then rewrite the explicit-stack version without looking at the answer.  
If you can reliably finish LeetCode 94 in about three minutes, your tree-traversal fundamentals are already in place.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    res = []
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res


if __name__ == "__main__":
    root = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
    print(inorder_traversal(root))
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

int* inorderTraversal(struct TreeNode* root, int* returnSize) {
    struct TreeNode* stack[128];
    int top = 0;
    int* res = (int*)malloc(sizeof(int) * 128);
    *returnSize = 0;
    struct TreeNode* cur = root;

    while (cur != NULL || top > 0) {
        while (cur != NULL) {
            stack[top++] = cur;
            cur = cur->left;
        }
        cur = stack[--top];
        res[(*returnSize)++] = cur->val;
        cur = cur->right;
    }
    return res;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->right = new_node(2);
    root->right->left = new_node(3);

    int n = 0;
    int* ans = inorderTraversal(root, &n);
    for (int i = 0; i < n; ++i) {
        printf("%d%s", ans[i], i + 1 == n ? "\n" : " ");
    }

    free(ans);
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <stack>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

std::vector<int> inorderTraversal(TreeNode* root) {
    std::vector<int> res;
    std::stack<TreeNode*> st;
    TreeNode* cur = root;
    while (cur || !st.empty()) {
        while (cur) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        res.push_back(cur->val);
        cur = cur->right;
    }
    return res;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->right = new TreeNode(2);
    root->right->left = new TreeNode(3);

    auto ans = inorderTraversal(root);
    for (size_t i = 0; i < ans.size(); ++i) {
        std::cout << ans[i] << (i + 1 == ans.size() ? '\n' : ' ');
    }

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

func inorderTraversal(root *TreeNode) []int {
	res := []int{}
	stack := []*TreeNode{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, cur.Val)
		cur = cur.Right
	}
	return res
}

func main() {
	root := &TreeNode{Val: 1}
	root.Right = &TreeNode{Val: 2, Left: &TreeNode{Val: 3}}
	fmt.Println(inorderTraversal(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn inorder_traversal(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    fn dfs(node: &Option<Box<TreeNode>>, res: &mut Vec<i32>) {
        if let Some(node) = node {
            dfs(&node.left, res);
            res.push(node.val);
            dfs(&node.right, res);
        }
    }

    let mut res = vec![];
    dfs(root, &mut res);
    res
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: None,
        right: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 3,
                left: None,
                right: None,
            })),
            right: None,
        })),
    }));

    println!("{:?}", inorder_traversal(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function inorderTraversal(root) {
  const res = [];
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    res.push(cur.val);
    cur = cur.right;
  }
  return res;
}

const root = new TreeNode(1, null, new TreeNode(2, new TreeNode(3), null));
console.log(inorderTraversal(root));
```
