---
title: "Hot100: Kth Smallest Element in a BST (Inorder Counting / Early Stop ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "BST", "inorder traversal", "stack", "LeetCode 230"]
description: "A practical guide to LeetCode 230 covering why inorder traversal of a BST is sorted, how to count visits with an explicit stack, and how to stop immediately at the k-th node."
keywords: ["Kth Smallest Element in a BST", "BST", "inorder traversal", "stack", "LeetCode 230", "Hot100"]
---

> **Subtitle / Summary**
> LeetCode 230 is not really about tree traversal mechanics alone. It is about turning BST order into a useful query. Once you see that the k-th smallest value is simply the k-th node visited in inorder traversal, the problem becomes a very stable counting task.

- **Reading time**: 11-14 min
- **Tags**: `Hot100`, `binary tree`, `BST`, `inorder traversal`, `stack`
- **SEO keywords**: Kth Smallest Element in a BST, BST, inorder traversal, stack, LeetCode 230
- **Meta description**: Learn LeetCode 230 from BST inorder ordering, explicit-stack counting, and early-stop traversal, with runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given the root `root` of a binary search tree and an integer `k`, return the `k`-th smallest value in the tree.

Here `k` is 1-indexed.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the BST |
| k | int | ranking position, `1 <= k <= n` |
| return | int | the `k`-th smallest node value |

### Example 1

```text
input: root = [3,1,4,null,2], k = 1
output: 1
```

### Example 2

```text
input: root = [5,3,6,2,4,null,null,1], k = 3
output: 3
```

### Constraints

- The number of nodes in the tree is `n`
- `1 <= k <= n <= 10^4`
- `0 <= Node.val <= 10^4`

### Follow-up

If the BST is modified often and you need to query the k-th smallest frequently, how would you optimize it?

---

## Target Readers

- Hot100 learners who know that BST inorder traversal is sorted but have not yet turned that fact into query logic
- Developers who want to connect problems 94, 98, and 230 into one BST skill chain
- Engineers who want a clean "rank query on an ordered tree" template

## Background / Motivation

This problem is good training for one key transition:

- from "tree structure thinking" to "ordered sequence thinking"

A naive solution might be:

- do an inorder traversal
- collect all values into an array
- return the element at position `k - 1`

That works, but it does more work than necessary.
A BST already gives you sorted order for free.

So the real question is:

> How do we count nodes in inorder order and stop immediately once we reach the k-th one?

## Core Concepts

- **BST inorder ordering**: inorder traversal visits values in sorted ascending order
- **Explicit stack**: simulate recursive inorder traversal with full control
- **Visit counting**: each pop from the stack means one more sorted element has appeared
- **Early stop**: once the k-th node is reached, the rest of the tree is irrelevant

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Rewrite the problem in inorder language

Take the smallest useful example:

```text
    2
   / \
  1   3

k = 2
```

Its inorder traversal is:

```text
[1,2,3]
```

So the 2nd smallest value is simply the 2nd visited node in inorder order.

This means the real problem is:

> Find the node visited at the k-th step of inorder traversal.

#### Step 2: Decide what state the iterative traversal needs

If we want to control inorder traversal manually, we need:

- a `stack` for ancestors we still need to process
- a `cur` pointer to walk left

That is the standard iterative inorder state.

#### Step 3: Define the smaller subproblem

At each point, we are repeatedly solving this smaller task:

> Find the next smallest unvisited node.

The rule is:

- keep pushing the current node and go left
- when left is exhausted, pop the stack

The popped node is the next value in sorted order.

#### Step 4: Define when the work is complete

A node is truly visited when it is popped from the stack:

```python
cur = stack.pop()
count += 1
```

If:

```python
count == k
```

then we already have the answer and can return immediately.

#### Step 5: Push the left chain first

The inorder rule is left, root, right.
So as long as `cur` exists, go left:

```python
while cur is not None:
    stack.append(cur)
    cur = cur.left
```

This guarantees the next popped node is the next smallest value.

#### Step 6: Count only when the node is actually visited

Do not count at push time.
The correct counting point is:

```python
cur = stack.pop()
count += 1
```

because only then has the node been reached in true inorder order.

#### Step 7: Continue with the right subtree

After visiting the current node, inorder traversal must move to the right subtree:

```python
cur = cur.right
```

Then the whole "go left as much as possible" process repeats.

#### Step 8: Walk the official example slowly

Use:

```text
root = [3,1,4,null,2], k = 1
```

1. Start at `3`, push it, go left to `1`
2. Push `1`, go left to null
3. Pop `1`
4. This is visit number 1, so the answer is immediately `1`

We never need to traverse the rest of the tree.
That is the value of early stopping.

#### Step 9: Connect this to the follow-up

For a single query, iterative inorder traversal is excellent.

For many queries with frequent insertions and deletions, the follow-up suggests a stronger structure:

- store subtree sizes
- walk downward by rank instead of traversing from scratch

But for the main problem, inorder counting is the right baseline.

### Assemble the Full Code

Now combine the traversal and counting rules into the first full working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(root, k):
    stack = []
    cur = root
    count = 0

    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left

        cur = stack.pop()
        count += 1
        if count == k:
            return cur.val
        cur = cur.right

    return -1


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))
    print(kth_smallest(root, 1))
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
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root

        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right

        return -1
```

### What method did we just build?

You could call it:

- BST inorder counting
- iterative inorder traversal
- rank query by traversal

But the real takeaway is:

> In a BST, the k-th smallest element is exactly the k-th node visited by inorder traversal.

---

## E — Engineering

### Scenario 1: Query the k-th threshold in an ordered rule tree (Python)

**Background**: a system stores thresholds inside a BST-like structure and needs rank-based access.  
**Why it fits**: inorder traversal yields sorted order without extra sorting work.

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth(root, k):
    stack = []
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right


root = Node(20, Node(10), Node(30))
print(kth(root, 2))
```

### Scenario 2: Rank query inside a read-heavy tree index (Go)

**Background**: a mostly static tree index needs occasional rank-based key lookup.  
**Why it fits**: explicit-stack inorder traversal is simple, stable, and avoids materializing the full order.

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func kth(root *Node, k int) int {
	stack := []*Node{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return cur.Val
		}
		cur = cur.Right
	}
	return -1
}

func main() {
	root := &Node{Val: 2, Left: &Node{Val: 1}, Right: &Node{Val: 3}}
	fmt.Println(kth(root, 3))
}
```

### Scenario 3: Pick the k-th display node in a ranked UI tree (JavaScript)

**Background**: a UI keeps ranked items in a tree-shaped structure and wants one item by order.  
**Why it fits**: inorder traversal gives the ranking stream directly.

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function kth(root, k) {
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    k -= 1;
    if (k === 0) return cur.val;
    cur = cur.right;
  }
  return null;
}

const root = new Node(4, new Node(2, new Node(1), new Node(3)), new Node(6));
console.log(kth(root, 4));
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: worst-case `O(n)`, but often closer to `O(h + k)` because traversal can stop early
- **Space complexity**: `O(h)` for the explicit stack

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Inorder traversal with early stop | worst `O(n)` | `O(h)` | Best baseline for one query |
| Full inorder array | `O(n)` | `O(n)` | Works, but stores more than needed |
| Subtree-size augmentation | `O(h)` query | extra bookkeeping | Best for frequent updates + frequent rank queries |

### Common Mistakes

1. **Treating the BST like a generic binary tree**: then you miss the inorder-sorted property entirely.  
2. **Counting when pushing nodes**: the correct counting point is when the node is popped and truly visited.  
3. **Forgetting early stop**: then you keep traversing nodes that cannot change the answer.  
4. **Off-by-one with `k`**: the problem uses 1-based counting, not 0-based indexing.

## FAQ and Notes

### 1. Why is this so closely related to problem 94?

Because 94 gives you the traversal order.
Problem 230 simply adds counting and early stopping on top of that order.

### 2. What does the follow-up optimization mean?

If updates and rank queries are both frequent, you can store subtree sizes.
Then instead of traversing in full inorder order, you walk down the tree by comparing `k` with the left subtree size.

### 3. Can recursion solve this too?

Yes.
You can keep:

- a global or nonlocal counter
- an answer variable

But the explicit-stack version mirrors the iterative inorder template more clearly.

## Best Practices

- If you see "BST + k-th smallest", think inorder immediately
- Count when the node is popped, not when it is pushed
- Stop as soon as the k-th node is reached
- If the interviewer asks about many updates and many queries, bring up subtree sizes

## S — Summary

- The core of problem 230 is turning BST order into a rank query
- The k-th smallest node is exactly the k-th node in inorder traversal
- Explicit-stack traversal gives precise visit order and clean early stopping
- For one query, inorder counting is usually enough
- Problems 94, 98, and 230 form a very clean BST fundamentals sequence

## Further Reading

- [LeetCode 230: Kth Smallest Element in a BST](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)
- LeetCode 94: Binary Tree Inorder Traversal
- LeetCode 98: Validate Binary Search Tree
- LeetCode 173: Binary Search Tree Iterator

## CTA

Practice `94 + 98 + 230` together.
They form a compact BST chain: traversal, validation, and ordered query.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(root, k):
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right
    return -1


if __name__ == "__main__":
    root = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4)), TreeNode(6))
    print(kth_smallest(root, 3))
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

int kthSmallest(struct TreeNode* root, int k) {
    struct TreeNode* stack[10016];
    int top = 0;
    struct TreeNode* cur = root;

    while (cur != NULL || top > 0) {
        while (cur != NULL) {
            stack[top++] = cur;
            cur = cur->left;
        }
        cur = stack[--top];
        k--;
        if (k == 0) return cur->val;
        cur = cur->right;
    }
    return -1;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(5);
    root->left = new_node(3);
    root->right = new_node(6);
    root->left->left = new_node(2);
    root->left->right = new_node(4);
    root->left->left->left = new_node(1);
    printf("%d\n", kthSmallest(root, 3));
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <stack>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int kthSmallest(TreeNode* root, int k) {
    std::stack<TreeNode*> st;
    TreeNode* cur = root;
    while (cur || !st.empty()) {
        while (cur) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        if (--k == 0) return cur->val;
        cur = cur->right;
    }
    return -1;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(5);
    root->left = new TreeNode(3);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(2);
    root->left->right = new TreeNode(4);
    root->left->left->left = new TreeNode(1);
    std::cout << kthSmallest(root, 3) << '\n';
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

func kthSmallest(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return cur.Val
		}
		cur = cur.Right
	}
	return -1
}

func main() {
	root := &TreeNode{
		Val: 5,
		Left: &TreeNode{
			Val: 3,
			Left: &TreeNode{
				Val:  2,
				Left: &TreeNode{Val: 1},
			},
			Right: &TreeNode{Val: 4},
		},
		Right: &TreeNode{Val: 6},
	}
	fmt.Println(kthSmallest(root, 3))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn kth_smallest(root: &Option<Box<TreeNode>>, mut k: i32) -> i32 {
    let mut stack: Vec<&TreeNode> = Vec::new();
    let mut cur = root.as_deref();

    while cur.is_some() || !stack.is_empty() {
        while let Some(node) = cur {
            stack.push(node);
            cur = node.left.as_deref();
        }
        let node = stack.pop().unwrap();
        k -= 1;
        if k == 0 {
            return node.val;
        }
        cur = node.right.as_deref();
    }
    -1
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 5,
        left: Some(Box::new(TreeNode {
            val: 3,
            left: Some(Box::new(TreeNode {
                val: 2,
                left: Some(Box::new(TreeNode {
                    val: 1,
                    left: None,
                    right: None,
                })),
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 6,
            left: None,
            right: None,
        })),
    }));

    println!("{}", kth_smallest(&root, 3));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function kthSmallest(root, k) {
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    k -= 1;
    if (k === 0) return cur.val;
    cur = cur.right;
  }
  return null;
}

const root = new TreeNode(
  5,
  new TreeNode(3, new TreeNode(2, new TreeNode(1)), new TreeNode(4)),
  new TreeNode(6)
);
console.log(kthSmallest(root, 3));
```
