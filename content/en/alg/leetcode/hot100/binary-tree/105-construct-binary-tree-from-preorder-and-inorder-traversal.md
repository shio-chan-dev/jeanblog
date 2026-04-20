---
title: "Hot100: Construct Binary Tree from Preorder and Inorder Traversal (Indexed Divide-and-Conquer ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "divide and conquer", "hash map", "preorder", "LeetCode 105"]
description: "A practical guide to LeetCode 105 covering how preorder chooses the root, how inorder splits the subtrees, and how a hash map plus index recursion reduces reconstruction to O(n)."
keywords: ["Construct Binary Tree from Preorder and Inorder Traversal", "preorder", "inorder", "divide and conquer", "hash map", "LeetCode 105", "Hot100"]
---

> **Subtitle / Summary**
> The key to LeetCode 105 is not memorizing that preorder and inorder can reconstruct a tree. It is understanding what each traversal contributes: preorder tells you the root, inorder tells you the left/right boundary. Once those roles are clear, the whole problem becomes a clean indexed divide-and-conquer.

- **Reading time**: 12-15 min
- **Tags**: `Hot100`, `binary tree`, `divide and conquer`, `hash map`, `preorder`
- **SEO keywords**: Construct Binary Tree from Preorder and Inorder Traversal, preorder, inorder, divide and conquer, hash map, LeetCode 105
- **Meta description**: Learn LeetCode 105 from traversal roles, indexed recursion, and hash-map root lookup, with runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given the preorder traversal `preorder` and inorder traversal `inorder` of a binary tree, reconstruct the tree and return its root.

The problem guarantees:

- all node values are unique
- both arrays come from the same valid binary tree

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| preorder | `int[]` | preorder traversal of the tree |
| inorder | `int[]` | inorder traversal of the same tree |
| return | TreeNode | reconstructed root |

### Example 1

```text
input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
output: [3,9,20,null,null,15,7]
```

### Example 2

```text
input: preorder = [-1], inorder = [-1]
output: [-1]
```

### Constraints

- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` and `inorder` consist of unique values
- every value of `inorder` appears in `preorder`
- `preorder` is guaranteed to be a valid preorder traversal
- `inorder` is guaranteed to be a valid inorder traversal

---

## Target Readers

- Hot100 learners who still mix up the roles of preorder and inorder
- Developers who want one stable "rebuild the tree from traversals" template
- Readers connecting traversal order with actual tree structure

## Background / Motivation

This problem is valuable because it teaches a deeper question:

- what kind of structural information does each traversal order actually contain?

Many people remember a slogan:

- preorder + inorder can rebuild the tree

But if you do not keep asking why, the code stays mechanical.

The real points to understand are:

- why the first preorder value must be the root
- why the root position in inorder splits left and right subtrees
- why the recursion must build the left subtree before the right subtree

Once those are stable, the implementation is straightforward.

## Core Concepts

- **Preorder traversal**: root, left, right, so the first element is always the current root
- **Inorder traversal**: left, root, right, so the root position splits left and right ranges
- **Interval recursion**: represent the current subtree by an inorder interval `[l, r]`
- **Hash map lookup**: use `value -> inorderIndex` for `O(1)` split-point lookup

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the two key traversal facts

Use:

```text
preorder = [3,9,20,15,7]
inorder  = [9,3,15,20,7]
```

The first preorder element is `3`.
Since preorder is root, left, right:

> `3` must be the root of the whole tree.

Now locate `3` in the inorder array:

```text
[9, 3, 15, 20, 7]
```

Everything left of `3` belongs to the left subtree, and everything right of `3` belongs to the right subtree.

That is the essential split of the entire problem.

#### Step 2: Decide what state the recursion really needs

Cutting arrays into slices works, but copies too much.
The cleaner state is:

- `pre_idx`: which preorder value is the next root
- `l, r`: the current inorder interval

So the subproblem becomes:

> build the subtree whose inorder range is `inorder[l..r]`, using preorder roots in sequence.

#### Step 3: Define the smaller subproblem

Once the current root is known, the remaining work splits naturally:

- build the left subtree from the left inorder segment
- build the right subtree from the right inorder segment

So the recursion shape is:

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 4: Define the base case

If the current inorder interval is empty, there is no subtree:

```python
if l > r:
    return None
```

#### Step 5: Read the current root from preorder

Because preorder always lists the root first:

```python
root_val = preorder[pre_idx]
pre_idx += 1
```

That gives the root of the current subtree immediately.

#### Step 6: Find the split point quickly

Now locate the root inside `inorder`:

```python
mid = index[root_val]
```

To avoid scanning linearly every time, precompute:

```python
index = {value: i for i, value in enumerate(inorder)}
```

Then each lookup is `O(1)`.

#### Step 7: Why must the recursion build left before right?

Because preorder order is:

```text
root -> left -> right
```

After consuming the current root, the next preorder value belongs to the left subtree, not the right subtree.

So the correct order is:

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

If you swap them, `pre_idx` starts consuming right-subtree roots too early.

#### Step 8: Walk the official example slowly

Use:

```text
preorder = [3,9,20,15,7]
inorder  = [9,3,15,20,7]
```

1. `3` is the first preorder value, so `3` is the root
2. In inorder, `3` sits between `[9]` and `[15,20,7]`
3. The next preorder value is `9`, so that must be the left subtree root
4. Only after the left subtree is finished do we move to preorder values for the right subtree

This shows the exact division of labor:

- preorder chooses roots
- inorder chooses boundaries

#### Step 9: Connect this to the general reconstruction pattern

This is not special to preorder + inorder only.
The broader pattern is:

- one traversal identifies the current root
- another traversal tells you how the tree splits around that root

That is the deeper reusable idea.

### Assemble the Full Code

Now combine the traversal roles into the first full working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(preorder, inorder):
    index = {value: i for i, value in enumerate(inorder)}
    pre_idx = 0

    def build(l, r):
        nonlocal pre_idx
        if l > r:
            return None

        root_val = preorder[pre_idx]
        pre_idx += 1
        root = TreeNode(root_val)

        mid = index[root_val]
        root.left = build(l, mid - 1)
        root.right = build(mid + 1, r)
        return root

    return build(0, len(inorder) - 1)


def preorder_traversal(root):
    if root is None:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)


if __name__ == "__main__":
    root = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    print(preorder_traversal(root))
```

### Reference Answer

The LeetCode-style submission version is:

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        index = {value: i for i, value in enumerate(inorder)}
        pre_idx = 0

        def build(l: int, r: int) -> Optional[TreeNode]:
            nonlocal pre_idx
            if l > r:
                return None

            root_val = preorder[pre_idx]
            pre_idx += 1
            root = TreeNode(root_val)

            mid = index[root_val]
            root.left = build(l, mid - 1)
            root.right = build(mid + 1, r)
            return root

        return build(0, len(inorder) - 1)
```

### What method did we just build?

You could call it:

- traversal-based reconstruction
- indexed divide and conquer
- hash-assisted tree rebuilding

But the core takeaway is:

> Preorder chooses the root, and inorder chooses the subtree boundary.

---

## E — Engineering

### Scenario 1: Rebuild a tree from root-first and inorder snapshots (Python)

**Background**: a system stores two traversal snapshots and later needs to reconstruct the exact tree.  
**Why it fits**: one sequence gives the root order, the other gives the subtree boundary.

```python
def rebuild(pre, ino):
    pos = {v: i for i, v in enumerate(ino)}
    idx = 0

    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def dfs(l, r):
        nonlocal idx
        if l > r:
            return None
        val = pre[idx]
        idx += 1
        m = pos[val]
        return Node(val, dfs(l, m - 1), dfs(m + 1, r))

    return dfs(0, len(ino) - 1)


root = rebuild([2, 1, 3], [1, 2, 3])
print(root.val)
```

### Scenario 2: Recover an expression tree from traversal logs (Go)

**Background**: a parser stores traversal sequences for later reconstruction.  
**Why it fits**: preorder gives the operator root, inorder gives the left/right operand split.

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func buildTree(pre, ino []int) *Node {
	pos := map[int]int{}
	for i, v := range ino {
		pos[v] = i
	}
	idx := 0
	var dfs func(int, int) *Node
	dfs = func(l, r int) *Node {
		if l > r {
			return nil
		}
		val := pre[idx]
		idx++
		m := pos[val]
		root := &Node{Val: val}
		root.Left = dfs(l, m-1)
		root.Right = dfs(m+1, r)
		return root
	}
	return dfs(0, len(ino)-1)
}

func main() {
	root := buildTree([]int{2, 1, 3}, []int{1, 2, 3})
	fmt.Println(root.Val)
}
```

### Scenario 3: Restore a UI component hierarchy from traversal data (JavaScript)

**Background**: a low-code system saves traversal outputs and wants to reconstruct the component tree later.  
**Why it fits**: unique node IDs plus root order plus inorder boundaries are enough.

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildTree(preorder, inorder) {
  const pos = new Map();
  inorder.forEach((v, i) => pos.set(v, i));
  let idx = 0;

  function dfs(l, r) {
    if (l > r) return null;
    const val = preorder[idx++];
    const m = pos.get(val);
    return new Node(val, dfs(l, m - 1), dfs(m + 1, r));
  }

  return dfs(0, inorder.length - 1);
}

const root = buildTree([2, 1, 3], [1, 2, 3]);
console.log(root.val);
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)` because each node is created once and each split-point lookup is `O(1)`
- **Space complexity**: `O(n)` from the hash map plus recursion stack

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Hash map + index recursion | `O(n)` | `O(n)` | Best standard solution |
| Linear scan in inorder every time | worst `O(n^2)` | `O(h)` | Can degrade badly |
| Recursive array slicing | `O(n)` to `O(n^2)` | higher | Simpler to read, but copies too much |

### Common Mistakes

1. **Building the right subtree before the left**: this breaks preorder consumption order.  
2. **Skipping the hash map**: then every recursive step may scan `inorder` again.  
3. **Ignoring the uniqueness requirement**: duplicate values would make the split ambiguous.  
4. **Mixing interval boundaries with preorder index semantics**: keep those two responsibilities separate.

## FAQ and Notes

### 1. Why is preorder alone not enough?

Preorder tells you root order, but not where the left subtree stops and the right subtree starts.

### 2. Why is inorder alone not enough?

Inorder tells you relative left/root/right arrangement, but not which node should become the current root first.

### 3. Can postorder plus inorder reconstruct a tree too?

Yes.
The same pattern still holds:

- one traversal identifies the root position
- the other gives the subtree boundary

## Best Practices

- Write one clear sentence for what preorder contributes and what inorder contributes
- Use inorder intervals rather than slicing arrays
- Let `pre_idx` do one thing only: consume roots in preorder order
- Pair this with problem 106 to reinforce the reconstruction pattern

## S — Summary

- The heart of problem 105 is the division of labor between preorder and inorder
- Preorder gives the root; inorder gives the left/right split
- A hash map turns split-point lookup into `O(1)` and keeps the whole algorithm at `O(n)`
- The left subtree must be built before the right subtree because preorder is root-left-right
- This problem is one of the cleanest traversal-to-structure reconstruction exercises

## Further Reading

- [LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- LeetCode 106: Construct Binary Tree from Inorder and Postorder Traversal
- LeetCode 114: Flatten Binary Tree to Linked List
- LeetCode 94: Binary Tree Inorder Traversal

## CTA

Practice `94 + 105 + 106 + 114` together.
They build a very strong intuition for the relationship between traversal order and actual tree structure.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(preorder, inorder):
    pos = {v: i for i, v in enumerate(inorder)}
    idx = 0

    def build(l, r):
        nonlocal idx
        if l > r:
            return None
        val = preorder[idx]
        idx += 1
        root = TreeNode(val)
        m = pos[val]
        root.left = build(l, m - 1)
        root.right = build(m + 1, r)
        return root

    return build(0, len(inorder) - 1)


def preorder_traversal(root):
    if root is None:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)


if __name__ == "__main__":
    root = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    print(preorder_traversal(root))
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

struct TreeNode* build(int* preorder, int* pos, int* idx, int l, int r) {
    if (l > r) return NULL;
    int val = preorder[(*idx)++];
    struct TreeNode* root = new_node(val);
    int m = pos[val + 3000];
    root->left = build(preorder, pos, idx, l, m - 1);
    root->right = build(preorder, pos, idx, m + 1, r);
    return root;
}

void preorder_print(struct TreeNode* root) {
    if (!root) return;
    printf("%d ", root->val);
    preorder_print(root->left);
    preorder_print(root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    int preorder[] = {3, 9, 20, 15, 7};
    int inorder[] = {9, 3, 15, 20, 7};
    int pos[6001] = {0};
    for (int i = 0; i < 5; ++i) pos[inorder[i] + 3000] = i;
    int idx = 0;
    struct TreeNode* root = build(preorder, pos, &idx, 0, 4);
    preorder_print(root);
    printf("\n");
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* build(const std::vector<int>& preorder, int& idx, int l, int r,
                const std::unordered_map<int, int>& pos) {
    if (l > r) return nullptr;
    int val = preorder[idx++];
    TreeNode* root = new TreeNode(val);
    int m = pos.at(val);
    root->left = build(preorder, idx, l, m - 1, pos);
    root->right = build(preorder, idx, m + 1, r, pos);
    return root;
}

void preorderPrint(TreeNode* root) {
    if (!root) return;
    std::cout << root->val << ' ';
    preorderPrint(root->left);
    preorderPrint(root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    std::vector<int> preorder{3, 9, 20, 15, 7};
    std::vector<int> inorder{9, 3, 15, 20, 7};
    std::unordered_map<int, int> pos;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i) pos[inorder[i]] = i;
    int idx = 0;
    TreeNode* root = build(preorder, idx, 0, static_cast<int>(inorder.size()) - 1, pos);
    preorderPrint(root);
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

func buildTree(preorder []int, inorder []int) *TreeNode {
	pos := map[int]int{}
	for i, v := range inorder {
		pos[v] = i
	}
	idx := 0
	var build func(int, int) *TreeNode
	build = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		val := preorder[idx]
		idx++
		root := &TreeNode{Val: val}
		m := pos[val]
		root.Left = build(l, m-1)
		root.Right = build(m+1, r)
		return root
	}
	return build(0, len(inorder)-1)
}

func preorderPrint(root *TreeNode) {
	if root == nil {
		return
	}
	fmt.Print(root.Val, " ")
	preorderPrint(root.Left)
	preorderPrint(root.Right)
}

func main() {
	root := buildTree([]int{3, 9, 20, 15, 7}, []int{9, 3, 15, 20, 7})
	preorderPrint(root)
	fmt.Println()
}
```

```rust
use std::collections::HashMap;

#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn build(
    preorder: &[i32],
    idx: &mut usize,
    l: i32,
    r: i32,
    pos: &HashMap<i32, i32>,
) -> Option<Box<TreeNode>> {
    if l > r {
        return None;
    }
    let val = preorder[*idx];
    *idx += 1;
    let m = *pos.get(&val).unwrap();
    Some(Box::new(TreeNode {
        val,
        left: build(preorder, idx, l, m - 1, pos),
        right: build(preorder, idx, m + 1, r, pos),
    }))
}

fn preorder_collect(root: &Option<Box<TreeNode>>, out: &mut Vec<i32>) {
    if let Some(node) = root {
        out.push(node.val);
        preorder_collect(&node.left, out);
        preorder_collect(&node.right, out);
    }
}

fn main() {
    let preorder = vec![3, 9, 20, 15, 7];
    let inorder = [9, 3, 15, 20, 7];
    let mut pos = HashMap::new();
    for (i, v) in inorder.iter().enumerate() {
        pos.insert(*v, i as i32);
    }
    let mut idx = 0usize;
    let root = build(&preorder, &mut idx, 0, inorder.len() as i32 - 1, &pos);
    let mut out = Vec::new();
    preorder_collect(&root, &mut out);
    println!("{:?}", out);
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildTree(preorder, inorder) {
  const pos = new Map();
  inorder.forEach((v, i) => pos.set(v, i));
  let idx = 0;

  function build(l, r) {
    if (l > r) return null;
    const val = preorder[idx++];
    const m = pos.get(val);
    return new TreeNode(val, build(l, m - 1), build(m + 1, r));
  }

  return build(0, inorder.length - 1);
}

function preorderCollect(root, out = []) {
  if (!root) return out;
  out.push(root.val);
  preorderCollect(root.left, out);
  preorderCollect(root.right, out);
  return out;
}

const root = buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]);
console.log(preorderCollect(root));
```
