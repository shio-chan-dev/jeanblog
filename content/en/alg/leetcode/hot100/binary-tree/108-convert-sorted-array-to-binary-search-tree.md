---
title: "Hot100: Convert Sorted Array to Binary Search Tree (Midpoint Divide-and-Conquer ACERS Guide)"
date: 2026-04-20T10:01:46+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "BST", "divide and conquer", "recursion", "LeetCode 108"]
description: "A practical guide to LeetCode 108 covering why the midpoint is the natural root, how sorted order and balance requirements work together, and how to build the BST with stable index recursion."
keywords: ["Convert Sorted Array to Binary Search Tree", "BST", "balanced BST", "divide and conquer", "recursion", "LeetCode 108", "Hot100"]
---

> **Subtitle / Summary**
> The key to LeetCode 108 is not recursion by itself. It is noticing that the problem wants two things at the same time: BST ordering and height balance. Once you read those two constraints together, "pick the middle element as the root" stops being a trick and becomes the natural construction rule.

- **Reading time**: 11-14 min
- **Tags**: `Hot100`, `binary tree`, `BST`, `divide and conquer`, `recursion`
- **SEO keywords**: Convert Sorted Array to Binary Search Tree, BST, balanced BST, divide and conquer, recursion, LeetCode 108
- **Meta description**: Learn LeetCode 108 from the midpoint-construction idea, with step-by-step derivation, engineering mappings, and runnable multi-language implementations.

---

## A — Algorithm

### Problem Restatement

Given an integer array `nums` sorted in strictly increasing order, convert it into a height-balanced binary search tree.

That means the output tree must satisfy two requirements:

- it must be a valid BST
- it should be as balanced as possible

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | `int[]` | strictly increasing sorted array |
| return | TreeNode | root of any valid height-balanced BST |

### Example 1

```text
input: nums = [-10,-3,0,5,9]
output: [0,-3,9,-10,null,5]
explanation: [0,-10,5,null,-3,null,9] is also correct.
```

### Example 2

```text
input: nums = [1,3]
output: [3,1]
explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
```

### Constraints

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` is sorted in strictly increasing order

---

## Target Readers

- Hot100 learners who want one stable "array to BST" construction template
- Developers who already know BST validation but still feel shaky about BST construction
- Engineers who want to understand why midpoint splitting is a direct consequence of the problem constraints

## Background / Motivation

This problem is good training for one important habit:

- when a problem gives multiple structural requirements at once, find the split rule that satisfies all of them together

A first instinct might be:

- insert the values into a BST one by one
- or pick any root, then recursively attach the rest

But the problem is not just asking for a BST.
It is asking for a **balanced** BST built from an already sorted array.

So the right question is:

> Which element should become the root if we want both ordering and balance?

Once you ask that, the midpoint choice becomes almost inevitable.

## Core Concepts

- **BST ordering**: all values on the left are smaller; all values on the right are larger
- **Height balance**: left and right subtree heights should stay close
- **Interval recursion**: use an array segment `[l, r]` to represent the subtree to build
- **Midpoint root**: splitting near the center keeps the tree shallow

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the smallest non-trivial example

Consider:

```text
nums = [1,2,3]
```

If you choose `1` as the root, the tree becomes:

```text
1
 \
  2
   \
    3
```

That is still a BST, but it is obviously not balanced.

If you choose the middle value `2`:

```text
  2
 / \
1   3
```

Now both requirements are satisfied.

#### Step 2: Decide what state the recursion really needs

We do not need to copy subarrays.
We only need to know:

- where the current segment starts
- where the current segment ends

So the recursive state is just:

- `l`
- `r`

The subproblem becomes:

> build a height-balanced BST from `nums[l..r]`.

#### Step 3: Define the smaller subproblem

Once the root position is chosen, the remaining work splits naturally:

- build the left subtree from the left half
- build the right subtree from the right half

So the recursion shape is:

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 4: Define the base case

If the current interval is empty, there is no subtree to build:

```python
if l > r:
    return None
```

That is the standard tree-construction base case.

#### Step 5: Why is the midpoint the right root candidate?

The tree should be height-balanced.
The most direct way to encourage balance is to keep the left and right subtree sizes as close as possible.

That means the root should split the interval roughly in half:

```python
mid = (l + r) // 2
```

For even-length segments, either left-middle or right-middle is acceptable as long as you are consistent.

#### Step 6: Build the current root

Once `mid` is fixed, create the root node:

```python
root = TreeNode(nums[mid])
```

Then recursively attach both sides:

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 7: Why does this automatically satisfy BST ordering?

Because the input array is strictly increasing.

That means:

- every value left of `mid` is smaller than `nums[mid]`
- every value right of `mid` is larger than `nums[mid]`

So BST ordering is already built into the array positions.

#### Step 8: Why does this also keep the tree balanced?

Because every step splits the current segment as evenly as possible.

So:

- left and right subtree sizes differ by at most 1 at the current split
- repeated midpoint splits keep the tree height low

This is not an extra trick.
It follows directly from the balancing requirement.

#### Step 9: Walk the official example slowly

Use:

```text
nums = [-10,-3,0,5,9]
```

1. The middle value is `0`, so `0` becomes the root
2. The left segment `[-10,-3]` builds the left subtree
3. The right segment `[5,9]` builds the right subtree
4. Each side repeats the same midpoint rule

The entire construction is just one simple rule applied recursively.

### Assemble the Full Code

Now combine those pieces into the first full working solution.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst(nums):
    def build(l, r):
        if l > r:
            return None
        mid = (l + r) // 2
        root = TreeNode(nums[mid])
        root.left = build(l, mid - 1)
        root.right = build(mid + 1, r)
        return root

    return build(0, len(nums) - 1)


def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


if __name__ == "__main__":
    root = sorted_array_to_bst([-10, -3, 0, 5, 9])
    print(inorder(root))
```

### Reference Answer

If you want the LeetCode-style submission version, it becomes:

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(l: int, r: int) -> Optional[TreeNode]:
            if l > r:
                return None
            mid = (l + r) // 2
            root = TreeNode(nums[mid])
            root.left = build(l, mid - 1)
            root.right = build(mid + 1, r)
            return root

        return build(0, len(nums) - 1)
```

### What method did we just build?

You could call it:

- divide and conquer
- interval recursion
- midpoint BST construction

But the real takeaway is this:

> When a problem wants both sorted BST order and good balance, midpoint splitting is the natural construction rule.

---

## E — Engineering

### Scenario 1: Build a balanced in-memory index from sorted keys (Python)

**Background**: an offline job has already sorted the keys, and a service wants to rebuild an in-memory tree quickly at startup.  
**Why it fits**: sequential insertion can degrade badly, while midpoint construction creates a much shallower search tree immediately.

```python
class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


def build_balanced(keys):
    def dfs(l, r):
        if l > r:
            return None
        m = (l + r) // 2
        return Node(keys[m], dfs(l, m - 1), dfs(m + 1, r))

    return dfs(0, len(keys) - 1)


root = build_balanced([10, 20, 30, 40, 50])
print(root.key)
```

### Scenario 2: Build a shallow decision tree from ordered thresholds (C++)

**Background**: some read-heavy systems pre-build a threshold tree so lookup depth stays predictable.  
**Why it fits**: midpoint splitting keeps the worst-case comparison path close to logarithmic depth.

```cpp
#include <iostream>
#include <vector>

struct Node {
    int val;
    Node* left;
    Node* right;
    explicit Node(int x) : val(x), left(nullptr), right(nullptr) {}
};

Node* build(const std::vector<int>& a, int l, int r) {
    if (l > r) return nullptr;
    int m = (l + r) / 2;
    Node* root = new Node(a[m]);
    root->left = build(a, l, m - 1);
    root->right = build(a, m + 1, r);
    return root;
}

int main() {
    std::vector<int> a{5, 10, 20, 40, 80};
    Node* root = build(a, 0, static_cast<int>(a.size()) - 1);
    std::cout << root->val << '\n';
}
```

### Scenario 3: Convert sorted breakpoints into a balanced rule tree (JavaScript)

**Background**: a UI rule engine stores ordered breakpoints and wants a tree-shaped runtime matcher.  
**Why it fits**: a balanced tree reduces the maximum number of comparisons during rule selection.

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildBalanced(arr, l = 0, r = arr.length - 1) {
  if (l > r) return null;
  const m = Math.floor((l + r) / 2);
  return new Node(arr[m], buildBalanced(arr, l, m - 1), buildBalanced(arr, m + 1, r));
}

const root = buildBalanced([320, 480, 768, 1024, 1280]);
console.log(root.val);
```

---

## R — Reflection

### Complexity Analysis

- **Time complexity**: `O(n)` because each array element becomes exactly one tree node
- **Space complexity**: `O(log n)` to `O(n)` depending on recursion depth; with midpoint splitting it is typically `O(log n)`

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Midpoint divide-and-conquer | `O(n)` | `O(log n)` | Best match for the problem |
| Insert values one by one into BST | best `O(n log n)`, worst `O(n^2)` | variable | Can easily degenerate |
| Slice arrays at each recursive step | `O(n)` to `O(n log n)` | higher | Simpler to read, but copies too much |

### Common Mistakes

1. **Thinking only about BST order and ignoring balance**: then sequential insertion starts looking attractive, but it misses the point of the problem.  
2. **Slicing arrays inside recursion**: it works, but adds unnecessary copying.  
3. **Mixing interval boundaries**: `mid - 1` and `mid + 1` are easy to swap by mistake.  
4. **Assuming the answer is unique**: the problem allows multiple valid balanced BSTs.

## FAQ and Notes

### 1. Why are both left-middle and right-middle valid for even lengths?

Because the subtree sizes still differ by at most 1.
The problem does not require a unique tree shape.

### 2. Why do we not need explicit BST validation here?

Because BST ordering is already guaranteed by the sorted array:

- everything left of `mid` is smaller
- everything right of `mid` is larger

### 3. Is this basically binary search?

The split looks similar, but the goal is different:

- binary search finds one target
- this problem uses binary-style splitting to construct a low-height tree

## Best Practices

- Define clearly what the current interval means before writing recursion
- Pass indices instead of copying subarrays
- When a problem says "sorted" plus "balanced", test midpoint splitting first
- Pair this problem with 98 and 230 to complete the BST construction / validation / query trio

## S — Summary

- LeetCode 108 is really about reading "sorted" and "balanced" together
- Choosing the midpoint as the root satisfies BST ordering and balance at the same time
- The recursion only needs an interval `[l, r]`; subarray copies are unnecessary
- This is a clean divide-and-conquer construction template worth memorizing
- Problem 108 pairs naturally with 98 and 230 in a BST learning path

## Further Reading

- [LeetCode 108: Convert Sorted Array to Binary Search Tree](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)
- LeetCode 98: Validate Binary Search Tree
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 230: Kth Smallest Element in a BST

## CTA

Try practicing `108 + 98 + 230` together.
They form a very clean BST sequence: construction, validation, and ordered query.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst(nums):
    def build(l, r):
        if l > r:
            return None
        m = (l + r) // 2
        root = TreeNode(nums[m])
        root.left = build(l, m - 1)
        root.right = build(m + 1, r)
        return root

    return build(0, len(nums) - 1)


def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


if __name__ == "__main__":
    root = sorted_array_to_bst([-10, -3, 0, 5, 9])
    print(inorder(root))
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

struct TreeNode* build(int* nums, int l, int r) {
    if (l > r) return NULL;
    int m = (l + r) / 2;
    struct TreeNode* root = new_node(nums[m]);
    root->left = build(nums, l, m - 1);
    root->right = build(nums, m + 1, r);
    return root;
}

void inorder(struct TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->val);
    inorder(root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    int nums[] = {-10, -3, 0, 5, 9};
    struct TreeNode* root = build(nums, 0, 4);
    inorder(root);
    printf("\n");
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* build(const std::vector<int>& nums, int l, int r) {
    if (l > r) return nullptr;
    int m = (l + r) / 2;
    TreeNode* root = new TreeNode(nums[m]);
    root->left = build(nums, l, m - 1);
    root->right = build(nums, m + 1, r);
    return root;
}

void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    std::cout << root->val << ' ';
    inorder(root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    std::vector<int> nums{-10, -3, 0, 5, 9};
    TreeNode* root = build(nums, 0, static_cast<int>(nums.size()) - 1);
    inorder(root);
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

func sortedArrayToBST(nums []int) *TreeNode {
	var build func(int, int) *TreeNode
	build = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		m := (l + r) / 2
		root := &TreeNode{Val: nums[m]}
		root.Left = build(l, m-1)
		root.Right = build(m+1, r)
		return root
	}
	return build(0, len(nums)-1)
}

func inorder(root *TreeNode) {
	if root == nil {
		return
	}
	inorder(root.Left)
	fmt.Print(root.Val, " ")
	inorder(root.Right)
}

func main() {
	root := sortedArrayToBST([]int{-10, -3, 0, 5, 9})
	inorder(root)
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

fn build(nums: &[i32], l: i32, r: i32) -> Option<Box<TreeNode>> {
    if l > r {
        return None;
    }
    let m = (l + r) / 2;
    Some(Box::new(TreeNode {
        val: nums[m as usize],
        left: build(nums, l, m - 1),
        right: build(nums, m + 1, r),
    }))
}

fn inorder(root: &Option<Box<TreeNode>>, out: &mut Vec<i32>) {
    if let Some(node) = root {
        inorder(&node.left, out);
        out.push(node.val);
        inorder(&node.right, out);
    }
}

fn main() {
    let nums = vec![-10, -3, 0, 5, 9];
    let root = build(&nums, 0, nums.len() as i32 - 1);
    let mut out = Vec::new();
    inorder(&root, &mut out);
    println!("{:?}", out);
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function sortedArrayToBST(nums) {
  function build(l, r) {
    if (l > r) return null;
    const m = Math.floor((l + r) / 2);
    return new TreeNode(nums[m], build(l, m - 1), build(m + 1, r));
  }
  return build(0, nums.length - 1);
}

function inorder(root, out = []) {
  if (!root) return out;
  inorder(root.left, out);
  out.push(root.val);
  inorder(root.right, out);
  return out;
}

const root = sortedArrayToBST([-10, -3, 0, 5, 9]);
console.log(inorder(root));
```
