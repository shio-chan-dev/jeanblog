---
title: "Hot100: Binary Tree Level Order Traversal (BFS / DFS ACERS Guide)"
date: 2026-03-16T13:00:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "binary tree", "BFS", "DFS", "queue", "level order traversal", "LeetCode 102"]
description: "A practical guide to LeetCode 102 covering level-by-level BFS, level-size boundaries, DFS depth buckets, and runnable multi-language implementations."
keywords: ["Binary Tree Level Order Traversal", "level order traversal", "BFS", "queue", "LeetCode 102", "Hot100"]
---

> **Subtitle / Summary**  
> Level order traversal is the entry point of the binary-tree BFS template. The real key is not merely "use a queue", but "separate one level from the next correctly". This ACERS guide explains the level-size pattern, the DFS depth-bucket alternative, and engineering situations where grouped-by-depth traversal is useful.

- **Reading time**: 10-12 min  
- **Tags**: `Hot100`, `binary tree`, `BFS`, `DFS`, `queue`, `level order traversal`  
- **SEO keywords**: Hot100, Binary Tree Level Order Traversal, BFS, queue, level order traversal, LeetCode 102  
- **Meta description**: A systematic guide to LeetCode 102 from level-by-level BFS to DFS depth buckets, with engineering scenarios and runnable multi-language implementations.  

---

## Target Readers

- Hot100 learners who want to make the BFS tree template stable
- Developers who can traverse a tree but still mix up current-level and next-level boundaries
- Engineers who need to group tree-shaped data by depth for display or execution

## Background / Motivation

LeetCode 102 is one of the most standard tree-BFS starter problems.

What it really trains is not just "visit all nodes", but two more important skills:

- use a queue to maintain the next batch of nodes to process
- separate the current level from the next level cleanly

Many BFS bugs come exactly from this boundary issue:

- using the changing `queue.length` directly while iterating the current level
- mixing newly pushed children into the current level's answer
- forgetting the empty-tree check and touching null immediately

If you stabilize the 102 template, later problems like:

- right side view
- average of levels
- zigzag level order traversal
- minimum depth or maximum depth via BFS

become much easier.

## Core Concepts

- **Level order traversal**: visit nodes level by level from top to bottom and left to right
- **BFS (breadth-first search)**: process the current layer first, then expand the next layer
- **Level-size snapshot**: record the queue length before processing a level; that number is exactly how many nodes belong to this level
- **Depth bucket**: the DFS alternative, where values are stored in `res[depth]`

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Given the root node `root` of a binary tree, return the level order traversal of its node values.

In other words, return the values level by level from left to right.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree, may be null |
| return | `List[List[int]]` | node values grouped by level |

### Example 1

```text
input: root = [3,9,20,null,null,15,7]
output: [[3],[9,20],[15,7]]
explanation:
level 1 -> [3]
level 2 -> [9,20]
level 3 -> [15,7]
```

### Example 2

```text
input: root = [1]
output: [[1]]
```

### Example 3

```text
input: root = []
output: []
```

### Constraints

- The number of nodes is in the range `[0, 2000]`
- `-1000 <= Node.val <= 1000`

---

## C - Concepts (Core Ideas)

### Thought Process: The key is not the queue itself, but the level boundary

If the task were only "visit every node", ordinary BFS would be enough.  
But this problem asks for a grouped result like `[[level1], [level2], ...]`, so you must know:

- which nodes popped from the queue belong to the current level
- which children should be saved for the next round

The most stable pattern is:

1. before processing a level, record `level_size = len(queue)`
2. pop exactly `level_size` nodes
3. place the values of those nodes into the same `level` array
4. push their children into the queue for the next round

### Why we must record `level_size` first

Because while you process the current level, you keep pushing children from the next level into the same queue.

If you use the changing queue length directly as the loop condition, the current level and the next level will get mixed together.

### Method Category

- **BFS with queue**
- **Level grouping**
- **DFS with depth buckets (alternative)**

### Why DFS can also work

If you use DFS, carry the current depth `depth` in the recursive call:

- if `depth == len(res)`, create a new bucket for that level
- append the current value to `res[depth]`

That also produces a grouped-by-level result, although BFS is the more direct first choice for this problem.

---

## Practice Guide / Steps

### Recommended Approach: Level-by-level BFS

1. Return `[]` immediately if the root is null
2. Initialize a queue with the root
3. At the start of each round, record `level_size`
4. Pop exactly `level_size` nodes and collect their values
5. Push their children into the queue
6. Append the finished `level` array to the answer

Runnable Python example:

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root):
    if root is None:
        return []

    ans = []
    q = deque([root])
    while q:
        level_size = len(q)
        level = []
        for _ in range(level_size):
            node = q.popleft()
            level.append(node.val)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
        ans.append(level)
    return ans


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(level_order(root))
```

### DFS Alternative

If you want to practice the depth-bucket idea:

1. carry `depth` in recursion
2. when `depth == len(res)`, create a new level array
3. append the current node value to `res[depth]`

This is handy when the same traversal also needs other DFS-style statistics, but for LeetCode 102, BFS is more intuitive.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: Group an org chart by management level (Python)

**Background**: reporting structures and org charts are naturally tree-shaped.  
**Why it fits**: UI displays often need data grouped by CEO, VP, director, and manager layers.

```python
from collections import deque


def group_by_level(root):
    if root is None:
        return []
    q = deque([root])
    ans = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node["name"])
            for child in node.get("children", []):
                q.append(child)
        ans.append(level)
    return ans


org = {"name": "CEO", "children": [{"name": "VP1"}, {"name": "VP2"}]}
print(group_by_level(org))
```

### Scenario 2: Render a menu tree level by level (JavaScript)

**Background**: admin menus and site navigation trees are often stored as hierarchical configs.  
**Why it fits**: some interfaces progressively render or lazy-load one depth at a time to reduce initial complexity.

```javascript
function levelOrder(root) {
  if (!root) return [];
  const queue = [root];
  const ans = [];
  while (queue.length) {
    const size = queue.length;
    const level = [];
    for (let i = 0; i < size; i += 1) {
      const node = queue.shift();
      level.push(node.name);
      for (const child of node.children || []) queue.push(child);
    }
    ans.push(level);
  }
  return ans;
}

const menu = { name: "root", children: [{ name: "docs", children: [] }, { name: "blog", children: [] }] };
console.log(levelOrder(menu));
```

### Scenario 3: Execute tree-shaped tasks wave by wave (Go)

**Background**: some workflow engines treat dependent tasks as a tree.  
**Why it fits**: tasks at the same depth can be executed or inspected as one wave before moving to the next depth.

```go
package main

import "fmt"

type Task struct {
	Name     string
	Children []*Task
}

func waves(root *Task) [][]string {
	if root == nil {
		return [][]string{}
	}
	q := []*Task{root}
	ans := [][]string{}
	for len(q) > 0 {
		size := len(q)
		level := make([]string, 0, size)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			level = append(level, node.Name)
			q = append(q, node.Children...)
		}
		ans = append(ans, level)
	}
	return ans
}

func main() {
	root := &Task{
		Name: "build",
		Children: []*Task{
			{Name: "unit-test"},
			{Name: "lint"},
		},
	}
	fmt.Println(waves(root))
}
```

---

## R - Reflection (Analysis and Deeper Understanding)

### Complexity Analysis

- **Time complexity**: `O(n)`, because each node is pushed and popped once
- **Space complexity**:
  - BFS: `O(w)`, where `w` is the maximum width of the tree
  - DFS depth buckets: `O(h)` recursion stack, plus the result array itself

### Alternative Approaches

| Method | Time | Extra Space | Notes |
| --- | --- | --- | --- |
| Level-by-level BFS | `O(n)` | `O(w)` | Most natural and the recommended template |
| DFS depth buckets | `O(n)` | `O(h)` | Works well, but the level-order intuition is less direct |
| Traverse first, regroup later | `O(n)` | Extra map or array | Possible, but more indirect than grouping during traversal |

### Common Mistakes and Pitfalls

- Not recording `level_size` first, so newly added children get mixed into the current level
- Forgetting to return an empty array when the root is null
- Defining the `level` array outside the outer loop and accidentally reusing the same array for all levels
- In JavaScript, iterating with a changing `queue.length` and breaking the level boundary

## Common Questions and Notes

### 1. Why must we record the queue length before each level starts?

Because the queue changes during processing.  
Only the original queue length at the start of the round tells you how many nodes belong to the current level.

### 2. Does this problem have to be solved with BFS?

No. DFS with depth tracking also works. But LeetCode 102 is the canonical BFS level-order template, so BFS is the best first answer.

### 3. What should an empty tree return?

Return `[]`, not `[[]]`.

## Best Practices and Suggestions

- For every "output by level" tree problem, think of the `level_size` template first
- Keep responsibilities clear: the queue stores nodes, while the `level` array stores values
- When practicing DFS, remember the trigger: `depth == len(res)` means create a new level
- Problems 102, 107, 199, and 637 make a good mini-series for BFS level-order variations

## S - Summary

- The heart of LeetCode 102 is not the queue alone, but the level boundary
- Recording `level_size` first is the most important stabilizing technique in the whole problem
- BFS is the primary template here, while DFS depth buckets are a strong alternative
- Any tree-shaped data that needs grouping by depth can reuse the same idea
- Once 102 is stable, the whole family of level-order problems becomes much easier

## References and Further Reading

- [LeetCode 102: Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- LeetCode 104: Maximum Depth of Binary Tree
- LeetCode 199: Binary Tree Right Side View
- LeetCode 637: Average of Levels in Binary Tree
- LeetCode 103: Binary Tree Zigzag Level Order Traversal

## CTA

Practice 102, 107, and 199 together.  
They are all variations of the same **level-by-level BFS** template. Only the output rule changes, which makes them ideal for locking in the queue-and-boundary pattern.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root):
    if root is None:
        return []

    ans = []
    q = deque([root])
    while q:
        level_size = len(q)
        level = []
        for _ in range(level_size):
            node = q.popleft()
            level.append(node.val)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
        ans.append(level)
    return ans


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
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

struct LevelOrderResult {
    int** levels;
    int* sizes;
    int count;
};

struct TreeNode* new_node(int val) {
    struct TreeNode* node = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    node->val = val;
    node->left = NULL;
    node->right = NULL;
    return node;
}

struct LevelOrderResult levelOrder(struct TreeNode* root) {
    struct LevelOrderResult res = {NULL, NULL, 0};
    if (root == NULL) return res;

    struct TreeNode* queue[4096];
    int front = 0;
    int back = 0;
    queue[back++] = root;

    res.levels = (int**)malloc(sizeof(int*) * 2048);
    res.sizes = (int*)malloc(sizeof(int) * 2048);

    while (front < back) {
        int levelSize = back - front;
        res.levels[res.count] = (int*)malloc(sizeof(int) * levelSize);
        res.sizes[res.count] = levelSize;
        for (int i = 0; i < levelSize; ++i) {
            struct TreeNode* node = queue[front++];
            res.levels[res.count][i] = node->val;
            if (node->left) queue[back++] = node->left;
            if (node->right) queue[back++] = node->right;
        }
        res.count++;
    }
    return res;
}

void print_result(struct LevelOrderResult* res) {
    printf("[");
    for (int i = 0; i < res->count; ++i) {
        printf("[");
        for (int j = 0; j < res->sizes[i]; ++j) {
            printf("%d%s", res->levels[i][j], j + 1 == res->sizes[i] ? "" : ",");
        }
        printf("]%s", i + 1 == res->count ? "" : ",");
    }
    printf("]\n");
}

void free_result(struct LevelOrderResult* res) {
    if (!res->levels || !res->sizes) return;
    for (int i = 0; i < res->count; ++i) {
        free(res->levels[i]);
    }
    free(res->levels);
    free(res->sizes);
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

    struct LevelOrderResult res = levelOrder(root);
    print_result(&res);
    free_result(&res);
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

std::vector<std::vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    std::vector<std::vector<int>> ans;
    std::queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        std::vector<int> level;
        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        ans.push_back(level);
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
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);

    auto ans = levelOrder(root);
    std::cout << "[";
    for (size_t i = 0; i < ans.size(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < ans[i].size(); ++j) {
            std::cout << ans[i][j] << (j + 1 == ans[i].size() ? "" : ",");
        }
        std::cout << "]" << (i + 1 == ans.size() ? "" : ",");
    }
    std::cout << "]\n";

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

func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	q := []*TreeNode{root}
	ans := [][]int{}
	for len(q) > 0 {
		size := len(q)
		level := make([]int, 0, size)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
		ans = append(ans, level)
	}
	return ans
}

func main() {
	root := &TreeNode{
		Val:   3,
		Left:  &TreeNode{Val: 9},
		Right: &TreeNode{Val: 20, Left: &TreeNode{Val: 15}, Right: &TreeNode{Val: 7}},
	}
	fmt.Println(levelOrder(root))
}
```

```rust
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

#[derive(Debug, Clone)]
struct TreeNode {
    val: i32,
    left: Node,
    right: Node,
}

impl TreeNode {
    fn new(val: i32) -> Rc<RefCell<TreeNode>> {
        Rc::new(RefCell::new(TreeNode {
            val,
            left: None,
            right: None,
        }))
    }
}

fn level_order(root: &Node) -> Vec<Vec<i32>> {
    let mut ans = Vec::new();
    let mut q = VecDeque::new();
    if let Some(node) = root {
        q.push_back(node.clone());
    } else {
        return ans;
    }

    while !q.is_empty() {
        let size = q.len();
        let mut level = Vec::with_capacity(size);
        for _ in 0..size {
            let node = q.pop_front().unwrap();
            let node_ref = node.borrow();
            level.push(node_ref.val);
            if let Some(left) = &node_ref.left {
                q.push_back(left.clone());
            }
            if let Some(right) = &node_ref.right {
                q.push_back(right.clone());
            }
        }
        ans.push(level);
    }
    ans
}

fn main() {
    let root = Some(TreeNode::new(3));
    if let Some(node) = &root {
        node.borrow_mut().left = Some(TreeNode::new(9));
        let right = Some(TreeNode::new(20));
        node.borrow_mut().right = right.clone();
        if let Some(r) = right {
            r.borrow_mut().left = Some(TreeNode::new(15));
            r.borrow_mut().right = Some(TreeNode::new(7));
        }
    }
    println!("{:?}", level_order(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function levelOrder(root) {
  if (root === null) return [];
  const queue = [root];
  const ans = [];
  while (queue.length) {
    const size = queue.length;
    const level = [];
    for (let i = 0; i < size; i += 1) {
      const node = queue.shift();
      level.push(node.val);
      if (node.left !== null) queue.push(node.left);
      if (node.right !== null) queue.push(node.right);
    }
    ans.push(level);
  }
  return ans;
}

const root = new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
console.log(levelOrder(root));
```
