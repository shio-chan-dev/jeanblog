---
title: "Path Sum III: Prefix Sum + Hash Map Counting Downward Paths (LeetCode 437) ACERS Guide"
date: 2026-02-04T16:02:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary tree", "DFS", "prefix sum", "hash map", "LeetCode 437"]
description: "Count downward paths in a binary tree whose sum equals targetSum in O(n) using prefix sums and a frequency hash map, with derivation, engineering mapping, and multi-language implementations."
keywords: ["Path Sum III", "tree prefix sum", "prefix sum hash map", "DFS", "LeetCode 437", "O(n)"]
---

> **Subtitle / Summary**  
> The constraint “the path can start anywhere, but must go downward” makes root-to-leaf DP insufficient. This ACERS guide explains **prefix sums on trees**: convert any downward path into a difference of two prefix sums, maintain a frequency hash map during one DFS, and finish in O(n).

- **Reading time**: 12–15 min  
- **Tags**: `binary tree`, `prefix sum`, `DFS`, `hash map`  
- **SEO keywords**: Path Sum III, tree prefix sum, prefix-sum hash, LeetCode 437  
- **Meta description**: Count downward paths whose sum equals targetSum in O(n) via prefix sum + hash map, with derivation, tradeoffs, and multi-language implementations.  

---

## Target Readers

- LeetCode learners who want a reusable “tree + hash map” template  
- People who tend to write O(n²) when the path does not have to start at the root  
- Engineers working with hierarchical data (call traces, org trees) who need “downward segment” statistics

## Background / Motivation

Many “tree path” problems hide a trap:
you naturally assume paths start at the root, or end at leaves — but this problem allows the path to start and end at **any nodes**, as long as the direction is downward (parent → child).

That means:

- Maintaining only a “root-to-current” DP state is not enough  
- Enumerating all start nodes degrades to O(n²) in a skewed tree  
- Sliding window does not apply (node values can be negative, so there is no monotonicity)

The key skill worth internalizing is:

> Turn “any downward path” into “the difference of two prefix sums on the same DFS path”.

Once you own this model, a lot of tree counting problems collapse into the familiar recipe: **prefix sum + frequency map**.

## Core Concepts

- **Downward path**: can only go from parent to child (no backtracking, no cross-branch jumps)  
- **Prefix sum**: the sum along the path from the root to the current node  
- **Difference counting**: if `curSum - prevSum = target`, then `prevSum = curSum - target`  
- **Path-local hash map**: the map must represent prefix sums on the *current DFS stack*; you must undo it on backtracking

---

## A — Algorithm (Problem & Algorithm)

### Problem Restatement

Given the root `root` of a binary tree and an integer `targetSum`, return the number of **downward paths** whose node values sum to `targetSum`.
The path does not need to start at the root or end at a leaf, but it must go downward (parent → child).

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| root | TreeNode | root of the binary tree |
| targetSum | int | target path sum |
| return | int | number of valid downward paths |

### Example 1

```text
       10
      /  \
     5   -3
    / \    \
   3   2    11
  / \   \
 3  -2   1

targetSum = 8
output: 3
explain: 5->3, 5->2->1, -3->11
```

### Example 2

```text
    1
   / \
  2   3

targetSum = 3
output: 2
explain: 1->2, 3
```

---

## C — Concepts (Core Ideas)

### Derivation: from O(n²) enumeration to O(n) prefix sum

1) **Naive approach: start DFS from every node**  
For each node `start`, count downward paths starting at `start` with sum `targetSum`.
This can degrade to O(n²) on a chain-like tree and repeats work.

2) **Key observation: any downward path is a contiguous segment of a root-to-current path**  
During a DFS, we are always standing on one root → current path (the recursion stack).
If we define:

- `curSum`: prefix sum from root to current node  
- `prevSum`: prefix sum from root to some ancestor node  

Then the sum of the downward path “(ancestor’s child) → current” equals:

```text
curSum - prevSum
```

To make it equal `targetSum`, we need:

```text
prevSum = curSum - targetSum
```

3) **Method choice: frequency map of prefix sums on the current DFS path**  
When we visit a node:

- compute `curSum`  
- add `cnt[curSum - targetSum]` to the answer (all paths ending at this node)  
- increment `cnt[curSum]` and recurse to children  
- decrement `cnt[curSum]` on backtracking (do not leak into sibling branches)

### Method Category

- **Prefix sum on tree**  
- **DFS + frequency hash map**  
- **Backtracking to maintain path-local state**

### Key Invariant (the thing that makes it correct)

When processing node `x`, the map `cnt` contains prefix sum counts for the path from the root to **x’s parent** only.
That’s why `cnt[curSum - targetSum]` exactly means “how many ancestors produce a valid downward segment ending at x”.

Initialization `cnt[0] = 1` matters:
we treat the “empty prefix” as occurring once, so when `curSum == targetSum` (a path starting at the root), it is counted.

---

## Practice Guide / Steps

1. Run DFS with arguments: `node`, current prefix sum `curSum`  
2. On entry: update `curSum += node.val`  
3. Add `cnt[curSum - targetSum]` to the answer  
4. Record `cnt[curSum] += 1`  
5. Recurse to left/right and accumulate counts  
6. Backtrack: `cnt[curSum] -= 1`  
7. Return the accumulated answer

Runnable Python example (save as `path_sum_iii.py`):

```python
from typing import Dict, Optional


class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


def path_sum(root: Optional[TreeNode], target_sum: int) -> int:
    cnt: Dict[int, int] = {0: 1}

    def dfs(node: Optional[TreeNode], cur: int) -> int:
        if node is None:
            return 0

        cur += node.val
        ans = cnt.get(cur - target_sum, 0)

        cnt[cur] = cnt.get(cur, 0) + 1
        ans += dfs(node.left, cur)
        ans += dfs(node.right, cur)
        cnt[cur] -= 1

        return ans

    return dfs(root, 0)


if __name__ == "__main__":
    # Example 2
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    print(path_sum(root, 3))  # 2
```

---

## E — Engineering (Real-world Scenarios)

> The transferable value of this problem is: **counting “downward contiguous segments” in hierarchical data**.  
> If your data can be modeled as a parent→child tree and each node has an additive value, you can apply the same template.

### Scenario 1: Trace tree — count “downward segments with exact total cost” (Go)

**Background**: a request trace forms a tree of spans; each span has a cost (latency) or a score.  
**Why it fits**: you may want to count how many downward sub-chains sum to a threshold (feature construction, pattern detection).

```go
package main

import "fmt"

type Span struct {
    Cost int64
    Next []*Span
}

func countPaths(root *Span, target int64) int64 {
    cnt := map[int64]int64{0: 1}
    var dfs func(*Span, int64) int64
    dfs = func(node *Span, cur int64) int64 {
        if node == nil {
            return 0
        }
        cur += node.Cost
        ans := cnt[cur-target]
        cnt[cur]++
        for _, ch := range node.Next {
            ans += dfs(ch, cur)
        }
        cnt[cur]--
        return ans
    }
    return dfs(root, 0)
}

func main() {
    root := &Span{Cost: 1, Next: []*Span{{Cost: 2}, {Cost: 3}}}
    fmt.Println(countPaths(root, 3)) // 2: 1->2, 3
}
```

### Scenario 2: Org tree / directory tree — count “budget segments” (Python)

**Background**: an org tree where each node carries a budget delta or cost.  
**Why it fits**: count downward segments whose total equals a target (compliance rules, feature engineering).

```python
from collections import defaultdict


class Node:
    def __init__(self, v, children=None):
        self.v = v
        self.children = children or []


def count_paths(root, target):
    cnt = defaultdict(int)
    cnt[0] = 1

    def dfs(node, cur):
        if node is None:
            return 0
        cur += node.v
        ans = cnt[cur - target]
        cnt[cur] += 1
        for ch in node.children:
            ans += dfs(ch, cur)
        cnt[cur] -= 1
        return ans

    return dfs(root, 0)


if __name__ == "__main__":
    root = Node(1, [Node(2), Node(3)])
    print(count_paths(root, 3))
```

### Scenario 3: Frontend component tree — count “downward weight segments” (JavaScript)

**Background**: component/menu trees where each node has a weight (exposure score, risk score, cost score).  
**Why it fits**: count how many downward segments match a target sum for debugging or rule matching.

```javascript
function Node(v, children = []) {
  this.v = v;
  this.children = children;
}

function countPaths(root, target) {
  const cnt = new Map();
  cnt.set(0, 1);

  function dfs(node, cur) {
    if (!node) return 0;
    cur += node.v;
    const need = cur - target;
    let ans = cnt.get(need) || 0;
    cnt.set(cur, (cnt.get(cur) || 0) + 1);
    for (const ch of node.children) ans += dfs(ch, cur);
    cnt.set(cur, cnt.get(cur) - 1);
    return ans;
  }

  return dfs(root, 0);
}

const root = new Node(1, [new Node(2), new Node(3)]);
console.log(countPaths(root, 3));
```

---

## R — Reflection (Tradeoffs & Deeper Notes)

### Complexity

- **Time**: O(n)  
  Each node is visited once, and hash operations are O(1) amortized.
- **Space**: O(h) ~ O(n)  
  The map stores prefix sums along the current root-to-node path; in the worst case, a skewed tree has height `h = n`.

### Alternatives Comparison

| Method | Idea | Complexity | Issue |
| --- | --- | --- | --- |
| DFS from every node | enumerate start nodes | worst O(n²) | TLE on skewed trees |
| root-to-leaf DP only | classic Path Sum DP | O(n) | violates “start anywhere / end anywhere” |
| **prefix sum + hash map** | difference counting | **O(n)** | must backtrack correctly |

### Common Pitfalls (high-frequency mistakes)

1. **Forgetting `cnt[curSum] -= 1` on backtracking**: you mix prefix sums from sibling branches and count invalid cross-branch paths.  
2. **Trying sliding window**: node values can be negative, so the window has no monotonic property.  
3. **Counting only paths starting at the root**: you will miss paths starting in the middle.  
4. **Overflow in prefix sums**: in production, prefer `int64/long long` for prefix sums.

---

## Explanation / Why it works

This is essentially the “tree version of LeetCode 560 (Subarray Sum Equals K)”:

- In arrays: subarray sum = difference of two prefix sums  
- In trees: downward path sum = difference of two prefix sums on the same DFS stack path

The only extra requirement is handling branching:
the hash map must represent **only the current path**, so we must do “enter +1, exit -1” to keep the counting scope correct.

---

## FAQs and Notes

1. **Why `cnt[0] = 1`?**  
   It counts paths that start at the root: if `curSum == targetSum`, then `curSum - targetSum == 0`, which matches the “empty prefix”.

2. **Can I write it as iterative DFS?**  
   Yes, but you must model backtracking explicitly (push enter/exit events). Recursion is simpler; if stack depth is a concern, use an explicit stack.

3. **Does the path need to end at a leaf?**  
   No. We count paths ending at any node, so we add `cnt[cur-target]` at every node.

---

## Best Practices

- Use `int64/long long` for prefix sums to avoid overflow  
- Keep the meaning of `cnt` crystal clear: it is **path-local** (current DFS stack only)  
- Keep a fixed update order: **count first → `cnt[cur]++` → recurse → `cnt[cur]--`**  
- Hand-simulate 2–3 steps on a tiny tree to verify backtracking restores the state

---

## S — Summary

### Key Takeaways

- For “start anywhere, end anywhere, but downward only” tree path counting, think “prefix sum on tree” first  
- Any downward path sum can be written as a difference of two prefix sums on the same DFS path  
- A frequency map of prefix sums lets you count all paths ending at the current node online in O(1) amortized  
- Backtracking the frequency map is the correctness linchpin (prevents cross-branch pollution)

### Conclusion

The key in LeetCode 437 is not “DFS itself”, but modeling it as **difference counting on tree prefix sums**.
Once you see it this way, the solution becomes short, fast, and reusable.

### References and Further Reading

- LeetCode 437. Path Sum III  
- LeetCode 560. Subarray Sum Equals K (same idea on arrays)  
- LeetCode 112/113. Path Sum / Path Sum II (different constraints; good for comparison)  
- A standard DFS backtracking pattern: mutate state on entry, undo on exit

---

## Meta

- **Reading time**: 12–15 min  
- **Tags**: binary tree, prefix sum, DFS, LeetCode 437  
- **SEO keywords**: Path Sum III, tree prefix sum, prefix-sum hash, LeetCode 437  
- **Meta description**: Prefix sum + hash map to count downward paths with sum targetSum, with derivation and multi-language implementations.  

---

## Call to Action

If you want to solidify this template, do these two next:

1) LeetCode 560 (array prefix-sum difference counting)  
2) LeetCode 112/113 (Path Sum variants to see how constraints change the solution)

If you want an ACERS-style write-up of 560 as a “prefix sum template”, tell me.

---

## Multi-language Reference Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import Optional, Dict


class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


def pathSum(root: Optional[TreeNode], targetSum: int) -> int:
    cnt: Dict[int, int] = {0: 1}

    def dfs(node: Optional[TreeNode], cur: int) -> int:
        if node is None:
            return 0
        cur += node.val
        ans = cnt.get(cur - targetSum, 0)
        cnt[cur] = cnt.get(cur, 0) + 1
        ans += dfs(node.left, cur)
        ans += dfs(node.right, cur)
        cnt[cur] -= 1
        return ans

    return dfs(root, 0)


if __name__ == "__main__":
    # Example 1
    root = TreeNode(
        10,
        TreeNode(
            5,
            TreeNode(3, TreeNode(3), TreeNode(-2)),
            TreeNode(2, None, TreeNode(1)),
        ),
        TreeNode(-3, None, TreeNode(11)),
    )
    print(pathSum(root, 8))  # 3
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef long long i64;

typedef struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

static int count_nodes(const TreeNode* root) {
    if (!root) return 0;
    return 1 + count_nodes(root->left) + count_nodes(root->right);
}

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

typedef struct {
    i64* keys;
    int* vals;
    unsigned char* used;
    size_t cap;
} Map;

static Map map_new(size_t cap) {
    Map m;
    m.cap = cap;
    m.keys = (i64*)calloc(cap, sizeof(i64));
    m.vals = (int*)calloc(cap, sizeof(int));
    m.used = (unsigned char*)calloc(cap, sizeof(unsigned char));
    return m;
}

static void map_free(Map* m) {
    free(m->keys);
    free(m->vals);
    free(m->used);
}

static int map_get(const Map* m, i64 key) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)mix64((uint64_t)key) & mask;
    while (m->used[i]) {
        if (m->keys[i] == key) return m->vals[i];
        i = (i + 1) & mask;
    }
    return 0;
}

static void map_add(Map* m, i64 key, int delta) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)mix64((uint64_t)key) & mask;
    while (m->used[i]) {
        if (m->keys[i] == key) {
            m->vals[i] += delta;
            return;
        }
        i = (i + 1) & mask;
    }
    m->used[i] = 1;
    m->keys[i] = key;
    m->vals[i] = delta;
}

static int dfs(TreeNode* node, i64 cur, i64 target, Map* cnt) {
    if (!node) return 0;
    cur += (i64)node->val;
    int ans = map_get(cnt, cur - target);
    map_add(cnt, cur, 1);
    ans += dfs(node->left, cur, target, cnt);
    ans += dfs(node->right, cur, target, cnt);
    map_add(cnt, cur, -1);
    return ans;
}

static int pathSum(TreeNode* root, int targetSum) {
    int n = count_nodes(root);
    size_t cap = 1;
    while (cap < (size_t)(n * 4 + 8)) cap <<= 1; /* keep load factor low */
    Map cnt = map_new(cap);
    map_add(&cnt, 0, 1);
    int ans = dfs(root, 0, (i64)targetSum, &cnt);
    map_free(&cnt);
    return ans;
}

static TreeNode* node(int v, TreeNode* l, TreeNode* r) {
    TreeNode* n = (TreeNode*)malloc(sizeof(TreeNode));
    n->val = v;
    n->left = l;
    n->right = r;
    return n;
}

static void free_tree(TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    /* Example 1 */
    TreeNode* root =
        node(10,
             node(5,
                  node(3, node(3, NULL, NULL), node(-2, NULL, NULL)),
                  node(2, NULL, node(1, NULL, NULL))),
             node(-3, NULL, node(11, NULL, NULL)));
    printf("%d\n", pathSum(root, 8)); /* 3 */
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

static int dfs(TreeNode* node, long long cur, long long target, std::unordered_map<long long, int>& cnt) {
    if (!node) return 0;
    cur += node->val;
    int ans = 0;
    auto it = cnt.find(cur - target);
    if (it != cnt.end()) ans += it->second;
    cnt[cur] += 1;
    ans += dfs(node->left, cur, target, cnt);
    ans += dfs(node->right, cur, target, cnt);
    cnt[cur] -= 1;
    return ans;
}

int pathSum(TreeNode* root, int targetSum) {
    std::unordered_map<long long, int> cnt;
    cnt[0] = 1;
    return dfs(root, 0, targetSum, cnt);
}

int main() {
    // Example 1
    auto* root = new TreeNode(10);
    root->left = new TreeNode(5);
    root->right = new TreeNode(-3);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(2);
    root->right->right = new TreeNode(11);
    root->left->left->left = new TreeNode(3);
    root->left->left->right = new TreeNode(-2);
    root->left->right->right = new TreeNode(1);

    std::cout << pathSum(root, 8) << "\n"; // 3

    // Omit delete in this demo. In production, release memory properly.
    return 0;
}
```

```go
package main

import "fmt"

type TreeNode struct {
    Val   int64
    Left  *TreeNode
    Right *TreeNode
}

func pathSum(root *TreeNode, targetSum int64) int64 {
    cnt := map[int64]int64{0: 1}
    var dfs func(*TreeNode, int64) int64
    dfs = func(node *TreeNode, cur int64) int64 {
        if node == nil {
            return 0
        }
        cur += node.Val
        ans := cnt[cur-targetSum]
        cnt[cur]++
        ans += dfs(node.Left, cur)
        ans += dfs(node.Right, cur)
        cnt[cur]--
        return ans
    }
    return dfs(root, 0)
}

func main() {
    // Example 1
    root := &TreeNode{Val: 10}
    root.Left = &TreeNode{Val: 5}
    root.Right = &TreeNode{Val: -3}
    root.Left.Left = &TreeNode{Val: 3}
    root.Left.Right = &TreeNode{Val: 2}
    root.Right.Right = &TreeNode{Val: 11}
    root.Left.Left.Left = &TreeNode{Val: 3}
    root.Left.Left.Right = &TreeNode{Val: -2}
    root.Left.Right.Right = &TreeNode{Val: 1}

    fmt.Println(pathSum(root, 8)) // 3
}
```

```rust
use std::collections::HashMap;

#[derive(Debug)]
struct TreeNode {
    val: i64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new(val: i64) -> Self {
        TreeNode { val, left: None, right: None }
    }
}

fn dfs(node: &Option<Box<TreeNode>>, cur: i64, target: i64, cnt: &mut HashMap<i64, i32>) -> i32 {
    let Some(n) = node.as_ref() else { return 0 };
    let cur = cur + n.val;
    let mut ans = *cnt.get(&(cur - target)).unwrap_or(&0);
    *cnt.entry(cur).or_insert(0) += 1;
    ans += dfs(&n.left, cur, target, cnt);
    ans += dfs(&n.right, cur, target, cnt);
    if let Some(v) = cnt.get_mut(&cur) {
        *v -= 1;
    }
    ans
}

fn path_sum(root: &Option<Box<TreeNode>>, target: i64) -> i32 {
    let mut cnt: HashMap<i64, i32> = HashMap::new();
    cnt.insert(0, 1);
    dfs(root, 0, target, &mut cnt)
}

fn main() {
    // Example 1
    let mut root = Box::new(TreeNode::new(10));
    root.left = Some(Box::new(TreeNode::new(5)));
    root.right = Some(Box::new(TreeNode::new(-3)));

    {
        let left = root.left.as_mut().unwrap();
        left.left = Some(Box::new(TreeNode::new(3)));
        left.right = Some(Box::new(TreeNode::new(2)));
        let ll = left.left.as_mut().unwrap();
        ll.left = Some(Box::new(TreeNode::new(3)));
        ll.right = Some(Box::new(TreeNode::new(-2)));
        let lr = left.right.as_mut().unwrap();
        lr.right = Some(Box::new(TreeNode::new(1)));
    }
    {
        let right = root.right.as_mut().unwrap();
        right.right = Some(Box::new(TreeNode::new(11)));
    }

    let root = Some(root);
    println!("{}", path_sum(&root, 8)); // 3
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function pathSum(root, targetSum) {
  const cnt = new Map();
  cnt.set(0, 1);

  function dfs(node, cur) {
    if (!node) return 0;
    cur += node.val;
    let ans = cnt.get(cur - targetSum) || 0;
    cnt.set(cur, (cnt.get(cur) || 0) + 1);
    ans += dfs(node.left, cur);
    ans += dfs(node.right, cur);
    cnt.set(cur, cnt.get(cur) - 1);
    return ans;
  }

  return dfs(root, 0);
}

// Example 1
const root = new TreeNode(
  10,
  new TreeNode(5, new TreeNode(3, new TreeNode(3), new TreeNode(-2)), new TreeNode(2, null, new TreeNode(1))),
  new TreeNode(-3, null, new TreeNode(11)),
);
console.log(pathSum(root, 8)); // 3
```

