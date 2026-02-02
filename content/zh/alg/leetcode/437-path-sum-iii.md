---
title: "路径和 III：前缀和 + 哈希表统计向下路径（LeetCode 437）ACERS 解析"
date: 2026-02-02T22:13:45+08:00
draft: false
categories: ["LeetCode"]
tags: ["二叉树", "DFS", "前缀和", "哈希表", "LeetCode 437"]
description: "用前缀和 + 计数哈希表在 O(n) 时间统计二叉树中和为 targetSum 的向下路径数，含推导、工程迁移与多语言实现。"
keywords: ["Path Sum III", "路径和 III", "前缀和", "哈希表", "DFS", "LeetCode 437", "O(n)"]
---

> **副标题 / 摘要**  
> “路径不必从根开始、但必须向下”使得这题无法用简单的根到叶 DP 解决。本文用 ACERS 结构讲透 **树上前缀和**：把任意向下路径转化为“两个前缀和的差”，用哈希表在线计数，做到 O(n) 一次 DFS 统计所有答案。

- **预计阅读时长**：12~15 分钟  
- **标签**：`二叉树`、`前缀和`、`DFS`、`哈希表`  
- **SEO 关键词**：Path Sum III, 路径和 III, 树上前缀和, 前缀和哈希, LeetCode 437  
- **元描述**：前缀和 + 哈希表在线统计二叉树向下路径和等于 targetSum 的条数，包含推导、复杂度对比与多语言实现。  

---

## 目标读者

- 刷 LeetCode、希望把“树 + 哈希”题型沉淀成模板的学习者  
- 对“路径不从根开始”的树题容易写成 O(n^2) 的同学  
- 做日志调用链 / 层级数据分析，需要在树结构上做区间统计的工程师

## 背景 / 动机

很多“树上的路径问题”都有一个坑：  
你以为要从根出发、或要到叶子结束，但题目允许 **从任意节点开始、到任意节点结束**（但方向必须向下）。  
这意味着：

- 你不能只维护“从根到当前”的一种状态就完事；  
- 也不能枚举所有起点（那会退化成 O(n^2)）；  
- 更不能用滑动窗口（节点值可正可负，窗口单调性不存在）。

这题最值得掌握的点是：**把“树上任意向下路径”化为“同一路径上的两个前缀和之差”**。  
一旦你掌握了这个模型，很多树上统计题都会变成“前缀和 + 哈希表”的熟悉配方。

## 核心概念

- **向下路径**：只能从父到子（不能回头、不能跨分支）  
- **前缀和（prefix sum）**：从根到当前节点路径上所有节点值的累加  
- **差分计数**：若 `curSum - prevSum = target`，则 `prevSum = curSum - target`  
- **路径内哈希表**：只统计“当前 DFS 路径上的前缀和”，回溯时必须撤销（否则会把不同分支混在一起）

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个二叉树的根节点 `root` 和整数 `targetSum`，求二叉树里 **节点值之和等于 targetSum 的向下路径** 的数目。  
路径不需要从根节点开始，也不需要在叶子节点结束，但路径方向必须向下（只能从父节点到子节点）。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点 |
| targetSum | int | 目标路径和 |
| 返回 | int | 满足条件的向下路径条数 |

### 示例 1（常见示例）

```text
       10
      /  \
     5   -3
    / \    \
   3   2    11
  / \   \
 3  -2   1

targetSum = 8
输出: 3
解释: 5->3, 5->2->1, -3->11
```

### 示例 2（自拟）

```text
    1
   / \
  2   3

targetSum = 3
输出: 2
解释: 1->2, 3
```

---

## C — Concepts（核心思想）

### 思路推导：从 O(n^2) 枚举到 O(n) 前缀和

1. **朴素做法：以每个节点为起点做一次 DFS**  
   对每个节点 `start`，统计所有从 `start` 向下的路径和是否等于 target。  
   - 在链状树（极度不平衡）里会退化成 O(n^2)  
   - 代码也更容易写重复逻辑

2. **关键观察：任何向下路径都是“同一根到叶路径”的一段连续片段**  
   在一次 DFS 中，我们始终走在某条根到当前节点的路径上。  
   如果我们记录：

   - `curSum`：根到当前节点的前缀和  
   - `prevSum`：根到某个祖先节点的前缀和  

   那么祖先的下一个节点到当前节点的路径和就是：

```text
curSum - prevSum
```

   要让它等于 `targetSum`，就要求：

```text
prevSum = curSum - targetSum
```

3. **方法选择：路径内前缀和计数表（HashMap）**  
   当我们在 DFS 到达某个节点时：

   - 计算当前 `curSum`  
   - 需要的答案增量是：`count[curSum - targetSum]`  
   - 然后把 `curSum` 计数 +1，递归左右子树  
   - 回溯时把 `curSum` 计数 -1（只统计当前路径，不能污染兄弟分支）

### 方法归类

- **树上前缀和（Prefix Sum on Tree）**  
- **DFS + 哈希表计数（DFS with frequency map）**  
- **回溯（Backtracking）维护路径状态**

### 关键不变量（写对的核心）

在访问节点 `x` 时，哈希表 `cnt` 只包含“从根到 `x` 的父节点”这条路径上的前缀和计数。  
这样 `cnt[curSum - targetSum]` 才表示“从某个祖先之后开始，到 x 结束”的合法向下路径数量。

初始化 `cnt[0] = 1` 的意义：  
把“空前缀”也当成一次出现，这样当 `curSum == targetSum` 时（路径从根开始），也能被计数到。

---

## 实践指南 / 步骤

1. 定义 DFS：入参为 `node`、当前前缀和 `curSum`  
2. 访问节点时更新 `curSum += node.val`  
3. `ans += cnt[curSum - targetSum]`（统计以当前节点为终点的路径条数）  
4. `cnt[curSum] += 1`（把当前前缀和加入路径）  
5. 递归左右子树，把返回值累加  
6. 回溯：`cnt[curSum] -= 1`（离开该节点时撤销影响）  
7. 返回累计答案

Python 可运行示例（保存为 `path_sum_iii.py`）：

```python
from typing import Optional, Dict


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
    # 示例 2（自拟）
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    print(path_sum(root, 3))  # 2
```

---

## E — Engineering（工程应用）

> 这道题的工程迁移价值在于：**在层级结构里统计“任意起点到任意终点的向下连续片段”数量**。  
> 只要你的数据能抽象为“父 -> 子”的树状关系，并且每个节点有一个可累加的数值，就可以套这个模板。

### 场景 1：调用链（trace tree）里统计“连续片段耗时等于阈值”的次数（Go）

**背景**：一次请求的 trace 形成树形 span 结构，每个 span 有耗时（或打分）。  
**为什么适用**：你可能希望统计“某段连续向下调用链”累计耗时恰好为某个阈值的次数（例如合成特征、检测固定模式）。

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

### 场景 2：组织结构/目录树里统计“从任意部门到下级的预算片段”数量（Python）

**背景**：部门树每个节点带一个预算增量或成本。  
**为什么适用**：你可能需要统计“任意管理链上连续片段的预算和等于 target”的次数，用于合规或特征工程。

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

### 场景 3：前端组件树“向下连续权重片段”统计（JavaScript）

**背景**：组件树/菜单树每个节点带一个权重（曝光分、风险分、成本分）。  
**为什么适用**：你可能需要统计满足特定累计分值的连续向下片段数量，用于 debug 或规则匹配。

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

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
  每个节点在 DFS 中只被访问一次，且每次哈希操作均摊 O(1)。
- **空间复杂度**：O(h) ~ O(n)  
  哈希表保存的是“当前根到叶路径上的前缀和计数”，最坏情况下链状树高度 h = n。

### 替代方案对比

| 方法 | 思路 | 复杂度 | 问题 |
| --- | --- | --- | --- |
| 每点做一次向下 DFS | 枚举起点 | 最坏 O(n^2) | 链状树会超时 |
| 只算根到叶 | 典型路径和 DP | O(n) | 不满足“起点任意/终点任意” |
| **前缀和 + 哈希（本文）** | 差分计数 | **O(n)** | 必须正确回溯撤销计数 |

### 常见误区（高频翻车点）

1. **忘记回溯 `cnt[curSum] -= 1`**：会把兄弟分支的前缀和“串起来”，得到虚假的跨分支路径。  
2. **用滑动窗口**：节点值可以为负，窗口不具备单调性，思路不成立。  
3. **只统计从根开始的路径**：会漏掉从任意节点开始的合法路径。  
4. **前缀和用 int 溢出**：工程里建议用 `int64/long long` 存前缀和。

---

## 解释与原理（为什么这么做）

本质上，我们在做“树上的 560 题”（数组的 Subarray Sum Equals K）：  
数组里“子数组和 = 两个前缀和之差”，树里“向下路径和 = 同一 DFS 路径上两个前缀和之差”。

唯一的区别在于：树会分叉，所以哈希表必须只代表**当前路径**。  
因此要用“进入节点 +1、离开节点 -1”的回溯操作来维持正确的计数域。

---

## 常见问题与注意事项

1. **为什么 `cnt[0] = 1`？**  
   让“从根开始到某个节点”的路径也能被计数：当 `curSum == targetSum` 时，`curSum - targetSum == 0`，对应这一次“空前缀”。

2. **可以改成迭代 DFS 吗？**  
   可以，但实现要额外处理“回溯时机”。递归更直观；若担心栈深，可改为显式栈并区分入栈/出栈事件。

3. **路径必须以叶子结束吗？**  
   不需要。我们统计的是“以任意节点为终点”的向下路径，因此在每个节点都做一次 `cnt[cur-target]` 计数。

---

## 最佳实践与建议

- 用 `int64/long long` 存前缀和，避免溢出  
- 把 `cnt` 的含义写在脑中：它只属于“当前 DFS 路径”  
- 写代码时固定顺序：**先计数 ans，再 cnt[cur]++，递归，最后 cnt[cur]--**  
- 先用小树手算 2~3 轮，验证“回溯撤销”确实把状态还原

---

## S — Summary（总结）

### 核心收获

- “起点任意、终点任意但必须向下”的路径题，优先想到“树上前缀和”  
- 任意向下路径和可以转化为同一路径上的两个前缀和之差  
- 用哈希表记录当前路径前缀和出现次数，可在线统计以当前节点为终点的答案增量  
- 回溯撤销计数是正确性的关键（避免跨分支污染）  

### 小结 / 结论

LeetCode 437 的关键不是 DFS，而是把它看成“树上的前缀和差分计数”。  
掌握这题，你相当于把数组前缀和的经典模型升级到了树结构上。

### 参考与延伸阅读

- LeetCode 437. Path Sum III  
- LeetCode 560. Subarray Sum Equals K（同一思想在数组上的版本）  
- LeetCode 112/113. Path Sum（路径起点/终点限制不同，便于对比）  
- 树上 DFS 回溯的典型范式：进入修改状态、离开撤销状态

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：二叉树、前缀和、DFS、LeetCode 437  
- **SEO 关键词**：Path Sum III, 树上前缀和, 前缀和哈希, LeetCode 437  
- **元描述**：前缀和 + 哈希表在线统计二叉树向下路径和等于 targetSum 的条数，附推导与多语言实现。  

---

## 行动号召（CTA）

建议你用同一个模板，立刻去做两题巩固迁移：

1) LeetCode 560（数组前缀和差分计数）  
2) LeetCode 142（链表判环入口定位：同样依赖“不变量 + 结构推理”）

如果你希望我把 560 也按 ACERS 模板整理成一篇“可复用前缀和模型”文章，告诉我即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

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
    # 示例 1
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
    /* 示例 1 */
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
    // 示例 1
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

    // 省略 delete（示例代码），工程里请释放
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
    // 示例 1
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
    // 示例 1
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

// 示例 1
const root = new TreeNode(
  10,
  new TreeNode(5, new TreeNode(3, new TreeNode(3), new TreeNode(-2)), new TreeNode(2, null, new TreeNode(1))),
  new TreeNode(-3, null, new TreeNode(11)),
);
console.log(pathSum(root, 8)); // 3
```
