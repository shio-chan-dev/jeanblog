---
title: "Hot100：二叉树的层序遍历（Binary Tree Level Order Traversal）BFS / DFS ACERS 解析"
date: 2026-03-15T21:29:44+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "BFS", "DFS", "队列", "层序遍历", "LeetCode 102"]
description: "讲透 LeetCode 102 的按层 BFS、层宽控制与 DFS depth bucket 备选方案，附工程迁移和多语言实现。"
keywords: ["Binary Tree Level Order Traversal", "二叉树的层序遍历", "BFS", "队列", "LeetCode 102", "Hot100"]
---

> **副标题 / 摘要**  
> 层序遍历是二叉树 BFS 模板的起点。真正关键的不是“用队列”，而是“如何把同一层的节点切分出来”。本文按 ACERS 结构拆解 LeetCode 102 的按层处理方法、DFS 深度分桶备选方案，以及工程里常见的分层遍历场景。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`二叉树`、`BFS`、`DFS`、`队列`、`层序遍历`  
- **SEO 关键词**：Hot100, Binary Tree Level Order Traversal, 二叉树的层序遍历, BFS, 队列, LeetCode 102  
- **元描述**：系统讲透 LeetCode 102 的层序 BFS、层宽控制与 DFS 深度分桶思路，并延伸到组织树、菜单树和波次执行等工程场景。  

---

## 目标读者

- 想把 BFS 模板真正固定下来的 Hot100 刷题读者
- 会普通遍历，但一到“按层输出”就容易把层边界写乱的开发者
- 需要按深度分组展示树形结构的工程师

## 背景 / 动机

LeetCode 102 是树题里最标准的 BFS 入门题之一。  
它训练的不是“遍历所有节点”，而是两件更重要的事：

- 如何用队列维护“下一批待处理节点”
- 如何准确切出“这一层”和“下一层”的边界

很多 BFS bug 都来自这里：

- 在遍历当前层时直接用不断变化的 `queue.length`
- 一边弹当前层，一边把新孩子混进当前层结果
- 忘记空树处理，导致访问空指针

把 102 的模板写稳，后面的：

- 右视图
- 每层平均值
- 锯齿层序遍历
- 最小深度 / 最大深度的 BFS 写法

都会自然很多。

## 核心概念

- **层序遍历**：按照树的层级从上到下、从左到右访问节点
- **BFS（广度优先搜索）**：先处理当前层，再扩展下一层
- **层宽快照**：在处理当前层前，先记录队列长度，表示这一层有多少节点
- **depth bucket**：DFS 备选做法，用深度 `depth` 把节点值放进 `res[depth]`

---

## A — Algorithm（题目与算法）

### 题目还原

给你二叉树的根节点 `root`，返回其节点值的层序遍历结果。  
也就是逐层地，从左到右访问所有节点。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点，可以为空 |
| 返回值 | `List[List[int]]` | 每一层的节点值列表 |

### 示例 1

```text
输入: root = [3,9,20,null,null,15,7]
输出: [[3],[9,20],[15,7]]
解释:
第 1 层是 [3]
第 2 层是 [9,20]
第 3 层是 [15,7]
```

### 示例 2

```text
输入: root = [1]
输出: [[1]]
```

### 示例 3

```text
输入: root = []
输出: []
```

### 约束

- 树中节点数目在 `[0, 2000]` 范围内
- `-1000 <= Node.val <= 1000`

---

## C — Concepts（核心思想）

### 思路推导：关键不是队列，而是层边界

如果只要求“访问所有节点”，普通 BFS 很容易。  
但这题要的是 `[[第一层], [第二层], ...]` 这样的二维结果，所以你必须知道：

- 当前从队列里弹出的哪些节点属于同一层
- 新加入队列的孩子节点属于下一层

最稳定的做法就是：

1. 在处理当前层前，记录 `level_size = len(queue)`
2. 接下来只弹出 `level_size` 个节点
3. 这 `level_size` 个节点的值放进同一个 `level` 数组
4. 它们产生的孩子自动留在队列中，等待下一轮处理

### 为什么一定要先记录 `level_size`

因为你在处理当前层时，队列会不断加入下一层的孩子。  
如果直接用变化中的队列长度做循环条件，当前层和下一层就会混在一起。

### 方法归类

- **BFS / 队列**
- **按层分组**
- **DFS 深度分桶（备选）**

### DFS 为什么也能做

如果你用 DFS，每到一个节点就带上当前深度 `depth`：

- 如果 `depth == len(res)`，说明这是新的一层，先创建一个空数组
- 然后把当前值放入 `res[depth]`

这样也能得到按层结果，只是题目直觉上更推荐 BFS。

---

## 实践指南 / 步骤

### 推荐写法：BFS 按层遍历

1. 根节点为空时直接返回空数组
2. 准备队列，把根节点入队
3. 每轮先记录当前层节点数 `level_size`
4. 连续弹出 `level_size` 个节点，收集当前层值
5. 把左右孩子加入队列，进入下一轮

Python 可运行示例：

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

### DFS 备选写法

如果你想练“深度分桶”思维，也可以写 DFS：

1. 递归参数带上 `depth`
2. 首次到达某层时，先创建 `res[depth]`
3. 再把当前节点值加入对应层

这种写法在“顺手还要做别的 DFS 统计”时很方便，但作为 102 的首选模板，BFS 更直观。

---

## E — Engineering（工程应用）

### 场景 1：组织架构按层展示（Python）

**背景**：组织架构、汇报链路常被组织成树。  
**为什么适用**：前端展示时，经常要按层输出 CEO、总监、经理等不同层级。

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

### 场景 2：菜单树逐层渲染（JavaScript）

**背景**：后台菜单和站点导航常是树形配置。  
**为什么适用**：有些页面需要按层懒加载或逐层渲染，避免一次性展开过多节点。

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

### 场景 3：按波次执行树形任务（Go）

**背景**：部署系统或拓扑巡检有时会按“层级波次”推进任务。  
**为什么适用**：层序遍历天然就是一波处理一层，适合逐层展开任务。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func waves(root *Node) [][]string {
	if root == nil {
		return nil
	}
	q := []*Node{root}
	ans := [][]string{}
	for len(q) > 0 {
		size := len(q)
		level := []string{}
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			level = append(level, node.Name)
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
	root := &Node{"root", &Node{"A", nil, nil}, &Node{"B", nil, nil}}
	fmt.Println(waves(root))
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点入队出队各一次
- **空间复杂度**：
  - BFS：`O(w)`，`w` 为树的最大层宽
  - DFS 分桶：`O(h)` 调用栈，再加上结果数组本身

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| BFS 按层遍历 | `O(n)` | `O(w)` | 最自然，推荐 |
| DFS 深度分桶 | `O(n)` | `O(h)` | 能做，但“按层”直觉不如 BFS |
| 递归后再按深度重组 | `O(n)` | 额外映射/数组 | 可行，但不如直接分层 |

### 常见错误与注意事项

- 没有先记录 `level_size`，导致新入队节点被误算进当前层
- 根节点为空时没有提前返回空数组
- 把“当前层结果”定义在循环外，结果所有层共用同一个数组
- JS 中直接遍历变化中的 `queue.length`，导致层边界错乱

## 常见问题与注意事项

### 1. 为什么每层开始前一定要记录队列长度？

因为队列会在处理过程中加入下一层节点。  
只有先记住当前层原始大小，才能准确切出这一层。

### 2. 这题一定要用 BFS 吗？

不一定。DFS 带深度也能做，但 102 最推荐的模板仍然是 BFS。

### 3. 空树应该返回什么？

返回空数组 `[]`，不是 `[[]]`。

## 最佳实践与建议

- 所有“按层输出”的树题，优先联想 `level_size` 模板
- 队列里放节点，层数组里放值，职责分清更不容易写乱
- 想练 DFS 时，记住 `depth == len(res)` 就开新层
- 102、107、199、637 这几题适合放成一组练 BFS 变形

## S — Summary（总结）

- 102 的核心不是“会用队列”，而是“会切层边界”
- 先记录 `level_size` 是整题最重要的稳定技巧
- BFS 是这题的首选模板，DFS 深度分桶是很好的备选思路
- 任何需要按深度分组展示的树形数据，都能复用这套方法
- 把 102 写稳后，层序系列题会明显更顺

## 参考与延伸阅读

- [LeetCode 102: Binary Tree Level Order Traversal](https://leetcode.cn/problems/binary-tree-level-order-traversal/)
- LeetCode 104：二叉树的最大深度
- LeetCode 199：二叉树的右视图
- LeetCode 637：二叉树的层平均值
- LeetCode 103：二叉树的锯齿形层序遍历

## CTA

建议把 102、107、199 连起来做。  
它们都是同一套“按层 BFS”模板的变形题，只是每层输出规则不同，特别适合拿来固化队列分层思维。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

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
    let mut q: VecDeque<Rc<RefCell<TreeNode>>> = VecDeque::new();
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
            let n = node.borrow();
            level.push(n.val);
            if let Some(left) = &n.left {
                q.push_back(left.clone());
            }
            if let Some(right) = &n.right {
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
        if let Some(r) = &right {
            r.borrow_mut().left = Some(TreeNode::new(15));
            r.borrow_mut().right = Some(TreeNode::new(7));
        }
        node.borrow_mut().right = right;
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

const root = new TreeNode(
  3,
  new TreeNode(9),
  new TreeNode(20, new TreeNode(15), new TreeNode(7))
);
console.log(levelOrder(root));
```
