---
title: "Hot100：二叉树的最大深度（Maximum Depth of Binary Tree）DFS / BFS ACERS 解析"
date: 2026-03-06T17:58:22+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "DFS", "BFS", "递归", "LeetCode 104"]
description: "讲透 LeetCode 104 的深度定义、递归 DFS 与层序 BFS 两种主流解法，附工程迁移和多语言实现。"
keywords: ["Maximum Depth of Binary Tree", "二叉树的最大深度", "DFS", "BFS", "递归", "LeetCode 104", "Hot100"]
---

> **副标题 / 摘要**  
> “最大深度”是树递归最标准的起手式。你只要真正理解“当前树的答案依赖左右子树答案”的定义，整类树形 DP / DFS 题都会顺很多。本文以 LeetCode 104 为核心，系统讲解递归 DFS、层序 BFS 与工程迁移方法。

- **预计阅读时长**：9~11 分钟  
- **标签**：`Hot100`、`二叉树`、`DFS`、`BFS`、`递归`  
- **SEO 关键词**：Hot100, Maximum Depth of Binary Tree, 二叉树的最大深度, DFS, BFS, LeetCode 104  
- **元描述**：从深度定义出发，讲清 LeetCode 104 的 DFS 和 BFS 解法，并附多语言可运行代码。  

---

## 目标读者

- 刚开始刷树题，想把“树递归返回值”真正吃透的同学
- 能写遍历，但一遇到“求高度 / 求路径 / 求答案”就容易混乱的开发者
- 需要在菜单树、组织架构、嵌套 JSON 等层级数据里做深度分析的工程师

## 背景 / 动机

LeetCode 104 看起来像一道“送分题”，但它几乎是所有树递归的母题：

- 你需要先回答“**空树深度是多少**”
- 再回答“**当前节点的答案依赖谁**”
- 最后把关系写成 `1 + max(left, right)`

一旦这个递归定义真正建立起来，后续的平衡二叉树、直径、路径和、最近公共祖先都会更容易进入状态。

## 核心概念

- **深度 / 高度**：这里按题意，根到最远叶子节点的节点数
- **后序式思维**：想知道当前节点答案，必须先知道左右子树答案
- **DFS**：递归向下，回溯时组合答案
- **BFS**：按层遍历，最后一层编号就是树深度

---

## A — Algorithm（题目与算法）

### 题目还原

给定二叉树根节点 `root`，返回其 **最大深度**。

最大深度是指：从根节点到最远叶子节点的最长路径上，经过的节点数量。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点，可以为空 |
| 返回值 | int | 树的最大深度 |

### 示例 1

```text
输入: root = [3,9,20,null,null,15,7]
输出: 3
解释:
第 1 层: 3
第 2 层: 9, 20
第 3 层: 15, 7
所以最大深度为 3。
```

### 示例 2

```text
输入: root = [1,null,2]
输出: 2
```

### 约束

- 树中节点数目在 `[0, 10^4]` 内
- `-100 <= Node.val <= 100`

---

## C — Concepts（核心思想）

### 思路推导：为什么递归公式是 `1 + max(left, right)`

对任意节点 `node` 来说：

- 如果它为空，深度就是 `0`
- 如果它不为空，那么从它出发的最大深度，等于：
  - 当前节点这一层贡献 `1`
  - 加上左右子树中更深的那一边

所以状态转移非常直接：

```text
depth(node) = 1 + max(depth(node.left), depth(node.right))
```

### 方法归类

- **树形递归 / DFS**
- **层序遍历 / BFS**
- **树问题中的“自底向上合并答案”**

### DFS 和 BFS 各适合什么场景

1. **DFS 递归**
   - 代码最短
   - 最符合定义
   - 适合大部分面试与题解讲解

2. **BFS 层序**
   - 很适合“按层统计”的问题
   - 如果你同时想拿到每层节点分布，BFS 更顺手

### 为什么 DFS 是这题的推荐模板

因为这题不是要求打印每层节点，而是只要一个最终数值。  
DFS 直接按定义写，表达最清晰，错误率最低。

---

## 实践指南 / 步骤

### 推荐写法：递归 DFS

1. 如果节点为空，返回 `0`
2. 递归计算左子树最大深度
3. 递归计算右子树最大深度
4. 返回 `1 + max(leftDepth, rightDepth)`

Python 可运行示例：

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

### BFS 备选写法

如果你偏爱按层遍历，也可以：

1. 用队列保存当前层节点
2. 每处理完一层，深度加一
3. 队列为空时结束

这种方法也很常见，尤其在题目还要求“返回每层结果”时更顺手。

---

## E — Engineering（工程应用）

### 场景 1：前端菜单配置最大嵌套层级（JavaScript）

**背景**：后台常允许菜单配置为树形结构。  
**为什么适用**：发布前可以检查菜单是否超过设计允许的最大层级。

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

### 场景 2：组织架构的最长汇报链（Go）

**背景**：组织架构或审批链路常用树表示。  
**为什么适用**：最大深度能衡量层级复杂度，辅助流程优化和权限设计。

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

### 场景 3：嵌套 JSON 的最大层数校验（Python）

**背景**：日志、配置和 ETL 数据里常有深层嵌套 JSON。  
**为什么适用**：过深的数据容易影响可读性和下游处理，可在入口处先做深度限制。

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

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点访问一次
- **空间复杂度**：
  - DFS 递归：`O(h)`
  - BFS 队列：最坏 `O(n)`，更准确地说是 `O(w)`，其中 `w` 是树的最大层宽

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| DFS 递归 | `O(n)` | `O(h)` | 最符合定义，推荐 |
| BFS 层序 | `O(n)` | `O(w)` | 按层问题很顺手 |
| 显式栈 DFS | `O(n)` | `O(h)` | 不想用递归时可选 |

### 常见错误与注意事项

- 把空节点深度写成 `1`，导致整体多一层
- 把“边数”和“节点数”概念混掉，这题按 **节点数** 计
- 只在一边递归，忘了取 `max(left, right)`
- BFS 时每弹一个节点就加一，结果把“节点数”错当成“层数”

## 常见问题与注意事项

### 1. 这题是前序、中序还是后序？

更准确地说，它属于“**后序式合并**”思维：因为当前节点答案依赖左右子树答案。

### 2. DFS 和 BFS 哪个更好？

如果只求一个深度值，DFS 更简洁；如果还要顺手得到每层节点，BFS 更自然。

### 3. 递归会不会爆栈？

极端退化树确实可能。工程里如果树深不可控，可改成显式栈或 BFS。

## 最佳实践与建议

- 树题先写清 base case：`node == null` 时返回什么
- 遇到“当前答案依赖左右子树答案”的树题，优先联想递归返回值
- 写复杂度时区分 `O(h)` 和 `O(w)`，表达更准确
- 能说清 DFS / BFS 各自适用场景，比只会背代码更重要

## S — Summary（总结）

- 104 的核心不是代码，而是深度定义本身
- 只要想清 `depth(node) = 1 + max(left, right)`，递归就会自然写出来
- DFS 是这题最推荐的模板，BFS 是非常好的按层替代方案
- 这题是后续平衡树、树直径、路径和等题的基础
- 工程里，任何层级结构的“最大嵌套深度”都能复用这套思路

## 参考与延伸阅读

- [LeetCode 104: Maximum Depth of Binary Tree](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)
- LeetCode 111：二叉树的最小深度
- LeetCode 110：平衡二叉树
- LeetCode 543：二叉树的直径
- LeetCode 102：二叉树的层序遍历

## CTA

建议把 104 和 111 放在一起练。一个是 `max`，一个常常要特别注意空子树处理；两题一起做，树递归的 base case 会更稳。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

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
