---
title: "Hot100：翻转二叉树（Invert Binary Tree）递归 / BFS ACERS 解析"
date: 2026-03-06T17:58:23+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "递归", "BFS", "树变换", "LeetCode 226"]
description: "围绕 LeetCode 226 讲清树镜像变换、递归交换左右子树的本质，以及工程里的结构镜像场景。"
keywords: ["Invert Binary Tree", "翻转二叉树", "树镜像", "递归", "BFS", "LeetCode 226", "Hot100"]
---

> **副标题 / 摘要**  
> 翻转二叉树是一道看起来非常短、却能快速检验你是否真正理解递归结构的题。本文围绕 LeetCode 226 拆解“交换左右子树”的本质，给出递归 / BFS 两种做法，以及结构镜像在工程中的迁移思路。

- **预计阅读时长**：8~10 分钟  
- **标签**：`Hot100`、`二叉树`、`递归`、`BFS`、`树变换`  
- **SEO 关键词**：Hot100, Invert Binary Tree, 翻转二叉树, 树镜像, 递归, LeetCode 226  
- **元描述**：讲清 LeetCode 226 的递归与 BFS 解法，并延伸到布局镜像、结构变换等工程场景。  

---

## 目标读者

- 想检验自己是否真正理解“递归作用在整棵树每个节点上”的刷题读者
- 看到树题就下意识写遍历，但不确定该在什么时机处理当前节点的开发者
- 需要做树形结构镜像、布局翻转或对称转换的工程师

## 背景 / 动机

226 的代码通常很短，但它的思维非常典型：

- 当前节点要做什么？  
  把 `left` 和 `right` 交换。

- 子问题是什么？  
  左右子树本身也都要继续翻转。

这就是非常纯粹的“**当前操作 + 递归处理子问题**”。

如果你对这题没有完全吃透，往往会出现：

- 只交换根节点，不继续处理子树
- 交换后递归方向写乱
- 把本来能原地完成的事，额外重建一棵新树

## 核心概念

- **树镜像（mirror）**：把每个节点的左子树与右子树对调
- **原地变换（in-place transform）**：不新建整棵树，只交换指针
- **递归分治**：当前节点处理完后，左右子树仍是同类型问题
- **BFS 层序变换**：也可以按层把每个节点的左右孩子交换

---

## A — Algorithm（题目与算法）

### 题目还原

给你一棵二叉树的根节点 `root`，请将这棵树翻转，并返回翻转后的根节点。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点，可以为空 |
| 返回值 | TreeNode | 翻转后的根节点 |

### 示例 1

```text
输入: root = [4,2,7,1,3,6,9]
输出: [4,7,2,9,6,3,1]
解释:
原树左右子树整体对调后，所有节点都完成镜像翻转。
```

### 示例 2

```text
输入: root = [2,1,3]
输出: [2,3,1]
```

### 示例 3

```text
输入: root = []
输出: []
```

### 约束

- 树中节点数目在 `[0, 100]` 内
- `-100 <= Node.val <= 100`

---

## C — Concepts（核心思想）

### 思路推导：为什么“交换 + 递归”就够了

假设当前节点是 `node`，我们要做的事情只有两步：

1. 交换 `node.left` 和 `node.right`
2. 递归翻转新的左子树和新的右子树

写成伪代码非常短：

```text
invert(node):
    if node 为空:
        return null
    交换 node.left 和 node.right
    invert(node.left)
    invert(node.right)
    return node
```

### 为什么这就是完整答案

因为翻转整棵树，本质上就是“**每个节点都做一次左右交换**”。  
而树的每个局部子树仍然是一棵树，所以递归天然成立。

### 方法归类

- **树递归**
- **原地结构变换**
- **BFS / 队列遍历**

### 递归与 BFS 的取舍

1. **递归**
   - 代码最短
   - 最贴合树定义
   - 推荐作为主解

2. **BFS**
   - 每层逐个交换
   - 如果你想顺手做层级统计或可视化处理，BFS 也很合适

---

## 实践指南 / 步骤

### 推荐写法：递归

1. 判空
2. 交换左右子节点
3. 递归翻转左子树
4. 递归翻转右子树
5. 返回当前节点

Python 可运行示例：

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

## E — Engineering（工程应用）

### 场景 1：左右分栏布局镜像预览（JavaScript）

**背景**：可视化编辑器常把分栏布局组织成二叉树。  
**为什么适用**：做“镜像预览”时，本质就是把每个分割节点的左右区域交换。

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

### 场景 2：教学工具里的树镜像演示（Python）

**背景**：算法教学平台经常需要动态演示“镜像”概念。  
**为什么适用**：226 的解法就是最标准的树镜像变换。

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

### 场景 3：规则树的左右分支翻转测试（Go）

**背景**：有些规则引擎会把“命中 / 未命中”分支组织为二叉树。  
**为什么适用**：做镜像测试时，可以快速验证左右分支逻辑是否对称。

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

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点交换一次
- **空间复杂度**：
  - 递归：`O(h)`
  - BFS：`O(w)`，`w` 为最大层宽

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 递归 | `O(n)` | `O(h)` | 最简洁，推荐 |
| BFS 队列 | `O(n)` | `O(w)` | 便于按层处理 |
| 新建镜像树 | `O(n)` | `O(n)` | 不必要，额外分配更多内存 |

### 常见错误与注意事项

- 只在根节点做一次交换，忘了对子树继续递归
- 交换后又沿着旧引用递归，导致逻辑混乱
- 明明可以原地做，却重新 new 一整棵树
- 把“翻转二叉树”和“反转链表”类比错了，误以为需要线性重连顺序

## 常见问题与注意事项

### 1. 先递归再交换可以吗？

可以，只要保证每个节点最终都完成左右交换即可。但“先交换再递归”通常最直观。

### 2. 递归和 BFS 谁更适合面试？

这题首选递归。BFS 更像备选写法，用来展示你对遍历方式的掌握。

### 3. 这题属于前序还是后序？

更像“前序式处理”：因为当前节点一上来就先做交换，然后再处理子树。

## 最佳实践与建议

- 树变换题先想“当前节点要改什么”，再想“子问题是否同型”
- 能原地交换就原地交换，避免不必要的新对象分配
- 写递归时尽量让函数语义简单明确：传入一棵树，返回翻转后的这棵树
- 别只背代码，最好能口头说清这题为什么天然适合递归

## S — Summary（总结）

- 226 的本质是对每个节点执行一次左右交换
- 递归成立的原因是：每个子树仍然是相同问题
- 这题是非常典型的“当前处理 + 递归处理子问题”模板
- BFS 也能做，但递归表达更直接
- 工程里，布局镜像、可视化镜像、规则树对称测试都能复用这类思路

## 参考与延伸阅读

- [LeetCode 226: Invert Binary Tree](https://leetcode.cn/problems/invert-binary-tree/)
- LeetCode 101：对称二叉树
- LeetCode 100：相同的树
- LeetCode 104：二叉树的最大深度
- LeetCode 102：二叉树的层序遍历

## CTA

可以把 226、101、100 连着做。一个练“结构变换”，一个练“结构比较”，一组下来对树递归的感觉会扎实很多。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

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
