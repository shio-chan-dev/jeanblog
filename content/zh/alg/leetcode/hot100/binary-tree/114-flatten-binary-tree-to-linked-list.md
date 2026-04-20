---
title: "Hot100：二叉树展开为链表（Flatten Binary Tree to Linked List）反向先序重连 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "先序遍历", "原地修改", "递归", "LeetCode 114"]
description: "围绕 LeetCode 114 讲清为什么展开后的链表顺序就是先序遍历，以及如何用反向先序和 prev 指针把整棵树原地重连。"
keywords: ["Flatten Binary Tree to Linked List", "二叉树展开为链表", "先序遍历", "原地修改", "反向先序", "LeetCode 114", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 114 的真正难点不是把树“拍平”，而是想清楚重连顺序。只要抓住“目标链表等于先序遍历顺序”，再把处理方向反过来，`prev` 指针会让整题变成一段非常稳定的原地递归。

- **预计阅读时长**：12~15 分钟
- **标签**：`Hot100`、`二叉树`、`先序遍历`、`原地修改`、`递归`
- **SEO 关键词**：Flatten Binary Tree to Linked List, 二叉树展开为链表, 先序遍历, 原地修改, 反向先序, LeetCode 114
- **元描述**：系统讲透 LeetCode 114 的反向先序重连思路，解释 prev 指针为什么有效，并补充工程迁移、复杂度与进阶 O(1) 方案。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一棵二叉树，请把它原地展开成一个“只沿 `right` 指针连接”的单链表。

展开后要满足两条规则：

- 每个节点的 `left` 都必须变成 `null`
- 沿 `right` 走出来的顺序，必须与原树的先序遍历顺序完全一致

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `root` | `TreeNode` | 二叉树根节点 |
| 返回值 | 无 | 原地修改 `root` |

### 示例 1

```text
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

### 示例 2

```text
输入：root = []
输出：[]
```

### 示例 3

```text
输入：root = [0]
输出：[0]
```

### 提示

- 树中结点数在范围 `[0, 2000]` 内
- `-100 <= Node.val <= 100`

### 进阶

你可以使用原地算法（`O(1)` 额外空间）展开这棵树吗？

---

## 目标读者

- 已经会先序遍历，但还不会把遍历顺序变成“链式结构”的学习者
- 一写树上原地修改就容易丢指针、覆盖结构的开发者
- 想把 114 和 105、199 这类“树结构重组题”串起来的读者

## 背景 / 动机

这题训练的不是遍历本身，而是：

- 在修改指针的同时，仍然保持目标顺序正确

很多人第一次看到 114，会下意识想：

- 先做一遍先序遍历，把节点放进数组
- 再第二遍把数组里的节点串成链表

这样能做，但会多用 `O(n)` 额外空间，而且也没真正掌握“树怎么原地重连”。

更本质的问题应该是：

> 如果目标链表顺序就是先序遍历顺序，我应该按什么方向修改指针，才不会把后面的结构弄丢？

## 核心概念

- **先序遍历**：根、左、右
- **反向先序**：右、左、根
- **`prev` 指针**：保存当前节点在展开链表里的后继
- **原地重连**：直接改动原树节点的左右指针，不新建节点

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先看目标链表到底长什么样

看官方示例：

```text
    1
   / \
  2   5
 / \   \
3   4   6
```

这棵树的先序遍历是：

```text
[1,2,3,4,5,6]
```

展开后的链表也必须正好是：

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

所以题目并不是随便拍平，而是要求：

> 按先序遍历顺序把节点串成一条 `right` 链。

#### Step 2：如果从前往后做，我们最容易丢什么信息？

假设你现在在节点 `1`，想立刻把：

```text
1.right = 2
1.left = None
```

你会马上遇到一个风险：

- 原来 `1.right` 指向的那棵右子树 `5` 可能会丢

所以直接按“先序顺序正向改”很容易破坏还没处理的结构。

#### Step 3：那能不能把方向反过来？

既然正向改容易丢后继，那我们就反过来想：

> 如果我先把链表后半段准备好，再回头处理当前节点，是不是更安全？

目标顺序是：

```text
根 -> 左 -> 右
```

反过来处理就是：

```text
右 -> 左 -> 根
```

这就是“反向先序”。

#### Step 4：什么时候说明递归工作已经完成？

最基本的 base case 还是：

```python
if node is None:
    return
```

空节点不需要展开。

#### Step 5：为什么递归顺序必须是“先右后左”？

因为我们希望当前节点在重连时，它后面的那条链已经准备好了。

而先序目标顺序是：

```text
node -> node.left -> node.right
```

所以反向处理必须先搞定：

```text
node.right
node.left
node
```

代码就是：

```python
dfs(node.right)
dfs(node.left)
```

#### Step 6：`prev` 到底表示什么？

当我们按右、左、根的顺序回溯时，`prev` 始终表示：

> 在展开后的链表里，当前节点后面应该接上的那个节点。

于是当前节点的重连动作就非常直接：

```python
node.right = prev
node.left = None
prev = node
```

#### Step 7：为什么这三句就够了？

因为此时：

- `node.right` 子树已经被展开好了
- `node.left` 子树也已经被展开好了
- `prev` 指向的是当前节点在最终链表中的下一个节点

所以只要：

- 把 `right` 接到 `prev`
- 把 `left` 清空
- 再让 `prev` 更新成当前节点

整条链就会一步步往前长出来。

#### Step 8：慢速走一遍关键分支

仍看示例：

```text
[1,2,5,3,4,null,6]
```

反向先序访问顺序是：

```text
6 -> 5 -> 4 -> 3 -> 2 -> 1
```

回溯过程里：

1. `6.right = None`，`prev = 6`
2. `5.right = 6`，`prev = 5`
3. `4.right = 5`，`prev = 4`
4. `3.right = 4`，`prev = 3`
5. `2.right = 3`，`prev = 2`
6. `1.right = 2`，`prev = 1`

最后得到的正好就是先序链。

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- 反向先序顺序：右、左、根
- `prev` 的语义
- 三句重连代码

现在只差把它们合成一个完整递归。

### Assemble the Full Code

先给一版可直接运行的 Python 示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


def collect_chain(root):
    ans = []
    cur = root
    while cur:
        ans.append(cur.val)
        cur = cur.right
    return ans


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(5, None, TreeNode(6)),
    )
    flatten(root)
    print(collect_chain(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以整理成下面这样：

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        prev = None

        def dfs(node: Optional[TreeNode]) -> None:
            nonlocal prev
            if node is None:
                return
            dfs(node.right)
            dfs(node.left)
            node.right = prev
            node.left = None
            prev = node

        dfs(root)
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 反向先序递归
- 树上原地重连
- `prev` 后继拼接法

但真正要记住的是：

> 当前节点如果想安全地指向“先序里的下一个节点”，最好先把后半段链表准备好，再回头接自己。

---

## E — Engineering（工程应用）

### 场景 1：把树形菜单拍平成启动顺序链（Python）

**背景**：有些系统会先用树表达模块依赖，再在运行前把它整理成一条固定执行链。  
**为什么适用**：如果目标顺序已经明确，树上原地重连可以避免额外复制节点。

```python
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


root = Node("core", Node("auth"), Node("feed"))
flatten(root)
print(root.name, root.right.name)
```

### 场景 2：任务树转单链执行序列（Go）

**背景**：一棵任务树需要被转换成严格顺序执行的流水链。  
**为什么适用**：反向先序能让“后继任务”在回溯时自然就位。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func flatten(root *Node) {
	var prev *Node
	var dfs func(*Node)
	dfs = func(node *Node) {
		if node == nil {
			return
		}
		dfs(node.Right)
		dfs(node.Left)
		node.Right = prev
		node.Left = nil
		prev = node
	}
	dfs(root)
}

func main() {
	root := &Node{Name: "A", Left: &Node{Name: "B"}, Right: &Node{Name: "C"}}
	flatten(root)
	fmt.Println(root.Name, root.Right.Name)
}
```

### 场景 3：前端把树形导航压成线性 Tab 顺序（JavaScript）

**背景**：某些简化模式下，树形导航需要变成单线性焦点流。  
**为什么适用**：只要目标顺序固定，就可以直接复用“树到右链”的重连思路。

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function flatten(root) {
  let prev = null;
  function dfs(node) {
    if (!node) return;
    dfs(node.right);
    dfs(node.left);
    node.right = prev;
    node.left = null;
    prev = node;
  }
  dfs(root);
}

const root = new Node("Home", new Node("Docs"), new Node("Blog"));
flatten(root);
console.log(root.name, root.right.name);
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点重连一次
- **空间复杂度**：`O(h)`，来自递归栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 反向先序 + `prev` | `O(n)` | `O(h)` | 最好理解，写法稳定 |
| 先序遍历收集到数组再重连 | `O(n)` | `O(n)` | 简单，但空间更多 |
| 栈模拟先序再重连 | `O(n)` | `O(n)` | 可做，但维护更重 |
| 前驱拼接迭代法 | `O(n)` | `O(1)` | 满足进阶，但推导难度更高 |

### 常见错误与注意事项

1. **按先序正向改指针**：很容易把原右子树提前覆盖掉。  
2. **忘记清空 `left`**：链表要求所有左指针都为 `null`。  
3. **`prev` 定义不清**：它不是“上一个访问节点”，而是“当前节点在最终链表里的后继”。  
4. **把递归顺序写成左、右、根**：那样重连顺序就不对了。

## 常见问题与注意事项

### 1. 为什么要先递归右子树，再递归左子树？

因为最终链表顺序是：

```text
根 -> 左 -> 右
```

反向处理时就必须是：

```text
右 -> 左 -> 根
```

这样当前节点回溯时，它后面的链已经完整了。

### 2. `prev` 为什么能直接作为当前节点的 `right`？

因为在反向先序回溯到当前节点时，`prev` 恰好就是先序序列里“当前节点后面的那一串链”的头节点。

### 3. 进阶 `O(1)` 怎么做？

更进一步可以用迭代前驱拼接：

- 找到当前节点左子树最右侧节点
- 把它接到当前节点原来的右子树前面
- 再把左子树整体挪到右边

这个方法空间更优，但首次理解门槛更高。

## 最佳实践与建议

- 先把目标顺序写出来，再决定遍历方向
- 一旦发现正向修改容易丢后继，就试着反向处理
- 写树上原地重连时，先给状态变量写一句中文定义
- 做完 114 后，顺手练 144 和 105，树的“遍历顺序”和“结构重建”会更稳

## S — Summary（总结）

- 114 的目标链表顺序就是先序遍历顺序
- 为了不丢后继，最稳定的写法是用反向先序和 `prev` 指针回溯重连
- `prev` 表示“当前节点在最终链表里的下一个节点”，这一定义非常关键
- 这题本质上是在练“树结构原地改造”，不只是普通遍历
- 如果你还想进一步压空间，可以再挑战前驱拼接的 `O(1)` 进阶写法

## 参考与延伸阅读

- [LeetCode 114：二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)
- LeetCode 144：二叉树的前序遍历
- LeetCode 105：从前序与中序遍历序列构造二叉树
- LeetCode 173：二叉搜索树迭代器

## CTA

建议把 `144 + 114 + 105` 一起练。
144 帮你熟悉先序顺序，114 让你把顺序变成结构，105 则反过来用顺序重建结构，这三题连起来特别适合巩固“遍历顺序和树结构之间的关系”。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if node is None:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


def collect(root):
    ans = []
    cur = root
    while cur:
        ans.append(cur.val)
        cur = cur.right
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5, None, TreeNode(6)))
    flatten(root)
    print(collect(root))
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

void dfs(struct TreeNode* node, struct TreeNode** prev) {
    if (!node) return;
    dfs(node->right, prev);
    dfs(node->left, prev);
    node->right = *prev;
    node->left = NULL;
    *prev = node;
}

void flatten(struct TreeNode* root) {
    struct TreeNode* prev = NULL;
    dfs(root, &prev);
}

void print_chain(struct TreeNode* root) {
    while (root) {
        printf("%d ", root->val);
        root = root->right;
    }
    printf("\n");
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->left = new_node(2);
    root->right = new_node(5);
    root->left->left = new_node(3);
    root->left->right = new_node(4);
    root->right->right = new_node(6);
    flatten(root);
    print_chain(root);
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

void dfs(TreeNode* node, TreeNode*& prev) {
    if (!node) return;
    dfs(node->right, prev);
    dfs(node->left, prev);
    node->right = prev;
    node->left = nullptr;
    prev = node;
}

void flatten(TreeNode* root) {
    TreeNode* prev = nullptr;
    dfs(root, prev);
}

void printChain(TreeNode* root) {
    while (root) {
        std::cout << root->val << ' ';
        root = root->right;
    }
    std::cout << '\n';
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(5);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->right = new TreeNode(6);
    flatten(root);
    printChain(root);
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

func flatten(root *TreeNode) {
	var prev *TreeNode
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		dfs(node.Left)
		node.Right = prev
		node.Left = nil
		prev = node
	}
	dfs(root)
}

func printChain(root *TreeNode) {
	for root != nil {
		fmt.Print(root.Val, " ")
		root = root.Right
	}
	fmt.Println()
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 3},
			Right: &TreeNode{Val: 4},
		},
		Right: &TreeNode{
			Val:   5,
			Right: &TreeNode{Val: 6},
		},
	}
	flatten(root)
	printChain(root)
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn flatten(root: &mut Option<Box<TreeNode>>) {
    fn dfs(node: &mut Option<Box<TreeNode>>, prev: &mut Option<Box<TreeNode>>) {
        if let Some(mut cur) = node.take() {
            dfs(&mut cur.right, prev);
            dfs(&mut cur.left, prev);
            cur.right = prev.take();
            cur.left = None;
            *prev = Some(cur);
        }
    }

    let mut prev = None;
    dfs(root, &mut prev);
    *root = prev;
}

fn collect(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    let mut ans = Vec::new();
    let mut cur = root.as_deref();
    while let Some(node) = cur {
        ans.push(node.val);
        cur = node.right.as_deref();
    }
    ans
}

fn main() {
    let mut root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 3,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 5,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 6,
                left: None,
                right: None,
            })),
        })),
    }));

    flatten(&mut root);
    println!("{:?}", collect(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function flatten(root) {
  let prev = null;
  function dfs(node) {
    if (!node) return;
    dfs(node.right);
    dfs(node.left);
    node.right = prev;
    node.left = null;
    prev = node;
  }
  dfs(root);
}

function collect(root) {
  const ans = [];
  let cur = root;
  while (cur) {
    ans.push(cur.val);
    cur = cur.right;
  }
  return ans;
}

const root = new TreeNode(
  1,
  new TreeNode(2, new TreeNode(3), new TreeNode(4)),
  new TreeNode(5, null, new TreeNode(6))
);
flatten(root);
console.log(collect(root));
```
