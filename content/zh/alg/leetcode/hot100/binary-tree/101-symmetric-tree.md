---
title: "Hot100：对称二叉树（Symmetric Tree）镜像递归 / BFS ACERS 解析"
date: 2026-03-15T21:29:43+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "DFS", "BFS", "对称性", "LeetCode 101"]
description: "围绕 LeetCode 101 讲清镜像递归、BFS 成对入队与树形结构对称校验，附工程迁移和多语言实现。"
keywords: ["Symmetric Tree", "对称二叉树", "镜像递归", "BFS", "LeetCode 101", "Hot100"]
---

> **副标题 / 摘要**  
> 对称二叉树的难点不在遍历，而在“比较方向”。你比较的不是左对左、右对右，而是镜像位置上的节点对。本文按 ACERS 结构拆解 LeetCode 101 的镜像递归合同、BFS 成对入队写法，以及工程中的对称结构校验场景。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`二叉树`、`DFS`、`BFS`、`对称性`  
- **SEO 关键词**：Hot100, Symmetric Tree, 对称二叉树, 镜像递归, BFS, LeetCode 101  
- **元描述**：系统讲透 LeetCode 101 的镜像递归与 BFS 对称校验思路，并延伸到布局树与拓扑模板的对称检查。  

---

## 目标读者

- 刚从 100 相同的树过渡到“镜像比较”的刷题读者
- 会写普通树递归，但对“外侧 / 内侧”比较关系容易写乱的开发者
- 需要在布局树、模板树、镜像结构里做左右对称校验的工程师

## 背景 / 动机

LeetCode 101 很适合作为树题里的“方向感”训练：

- 你要先意识到，对称不是“左右子树完全一样”
- 它要求的是“左边看过去”和“右边镜像过来”之后一致
- 也就是说，比较方向从“同向”变成了“交叉”

很多人做这题时容易犯三类错误：

- 还沿用 100 的思路，写成 `left.left` 对 `right.left`
- 只比较节点值，不比较空节点位置
- 先翻转一棵子树再比较，结果多做了一轮变换，逻辑也更绕

这题真正训练的是“**镜像递归模板**”。掌握后，树对称、树镜像、结构匹配等题都会更清楚。

## 核心概念

- **镜像关系**：左子树的左边，要和右子树的右边对应；左子树的右边，要和右子树的左边对应
- **外侧 / 内侧配对**：`left.left` 对 `right.right`，`left.right` 对 `right.left`
- **成对递归**：递归函数参数是两个节点，表示“这两个位置是否互为镜像”
- **成对入队**：BFS 里队列保存的不是单个节点，而是需要一起比较的节点对

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个二叉树的根节点 `root`，检查它是否轴对称。

如果一棵树的左子树和右子树互为镜像，则它是对称的。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点 |
| 返回值 | bool | 该树是否对称 |

### 示例 1

```text
输入: root = [1,2,2,3,4,4,3]
输出: true
解释:
左子树与右子树的镜像结构和节点值都一一对应，因此整棵树对称。
```

### 示例 2

```text
输入: root = [1,2,2,null,3,null,3]
输出: false
解释:
左子树的右孩子和右子树的右孩子出现在同一方向，不构成镜像。
```

### 约束

- 树中节点数目在 `[1, 1000]` 范围内
- `-100 <= Node.val <= 100`

---

## C — Concepts（核心思想）

### 思路推导：对称性要比较“镜像位置”

假设当前正在比较两个节点 `a` 和 `b`，它们要想互为镜像，必须同时满足：

1. **都为空**：这对位置匹配，返回 `true`
2. **只有一个为空**：结构破坏，返回 `false`
3. **值不同**：镜像节点值不一致，返回 `false`
4. **值相同且都非空**：
   - 比较 `a.left` 和 `b.right`
   - 比较 `a.right` 和 `b.left`

写成公式就是：

```text
mirror(a, b) =
    true, if a == null and b == null
    false, if exactly one is null
    false, if a.val != b.val
    mirror(a.left, b.right) and mirror(a.right, b.left), otherwise
```

### 为什么不能“左对左、右对右”

那样比较的是“相同”，不是“镜像”。  
101 和 100 的最大差异正好在这里：

- **100 相同的树**：`left.left` 对 `right.left`
- **101 对称二叉树**：`left.left` 对 `right.right`

这就是“同向比较”和“镜像比较”的本质区别。

### 方法归类

- **树镜像递归 / DFS**
- **BFS 成对入队**
- **结构对称性校验**

### BFS 为什么也适合

如果你不想写递归，也可以把镜像节点对放进队列：

1. 队列初始放入 `root.left` 和 `root.right`
2. 每次弹出一对节点，按镜像合同判断
3. 若当前匹配，则继续入队：
   - `left.left` 和 `right.right`
   - `left.right` 和 `right.left`

这样就把递归的“节点对”显式展开了。

---

## 实践指南 / 步骤

### 推荐写法：镜像递归

1. 如果根节点为空，直接返回 `true`
2. 定义辅助函数 `is_mirror(a, b)`
3. 在辅助函数里按“空 / 单空 / 值不同 / 递归比较”的顺序写
4. 最终返回 `is_mirror(root.left, root.right)`

Python 可运行示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return is_mirror(a.left, b.right) and is_mirror(a.right, b.left)

    return True if root is None else is_mirror(root.left, root.right)


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(2, TreeNode(4), TreeNode(3)),
    )
    print(is_symmetric(root))
```

### BFS 备选写法

非递归版本通常这样写：

1. 用队列保存镜像节点对
2. 每次弹出两个节点做比较
3. 匹配成功后按“外侧一对、内侧一对”的顺序继续入队

如果你需要显式记录是哪个节点对不对称，BFS 会更便于调试。

---

## E — Engineering（工程应用）

### 场景 1：双栏页面布局镜像校验（JavaScript）

**背景**：可视化编辑器里，常会有“左右镜像布局”模板。  
**为什么适用**：模板发布前，往往要校验左右区域是不是严格镜像，避免交互区错位。

```javascript
function isMirror(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (a.type !== b.type) return false;
  return isMirror(a.left, b.right) && isMirror(a.right, b.left);
}

const left = { type: "Split", left: { type: "Menu" }, right: { type: "Detail" } };
const right = { type: "Split", left: { type: "Detail" }, right: { type: "Menu" } };
console.log(isMirror(left, right));
```

### 场景 2：双活机房拓扑模板对称检查（Python）

**背景**：双活架构常要求左右两侧机房拓扑在角色与层级上互为镜像。  
**为什么适用**：上线前可快速检查模板树是否保持对称，避免一侧缺少节点或角色错位。

```python
def mirror_role(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a["role"] != b["role"]:
        return False
    return mirror_role(a.get("left"), b.get("right")) and mirror_role(a.get("right"), b.get("left"))


left_dc = {"role": "gateway", "left": {"role": "api"}, "right": {"role": "db"}}
right_dc = {"role": "gateway", "left": {"role": "db"}, "right": {"role": "api"}}
print(mirror_role(left_dc, right_dc))
```

### 场景 3：教学工具里的镜像树验收（Go）

**背景**：算法教学平台常会让学生构造一棵与目标模板镜像对应的树。  
**为什么适用**：判题时不只是看节点值，还要看镜像位置是否正确。

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func mirror(a, b *Node) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Val != b.Val {
		return false
	}
	return mirror(a.Left, b.Right) && mirror(a.Right, b.Left)
}

func main() {
	left := &Node{2, &Node{3, nil, nil}, &Node{4, nil, nil}}
	right := &Node{2, &Node{4, nil, nil}, &Node{3, nil, nil}}
	fmt.Println(mirror(left, right))
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点最多被比较一次
- **空间复杂度**：
  - 递归 DFS：`O(h)`，`h` 是树高
  - BFS 队列：最坏 `O(w)`，`w` 是某一层最大宽度

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 镜像递归 | `O(n)` | `O(h)` | 最符合定义，推荐 |
| BFS 成对入队 | `O(n)` | `O(w)` | 显式流程，调试方便 |
| 先翻转一侧再比较 | `O(n)` | `O(h)` 或 `O(w)` | 多了一次变换，且可能修改原树 |
| 序列化后比镜像序列 | `O(n)` | `O(n)` | 实现更绕，还要处理空节点占位 |

### 常见错误与注意事项

- 把 101 写成 100，仍然比较 `left.left` 对 `right.left`
- 只比较节点值，不比较空节点位置
- 先翻转树再判断，对本题来说不必要，还可能引入副作用
- BFS 时队列只存单个节点，而不是成对节点，导致信息丢失

## 常见问题与注意事项

### 1. 单节点树算对称吗？

算。因为它的左子树和右子树都为空，天然互为镜像。

### 2. 为什么不直接翻转左子树再和右子树比较？

可以做，但不推荐。那样多了一步结构变换，思路更绕，也可能修改原树。

### 3. DFS 和 BFS 该怎么选？

刷题和讲解时优先递归；如果你要显式记录比较路径、避免深递归，BFS 更顺手。

## 最佳实践与建议

- 做 101 前，先在纸上画出“外侧 / 内侧”配对关系
- 牢记模板：`left.left` 对 `right.right`，`left.right` 对 `right.left`
- 这题和 100 最适合对比着练，能快速建立方向感
- 如果递归一开始总写乱，先用 BFS 成对入队帮助自己把比较对固定下来

## S — Summary（总结）

- 对称二叉树的核心不是遍历，而是镜像位置比较
- 镜像递归只要守住“外侧对外侧、内侧对内侧”的合同，代码就会稳定
- BFS 版本本质相同，只是把递归节点对显式化
- 这题非常适合和 100、226 组成一组练，建立树结构判断的基础模板
- 工程里，凡是涉及左右镜像模板、对称布局、镜像拓扑，都能复用这套思路

## 参考与延伸阅读

- [LeetCode 101: Symmetric Tree](https://leetcode.cn/problems/symmetric-tree/)
- LeetCode 100：相同的树
- LeetCode 226：翻转二叉树
- LeetCode 104：二叉树的最大深度
- LeetCode 102：二叉树的层序遍历

## CTA

建议把 100、101、226 放成一个小专题连续练。  
100 练“同向比较”，101 练“镜像比较”，226 练“结构变换”；这三题放在一起，树结构直觉会很快成型。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return is_mirror(a.left, b.right) and is_mirror(a.right, b.left)

    return True if root is None else is_mirror(root.left, root.right)


if __name__ == "__main__":
    root = TreeNode(
        1,
        TreeNode(2, TreeNode(3), TreeNode(4)),
        TreeNode(2, TreeNode(4), TreeNode(3)),
    )
    print(is_symmetric(root))
```

```c
#include <stdbool.h>
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

bool isMirror(struct TreeNode* a, struct TreeNode* b) {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    if (a->val != b->val) return false;
    return isMirror(a->left, b->right) && isMirror(a->right, b->left);
}

bool isSymmetric(struct TreeNode* root) {
    if (root == NULL) return true;
    return isMirror(root->left, root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->left = new_node(2);
    root->right = new_node(2);
    root->left->left = new_node(3);
    root->left->right = new_node(4);
    root->right->left = new_node(4);
    root->right->right = new_node(3);

    printf("%s\n", isSymmetric(root) ? "true" : "false");
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

bool isMirror(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    if (a->val != b->val) return false;
    return isMirror(a->left, b->right) && isMirror(a->right, b->left);
}

bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return isMirror(root->left, root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(2);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(3);

    std::cout << (isSymmetric(root) ? "true" : "false") << '\n';
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

func isMirror(a *TreeNode, b *TreeNode) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Val != b.Val {
		return false
	}
	return isMirror(a.Left, b.Right) && isMirror(a.Right, b.Left)
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isMirror(root.Left, root.Right)
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
			Val:   2,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 3},
		},
	}
	fmt.Println(isSymmetric(root))
}
```

```rust
use std::cell::RefCell;
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

fn is_mirror(a: &Node, b: &Node) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            let xr = x.borrow();
            let yr = y.borrow();
            xr.val == yr.val
                && is_mirror(&xr.left, &yr.right)
                && is_mirror(&xr.right, &yr.left)
        }
        _ => false,
    }
}

fn is_symmetric(root: &Node) -> bool {
    match root {
        None => true,
        Some(node) => {
            let n = node.borrow();
            is_mirror(&n.left, &n.right)
        }
    }
}

fn main() {
    let root = Some(TreeNode::new(1));
    if let Some(node) = &root {
        let left = Some(TreeNode::new(2));
        let right = Some(TreeNode::new(2));
        if let Some(l) = &left {
            l.borrow_mut().left = Some(TreeNode::new(3));
            l.borrow_mut().right = Some(TreeNode::new(4));
        }
        if let Some(r) = &right {
            r.borrow_mut().left = Some(TreeNode::new(4));
            r.borrow_mut().right = Some(TreeNode::new(3));
        }
        node.borrow_mut().left = left;
        node.borrow_mut().right = right;
    }
    println!("{}", is_symmetric(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function isMirror(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (a.val !== b.val) return false;
  return isMirror(a.left, b.right) && isMirror(a.right, b.left);
}

function isSymmetric(root) {
  if (root === null) return true;
  return isMirror(root.left, root.right);
}

const root = new TreeNode(
  1,
  new TreeNode(2, new TreeNode(3), new TreeNode(4)),
  new TreeNode(2, new TreeNode(4), new TreeNode(3))
);
console.log(isSymmetric(root));
```
