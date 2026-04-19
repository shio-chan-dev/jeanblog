---
title: "Hot100：二叉树的最近公共祖先（Lowest Common Ancestor of a Binary Tree）后序返回值语义 ACERS 解析"
date: 2026-04-19T14:52:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "LCA", "DFS", "后序遍历", "LeetCode 236"]
description: "围绕 LeetCode 236 讲清最近公共祖先的后序返回值语义：子树返回什么、什么时候当前节点成为答案、以及为什么节点自己也可以是祖先。"
keywords: ["Lowest Common Ancestor of a Binary Tree", "二叉树的最近公共祖先", "LCA", "后序遍历", "DFS", "LeetCode 236", "Hot100"]
---

> **副标题 / 摘要**
> LeetCode 236 的真正难点不是“记住 LCA 模板”，而是先定义清楚：递归函数到底要向父节点返回什么信息。只要这个返回值语义稳定，整题就会自然落成一段非常短但非常强的后序递归。

- **预计阅读时长**：11~14 分钟
- **标签**：`Hot100`、`二叉树`、`LCA`、`DFS`、`后序遍历`
- **SEO 关键词**：Lowest Common Ancestor of a Binary Tree, 二叉树的最近公共祖先, LCA, 后序遍历, DFS, LeetCode 236
- **元描述**：系统讲透 LeetCode 236 的后序返回值定义、节点自祖先规则、递归推导过程、工程迁移和多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一棵二叉树，找到该树中两个指定节点 `p` 和 `q` 的最近公共祖先。

最近公共祖先（LCA）的定义是：

- `x` 同时是 `p` 和 `q` 的祖先
- 在满足上面条件的节点里，`x` 的深度尽可能大
- 一个节点也可以是它自己的祖先

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点 |
| p, q | TreeNode | 树中的两个指定节点；示例输入里用它们的唯一值表示 |
| 返回 | TreeNode | `p` 和 `q` 的最近公共祖先 |

### 示例 1

```text
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

### 示例 2

```text
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```

### 示例 3

```text
输入：root = [1,2], p = 1, q = 2
输出：1
```

### 提示

- 树中节点数目在范围 `[2, 10^5]` 内
- `-10^9 <= Node.val <= 10^9`
- 所有 `Node.val` 互不相同
- `p != q`
- `p` 和 `q` 均存在于给定的二叉树中

---

## 目标读者

- 已经会写普通树递归，但一到“最近公共祖先”就容易卡在返回值定义上的学习者
- 想把后序分治写法固定成稳定模板的开发者
- 在工程里处理组织树、组件树、目录树共享祖先问题的工程师

## 背景 / 动机

很多人第一次做 236，会下意识想到：

- 给节点加 parent 指针
- 从 `p` 往上走，再从 `q` 往上走
- 或者分别找两条根到节点路径再比较

这些方法都能做，但这题最值得学的是更干净的版本：

- 不额外建图
- 不额外存父指针
- 一次后序递归就把答案找出来

它的关键不在于代码短，而在于你是否先想清：

> 每个子树递归完成后，应该向父节点报告什么？

## 核心概念

- **后序递归**：先处理左右子树，再用左右结果决定当前节点含义
- **返回值语义**：返回“当前子树里已经找到的目标节点，或已经确认的 LCA”
- **节点自祖先**：如果 `p` 本身就是 `q` 的祖先，答案可以直接是 `p`
- **向上冒泡**：某个子树已经带着答案回来时，父节点只需要继续上传

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先从“节点自己也可以是祖先”这个事实出发

看官方示例 2：

```text
p = 5, q = 4
答案 = 5
```

这说明一件很重要的事：

- 我们不能把“公共祖先”理解成“必须严格在上面”
- 只要一个节点本身就是另一个节点的祖先，它就可能直接是答案

所以写递归时，遇到 `p` 或 `q` 不能随便跳过，这个信息非常重要。

#### Step 2：当前子树最少要向父节点返回什么？

递归写不顺，通常是因为返回值语义不稳定。
这题最好的定义是：

> `dfs(node)` 返回“在以 `node` 为根的子树里，目前已经确认的那个关键节点”。

这个“关键节点”可能是三种情况之一：

- 没找到任何目标，返回 `None`
- 找到了 `p` 或 `q` 其中之一，返回那个节点
- 左右两边各找到一个，此时当前节点就是 LCA，返回当前节点

这一定义一旦稳定，后面代码会非常自然。

#### Step 3：递归真正要解决的子问题是什么？

把原题拆小一点：

> 如果左右子树都已经告诉我它们各自找到了什么，那我怎么判断当前节点要返回什么？

这就是典型的后序分治：

```python
left = dfs(node.left)
right = dfs(node.right)
```

#### Step 4：什么时候可以立刻返回，不必继续往下？

有两个最关键的 base case：

```python
if node is None:
    return None
if node.val == p or node.val == q:
    return node
```

这里第二个判断尤其重要。
一旦当前节点就是目标之一，这个节点必须立刻被保留下来，继续向上汇报。

#### Step 5：左右子树返回结果后，有哪些可能性？

拿到 `left` 和 `right` 之后，情况只剩下三类：

- 左右都为空：当前子树什么也没找到
- 只有一边非空：说明目标节点或答案在那一边
- 左右都非空：说明 `p`、`q` 分别落在两边，当前节点就是最近公共祖先

这一步其实就是整题的判定核心。

#### Step 6：什么时候当前节点就是答案？

如果左右都非空：

```python
if left and right:
    return node
```

因为这说明：

- 左子树里至少命中了一个目标
- 右子树里至少命中了另一个目标

而当前节点是它们第一次汇合的地方。

#### Step 7：如果只有一边非空，为什么要直接向上返回？

因为当前节点还不能确认自己是答案。
它只是知道：

- 有一个目标节点，或者已经确认的 LCA，在某一边

所以直接把非空结果向上冒泡即可：

```python
return left if left else right
```

#### Step 8：慢速走一条分支

看官方示例 1：

```text
root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
```

递归到节点 `5` 时，因为它本身就是目标之一，所以直接返回节点 `5`。  
递归到节点 `1` 时，同理返回节点 `1`。

回到根节点 `3`：

- `left = 5`
- `right = 1`

左右都非空，所以根节点 `3` 就是最近公共祖先。

#### Step 9：为了让示例可运行，代码里传“值”还是传“节点”？

LeetCode 题面语义是传节点 `p` 和 `q`。  
但因为本题明确保证 `Node.val` 互不相同，所以在博客里的可运行示例中，直接传 `pVal` 和 `qVal` 也能得到完全等价的判定结果。

这能让示例更自包含，尤其是多语言版本更容易直接运行。

### Assemble the Full Code

下面把刚才的碎片拼成第一版完整代码。
这版示例用 `pVal` 和 `qVal` 传入目标节点值，便于直接运行。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root, pVal, qVal):
    def dfs(node):
        if node is None:
            return None
        if node.val == pVal or node.val == qVal:
            return node

        left = dfs(node.left)
        right = dfs(node.right)

        if left and right:
            return node
        return left if left else right

    return dfs(root)


if __name__ == "__main__":
    root = TreeNode(
        3,
        TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4))),
        TreeNode(1, TreeNode(0), TreeNode(8)),
    )
    ans = lowest_common_ancestor(root, 5, 1)
    print(ans.val if ans else None)
```

### Reference Answer

如果你要提交到 LeetCode，核心逻辑其实就是下面这几行：

```python
from typing import Optional


class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class Solution:
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        if root is None or root is p or root is q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        return left if left else right
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 后序递归
- 树上分治
- LCA 返回值语义法

但更重要的是这个思维顺序：

1. 先定义子树要返回什么  
2. 再写左右子树递归  
3. 最后用左右结果决定当前节点含义

---

## E — Engineering（工程应用）

### 场景 1：组织架构里找最近共同上级（Python）

**背景**：在组织树里，常常要找两个员工最近的共同汇报节点。  
**为什么适用**：这就是 LCA 的原型问题，答案可能是更高层领导，也可能就是某个员工本人。

```python
class Staff:
    def __init__(self, id_, left=None, right=None):
        self.id = id_
        self.left = left
        self.right = right


def lca(root, a, b):
    if root is None or root.id == a or root.id == b:
        return root
    left = lca(root.left, a, b)
    right = lca(root.right, a, b)
    if left and right:
        return root
    return left or right


root = Staff(3, Staff(5), Staff(1))
print(lca(root, 5, 1).id)
```

### 场景 2：前端组件树里找最近公共容器（JavaScript）

**背景**：调试两个元素共用的布局容器或事件边界时，经常要在组件树里找最近公共父容器。  
**为什么适用**：LCA 直接描述了“两个节点第一次在树上汇合的位置”。

```javascript
function Node(id, left = null, right = null) {
  this.id = id;
  this.left = left;
  this.right = right;
}

function lca(root, a, b) {
  if (!root || root.id === a || root.id === b) return root;
  const left = lca(root.left, a, b);
  const right = lca(root.right, a, b);
  if (left && right) return root;
  return left || right;
}

const root = new Node("page", new Node("sidebar"), new Node("content"));
console.log(lca(root, "sidebar", "content").id);
```

### 场景 3：目录树里找最近公共父目录（Go）

**背景**：对两个文件或两个目录做批量操作时，常要先定位它们最近的共同父目录。  
**为什么适用**：目录树上的“最近共同父节点”就是 LCA 问题的工程映射。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func lca(root *Node, a, b string) *Node {
	if root == nil || root.Name == a || root.Name == b {
		return root
	}
	left := lca(root.Left, a, b)
	right := lca(root.Right, a, b)
	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	}
	return right
}

func main() {
	root := &Node{Name: "root", Left: &Node{Name: "docs"}, Right: &Node{Name: "blog"}}
	fmt.Println(lca(root, "docs", "blog").Name)
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点最多访问一次
- **空间复杂度**：`O(h)`，来自递归栈，`h` 为树高

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 后序递归返回值语义 | `O(n)` | `O(h)` | 最简洁，最贴题 |
| parent map + 祖先集合 | `O(n)` | `O(n)` | 适合迭代写法，但要额外建父指针 |
| 根到节点路径比较 | `O(n)` | `O(h)` 到 `O(n)` | 思路直观，但要做两次路径搜索 |

### 常见错误与注意事项

1. **忘记“节点自己也可以是祖先”**：这会导致示例 2 之类的情况直接写错。  
2. **没有把 `root == p` 或 `root == q` 作为 base case**：这样目标节点的信息就传不上去。  
3. **把“找到一个目标”误判成“已经找到答案”**：只有左右两边都非空时，当前节点才是 LCA。  
4. **在值不唯一的树上直接按值比较**：本文示例这样写是因为题目明确保证 `Node.val` 唯一。

## 常见问题与注意事项

### 1. 如果 `p` 本身就是 `q` 的祖先怎么办？

答案就是 `p`。
这正是题面里“一个节点也可以是它自己的祖先”的含义。

### 2. 为什么这题适合后序遍历？

因为当前节点要先知道左右子树分别找到了什么，才能判断自己是不是第一次汇合点。
这天然就是“先左右、再当前”的后序逻辑。

### 3. 如果题目不保证 `p` 和 `q` 一定存在呢？

那返回值设计就要更严格，通常要额外携带“找到几个目标节点”的计数。
本题因为明确保证两者都存在，所以标准写法可以更简洁。

## 最佳实践与建议

- 先把“返回值语义”用一句中文写清，再开始写代码
- 一进递归先判断 `None / p / q`，这是最稳定的入口
- 看到“两个子树各返回一个非空”时，立刻想到“当前节点就是汇合点”
- 做完 236 后，再去做 235，对比“普通二叉树”和“BST 上的 LCA”差别

## S — Summary（总结）

- 236 的核心不是模板名，而是递归返回值到底代表什么
- 一旦把返回值定义成“子树里已经确认的关键节点”，整题会非常顺
- 后序递归适合这题，因为当前节点必须先拿到左右子树结果再决策
- “节点自己也可以是祖先”是这题最容易忽略却最关键的规则
- 这道题会明显提升你写树上分治和结果向上冒泡的稳定性

## 参考与延伸阅读

- [LeetCode 236：二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)
- LeetCode 235：二叉搜索树的最近公共祖先
- LeetCode 104：二叉树的最大深度
- LeetCode 543：二叉树的直径

## CTA

建议把 `236 + 235` 放在一起做。
前者训练“后序返回值语义”，后者训练“利用 BST 有序性剪枝”，两题放一起对比，LCA 这一类题会立刻清楚很多。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root, pVal, qVal):
    if root is None or root.val == pVal or root.val == qVal:
        return root

    left = lowest_common_ancestor(root.left, pVal, qVal)
    right = lowest_common_ancestor(root.right, pVal, qVal)

    if left and right:
        return root
    return left if left else right


if __name__ == "__main__":
    root = TreeNode(
        3,
        TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4))),
        TreeNode(1, TreeNode(0), TreeNode(8)),
    )
    ans = lowest_common_ancestor(root, 5, 1)
    print(ans.val if ans else None)
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

struct TreeNode* lowestCommonAncestor(struct TreeNode* root, int p, int q) {
    if (root == NULL || root->val == p || root->val == q) return root;

    struct TreeNode* left = lowestCommonAncestor(root->left, p, q);
    struct TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(3);
    root->left = new_node(5);
    root->right = new_node(1);
    root->left->left = new_node(6);
    root->left->right = new_node(2);
    root->left->right->left = new_node(7);
    root->left->right->right = new_node(4);
    root->right->left = new_node(0);
    root->right->right = new_node(8);

    struct TreeNode* ans = lowestCommonAncestor(root, 5, 1);
    printf("%d\n", ans ? ans->val : -1);
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

TreeNode* lowestCommonAncestor(TreeNode* root, int p, int q) {
    if (!root || root->val == p || root->val == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(5);
    root->right = new TreeNode(1);
    root->left->left = new TreeNode(6);
    root->left->right = new TreeNode(2);
    root->left->right->left = new TreeNode(7);
    root->left->right->right = new TreeNode(4);
    root->right->left = new TreeNode(0);
    root->right->right = new TreeNode(8);

    TreeNode* ans = lowestCommonAncestor(root, 5, 1);
    std::cout << (ans ? ans->val : -1) << '\n';
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

func lowestCommonAncestor(root *TreeNode, p, q int) *TreeNode {
	if root == nil || root.Val == p || root.Val == q {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)

	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	}
	return right
}

func main() {
	root := &TreeNode{
		Val: 3,
		Left: &TreeNode{
			Val: 5,
			Left: &TreeNode{Val: 6},
			Right: &TreeNode{
				Val:   2,
				Left:  &TreeNode{Val: 7},
				Right: &TreeNode{Val: 4},
			},
		},
		Right: &TreeNode{
			Val:   1,
			Left:  &TreeNode{Val: 0},
			Right: &TreeNode{Val: 8},
		},
	}

	ans := lowestCommonAncestor(root, 5, 1)
	fmt.Println(ans.Val)
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn lowest_common_ancestor(root: &Option<Box<TreeNode>>, p: i32, q: i32) -> Option<i32> {
    match root {
        None => None,
        Some(node) => {
            if node.val == p || node.val == q {
                return Some(node.val);
            }

            let left = lowest_common_ancestor(&node.left, p, q);
            let right = lowest_common_ancestor(&node.right, p, q);

            if left.is_some() && right.is_some() {
                return Some(node.val);
            }
            if left.is_some() {
                return left;
            }
            right
        }
    }
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 3,
        left: Some(Box::new(TreeNode {
            val: 5,
            left: Some(Box::new(TreeNode {
                val: 6,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 2,
                left: Some(Box::new(TreeNode {
                    val: 7,
                    left: None,
                    right: None,
                })),
                right: Some(Box::new(TreeNode {
                    val: 4,
                    left: None,
                    right: None,
                })),
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 1,
            left: Some(Box::new(TreeNode {
                val: 0,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 8,
                left: None,
                right: None,
            })),
        })),
    }));

    println!("{:?}", lowest_common_ancestor(&root, 5, 1));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function lowestCommonAncestor(root, p, q) {
  if (!root || root.val === p || root.val === q) return root;

  const left = lowestCommonAncestor(root.left, p, q);
  const right = lowestCommonAncestor(root.right, p, q);

  if (left && right) return root;
  return left || right;
}

const root = new TreeNode(
  3,
  new TreeNode(5, new TreeNode(6), new TreeNode(2, new TreeNode(7), new TreeNode(4))),
  new TreeNode(1, new TreeNode(0), new TreeNode(8))
);

const ans = lowestCommonAncestor(root, 5, 1);
console.log(ans ? ans.val : null);
```
