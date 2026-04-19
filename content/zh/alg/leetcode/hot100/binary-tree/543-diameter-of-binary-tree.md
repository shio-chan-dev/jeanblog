---
title: "Hot100：二叉树的直径（Diameter of Binary Tree）树形 DP / 高度回传 ACERS 解析"
date: 2026-04-19T14:52:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "树形DP", "DFS", "后序遍历", "LeetCode 543"]
description: "围绕 LeetCode 543 讲清二叉树直径的核心：递归返回高度、在当前节点更新全局直径，以及为什么最长路径不一定只看根到叶。"
keywords: ["Diameter of Binary Tree", "二叉树的直径", "树形DP", "高度", "DFS", "LeetCode 543", "Hot100"]
---

> **副标题 / 摘要**
> LeetCode 543 最容易混乱的点是：递归函数到底应该返回“高度”还是“直径”。这题的稳定写法是后序遍历时向上返回高度，同时在每个节点尝试更新“经过当前节点的最长路径”。理解这个分工后，树形 DP 会一下子清楚很多。

- **预计阅读时长**：10~13 分钟
- **标签**：`Hot100`、`二叉树`、`树形DP`、`DFS`、`后序遍历`
- **SEO 关键词**：Diameter of Binary Tree, 二叉树的直径, 树形DP, 高度回传, DFS, LeetCode 543
- **元描述**：系统讲透 LeetCode 543 的后序高度回传法，包含递推推导、工程迁移、复杂度分析与多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一棵二叉树的根节点 `root`，返回该树的直径。

二叉树的直径指的是：

- 树中任意两个节点之间最长路径的长度
- 这条路径可以经过根节点，也可以不经过根节点
- 路径长度按**边数**计算

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点 |
| 返回 | int | 树的直径（最长路径边数） |

### 示例 1

```text
输入：root = [1,2,3,4,5]
输出：3
解释：长度为 3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 。
```

### 示例 2

```text
输入：root = [1,2]
输出：1
```

### 提示

- 树中节点数目在范围 `[1, 10^4]` 内
- `-100 <= Node.val <= 100`

---

## 目标读者

- 已经会写树深度递归，准备进入树形 DP 视角的学习者
- 容易把“返回值”和“全局答案”混在一起的开发者
- 在工程里处理层级传播链、最长链路或树状结构跨度问题的工程师

## 背景 / 动机

看到“二叉树的直径”，很多人第一反应会是：

- 求最大深度？
- 或者看根到叶的最长路径？

但这题真正问的是：

> 树里任意两个节点之间的最长路径有多长？

所以最长路径可能：

- 经过根
- 也可能完全藏在某一棵子树内部

这就是为什么 543 很适合作为“树形 DP 入门题”。
它迫使你把“子问题返回什么”和“全局最优怎么更新”拆开思考。

## 核心概念

- **高度（height）**：从当前节点往下走到最深叶子，最多能走出多长的单边路径
- **直径候选**：经过当前节点时，最长路径长度 = 左高度 + 右高度
- **后序遍历**：必须先拿到左右子树高度，才能计算当前节点的候选直径
- **全局答案**：直径是全树范围内的最大值，不能只靠单个返回值表达

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先抓住题目里最容易忽略的词

题目说的是：

- 任意两个节点之间
- 最长路径
- 按边数计算

这意味着它和“最大深度”不是同一个问题。
最大深度只看“从当前节点往下的一条链”，而直径看的是“某个节点左右两边拼起来的一整条路径”。

#### Step 2：如果想在当前节点算出一条候选最长路径，最少要知道什么？

假设当前节点是 `node`。
如果一条最长路径经过 `node`，那它一定长这样：

```text
左边最深叶子 -> ... -> node -> ... -> 右边最深叶子
```

所以我们最少要知道：

- 左子树能向下走多深
- 右子树能向下走多深

这就引出了递归返回值：**高度**。

#### Step 3：递归真正要解决的子问题是什么？

可以把问题拆成：

> 对每个节点，先求出左右子树高度，再用这两个高度更新经过当前节点的直径候选。

代码骨架自然就是后序：

```python
left = dfs(node.left)
right = dfs(node.right)
```

#### Step 4：空节点的高度应该是什么？

最稳定的定义是：

```python
if node is None:
    return 0
```

也就是把空节点高度记成 `0`。

这样：

- 叶子节点左右高度都是 `0`
- 叶子自身返回 `1`

这个定义对代码最顺手。

#### Step 5：为什么经过当前节点的候选直径是 `left + right`？

注意这里我们让 `dfs(node)` 返回的是：

> 以 `node` 为起点，向下最多能走出的节点层数

于是：

- `left` 表示左边单链长度
- `right` 表示右边单链长度

把它们在当前节点拼起来，得到的边数正好就是：

```python
left + right
```

例如叶子节点：

- `left = 0`
- `right = 0`
- 候选直径是 `0`

这正好符合“单个节点没有边”的定义。

#### Step 6：全局答案应该在哪里更新？

直径是全树范围的最大值，所以不能直接作为递归返回值一路传上去。
更稳的做法是：

- 递归只负责返回高度
- 直径放在外层变量 `ans` 里维护

```python
ans = max(ans, left + right)
```

#### Step 7：当前节点应该向父节点返回什么？

父节点并不需要知道“当前子树直径是多少”，它只需要知道：

> 如果要把当前节点接到更高层去，当前这边最多还能往下延伸多长？

所以返回值应该是：

```python
return 1 + max(left, right)
```

这一步特别关键：

- `ans` 负责全局最优
- `dfs` 返回值只负责向上提供高度

#### Step 8：慢速走一条分支

看官方示例 1：

```text
root = [1,2,3,4,5]
```

从底往上算：

- 节点 `4` 高度是 `1`
- 节点 `5` 高度是 `1`
- 节点 `2` 的左高度 `1`，右高度 `1`
- 所以经过节点 `2` 的候选直径是 `2`
- 节点 `2` 返回高度 `2`

再看根节点 `1`：

- 左高度 `2`
- 右高度 `1`

于是经过根节点的候选直径是 `3`，这就是答案。

#### Step 9：把规则压缩成一句能复用的话

这题的完整递推可以浓缩成一句：

> 后序遍历时，`dfs(node)` 返回高度；每到一个节点，用 `leftHeight + rightHeight` 更新全局直径。

一旦这句话稳了，543 基本就不会写乱。

### Assemble the Full Code

下面把刚才得到的规则拼成第一版完整代码。
这版代码可以直接运行。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_of_binary_tree(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0

        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    print(diameter_of_binary_tree(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0

        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return 0

            left = dfs(node.left)
            right = dfs(node.right)
            ans = max(ans, left + right)
            return 1 + max(left, right)

        dfs(root)
        return ans
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 树形 DP
- 后序递归
- 高度回传 + 全局最优更新

但真正要记住的不是名字，而是这个职责分离：

- **返回值**：只返回高度
- **全局变量**：只记录直径最大值

---

## E — Engineering（工程应用）

### 场景 1：组织树里的最长沟通链（Python）

**背景**：在组织结构里，你可能想知道两名员工之间最远的汇报跨度。  
**为什么适用**：这本质上就是树中两个节点之间的最长距离。

```python
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def longest_chain(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


root = Node("CEO", Node("VP1", Node("M1"), Node("M2")), Node("VP2"))
print(longest_chain(root))
```

### 场景 2：服务调用树里的最远传播跳数（Go）

**背景**：分析一次请求的调用树时，常要知道最远的传播链路。  
**为什么适用**：只要结构是树，最长路径就能用同样的“高度回传 + 全局更新”模式计算。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func diameter(root *Node) int {
	ans := 0
	var dfs func(*Node) int
	dfs = func(node *Node) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left+right > ans {
			ans = left + right
		}
		if left > right {
			return 1 + left
		}
		return 1 + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &Node{Name: "A", Left: &Node{Name: "B"}, Right: &Node{Name: "C"}}
	fmt.Println(diameter(root))
}
```

### 场景 3：前端组件树里的最远事件传播链（JavaScript）

**背景**：在复杂页面里，组件层级过深会让事件传播和状态同步链路变长。  
**为什么适用**：最长链路本质上就是组件树的直径。

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function diameter(root) {
  let ans = 0;
  function dfs(node) {
    if (!node) return 0;
    const left = dfs(node.left);
    const right = dfs(node.right);
    ans = Math.max(ans, left + right);
    return 1 + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new Node("App", new Node("Sidebar"), new Node("Content"));
console.log(diameter(root));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点只访问一次
- **空间复杂度**：`O(h)`，来自递归栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 后序高度回传 | `O(n)` | `O(h)` | 最经典、最推荐 |
| 每个节点都重新算左右高度 | 最坏 `O(n^2)` | `O(h)` | 重复计算严重，链状树最差 |
| 转无向图再做两次 BFS/DFS | `O(n)` | `O(n)` | 思路可行，但对这题过重 |

### 常见错误与注意事项

1. **把递归返回值写成直径**：父节点真正需要的是高度，不是当前子树直径。  
2. **忘记题目按边数计数**：这会导致答案多算或少算 `1`。  
3. **只检查经过根节点的路径**：最长路径可能完全在某棵子树里。  
4. **更新 `ans` 的时机放错**：必须在拿到左右高度之后，立刻尝试 `left + right`。

## 常见问题与注意事项

### 1. 为什么 `left + right` 就是边数？

因为这里的 `left` 和 `right` 表示从当前节点分别向左、向右最多能延伸出的单边长度。
把两边在当前节点拼起来，正好得到一条完整路径的边数。

### 2. 为什么不能直接返回 `max(leftDiameter, rightDiameter, left + right)`？

可以写成更复杂的结构体返回多个信息，但这题完全没必要。
只返回高度，再用一个外层变量维护全局直径，已经足够且更简洁。

### 3. 这题和 104 最大深度是什么关系？

104 只关心一条向下链的最长长度。  
543 要在“左右两边的高度”基础上再拼成一条跨节点路径。
所以 104 是 543 的基础，但 543 多了一层“全局最优更新”。

## 最佳实践与建议

- 写代码前先明确：`dfs` 返回的是高度，不是答案
- `ans` 和 `dfs` 返回值分工必须分清，别混成一个变量
- 手算一个 5 个节点的小树，检查“边数”和“节点数”是否混了
- 做完 543 后，顺手练 104 和 110，树形递推会更稳

## S — Summary（总结）

- 二叉树直径不是最大深度，也不是只看根到叶路径
- 这题最关键的设计是：递归返回高度，全局变量记录直径
- 每个节点都可以贡献一条候选直径 `left + right`
- 最长路径可能经过根，也可能完全在某棵子树内部
- 543 是理解树形 DP 和后序信息汇总的非常好的一题

## 参考与延伸阅读

- [LeetCode 543：二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)
- LeetCode 104：二叉树的最大深度
- LeetCode 110：平衡二叉树
- LeetCode 236：二叉树的最近公共祖先

## CTA

建议把 `104 + 543 + 110` 当成一组做。
它们都在训练“子树先返回什么、当前节点再怎么汇总”的树形递推能力，非常适合连续固化。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_of_binary_tree(root):
    ans = 0

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        ans = max(ans, left + right)
        return 1 + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    print(diameter_of_binary_tree(root))
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

int max_int(int a, int b) {
    return a > b ? a : b;
}

int dfs(struct TreeNode* node, int* ans) {
    if (node == NULL) return 0;

    int left = dfs(node->left, ans);
    int right = dfs(node->right, ans);
    if (left + right > *ans) *ans = left + right;
    return 1 + max_int(left, right);
}

int diameterOfBinaryTree(struct TreeNode* root) {
    int ans = 0;
    dfs(root, &ans);
    return ans;
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
    root->right = new_node(3);
    root->left->left = new_node(4);
    root->left->right = new_node(5);
    printf("%d\n", diameterOfBinaryTree(root));
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

int dfs(TreeNode* node, int& ans) {
    if (!node) return 0;
    int left = dfs(node->left, ans);
    int right = dfs(node->right, ans);
    ans = std::max(ans, left + right);
    return 1 + std::max(left, right);
}

int diameterOfBinaryTree(TreeNode* root) {
    int ans = 0;
    dfs(root, ans);
    return ans;
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
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    std::cout << diameterOfBinaryTree(root) << '\n';
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

func diameterOfBinaryTree(root *TreeNode) int {
	ans := 0

	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		left := dfs(node.Left)
		right := dfs(node.Right)
		if left+right > ans {
			ans = left + right
		}
		if left > right {
			return 1 + left
		}
		return 1 + right
	}

	dfs(root)
	return ans
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Left:  &TreeNode{Val: 4},
			Right: &TreeNode{Val: 5},
		},
		Right: &TreeNode{Val: 3},
	}
	fmt.Println(diameterOfBinaryTree(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn dfs(node: &Option<Box<TreeNode>>, ans: &mut i32) -> i32 {
    match node {
        None => 0,
        Some(n) => {
            let left = dfs(&n.left, ans);
            let right = dfs(&n.right, ans);
            *ans = (*ans).max(left + right);
            1 + left.max(right)
        }
    }
}

fn diameter_of_binary_tree(root: &Option<Box<TreeNode>>) -> i32 {
    let mut ans = 0;
    dfs(root, &mut ans);
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 5,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: None,
        })),
    }));

    println!("{}", diameter_of_binary_tree(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function diameterOfBinaryTree(root) {
  let ans = 0;

  function dfs(node) {
    if (!node) return 0;
    const left = dfs(node.left);
    const right = dfs(node.right);
    ans = Math.max(ans, left + right);
    return 1 + Math.max(left, right);
  }

  dfs(root);
  return ans;
}

const root = new TreeNode(1, new TreeNode(2, new TreeNode(4), new TreeNode(5)), new TreeNode(3));
console.log(diameterOfBinaryTree(root));
```
