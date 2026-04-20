---
title: "Hot100：二叉树中的最大路径和（Binary Tree Maximum Path Sum）树形 DP / 单边贡献 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "树形DP", "DFS", "后序遍历", "LeetCode 124"]
description: "围绕 LeetCode 124 讲清最大路径和的关键设计：递归向上只返回单边贡献，而经过当前节点的完整路径用来更新全局答案。"
keywords: ["Binary Tree Maximum Path Sum", "二叉树中的最大路径和", "树形DP", "单边贡献", "后序遍历", "LeetCode 124", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 124 最容易混淆的地方是：递归到底该返回“整条最大路径”还是“能继续向上接的那一段贡献”。只要把这两个角色分开，这题就会变成一个非常典型的树形 DP。

- **预计阅读时长**：12~15 分钟
- **标签**：`Hot100`、`二叉树`、`树形DP`、`DFS`、`后序遍历`
- **SEO 关键词**：Binary Tree Maximum Path Sum, 二叉树中的最大路径和, 树形DP, 单边贡献, 后序遍历, LeetCode 124
- **元描述**：系统讲透 LeetCode 124 的单边贡献返回值、全局最大路径更新、负贡献剪枝与多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

二叉树中的一条路径定义为：

- 由若干个节点组成
- 相邻节点之间必须有边相连
- 同一个节点在一条路径里最多出现一次
- 路径至少包含一个节点
- 路径不一定经过根节点

路径和就是路径上所有节点值之和。  
题目要求返回整棵树里的最大路径和。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `root` | `TreeNode` | 二叉树根节点 |
| 返回值 | `int` | 最大路径和 |

### 示例 1

```text
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3，路径和为 6。
```

### 示例 2

```text
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7，路径和为 42。
```

### 提示

- 树中节点数目范围是 `[1, 3 * 10^4]`
- `-1000 <= Node.val <= 1000`

---

## 目标读者

- 已经做过 543，但还没完全理解“树上路径题”返回值该如何设计的学习者
- 一写最大路径和就会把“经过当前节点的完整路径”和“向上返回的贡献”混掉的开发者
- 想系统掌握树形 DP 基本套路的读者

## 背景 / 动机

这题的关键训练点是：

- 一个递归函数不一定返回“全局答案”，它也可以只返回“父节点真正需要的信息”

很多人第一次做 124，会卡在两个问题上：

- 最长路径可能同时经过左右子树，那返回值怎么表达？
- 如果某个子树和是负数，要不要把它接进来？

这两个问题如果不拆开，代码就很容易混乱。

真正稳定的写法是把角色分成两类：

- **全局答案**：记录整棵树里目前出现过的最大路径和
- **递归返回值**：只返回“当前节点向上最多能提供多少单边贡献”

## 核心概念

- **后序递归**：先拿左右子树结果，再决定当前节点信息
- **单边贡献**：当前节点能向父节点延伸出去的最好一条链
- **完整路径候选**：在当前节点处把左右两边都接上形成的一条路径
- **负贡献剪枝**：如果某一边是负收益，宁可不要

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先弄清“答案”长什么样

看最简单的例子：

```text
    1
   / \
  2   3
```

最大路径和不是单条向下链，而是：

```text
2 -> 1 -> 3
```

也就是说，某个节点可以成为“拐点”，把左右两边都接起来。

这和 104、543 那类只看单边高度的题不一样。

#### Step 2：父节点到底需要子节点返回什么？

虽然当前节点处的最大路径可能同时用到左右两边，但父节点继续往上接时，只能选一边。

因为一条路径一旦往上延伸，不能在父节点处同时带着两条分叉继续走。

所以递归返回值应该定义成：

> 以当前节点为起点，向下延伸时，能提供给父节点的最大单边贡献。

#### Step 3：更小的子问题是什么？

如果左右子树都已经告诉我们各自的最大单边贡献，那么当前节点就能计算出：

- 经过当前节点的完整路径值
- 当前节点向上返回的单边贡献值

这正是典型后序递归。

#### Step 4：什么时候递归可以直接结束？

空节点对路径没有正贡献，所以可以返回：

```python
if node is None:
    return 0
```

这里返回的是“贡献值”，不是答案。

#### Step 5：当前节点处的完整路径候选怎么算？

先拿左右贡献：

```python
left = dfs(node.left)
right = dfs(node.right)
```

如果某一边是负数，接上去只会让路径更差，所以可以直接舍弃：

```python
left = max(left, 0)
right = max(right, 0)
```

于是经过当前节点的完整路径候选就是：

```python
candidate = node.val + left + right
```

#### Step 6：全局答案什么时候更新？

只要你拿到了这个 `candidate`，就应该立刻尝试更新全局最大值：

```python
ans = max(ans, candidate)
```

因为这条路径可能是整棵树目前最好的答案。

#### Step 7：为什么递归向上只能返回一边？

父节点如果要继续把当前节点接进更长路径，它只能从当前节点往下选一条分支。

所以返回值必须是：

```python
return node.val + max(left, right)
```

这里的 `left` 和 `right` 已经经过了 `max(., 0)` 处理。

#### Step 8：慢速走一遍官方示例

看：

```text
root = [-10,9,20,null,null,15,7]
```

到节点 `20` 时：

- 左贡献是 `15`
- 右贡献是 `7`

所以经过 `20` 的完整路径候选是：

```text
20 + 15 + 7 = 42
```

这会更新全局答案为 `42`。

但 `20` 向上返回给 `-10` 的，只能是一条单边链：

```text
20 + max(15, 7) = 35
```

这就是“完整路径”和“返回贡献”必须分开的原因。

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- `dfs` 返回单边贡献
- `candidate` 用来更新全局答案
- 负贡献直接丢弃

现在只差把它们装成完整函数。

### Assemble the Full Code

先给一版可直接运行的 Python 示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0

        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)

        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_path_sum(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以写成下面这样：

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = float("-inf")

        def dfs(node: Optional[TreeNode]) -> int:
            nonlocal ans
            if node is None:
                return 0

            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)

            ans = max(ans, node.val + left + right)
            return node.val + max(left, right)

        dfs(root)
        return ans
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 树形 DP
- 后序贡献回传
- 最大路径和分治

但真正要记住的是这句：

> 当前节点处的完整答案可以同时接左右两边，但向父节点返回时只能带一边上去。

---

## E — Engineering（工程应用）

### 场景 1：收益依赖树里找最大收益链路（Python）

**背景**：一棵依赖树上，每个节点有正负收益，想找总收益最大的那条连接路径。  
**为什么适用**：负收益分支应该被主动丢弃，这和本题完全同构。

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def best_path(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


root = Node(-5, Node(10), Node(20))
print(best_path(root))
```

### 场景 2：服务调用树里寻找最高价值传播路径（Go）

**背景**：一棵调用树上的每个节点代表净收益或净耗损，需要找最大总价值链路。  
**为什么适用**：负值分支应被裁掉，而局部拐点可能形成全局最优。

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func maxPath(root *Node) int {
	if root == nil {
		return 0
	}
	ans := root.Val
	var dfs func(*Node) int
	dfs = func(node *Node) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		if left < 0 {
			left = 0
		}
		right := dfs(node.Right)
		if right < 0 {
			right = 0
		}
		sum := node.Val + left + right
		if sum > ans {
			ans = sum
		}
		if left > right {
			return node.Val + left
		}
		return node.Val + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &Node{Val: -10, Left: &Node{Val: 9}, Right: &Node{Val: 20, Left: &Node{Val: 15}, Right: &Node{Val: 7}}}
	fmt.Println(maxPath(root))
}
```

### 场景 3：前端技能树里找最大分数路径（JavaScript）

**背景**：游戏或可视化系统里，节点有正负得分，想找总分最高的一条相连路径。  
**为什么适用**：同样是“局部贡献上行 + 全局答案单独更新”的结构。

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function maxPath(root) {
  let ans = -Infinity;
  function dfs(node) {
    if (!node) return 0;
    const left = Math.max(dfs(node.left), 0);
    const right = Math.max(dfs(node.right), 0);
    ans = Math.max(ans, node.val + left + right);
    return node.val + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new Node(-10, new Node(9), new Node(20, new Node(15), new Node(7)));
console.log(maxPath(root));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点访问一次
- **空间复杂度**：`O(h)`，来自递归栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 后序单边贡献 + 全局更新 | `O(n)` | `O(h)` | 最推荐 |
| 枚举所有路径 | 极高 | 极高 | 几乎不可用 |
| 把返回值当成完整路径和 | 错误 | - | 父节点无法继续正确拼接 |

### 常见错误与注意事项

1. **把递归返回值写成完整路径和**：父节点根本没法继续往上接。  
2. **忘记丢弃负贡献**：这样会把答案越加越小。  
3. **把全局答案初值设成 `0`**：如果整棵树全是负数，会得到错误结果。  
4. **和 543 混淆**：543 按边数看直径，124 按节点值和看收益，且要处理负数。

## 常见问题与注意事项

### 1. 为什么空节点返回 `0` 不会错？

因为这里返回的是“贡献值”。
对于父节点来说，空节点没有可用贡献，返回 `0` 正合适。

### 2. 为什么全局答案不能初始化为 `0`？

因为树可能全是负数。
例如只有一个节点 `-3` 时，答案必须是 `-3`，不是 `0`。

### 3. 这题和 543 的核心区别是什么？

两题都用后序递归，但：

- 543 返回高度，更新的是边数直径
- 124 返回单边贡献，更新的是节点值路径和

124 还多了一个关键动作：负贡献剪枝。

## 最佳实践与建议

- 先问自己：父节点真正需要我返回什么，而不是一上来就求全局答案
- 区分“当前节点处的完整候选路径”和“可向上延伸的单边贡献”
- 看到负权时，优先考虑是否应该把负贡献截断
- 做完 124 后，回头再看 543，会更容易理解树形 DP 的分工

## S — Summary（总结）

- 124 的核心设计是：递归向上只返回单边贡献，全局答案单独更新
- 经过当前节点的完整路径可以接左右两边，但向父节点只能带一边
- 负贡献不值得保留，所以要用 `max(gain, 0)` 裁掉
- 这题和 543 形式相似，但 124 还要处理节点权值和负数情况
- 一旦吃透 124，很多树上路径题都会变得更清楚

## 参考与延伸阅读

- [LeetCode 124：二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)
- LeetCode 543：二叉树的直径
- LeetCode 104：二叉树的最大深度
- LeetCode 337：打家劫舍 III

## CTA

建议把 `104 + 543 + 124` 一起练。
104 练单边高度，543 练结构路径，124 练带权路径，这三题连起来非常适合建立树形 DP 的完整直觉。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans
        if node is None:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        ans = max(ans, node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return ans


if __name__ == "__main__":
    root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    print(max_path_sum(root))
```

```c
#include <limits.h>
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
    if (!node) return 0;
    int left = max_int(dfs(node->left, ans), 0);
    int right = max_int(dfs(node->right, ans), 0);
    int candidate = node->val + left + right;
    if (candidate > *ans) *ans = candidate;
    return node->val + max_int(left, right);
}

int maxPathSum(struct TreeNode* root) {
    int ans = INT_MIN;
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
    struct TreeNode* root = new_node(-10);
    root->left = new_node(9);
    root->right = new_node(20);
    root->right->left = new_node(15);
    root->right->right = new_node(7);
    printf("%d\n", maxPathSum(root));
    free_tree(root);
    return 0;
}
```

```cpp
#include <algorithm>
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int dfs(TreeNode* node, int& ans) {
    if (!node) return 0;
    int left = std::max(dfs(node->left, ans), 0);
    int right = std::max(dfs(node->right, ans), 0);
    ans = std::max(ans, node->val + left + right);
    return node->val + std::max(left, right);
}

int maxPathSum(TreeNode* root) {
    int ans = INT_MIN;
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
    TreeNode* root = new TreeNode(-10);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);
    std::cout << maxPathSum(root) << '\n';
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

func maxPathSum(root *TreeNode) int {
	if root == nil {
		return 0
	}
	ans := root.Val
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		if left < 0 {
			left = 0
		}
		right := dfs(node.Right)
		if right < 0 {
			right = 0
		}
		if node.Val+left+right > ans {
			ans = node.Val + left + right
		}
		if left > right {
			return node.Val + left
		}
		return node.Val + right
	}
	dfs(root)
	return ans
}

func main() {
	root := &TreeNode{
		Val:  -10,
		Left: &TreeNode{Val: 9},
		Right: &TreeNode{
			Val:   20,
			Left:  &TreeNode{Val: 15},
			Right: &TreeNode{Val: 7},
		},
	}
	fmt.Println(maxPathSum(root))
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
            let left = dfs(&n.left, ans).max(0);
            let right = dfs(&n.right, ans).max(0);
            *ans = (*ans).max(n.val + left + right);
            n.val + left.max(right)
        }
    }
}

fn max_path_sum(root: &Option<Box<TreeNode>>) -> i32 {
    let mut ans = i32::MIN;
    dfs(root, &mut ans);
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: -10,
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

    println!("{}", max_path_sum(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function maxPathSum(root) {
  let ans = -Infinity;
  function dfs(node) {
    if (!node) return 0;
    const left = Math.max(dfs(node.left), 0);
    const right = Math.max(dfs(node.right), 0);
    ans = Math.max(ans, node.val + left + right);
    return node.val + Math.max(left, right);
  }
  dfs(root);
  return ans;
}

const root = new TreeNode(-10, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
console.log(maxPathSum(root));
```
