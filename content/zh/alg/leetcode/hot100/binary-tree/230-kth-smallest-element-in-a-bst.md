---
title: "Hot100：二叉搜索树中第 K 小的元素（Kth Smallest Element in a BST）中序计数 / 提前停止 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "BST", "中序遍历", "栈", "LeetCode 230"]
description: "围绕 LeetCode 230 讲清为什么 BST 的中序遍历天然有序，以及如何用显式栈做到边遍历边计数、命中第 k 个就提前停止。"
keywords: ["Kth Smallest Element in a BST", "二叉搜索树中第 K 小的元素", "BST", "中序遍历", "显式栈", "LeetCode 230", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 230 的难点不是“会中序遍历”，而是把 BST 的有序性用成真正有用的信息。只要抓住“中序第 `k` 次访问到的节点就是答案”，整题就会变成一个非常稳定的计数问题。

- **预计阅读时长**：11~14 分钟
- **标签**：`Hot100`、`二叉树`、`BST`、`中序遍历`、`栈`
- **SEO 关键词**：Kth Smallest Element in a BST, 二叉搜索树中第 K 小的元素, BST, 中序遍历, 显式栈, LeetCode 230
- **元描述**：系统讲透 LeetCode 230 的 BST 中序有序性、显式栈计数与提前停止技巧，并给出工程迁移与多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一棵二叉搜索树 `root` 和一个整数 `k`，请找出这棵树中第 `k` 小的元素。

这里的 `k` 从 `1` 开始计数。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `root` | `TreeNode` | 二叉搜索树根节点 |
| `k` | `int` | 第几个最小元素，`1 <= k <= n` |
| 返回值 | `int` | 第 `k` 小的节点值 |

### 示例 1

```text
输入：root = [3,1,4,null,2], k = 1
输出：1
```

### 示例 2

```text
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```

### 提示

- 树中的节点数为 `n`
- `1 <= k <= n <= 10^4`
- `0 <= Node.val <= 10^4`

### 进阶

如果 BST 经常插入、删除，并且需要频繁查询第 `k` 小，你会怎么优化？

---

## 目标读者

- 已经知道 BST 中序有序，但还不会把这个性质变成查询代码的学习者
- 想把 94、98、230 这组 BST 基础题串起来的开发者
- 想理解为什么“遍历时提前停止”能省掉无意义访问的读者

## 背景 / 动机

这题非常适合训练一个关键转换：

- 从“树结构题”切换到“有序序列上的排名题”

很多人第一次看到第 `k` 小，会先想到：

- 先整棵树中序遍历成数组
- 排序
- 取第 `k - 1` 个位置

虽然能做，但不够贴题。
BST 已经免费给了我们顺序结构，中序遍历时节点就是按从小到大出现的。

所以问题真正应该变成：

> 怎样在中序遍历的过程中，一边访问一边计数，并在第 `k` 个节点处立即停下？

## 核心概念

- **BST 中序有序性**：左、根、右访问顺序会得到严格递增序列
- **显式栈**：手动模拟递归调用栈，精确控制访问顺序
- **访问计数**：每弹出一个节点，就意味着序列中出现了下一个更大的元素
- **提前停止**：一旦计数达到 `k`，后续节点无需再看

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先把题目翻译成一个更直接的问题

看最小例子：

```text
    2
   / \
  1   3

k = 2
```

这棵 BST 的中序结果是：

```text
[1,2,3]
```

第 `2` 小就是中序序列里的第 `2` 个元素，也就是 `2`。

所以整题其实不是“神秘的 BST 技巧题”，而是：

> 找 BST 中序遍历时第 `k` 次访问到的节点。

#### Step 2：为了按顺序访问，我们最少需要什么状态？

如果用递归，中序顺序是：

- 先一路走左边
- 回来访问当前节点
- 再转向右边

如果想自己控制这个过程，就需要：

- 一个 `stack` 保存还没来得及访问的祖先
- 一个 `cur` 指针一路往左走

#### Step 3：更小的子问题是什么？

当前不是在“求整个答案”，而是在重复做一件更小的事：

> 找到当前还没访问过的最小节点。

怎么找？

- 只要 `cur` 不为空，就持续入栈并往左走
- 左边走到底时，栈顶就是下一个最小值

#### Step 4：什么时候说明我们已经完成了目标？

每弹出一个节点，就代表：

- 中序序列里又出现了一个新元素
- 计数器应该加 `1`

当：

```python
count == k
```

就可以直接返回当前节点值。

#### Step 5：中序遍历里“下一步会发生什么”？

标准流程是：

```python
while cur is not None:
    stack.append(cur)
    cur = cur.left
```

这一步做完之后，最左侧还没访问的节点会来到栈顶。

#### Step 6：真正计数应该放在哪个时机？

不是入栈时，也不是走向右子树时，而是：

```python
cur = stack.pop()
count += 1
```

因为只有弹栈这一刻，当前节点才真正按照中序顺序被访问到。

#### Step 7：访问完一个节点后，后续信息怎么延续？

一旦访问完 `cur`，中序遍历下一步一定是它的右子树：

```python
cur = cur.right
```

然后重复“继续向左压栈”的流程。

这就是整题最稳定的循环骨架。

#### Step 8：慢速走一遍官方示例

看：

```text
root = [3,1,4,null,2], k = 1
```

1. 从 `3` 出发，一路往左，把 `3`、`1` 压栈  
2. `1` 没有左子树，弹出它  
3. 这是第 `1` 次访问，所以答案立刻就是 `1`

你会发现，我们甚至不需要把后面的 `2` 和 `4` 全看完。
这就是提前停止的价值。

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- 中序顺序
- `stack + cur`
- 弹栈时计数
- 命中 `k` 时返回

现在只差把这些片段合成一个完整循环。

### Assemble the Full Code

先给一版可以直接运行的 Python 示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(root, k):
    stack = []
    cur = root
    count = 0

    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left

        cur = stack.pop()
        count += 1
        if count == k:
            return cur.val
        cur = cur.right

    raise ValueError("k is out of range")


if __name__ == "__main__":
    root = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))
    print(kth_smallest(root, 1))
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
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root

        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right

        raise ValueError("invalid input")
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- BST 中序计数
- 显式栈中序遍历
- 排名查询

但真正要记住的是这句话：

> BST 的第 `k` 小，就是中序遍历时第 `k` 次访问到的节点。

---

## E — Engineering（工程应用）

### 场景 1：有序配置树里查第 k 个阈值（Python）

**背景**：一些配置系统会把阈值按 BST 结构保存在内存中，运维需要查询第 `k` 个阈值。  
**为什么适用**：BST 中序天然有序，不必额外排序。

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth(root, k):
    stack = []
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right


root = Node(20, Node(10), Node(30))
print(kth(root, 2))
```

### 场景 2：读多写少索引里做排名查询（Go）

**背景**：一棵几乎不更新的索引树，需要按排名查询某个位置的键值。  
**为什么适用**：中序计数就相当于顺着有序流走到第 `k` 个元素。

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func kth(root *Node, k int) int {
	stack := []*Node{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return cur.Val
		}
		cur = cur.Right
	}
	return -1
}

func main() {
	root := &Node{Val: 2, Left: &Node{Val: 1}, Right: &Node{Val: 3}}
	fmt.Println(kth(root, 3))
}
```

### 场景 3：前端排序树里按名次取展示节点（JavaScript）

**背景**：页面上维护一棵按优先级组织的搜索树，需要按序号抽取第 `k` 个节点展示。  
**为什么适用**：不必先拍平成数组，只要按中序走到目标位置即可。

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function kth(root, k) {
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    k -= 1;
    if (k === 0) return cur.val;
    cur = cur.right;
  }
  return null;
}

const root = new Node(4, new Node(2, new Node(1), new Node(3)), new Node(6));
console.log(kth(root, 4));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：最坏 `O(h + k)` 到 `O(n)`  
  如果很早就命中第 `k` 个元素，可以提前停止；最坏仍可能访问整棵树
- **空间复杂度**：`O(h)`，来自显式栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 中序遍历 + 提前停止 | 最坏 `O(n)` | `O(h)` | 最直接，最适合单次查询 |
| 中序拍平成数组 | `O(n)` | `O(n)` | 好写，但做了不必要存储 |
| 维护子树大小字段 | 查询 `O(h)` | 视实现而定 | 适合频繁更新 + 高频排名查询 |

### 常见错误与注意事项

1. **把 BST 当普通二叉树**：这会错过中序有序这个最值钱的信息。  
2. **入栈时就计数**：真正被访问是在弹栈时。  
3. **忘记提前停止**：会继续无意义遍历右侧更大的节点。  
4. **`k` 从 1 开始而不是 0 开始**：这是最常见的 off-by-one 错误。

## 常见问题与注意事项

### 1. 为什么这题和 94 中序遍历几乎是一回事？

因为 BST 的排名就是中序序列的位置。
94 负责“怎么按顺序访问”，230 负责“在访问过程中做计数并提前停下”。

### 2. 进阶里说频繁修改怎么办？

如果查询很多、更新也很多，更常见的优化是：

- 给每个节点维护子树大小
- 查第 `k` 小时像二分一样沿树往下走

这样单次查询可以做到 `O(h)`。

### 3. 这题能用递归写吗？

当然可以。
只要维护：

- 访问计数 `count`
- 命中答案 `ans`

但显式栈版本更贴近 94 的迭代模板，也更容易直接提前停止。

## 最佳实践与建议

- 做 230 前先把 94 和 98 写顺，BST 线会明显更清楚
- 一旦看到“BST + 第 k 小”，先想到中序遍历
- 计数要放在弹栈访问时，而不是压栈时
- 如果题目追问高频查询，再考虑子树大小增强版

## S — Summary（总结）

- 230 的核心不是树，而是把 BST 中序有序性转成排名查询
- 第 `k` 小元素就是中序遍历时第 `k` 次访问到的节点
- 显式栈版本最稳定，因为它能精确控制访问顺序和提前停止
- 单次查询时，中序计数通常已经足够好，不需要更重的数据结构
- 230 和 94、98 组成了一条很完整的 BST 基础训练线

## 参考与延伸阅读

- [LeetCode 230：二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)
- LeetCode 94：二叉树的中序遍历
- LeetCode 98：验证二叉搜索树
- LeetCode 173：二叉搜索树迭代器

## CTA

建议把 `94 + 98 + 230` 连着做。
94 练遍历顺序，98 练 BST 不变量，230 练“利用有序性做查询”，三题放在一起非常适合固化 BST 基本功。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(root, k):
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right


if __name__ == "__main__":
    root = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4)), TreeNode(6))
    print(kth_smallest(root, 3))
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

int kthSmallest(struct TreeNode* root, int k) {
    struct TreeNode* stack[10016];
    int top = 0;
    struct TreeNode* cur = root;

    while (cur != NULL || top > 0) {
        while (cur != NULL) {
            stack[top++] = cur;
            cur = cur->left;
        }
        cur = stack[--top];
        k--;
        if (k == 0) return cur->val;
        cur = cur->right;
    }
    return -1;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(5);
    root->left = new_node(3);
    root->right = new_node(6);
    root->left->left = new_node(2);
    root->left->right = new_node(4);
    root->left->left->left = new_node(1);
    printf("%d\n", kthSmallest(root, 3));
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <stack>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int kthSmallest(TreeNode* root, int k) {
    std::stack<TreeNode*> st;
    TreeNode* cur = root;
    while (cur || !st.empty()) {
        while (cur) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        if (--k == 0) return cur->val;
        cur = cur->right;
    }
    return -1;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(5);
    root->left = new TreeNode(3);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(2);
    root->left->right = new TreeNode(4);
    root->left->left->left = new TreeNode(1);
    std::cout << kthSmallest(root, 3) << '\n';
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

func kthSmallest(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return cur.Val
		}
		cur = cur.Right
	}
	return -1
}

func main() {
	root := &TreeNode{
		Val: 5,
		Left: &TreeNode{
			Val: 3,
			Left: &TreeNode{
				Val:  2,
				Left: &TreeNode{Val: 1},
			},
			Right: &TreeNode{Val: 4},
		},
		Right: &TreeNode{Val: 6},
	}
	fmt.Println(kthSmallest(root, 3))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn kth_smallest(root: &Option<Box<TreeNode>>, mut k: i32) -> i32 {
    let mut stack: Vec<&TreeNode> = Vec::new();
    let mut cur = root.as_deref();

    while cur.is_some() || !stack.is_empty() {
        while let Some(node) = cur {
            stack.push(node);
            cur = node.left.as_deref();
        }
        let node = stack.pop().unwrap();
        k -= 1;
        if k == 0 {
            return node.val;
        }
        cur = node.right.as_deref();
    }
    -1
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 5,
        left: Some(Box::new(TreeNode {
            val: 3,
            left: Some(Box::new(TreeNode {
                val: 2,
                left: Some(Box::new(TreeNode {
                    val: 1,
                    left: None,
                    right: None,
                })),
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 6,
            left: None,
            right: None,
        })),
    }));

    println!("{}", kth_smallest(&root, 3));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function kthSmallest(root, k) {
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    k -= 1;
    if (k === 0) return cur.val;
    cur = cur.right;
  }
  return null;
}

const root = new TreeNode(
  5,
  new TreeNode(3, new TreeNode(2, new TreeNode(1)), new TreeNode(4)),
  new TreeNode(6)
);
console.log(kthSmallest(root, 3));
```
