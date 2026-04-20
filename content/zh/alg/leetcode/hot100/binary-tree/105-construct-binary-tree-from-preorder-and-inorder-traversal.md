---
title: "Hot100：从前序与中序遍历序列构造二叉树（Construct Binary Tree from Preorder and Inorder Traversal）索引分治 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "分治", "哈希表", "前序遍历", "LeetCode 105"]
description: "围绕 LeetCode 105 讲清前序负责定根、中序负责切左右的分工，以及如何用索引区间和哈希表把构树复杂度降到 O(n)。"
keywords: ["Construct Binary Tree from Preorder and Inorder Traversal", "从前序与中序遍历序列构造二叉树", "前序遍历", "中序遍历", "分治", "LeetCode 105", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 105 的关键不是死记“前序 + 中序能构树”，而是先看懂两种遍历各自提供什么信息。前序负责告诉你谁是根，中序负责告诉你左右边界，组合起来就能自然落成一个哈希定位的区间分治。

- **预计阅读时长**：12~15 分钟
- **标签**：`Hot100`、`二叉树`、`分治`、`哈希表`、`前序遍历`
- **SEO 关键词**：Construct Binary Tree from Preorder and Inorder Traversal, 从前序与中序遍历序列构造二叉树, 前序遍历, 中序遍历, 分治, LeetCode 105
- **元描述**：系统讲透 LeetCode 105 的构树推导、索引分治、哈希优化与多语言实现，并解释为什么一定要先建左子树。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一棵二叉树的前序遍历 `preorder` 和中序遍历 `inorder`，请你重建这棵二叉树并返回它的根节点。

题目保证：

- 树中元素没有重复
- 给出的两个数组一定来自同一棵合法二叉树

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `preorder` | `int[]` | 前序遍历结果 |
| `inorder` | `int[]` | 中序遍历结果 |
| 返回值 | `TreeNode` | 重建后的二叉树根节点 |

### 示例 1

```text
输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出：[3,9,20,null,null,15,7]
```

### 示例 2

```text
输入：preorder = [-1], inorder = [-1]
输出：[-1]
```

### 提示

- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` 和 `inorder` 都没有重复元素
- `inorder` 中的每个值都出现在 `preorder` 中
- `preorder` 保证是某棵树的前序遍历
- `inorder` 保证是同一棵树的中序遍历

---

## 目标读者

- 一看到“根据遍历结果构树”就会混淆前序和中序职责的学习者
- 想把“树的遍历顺序”和“树的结构恢复”建立稳定联系的开发者
- 做过 94、114 之后，想继续巩固树结构题的读者

## 背景 / 动机

这题最值得学的地方是：

- 不同遍历顺序到底提供了哪一类结构信息

很多人第一次做 105，会记住一个口号：

- 前序 + 中序可以构树

但如果不继续追问“为什么”，代码很容易写得很机械。

真正该想清的是：

- 前序里第一个值为什么一定是根？
- 中序里根的位置为什么刚好能把左右子树切开？
- 为什么递归时必须先建左子树，再建右子树？

一旦这三点稳定，这题就很顺。

## 核心概念

- **前序遍历**：根、左、右，所以第一个元素总是当前子树的根
- **中序遍历**：左、根、右，所以根的位置能切出左右子树范围
- **区间分治**：用 `inorder` 的索引区间表示当前子树
- **哈希表定位**：用 `value -> inorderIndex` 快速找到切分点

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先从最关键的两条遍历性质出发

看示例：

```text
preorder = [3,9,20,15,7]
inorder  = [9,3,15,20,7]
```

前序第一个元素是 `3`。
因为前序顺序是“根、左、右”，所以：

> `3` 一定是整棵树的根。

接着去中序里找 `3`：

```text
[9, 3, 15, 20, 7]
```

根左边 `[9]` 是左子树，根右边 `[15,20,7]` 是右子树。

这就是整题最核心的分工。

#### Step 2：递归最少需要什么状态？

如果每次都真的切片，代码虽然能写，但会多做很多复制。

更稳定的状态是：

- `pre_idx`：当前该从 `preorder` 读哪个根
- `l, r`：当前子树在 `inorder` 里的区间边界

所以子问题可以写成：

> 用 `preorder[pre_idx...]` 和 `inorder[l..r]` 构造当前子树。

#### Step 3：更小的子问题到底是什么？

一旦当前根节点确定了，问题自然分成两个同类子问题：

- 用中序左半段构造左子树
- 用中序右半段构造右子树

所以递归框架天然成立：

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 4：什么时候这个区间已经没树可建了？

如果当前 `inorder` 区间为空：

```python
if l > r:
    return None
```

说明当前子树不存在，这就是 base case。

#### Step 5：当前根节点该怎么取？

因为前序遍历总是先给根，所以：

```python
root_val = preorder[pre_idx]
pre_idx += 1
```

这一步表示：

- 当前子树的根已经确定
- 下一个前序位置将用于更深层子树

#### Step 6：中序里的切分点怎么快速找到？

根值一旦知道，就要在 `inorder` 里找到它的位置：

```python
mid = index[root_val]
```

为了避免每次线性搜索，最好提前建一个哈希表：

```python
index = {value: i for i, value in enumerate(inorder)}
```

这样每次定位根的位置都是 `O(1)`。

#### Step 7：为什么必须先递归左子树，再递归右子树？

因为前序顺序是：

```text
根 -> 左 -> 右
```

当前根读完后，`preorder` 下一个值一定属于左子树，而不是右子树。

所以必须：

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

如果顺序写反，`pre_idx` 就会把右子树错误地吃成左子树。

#### Step 8：慢速走一遍官方示例

看：

```text
preorder = [3,9,20,15,7]
inorder  = [9,3,15,20,7]
```

1. `preorder[0] = 3`，根是 `3`
2. `3` 在中序里的位置是 `1`
3. 左区间是 `[9]`，右区间是 `[15,20,7]`
4. 接下来 `preorder[1] = 9`，自然成为左子树根
5. 左子树建完后，再消费 `20,15,7` 去建右子树

整个过程里，前序负责“取根”，中序负责“切边界”。

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- `pre_idx`
- `inorder` 区间
- 哈希表定位
- 左右递归顺序

现在只差把这些片段装进一个完整递归。

### Assemble the Full Code

先给一版可直接运行的 Python 示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(preorder, inorder):
    index = {value: i for i, value in enumerate(inorder)}
    pre_idx = 0

    def build(l, r):
        nonlocal pre_idx
        if l > r:
            return None

        root_val = preorder[pre_idx]
        pre_idx += 1
        root = TreeNode(root_val)

        mid = index[root_val]
        root.left = build(l, mid - 1)
        root.right = build(mid + 1, r)
        return root

    return build(0, len(inorder) - 1)


def preorder_traversal(root):
    if root is None:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)


if __name__ == "__main__":
    root = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    print(preorder_traversal(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以写成下面这样：

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        index = {value: i for i, value in enumerate(inorder)}
        pre_idx = 0

        def build(l: int, r: int) -> Optional[TreeNode]:
            nonlocal pre_idx
            if l > r:
                return None

            root_val = preorder[pre_idx]
            pre_idx += 1
            root = TreeNode(root_val)

            mid = index[root_val]
            root.left = build(l, mid - 1)
            root.right = build(mid + 1, r)
            return root

        return build(0, len(inorder) - 1)
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 遍历序列构树
- 哈希定位分治
- 区间递归建树

但更重要的是这个分工：

> 前序负责定根，中序负责切左右。

---

## E — Engineering（工程应用）

### 场景 1：根据根优先快照恢复树结构（Python）

**背景**：某些系统会分别保存“根优先访问顺序”和“中序布局顺序”，之后需要恢复原树。  
**为什么适用**：一份序列给根，一份序列给边界，正好可以重建结构。

```python
def rebuild(pre, ino):
    pos = {v: i for i, v in enumerate(ino)}
    idx = 0

    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def dfs(l, r):
        nonlocal idx
        if l > r:
            return None
        val = pre[idx]
        idx += 1
        m = pos[val]
        return Node(val, dfs(l, m - 1), dfs(m + 1, r))

    return dfs(0, len(ino) - 1)


root = rebuild([2, 1, 3], [1, 2, 3])
print(root.val)
```

### 场景 2：表达式树从遍历日志恢复（Go）

**背景**：编译或解释流程里，有时会把表达式树的多种遍历结果落盘，用于后续恢复。  
**为什么适用**：前序给运算根顺序，中序给操作数左右边界。

```go
package main

import "fmt"

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func buildTree(pre, ino []int) *Node {
	pos := map[int]int{}
	for i, v := range ino {
		pos[v] = i
	}
	idx := 0
	var dfs func(int, int) *Node
	dfs = func(l, r int) *Node {
		if l > r {
			return nil
		}
		val := pre[idx]
		idx++
		m := pos[val]
		root := &Node{Val: val}
		root.Left = dfs(l, m-1)
		root.Right = dfs(m+1, r)
		return root
	}
	return dfs(0, len(ino)-1)
}

func main() {
	root := buildTree([]int{2, 1, 3}, []int{1, 2, 3})
	fmt.Println(root.Val)
}
```

### 场景 3：前端恢复树形组件层级（JavaScript）

**背景**：有些低代码系统会保存组件树的遍历结果，重新打开页面时再还原层级。  
**为什么适用**：只要节点值唯一，这套构树逻辑就能直接复用。

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildTree(preorder, inorder) {
  const pos = new Map();
  inorder.forEach((v, i) => pos.set(v, i));
  let idx = 0;

  function dfs(l, r) {
    if (l > r) return null;
    const val = preorder[idx++];
    const m = pos.get(val);
    return new Node(val, dfs(l, m - 1), dfs(m + 1, r));
  }

  return dfs(0, inorder.length - 1);
}

const root = buildTree([2, 1, 3], [1, 2, 3]);
console.log(root.val);
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`  
  每个节点创建一次，哈希表定位根位置是 `O(1)`
- **空间复杂度**：`O(n)`  
  来自哈希表和递归栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 哈希表 + 区间分治 | `O(n)` | `O(n)` | 最推荐 |
| 每次在线性扫描中序找根 | 最坏 `O(n^2)` | `O(h)` | 链状树时会退化 |
| 递归里不断切片 | `O(n)` 到 `O(n^2)` | 更高 | 容易写，但复制成本大 |

### 常见错误与注意事项

1. **左右子树递归顺序写反**：会把 `preorder` 的消费顺序彻底打乱。  
2. **没有哈希表**：每层都去中序里线性找根，复杂度容易退化。  
3. **题目要求唯一值却没意识到其必要性**：如果有重复值，中序切分点会变得不唯一。  
4. **把区间边界和 `pre_idx` 混在一起**：这题最稳定的写法是让二者职责分离。

## 常见问题与注意事项

### 1. 为什么只用前序不能唯一构树？

因为前序只告诉你：

- 谁先出现
- 谁是根

但不能告诉你左右子树各有多大。  
中序里的根位置才提供了切分边界。

### 2. 为什么只用中序也不行？

中序只能告诉你左、根、右的相对位置，但不能告诉你哪个节点先当根，所以信息不够。

### 3. 后序 + 中序也可以构树吗？

可以。  
区别只是：

- 后序最后一个元素是根
- 前序第一个元素是根

核心思想不变，都是“一种遍历定根，一种遍历切边界”。

## 最佳实践与建议

- 先给前序和中序各写一句“它提供什么信息”
- 用 `inorder` 区间表示当前子树，比切片稳定得多
- `pre_idx` 只负责按前序顺序读根，别让它承担别的语义
- 做完 105 后，顺手练 106，前序版和后序版会形成对照记忆

## S — Summary（总结）

- 105 的核心不是背答案，而是明确前序与中序的职责分工
- 前序的第一个元素总是当前子树根，中序里的根位置负责切出左右区间
- 哈希表让根位置查找从 `O(n)` 降到 `O(1)`，整题复杂度稳定在 `O(n)`
- 递归时一定要先建左子树，再建右子树，因为前序消费顺序就是根、左、右
- 这题是非常典型的“遍历顺序和树结构互相转化”训练题

## 参考与延伸阅读

- [LeetCode 105：从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- LeetCode 106：从中序与后序遍历序列构造二叉树
- LeetCode 114：二叉树展开为链表
- LeetCode 94：二叉树的中序遍历

## CTA

建议把 `94 + 105 + 106 + 114` 连着做。
94 让你熟悉遍历顺序，105/106 让你用顺序重建结构，114 则让你把结构再变回顺序，这组题非常适合系统建立“遍历和结构互相转换”的直觉。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(preorder, inorder):
    pos = {v: i for i, v in enumerate(inorder)}
    idx = 0

    def build(l, r):
        nonlocal idx
        if l > r:
            return None
        val = preorder[idx]
        idx += 1
        root = TreeNode(val)
        m = pos[val]
        root.left = build(l, m - 1)
        root.right = build(m + 1, r)
        return root

    return build(0, len(inorder) - 1)


def preorder_traversal(root):
    if root is None:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)


if __name__ == "__main__":
    root = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    print(preorder_traversal(root))
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

struct TreeNode* build(int* preorder, int* pos, int* idx, int l, int r) {
    if (l > r) return NULL;
    int val = preorder[(*idx)++];
    struct TreeNode* root = new_node(val);
    int m = pos[val + 3000];
    root->left = build(preorder, pos, idx, l, m - 1);
    root->right = build(preorder, pos, idx, m + 1, r);
    return root;
}

void preorder_print(struct TreeNode* root) {
    if (!root) return;
    printf("%d ", root->val);
    preorder_print(root->left);
    preorder_print(root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    int preorder[] = {3, 9, 20, 15, 7};
    int inorder[] = {9, 3, 15, 20, 7};
    int pos[6001] = {0};
    for (int i = 0; i < 5; ++i) pos[inorder[i] + 3000] = i;
    int idx = 0;
    struct TreeNode* root = build(preorder, pos, &idx, 0, 4);
    preorder_print(root);
    printf("\n");
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* build(const std::vector<int>& preorder, int& idx, int l, int r,
                const std::unordered_map<int, int>& pos) {
    if (l > r) return nullptr;
    int val = preorder[idx++];
    TreeNode* root = new TreeNode(val);
    int m = pos.at(val);
    root->left = build(preorder, idx, l, m - 1, pos);
    root->right = build(preorder, idx, m + 1, r, pos);
    return root;
}

void preorderPrint(TreeNode* root) {
    if (!root) return;
    std::cout << root->val << ' ';
    preorderPrint(root->left);
    preorderPrint(root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    std::vector<int> preorder{3, 9, 20, 15, 7};
    std::vector<int> inorder{9, 3, 15, 20, 7};
    std::unordered_map<int, int> pos;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i) pos[inorder[i]] = i;
    int idx = 0;
    TreeNode* root = build(preorder, idx, 0, static_cast<int>(inorder.size()) - 1, pos);
    preorderPrint(root);
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

func buildTree(preorder []int, inorder []int) *TreeNode {
	pos := map[int]int{}
	for i, v := range inorder {
		pos[v] = i
	}
	idx := 0
	var build func(int, int) *TreeNode
	build = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		val := preorder[idx]
		idx++
		root := &TreeNode{Val: val}
		m := pos[val]
		root.Left = build(l, m-1)
		root.Right = build(m+1, r)
		return root
	}
	return build(0, len(inorder)-1)
}

func preorderPrint(root *TreeNode) {
	if root == nil {
		return
	}
	fmt.Print(root.Val, " ")
	preorderPrint(root.Left)
	preorderPrint(root.Right)
}

func main() {
	root := buildTree([]int{3, 9, 20, 15, 7}, []int{9, 3, 15, 20, 7})
	preorderPrint(root)
	fmt.Println()
}
```

```rust
use std::collections::HashMap;

#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn build(
    preorder: &[i32],
    idx: &mut usize,
    l: i32,
    r: i32,
    pos: &HashMap<i32, i32>,
) -> Option<Box<TreeNode>> {
    if l > r {
        return None;
    }
    let val = preorder[*idx];
    *idx += 1;
    let m = *pos.get(&val).unwrap();
    Some(Box::new(TreeNode {
        val,
        left: build(preorder, idx, l, m - 1, pos),
        right: build(preorder, idx, m + 1, r, pos),
    }))
}

fn preorder_collect(root: &Option<Box<TreeNode>>, out: &mut Vec<i32>) {
    if let Some(node) = root {
        out.push(node.val);
        preorder_collect(&node.left, out);
        preorder_collect(&node.right, out);
    }
}

fn main() {
    let preorder = vec![3, 9, 20, 15, 7];
    let inorder = [9, 3, 15, 20, 7];
    let mut pos = HashMap::new();
    for (i, v) in inorder.iter().enumerate() {
        pos.insert(*v, i as i32);
    }
    let mut idx = 0usize;
    let root = build(&preorder, &mut idx, 0, inorder.len() as i32 - 1, &pos);
    let mut out = Vec::new();
    preorder_collect(&root, &mut out);
    println!("{:?}", out);
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildTree(preorder, inorder) {
  const pos = new Map();
  inorder.forEach((v, i) => pos.set(v, i));
  let idx = 0;

  function build(l, r) {
    if (l > r) return null;
    const val = preorder[idx++];
    const m = pos.get(val);
    return new TreeNode(val, build(l, m - 1), build(m + 1, r));
  }

  return build(0, inorder.length - 1);
}

function preorderCollect(root, out = []) {
  if (!root) return out;
  out.push(root.val);
  preorderCollect(root.left, out);
  preorderCollect(root.right, out);
  return out;
}

const root = buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]);
console.log(preorderCollect(root));
```
