---
title: "Hot100：二叉树的右视图（Binary Tree Right Side View）层序遍历取每层最后一个 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "BFS", "层序遍历", "队列", "LeetCode 199"]
description: "围绕 LeetCode 199 讲清为什么右视图等价于“每层最后一个节点”，以及按层 BFS 和右优先 DFS 两种稳定写法。"
keywords: ["Binary Tree Right Side View", "二叉树的右视图", "层序遍历", "BFS", "右优先 DFS", "LeetCode 199", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 199 不是在考“看图想象力”，而是在考你能不能把视角问题翻译成层级问题。只要意识到右视图就是每一层最右边那个节点，这题就会立刻变成一个标准层序遍历。

- **预计阅读时长**：10~13 分钟
- **标签**：`Hot100`、`二叉树`、`BFS`、`层序遍历`、`队列`
- **SEO 关键词**：Binary Tree Right Side View, 二叉树的右视图, 层序遍历, BFS, 右优先 DFS, LeetCode 199
- **元描述**：系统讲透 LeetCode 199 的层序遍历解法，解释“右视图 = 每层最后一个节点”的本质，并补充右优先 DFS 视角。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一棵二叉树的根节点 `root`，想象你站在它的右侧，从上到下观察这棵树，返回你能看到的节点值。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `root` | `TreeNode` | 二叉树根节点 |
| 返回值 | `int[]` | 从上到下看到的右视图节点值 |

### 示例 1

```text
输入：root = [1,2,3,null,5,null,4]
输出：[1,3,4]
```

### 示例 2

```text
输入：root = [1,2,3,4,null,null,null,5]
输出：[1,3,4,5]
```

### 示例 3

```text
输入：root = [1,null,3]
输出：[1,3]
```

### 示例 4

```text
输入：root = []
输出：[]
```

### 提示

- 二叉树的节点个数范围是 `[0, 100]`
- `-100 <= Node.val <= 100`

---

## 目标读者

- 已经会层序遍历，但不够熟悉“每层保留哪个节点”这类变形题的学习者
- 一看到“从某个方向看到的节点”就容易被题面叙述绕进去的开发者
- 想把 `102 + 199` 这一组 BFS 树题系统化的读者

## 背景 / 动机

这题很适合练习一个非常重要的动作：

- 把视觉描述改写成数据结构上的层级规则

很多人第一次看到“右视图”时，会先想：

- 每一层最靠右的是谁？
- 要不要真的去算几何位置？
- 会不会和树的宽度、坐标有关？

其实都不用。
题面说的是“从上到下看到的节点”，这说明答案天然按层组织。

一旦你把题目翻译成：

> 对每一层，只保留最右边那个节点。

它就和普通层序遍历只差最后一步。

## 核心概念

- **层序遍历**：按层从上到下访问节点
- **队列**：BFS 最常见的状态容器
- **层宽控制**：先记住当前层有多少节点，再只处理这一层
- **右视图**：每层最后一个被处理到的节点，或者右优先 DFS 下每个深度第一个节点

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先用最小例子看“右视图”到底在挑什么

看：

```text
    1
   / \
  2   3
   \   \
    5   4
```

从右边看过去：

- 第 1 层看到 `1`
- 第 2 层看到 `3`
- 第 3 层看到 `4`

也就是：

```text
[1,3,4]
```

你会发现答案并不是“整棵树右边一条链”，而是：

> 每一层最右边那个节点。

#### Step 2：要按层处理，我们最先需要什么状态？

既然答案是按层出来的，最直接的做法就是 BFS。

所以第一批状态只有两个：

- `queue`：保存还没处理的节点
- `ans`：保存每层最终可见的节点值

#### Step 3：更小的子问题是什么？

不是“一次求完整个右视图”，而是：

> 当前这一层有哪些节点？这层结束后，谁应该被记入答案？

所以每轮循环先固定当前层大小：

```python
level_size = len(queue)
```

这样我们就知道接下来这 `level_size` 个节点都属于同一层。

#### Step 4：什么时候这一层算处理完？

当这一层的 `level_size` 个节点全部出队之后，这一层就结束了。

而其中最后一个出队的节点，就是这一层最右边那个节点。

所以在循环里判断：

```python
if i == level_size - 1:
    ans.append(node.val)
```

#### Step 5：下一层节点从哪里来？

对当前层每个节点，都按正常 BFS 方式扩展孩子：

```python
if node.left:
    queue.append(node.left)
if node.right:
    queue.append(node.right)
```

这一步没有任何“右视图特技”，就是标准层序遍历。

#### Step 6：为什么“每层最后一个”一定是右视图？

因为当前层节点是按从左到右进入队列并被处理的。

所以：

- 第一个处理的是这一层更靠左的节点
- 最后一个处理的是这一层更靠右的节点

从右边看过去，能留下的正是最后那个。

#### Step 7：除了 BFS，还能怎么理解这个题？

如果你做右优先 DFS：

- 先访问右子树
- 再访问左子树

那么每个深度第一次到达的节点，也会是这一层最右边的节点。

所以这题其实有两种很自然的视角：

- BFS：记录每层最后一个
- 右优先 DFS：记录每个深度第一个

#### Step 8：慢速走一遍官方示例

看：

```text
root = [1,2,3,null,5,null,4]
```

1. 第 1 层只有 `1`，记录 `1`
2. 第 2 层是 `2,3`，最后一个是 `3`
3. 第 3 层是 `5,4`，最后一个是 `4`

于是答案就是：

```text
[1,3,4]
```

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- 队列
- 当前层宽度
- 最后一个节点入答案
- 正常扩展左右孩子

现在只差把它们合到一个 BFS 模板里。

### Assemble the Full Code

先给一版可直接运行的 Python 示例：

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def right_side_view(root):
    if root is None:
        return []

    ans = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if i == level_size - 1:
                ans.append(node.val)

    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
    print(right_side_view(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以写成下面这样：

```python
from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []

        ans = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == level_size - 1:
                    ans.append(node.val)

        return ans
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 层序遍历
- 按层 BFS
- 每层取最后一个节点

但这题真正要记住的是这个等价转换：

> 右视图不是几何问题，而是“每一层保留最右边节点”的层级问题。

---

## E — Engineering（工程应用）

### 场景 1：组织树里展示每层最外侧负责人（Python）

**背景**：可视化组织架构时，某些简化视图只想展示每层最外侧负责人。  
**为什么适用**：这本质上就是“每层保留一个代表节点”。

```python
from collections import deque


class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def rightmost_each_level(root):
    if root is None:
        return []
    q = deque([root])
    ans = []
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if i == size - 1:
                ans.append(node.name)
    return ans


root = Node("CEO", Node("Eng"), Node("Sales"))
print(rightmost_each_level(root))
```

### 场景 2：服务拓扑里抽取每层最外侧暴露节点（Go）

**背景**：一棵依赖树按层展开时，监控面板只想显示每层最外侧对外暴露的服务。  
**为什么适用**：把“每层最后一个”当成该层展示代表即可。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func rightView(root *Node) []string {
	if root == nil {
		return nil
	}
	q := []*Node{root}
	ans := []string{}
	for len(q) > 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
			if i == size-1 {
				ans = append(ans, node.Name)
			}
		}
	}
	return ans
}

func main() {
	root := &Node{Name: "api", Left: &Node{Name: "auth"}, Right: &Node{Name: "feed"}}
	fmt.Println(rightView(root))
}
```

### 场景 3：前端树形导航里每层只保留末尾入口（JavaScript）

**背景**：有些缩略导航只展示每层最后一个入口，避免整棵树全部展开。  
**为什么适用**：本质上仍然是按层取最后一个节点。

```javascript
function Node(name, left = null, right = null) {
  this.name = name;
  this.left = left;
  this.right = right;
}

function rightView(root) {
  if (!root) return [];
  const q = [root];
  const ans = [];
  while (q.length) {
    const size = q.length;
    for (let i = 0; i < size; i++) {
      const node = q.shift();
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
      if (i === size - 1) ans.push(node.name);
    }
  }
  return ans;
}

const root = new Node("Home", new Node("Docs"), new Node("Blog"));
console.log(rightView(root));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点访问一次
- **空间复杂度**：`O(w)`，`w` 为树的最大层宽

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 按层 BFS | `O(n)` | `O(w)` | 最贴题，最直接 |
| 右优先 DFS | `O(n)` | `O(h)` | 也很优雅，适合递归思维 |
| 每层先收集再取最后一个 | `O(n)` | 更高 | 可以做，但没有必要额外存整层 |

### 常见错误与注意事项

1. **把题目理解成“找最右链”**：这会漏掉像示例 1 那样的跨层情况。  
2. **没有固定层宽**：一旦层与层混在一起，“最后一个节点”就失去意义。  
3. **误以为必须先看右子树**：那是 DFS 写法的视角，不是 BFS 的必要条件。  
4. **空树没单独处理**：根为空时应直接返回空数组。

## 常见问题与注意事项

### 1. 为什么 BFS 里按左后右入队，最后一个仍然是右边节点？

因为同一层节点的处理顺序本来就是从左到右。
所以最后处理到的，正好就是这层最右边的那个。

### 2. 这题和 102 层序遍历是什么关系？

199 本质上就是 102 的一个轻微变形：

- 102 保留整层
- 199 只保留整层里的最后一个节点

### 3. DFS 为什么也能做？

如果你总是先走右子树，再走左子树，那么某个深度第一次到达的节点，一定是这一层最右边的节点。

## 最佳实践与建议

- 一看到“从上到下 / 每层 / 某一侧视图”，优先想层序遍历
- 写 BFS 时先固定 `level_size`，别让不同层混在一起
- 如果你已经熟悉递归，也可以顺手把右优先 DFS 当作备选模板
- 做完 199 后，顺手练 102 和 637，层序题会更稳

## S — Summary（总结）

- 199 的核心不是“右边看”，而是把右视图翻译成“每层最后一个节点”
- 按层 BFS 是最直接的写法，因为题目本身就要求从上到下观察
- 右优先 DFS 也成立，因为每层第一次被看到的节点就是最右节点
- 这题和 102 的层序遍历几乎是一脉相承的变形
- 一旦理解这种转换，很多“树的视图题”都会简单很多

## 参考与延伸阅读

- [LeetCode 199：二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)
- LeetCode 102：二叉树的层序遍历
- LeetCode 637：二叉树的层平均值
- LeetCode 103：二叉树的锯齿形层序遍历

## CTA

建议把 `102 + 199 + 637` 一起做。
这三题都在训练“按层组织信息”，差别只是每层最后保留什么结果，非常适合打包固化。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def right_side_view(root):
    if root is None:
        return []
    ans = []
    q = deque([root])
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if i == size - 1:
                ans.append(node.val)
    return ans


if __name__ == "__main__":
    root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
    print(right_side_view(root))
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

int* rightSideView(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    if (!root) return NULL;

    struct TreeNode* queue[256];
    int front = 0, back = 0;
    int* ans = (int*)malloc(sizeof(int) * 256);
    queue[back++] = root;

    while (front < back) {
        int size = back - front;
        for (int i = 0; i < size; ++i) {
            struct TreeNode* node = queue[front++];
            if (node->left) queue[back++] = node->left;
            if (node->right) queue[back++] = node->right;
            if (i == size - 1) ans[(*returnSize)++] = node->val;
        }
    }
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
    root->left->right = new_node(5);
    root->right->right = new_node(4);

    int size = 0;
    int* ans = rightSideView(root, &size);
    for (int i = 0; i < size; ++i) printf("%d ", ans[i]);
    printf("\n");

    free(ans);
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

std::vector<int> rightSideView(TreeNode* root) {
    if (!root) return {};
    std::queue<TreeNode*> q;
    std::vector<int> ans;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
            if (i == size - 1) ans.push_back(node->val);
        }
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
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->right = new TreeNode(5);
    root->right->right = new TreeNode(4);
    for (int x : rightSideView(root)) std::cout << x << ' ';
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

func rightSideView(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	q := []*TreeNode{root}
	ans := []int{}
	for len(q) > 0 {
		size := len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
			if i == size-1 {
				ans = append(ans, node.Val)
			}
		}
	}
	return ans
}

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val:   2,
			Right: &TreeNode{Val: 5},
		},
		Right: &TreeNode{
			Val:   3,
			Right: &TreeNode{Val: 4},
		},
	}
	fmt.Println(rightSideView(root))
}
```

```rust
use std::collections::VecDeque;

#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn right_side_view(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    let mut ans = Vec::new();
    let mut q: VecDeque<&TreeNode> = VecDeque::new();
    if let Some(node) = root.as_deref() {
        q.push_back(node);
    } else {
        return ans;
    }

    while !q.is_empty() {
        let size = q.len();
        for i in 0..size {
            let node = q.pop_front().unwrap();
            if let Some(left) = node.left.as_deref() {
                q.push_back(left);
            }
            if let Some(right) = node.right.as_deref() {
                q.push_back(right);
            }
            if i + 1 == size {
                ans.push(node.val);
            }
        }
    }
    ans
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: Some(Box::new(TreeNode {
            val: 2,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 5,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: Some(Box::new(TreeNode {
                val: 4,
                left: None,
                right: None,
            })),
        })),
    }));

    println!("{:?}", right_side_view(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function rightSideView(root) {
  if (!root) return [];
  const q = [root];
  const ans = [];
  while (q.length) {
    const size = q.length;
    for (let i = 0; i < size; i++) {
      const node = q.shift();
      if (node.left) q.push(node.left);
      if (node.right) q.push(node.right);
      if (i === size - 1) ans.push(node.val);
    }
  }
  return ans;
}

const root = new TreeNode(1, new TreeNode(2, null, new TreeNode(5)), new TreeNode(3, null, new TreeNode(4)));
console.log(rightSideView(root));
```
