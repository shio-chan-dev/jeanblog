---
title: "Hot100：二叉树的中序遍历（Binary Tree Inorder Traversal）递归 / 显式栈 ACERS 解析"
date: 2026-03-06T17:58:21+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "DFS", "栈", "中序遍历", "LeetCode 94"]
description: "用左-根-右模板讲透 LeetCode 94，覆盖递归、显式栈与工程迁移场景，附多语言可运行实现。"
keywords: ["Binary Tree Inorder Traversal", "二叉树的中序遍历", "中序遍历", "显式栈", "DFS", "LeetCode 94", "Hot100"]
---

> **副标题 / 摘要**  
> 二叉树遍历是树题模板的起点，中序遍历则是“递归思维”和“显式栈模拟”最典型的一题。本文按 ACERS 结构拆解 LeetCode 94，把左-根-右的访问顺序、迭代栈写法和工程迁移价值一次讲清。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`二叉树`、`DFS`、`栈`、`中序遍历`  
- **SEO 关键词**：Hot100, Binary Tree Inorder Traversal, 二叉树的中序遍历, 中序遍历, 显式栈, LeetCode 94  
- **元描述**：从递归到显式栈，系统讲透 LeetCode 94 二叉树中序遍历，并给出工程场景迁移与多语言实现。  

---

## 目标读者

- 正在刷 Hot100，希望把树遍历模板固定下来的同学
- 刚从数组 / 链表过渡到树结构，容易把前序、中序、后序顺序写混的开发者
- 需要在 BST、表达式树、抽象语法树里复用“左-根-右”思想的工程师

## 背景 / 动机

中序遍历本身不复杂，但它的训练价值很高：

- 它是“**递归 = 隐式栈**，迭代 = **显式栈**”最容易建立直觉的一题
- 它能帮助你稳定掌握“先一路向左，再回退访问根，再转向右子树”的过程
- 在 **二叉搜索树（BST）** 里，中序遍历天然得到有序序列，工程迁移价值很强

很多人第一次写树题不是逻辑不会，而是：

- 不清楚访问顺序到底是谁先谁后
- 迭代版不知道什么时候入栈、什么时候出栈
- 一旦树为空或只有单边链，代码就容易写乱

这题把模板练熟，后面的验证 BST、找第 k 小元素、恢复二叉搜索树等题会更顺。

## 核心概念

- **中序遍历**：按照 `左子树 -> 根节点 -> 右子树` 的顺序访问
- **DFS（深度优先搜索）**：树遍历最常见的组织方式，中序遍历就是 DFS 的一种访问顺序
- **显式栈**：把递归调用栈手动写出来，用栈保存“回头还要处理的节点”
- **树高 h**：空间复杂度通常写成 `O(h)`，平衡树约为 `O(log n)`，极端退化链表时是 `O(n)`

---

## A — Algorithm（题目与算法）

### 题目还原

给定二叉树根节点 `root`，返回它的 **中序遍历** 结果。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点，可以为空 |
| 返回值 | `int[]` / `List[int]` | 按中序顺序得到的节点值序列 |

### 示例 1

```text
输入: root = [1,null,2,3]
输出: [1,3,2]
解释:
    1
     \
      2
     /
    3

中序顺序是 左 -> 根 -> 右，因此得到 [1,3,2]。
```

### 示例 2

```text
输入: root = []
输出: []
```

### 示例 3

```text
输入: root = [1]
输出: [1]
```

### 约束

- 树中节点数目在 `[0, 100]` 内
- `-100 <= Node.val <= 100`

---

## C — Concepts（核心思想）

### 思路推导：从递归定义到显式栈模板

1. **最自然的写法是递归**  
   对每个节点 `node`：
   - 先遍历左子树
   - 再访问当前节点
   - 最后遍历右子树

   这正好与“中序”的定义一致，代码非常短。

2. **但面试常追问：你能不用递归吗？**  
   因为递归本质上依赖函数调用栈，所以面试官常要求你把这个过程显式写出来。

3. **关键观察：为什么要一路向左入栈？**  
   因为中序顺序要求先处理左子树，所以只要当前节点不为空，就先把它压栈并继续走向 `left`。  
   当走到空节点时，说明“最左侧链”已经到底，这时栈顶就是下一个该访问的根节点。

### 方法归类

- **树 DFS**
- **递归遍历**
- **栈模拟递归**

### 显式栈模板

迭代版可以稳定记成下面四步：

1. `cur = root`
2. 当 `cur != null` 时，一路向左压栈
3. 左边走到底后，弹出栈顶并记录它的值
4. 把 `cur` 切到被弹出节点的右子树，重复上述过程

伪流程如下：

```text
while cur 非空 或 栈非空:
    while cur 非空:
        栈.push(cur)
        cur = cur.left

    cur = 栈.pop()
    记录 cur.val
    cur = cur.right
```

### 为什么这个顺序一定正确

- 每个节点在“左链回退”时恰好被访问一次
- 左子树总是在节点本身之前完成
- 右子树只有在根节点访问之后才会进入处理流程

这正好等价于中序遍历定义，因此结果正确。

---

## 实践指南 / 步骤

### 推荐写法：显式栈迭代版

1. 准备结果数组 `res` 和栈 `stack`
2. `cur` 从根开始
3. 不断把左链入栈
4. 弹栈访问根
5. 转向右子树
6. 栈空且 `cur` 为空时结束

Python 可运行示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    res = []
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res


if __name__ == "__main__":
    root = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
    print(inorder_traversal(root))
```

---

## E — Engineering（工程应用）

### 场景 1：BST 导出有序主键（Python）

**背景**：很多内存索引、缓存字典、教学性质搜索树都会用 BST 存数据。  
**为什么适用**：BST 的中序遍历天然得到升序序列，可以快速导出审计结果或调试快照。

```python
class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


def inorder(node, out):
    if node is None:
        return
    inorder(node.left, out)
    out.append(node.key)
    inorder(node.right, out)


root = Node(5, Node(3, Node(2), Node(4)), Node(7))
result = []
inorder(root, result)
print(result)
```

### 场景 2：表达式树转中缀表达式（JavaScript）

**背景**：编译器、公式编辑器、规则引擎里经常会把表达式组织成二叉树。  
**为什么适用**：中序遍历天然接近“中缀表达式”的阅读顺序，便于展示给人看。

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function inorder(node) {
  if (!node) return "";
  if (!node.left && !node.right) return String(node.val);
  return `(${inorder(node.left)} ${node.val} ${inorder(node.right)})`;
}

const tree = new Node("*", new Node("+", new Node(1), new Node(2)), new Node(3));
console.log(inorder(tree));
```

### 场景 3：调试树形配置的局部顺序（Go）

**背景**：有些规则系统会把“左分支 / 当前节点 / 右分支”作为一种稳定的人工检查顺序。  
**为什么适用**：中序遍历能让开发者按照固定局部顺序看节点，便于做 diff 和人工核对。

```go
package main

import "fmt"

type Node struct {
	Name  string
	Left  *Node
	Right *Node
}

func inorder(node *Node, out *[]string) {
	if node == nil {
		return
	}
	inorder(node.Left, out)
	*out = append(*out, node.Name)
	inorder(node.Right, out)
}

func main() {
	root := &Node{"root", &Node{"L", nil, nil}, &Node{"R", nil, nil}}
	order := []string{}
	inorder(root, &order)
	fmt.Println(order)
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点恰好处理一次
- **空间复杂度**：
  - 递归版：`O(h)` 调用栈
  - 显式栈版：`O(h)` 辅助栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 递归 | `O(n)` | `O(h)` | 最直观，代码最短 |
| 显式栈 | `O(n)` | `O(h)` | 面试最常考，模板复用强 |
| Morris 遍历 | `O(n)` | `O(1)` | 会临时改树结构，理解成本更高 |

### 常见错误与注意事项

- 把“前序 / 中序 / 后序”的访问点写混
- 迭代时忘了在弹栈后转去 `cur.right`
- 只写 `while cur != null`，遗漏“栈里还有节点没处理”的情况
- 递归时没有先判空，直接访问 `node.left`

## 常见问题与注意事项

### 1. 中序遍历一定有序吗？

不是。只有当树满足 **BST 性质** 时，中序结果才是升序。

### 2. 递归和迭代谁更推荐？

面试里两种都要会。刷题初期先用递归建立定义感，再掌握显式栈模板最稳。

### 3. Morris 值得背吗？

可以了解，但不建议在基础题阶段优先记忆。先把递归和显式栈写稳更重要。

## 最佳实践与建议

- 先用一句话记住定义：**左、根、右**
- 迭代模板直接背“左链入栈 -> 弹出访问 -> 转向右子树”
- 看到 BST，优先联想到“中序 = 有序”
- 写树题时把空间复杂度统一写成 `O(h)`，表达更准确

## S — Summary（总结）

- 中序遍历的核心是固定访问顺序：`左 -> 根 -> 右`
- 递归版最符合定义，显式栈版最适合作为面试模板
- 这题训练的是“树递归”和“手动模拟调用栈”两种能力
- 在 BST、表达式树、配置树里，中序思想都有现实工程价值
- 把 94 写稳后，验证 BST、找第 k 小元素等题会明显更顺

## 参考与延伸阅读

- [LeetCode 94: Binary Tree Inorder Traversal](https://leetcode.cn/problems/binary-tree-inorder-traversal/)
- LeetCode 144：二叉树的前序遍历
- LeetCode 145：二叉树的后序遍历
- LeetCode 98：验证二叉搜索树
- LeetCode 230：二叉搜索树中第 K 小的元素

## CTA

可以先自己手写一遍递归版，再不看答案写一遍显式栈版。等你能在 3 分钟内稳定写出 94，树遍历题的基本盘就立住了。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    res = []
    stack = []
    cur = root
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res


if __name__ == "__main__":
    root = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
    print(inorder_traversal(root))
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

int* inorderTraversal(struct TreeNode* root, int* returnSize) {
    struct TreeNode* stack[128];
    int top = 0;
    int* res = (int*)malloc(sizeof(int) * 128);
    *returnSize = 0;
    struct TreeNode* cur = root;

    while (cur != NULL || top > 0) {
        while (cur != NULL) {
            stack[top++] = cur;
            cur = cur->left;
        }
        cur = stack[--top];
        res[(*returnSize)++] = cur->val;
        cur = cur->right;
    }
    return res;
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(1);
    root->right = new_node(2);
    root->right->left = new_node(3);

    int n = 0;
    int* ans = inorderTraversal(root, &n);
    for (int i = 0; i < n; ++i) {
        printf("%d%s", ans[i], i + 1 == n ? "\n" : " ");
    }

    free(ans);
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <stack>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

std::vector<int> inorderTraversal(TreeNode* root) {
    std::vector<int> res;
    std::stack<TreeNode*> st;
    TreeNode* cur = root;
    while (cur || !st.empty()) {
        while (cur) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        st.pop();
        res.push_back(cur->val);
        cur = cur->right;
    }
    return res;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->right = new TreeNode(2);
    root->right->left = new TreeNode(3);

    auto ans = inorderTraversal(root);
    for (size_t i = 0; i < ans.size(); ++i) {
        std::cout << ans[i] << (i + 1 == ans.size() ? '\n' : ' ');
    }

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

func inorderTraversal(root *TreeNode) []int {
	res := []int{}
	stack := []*TreeNode{}
	cur := root
	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, cur.Val)
		cur = cur.Right
	}
	return res
}

func main() {
	root := &TreeNode{Val: 1}
	root.Right = &TreeNode{Val: 2, Left: &TreeNode{Val: 3}}
	fmt.Println(inorderTraversal(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn inorder_traversal(root: &Option<Box<TreeNode>>) -> Vec<i32> {
    fn dfs(node: &Option<Box<TreeNode>>, res: &mut Vec<i32>) {
        if let Some(node) = node {
            dfs(&node.left, res);
            res.push(node.val);
            dfs(&node.right, res);
        }
    }

    let mut res = vec![];
    dfs(root, &mut res);
    res
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 1,
        left: None,
        right: Some(Box::new(TreeNode {
            val: 2,
            left: Some(Box::new(TreeNode {
                val: 3,
                left: None,
                right: None,
            })),
            right: None,
        })),
    }));

    println!("{:?}", inorder_traversal(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function inorderTraversal(root) {
  const res = [];
  const stack = [];
  let cur = root;
  while (cur || stack.length) {
    while (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    cur = stack.pop();
    res.push(cur.val);
    cur = cur.right;
  }
  return res;
}

const root = new TreeNode(1, null, new TreeNode(2, new TreeNode(3), null));
console.log(inorderTraversal(root));
```
