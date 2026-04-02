---
title: "相同的树（Same Tree）同步递归 / BFS ACERS 解析"
date: 2026-03-15T21:29:42+08:00
draft: false
aliases: ["/alg/leetcode/hot100/binary-tree/100-same-tree/"]
categories: ["LeetCode"]
tags: ["二叉树", "DFS", "BFS", "树比较", "LeetCode 100"]
description: "围绕 LeetCode 100 讲清同步递归比较、队列成对校验与结构等价判断，附工程迁移与多语言实现。"
keywords: ["Same Tree", "相同的树", "二叉树比较", "同步递归", "BFS", "LeetCode 100"]
---

> **副标题 / 摘要**  
> LeetCode 100 的关键不在“会不会遍历树”，而在“能不能把两棵树当成一对一对的节点同步比较”。本文按 ACERS 结构拆解同步递归的判断合同、BFS 成对校验写法，以及工程里常见的结构等价判断场景。

- **预计阅读时长**：9~11 分钟  
- **标签**：`二叉树`、`DFS`、`BFS`、`树比较`  
- **SEO 关键词**：Same Tree, 相同的树, 二叉树比较, 同步递归, LeetCode 100  
- **元描述**：系统讲透 LeetCode 100 的同步递归与 BFS 成对比较思路，并延伸到配置树、组件树和语法树的等价判断。  

---

## 目标读者

- 刚开始刷树题，想建立“成对递归”思维的读者
- 能写单棵树 DFS，但一涉及“两棵树同时比较”就容易混乱的开发者
- 需要在配置树、组件树、语法树里判断结构是否一致的工程师

## 背景 / 动机

很多人第一次做 100，会本能地把问题理解成“分别遍历两棵树，再比较结果”。  
这当然能做，但它绕远了。题目真正考的是：

- 你能不能把 `p` 和 `q` 上的对应节点同时拿出来看
- 你能不能把“相同”的定义拆成一套稳定的判断合同
- 你能不能在递归里先处理空节点，再处理值和子树

这类思维在后续很多树题里都会反复出现，比如：

- 判断一棵树是否是另一棵树的子树
- 判断左右子树是否镜像对称
- 校验两份树形配置是否结构等价

所以 100 虽然简单，但它是“**双树同步递归模板**”的起点。

## 核心概念

- **同步递归**：递归函数参数不是一个节点，而是一对节点 `p` 和 `q`
- **结构相同**：对应位置都存在节点，且左右子树结构也一致
- **值相同**：对应位置的节点值相等
- **成对遍历**：无论 DFS 还是 BFS，核心都是“每次处理一对节点”

---

## A — Algorithm（题目与算法）

### 题目还原

给你两棵二叉树的根节点 `p` 和 `q`，编写一个函数来检验这两棵树是否相同。

如果两棵树在结构上完全相同，并且对应节点的值也都相同，则认为它们是相同的树。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| p | TreeNode | 第一棵二叉树的根节点 |
| q | TreeNode | 第二棵二叉树的根节点 |
| 返回值 | bool | 两棵树是否完全相同 |

### 示例 1

```text
输入: p = [1,2,3], q = [1,2,3]
输出: true
解释:
两棵树结构一致，且每个对应节点值都相同。
```

### 示例 2

```text
输入: p = [1,2], q = [1,null,2]
输出: false
解释:
第二层对应位置一个在左边，一个在右边，结构不同。
```

### 示例 3

```text
输入: p = [1,2,1], q = [1,1,2]
输出: false
解释:
结构相同，但第二层对应节点的值不同。
```

### 约束

- 两棵树上的节点数目都在 `[0, 100]` 范围内
- `-10^4 <= Node.val <= 10^4`

---

## C — Concepts（核心思想）

### 思路推导：把“相同”拆成四条判断合同

对任意一对节点 `(p, q)`，你只需要按下面四步判断：

1. **都为空**：说明这一对位置匹配，返回 `true`
2. **只有一个为空**：结构已经不同，返回 `false`
3. **值不同**：对应节点不一致，返回 `false`
4. **值相同且都非空**：继续递归比较
   - `p.left` 和 `q.left`
   - `p.right` 和 `q.right`

写成公式就是：

```text
same(p, q) =
    true,                      if p == null and q == null
    false,                     if exactly one is null
    false,                     if p.val != q.val
    same(p.left, q.left) and same(p.right, q.right), otherwise
```

### 为什么这就是完整答案

题目要求的“相同”只有两部分：

- **结构相同**
- **值相同**

而递归每次都在检查“当前对应位置是否满足这两个条件”，再把问题缩小到左右孩子。  
这就是典型的“局部合同 + 子问题同构”。

### 方法归类

- **树同步递归 / DFS**
- **队列成对校验 / BFS**
- **结构等价判断**

### BFS 为什么也能做

如果你不想用递归，也可以把“节点对”放进队列：

1. 每次弹出一对节点
2. 按与递归同样的四条规则判断
3. 若当前匹配，则把 `(left,left)` 与 `(right,right)` 继续入队

本质没有变，变的只是“调用栈”换成了“显式队列”。

---

## 实践指南 / 步骤

### 推荐写法：同步递归

1. 定义函数 `is_same(p, q)`
2. 先处理空节点组合
3. 再比较节点值
4. 最后递归比较左右子树

Python 可运行示例：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


if __name__ == "__main__":
    a = TreeNode(1, TreeNode(2), TreeNode(3))
    b = TreeNode(1, TreeNode(2), TreeNode(3))
    print(is_same_tree(a, b))
```

### BFS 备选写法

如果你想避免递归深度问题，或者更喜欢显式流程，可以使用队列：

1. 队列里存 `(p, q)` 这样的节点对
2. 每次弹出一对进行比较
3. 满足当前匹配时，把左右孩子成对放回队列

这种写法在“需要顺手打印比较路径”或“需要和非递归框架集成”的工程场景中更方便。

---

## E — Engineering（工程应用）

### 场景 1：树形配置是否发生结构漂移（Python）

**背景**：配置中心常把灰度规则、权限继承、路由树表示成嵌套节点。  
**为什么适用**：发布前常需要判断“线上配置树”和“预发布配置树”是否完全一致。

```python
def same_config(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a["name"] != b["name"]:
        return False
    return same_config(a.get("left"), b.get("left")) and same_config(a.get("right"), b.get("right"))


cfg1 = {"name": "root", "left": {"name": "A"}, "right": {"name": "B"}}
cfg2 = {"name": "root", "left": {"name": "A"}, "right": {"name": "B"}}
print(same_config(cfg1, cfg2))
```

### 场景 2：组件树快照是否等价（JavaScript）

**背景**：前端低代码系统经常把页面布局保存成组件树。  
**为什么适用**：做发布校验或回归测试时，需要确认两份布局快照不仅节点值一样，连层级关系也一致。

```javascript
function sameTree(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (a.type !== b.type) return false;
  return sameTree(a.left, b.left) && sameTree(a.right, b.right);
}

const oldTree = { type: "Split", left: { type: "List" }, right: { type: "Form" } };
const newTree = { type: "Split", left: { type: "List" }, right: { type: "Form" } };
console.log(sameTree(oldTree, newTree));
```

### 场景 3：语法树重写后做结构一致性检查（Go）

**背景**：编译器或规则引擎在重写表达式后，常需要确认重写前后结构是否符合预期。  
**为什么适用**：很多时候你要比较的是“整棵树的形态 + 每个节点标签”。

```go
package main

import "fmt"

type Node struct {
	Label string
	Left  *Node
	Right *Node
}

func same(a, b *Node) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Label != b.Label {
		return false
	}
	return same(a.Left, b.Left) && same(a.Right, b.Right)
}

func main() {
	x := &Node{"Add", &Node{"Num", nil, nil}, &Node{"Num", nil, nil}}
	y := &Node{"Add", &Node{"Num", nil, nil}, &Node{"Num", nil, nil}}
	fmt.Println(same(x, y))
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，其中 `n` 是两棵树中被比较到的节点数；最坏会访问所有对应节点
- **空间复杂度**：
  - 递归 DFS：`O(h)`，`h` 为树高
  - BFS 队列：最坏 `O(w)`，`w` 为某一层的最大宽度

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 同步递归 | `O(n)` | `O(h)` | 最符合题意，推荐 |
| BFS 成对入队 | `O(n)` | `O(w)` | 非递归，便于调试 |
| 序列化后比较 | `O(n)` | `O(n)` | 要小心空节点占位，否则会误判 |
| 哈希签名比较 | 视实现而定 | 额外哈希存储 | 工程上可用于快速过滤，但不如直接比较直观 |

### 常见错误与注意事项

- 只比较中序或前序结果，却没有把空节点位置编码进去，导致不同结构被误判为相同
- 写成 `same(p.left, q.right)` 这种镜像比较，实际上那是 101 的思路
- 空节点判断顺序混乱，先取 `p.val` 结果直接空指针
- 把“值相同”误当成“节点对象地址相同”

## 常见问题与注意事项

### 1. 为什么不能只比较遍历结果？

因为不同结构的树可能产生同样的节点值序列。  
如果你真要序列化，必须把空节点位置也编码进去。

### 2. 这题比较的是“同一个对象”吗？

不是。题目比较的是 **结构和值是否相同**，不是两个根指针是不是同一块内存。

### 3. BFS 和 DFS 谁更推荐？

刷题和面试里，递归更短、更贴近定义；工程里若要避免深递归或记录比较路径，BFS 更顺手。

## 最佳实践与建议

- 双树问题先问自己：递归参数是不是应该有两个节点
- 判空顺序放在最前面，能大幅降低 bug 率
- 明确区分“同值”“同结构”“同引用”三个概念
- 看到 100、101、572 这类题时，优先联想“同步比较模板”

## S — Summary（总结）

- LeetCode 100 的本质是“成对比较”，不是“分别遍历”
- 同步递归只要守住四条判断合同，代码就会非常稳定
- BFS 版本只是把递归里的节点对换成了显式队列
- 结构等价判断在配置树、组件树、语法树里都有直接工程价值
- 把 100 写稳后，再做 101 对称二叉树会顺很多

## 参考与延伸阅读

- [LeetCode 100: Same Tree](https://leetcode.cn/problems/same-tree/)
- LeetCode 101：对称二叉树
- LeetCode 572：另一棵树的子树
- LeetCode 226：翻转二叉树
- LeetCode 102：二叉树的层序遍历

## CTA

建议把 100 和 101 连着练。  
100 是“同方向比较”，101 是“镜像方向比较”；把这两个模板放在一起理解，二叉树判断题会清楚很多。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


if __name__ == "__main__":
    p = TreeNode(1, TreeNode(2), TreeNode(3))
    q = TreeNode(1, TreeNode(2), TreeNode(3))
    print(is_same_tree(p, q))
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

bool isSameTree(struct TreeNode* p, struct TreeNode* q) {
    if (p == NULL && q == NULL) return true;
    if (p == NULL || q == NULL) return false;
    if (p->val != q->val) return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* p = new_node(1);
    p->left = new_node(2);
    p->right = new_node(3);

    struct TreeNode* q = new_node(1);
    q->left = new_node(2);
    q->right = new_node(3);

    printf("%s\n", isSameTree(p, q) ? "true" : "false");
    free_tree(p);
    free_tree(q);
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

bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    if (p->val != q->val) return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* p = new TreeNode(1);
    p->left = new TreeNode(2);
    p->right = new TreeNode(3);

    TreeNode* q = new TreeNode(1);
    q->left = new TreeNode(2);
    q->right = new TreeNode(3);

    std::cout << (isSameTree(p, q) ? "true" : "false") << '\n';
    freeTree(p);
    freeTree(q);
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

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

func main() {
	p := &TreeNode{Val: 1, Left: &TreeNode{Val: 2}, Right: &TreeNode{Val: 3}}
	q := &TreeNode{Val: 1, Left: &TreeNode{Val: 2}, Right: &TreeNode{Val: 3}}
	fmt.Println(isSameTree(p, q))
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

fn is_same_tree(p: &Node, q: &Node) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let a_ref = a.borrow();
            let b_ref = b.borrow();
            a_ref.val == b_ref.val
                && is_same_tree(&a_ref.left, &b_ref.left)
                && is_same_tree(&a_ref.right, &b_ref.right)
        }
        _ => false,
    }
}

fn main() {
    let p = Some(TreeNode::new(1));
    let q = Some(TreeNode::new(1));
    if let Some(root) = &p {
        root.borrow_mut().left = Some(TreeNode::new(2));
        root.borrow_mut().right = Some(TreeNode::new(3));
    }
    if let Some(root) = &q {
        root.borrow_mut().left = Some(TreeNode::new(2));
        root.borrow_mut().right = Some(TreeNode::new(3));
    }
    println!("{}", is_same_tree(&p, &q));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function isSameTree(p, q) {
  if (p === null && q === null) return true;
  if (p === null || q === null) return false;
  if (p.val !== q.val) return false;
  return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
}

const p = new TreeNode(1, new TreeNode(2), new TreeNode(3));
const q = new TreeNode(1, new TreeNode(2), new TreeNode(3));
console.log(isSameTree(p, q));
```
