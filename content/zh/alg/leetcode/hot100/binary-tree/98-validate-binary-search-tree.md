---
title: "Hot100：验证二叉搜索树（Validate Binary Search Tree）区间约束 / 中序判序 ACERS 解析"
date: 2026-04-19T14:52:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "BST", "DFS", "中序遍历", "LeetCode 98"]
description: "围绕 LeetCode 98 讲清 BST 的祖先区间约束、不只和父节点比较的核心原因，以及区间递归与中序判序两种稳定写法。"
keywords: ["Validate Binary Search Tree", "验证二叉搜索树", "BST", "区间约束", "中序遍历", "LeetCode 98", "Hot100"]
---

> **副标题 / 摘要**
> LeetCode 98 最容易写错的地方，不是“不会递归”，而是误以为每个节点只要和自己的父节点比较就够了。真正的 BST 校验要把祖先留下来的上下界一路向下传递。本文按 ACERS 结构把这个不变量讲透，再补上中序遍历判序的等价视角。

- **预计阅读时长**：11~14 分钟
- **标签**：`Hot100`、`二叉树`、`BST`、`DFS`、`中序遍历`
- **SEO 关键词**：Validate Binary Search Tree, 验证二叉搜索树, BST, 区间约束, 中序遍历, LeetCode 98
- **元描述**：系统讲透 LeetCode 98 的区间递归写法与中序递增判定思路，包含推导、工程迁移、多语言实现与高频误区。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个二叉树的根节点 `root`，判断它是否是一棵有效的二叉搜索树（BST）。

有效 BST 需要同时满足：

- 左子树所有节点值都严格小于当前节点值
- 右子树所有节点值都严格大于当前节点值
- 左右子树本身也都必须是 BST

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| root | TreeNode | 二叉树根节点 |
| 返回 | bool | 是否为有效 BST |

### 示例 1

```text
输入：root = [2,1,3]
输出：true
```

### 示例 2

```text
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5，但是右子节点的值是 4 。
```

### 提示

- 树中节点数目范围在 `[1, 10^4]` 内
- `-2^31 <= Node.val <= 2^31 - 1`

---

## 目标读者

- 刷 Hot100，准备把 BST 判断模板彻底固定下来的学习者
- 会写树递归，但一遇到“全局约束”就容易只写局部判断的开发者
- 在工程里处理树形有序结构、范围规则树、层级校验逻辑的工程师

## 背景 / 动机

这题看起来像一题“简单树递归”，但它真正训练的是：

- 什么时候局部条件不够
- 什么时候必须把祖先信息沿递归向下传递

很多人第一次写 98，会写成下面这种错误逻辑：

- `node.left.val < node.val`
- `node.right.val > node.val`

这只能检查“父子关系”，却检查不了“祖先关系”。
而 BST 的本质恰恰是一个**整棵子树都要服从祖先区间**的结构约束。

## 核心概念

- **祖先区间约束**：当前节点允许出现的值范围，不是只看父节点，而是由所有祖先共同决定
- **开区间 `(low, high)`**：当前节点值必须满足 `low < val < high`
- **递归传边界**：左子树收紧上界，右子树收紧下界
- **中序判序**：BST 的中序遍历结果必须严格递增

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先从题目给的反例看出“局部比较不够”

看官方示例 2：

```text
root = [5,1,4,null,null,3,6]
```

一旦根节点是 `5`，那么它的整棵右子树都必须严格大于 `5`。
这时右孩子 `4` 已经直接违规了。

这个反例暴露出的关键不是“4 比 5 小”，而是：

- 我们往右子树走下去之后
- 仍然必须记得祖先 `5` 留下来的下界

也就是说，节点是否合法，不只由它的父节点决定。

#### Step 2：当前部分答案最少要记住什么？

如果当前递归已经走到某个节点，那么我们最少要知道两件事：

- 这个节点允许的最小值 `low`
- 这个节点允许的最大值 `high`

```python
def dfs(node, low, high):
    ...
```

这里的 `(low, high)` 表示一个严格开区间。

#### Step 3：递归真正要解决的子问题是什么？

我们可以把原题改写成一个更小的子问题：

> 给定一棵子树根节点 `node`，以及它允许落入的开区间 `(low, high)`，判断这棵子树是否全部合法。

一旦这样定义，递归语义就非常稳定了。

#### Step 4：什么时候说明这部分已经合法完成？

如果当前节点是空节点，说明这棵子树没有任何违例，自然合法。

```python
if node is None:
    return True
```

空树是合法 BST，这也是树递归最标准的 base case。

#### Step 5：当前节点本身要先满足什么条件？

在继续看左右子树之前，先检查当前值是否落在允许区间里：

```python
if node.val <= low or node.val >= high:
    return False
```

注意这里必须是**严格不等号**。
因为题目要求左边严格小于、右边严格大于，BST 不允许重复值出现在边界上。

#### Step 6：左右子树分别继承什么新边界？

当前节点值一旦确定，左右子树的区间就会进一步收紧：

- 左子树必须落在 `(low, node.val)`
- 右子树必须落在 `(node.val, high)`

```python
return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
```

这一步就是整题的核心。

#### Step 7：这题为什么不需要“撤销状态”？

因为这里传递的是参数，不是共享可变数组。
每一层递归只把更窄的边界传给下一层：

- 不需要 `path`
- 不需要 `used`
- 不需要回溯撤销

真正被“携带向下”的，是不变量：**祖先留下来的合法取值区间**。

#### Step 8：慢速走一条分支

还是看示例 2：

- 根节点 `5`，允许区间是 `(-inf, +inf)`，合法
- 去左子树 `1`，允许区间是 `(-inf, 5)`，合法
- 去右子树 `4`，允许区间是 `(5, +inf)`

走到 `4` 时马上发现：

```text
4 <= 5
```

所以这棵树不合法，根本不需要再继续往下检查。

#### Step 9：中序遍历为什么也成立？

BST 还有一个等价性质：

> 对 BST 做中序遍历，得到的序列必须严格递增。

所以你也可以：

1. 中序遍历整棵树
2. 维护上一个访问到的值 `prev`
3. 如果当前值 `<= prev`，就返回 `False`

不过从“题目是怎么推出来的”角度看，区间约束递归更直接，也更能说明 BST 的本质。

### Assemble the Full Code

下面把刚才得到的片段拼成第一版完整代码。
这版代码可以直接运行。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root):
    def dfs(node, low, high):
        if node is None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root, float("-inf"), float("inf"))


if __name__ == "__main__":
    root = TreeNode(2, TreeNode(1), TreeNode(3))
    print(is_valid_bst(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以整理成更贴近提交环境的写法：

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], low: float, high: float) -> bool:
            if node is None:
                return True
            if node.val <= low or node.val >= high:
                return False
            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

        return dfs(root, float("-inf"), float("inf"))
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 树递归
- 祖先区间约束
- DFS 边界传递

但这题真正要固定下来的不是名字，而是这个不变量：

> 每个节点都必须对“所有祖先一起决定的区间”负责，而不是只对父节点负责。

---

## E — Engineering（工程应用）

### 场景 1：层级额度树的全局范围校验（Python）

**背景**：有些风控或配额系统会把规则组织成树，子规则必须落在祖先给定的允许范围内。  
**为什么适用**：这和 BST 的“祖先区间逐层收紧”完全同构。

```python
class RuleNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def validate_rule_tree(root, low=float("-inf"), high=float("inf")):
    if root is None:
        return True
    if root.value <= low or root.value >= high:
        return False
    return validate_rule_tree(root.left, low, root.value) and validate_rule_tree(root.right, root.value, high)


tree = RuleNode(50, RuleNode(20), RuleNode(80))
print(validate_rule_tree(tree))
```

### 场景 2：反序列化后的内存索引树校验（Go）

**背景**：服务启动时把磁盘中的有序树结构恢复到内存，加载后往往需要做一次一致性检查。  
**为什么适用**：只比较父子不够，必须验证整棵子树是否仍然满足祖先边界。

```go
package main

import "fmt"

type Node struct {
	Key   int
	Left  *Node
	Right *Node
}

func validate(root *Node, low, high int64) bool {
	if root == nil {
		return true
	}
	v := int64(root.Key)
	if v <= low || v >= high {
		return false
	}
	return validate(root.Left, low, v) && validate(root.Right, v, high)
}

func main() {
	root := &Node{Key: 20, Left: &Node{Key: 10}, Right: &Node{Key: 30}}
	fmt.Println(validate(root, -1<<63, 1<<63-1))
}
```

### 场景 3：前端优先级规则树的顺序校验（JavaScript）

**背景**：前端配置中心有时会把规则组织成二叉决策树，节点优先级必须满足全局顺序约束。  
**为什么适用**：一旦子树越界，就说明配置被错误拼接或拖拽。

```javascript
function Node(priority, left = null, right = null) {
  this.priority = priority;
  this.left = left;
  this.right = right;
}

function validate(node, low = -Infinity, high = Infinity) {
  if (!node) return true;
  if (node.priority <= low || node.priority >= high) return false;
  return validate(node.left, low, node.priority) && validate(node.right, node.priority, high);
}

const root = new Node(10, new Node(5), new Node(15));
console.log(validate(root));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个节点只访问一次
- **空间复杂度**：`O(h)`，`h` 为树高，来自递归调用栈

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 祖先区间递归 | `O(n)` | `O(h)` | 最直接，最能体现 BST 本质 |
| 中序遍历判严格递增 | `O(n)` | `O(h)` | 也很稳定，但更像“性质验证” |
| 只比较父子节点 | 看似 `O(n)` | `O(h)` | **错误**，会漏掉祖先约束 |

### 常见错误与注意事项

1. **只比较父节点**：这是最常见的逻辑错误，无法检查更高祖先留下的限制。  
2. **把严格不等号写成非严格不等号**：BST 这里不允许重复值贴边通过。  
3. **边界用 `int` 哨兵直接顶满**：节点值范围已经覆盖 `int32` 两端，工程里更稳的是 `long long / int64` 或 `None` 边界。  
4. **中序遍历忘记维护全局 `prev`**：局部变量会被递归层覆盖，导致判序失真。

## 常见问题与注意事项

### 1. 为什么不能只看当前节点和左右孩子？

因为 BST 约束的是整棵子树。
比如一个节点在右子树里，就必须同时大于所有把它带到右边去的祖先。

### 2. 为什么中序遍历严格递增就能说明它是 BST？

因为 BST 的定义决定了中序访问顺序一定是“从小到大”。
反过来，如果整棵树中序结果严格递增，也能证明其满足 BST 性质。

### 3. 区间递归和中序判序应该优先记哪一个？

面试和学习阶段更建议优先记区间递归。
它更贴合题意，也更容易迁移到“祖先约束传递”的其他树题。

## 最佳实践与建议

- 首选“`low < val < high`”这个不变量来写递归
- 边界类型尽量放宽一层，避免和节点值域撞边
- 手算一遍“右子树继承下界、左子树继承上界”的流程
- 做完 98 后，顺手把 94 和 230 一起练，BST + 中序会更顺

## S — Summary（总结）

- BST 校验的核心不是父子大小关系，而是祖先一路传下来的区间约束
- 写成 `dfs(node, low, high)` 后，整题结构会非常稳定
- `low < val < high` 必须是严格开区间，不能放宽为允许相等
- 中序遍历严格递增是一个等价判断，但区间递归更能解释题目本质
- 这题练熟以后，很多“树上的全局约束”题都会更容易进入状态

## 参考与延伸阅读

- [LeetCode 98：验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)
- LeetCode 94：二叉树的中序遍历
- LeetCode 230：二叉搜索树中第 K 小的元素
- LeetCode 700：二叉搜索树中的搜索

## CTA

建议把 `98 + 94 + 230` 连起来做。
这三题能把 BST、中序遍历和“有序树上的局部查询”一起串起来，特别适合固化一组模板。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root):
    def dfs(node, low, high):
        if node is None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

    return dfs(root, float("-inf"), float("inf"))


if __name__ == "__main__":
    root = TreeNode(2, TreeNode(1), TreeNode(3))
    print(is_valid_bst(root))
```

```c
#include <limits.h>
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

bool dfs(struct TreeNode* node, long long low, long long high) {
    if (node == NULL) return true;
    long long v = node->val;
    if (v <= low || v >= high) return false;
    return dfs(node->left, low, v) && dfs(node->right, v, high);
}

bool isValidBST(struct TreeNode* root) {
    return dfs(root, LLONG_MIN, LLONG_MAX);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    struct TreeNode* root = new_node(2);
    root->left = new_node(1);
    root->right = new_node(3);
    printf("%s\n", isValidBST(root) ? "true" : "false");
    free_tree(root);
    return 0;
}
```

```cpp
#include <climits>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

bool dfs(TreeNode* node, long long low, long long high) {
    if (!node) return true;
    long long v = node->val;
    if (v <= low || v >= high) return false;
    return dfs(node->left, low, v) && dfs(node->right, v, high);
}

bool isValidBST(TreeNode* root) {
    return dfs(root, LLONG_MIN, LLONG_MAX);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    TreeNode* root = new TreeNode(2);
    root->left = new TreeNode(1);
    root->right = new TreeNode(3);
    std::cout << (isValidBST(root) ? "true" : "false") << '\n';
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

func dfs(node *TreeNode, low, high int64) bool {
	if node == nil {
		return true
	}
	v := int64(node.Val)
	if v <= low || v >= high {
		return false
	}
	return dfs(node.Left, low, v) && dfs(node.Right, v, high)
}

func isValidBST(root *TreeNode) bool {
	return dfs(root, -1<<63, 1<<63-1)
}

func main() {
	root := &TreeNode{
		Val:   2,
		Left:  &TreeNode{Val: 1},
		Right: &TreeNode{Val: 3},
	}
	fmt.Println(isValidBST(root))
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn dfs(node: &Option<Box<TreeNode>>, low: i64, high: i64) -> bool {
    match node {
        None => true,
        Some(n) => {
            let v = n.val as i64;
            if v <= low || v >= high {
                return false;
            }
            dfs(&n.left, low, v) && dfs(&n.right, v, high)
        }
    }
}

fn is_valid_bst(root: &Option<Box<TreeNode>>) -> bool {
    dfs(root, i64::MIN, i64::MAX)
}

fn main() {
    let root = Some(Box::new(TreeNode {
        val: 2,
        left: Some(Box::new(TreeNode {
            val: 1,
            left: None,
            right: None,
        })),
        right: Some(Box::new(TreeNode {
            val: 3,
            left: None,
            right: None,
        })),
    }));

    println!("{}", is_valid_bst(&root));
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function dfs(node, low, high) {
  if (!node) return true;
  if (node.val <= low || node.val >= high) return false;
  return dfs(node.left, low, node.val) && dfs(node.right, node.val, high);
}

function isValidBST(root) {
  return dfs(root, -Infinity, Infinity);
}

const root = new TreeNode(2, new TreeNode(1), new TreeNode(3));
console.log(isValidBST(root));
```
