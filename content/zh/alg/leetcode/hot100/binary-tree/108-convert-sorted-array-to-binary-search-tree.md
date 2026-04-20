---
title: "Hot100：将有序数组转换为二叉搜索树（Convert Sorted Array to Binary Search Tree）分治选中点 ACERS 解析"
date: 2026-04-20T09:37:25+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "二叉树", "BST", "分治", "递归", "LeetCode 108"]
description: "围绕 LeetCode 108 讲清为什么每次选中点就能同时满足 BST 有序性和平衡性，以及索引分治写法为什么最稳定。"
keywords: ["Convert Sorted Array to Binary Search Tree", "将有序数组转换为二叉搜索树", "平衡二叉搜索树", "BST", "分治", "LeetCode 108", "Hot100"]
---

> **副标题 / 摘要**  
> LeetCode 108 的关键不在“会递归”，而在于看懂题目同时要求两件事：既要保持 BST 的有序性，又要尽量平衡。只要抓住“中点做根”这个证据，整题就会自然落成一个非常干净的区间分治。

- **预计阅读时长**：11~14 分钟
- **标签**：`Hot100`、`二叉树`、`BST`、`分治`、`递归`
- **SEO 关键词**：Convert Sorted Array to Binary Search Tree, 将有序数组转换为二叉搜索树, 平衡二叉搜索树, BST, 分治, LeetCode 108
- **元描述**：系统讲透 LeetCode 108 的中点分治构造法，覆盖题意推导、正确性解释、工程映射与多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个严格递增的整数数组 `nums`，请把它转换成一棵高度平衡的二叉搜索树。

这里同时包含两个目标：

- 新树必须满足 BST 的大小关系
- 新树还必须尽量平衡，也就是任意节点左右子树高度差不超过 `1`

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `nums` | `int[]` | 严格递增数组 |
| 返回值 | `TreeNode` | 任意一棵合法的高度平衡 BST 根节点 |

### 示例 1

```text
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也同样正确。
```

### 示例 2

```text
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

### 提示

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` 按严格递增顺序排列

---

## 目标读者

- 正在刷 Hot100，希望把“数组转树”的分治模板固定下来的学习者
- 已经会写 BST 判断，但还不够熟悉“BST 构造题”如何从题意推出来的开发者
- 想理解为什么“中点做根”不是技巧，而是由约束直接逼出来的读者

## 背景 / 动机

这题很适合拿来训练一个能力：

- 当题目同时给出多个结构约束时，怎么找那个能一次满足它们的拆分点

很多人第一次做 108，会先想到：

- 顺着数组从左到右插入 BST
- 或者随便挑一个数做根，再递归处理剩下元素

这些方法不一定错，但都不够稳定。
真正可靠的写法应该来自题面本身：

- 数组已经有序
- 树必须是 BST
- 树还要尽量平衡

只要把这三条信息放在一起看，“中点做根”几乎就是唯一自然的选择。

## 核心概念

- **BST 有序性**：左子树所有值都小于根，右子树所有值都大于根
- **高度平衡**：左右子树规模不能长期严重失衡
- **区间分治**：用数组区间 `[l, r]` 表示当前要构造的子树
- **中点做根**：让左右两边元素数量尽可能接近，天然更平衡

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先用最小非平凡例子看“根应该选谁”

假设：

```text
nums = [1,2,3]
```

如果你选 `1` 做根，那么树会变成：

```text
1
 \
  2
   \
    3
```

这棵树仍然是 BST，但明显不平衡。

如果你选中点 `2` 做根：

```text
  2
 / \
1   3
```

BST 和平衡性就同时满足了。

#### Step 2：当前递归最少需要记住什么信息？

我们不需要真的切片复制数组，只需要知道：

> 当前子树应该由 `nums` 的哪一段来构造。

所以最自然的状态是：

- 左边界 `l`
- 右边界 `r`

也就是说，子问题可以写成：

> 用 `nums[l..r]` 这一段构造一棵高度平衡 BST。

#### Step 3：更小的子问题到底是什么？

一旦根选好了，左右子树其实就是两个更小的同类问题：

- 左子树来自左半段
- 右子树来自右半段

所以递归天然成立：

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 4：什么时候说明这一段已经构造完了？

如果区间为空，也就是：

```python
if l > r:
    return None
```

这就是最标准的 base case。

#### Step 5：根节点为什么一定优先考虑中点？

因为题目要“高度平衡”。
想让一段数组生成的树尽量平衡，最直接的方法就是让左右子树规模尽量接近。

这意味着根节点应该尽量把数组平均分成两半：

```python
mid = (l + r) // 2
```

偶数长度时取左中点或右中点都可以，只要你始终一致。

#### Step 6：拿到中点后，代码第一步该做什么？

先把这个值变成当前根节点：

```python
root = TreeNode(nums[mid])
```

然后递归去接上左右子树：

```python
root.left = build(l, mid - 1)
root.right = build(mid + 1, r)
```

#### Step 7：为什么这样一定同时满足 BST 和平衡？

先看 BST：

- 数组严格递增
- `mid` 左边所有元素都小于 `nums[mid]`
- `mid` 右边所有元素都大于 `nums[mid]`

所以左右子树天然满足 BST 大小关系。

再看平衡性：

- 中点把区间尽量均分
- 左右区间长度差最多为 `1`
- 递归地继续均分后，整体高度就会保持在较低水平

所以这不是“记模板”，而是题意直接推出的。

#### Step 8：慢速走一遍官方示例

看：

```text
nums = [-10,-3,0,5,9]
```

1. 整段中点是 `0`，它做根  
2. 左半段 `[-10,-3]` 的中点是 `-10` 或 `-3`，任取一种都合法  
3. 右半段 `[5,9]` 同理  
4. 每一层都在重复“中点做根 + 两边递归”

你会发现整题其实没有新的技巧，只有一个规则反复复用。

#### Step 9：把碎片拼成第一版完整代码

我们已经有了：

- 状态：`l, r`
- base case：`l > r`
- 根的选择：`mid`
- 递归拼接：左右两半

现在只差把它们组装起来。

### Assemble the Full Code

下面先给一版可直接运行的 Python 示例，并用中序遍历检查构造结果是否仍然有序。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst(nums):
    def build(l, r):
        if l > r:
            return None
        mid = (l + r) // 2
        root = TreeNode(nums[mid])
        root.left = build(l, mid - 1)
        root.right = build(mid + 1, r)
        return root

    return build(0, len(nums) - 1)


def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


if __name__ == "__main__":
    nums = [-10, -3, 0, 5, 9]
    root = sorted_array_to_bst(nums)
    print(inorder(root))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(l: int, r: int) -> Optional[TreeNode]:
            if l > r:
                return None
            mid = (l + r) // 2
            root = TreeNode(nums[mid])
            root.left = build(l, mid - 1)
            root.right = build(mid + 1, r)
            return root

        return build(0, len(nums) - 1)
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 分治构造
- 区间递归
- 中点平衡构造

但这题真正要固定下来的不是名字，而是这个判断：

> 当题目要求“有序 + 平衡”时，优先想“中点做根，区间递归”。

---

## E — Engineering（工程应用）

### 场景 1：把排序后的主键批量装配成平衡内存索引（Python）

**背景**：离线任务已经把主键排好序，服务启动时要把它们装进内存树结构。  
**为什么适用**：如果按顺序逐个插入，很容易退化；中点分治能一步构造出更平衡的树。

```python
class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


def build_balanced(keys):
    def dfs(l, r):
        if l > r:
            return None
        m = (l + r) // 2
        return Node(keys[m], dfs(l, m - 1), dfs(m + 1, r))

    return dfs(0, len(keys) - 1)


root = build_balanced([10, 20, 30, 40, 50])
print(root.key)
```

### 场景 2：静态查询系统里构造低高度判定树（C++）

**背景**：某些读多写少的系统会提前把有序阈值构造成判定树，减少最坏查询深度。  
**为什么适用**：中点分治天然让比较次数接近 `O(log n)`。

```cpp
#include <iostream>
#include <vector>

struct Node {
    int val;
    Node* left;
    Node* right;
    explicit Node(int x) : val(x), left(nullptr), right(nullptr) {}
};

Node* build(const std::vector<int>& a, int l, int r) {
    if (l > r) return nullptr;
    int m = (l + r) / 2;
    Node* root = new Node(a[m]);
    root->left = build(a, l, m - 1);
    root->right = build(a, m + 1, r);
    return root;
}

int main() {
    std::vector<int> a{5, 10, 20, 40, 80};
    Node* root = build(a, 0, static_cast<int>(a.size()) - 1);
    std::cout << root->val << '\n';
}
```

### 场景 3：前端把有序断点列表转换为平衡规则树（JavaScript）

**背景**：界面规则或媒体查询断点有时会先离线排序，再装配成树状匹配结构。  
**为什么适用**：平衡树能减少单次匹配时的最坏比较层数。

```javascript
function Node(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function buildBalanced(arr, l = 0, r = arr.length - 1) {
  if (l > r) return null;
  const m = Math.floor((l + r) / 2);
  return new Node(arr[m], buildBalanced(arr, l, m - 1), buildBalanced(arr, m + 1, r));
}

const root = buildBalanced([320, 480, 768, 1024, 1280]);
console.log(root.val);
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：`O(n)`，每个数组元素恰好变成一个树节点
- **空间复杂度**：`O(log n)` 到 `O(n)`  
  对本题这种中点均分写法，递归栈高度通常是 `O(log n)`

### 替代方案对比

| 方法 | 时间 | 额外空间 | 说明 |
| --- | --- | --- | --- |
| 中点分治构造 | `O(n)` | `O(log n)` | 最稳定，直接满足题意 |
| 顺序插入 BST | 最好 `O(n log n)`，最坏 `O(n^2)` | `O(1)` 到 `O(n)` | 很容易退化成链 |
| 每层切片再递归 | `O(n)` 到 `O(n log n)` | 更高 | 可写，但会有额外复制成本 |

### 常见错误与注意事项

1. **只想着“BST”，忘了“平衡”**：顺序插入会过，但不满足题目真正训练点。  
2. **递归里直接切片**：代码短，但会产生额外拷贝。  
3. **边界写错**：`mid - 1` 和 `mid + 1` 非常容易写反。  
4. **误以为答案唯一**：这题往往有多棵合法树，只要满足 BST 和平衡即可。

## 常见问题与注意事项

### 1. 为什么偶数长度时取左中点或右中点都可以？

因为两边元素个数都只会差 `1`，依然满足平衡要求。  
这题不要求唯一结构，所以两种都合法。

### 2. 这题为什么不需要像 98 那样显式验证 BST？

因为数组本来就是严格递增的，我们在中点处分开后：

- 左边天然全小于根
- 右边天然全大于根

BST 性质已经由输入顺序直接保证。

### 3. 这题和二分查找有什么关系？

拆分动作很像二分，但目标不同：

- 二分查找是在找一个值
- 这里是在用“二分式拆分”构造一棵低高度树

## 最佳实践与建议

- 写树构造题时，优先把“当前区间代表什么”写清楚
- 尽量传索引，不要在递归里频繁切片
- 遇到“有序 + 平衡”的组合条件，先试中点分治
- 做完 108 后，顺手练 98 和 230，BST 直觉会连起来

## S — Summary（总结）

- LeetCode 108 的关键不是“会递归”，而是从题面同时读出“有序”和“平衡”
- 用数组中点做根，能一次同时满足 BST 结构约束和高度平衡目标
- 递归状态只需要一个区间 `[l, r]`，不需要拷贝子数组
- 这题是非常标准的区间分治构造题，适合固化树构造模板
- 108 和 98、230 放在一起练，BST 的“构造 + 校验 + 查询”会更完整

## 参考与延伸阅读

- [LeetCode 108：将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)
- LeetCode 98：验证二叉搜索树
- LeetCode 104：二叉树的最大深度
- LeetCode 230：二叉搜索树中第 K 小的元素

## CTA

建议把 `108 + 98 + 230` 当成一组做。
108 负责“构造”，98 负责“校验”，230 负责“查询”，这三题刚好能把 BST 最常见的三类问题串成一条线。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst(nums):
    def build(l, r):
        if l > r:
            return None
        m = (l + r) // 2
        root = TreeNode(nums[m])
        root.left = build(l, m - 1)
        root.right = build(m + 1, r)
        return root

    return build(0, len(nums) - 1)


def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


if __name__ == "__main__":
    root = sorted_array_to_bst([-10, -3, 0, 5, 9])
    print(inorder(root))
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

struct TreeNode* build(int* nums, int l, int r) {
    if (l > r) return NULL;
    int m = (l + r) / 2;
    struct TreeNode* root = new_node(nums[m]);
    root->left = build(nums, l, m - 1);
    root->right = build(nums, m + 1, r);
    return root;
}

void inorder(struct TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->val);
    inorder(root->right);
}

void free_tree(struct TreeNode* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    int nums[] = {-10, -3, 0, 5, 9};
    struct TreeNode* root = build(nums, 0, 4);
    inorder(root);
    printf("\n");
    free_tree(root);
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* build(const std::vector<int>& nums, int l, int r) {
    if (l > r) return nullptr;
    int m = (l + r) / 2;
    TreeNode* root = new TreeNode(nums[m]);
    root->left = build(nums, l, m - 1);
    root->right = build(nums, m + 1, r);
    return root;
}

void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    std::cout << root->val << ' ';
    inorder(root->right);
}

void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

int main() {
    std::vector<int> nums{-10, -3, 0, 5, 9};
    TreeNode* root = build(nums, 0, static_cast<int>(nums.size()) - 1);
    inorder(root);
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

func sortedArrayToBST(nums []int) *TreeNode {
	var build func(int, int) *TreeNode
	build = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		m := (l + r) / 2
		root := &TreeNode{Val: nums[m]}
		root.Left = build(l, m-1)
		root.Right = build(m+1, r)
		return root
	}
	return build(0, len(nums)-1)
}

func inorder(root *TreeNode) {
	if root == nil {
		return
	}
	inorder(root.Left)
	fmt.Print(root.Val, " ")
	inorder(root.Right)
}

func main() {
	root := sortedArrayToBST([]int{-10, -3, 0, 5, 9})
	inorder(root)
	fmt.Println()
}
```

```rust
#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

fn build(nums: &[i32], l: i32, r: i32) -> Option<Box<TreeNode>> {
    if l > r {
        return None;
    }
    let m = (l + r) / 2;
    Some(Box::new(TreeNode {
        val: nums[m as usize],
        left: build(nums, l, m - 1),
        right: build(nums, m + 1, r),
    }))
}

fn inorder(root: &Option<Box<TreeNode>>, out: &mut Vec<i32>) {
    if let Some(node) = root {
        inorder(&node.left, out);
        out.push(node.val);
        inorder(&node.right, out);
    }
}

fn main() {
    let nums = vec![-10, -3, 0, 5, 9];
    let root = build(&nums, 0, nums.len() as i32 - 1);
    let mut out = Vec::new();
    inorder(&root, &mut out);
    println!("{:?}", out);
}
```

```javascript
function TreeNode(val, left = null, right = null) {
  this.val = val;
  this.left = left;
  this.right = right;
}

function sortedArrayToBST(nums) {
  function build(l, r) {
    if (l > r) return null;
    const m = Math.floor((l + r) / 2);
    return new TreeNode(nums[m], build(l, m - 1), build(m + 1, r));
  }
  return build(0, nums.length - 1);
}

function inorder(root, out = []) {
  if (!root) return out;
  inorder(root.left, out);
  out.push(root.val);
  inorder(root.right, out);
  return out;
}

const root = sortedArrayToBST([-10, -3, 0, 5, 9]);
console.log(inorder(root));
```
