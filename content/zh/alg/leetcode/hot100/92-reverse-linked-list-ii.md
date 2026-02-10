---
title: "Hot100：反转链表 II（Reverse Linked List II）哑节点+头插法 ACERS 解析"
date: 2026-02-10T09:56:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "区间反转", "哑节点", "头插法", "LeetCode 92"]
description: "在单链表中仅反转区间 [left, right]：用哑节点定位前驱，再用头插法原地重排，时间 O(n)、额外空间 O(1)，附工程思路与多语言实现。"
keywords: ["Reverse Linked List II", "反转链表 II", "区间反转", "哑节点", "头插法", "LeetCode 92", "Hot100"]
---

> **副标题 / 摘要**  
> 反转链表 II 的关键不在“会反转”，而在“只反转中间一段且不破坏两端连接”。本文用 ACERS 结构讲清哑节点定位、头插法重排与边界处理，给出可复用模板与多语言代码。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`链表`、`区间反转`、`哑节点`  
- **SEO 关键词**：Reverse Linked List II, 反转链表 II, 区间反转, 哑节点, 头插法, LeetCode 92, Hot100  
- **元描述**：单链表区间反转的工程化模板：哑节点 + 头插法，O(n)/O(1)，附推导、常见坑与多语言实现。  

---

## 目标读者

- 正在刷 Hot100，已会 206 反转链表，想进一步掌握“局部反转”的同学
- 经常在链表题里卡边界（`left=1`、`right=n`）的中级开发者
- 希望把链表指针操作做成稳定模板的工程师

## 背景 / 动机

`Reverse Linked List`（206）是“整条反转”，而 92 要求“只反转一个闭区间”。

这类“局部重排”在工程里非常常见：

- 任务链中的某个分段要逆序重放
- 事件日志只对一段做回滚重连
- 数据结构需要在不重建节点的前提下原地调整

难点并非复杂算法，而是：

- 找准区间前驱与区间首节点
- 反转过程中不丢失后续链路
- 区间反转后把前后两端重新接回去

## 核心概念

- **哑节点（dummy）**：统一处理 `left = 1` 场景，避免头节点特判地狱
- **前驱指针 `prev`**：最终停在第 `left-1` 个节点（若 `left=1` 则停在 `dummy`）
- **当前指针 `cur`**：初始为区间首节点 `prev.next`
- **头插法（head insertion）**：每次把 `cur` 后面的一个节点摘下，插到 `prev` 后面

---

## A — Algorithm（题目与算法）

### 题目还原

给你单链表的头节点 `head` 和两个整数 `left`、`right`（`1 <= left <= right <= n`），
请你反转从位置 `left` 到位置 `right` 的链表节点，返回反转后的链表。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头节点 |
| left | int | 反转区间左边界（1-based） |
| right | int | 反转区间右边界（1-based） |
| 返回 | ListNode | 区间反转后的头节点 |

### 示例 1

```text
输入: head = 1 -> 2 -> 3 -> 4 -> 5, left = 2, right = 4
输出: 1 -> 4 -> 3 -> 2 -> 5
```

### 示例 2

```text
输入: head = 5, left = 1, right = 1
输出: 5
```

---

## 思路推导（从朴素到最优）

### 朴素思路：拆数组再写回

- 先把链表转数组
- 对 `[left-1, right-1]` 子数组反转
- 再重建链表或回写值

缺点：

- 需要 O(n) 额外空间
- 如果题目或工程要求“节点身份不变”（不是只改值），这种方案不可用

### 关键观察

我们不需要创建新节点，只需要原地重连 `next` 指针。

对于区间 `[left, right]`，若能拿到它前一个节点 `prev`，就可以反复执行：

1. 取出 `cur.next`（记为 `next`）
2. 把 `next` 从原位置摘下：`cur.next = next.next`
3. 把 `next` 插到 `prev` 后面：
   - `next.next = prev.next`
   - `prev.next = next`

每做一次，区间前端就多一个“已反转节点”。重复 `right-left` 次即可。

---

## C — Concepts（核心思想）

### 方法归类

- 链表原地操作（In-place Linked List Rewiring）
- 局部区间重排（Sublist Reordering）
- 头插法（Head Insertion）

### 不变量（正确性抓手）

在第 `i` 次迭代（`0 <= i <= right-left`）后：

- `prev` 永远指向“已反转区块”的前驱
- `prev.next` 永远是当前已反转区块的新头
- `cur` 永远是已反转区块尾部（也是未处理部分入口）

所以循环结束时：

- `prev.next` 指向反转后区间头
- `cur` 成为反转后区间尾，并已接回后续链路

### 示意（left=2, right=4）

```text
初始:
1 -> 2 -> 3 -> 4 -> 5
     ^
    cur (prev=1)

第1轮头插(摘3插到1后):
1 -> 3 -> 2 -> 4 -> 5
          ^
         cur

第2轮头插(摘4插到1后):
1 -> 4 -> 3 -> 2 -> 5
             ^
            cur
```

---

## 实践指南 / 步骤

1. 创建 `dummy`，并让 `dummy.next = head`
2. 让 `prev` 从 `dummy` 出发，前进 `left-1` 步
3. 令 `cur = prev.next`
4. 重复 `right-left` 次头插：
   - `next = cur.next`
   - `cur.next = next.next`
   - `next.next = prev.next`
   - `prev.next = next`
5. 返回 `dummy.next`

---

## 可运行示例（Python）

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    cur = prev.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next


def build(nums):
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head):
    ans = []
    while head:
        ans.append(head.val)
        head = head.next
    return ans


if __name__ == "__main__":
    h = build([1, 2, 3, 4, 5])
    h = reverse_between(h, 2, 4)
    print(to_list(h))  # [1, 4, 3, 2, 5]
```

---

## 解释与原理（为什么这么做）

与“整链反转”相比，这题本质是“局部摘插”。

如果把 `prev` 看成“固定锚点”，每次把 `cur` 后面的节点摘出来插到锚点后面，
相当于在局部不断把后继节点前置，从而完成区间逆序。

优势：

- 不需要切断并单独反转再拼接，代码路径更短
- 不需要额外容器，空间 O(1)
- 对 `left=1`、`right=n` 等边界都可统一处理

---

## E — Engineering（工程应用）

### 场景 1：任务补偿链局部逆序（Go）

**背景**：补偿任务链上某个区间执行顺序需要逆转。  
**为什么适用**：要保持节点对象不变，且不能整链重建。

```go
package main

import "fmt"

type Node struct {
	Val  int
	Next *Node
}

func reverseBetween(head *Node, left, right int) *Node {
	if head == nil || left == right {
		return head
	}
	dummy := &Node{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
	}
	cur := prev.Next
	for i := 0; i < right-left; i++ {
		nxt := cur.Next
		cur.Next = nxt.Next
		nxt.Next = prev.Next
		prev.Next = nxt
	}
	return dummy.Next
}

func main() {
	_ = fmt.Println
}
```

### 场景 2：链式审计日志局部回滚（Python）

**背景**：只回滚某一段事件，其他段保持顺序。  
**为什么适用**：局部重排、低内存、原地修改。

```python
# 直接复用上方 reverse_between，传入目标 left/right 即可
```

### 场景 3：前端可视化流程节点局部重排（JavaScript）

**背景**：流程编辑器里选中一段节点执行“逆序”。  
**为什么适用**：无需复制整段数据结构，操作成本稳定。

```javascript
function reverseBetween(head, left, right) {
  if (!head || left === right) return head;
  const dummy = { val: 0, next: head };
  let prev = dummy;
  for (let i = 0; i < left - 1; i += 1) prev = prev.next;
  const cur = prev.next;
  for (let i = 0; i < right - left; i += 1) {
    const nxt = cur.next;
    cur.next = nxt.next;
    nxt.next = prev.next;
    prev.next = nxt;
  }
  return dummy.next;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)
- 空间复杂度：O(1)

其中：

- 前驱定位最多走 `left-1` 步
- 局部头插共做 `right-left` 次

### 替代方案对比

| 方法 | 时间 | 空间 | 评价 |
| --- | --- | --- | --- |
| 数组重建 | O(n) | O(n) | 容易写，但不满足原地约束 |
| 切段+整段反转+拼接 | O(n) | O(1) | 可行，但指针拼接点更多 |
| 哑节点+头插法 | O(n) | O(1) | 实现短、边界统一、工程常用 |

### 常见错误

- 忘了 `dummy`，导致 `left=1` 处理混乱
- `prev` 前进步数写错（应前进 `left-1`）
- 头插顺序写反，导致断链
- 用节点值比较而不是节点引用（在链表题里风险很高）

### 为什么这套更稳

- 单一入口：统一从 `dummy` 出发
- 单一循环：只做 `right-left` 次局部搬移
- 单一返回：`dummy.next`

这三个“单一”减少了分支和心智负担。

---

## 常见问题与注意事项

1. **`left == right` 需要做什么？**  
   直接返回原链表，不需要任何操作。

2. **`right` 是不是一定在链表长度内？**  
   题目通常保证合法；工程代码建议先做参数校验。

3. **为什么不用递归？**  
   递归写法可以做，但边界与回溯连接更复杂，且有栈深风险。

4. **这题和 206 的关系？**  
   206 是整链反转；92 是整链反转思想在局部区间上的工程化扩展。

---

## 最佳实践与建议

- 把 `dummy + prev定位 + 头插循环` 记成固定模板
- 先画 5 节点样例手推两轮，确认每条指针变化
- 写完先测 4 组边界：
  - `left=1`
  - `right=n`
  - `left=right`
  - `n=1`
- 多语言迁移时优先保证“操作顺序一致”，再谈语法风格

---

## S — Summary（总结）

- 92 的核心是“局部指针重排”，不是值交换
- 哑节点统一了头部边界，让实现稳定可复用
- 头插法让区间反转在 O(1) 空间内完成
- 循环不变量是排错与证明正确性的关键
- 这是链表高级题（k 组反转、分段重排）的基础模板

### 推荐延伸阅读

- LeetCode 206 — Reverse Linked List
- LeetCode 25 — Reverse Nodes in k-Group
- LeetCode 24 — Swap Nodes in Pairs
- LeetCode 143 — Reorder List

---

## 小结 / 结论

当你把“哑节点 + 头插法”内化为模板后，
区间链表重排会从“高风险指针体操”变成“可预测的工程操作”。

---

## 参考与延伸阅读

- https://leetcode.com/problems/reverse-linked-list-ii/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/book/ch15-01-box.html
- https://go.dev/doc/effective_go

---

## 元信息

- **阅读时长**：12~15 分钟  
- **标签**：Hot100、链表、区间反转、哑节点  
- **SEO 关键词**：Reverse Linked List II, 反转链表 II, 头插法, LeetCode 92  
- **元描述**：区间反转单链表的 O(n)/O(1) 模板实现与工程实践。  

---

## 行动号召（CTA）

建议你现在立刻做两件事：

1. 手写一次 92，不看答案复现“头插法四句”
2. 再做 25（k 组反转），体会“局部模板 + 分组控制”的组合

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    cur = prev.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt

    return dummy.next
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

ListNode* reverseBetween(ListNode* head, int left, int right) {
    if (!head || left == right) return head;

    ListNode dummy;
    dummy.val = 0;
    dummy.next = head;

    ListNode* prev = &dummy;
    for (int i = 0; i < left - 1; ++i) prev = prev->next;

    ListNode* cur = prev->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode* nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = prev->next;
        prev->next = nxt;
    }

    return dummy.next;
}
```

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseBetween(ListNode* head, int left, int right) {
    if (!head || left == right) return head;

    ListNode dummy(0);
    dummy.next = head;

    ListNode* prev = &dummy;
    for (int i = 0; i < left - 1; ++i) prev = prev->next;

    ListNode* cur = prev->next;
    for (int i = 0; i < right - left; ++i) {
        ListNode* nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = prev->next;
        prev->next = nxt;
    }

    return dummy.next;
}
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head == nil || left == right {
		return head
	}
	dummy := &ListNode{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
	}
	cur := prev.Next
	for i := 0; i < right-left; i++ {
		nxt := cur.Next
		cur.Next = nxt.Next
		nxt.Next = prev.Next
		prev.Next = nxt
	}
	return dummy.Next
}
```

```rust
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn reverse_between(head: Option<Box<ListNode>>, left: i32, right: i32) -> Option<Box<ListNode>> {
    if left == right {
        return head;
    }

    let mut vals = Vec::new();
    let mut cursor = head.as_ref();
    while let Some(node) = cursor {
        vals.push(node.val);
        cursor = node.next.as_ref();
    }

    let l = (left - 1) as usize;
    let r = (right - 1) as usize;
    vals[l..=r].reverse();

    let mut dummy = Box::new(ListNode::new(0));
    let mut tail = &mut dummy;
    for v in vals {
        tail.next = Some(Box::new(ListNode::new(v)));
        tail = tail.next.as_mut().unwrap();
    }

    dummy.next
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseBetween(head, left, right) {
  if (!head || left === right) return head;

  const dummy = new ListNode(0, head);
  let prev = dummy;
  for (let i = 0; i < left - 1; i += 1) prev = prev.next;

  const cur = prev.next;
  for (let i = 0; i < right - left; i += 1) {
    const nxt = cur.next;
    cur.next = nxt.next;
    nxt.next = prev.next;
    prev.next = nxt;
  }

  return dummy.next;
}
```
