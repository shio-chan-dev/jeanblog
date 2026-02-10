---
title: "Hot100：K 个一组翻转链表（Reverse Nodes in k-Group）分组反转 ACERS 解析"
date: 2026-02-10T10:01:23+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "分组反转", "哑节点", "双指针", "LeetCode 25"]
description: "在单链表中每 k 个节点为一组进行原地反转，不足 k 的尾组保持不变。本文给出哑节点 + 分组扫描 + 原地反转模板，时间 O(n)、额外空间 O(1)，附多语言实现。"
keywords: ["Reverse Nodes in k-Group", "K个一组翻转链表", "分组反转", "链表", "LeetCode 25", "Hot100", "O(1)空间"]
---

> **副标题 / 摘要**  
> LeetCode 25 是“整链反转（206）”与“区间反转（92）”的组合升级：你要按组切分、组内反转、组间拼接，并正确处理不足 k 的尾组。本文用 ACERS 模板给出工程可复用解法。

- **预计阅读时长**：14~18 分钟  
- **标签**：`Hot100`、`链表`、`分组反转`、`哑节点`  
- **SEO 关键词**：Reverse Nodes in k-Group, K 个一组翻转链表, 分组反转, LeetCode 25, Hot100  
- **元描述**：K 组链表原地反转模板：分组扫描 + 区间反转 + 安全拼接，含复杂度分析、常见坑与多语言代码。  

---

## 目标读者

- 已掌握 206 / 92，希望攻克“多区间连续反转”的 Hot100 学习者
- 链表题常在边界和拼接步骤出错的中级开发者
- 需要构建稳定“链表分段处理”模板的工程师

## 背景 / 动机

在工程里，链式结构的批处理并不少见：

- 任务链按固定批次重排执行
- 流水线节点按批回滚或重放
- 数据清洗链表按批次做原地变换

这类场景的核心诉求是：

- **组内变换**（例如反转）
- **组间保持顺序**
- **尾部残组按规则保留**（不足 k 不反转）

LeetCode 25 正是这个能力的典型建模。

## 核心概念

- **哑节点（dummy）**：统一处理头节点参与反转的场景
- **组前驱（groupPrev）**：指向当前组前一个节点
- **组尾探针（kth）**：从 `groupPrev` 出发找第 k 个节点，判断是否够一组
- **组后继（groupNext）**：当前组反转后要接回的后半链头
- **组内原地反转**：只反转 `[groupStart, kth]` 区间

---

## A — Algorithm（题目与算法）

### 题目还原

给你链表头节点 `head` 和整数 `k`，每 k 个节点一组进行翻转，返回修改后的链表。  
若剩余节点数不足 k，则保持原顺序不变。  
要求只能改节点指针，不允许只改节点值。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头节点 |
| k | int | 每组大小（`k >= 1`） |
| 返回 | ListNode | 分组翻转后的新头节点 |

### 示例 1

```text
输入: head = 1 -> 2 -> 3 -> 4 -> 5, k = 2
输出: 2 -> 1 -> 4 -> 3 -> 5
```

### 示例 2

```text
输入: head = 1 -> 2 -> 3 -> 4 -> 5, k = 3
输出: 3 -> 2 -> 1 -> 4 -> 5
```

---

## 思路推导（从朴素到最优）

### 朴素做法：数组化再分段反转

- 把链表读到数组
- 每 k 个元素反转后重建链表

问题：

- 额外空间 O(n)
- 题目要求“改指针，不改值”时不满足约束

### 关键观察

任务可以拆成重复的三步：

1. 判断当前是否有完整 k 节点
2. 有则反转这一段
3. 反转后接回主链，推进到下一组

本质是“分组驱动的区间反转”，可以复用 92 的区间反转思想。

### 方法选择

采用：

- `dummy + groupPrev` 维护全局链接
- `kth` 判断分组完整性
- 每组做一次原地反转

满足：

- 时间 O(n)
- 空间 O(1)

---

## C — Concepts（核心思想）

### 方法归类

- 链表原地重连（In-place Rewiring）
- 分段处理（Chunk / Batch Processing）
- 双指针边界定位（Predecessor + Kth Scan）

### 循环不变量

在每轮处理开始时：

- `groupPrev.next` 是当前待处理组的首节点
- `groupPrev` 之前的链表已经是最终正确形态

处理一组后：

- 该组被正确反转并拼接
- `groupPrev` 移动到新组尾（即反转前组首）

因此，循环推进不会破坏已完成部分。

### 结构示意（k=3）

```text
dummy -> a -> b -> c -> d -> e -> f -> g
          ^         ^
      groupStart   kth

反转 [a,b,c] 后：
dummy -> c -> b -> a -> d -> e -> f -> g
                     ^
                 新 groupPrev
```

---

## 实践指南 / 步骤

1. 创建 `dummy.next = head`，初始化 `groupPrev = dummy`
2. 从 `groupPrev` 出发向后走 k 步，得到 `kth`
   - 若找不到，说明不足 k，结束
3. 记录 `groupNext = kth.next`
4. 反转 `[groupPrev.next, kth]`：
   - 用 `prev = groupNext` 作为反转初始后继
   - 指针翻转直到到达 `groupNext`
5. 把 `groupPrev.next` 接到反转后新头（原 kth）
6. 把 `groupPrev` 移到新尾（反转前组首），进入下一轮

---

## 可运行示例（Python）

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or k <= 1:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next

        group_next = kth.next
        prev = group_next
        cur = group_prev.next

        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        new_group_head = prev
        new_group_tail = group_prev.next
        group_prev.next = new_group_head
        group_prev = new_group_tail


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
    print(to_list(reverse_k_group(h, 2)))  # [2, 1, 4, 3, 5]
```

---

## 解释与原理（为什么这么做）

`groupPrev` 是整条链的“固定锚点”，每次只处理它后面的一个完整组。  
组内反转时把 `prev` 初始化为 `groupNext`，有两个好处：

- 反转结束后，组尾自动指向后续链
- 不需要额外再处理“组尾接回”分支

这让每组处理都能复用同一段逻辑，不需要按是否尾组做特殊代码路径。

---

## E — Engineering（工程应用）

### 场景 1：批处理任务链重排（Go）

**背景**：任务执行链按批次逆序回放（例如批次补偿）。  
**为什么适用**：每批独立、原地反转、尾批不足 k 保持。

```go
package main

type Node struct {
	Val  int
	Next *Node
}

func reverseKGroup(head *Node, k int) *Node {
	if head == nil || k <= 1 {
		return head
	}
	dummy := &Node{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k && kth != nil; i++ {
			kth = kth.Next
		}
		if kth == nil {
			break
		}
		groupNext := kth.Next
		prev, cur := groupNext, groupPrev.Next
		for cur != groupNext {
			nxt := cur.Next
			cur.Next = prev
			prev = cur
			cur = nxt
		}
		tail := groupPrev.Next
		groupPrev.Next = prev
		groupPrev = tail
	}
	return dummy.Next
}
```

### 场景 2：分段回滚事件链（Python）

**背景**：按固定批次回滚事件，尾部不足一批保持原顺序。  
**为什么适用**：和业务规则高度一致，且便于压测与验证。

```python
# 直接复用上方 reverse_k_group(head, k)
```

### 场景 3：前端节点流水线批量逆序（JavaScript）

**背景**：可视化流程编辑器支持“每 k 个节点倒序”。  
**为什么适用**：链式数据结构可原地更新，交互响应快。

```javascript
function reverseKGroup(head, k) {
  if (!head || k <= 1) return head;
  const dummy = { val: 0, next: head };
  let groupPrev = dummy;

  while (true) {
    let kth = groupPrev;
    for (let i = 0; i < k; i += 1) {
      kth = kth.next;
      if (!kth) return dummy.next;
    }

    const groupNext = kth.next;
    let prev = groupNext;
    let cur = groupPrev.next;
    while (cur !== groupNext) {
      const nxt = cur.next;
      cur.next = prev;
      prev = cur;
      cur = nxt;
    }

    const tail = groupPrev.next;
    groupPrev.next = prev;
    groupPrev = tail;
  }
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：O(n)  
  每个节点最多被访问和改指针常数次。
- 空间复杂度：O(1)  
  只使用常量级临时指针。

### 替代方案与取舍

| 方法 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 数组化重建 | O(n) | O(n) | 容易写，但不满足原地约束 |
| 递归分组反转 | O(n) | O(n/k)~O(n) 栈 | 思路简洁但有栈风险 |
| 迭代分组原地反转 | O(n) | O(1) | 最工程可行，稳定可控 |

### 常见错误思路

- 找 `kth` 时漏判空，导致空指针
- 反转结束后忘记更新 `groupPrev`，死循环
- 尾组不足 k 仍反转，违背题意
- 只交换值不改指针，违反约束

### 为什么当前方法最优

它同时满足：

- 原地（O(1) 额外空间）
- 单次线性扫描（O(n)）
- 边界统一（dummy 处理头组）

在面试和工程里都具备稳定复用价值。

---

## 常见问题与注意事项

1. **`k=1` 怎么处理？**  
   不需要反转，直接返回原链表。

2. **链表长度不是 k 的倍数怎么办？**  
   最后一组不足 k，保持原顺序。

3. **可以递归写吗？**  
   可以，但建议掌握迭代版作为工程默认，避免栈深风险。

4. **这题和 92 的关系？**  
   25 是“固定步长重复做 92 的区间反转”。

---

## 最佳实践与建议

- 固化模板：`dummy -> find kth -> reverse group -> reconnect -> move groupPrev`
- 调试时打印每轮组边界：`groupPrev`、`kth`、`groupNext`
- 先用 `k=2` 手推，再测 `k=3`、`k=len`、`k>len`
- 与 206/92 联动复习，形成“整链/区间/分组”三件套

---

## S — Summary（总结）

- LeetCode 25 的本质是“分组驱动的区间反转”
- dummy 节点让头部边界处理统一
- `kth` 探针决定组完整性，是正确性的前置条件
- 每组原地反转可做到 O(n)/O(1)
- 这题是链表高阶题（k 组操作、分段重排）的核心模板

### 推荐延伸阅读

- LeetCode 206 — Reverse Linked List
- LeetCode 92 — Reverse Linked List II
- LeetCode 24 — Swap Nodes in Pairs
- LeetCode 143 — Reorder List

---

## 小结 / 结论

当你把“分组扫描 + 区间反转 + 安全拼接”写成肌肉记忆后，
LeetCode 25 会从高压指针题，变成可预测的模板题。

---

## 参考与延伸阅读

- https://leetcode.com/problems/reverse-nodes-in-k-group/
- https://en.cppreference.com/w/cpp/container/forward_list
- https://doc.rust-lang.org/book/ch15-01-box.html
- https://go.dev/doc/effective_go

---

## 元信息

- **阅读时长**：14~18 分钟  
- **标签**：Hot100、链表、分组反转、哑节点  
- **SEO 关键词**：Reverse Nodes in k-Group, K 个一组翻转链表, LeetCode 25  
- **元描述**：K 组链表原地反转模板，含推导、复杂度、工程应用与多语言实现。  

---

## 行动号召（CTA）

建议你按这个顺序做一轮巩固：

1. 先闭卷写出 25 的迭代模板
2. 用同一模板改写 24（两两交换）
3. 对比 92，理解“单次区间”与“循环分组”控制差异

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or k <= 1:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next

        group_next = kth.next
        prev = group_next
        cur = group_prev.next

        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        new_group_tail = group_prev.next
        group_prev.next = prev
        group_prev = new_group_tail
```

```c
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

ListNode* reverseKGroup(ListNode* head, int k) {
    if (!head || k <= 1) return head;

    ListNode dummy;
    dummy.val = 0;
    dummy.next = head;

    ListNode* groupPrev = &dummy;

    while (1) {
        ListNode* kth = groupPrev;
        for (int i = 0; i < k; ++i) {
            kth = kth->next;
            if (!kth) return dummy.next;
        }

        ListNode* groupNext = kth->next;
        ListNode* prev = groupNext;
        ListNode* cur = groupPrev->next;

        while (cur != groupNext) {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }

        ListNode* newGroupTail = groupPrev->next;
        groupPrev->next = prev;
        groupPrev = newGroupTail;
    }
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head || k <= 1) return head;

        ListNode dummy(0);
        dummy.next = head;
        ListNode* groupPrev = &dummy;

        while (true) {
            ListNode* kth = groupPrev;
            for (int i = 0; i < k; ++i) {
                kth = kth->next;
                if (!kth) return dummy.next;
            }

            ListNode* groupNext = kth->next;
            ListNode* prev = groupNext;
            ListNode* cur = groupPrev->next;

            while (cur != groupNext) {
                ListNode* nxt = cur->next;
                cur->next = prev;
                prev = cur;
                cur = nxt;
            }

            ListNode* newGroupTail = groupPrev->next;
            groupPrev->next = prev;
            groupPrev = newGroupTail;
        }
    }
};
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil || k <= 1 {
		return head
	}
	dummy := &ListNode{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k; i++ {
			kth = kth.Next
			if kth == nil {
				return dummy.Next
			}
		}

		groupNext := kth.Next
		prev, cur := groupNext, groupPrev.Next
		for cur != groupNext {
			nxt := cur.Next
			cur.Next = prev
			prev = cur
			cur = nxt
		}

		newGroupTail := groupPrev.Next
		groupPrev.Next = prev
		groupPrev = newGroupTail
	}
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

pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let k = k as usize;
    if k <= 1 {
        return head;
    }

    let mut dummy = Box::new(ListNode { val: 0, next: head });
    let mut group_prev: &mut Box<ListNode> = &mut dummy;

    loop {
        let mut check = group_prev.next.as_ref();
        for _ in 0..k {
            match check {
                Some(node) => check = node.next.as_ref(),
                None => return dummy.next,
            }
        }

        let mut cur = group_prev.next.take();
        let mut rev: Option<Box<ListNode>> = None;
        for _ in 0..k {
            let mut node = cur.unwrap();
            cur = node.next.take();
            node.next = rev;
            rev = Some(node);
        }

        group_prev.next = rev;
        for _ in 0..k {
            group_prev = group_prev.next.as_mut().unwrap();
        }
        group_prev.next = cur;
    }
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseKGroup(head, k) {
  if (!head || k <= 1) return head;
  const dummy = new ListNode(0, head);
  let groupPrev = dummy;

  while (true) {
    let kth = groupPrev;
    for (let i = 0; i < k; i += 1) {
      kth = kth.next;
      if (!kth) return dummy.next;
    }

    const groupNext = kth.next;
    let prev = groupNext;
    let cur = groupPrev.next;

    while (cur !== groupNext) {
      const nxt = cur.next;
      cur.next = prev;
      prev = cur;
      cur = nxt;
    }

    const newGroupTail = groupPrev.next;
    groupPrev.next = prev;
    groupPrev = newGroupTail;
  }
}
```
