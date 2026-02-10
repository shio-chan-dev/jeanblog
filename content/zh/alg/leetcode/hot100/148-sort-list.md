---
title: "Hot100：排序链表（Sort List）链表归并排序 ACERS 解析"
date: 2026-02-10T17:07:38+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "归并排序", "快慢指针", "分治", "LeetCode 148"]
description: "在 O(n log n) 时间内对单链表排序。本文用“快慢指针找中点 + 分治归并”给出稳定模板，并对比数组化方案、常见坑和多语言实现。"
keywords: ["Sort List", "排序链表", "链表归并排序", "分治", "LeetCode 148", "Hot100", "O(n log n)"]
---

> **副标题 / 摘要**  
> LeetCode 148 的核心不是“会排序”，而是“在链表结构里选对排序算法”。数组可随机访问适合快排/堆排，而单链表最匹配的是归并排序：找中点、递归分治、线性归并。

- **预计阅读时长**：12~16 分钟  
- **标签**：`Hot100`、`链表`、`归并排序`、`分治`  
- **SEO 关键词**：Sort List, 排序链表, 链表归并排序, LeetCode 148, Hot100  
- **元描述**：用链表归并排序在 O(n log n) 内完成排序，覆盖思路推导、工程迁移、复杂度分析和多语言可运行实现。

---

## 目标读者

- 正在刷 Hot100，想把链表题模板系统化的同学  
- 做链表题经常在“切分和拼接”环节出错的开发者  
- 想搞清楚“为什么链表排序优先用归并”而不是快排的人

## 背景 / 动机

链表排序在工程里并不罕见：

- 合并来自多个来源的链式任务队列  
- 对按时间追加的链式日志做离线整理  
- 对内存敏感结构进行“尽量少拷贝”的重排

如果把数组排序思维直接搬过来，往往会遇到：

- 链表不支持 O(1) 随机访问，分区/堆操作代价高  
- 频繁节点移动容易写出复杂且脆弱的代码

所以这题本质是：**为链表选择正确的数据结构友好算法**。

## 核心概念

- **分治（Divide & Conquer）**：把链表二分到最小子问题，再向上合并  
- **快慢指针找中点**：`slow` 每次 1 步，`fast` 每次 2 步  
- **链表归并**：两个有序链表线性拼接成一个有序链表  
- **稳定排序**：相等元素相对次序可保持

---

## A — Algorithm（题目与算法）

### 题目还原

给你链表头节点 `head`，请将其按升序排序并返回排序后的链表。  
要求时间复杂度为 `O(n log n)`。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头节点（可能为空） |
| 返回 | ListNode | 升序排序后的头节点 |

### 示例 1

```text
输入: 4 -> 2 -> 1 -> 3
输出: 1 -> 2 -> 3 -> 4
```

### 示例 2

```text
输入: -1 -> 5 -> 3 -> 4 -> 0
输出: -1 -> 0 -> 3 -> 4 -> 5
```

---

## 思路推导（从朴素到最优）

### 朴素做法：转数组再排序

- 遍历链表把值放到数组  
- 调库排序后重建链表

问题：

- 额外空间 O(n)  
- 违背“基于链表结构做排序”的训练目标

### 关键观察

链表最擅长的是：

- 切分（断开 `next`）  
- 线性遍历  
- 拼接（改 `next`）

这正好匹配归并排序：

1. 找中点切成两半  
2. 分别排好序  
3. 线性归并

### 方法选择

采用 **Top-Down 归并排序**：

- 时间：`O(n log n)`  
- 额外空间：递归栈 `O(log n)`  
- 代码结构清晰，稳定可复用

---

## C — Concepts（核心思想）

### 方法归类

- 链表分治排序  
- 快慢指针切分  
- 归并模板复用（可直接复用 LeetCode 21）

### 正确性直觉

1. 递归基：空链表或单节点天然有序  
2. 归纳假设：左右子链递归后有序  
3. 归并步骤：两个有序链表线性归并后仍有序

所以整条链表最终有序。

### 复杂度推导

设 `T(n) = 2T(n/2) + O(n)`，由主定理得：

- `T(n) = O(n log n)`

---

## 实践指南 / 步骤

1. 若 `head` 为空或仅一个节点，直接返回  
2. 用快慢指针找到中点，并断开成左右两条链表  
3. 递归排序左右链表  
4. 用哨兵节点归并两个有序链表  
5. 返回归并结果头

Python 可运行示例（`sort_list.py`）：

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    if head is None or head.next is None:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)
    return merge(left, right)


def merge(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next
```

---

## E — Engineering（工程应用）

### 场景 1：后台任务链重排（Go）

**背景**：任务按插入顺序组成链式结构，但执行优先级需要排序。  
**为什么适用**：链式结构原地切分与归并，比数组来回复制更省内存。

```go
type Task struct {
	Priority int
	Next     *Task
}

func merge(a, b *Task) *Task {
	d := &Task{}
	t := d
	for a != nil && b != nil {
		if a.Priority <= b.Priority {
			t.Next, a = a, a.Next
		} else {
			t.Next, b = b, b.Next
		}
		t = t.Next
	}
	if a != nil { t.Next = a } else { t.Next = b }
	return d.Next
}
```

### 场景 2：日志链离线整理（Python）

**背景**：日志先按到达顺序挂链，再按时间戳二次排序。  
**为什么适用**：线性归并对大数据量更稳，便于批处理。

```python
def merge_sorted_logs(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
```

### 场景 3：前端增量列表合并（JavaScript）

**背景**：本地缓存和远端分页都已排序，需要合并展示。  
**为什么适用**：归并是稳定、可预测的线性合并。

```javascript
function mergeSortedByScore(a, b) {
  let i = 0, j = 0;
  const out = [];
  while (i < a.length && j < b.length) {
    if (a[i].score <= b[j].score) out.push(a[i++]);
    else out.push(b[j++]);
  }
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}
```

---

## R — Reflection（反思与深入）

### 复杂度

- 时间：`O(n log n)`  
- 空间：`O(log n)`（递归栈）

### 替代方案对比

| 方法 | 时间 | 空间 | 问题 |
| --- | --- | --- | --- |
| 转数组排序 | O(n log n) | O(n) | 额外内存大，链表优势丢失 |
| 链表快排 | 平均 O(n log n) | O(log n) | 分区实现复杂，最坏 O(n²) |
| 链表归并（本题） | O(n log n) | O(log n) | 稳定、实现清晰 |

### 常见错误

1. 中点切分忘了 `slow.next = None`，导致递归死循环  
2. 快慢指针起点写错，偶数长度切分不均  
3. 归并时遗漏尾部剩余链表  
4. 在递归中复用已断链节点不当，造成链表丢失

### 为什么这是最工程可行方案

因为它完全贴合链表能力边界：

- 不依赖随机访问  
- 每层只做线性操作  
- 模板可复用到 `merge k lists`、`链表去重归并` 等场景

---

## 常见问题与注意事项

1. **这题能做到 O(1) 额外空间吗？**  
   若用自顶向下递归，通常是 O(log n) 栈空间；要严格 O(1) 可用自底向上迭代归并。  

2. **为什么不直接用快排？**  
   链表快排分区成本高且最坏退化明显，归并更稳。  

3. **稳定性重要吗？**  
   当节点值相等但携带额外业务字段时，稳定排序通常更可控。

---

## 最佳实践与建议

- 把“找中点 + 断链 + 归并”做成固定模板  
- 写单测覆盖空链、单节点、偶数长度、重复值  
- 优先保证指针安全，再谈常数优化  
- 需要极限空间时再进阶到底向上迭代归并

---

## S — Summary（总结）

- 排序链表的最佳默认解是归并排序，不是快排  
- 关键步骤：中点切分、递归排序、线性归并  
- 复杂度满足题目目标：O(n log n)  
- 该模板是多题共用核心能力

### 推荐延伸阅读

- LeetCode 21. Merge Two Sorted Lists  
- LeetCode 23. Merge k Sorted Lists  
- LeetCode 147. Insertion Sort List  
- LeetCode 25. Reverse Nodes in k-Group

---

## 参考与延伸阅读

- https://leetcode.com/problems/sort-list/  
- https://en.cppreference.com/w/cpp/algorithm/stable_sort  
- https://docs.python.org/3/howto/sorting.html  
- https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html

---

## 元信息

- **阅读时长**：12~16 分钟  
- **标签**：Hot100、链表、归并排序、分治  
- **SEO 关键词**：Sort List, 排序链表, 链表归并排序, LeetCode 148  
- **元描述**：LeetCode 148 链表归并排序模板，覆盖推导、复杂度、工程场景与多语言实现。

---

## 行动号召（CTA）

建议你现在做两件事：

1. 手写一遍递归版链表归并排序（不看答案）  
2. 再实现自底向上迭代版，对比空间复杂度与代码复杂度

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sortList(head):
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sortList(head)
    right = sortList(mid)
    return merge(left, right)


def merge(a, b):
    dummy = ListNode()
    t = dummy
    while a and b:
        if a.val <= b.val:
            t.next, a = a, a.next
        else:
            t.next, b = b, b.next
        t = t.next
    t.next = a if a else b
    return dummy.next
```

```c
typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

static ListNode* merge(ListNode* a, ListNode* b) {
    ListNode dummy = {0, NULL};
    ListNode* t = &dummy;
    while (a && b) {
        if (a->val <= b->val) {
            t->next = a;
            a = a->next;
        } else {
            t->next = b;
            b = b->next;
        }
        t = t->next;
    }
    t->next = a ? a : b;
    return dummy.next;
}

ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* slow = head;
    ListNode* fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode* mid = slow->next;
    slow->next = NULL;
    ListNode* left = sortList(head);
    ListNode* right = sortList(mid);
    return merge(left, right);
}
```

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x=0, ListNode* n=nullptr): val(x), next(n) {}
};

class Solution {
    ListNode* merge(ListNode* a, ListNode* b) {
        ListNode dummy;
        ListNode* t = &dummy;
        while (a && b) {
            if (a->val <= b->val) t->next = a, a = a->next;
            else t->next = b, b = b->next;
            t = t->next;
        }
        t->next = a ? a : b;
        return dummy.next;
    }
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *slow = head, *fast = head->next;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* mid = slow->next;
        slow->next = nullptr;
        return merge(sortList(head), sortList(mid));
    }
};
```

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	mid := slow.Next
	slow.Next = nil
	left := sortList(head)
	right := sortList(mid)
	return merge(left, right)
}

func merge(a, b *ListNode) *ListNode {
	dummy := &ListNode{}
	t := dummy
	for a != nil && b != nil {
		if a.Val <= b.Val {
			t.Next = a
			a = a.Next
		} else {
			t.Next = b
			b = b.Next
		}
		t = t.Next
	}
	if a != nil {
		t.Next = a
	} else {
		t.Next = b
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
    fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut vals = Vec::new();
    let mut cur = head;
    let mut p = cur;
    while let Some(mut node) = p {
        vals.push(node.val);
        p = node.next.take();
    }
    vals.sort_unstable();
    let mut new_head = None;
    for v in vals.into_iter().rev() {
        let mut node = Box::new(ListNode::new(v));
        node.next = new_head;
        new_head = Some(node);
    }
    new_head
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function sortList(head) {
  if (!head || !head.next) return head;
  let slow = head, fast = head.next;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  const mid = slow.next;
  slow.next = null;
  return merge(sortList(head), sortList(mid));
}

function merge(a, b) {
  const dummy = new ListNode(0);
  let t = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      t.next = a;
      a = a.next;
    } else {
      t.next = b;
      b = b.next;
    }
    t = t.next;
  }
  t.next = a || b;
  return dummy.next;
}
```
