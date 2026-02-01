---
title: "Hot100：环形链表 II（Linked List Cycle II）Floyd 判环 + 定位入环点 ACERS 解析"
date: 2026-02-01T21:40:21+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "快慢指针", "Floyd", "双指针", "LeetCode 142"]
description: "不修改链表的前提下返回入环的第一个节点：Floyd 快慢指针先相遇判环，再从头与相遇点同步前进定位入环点；O(n) 时间、O(1) 额外空间。"
keywords: ["Linked List Cycle II", "环形链表 II", "入环点", "Floyd", "fast slow pointers", "O(1) space", "LeetCode 142"]
---

> **副标题 / 摘要**  
> 这题的价值在于把“判环”升级为“定位入环点”。最稳的工程化模板是 Floyd：先用快慢指针在环内相遇，再让一个指针回到头结点同步走，下一次相遇的位置就是入环点。全程不修改链表，O(n) 时间、O(1) 额外空间。

- **预计阅读时长**：12~16 分钟  
- **标签**：`Hot100`、`链表`、`快慢指针`、`Floyd`  
- **SEO 关键词**：环形链表 II, 入环点, Floyd 判圈, 快慢指针, O(1) 空间, LeetCode 142  
- **元描述**：Floyd 快慢指针判环并定位入环点：相遇后从头与相遇点同步前进，返回入环的第一个节点；O(n)/O(1)，不允许修改链表。  

---

## 目标读者

- 刷 Hot100，想把“判环/入环点定位”模板一次性吃透的学习者  
- 需要写健壮链式结构遍历（避免死循环）并能定位故障节点的工程师  
- 面试里被问到“为什么 reset 之后会在入环点相遇”的同学

## 背景 / 动机

链表一旦出现环，任何“遍历到 null 为止”的代码都可能进入死循环。  
工程里造成环的原因很多：指针写错、复用节点、数据结构被破坏、并发读写导致 next 异常等。  
因此除了“有没有环”，更重要的是：

- **环从哪里开始？**（入环点）

找到入环点可以帮助你定位哪一个节点的 `next` 被错误地连回去了，这比单纯返回 `true/false` 更有诊断价值。

题目还明确要求：**不允许修改链表**，所以不能用“打标记/改值/断链”等手段。

## 核心概念

| 概念 | 含义 | 作用 |
| --- | --- | --- |
| 环 | 沿 next 走能再次回到某节点 | 会导致遍历死循环 |
| 入环点 | 从头结点沿 next 首次进入环的那个节点 | 题目要求返回它 |
| Floyd 判圈 | 快慢指针：slow 每次 1 步，fast 每次 2 步 | O(1) 空间判环 |
| 相遇点 | slow 与 fast 在环内第一次相遇的位置 | 用来进一步定位入环点 |
| 引用相等 | 判断是否为同一节点对象/地址 | 不能用值相等代替 |

---

## A — Algorithm（题目与算法）

### 题目还原

给定链表头节点 `head`，返回链表开始入环的第一个节点；如果链表无环，返回 `null`。

说明：

- 评测用 `pos` 表示尾节点连接到链表中的位置（0-based），`pos=-1` 表示无环  
- `pos` 不会作为参数传入，只用于描述测试构造  
- 不允许修改链表

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头结点 |
| 返回 | ListNode / null | 入环点节点引用，或 null |

### 示例 1（有环，入环点在值为 2 的节点）

```text
head = 3 -> 2 -> 0 -> -4
                ^     |
                |_____|

输出: 节点(2)  （返回节点引用/地址，不是索引或数值）
```

### 示例 2（无环）

```text
head = 1 -> 2 -> 3
输出: null
```

---

## 思路推导：从哈希到 Floyd + 入环点定位

### 朴素解（容易写对）：哈希集合记录访问过的节点

遍历链表：

- 若当前节点已经在集合里，说明第一次“重复访问”的节点就是入环点  
- 否则加入集合继续走

复杂度：O(n) 时间，O(n) 空间。  
缺点：空间不满足“更优”的要求，且大链表会带来明显内存压力。

### 关键目标：O(1) 额外空间 + 不修改链表

这时就轮到 Floyd（龟兔赛跑）登场：

1) **判环**：快慢指针若能相遇，则必有环；否则 fast 先到 null，无环  
2) **定位入环点**：相遇后，把一个指针放回头结点；两个指针每次都走 1 步，再次相遇点即入环点

接下来最重要的问题是：**为什么这样一定会在入环点相遇？**

---

## C — Concepts（核心思想）

### 方法归类

- **Floyd cycle detection（快慢指针判环）**  
- **相遇点性质（Meeting point property）**  
- **距离对齐（Distance alignment by reset）**

### 正确性证明（入环点定位为什么对）

用最常见的距离记号：

- 从头结点到入环点的距离为 `a`
- 从入环点到相遇点沿环走的距离为 `b`
- 环的长度为 `c`

慢指针走了 `a + b` 步到达相遇点。  
快指针走了 `2(a + b)` 步到达相遇点。

快指针比慢指针多走的步数：

```
2(a + b) - (a + b) = a + b
```

而“多走的部分”一定是绕环的整数圈：

```
a + b = k * c   （k 是正整数）
```

于是：

```
a = k*c - b = (k-1)*c + (c - b)
```

注意 `(c - b)` 正是“从相遇点沿环走到入环点”的距离。  
因此：

- 指针 P 从头结点走 `a` 步会到入环点  
- 指针 Q 从相遇点走 `a` 步，也等价于先绕 `(k-1)` 圈再走 `(c-b)`，最终也到入环点

所以把一个指针放回头结点，另一个留在相遇点，二者同速前进，**必在入环点相遇**。

---

## 实践指南 / 步骤

1. 用 `slow`、`fast` 从 head 出发，slow 每次 1 步，fast 每次 2 步  
2. 若 fast 走到 null，返回 null（无环）  
3. 若 slow == fast（相遇），进入第二阶段  
4. 令 `p = head`，`q = slow`（或 fast）  
5. `p` 与 `q` 每次各走 1 步，直到 `p == q`，返回该节点（入环点）

Python 可运行示例（保存为 `cycle_entry.py`）：

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def detect_cycle(head: ListNode | None) -> ListNode | None:
    slow = head
    fast = head

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            q = slow
            while p is not q:
                p = p.next  # type: ignore[assignment]
                q = q.next  # type: ignore[assignment]
            return p

    return None


if __name__ == "__main__":
    # 3 -> 2 -> 0 -> -4 -> (back to 2)
    n3 = ListNode(3)
    n2 = ListNode(2)
    n0 = ListNode(0)
    n4 = ListNode(-4)
    n3.next = n2
    n2.next = n0
    n0.next = n4
    n4.next = n2  # cycle entry

    ans = detect_cycle(n3)
    print(ans.val if ans else None)  # 2
```

---

## E — Engineering（工程应用）

### 场景 1：任务编排/工作流 next 指针错误定位（Python）

**背景**：某些轻量工作流用“next 任务”链表表示执行顺序；配置错误会形成环，导致任务一直跑不完。  
**为什么适用**：你不仅要知道“有环”，更要知道“从哪一个任务开始进入环”，便于直接定位配置节点。  

```python
class Task:
    def __init__(self, name):
        self.name = name
        self.next = None


def entry(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p, q = head, slow
            while p is not q:
                p = p.next
                q = q.next
            return p
    return None


if __name__ == "__main__":
    a = Task("A"); b = Task("B"); c = Task("C"); d = Task("D")
    a.next = b; b.next = c; c.next = d; d.next = b  # loop at B
    hit = entry(a)
    print(hit.name if hit else "no cycle")
```

### 场景 2：free-list/对象池链表自检（C）

**背景**：系统工程里常见“空闲块链表（free list）/对象池链表”。一旦指针写坏形成环，会导致分配器遍历卡死。  
**为什么适用**：不额外分配内存、不会改结构，适合作为运行时自检（或 debug 模式下的断言）。  

```c
struct Node { struct Node* next; /* ... */ };

struct Node* detectCycle(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            struct Node* p = head;
            struct Node* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return 0;
}
```

### 场景 3：前端/脚本中的链式路由配置排错（JavaScript）

**背景**：有些页面路由或步骤导航用 next 指针串起来，配置错误可能回指造成无限跳转。  
**为什么适用**：JS 对象引用天然支持“节点相等”判断；返回入环点可直接标红提示哪一个 step 配错。  

```javascript
function detectCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p = head, q = slow;
      while (p !== q) {
        p = p.next;
        q = q.next;
      }
      return p;
    }
  }
  return null;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)（判环最多 O(n)，定位入环点最多再走 O(n)）  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 备注 |
| --- | --- | --- | --- | --- |
| 哈希集合 | 记录访问过的节点，重复即入环点 | O(n) | O(n) | 最直观，调试友好 |
| **Floyd + reset** | 快慢指针相遇后同步走定位入环点 | **O(n)** | **O(1)** | 最经典模板 |

### 常见坑

1. **用值相等代替节点相等**：入环点必须是“同一节点对象/地址”。  
2. **忽略空指针**：判环阶段必须检查 `fast` 和 `fast.next`。  
3. **相遇后直接返回相遇点**：相遇点不一定是入环点（通常不是）。  
4. **链表可能被修改**：题目不允许修改；工程里也建议把恢复/不改结构作为默认习惯。

---

## 常见问题与注意事项

1. **为什么相遇后把一个指针放回 head 就可以？**  
   因为推导得到 `a = (k-1)c + (c-b)`，从 head 与相遇点同速走 `a` 步都会到入环点。

2. **如果只有一个节点并且自环呢？**  
   head.next = head，快慢指针会相遇，最终返回 head，符合预期。

3. **题目说不允许修改链表，Floyd 算法会修改吗？**  
   不会。Floyd 只移动指针变量，不改节点的 `next`。

---

## 最佳实践与建议

- 默认背模板：`fast = fast.next.next`、`slow = slow.next`；相遇后 `p=head, q=slow` 同步走  
- 工程里如果结构可能“并发修改”，需要先保证遍历期间结构稳定（否则任何判环都可能不可靠）  
- 调试时如果你用哈希法更方便，线上/性能敏感再切换到 Floyd

---

## S — Summary（总结）

### 核心收获

- 环导致遍历死循环，工程里比“无环链表”更常见更危险  
- 题目要返回入环点，不能只判有无环  
- Floyd 快慢指针能 O(1) 空间判环并得到相遇点  
- 相遇后 reset 一个指针到 head 同速前进，会在入环点相遇（有严格距离推导）  
- 全程不修改链表结构，符合题目约束与工程安全性

### 参考与延伸阅读

- LeetCode 142. Linked List Cycle II
- Floyd Cycle Detection（龟兔赛跑）经典证明与应用
- 相关题：LeetCode 141（判环）、LeetCode 160（相交链表）、LeetCode 234（回文链表）

---

## 元信息

- **阅读时长**：12~16 分钟  
- **标签**：Hot100、链表、Floyd、入环点  
- **SEO 关键词**：环形链表 II, Linked List Cycle II, 入环点, Floyd, O(1) 空间  
- **元描述**：Floyd 快慢指针判环并定位入环点：相遇后从头与相遇点同步前进，返回入环的第一个节点；O(n)/O(1)，不修改链表。  

---

## 行动号召（CTA）

建议你把本文的推导自己手推一遍（`a,b,c` 的关系），并用 3 种用例自测：无环、短环（自环）、长链 + 中间入环。  
如果你希望我把 141（仅判环）也按同风格补成 Hot100 系列文章，我可以直接继续写。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from __future__ import annotations


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: ListNode | None = None


def detect_cycle(head: ListNode | None) -> ListNode | None:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            q = slow
            while p is not q:
                p = p.next  # type: ignore[assignment]
                q = q.next  # type: ignore[assignment]
            return p
    return None
```

```c
#include <stdio.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* detectCycle(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            struct ListNode* p = head;
            struct ListNode* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return 0;
}
```

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* detectCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            ListNode* p = head;
            ListNode* q = slow;
            while (p != q) {
                p = p->next;
                q = q->next;
            }
            return p;
        }
    }
    return nullptr;
}
```

```go
package main

type ListNode struct {
    Val  int
    Next *ListNode
}

func detectCycle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            p, q := head, slow
            for p != q {
                p = p.Next
                q = q.Next
            }
            return p
        }
    }
    return nil
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct ListNode {
    val: i32,
    next: Option<Rc<RefCell<ListNode>>>,
}

fn node(v: i32) -> Rc<RefCell<ListNode>> {
    Rc::new(RefCell::new(ListNode { val: v, next: None }))
}

fn next_of(n: &Option<Rc<RefCell<ListNode>>>) -> Option<Rc<RefCell<ListNode>>> {
    n.as_ref().and_then(|x| x.borrow().next.clone())
}

fn ptr_eq(a: &Option<Rc<RefCell<ListNode>>>, b: &Option<Rc<RefCell<ListNode>>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn detect_cycle(
    head: Option<Rc<RefCell<ListNode>>>,
) -> Option<Rc<RefCell<ListNode>>> {
    let mut slow = head.clone();
    let mut fast = head.clone();

    while fast.is_some() && next_of(&fast).is_some() {
        slow = next_of(&slow);
        fast = next_of(&next_of(&fast));
        if ptr_eq(&slow, &fast) {
            let mut p = head.clone();
            let mut q = slow.clone();
            while !ptr_eq(&p, &q) {
                p = next_of(&p);
                q = next_of(&q);
            }
            return p;
        }
    }
    None
}

fn main() {
    // 3 -> 2 -> 0 -> -4 -> back to 2
    let n3 = node(3);
    let n2 = node(2);
    let n0 = node(0);
    let n4 = node(-4);
    n3.borrow_mut().next = Some(n2.clone());
    n2.borrow_mut().next = Some(n0.clone());
    n0.borrow_mut().next = Some(n4.clone());
    n4.borrow_mut().next = Some(n2.clone());

    let ans = detect_cycle(Some(n3));
    match ans {
        Some(x) => println!("{}", x.borrow().val),
        None => println!("null"),
    }
}
```

```javascript
function detectCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p = head, q = slow;
      while (p !== q) {
        p = p.next;
        q = q.next;
      }
      return p;
    }
  }
  return null;
}
```

