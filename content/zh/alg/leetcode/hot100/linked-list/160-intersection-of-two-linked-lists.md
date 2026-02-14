---
title: "Hot100：相交链表（Intersection of Two Linked Lists）双指针换头 O(1) 空间 ACERS 解析"
date: 2026-02-01T16:29:40+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "双指针", "哈希", "LeetCode 160"]
description: "在不破坏链表结构的前提下找到两个单链表的相交起点：双指针“走完 A 再走 B”保证同步，O(m+n) 时间、O(1) 额外空间；含推导、工程迁移与多语言实现。"
keywords: ["Intersection of Two Linked Lists", "相交链表", "two pointers", "switch heads", "O(1) space", "LeetCode 160"]
---

> **副标题 / 摘要**  
> 相交链表的关键不是比较值，而是比较“节点引用/地址”。本文用 ACERS 结构把朴素哈希解法、长度对齐解法与最常用的“双指针换头”模板讲清楚，并给出多语言可运行实现（不修改链表、无环前提）。

- **预计阅读时长**：10~14 分钟  
- **标签**：`Hot100`、`链表`、`双指针`  
- **SEO 关键词**：相交链表, 双指针换头, O(1) 空间, LeetCode 160, Intersection of Two Linked Lists  
- **元描述**：双指针分别走完 A 与 B 后交换起点，保证在 m+n 步内相遇于交点或同时到达 null；O(m+n)/O(1) 且不修改链表结构。  

---

## 目标读者

- 刷 Hot100，希望把链表双指针模板一次性吃透的学习者  
- 经常把“节点值相等”误当作“节点相同”的初中级开发者  
- 需要在工程里处理共享链式结构（共享后缀/共享节点）的工程师

## 背景 / 动机

这题看似简单，但它强迫你分清三个概念：

1. **相交是“共享同一个节点对象/地址”**，不是值相等  
2. 不能破坏结构（不能改 `next`、不能打标记）  
3. 还要高效：把 O(mn) 降到线性

最经典的工程化答案是“双指针换头”：  
它不用额外集合、不需要先算长度，只靠指针走路就能在 `m+n` 步内完成同步。

## 核心概念

| 概念 | 含义 | 备注 |
| --- | --- | --- |
| 节点相同 | 两个指针指向同一块内存/同一个对象 | 语言里通常是引用相等/指针相等 |
| 共享后缀 | 两条链表在某个节点开始共享接下来的所有节点 | 交点之后完全一致 |
| 双指针换头 | 指针走到链尾后跳到另一条链表的头 | 让两指针走过相同总路程 |
| 无环保证 | 题目保证整个结构无环 | 否则需要额外环检测处理 |

---

## A — Algorithm（题目与算法）

### 题目还原

给你两个单链表的头节点 `headA` 和 `headB`，请你找出并返回两个单链表相交的起始节点。  
如果两个链表不存在相交节点，返回 `null`。

补充约束：

- 题目数据保证整个链式结构中不存在环  
- 返回结果后，链表必须保持其原始结构（不允许修改链表）

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| headA | ListNode | 链表 A 的头结点 |
| headB | ListNode | 链表 B 的头结点 |
| 返回 | ListNode / null | 相交起始节点（同一节点对象），或 null |

### 示例 1（图示场景）

```text
A: a1 -> a2 -> c1 -> c2 -> c3
B: b1 -> b2 -> b3 -> c1 -> c2 -> c3

输出: c1（返回节点引用，不是数值）
```

### 示例 2（不相交）

```text
A: 1 -> 2 -> 3
B: 4 -> 5

输出: null
```

---

## 思路推导：从哈希到 O(1) 空间模板

### 朴素思路：集合记录 A 的所有节点

1. 遍历 A，把每个节点地址放入哈希集合  
2. 遍历 B，第一个出现在集合里的节点就是交点  

优点：简单直观、容易写对。  
缺点：额外空间 O(m)。

### O(1) 空间思路 1：先算长度，再对齐起点

1. 计算 A 长度 `m`、B 长度 `n`  
2. 长链表先走 `abs(m-n)` 步，让两指针到尾部的距离相等  
3. 两指针同步前进，第一个相同节点即交点  

这也是 O(1) 空间，但需要两次遍历算长度。

### O(1) 空间思路 2（最常用）：双指针换头

定义两个指针 `pA=headA`，`pB=headB`：

- 每步各走一步  
- 若某指针走到 `null`，就把它切换到另一条链的头（换头）  

直觉理解：  
`pA` 走过的路径是 `A + B`，`pB` 走过的路径是 `B + A`。  
两条路径总长度相同，所以它们会在交点同步（或一起到达 null）。

---

## C — Concepts（核心思想）

### 方法归类

- **链表双指针（Two Pointers on Linked List）**  
- **路径长度对齐（Path Length Alignment）**：用“走完再换头”隐式对齐长度差  
- **不修改结构的引用相等判定**

### 为什么“双指针换头”一定会相遇？

设：

- A 的独有前缀长度为 `a`
- B 的独有前缀长度为 `b`
- 共享后缀长度为 `c`

那么：

- 链表 A 总长 `a + c`
- 链表 B 总长 `b + c`

指针走法：

- `pA` 走 `a+c` 到 null，然后从 B 再走 `b`，恰好到达交点起始  
- `pB` 走 `b+c` 到 null，然后从 A 再走 `a`，也恰好到达交点起始  

因此在不超过 `a+b+c`（也就是 `m+n`）步内，它们要么在交点相等，要么都变成 null（不相交）。

---

## 实践指南 / 步骤

1. 初始化 `pA=headA, pB=headB`  
2. 当 `pA != pB` 时循环：
   - `pA = pA.next`，若 `pA` 为 null 则 `pA = headB`
   - `pB = pB.next`，若 `pB` 为 null 则 `pB = headA`
3. 循环结束返回 `pA`（可能是交点节点，也可能是 null）

Python 可运行示例（保存为 `intersection.py`，运行 `python3 intersection.py`）：

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def get_intersection_node(head_a: ListNode | None, head_b: ListNode | None) -> ListNode | None:
    p, q = head_a, head_b
    while p is not q:
        p = p.next if p else head_b
        q = q.next if q else head_a
    return p


if __name__ == "__main__":
    # Build shared tail: c1 -> c2 -> c3
    c1 = ListNode(8)
    c2 = ListNode(4)
    c3 = ListNode(5)
    c1.next = c2
    c2.next = c3

    # A: a1 -> a2 -> c1
    a1 = ListNode(4)
    a2 = ListNode(1)
    a1.next = a2
    a2.next = c1

    # B: b1 -> b2 -> b3 -> c1
    b1 = ListNode(5)
    b2 = ListNode(6)
    b3 = ListNode(1)
    b1.next = b2
    b2.next = b3
    b3.next = c1

    ans = get_intersection_node(a1, b1)
    print(ans.val if ans else None)  # 8
```

---

## E — Engineering（工程应用）

### 场景 1：版本化流水线的“共享后缀”去重（Python）

**背景**：某些实验/任务流水线以链式节点表示步骤；多个实验可能共享一段相同的后续步骤（共享后缀）。  
**为什么适用**：找到交点就能把公共后缀只执行一次（或做缓存命中）。  

```python
class Step:
    def __init__(self, name):
        self.name = name
        self.next = None


def intersection(a, b):
    p, q = a, b
    while p is not q:
        p = p.next if p else b
        q = q.next if q else a
    return p


if __name__ == "__main__":
    common = Step("train")
    common.next = Step("evaluate")

    a = Step("clean"); a.next = Step("fe"); a.next.next = common
    b = Step("clean_v2"); b.next = common

    hit = intersection(a, b)
    print(hit.name if hit else "none")  # train
```

### 场景 2：避免双重释放（double free）的安全检查（C）

**背景**：在 C 项目里，两条链表如果意外共享节点，分别 free 会导致 double free。  
**为什么适用**：先检测交点，交点之后的共享段只能释放一次，能显著降低崩溃风险。  

```c
struct Node { int v; struct Node* next; };

struct Node* intersection(struct Node* a, struct Node* b) {
    struct Node* p = a;
    struct Node* q = b;
    while (p != q) {
        p = p ? p->next : b;
        q = q ? q->next : a;
    }
    return p; // may be NULL
}
```

### 场景 3：前端操作历史分叉的合并点定位（JavaScript）

**背景**：某些编辑器用链表表示操作历史；分叉后两条历史可能共享一段尾部（例如合并/回放）。  
**为什么适用**：找到相交节点就能定位“从哪里开始共享”，用于 UI 高亮/合并策略。  

```javascript
function intersection(headA, headB) {
  let p = headA, q = headB;
  while (p !== q) {
    p = p ? p.next : headB;
    q = q ? q.next : headA;
  }
  return p;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(m+n)  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 备注 |
| --- | --- | --- | --- | --- |
| 哈希集合 | 存 A 节点集合，扫 B 命中 | O(m+n) | O(m) | 最直观 |
| 长度对齐 | 算长度差，长链先走 | O(m+n) | O(1) | 需要先遍历算长度 |
| **双指针换头** | A 走完接 B，B 走完接 A | **O(m+n)** | **O(1)** | 模板最简洁 |

### 常见坑

1. **把“val 相等”当作相交**：相交必须是节点引用相同（同一对象/地址）。  
2. **忘记处理 null**：循环条件要用“指针相等”退出，最终可能返回 null。  
3. **结构有环时直接套模板**：题目保证无环；若可能有环，需要先做环检测（否则可能死循环）。

---

## 常见问题与注意事项

1. **为什么两指针不会无限循环？**  
   无环前提下，每个指针最多走 `m+n` 步就会到达“交点或 null”，循环必然结束。

2. **如果两个链表完全相同（headA == headB）？**  
   一开始就相等，直接返回 headA。

3. **能不能在节点上打标记（比如改 val）？**  
   不建议也不允许：题目要求保持原结构；工程里修改共享结构会带来副作用。

---

## 最佳实践与建议

- 把“双指针换头”记成模板：`p = p ? p->next : headB` / `q = q ? q->next : headA`  
- 任何“相交/共享尾部”问题，第一反应先确认：比较的是引用还是值  
- 工程中若结构可能有环：先用 Floyd 判环，再讨论交点（问题会更复杂）

---

## S — Summary（总结）

### 核心收获

- 相交链表的“相交”是节点引用相同，不是值相等  
- 朴素哈希法易写但占内存；长度对齐法 O(1) 但要先算长度  
- **双指针换头**用“走过相同总路程”隐式对齐长度差，O(m+n)/O(1) 且不修改结构  
- 无环保证是算法终止与正确性的基础前提  
- 该模板可迁移到任何“共享后缀/合并点/共同尾部”结构

### 参考与延伸阅读

- LeetCode 160. Intersection of Two Linked Lists
- Two pointers on linked list 的经典题型：找环、找中点、删除倒数第 k 个
- 工程中的共享结构与引用相等（pointer identity）基础概念

---

## 元信息

- **阅读时长**：10~14 分钟  
- **标签**：Hot100、链表、双指针、空间优化  
- **SEO 关键词**：相交链表, Intersection of Two Linked Lists, 双指针换头, O(1) 空间  
- **元描述**：双指针分别走完 A 与 B 后交换起点，保证在 m+n 步内相遇于交点或同时到达 null；O(m+n)/O(1)，不修改链表结构。  

---

## 行动号召（CTA）

建议你用同一套思路再做两题巩固：  
1) 链表判环（Floyd）；2) 删除倒数第 N 个节点（快慢指针）。  
如果你希望我把“可能有环时如何判断相交”的进阶版本也写一篇补充文，留言我就继续写。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from __future__ import annotations


class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: ListNode | None = None


def get_intersection_node(head_a: ListNode | None, head_b: ListNode | None) -> ListNode | None:
    p, q = head_a, head_b
    while p is not q:
        p = p.next if p else head_b
        q = q.next if q else head_a
    return p
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* getIntersectionNode(struct ListNode* headA, struct ListNode* headB) {
    struct ListNode* p = headA;
    struct ListNode* q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}

static struct ListNode* node(int v) {
    struct ListNode* n = (struct ListNode*)malloc(sizeof(struct ListNode));
    n->val = v;
    n->next = NULL;
    return n;
}

int main(void) {
    // shared: c1(8) -> c2(4) -> c3(5)
    struct ListNode* c1 = node(8);
    struct ListNode* c2 = node(4);
    struct ListNode* c3 = node(5);
    c1->next = c2;
    c2->next = c3;

    // A: 4 -> 1 -> c1
    struct ListNode* a1 = node(4);
    struct ListNode* a2 = node(1);
    a1->next = a2;
    a2->next = c1;

    // B: 5 -> 6 -> 1 -> c1
    struct ListNode* b1 = node(5);
    struct ListNode* b2 = node(6);
    struct ListNode* b3 = node(1);
    b1->next = b2;
    b2->next = b3;
    b3->next = c1;

    struct ListNode* ans = getIntersectionNode(a1, b1);
    if (ans) printf("%d\n", ans->val); else printf("null\n");

    // In real code, you must free nodes carefully: shared suffix should be freed once.
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

ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    ListNode* p = headA;
    ListNode* q = headB;
    while (p != q) {
        p = p ? p->next : headB;
        q = q ? q->next : headA;
    }
    return p;
}

int main() {
    // shared: c1 -> c2 -> c3
    auto* c1 = new ListNode(8);
    auto* c2 = new ListNode(4);
    auto* c3 = new ListNode(5);
    c1->next = c2; c2->next = c3;

    // A: 4 -> 1 -> c1
    auto* a1 = new ListNode(4);
    auto* a2 = new ListNode(1);
    a1->next = a2; a2->next = c1;

    // B: 5 -> 6 -> 1 -> c1
    auto* b1 = new ListNode(5);
    auto* b2 = new ListNode(6);
    auto* b3 = new ListNode(1);
    b1->next = b2; b2->next = b3; b3->next = c1;

    ListNode* ans = getIntersectionNode(a1, b1);
    std::cout << (ans ? std::to_string(ans->val) : std::string("null")) << "\n";

    // Demo only: free omitted.
    return 0;
}
```

```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
    p, q := headA, headB
    for p != q {
        if p == nil {
            p = headB
        } else {
            p = p.Next
        }
        if q == nil {
            q = headA
        } else {
            q = q.Next
        }
    }
    return p
}

func main() {
    // shared: c1(8) -> c2(4) -> c3(5)
    c3 := &ListNode{Val: 5}
    c2 := &ListNode{Val: 4, Next: c3}
    c1 := &ListNode{Val: 8, Next: c2}

    // A: 4 -> 1 -> c1
    a := &ListNode{Val: 4, Next: &ListNode{Val: 1, Next: c1}}

    // B: 5 -> 6 -> 1 -> c1
    b := &ListNode{Val: 5, Next: &ListNode{Val: 6, Next: &ListNode{Val: 1, Next: c1}}}

    ans := getIntersectionNode(a, b)
    if ans != nil {
        fmt.Println(ans.Val)
    } else {
        fmt.Println("null")
    }
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

fn node(val: i32) -> Rc<RefCell<ListNode>> {
    Rc::new(RefCell::new(ListNode { val, next: None }))
}

fn same(a: &Option<Rc<RefCell<ListNode>>>, b: &Option<Rc<RefCell<ListNode>>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => Rc::ptr_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn get_intersection_node(
    head_a: Option<Rc<RefCell<ListNode>>>,
    head_b: Option<Rc<RefCell<ListNode>>>,
) -> Option<Rc<RefCell<ListNode>>> {
    let mut p = head_a.clone();
    let mut q = head_b.clone();
    while !same(&p, &q) {
        p = if let Some(n) = p {
            n.borrow().next.clone()
        } else {
            head_b.clone()
        };
        q = if let Some(n) = q {
            n.borrow().next.clone()
        } else {
            head_a.clone()
        };
    }
    p
}

fn main() {
    // shared: c1(8) -> c2(4) -> c3(5)
    let c1 = node(8);
    let c2 = node(4);
    let c3 = node(5);
    c1.borrow_mut().next = Some(c2.clone());
    c2.borrow_mut().next = Some(c3.clone());

    // A: 4 -> 1 -> c1
    let a1 = node(4);
    let a2 = node(1);
    a1.borrow_mut().next = Some(a2.clone());
    a2.borrow_mut().next = Some(c1.clone());

    // B: 5 -> 6 -> 1 -> c1
    let b1 = node(5);
    let b2 = node(6);
    let b3 = node(1);
    b1.borrow_mut().next = Some(b2.clone());
    b2.borrow_mut().next = Some(b3.clone());
    b3.borrow_mut().next = Some(c1.clone());

    let ans = get_intersection_node(Some(a1), Some(b1));
    match ans {
        Some(n) => println!("{}", n.borrow().val),
        None => println!("null"),
    }
}
```

```javascript
class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

function getIntersectionNode(headA, headB) {
  let p = headA, q = headB;
  while (p !== q) {
    p = p ? p.next : headB;
    q = q ? q.next : headA;
  }
  return p;
}

// demo
const c1 = new ListNode(8);
const c2 = new ListNode(4);
const c3 = new ListNode(5);
c1.next = c2; c2.next = c3;

const a1 = new ListNode(4);
const a2 = new ListNode(1);
a1.next = a2; a2.next = c1;

const b1 = new ListNode(5);
const b2 = new ListNode(6);
const b3 = new ListNode(1);
b1.next = b2; b2.next = b3; b3.next = c1;

const ans = getIntersectionNode(a1, b1);
console.log(ans ? ans.val : null);
```

