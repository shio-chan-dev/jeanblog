---
title: "Hot100：环形链表（Linked List Cycle）Floyd 快慢指针 ACERS 解析"
date: 2026-02-01T18:45:44+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "双指针", "快慢指针", "Floyd", "LeetCode 141"]
description: "用 Floyd 快慢指针在 O(n) 时间、O(1) 额外空间判断单链表是否有环，并对比哈希表方案、常见坑与工程迁移，附多语言可运行实现（Hot100）。"
keywords: ["Linked List Cycle", "环形链表", "快慢指针", "Floyd", "判环", "LeetCode 141", "Hot100", "O(1)"]
---

> **副标题 / 摘要**  
> 判断链表是否有环，本质是“指针追及问题”。本文用 ACERS 结构讲透 Floyd 快慢指针判环：为什么一定能相遇、如何避免空指针、以及在工程里如何用同一思想识别循环引用/路由环路。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`链表`、`快慢指针`  
- **SEO 关键词**：Hot100, Linked List Cycle, 环形链表, 判环, Floyd, 快慢指针, LeetCode 141  
- **元描述**：Floyd 快慢指针 O(n)/O(1) 判断单链表是否有环，附替代方案对比、易错点与多语言实现。  

---

## 目标读者

- 正在刷 Hot100 / 准备面试的同学  
- 想把“链表双指针”沉淀成稳定模板的中级开发者  
- 在工程里需要识别循环引用、链式结构异常的同学（C/C++/Go/Rust/JS 皆适用）

## 背景 / 动机

链表出现环在工程里并不罕见：  
例如手写内存池的 free list、对象引用链、状态机/任务编排的 next 指针、配置链路的“下一跳”等。

一旦出现环：

- 遍历会进入死循环（CPU 占用飙高，日志刷爆）  
- 资源释放/回收会卡死（例如释放链表节点时无限循环）  
- 监控定位困难（看起来像“偶发卡死”，本质是结构性错误）

因此你需要一个**不依赖额外内存、可在线检测**的判环方法：Floyd 快慢指针就是这类问题的标准答案。

## 核心概念

- **环（Cycle）**：从某个节点开始，沿 `next` 指针走若干步能回到自己  
- **pos（评测用）**：题目描述里的 `pos` 仅用于评测系统构造数据；你的函数不会收到 `pos` 参数  
- **快慢指针（Floyd）**：慢指针每次走 1 步，快指针每次走 2 步；若存在环，二者必定在环内相遇  
- **指针相等 vs 值相等**：判环必须比较“节点身份”（引用/地址），不能只比 `val`（值可能重复）

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个链表的头节点 `head`，判断链表中是否有环。  
如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。

为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。  
注意：`pos` **不作为参数进行传递**，它只是用于标识链表的实际情况。

若存在环返回 `true`，否则返回 `false`。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头节点（可能为空） |
| 返回 | bool | 是否存在环 |

### 示例 1（自拟）

```text
head: 3 -> 2 -> 0 -> -4
               ^     |
               |_____|
输出: true
```

### 示例 2（自拟）

```text
head: 1 -> 2 -> null
输出: false
```

---

## C — Concepts（核心思想）

### 思路推导：从“记录访问过的节点”到“追及相遇”

1. **直观方案：哈希表记录 visited**  
   遍历链表，把每个节点的“身份”（引用/地址）放入集合；  
   - 再次遇到同一个节点 ⇒ 有环  
   - 走到 `null` ⇒ 无环  
   这很直观，但需要 O(n) 额外空间。

2. **关键观察：如果有环，快指针会在环内追上慢指针**  
   进入环后，快指针每轮比慢指针多走 1 步（2 - 1 = 1），相当于在环上做“追及”。  
   环的长度是有限的，因此“距离差”会在模环长意义下不断变化，最终变为 0 —— 两者相遇。

3. **方法选择：Floyd 判环（O(1) 额外空间）**  
   用两个指针：  
   - `slow = slow.next`  
   - `fast = fast.next.next`  
   若 `fast` 或 `fast.next` 变为 `null` ⇒ 无环；  
   若 `slow == fast`（同一节点）⇒ 有环。

### 方法归类

- **双指针（Two Pointers）**  
- **Floyd Cycle Detection（龟兔赛跑）**  
- **在线检测（Streaming / Online Check）**

### 为什么一定会相遇（直觉版证明）

进入环后，把环看成长度为 `L` 的跑道：

- 慢指针每轮前进 1 格  
- 快指针每轮前进 2 格  

因此快指针相对慢指针每轮“追近 1 格”。  
假设某一时刻它们在环上的相对距离为 `d`（0 ≤ d < L），每轮后变成 `(d - 1) mod L`。  
连续做 L 轮后，总会出现 d = 0，即相遇。

---

## 实践指南 / 步骤

1. 若 `head == null` 直接返回 `false`  
2. 初始化：`slow = head`，`fast = head`  
3. 循环：  
   - 先判断 `fast` 和 `fast.next` 是否为 `null`；若是，返回 `false`  
   - `slow = slow.next`  
   - `fast = fast.next.next`  
   - 若 `slow == fast`，返回 `true`  
4. 理论上循环一定会返回（无环会遇到 `null`，有环会相遇）

Python 可运行示例（保存为 `linked_list_cycle.py`）：

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def build_list(values: List[int], pos: int) -> Optional[ListNode]:
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


if __name__ == "__main__":
    head1 = build_list([3, 2, 0, -4], pos=1)
    print(has_cycle(head1))  # True
    head2 = build_list([1, 2], pos=-1)
    print(has_cycle(head2))  # False
```

---

## E — Engineering（工程应用）

> 判环不是“只为刷题”。它是一个非常通用的安全检查：  
> 在任何“单向 next 指针链”里，你都可以用 Floyd 来做 O(1) 额外空间的结构体检。

### 场景 1：内存池 free list 自检，避免回收链表成环（C）

**背景**：手写内存池或对象池时，空闲块常用单链表串起来（free list）。  
**为什么适用**：一旦 free list 成环，分配/释放可能卡死；用 Floyd 能在不分配额外内存的情况下做快速自检（尤其适合资源受限环境）。

```c
int has_cycle(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1;
    }
    return 0;
}
```

### 场景 2：后端任务编排 next 链路的健康检查（Go）

**背景**：有些系统用 next 指针把步骤串成“任务链”（例如简化版状态机/工作流）。  
**为什么适用**：配置/代码 bug 可能让 next 指向前面的步骤形成环，导致执行永不结束；Floyd 能在线检测并快速失败。

```go
func hasCycle(head *Node) bool {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true
        }
    }
    return false
}
```

### 场景 3：前端/脚本链式对象的循环引用检测（JavaScript）

**背景**：有时你会用 `next` 字段把对象串成链（如导航、节点关系、简化 AST 结构）。  
**为什么适用**：在调试/校验阶段，你希望快速发现循环引用，避免遍历卡死；Floyd 不需要额外集合（当然 JS 里用 Set 也很常见）。

```javascript
function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)  
  - 无环：快指针走到 `null` 结束  
  - 有环：进入环后至多 O(L) 轮相遇（L 为环长），整体仍是线性级别
- **空间复杂度**：O(1)（Floyd）；哈希表方案为 O(n)

### 替代方案对比

| 方法 | 思路 | 额外空间 | 典型问题 |
| --- | --- | --- | --- |
| visited 集合 | 记录访问过的节点 | O(n) | 内存开销大；在 C/嵌入式里不方便 |
| 修改节点标记 | 在节点上写标记位 | O(1) | 破坏数据结构；题目/工程未必允许 |
| **Floyd（本文）** | 快慢指针追及相遇 | **O(1)** | 必须小心 `fast.next` 的空指针判断 |

### 常见错误与注意事项（面试高频）

1. **用 `val` 判断重复**：值可能重复，必须比较节点引用/地址。  
2. **忘记判空**：`fast = fast.next.next` 前必须确保 `fast` 和 `fast.next` 非空。  
3. **把 `slow == fast` 写在更新前**：初始时二者都指向 head，会误判；应在移动后比较，或初始化为不同起点。  
4. **以为 `pos` 会传入函数**：不会。`pos` 只是评测系统用于构造用例。

---

## 解释与原理（为什么这么做）

把链表分成两段理解：

- **非环前缀**：从 head 出发到进入环之前的那段（可能为空）  
- **环**：进入环后会在环内循环

快慢指针在前缀段最多 O(前缀长度) 轮就会进入环。  
一旦两者都进入环，快指针每轮比慢指针多走 1 步，所以相对距离会不断变化，最终相遇。  
这就是为什么 Floyd 判环既省内存又可靠。

---

## 常见问题与注意事项

1. **链表非常长会超时吗？**  
   Floyd 是 O(n)，只要遍历一次量级，通常是最稳妥的选择。

2. **能不能顺便找环的入口？**  
   可以（LeetCode 142）。本题只要判断是否有环，返回 bool 即可。

3. **工程里什么时候用 Set 更合适？**  
   当你需要记录路径、输出环上节点、或节点不是“单向 next 链”而是一般图结构时，Set/Map 往往更直接。

---

## 最佳实践与建议

- 牢记循环条件：`while fast != null && fast.next != null`  
- 比较节点身份：Python 用 `is`，JS 用 `===`，C/C++/Go 用指针相等  
- 不要依赖 `pos`：它只存在于评测系统  
- 需要定位入口/环长时，在 Floyd 相遇基础上再扩展（142/环长计算）

---

## S — Summary（总结）

### 核心收获

- 判环必须比较“节点身份”，不能比较值  
- Floyd 快慢指针用 O(1) 额外空间完成判环  
- 进入环后快指针相对慢指针每轮追近 1 步，因此必相遇  
- 判空是实现的生命线：`fast` 与 `fast.next` 都要检查  

### 小结 / 结论

LeetCode 141 是链表双指针的入门模板题。  
把它写熟，你会在很多“链式结构体检”场景里直接复用这段逻辑。

### 参考与延伸阅读

- LeetCode 141. Linked List Cycle  
- LeetCode 142. Linked List Cycle II（找环入口）  
- Floyd Cycle Detection（龟兔赛跑）经典证明与变体

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、链表、快慢指针、LeetCode 141  
- **SEO 关键词**：Hot100, Linked List Cycle, 环形链表, 判环, Floyd, 快慢指针  
- **元描述**：Floyd 快慢指针 O(n)/O(1) 判断单链表是否有环，附替代方案对比与多语言实现。  

---

## 行动号召（CTA）

建议你把今天学到的模板立刻用在两道延伸题上：

1) LeetCode 142：在“相遇”基础上找环入口  
2) 任意链表题里加入 debug 断言：关键链表操作后跑一次判环自检（排查断链/误连）

如果你希望我把 142 也按 Hot100 的 ACERS 风格写出来，告诉我即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def hasCycle(head: Optional[ListNode]) -> bool:
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def build(values: List[int], pos: int) -> Optional[ListNode]:
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]
    return nodes[0]


if __name__ == "__main__":
    print(hasCycle(build([3, 2, 0, -4], 1)))  # True
    print(hasCycle(build([1, 2], -1)))        # False
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

int hasCycle(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1;
    }
    return 0;
}

int main(void) {
    struct ListNode* nodes[4];
    for (int i = 0; i < 4; ++i) {
        nodes[i] = (struct ListNode*)malloc(sizeof(struct ListNode));
        nodes[i]->val = i;
        nodes[i]->next = NULL;
    }
    nodes[0]->next = nodes[1];
    nodes[1]->next = nodes[2];
    nodes[2]->next = nodes[3];
    nodes[3]->next = nodes[1]; /* cycle */

    printf("%d\n", hasCycle(nodes[0])); /* 1 */

    nodes[3]->next = NULL; /* break cycle before free */
    for (int i = 0; i < 4; ++i) free(nodes[i]);
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int v) : val(v), next(nullptr) {}
};

bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}

int main() {
    std::vector<ListNode*> nodes;
    for (int i = 0; i < 4; ++i) nodes.push_back(new ListNode(i));
    nodes[0]->next = nodes[1];
    nodes[1]->next = nodes[2];
    nodes[2]->next = nodes[3];
    nodes[3]->next = nodes[1]; // cycle

    std::cout << std::boolalpha << hasCycle(nodes[0]) << "\n";

    nodes[3]->next = nullptr; // break cycle
    for (auto* p : nodes) delete p;
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

func hasCycle(head *ListNode) bool {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true
        }
    }
    return false
}

func main() {
    a := &ListNode{Val: 1}
    b := &ListNode{Val: 2}
    c := &ListNode{Val: 3}
    d := &ListNode{Val: 4}
    a.Next = b
    b.Next = c
    c.Next = d
    d.Next = b // cycle
    fmt.Println(hasCycle(a)) // true
    d.Next = nil
}
```

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    val: i32,
    next: Option<Rc<RefCell<Node>>>,
}

fn next(node: &Option<Rc<RefCell<Node>>>) -> Option<Rc<RefCell<Node>>> {
    node.as_ref().and_then(|rc| rc.borrow().next.clone())
}

fn has_cycle(head: Option<Rc<RefCell<Node>>>) -> bool {
    let mut slow = head.clone();
    let mut fast = head;

    loop {
        slow = next(&slow);
        fast = next(&fast);
        if fast.is_none() || slow.is_none() {
            return false;
        }
        fast = next(&fast);
        if fast.is_none() {
            return false;
        }

        if let (Some(ref s), Some(ref f)) = (&slow, &fast) {
            if Rc::ptr_eq(s, f) {
                return true;
            }
        } else {
            return false;
        }
    }
}

fn main() {
    let a = Rc::new(RefCell::new(Node { val: 1, next: None }));
    let b = Rc::new(RefCell::new(Node { val: 2, next: None }));
    let c = Rc::new(RefCell::new(Node { val: 3, next: None }));

    a.borrow_mut().next = Some(b.clone());
    b.borrow_mut().next = Some(c.clone());
    c.borrow_mut().next = Some(b.clone()); // cycle

    println!("{}", has_cycle(Some(a)));
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}

const a = new ListNode(1);
const b = new ListNode(2);
const c = new ListNode(3);
a.next = b;
b.next = c;
c.next = b; // cycle
console.log(hasCycle(a));
```
