---
title: "Hot100：反转链表（Reverse Linked List）三指针迭代/递归 ACERS 解析"
date: 2026-02-01T16:30:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "指针", "迭代", "递归", "LeetCode 206"]
description: "用三指针迭代在 O(n) 时间、O(1) 额外空间反转单链表，并对比递归写法与常见坑，附多语言可运行实现（Hot100）。"
keywords: ["Reverse Linked List", "反转链表", "三指针", "迭代", "递归", "LeetCode 206", "Hot100", "O(n)"]
---

> **副标题 / 摘要**  
> 反转链表是“指针重连”的入门必修课：看似简单，却最容易因为边界、断链、顺序写错而翻车。本文用 ACERS 结构把三指针迭代写法讲透，并给出递归对照与多语言可运行实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`链表`、`指针`、`迭代`  
- **SEO 关键词**：Hot100, Reverse Linked List, 反转链表, 三指针, 迭代, 递归, LeetCode 206  
- **元描述**：三指针迭代 O(n)/O(1) 反转单链表，附递归对比、工程迁移与多语言实现。  

---

## 目标读者

- 正在刷 Hot100 / 准备面试的同学  
- 写链表题经常断链/空指针、希望建立稳定模板的中级开发者  
- 需要在 C/C++/Rust 等语言里熟练处理指针与所有权的工程师

## 背景 / 动机

在真实工程里，“反转链表”不一定以 LeetCode 的形态出现，但它背后的能力非常通用：

- 你要在 **O(1) 额外空间** 下重排节点顺序（例如复用节点对象，避免额外分配）  
- 你要理解 **指针重连的顺序**：先保留 `next`，再改 `cur.next`，否则就会断链  
- 你要能写出 **不会特判地狱**、对 `head = null` 也稳的实现

把这题做成模板后，很多链表题（如反转区间、k 组反转、判断回文链表）都会变得顺手很多。

## 核心概念

- **单链表**：每个节点只有一个 `next` 指针指向后继  
- **断链风险**：一旦把 `cur.next` 改掉而没保存原来的 next，就丢失后半段  
- **三指针（prev / cur / next）**：用 `next` 暂存后继，再把 `cur.next` 指向 `prev`  
- **循环不变量**：`prev` 永远指向“已反转部分”的头；`cur` 永远指向“未处理部分”的头

---

## A — Algorithm（题目与算法）

### 题目还原

给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头节点（可能为空） |
| 返回 | ListNode | 反转后的新头节点 |

### 示例 1（自拟）

```text
输入: 1 -> 2 -> 3 -> 4 -> 5 -> null
输出: 5 -> 4 -> 3 -> 2 -> 1 -> null
```

### 示例 2（自拟）

```text
输入: 1 -> 2 -> null
输出: 2 -> 1 -> null
```

---

## C — Concepts（核心思想）

### 思路推导：从“重新建链”到“原地指针反转”

1. **朴素想法：把值拷贝出来再重建链表**  
   - 先遍历把值存到数组  
   - 再从后往前新建节点串起来  
   缺点：需要 O(n) 额外空间，而且“重建节点”在工程里通常意味着额外分配与 GC/内存碎片。

2. **关键观察：反转只是在重连 next 指针**  
   对于当前节点 `cur`，我们希望把：

```text
prev <- cur -> next
```

   变成：

```text
prev <- cur    next(待处理)
```

   本质操作就是：

```text
cur.next = prev
```

   但在做这句之前，必须先把原来的 `cur.next` 保存下来，否则链表后半段会丢失。

3. **方法选择：三指针迭代（O(1) 额外空间）**

### 方法归类

- **链表原地操作（In-place Linked List Manipulation）**  
- **迭代模拟（Iterative Simulation）**  
- **递归（Recursion）作为等价写法对照**

### 迭代版本的循环不变量

在每次循环开始时保持：

- `prev` 指向已经反转好的链表头（初始为 `null`）  
- `cur` 指向尚未处理的链表头（初始为 `head`）  

循环体做三步：

1. `next = cur.next`（保存后继，防断链）  
2. `cur.next = prev`（反转指针）  
3. `prev = cur; cur = next`（整体前进）  

当 `cur == null` 时，所有节点处理完毕，`prev` 就是新头节点。

### 递归版本（对照理解）

递归的核心是把问题拆成：

- 先反转 `head.next` 之后的链表，拿到新头 `newHead`  
- 再把 `head.next.next = head` 让第二个节点指回 head  
- 最后把 `head.next = null` 断开旧指针，避免成环  

递归写法更“优雅”，但会用到函数调用栈（空间不是 O(1)）。

---

## 实践指南 / 步骤

### 迭代三指针（推荐模板）

1. 初始化 `prev = null, cur = head`  
2. 循环直到 `cur == null`：  
   - 先保存：`next = cur.next`  
   - 再反转：`cur.next = prev`  
   - 再推进：`prev = cur; cur = next`  
3. 返回 `prev`

Python 可运行示例（保存为 `reverse_list.py`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    prev = None
    cur = head
    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def from_list(a):
    dummy = ListNode()
    tail = dummy
    for x in a:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head):
    res = []
    while head is not None:
        res.append(head.val)
        head = head.next
    return res


if __name__ == "__main__":
    head = from_list([1, 2, 3, 4, 5])
    print(to_list(reverse_list(head)))
```

---

## E — Engineering（工程应用）

> 反转链表最重要的工程迁移价值，是“**原地重连指针**”这一能力：  
> 你不依赖额外容器，也不创建新节点，只改变链接关系——这在性能敏感或内存受限场景尤其常见。

### 场景 1：内存池/空闲链表（free list）调整分配策略（C）

**背景**：很多嵌入式/高性能系统用单链表维护空闲块（free list）。  
**为什么适用**：你可能希望把 free list 的顺序翻转，以改变“优先复用最近释放的块”或“更均匀地复用块”的策略（具体策略取决于系统设计）。反转可以在 O(1) 额外空间完成。

```c
#include <stdio.h>

typedef struct Node {
    int id;
    struct Node* next;
} Node;

Node* reverse(Node* head) {
    Node* prev = NULL;
    Node* cur = head;
    while (cur) {
        Node* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

int main(void) {
    Node c = {3, NULL};
    Node b = {2, &c};
    Node a = {1, &b};
    Node* head = reverse(&a);
    for (Node* p = head; p; p = p->next) printf("%d ", p->id);
    printf("\n");
    return 0;
}
```

### 场景 2：服务端任务链（单链表）做 LIFO/回放顺序切换（Go）

**背景**：在一些简化实现里，你可能用单链表当作栈（stack）来记录任务或操作序列。  
**为什么适用**：当你需要把“后进先出（LIFO）”的记录变成“先进先出（FIFO）”的回放顺序时，反转链表是最直接的原地变换。

```go
package main

import "fmt"

type Node struct {
    Val  int
    Next *Node
}

func reverse(head *Node) *Node {
    var prev *Node
    cur := head
    for cur != nil {
        nxt := cur.Next
        cur.Next = prev
        prev = cur
        cur = nxt
    }
    return prev
}

func main() {
    head := &Node{1, &Node{2, &Node{3, nil}}}
    head = reverse(head)
    for p := head; p != nil; p = p.Next {
        fmt.Print(p.Val, " ")
    }
    fmt.Println()
}
```

### 场景 3：前端数据结构教学/可视化（JavaScript）

**背景**：在前端做算法教学、可视化动画时，经常会用 JS 对象模拟链表节点。  
**为什么适用**：反转链表能展示“引用指向变化”，非常适合做一步步的动画演示（保存 `prev/cur/next` 的变化轨迹即可）。

```javascript
function Node(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverse(head) {
  let prev = null;
  let cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}

let head = new Node(1, new Node(2, new Node(3)));
head = reverse(head);
const res = [];
for (let p = head; p; p = p.next) res.push(p.val);
console.log(res.join(" "));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)，每个节点只处理一次  
- **空间复杂度**：O(1)（迭代版）；递归版为 O(n)（调用栈）

### 替代方案对比

| 方法 | 思路 | 额外空间 | 典型问题 |
| --- | --- | --- | --- |
| 新建链表 | 值拷贝 + 重建节点 | O(n) | 额外分配/GC/碎片；不满足“原地” |
| 递归反转 | 反转子链表再回连 | O(n) | 栈深风险；在部分语言里容易栈溢出 |
| **迭代三指针（本文）** | prev/cur/next 重连 | **O(1)** | 需要严格顺序，避免断链 |

### 常见错误与注意事项（高频翻车点）

1. **先改 `cur.next` 再保存 next**：后半段会丢失（断链）。  
2. **忘记推进 `cur`**：死循环。  
3. **递归忘记 `head.next = null`**：容易形成环。  
4. **误以为要交换值**：题目要求反转链表结构（节点指向），不是仅反转值序列（面试常追问）。

### 为什么迭代更工程可行

- 不依赖递归栈：对超长链表更稳  
- 逻辑是“局部可验证”的三步：保存、反转、推进  
- 更适合低层语言（C/C++/Rust）与高性能场景

---

## 解释与原理（为什么这么做）

把链表写成：

```text
prev -> ... (已反转部分)
cur  -> ... (未处理部分)
```

每次循环只做一件事：把 `cur` 从“未处理部分”的开头摘出来，接到“已反转部分”的前面。

关键顺序就是：

1. 先记住 `nxt = cur.next`（否则你再也找不到未处理部分）  
2. 再执行 `cur.next = prev`（完成“接到前面”）  
3. 最后整体推进：`prev = cur; cur = nxt`

当 `cur` 走到 `null`，说明“未处理部分为空”，此时 `prev` 就是整个反转链表的头。

---

## 常见问题与注意事项

1. **空链表 / 单节点链表怎么办？**  
   迭代模板天然支持：`head == null` 返回 `null`；单节点反转还是它自己。

2. **是否必须用三指针？**  
   本质上必须保存后继，所以你至少要有一个变量保存 `next`（不一定叫“三指针”，但状态等价）。

3. **递归更短，为什么还推荐迭代？**  
   递归会占用调用栈，长链表可能导致栈溢出；迭代更稳定，更接近工程代码。

4. **怎么保证不丢节点？**  
   自检口诀：**先存 next，再改 next，最后移动指针**。

---

## 最佳实践与建议

- 把迭代三步写成肌肉记忆：`nxt = cur.next` → `cur.next = prev` → `prev = cur; cur = nxt`  
- 画一遍指针图再写代码，能显著减少 bug  
- 递归写法当作“理解指针回连”的练习即可，工程上默认迭代

---

## S — Summary（总结）

### 核心收获

- 反转链表的本质是 **next 指针重连**，不是值交换  
- 三指针迭代能在 **O(n) 时间、O(1) 额外空间** 完成反转  
- 正确顺序是：**先保存 next，再反转指针，再推进**  
- 递归写法可读但占用栈空间，工程上更偏向迭代  

### 小结 / 结论

把这题的迭代模板写熟，你就拥有了“链表指针操作”的基本功。  
后续很多题（反转区间、k 组反转、回文链表）本质都在这个模板上做局部改造。

### 参考与延伸阅读

- LeetCode 206. Reverse Linked List  
- LeetCode 92. Reverse Linked List II（区间反转）  
- LeetCode 25. Reverse Nodes in k-Group（k 组反转）  
- LeetCode 234. Palindrome Linked List（快慢指针 + 反转后半段）  

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、链表、指针、迭代、LeetCode 206  
- **SEO 关键词**：Hot100, Reverse Linked List, 反转链表, 三指针, 迭代, 递归, LeetCode 206  
- **元描述**：三指针迭代 O(n)/O(1) 反转单链表，附递归对比、工程迁移与多语言实现。  

---

## 行动号召（CTA）

建议你做两件事来彻底掌握这题：

1) 不看代码，自己手画 `prev/cur/nxt` 的指针变化（至少画 3 步）  
2) 再去做 LeetCode 92（区间反转）——你会发现它就是“在局部套用同一模板”

如果你希望我把 92 / 25 也按 Hot100 的 ACERS 风格写出来，留言告诉我。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    cur = head
    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def from_list(a: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in a:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    res: List[int] = []
    while head is not None:
        res.append(head.val)
        head = head.next
    return res


if __name__ == "__main__":
    head = from_list([1, 2, 3, 4, 5])
    print(to_list(reverseList(head)))
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* reverseList(struct ListNode* head) {
    struct ListNode* prev = NULL;
    struct ListNode* cur = head;
    while (cur) {
        struct ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

static struct ListNode* push_front(struct ListNode* head, int val) {
    struct ListNode* node = (struct ListNode*)malloc(sizeof(struct ListNode));
    node->val = val;
    node->next = head;
    return node;
}

static void free_list(struct ListNode* head) {
    while (head) {
        struct ListNode* nxt = head->next;
        free(head);
        head = nxt;
    }
}

static void print_list(struct ListNode* head) {
    for (struct ListNode* p = head; p; p = p->next) {
        if (p != head) printf(" -> ");
        printf("%d", p->val);
    }
    printf(" -> null\n");
}

int main(void) {
    struct ListNode* head = NULL;
    head = push_front(head, 5);
    head = push_front(head, 4);
    head = push_front(head, 3);
    head = push_front(head, 2);
    head = push_front(head, 1);
    print_list(head);
    head = reverseList(head);
    print_list(head);
    free_list(head);
    return 0;
}
```

```cpp
#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* cur = head;
    while (cur) {
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

ListNode* fromVec(const std::vector<int>& a) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int x : a) {
        tail->next = new ListNode(x);
        tail = tail->next;
    }
    return dummy.next;
}

void freeList(ListNode* head) {
    while (head) {
        ListNode* nxt = head->next;
        delete head;
        head = nxt;
    }
}

void printList(ListNode* head) {
    for (ListNode* p = head; p; p = p->next) {
        std::cout << p->val << (p->next ? " -> " : " -> null\n");
    }
    if (!head) std::cout << "null\n";
}

int main() {
    ListNode* head = fromVec({1, 2, 3, 4, 5});
    printList(head);
    head = reverseList(head);
    printList(head);
    freeList(head);
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

func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    cur := head
    for cur != nil {
        nxt := cur.Next
        cur.Next = prev
        prev = cur
        cur = nxt
    }
    return prev
}

func fromSlice(a []int) *ListNode {
    dummy := &ListNode{}
    tail := dummy
    for _, x := range a {
        tail.Next = &ListNode{Val: x}
        tail = tail.Next
    }
    return dummy.Next
}

func toSlice(head *ListNode) []int {
    res := []int{}
    for head != nil {
        res = append(res, head.Val)
        head = head.Next
    }
    return res
}

func main() {
    head := fromSlice([]int{1, 2, 3, 4, 5})
    fmt.Println(toSlice(reverseList(head)))
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    pub fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut prev: Option<Box<ListNode>> = None;
    let mut cur = head;

    while let Some(mut node) = cur {
        let nxt = node.next.take();
        node.next = prev;
        prev = Some(node);
        cur = nxt;
    }
    prev
}

fn from_vec(a: &[i32]) -> Option<Box<ListNode>> {
    let mut head: Option<Box<ListNode>> = None;
    for &x in a.iter().rev() {
        let mut node = Box::new(ListNode::new(x));
        node.next = head;
        head = Some(node);
    }
    head
}

fn to_vec(mut head: &Option<Box<ListNode>>) -> Vec<i32> {
    let mut res = vec![];
    while let Some(node) = head.as_ref() {
        res.push(node.val);
        head = &node.next;
    }
    res
}

fn main() {
    let head = from_vec(&[1, 2, 3, 4, 5]);
    let head = reverse_list(head);
    println!("{:?}", to_vec(&head));
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseList(head) {
  let prev = null;
  let cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}

function fromArray(a) {
  const dummy = new ListNode(0);
  let tail = dummy;
  for (const x of a) {
    tail.next = new ListNode(x);
    tail = tail.next;
  }
  return dummy.next;
}

function toArray(head) {
  const res = [];
  for (let p = head; p; p = p.next) res.push(p.val);
  return res;
}

const head = fromArray([1, 2, 3, 4, 5]);
console.log(toArray(reverseList(head)));
```
