---
title: "Hot100：回文链表（Palindrome Linked List）快慢指针 + 反转后半段 O(1) 空间 ACERS 解析"
date: 2026-02-01T18:44:01+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "快慢指针", "反转链表", "回文", "LeetCode 234"]
description: "判断单链表是否为回文：快慢指针定位中点、原地反转后半段并与前半段对比，最后再恢复链表；O(n) 时间、O(1) 额外空间。"
keywords: ["Palindrome Linked List", "回文链表", "fast slow pointers", "reverse second half", "O(1) space", "restore list", "LeetCode 234"]
---

> **副标题 / 摘要**  
> 回文链表的核心是“对称比较”，但单链表不能从尾部往前走。最稳的工程化解法是：**快慢指针找中点 -> 原地反转后半段 -> 比较 -> 再反转恢复结构**，做到 O(n) 时间、O(1) 额外空间且不破坏链表。

- **预计阅读时长**：10~14 分钟  
- **标签**：`Hot100`、`链表`、`快慢指针`、`原地反转`  
- **SEO 关键词**：回文链表, Palindrome Linked List, O(1) 空间, 快慢指针, 反转后半段, LeetCode 234  
- **元描述**：快慢指针定位中点，反转后半段与前半段逐一比较，最后恢复链表结构；O(n)/O(1) 判断单链表是否回文。  

---

## 目标读者

- 刷 Hot100，想掌握“链表中点 + 原地反转”组合拳的学习者  
- 面试中经常遇到“回文/对称/镜像”类题的开发者  
- 关注空间效率、且需要保证数据结构不被破坏的工程实践者

## 背景 / 动机

在数组里判断回文很简单：左右指针向中间收缩即可。  
但在单链表里，你只能顺着 `next` 单向走，无法从尾部回看，这就让“对称比较”变得不那么直接。

工程上常见的约束也与题目一致：

- 结构不能改（不能改值、不能打标记、不能改 next 永久化）  
- 额外内存有限（不想把所有节点拷贝到数组里）  

因此我们需要一个 **线性时间、常数空间、且能恢复结构** 的模板解法。

## 核心概念

| 概念 | 含义 | 作用 |
| --- | --- | --- |
| 回文 | 从左到右与从右到左相同 | 需要做“对称比较” |
| 快慢指针 | `fast` 每次两步、`slow` 每次一步 | O(n) 找到链表中点 |
| 原地反转 | 改指针方向把链表片段反转 | 把“后半段”变成可从前往后比较 |
| 结构恢复 | 比较完成后再反转回去并接回 | 满足“链表保持原结构”要求 |

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个单链表的头节点 `head`，请你判断该链表是否为回文链表：  
如果是回文，返回 `true`；否则返回 `false`。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| head | ListNode | 单链表头结点 |
| 返回 | bool | 是否为回文 |

### 示例 1

```text
输入: 1 -> 2 -> 2 -> 1
输出: true
```

### 示例 2

```text
输入: 1 -> 2
输出: false
```

---

## 思路推导：从“拷贝到数组”到“原地反转并恢复”

### 朴素方案：拷贝到数组再做双指针

1. 遍历链表把值放入数组 `arr`  
2. 用数组左右指针判断回文  

优点：简单、容易写对。  
缺点：额外空间 O(n)，在大链表或内存敏感场景不理想。

### 次优方案：用栈存前半段

用快慢指针找中点的同时，把前半段入栈；之后出栈与后半段比较。  
仍然需要 O(n) 额外空间（栈），只是把“数组”换成“栈”。

### 关键观察：如果能把后半段倒过来，就能像数组一样对称比较

单链表的问题在于“无法从尾部往前”。  
但如果我们把 **后半段原地反转**，后半段就变成“从尾到头”的顺序了：

```text
1 -> 2 -> 3 -> 2 -> 1
           ^ 反转这段后：
1 -> 2 -> 3 -> 1 -> 2
           ^ second_half_start
```

于是：

- 从头开始的指针 `p` 走 `1,2,3,...`
- 从反转后的后半段指针 `q` 走 `1,2,...`

逐一比较即可判断回文。  
比较完成后，再把后半段反转回去并接回去，就能恢复原结构。

---

## C — Concepts（核心思想）

### 方法归类

- **快慢指针找中点（Two pointers / Tortoise-Hare）**  
- **链表原地反转（In-place Reverse）**  
- **结构恢复（Restore after temporary mutation）**

### 如何稳定处理奇偶长度？

一种非常稳的写法是先求“前半段末尾”：

- 若长度为奇数：前半段末尾是正中间节点（中点不参与比较）  
- 若长度为偶数：前半段末尾是左中点

然后反转 `first_half_end.next` 作为后半段的头。  
比较时只需要遍历后半段长度即可（后半段长度 <= 前半段长度）。

---

## 实践指南 / 步骤

1. 若链表为空或只有 1 个节点，直接返回 true  
2. 用快慢指针找到前半段末尾 `first_half_end`  
3. 原地反转 `first_half_end.next`，得到 `second_half_start`  
4. 用两个指针从 `head` 与 `second_half_start` 同步比较  
5. 比较结束后，再反转 `second_half_start` 并接回 `first_half_end.next`（恢复结构）  
6. 返回比较结果

Python 可运行示例（保存为 `palindrome_list.py`）：

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def end_of_first_half(head: ListNode) -> ListNode:
    fast = head
    slow = head
    while fast.next and fast.next.next:
        fast = fast.next.next
        slow = slow.next  # type: ignore[assignment]
    return slow


def is_palindrome(head: ListNode | None) -> bool:
    if head is None or head.next is None:
        return True

    first_half_end = end_of_first_half(head)
    second_half_start = reverse_list(first_half_end.next)

    p1 = head
    p2 = second_half_start
    ok = True
    while ok and p2 is not None:
        if p1.val != p2.val:
            ok = False
        p1 = p1.next  # type: ignore[assignment]
        p2 = p2.next

    # Restore list.
    first_half_end.next = reverse_list(second_half_start)
    return ok


def build(vals):
    dummy = ListNode(0)
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


if __name__ == "__main__":
    a = build([1, 2, 2, 1])
    b = build([1, 2])
    print(is_palindrome(a))  # True
    print(is_palindrome(b))  # False
```

---

## E — Engineering（工程应用）

### 场景 1：事件序列的对称性校验（Python）

**背景**：在风控/行为分析中，你可能用链表（或链式结构）表达一次会话的事件序列；某些规则要求“对称结构”（例如进入/退出必须镜像匹配）。  
**为什么适用**：不想把所有事件复制到数组里；并且希望校验后结构仍可被后续处理复用。  

```python
def is_symmetric_sequence(head):
    # 直接复用 is_palindrome：把“对称性”抽象成回文
    return is_palindrome(head)
```

### 场景 2：内存受限设备上的数据帧回文检测（C）

**背景**：某些嵌入式系统把采样值串成链表（例如内存池 + next 指针），需要快速判断是否满足对称约束以触发告警/自检。  
**为什么适用**：O(1) 额外空间，无需申请大块缓冲区。  

```c
struct ListNode { int val; struct ListNode* next; };

static struct ListNode* reverse(struct ListNode* head) {
    struct ListNode* prev = 0;
    struct ListNode* cur = head;
    while (cur) {
        struct ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

static struct ListNode* endFirstHalf(struct ListNode* head) {
    struct ListNode* fast = head;
    struct ListNode* slow = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

int isPalindrome(struct ListNode* head) {
    if (!head || !head->next) return 1;
    struct ListNode* firstEnd = endFirstHalf(head);
    struct ListNode* second = reverse(firstEnd->next);
    int ok = 1;
    struct ListNode* p1 = head;
    struct ListNode* p2 = second;
    while (ok && p2) {
        if (p1->val != p2->val) ok = 0;
        p1 = p1->next;
        p2 = p2->next;
    }
    firstEnd->next = reverse(second); // restore
    return ok;
}
```

### 场景 3：前端编辑器的操作栈对称检测（JavaScript）

**背景**：某些编辑器用链表记录操作（undo/redo 历史）；你可能想检测“操作是否对称抵消”（例如某些模式下要求回文式操作序列）。  
**为什么适用**：链式结构本身可直接处理；校验结束后需要继续使用原结构。  

```javascript
function reverse(head) {
  let prev = null, cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}

function endFirstHalf(head) {
  let fast = head, slow = head;
  while (fast.next && fast.next.next) {
    fast = fast.next.next;
    slow = slow.next;
  }
  return slow;
}

function isPalindrome(head) {
  if (!head || !head.next) return true;
  const firstEnd = endFirstHalf(head);
  const secondStart = reverse(firstEnd.next);

  let p1 = head, p2 = secondStart;
  let ok = true;
  while (ok && p2) {
    if (p1.val !== p2.val) ok = false;
    p1 = p1.next;
    p2 = p2.next;
  }
  firstEnd.next = reverse(secondStart); // restore
  return ok;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(n)（找中点 + 反转 + 比较 + 恢复都是线性）  
- **空间复杂度**：O(1)

### 替代方案对比

| 方法 | 思路 | 时间 | 额外空间 | 问题 |
| --- | --- | --- | --- | --- |
| 数组拷贝 | 值拷贝到数组，双指针判断 | O(n) | O(n) | 内存占用高 |
| 栈 | 压入前半段，弹出比较后半段 | O(n) | O(n) | 仍占用线性空间 |
| 递归 | 利用递归栈回溯比较 | O(n) | O(n) | 容易栈溢出/不可控 |
| **反转后半段** | 中点 + 反转 + 比较 + 恢复 | **O(n)** | **O(1)** | 代码稍多但最工程可行 |

### 常见错误思路

1. **反转后半段后忘记恢复**：题目要求结构保持原样；工程里也常要求“校验不产生副作用”。  
2. **中点处理搞错（奇偶长度）**：导致比较区间不一致。推荐使用“前半段末尾 + 反转 next”的写法。  
3. **把节点对象当作值**：回文判断比较的是 `val`；但恢复结构比较的是指针链接。

---

## 常见问题与注意事项

1. **为什么比较时遍历 `p2`（后半段）就够了？**  
   后半段长度 <= 前半段长度；回文成立时对应位置都要匹配，遍历后半段即可覆盖所有需要比较的对称对。

2. **如果链表有环怎么办？**  
   题目保证无环。工程里若不保证，需要先判环（Floyd），否则“找中点/反转”可能死循环或破坏结构。

3. **反转会不会破坏原链表？**  
   会“临时改变”，但我们在返回前把后半段再反转一次并接回，保证外部观测到的结构不变。

---

## 最佳实践与建议

- 统一采用模板：`first_half_end` + `reverse(first_half_end.next)`，避免奇偶分支  
- 做题/写代码时强制写“恢复结构”步骤，防止遗漏  
- 测试用例至少覆盖：
  - 空链表、单节点  
  - 偶数长度回文与非回文  
  - 奇数长度回文与非回文  
  - 值重复很多的场景（容易误判）

---

## S — Summary（总结）

### 核心收获

- 单链表不能从尾部回看，回文判断需要“借力”结构变换  
- 快慢指针可在 O(n) 内定位前半段末尾  
- 反转后半段能把“从尾到头”的比较变成“从头到尾”的比较  
- 比较完成后再反转恢复，确保不破坏链表原结构  
- 该模板可复用到很多“链表中点 + 半段处理”的题目（如重排、分割、判环扩展）

### 参考与延伸阅读

- LeetCode 234. Palindrome Linked List
- 链表反转（Reverse Linked List）与快慢指针（Middle of Linked List）经典题
- 任何数据结构的“临时变换 + 恢复”工程实践（避免副作用）

---

## 元信息

- **阅读时长**：10~14 分钟  
- **标签**：Hot100、链表、回文、快慢指针、原地反转  
- **SEO 关键词**：回文链表, Palindrome Linked List, 反转后半段, O(1) 空间, LeetCode 234  
- **元描述**：快慢指针定位中点，反转后半段对比并恢复结构；O(n)/O(1) 判断单链表是否回文。  

---

## 行动号召（CTA）

建议你用同一套“中点 + 反转”模板再刷两题巩固：  
1) 重排链表（Reorder List）；2) 反转链表（Reverse Linked List）。  
如果你希望我把这些题也按同风格整理成 Hot100 系列文章，直接丢题目过来即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from __future__ import annotations


class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next: ListNode | None = None


def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def end_of_first_half(head: ListNode) -> ListNode:
    fast = head
    slow = head
    while fast.next and fast.next.next:
        fast = fast.next.next
        slow = slow.next  # type: ignore[assignment]
    return slow


def is_palindrome(head: ListNode | None) -> bool:
    if head is None or head.next is None:
        return True
    first_end = end_of_first_half(head)
    second = reverse_list(first_end.next)

    ok = True
    p1 = head
    p2 = second
    while ok and p2 is not None:
        if p1.val != p2.val:
            ok = False
        p1 = p1.next  # type: ignore[assignment]
        p2 = p2.next

    first_end.next = reverse_list(second)  # restore
    return ok
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

static struct ListNode* reverse(struct ListNode* head) {
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

static struct ListNode* endFirstHalf(struct ListNode* head) {
    struct ListNode* fast = head;
    struct ListNode* slow = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

int isPalindrome(struct ListNode* head) {
    if (!head || !head->next) return 1;
    struct ListNode* firstEnd = endFirstHalf(head);
    struct ListNode* second = reverse(firstEnd->next);
    int ok = 1;
    struct ListNode* p1 = head;
    struct ListNode* p2 = second;
    while (ok && p2) {
        if (p1->val != p2->val) ok = 0;
        p1 = p1->next;
        p2 = p2->next;
    }
    firstEnd->next = reverse(second); // restore
    return ok;
}

static struct ListNode* node(int v) {
    struct ListNode* n = (struct ListNode*)malloc(sizeof(struct ListNode));
    n->val = v;
    n->next = NULL;
    return n;
}

int main(void) {
    // 1 -> 2 -> 2 -> 1
    struct ListNode* a1 = node(1);
    struct ListNode* a2 = node(2);
    struct ListNode* a3 = node(2);
    struct ListNode* a4 = node(1);
    a1->next = a2; a2->next = a3; a3->next = a4;
    printf("%d\n", isPalindrome(a1)); // 1
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

static ListNode* reverse(ListNode* head) {
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

static ListNode* endFirstHalf(ListNode* head) {
    ListNode* fast = head;
    ListNode* slow = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

bool isPalindrome(ListNode* head) {
    if (!head || !head->next) return true;
    ListNode* firstEnd = endFirstHalf(head);
    ListNode* second = reverse(firstEnd->next);
    bool ok = true;
    ListNode* p1 = head;
    ListNode* p2 = second;
    while (ok && p2) {
        if (p1->val != p2->val) ok = false;
        p1 = p1->next;
        p2 = p2->next;
    }
    firstEnd->next = reverse(second); // restore
    return ok;
}

int main() {
    auto* a1 = new ListNode(1);
    auto* a2 = new ListNode(2);
    auto* a3 = new ListNode(2);
    auto* a4 = new ListNode(1);
    a1->next = a2; a2->next = a3; a3->next = a4;
    std::cout << std::boolalpha << isPalindrome(a1) << "\n";
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

func reverse(head *ListNode) *ListNode {
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

func endFirstHalf(head *ListNode) *ListNode {
    fast, slow := head, head
    for fast.Next != nil && fast.Next.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
    }
    return slow
}

func isPalindrome(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return true
    }
    firstEnd := endFirstHalf(head)
    second := reverse(firstEnd.Next)

    ok := true
    p1, p2 := head, second
    for ok && p2 != nil {
        if p1.Val != p2.Val {
            ok = false
        }
        p1 = p1.Next
        p2 = p2.Next
    }
    firstEnd.Next = reverse(second) // restore
    return ok
}

func main() {
    a4 := &ListNode{Val: 1}
    a3 := &ListNode{Val: 2, Next: a4}
    a2 := &ListNode{Val: 2, Next: a3}
    a1 := &ListNode{Val: 1, Next: a2}
    fmt.Println(isPalindrome(a1))
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

fn reverse(mut head: Option<Rc<RefCell<ListNode>>>) -> Option<Rc<RefCell<ListNode>>> {
    let mut prev: Option<Rc<RefCell<ListNode>>> = None;
    while let Some(cur) = head {
        let nxt = cur.borrow().next.clone();
        cur.borrow_mut().next = prev.clone();
        prev = Some(cur);
        head = nxt;
    }
    prev
}

fn end_first_half(head: &Option<Rc<RefCell<ListNode>>>) -> Option<Rc<RefCell<ListNode>>> {
    let mut fast = head.clone();
    let mut slow = head.clone();
    loop {
        let f1 = next_of(&fast);
        let f2 = next_of(&f1);
        if f1.is_none() || f2.is_none() {
            break;
        }
        fast = f2;
        slow = next_of(&slow);
    }
    slow
}

fn is_palindrome(head: Option<Rc<RefCell<ListNode>>>) -> bool {
    if head.is_none() || next_of(&head).is_none() {
        return true;
    }
    let first_end = end_first_half(&head).unwrap();
    let second_start = reverse(first_end.borrow().next.clone());

    let mut ok = true;
    let mut p1 = head.clone();
    let mut p2 = second_start.clone();
    while ok && p2.is_some() {
        let v1 = p1.as_ref().unwrap().borrow().val;
        let v2 = p2.as_ref().unwrap().borrow().val;
        if v1 != v2 {
            ok = false;
        }
        p1 = next_of(&p1);
        p2 = next_of(&p2);
    }

    // restore
    first_end.borrow_mut().next = reverse(second_start);
    ok
}

fn main() {
    // 1 -> 2 -> 2 -> 1
    let a1 = node(1);
    let a2 = node(2);
    let a3 = node(2);
    let a4 = node(1);
    a1.borrow_mut().next = Some(a2.clone());
    a2.borrow_mut().next = Some(a3.clone());
    a3.borrow_mut().next = Some(a4.clone());

    println!("{}", is_palindrome(Some(a1)));
}
```

```javascript
class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

function reverse(head) {
  let prev = null, cur = head;
  while (cur) {
    const nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
  }
  return prev;
}

function endFirstHalf(head) {
  let fast = head, slow = head;
  while (fast.next && fast.next.next) {
    fast = fast.next.next;
    slow = slow.next;
  }
  return slow;
}

function isPalindrome(head) {
  if (!head || !head.next) return true;
  const firstEnd = endFirstHalf(head);
  const secondStart = reverse(firstEnd.next);

  let ok = true;
  let p1 = head, p2 = secondStart;
  while (ok && p2) {
    if (p1.val !== p2.val) ok = false;
    p1 = p1.next;
    p2 = p2.next;
  }
  firstEnd.next = reverse(secondStart); // restore
  return ok;
}

// demo: 1 -> 2 -> 2 -> 1
const a1 = new ListNode(1);
const a2 = new ListNode(2);
const a3 = new ListNode(2);
const a4 = new ListNode(1);
a1.next = a2; a2.next = a3; a3.next = a4;
console.log(isPalindrome(a1));
```

