---
title: "LeetCode 19：删除链表的倒数第 N 个结点（双指针一趟扫描）ACERS 全解析"
date: 2026-02-11T07:56:02+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "双指针", "快慢指针", "哨兵节点", "LeetCode 19"]
description: "从朴素两趟遍历到一趟快慢指针，系统讲清删除链表倒数第 N 个结点的思路、正确性、工程场景与多语言可运行实现。"
keywords: ["LeetCode 19", "删除链表倒数第N个结点", "Remove Nth Node From End of List", "双指针", "快慢指针", "链表"]
---

> **副标题 / 摘要**  
> 这题的核心不是“删除节点”，而是“如何在单链表里定位倒数第 N 个节点的前驱”。本文从朴素思路推导到一趟双指针解法，用 ACERS 结构讲透正确性、边界处理与工程迁移。

- **预计阅读时长**：12~15 分钟  
- **适用场景标签**：`链表基础`、`双指针`、`面试高频`  
- **SEO 关键词**：LeetCode 19, Remove Nth Node From End of List, 删除链表倒数第 N 个结点, 快慢指针, 哨兵节点  
- **元描述（Meta Description）**：删除链表倒数第 N 个结点的完整 ACERS 解析：从暴力到一趟双指针，含复杂度、常见坑、工程示例与 Python/C/C++/Go/Rust/JS 代码。

---

## 目标读者

- 刚开始刷链表题，想建立稳定解题模板的同学  
- 知道快慢指针，但容易在边界条件上出错的开发者  
- 希望把“题解能力”迁移到工程链式数据处理场景的后端/系统工程师

## 背景 / 动机

“删除倒数第 N 个节点”是链表题里的经典中档题，常见难点不在删除本身，而在：

- 单链表不能回退，无法直接从尾部向前数；
- 可能删除头节点，导致返回值处理复杂；
- 一旦 `next` 指针处理失误，容易断链或越界。

掌握它的价值在于：  
你会形成一套可复用的“哨兵节点 + 双指针间距控制”模板，这对后续链表题（分组翻转、分割、合并）都很关键。

## 核心概念

- **单链表（Singly Linked List）**：每个节点只有 `next` 指针，只能向后遍历。  
- **哨兵节点（dummy）**：在头结点前增加一个虚拟节点，统一“删除头节点”和“删除中间节点”的处理逻辑。  
- **快慢指针固定间距**：先让 `fast` 领先 `slow` 共 `n` 步，再同步前进；当 `fast` 到达末尾时，`slow` 正好停在目标节点前驱。

---

## A — Algorithm（题目与算法）

### 题目重述

给你一个链表，删除链表的倒数第 `n` 个结点，并返回链表的头结点。

### 输入输出

| 项目 | 类型 | 含义 |
| --- | --- | --- |
| `head` | `ListNode` | 单链表头结点 |
| `n` | `int` | 倒数第 `n` 个位置 |
| 返回值 | `ListNode` | 删除目标节点后的头结点 |

### 示例 1

```text
输入: head = [1,2,3,4,5], n = 2
输出: [1,2,3,5]
```

解释：倒数第 2 个节点是 `4`，删除后得到 `[1,2,3,5]`。

### 示例 2

```text
输入: head = [1], n = 1
输出: []
```

解释：删除唯一节点后，链表为空。

### 示例 3

```text
输入: head = [1,2], n = 2
输出: [2]
```

解释：倒数第 2 个节点就是头结点 `1`。

### 图示（间距法）

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
fast 先走 n=2 步后：
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow
             fast

然后 slow/fast 同步走，直到 fast 到尾：
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
             slow           fast
此时 slow.next 就是待删除节点 4
```

---

## C — Concepts（核心思想）

### 思路推导：从朴素到最优

1. **朴素法：转数组后删除再重建**  
   - 能做，但需要 O(L) 额外空间；  
   - 链表题里属于“绕开链表特性”的解法，不够优雅。

2. **改进法：两趟遍历（先求长度，再找第 `L-n` 个）**  
   - 时间 O(L)，空间 O(1)，已经可接受；  
   - 但需要两次扫描，且头删仍要特判或引入 dummy。

3. **最佳法：一趟双指针 + 哨兵节点（本文主解）**  
   - `fast` 先走 `n` 步，保持与 `slow` 的固定间距；  
   - 两者同步前进直到 `fast.next == null`；  
   - `slow.next` 就是待删除节点，直接跳过它。

### 方法归类

- 双指针（Two Pointers）  
- 间距控制（Gap Maintenance）  
- 链表原地修改（In-place Pointer Rewire）

### 正确性直觉

设链表长度为 `L`。  
当 `fast` 与 `slow` 之间保持 `n` 个节点间距并一起向后走时：

- 当 `fast` 到达最后一个节点（下标 `L-1`）；
- `slow` 恰好在下标 `L-n-1`（目标前驱）；
- 删除 `slow.next` 就等价于删除倒数第 `n` 个节点。

这就是一趟算法成立的关键不变量。

---

## 实践指南 / 步骤

1. 创建哨兵节点：`dummy.next = head`。  
2. 初始化：`fast = dummy`, `slow = dummy`。  
3. `fast` 先走 `n` 步，制造间距。  
4. `while fast.next != null`：`fast` 与 `slow` 同时前进。  
5. 执行删除：`slow.next = slow.next.next`。  
6. 返回 `dummy.next`。

Python 可运行示例（含输入输出转换）：

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy

    for _ in range(n):
        fast = fast.next

    while fast.next is not None:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next


def from_list(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    print(to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)))  # [1,2,3,5]
    print(to_list(remove_nth_from_end(from_list([1]), 1)))              # []
    print(to_list(remove_nth_from_end(from_list([1, 2]), 2)))           # [2]
```

---

## E — Engineering（工程应用）

> 这道题本质是“在单向结构中删除倒数第 N 个元素”。  
> 工程里虽然不一定直接叫这个名字，但链式结构上的“末端相对定位删除”很常见。

### 场景 1：后台任务重试链裁剪（Go）

**背景**：微服务里常用单链结构记录任务重试轨迹。  
**为什么适用**：当要删掉“倒数第 N 次失败记录”时，可直接复用双指针模板。

```go
package main

import "fmt"

type Node struct {
	ID   int
	Next *Node
}

func removeNthFromEnd(head *Node, n int) *Node {
	dummy := &Node{Next: head}
	fast, slow := dummy, dummy

	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

func printList(head *Node) {
	for p := head; p != nil; p = p.Next {
		fmt.Printf("%d ", p.ID)
	}
	fmt.Println()
}

func main() {
	head := &Node{1, &Node{2, &Node{3, &Node{4, nil}}}}
	head = removeNthFromEnd(head, 2)
	printList(head) // 1 2 4
}
```

### 场景 2：系统空闲块链维护（C）

**背景**：简化内存管理器会维护空闲块单链表。  
**为什么适用**：按“距离尾部第 N 个块”剔除异常块时，单趟定位可减少扫描状态复杂度。

```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int addr;
    struct Node* next;
};

struct Node* remove_nth_from_end(struct Node* head, int n) {
    struct Node dummy = {0, head};
    struct Node *fast = &dummy, *slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }

    struct Node* del = slow->next;
    slow->next = del->next;
    free(del);
    return dummy.next;
}

int main() {
    struct Node* n4 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n3 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n2 = (struct Node*)malloc(sizeof(struct Node));
    struct Node* n1 = (struct Node*)malloc(sizeof(struct Node));
    n1->addr = 10; n1->next = n2;
    n2->addr = 20; n2->next = n3;
    n3->addr = 30; n3->next = n4;
    n4->addr = 40; n4->next = NULL;

    struct Node* head = remove_nth_from_end(n1, 3);
    for (struct Node* p = head; p; p = p->next) printf("%d ", p->addr);
    printf("\n");

    while (head) {
        struct Node* t = head;
        head = head->next;
        free(t);
    }
    return 0;
}
```

### 场景 3：前端撤销链路精简（JavaScript）

**背景**：编辑器可把历史操作组织成单向链。  
**为什么适用**：要删除“倒数第 N 个撤销快照”时，逻辑与本题完全一致。

```javascript
class Node {
  constructor(v, next = null) {
    this.v = v;
    this.next = next;
  }
}

function removeNthFromEnd(head, n) {
  const dummy = new Node(0, head);
  let fast = dummy;
  let slow = dummy;

  for (let i = 0; i < n; i++) fast = fast.next;
  while (fast.next !== null) {
    fast = fast.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;
  return dummy.next;
}

function print(head) {
  const arr = [];
  for (let p = head; p; p = p.next) arr.push(p.v);
  console.log(arr);
}

const head = new Node(1, new Node(2, new Node(3, new Node(4))));
print(removeNthFromEnd(head, 1)); // [1,2,3]
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(L)`，其中 `L` 为链表长度  
- 空间复杂度：`O(1)`（原地修改，常数额外指针）

### 方案对比

| 方案 | 时间 | 空间 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 转数组再删 | O(L) | O(L) | 实现直观 | 额外空间大，不像链表题 |
| 两趟遍历 | O(L) | O(1) | 稳定易懂 | 需要两次扫描 |
| 一趟双指针 + dummy | O(L) | O(1) | 一次扫描、边界统一 | 需要掌握间距不变量 |

### 常见错误思路

- 忘记加 `dummy`，删除头节点时出现分支爆炸  
- 让 `fast` 先走 `n+1` 或 `n-1` 步，造成 off-by-one  
- 删除后忘记处理被删节点内存（C/C++ 场景）

### 为什么当前方法更工程可行

- 模板化程度高，可迁移到大量链表变体题；  
- 边界行为稳定（尤其是删头）；  
- 对性能敏感环境友好（O(1) 额外空间）。

---

## 常见问题（FAQ）

### Q1：为什么循环条件是 `while fast.next != null`，不是 `while fast != null`？

因为我们需要让 `slow` 停在“待删除节点前驱”，当 `fast` 到最后一个节点时停止最合适。

### Q2：`n` 等于链表长度时会不会崩？

不会。由于使用了 `dummy`，此时 `slow` 最终停在 `dummy`，删除的正好是原头节点。

### Q3：可以用递归写吗？

可以，但递归通常带来 O(L) 栈空间，在深链场景下不如迭代稳定。

---

## 最佳实践与建议

- 先写 `dummy`，再考虑任何删除逻辑；  
- 统一采用“让 `fast` 先走 `n` 步”的写法，降低 off-by-one 风险；  
- 题解里先给两趟法，再过渡到一趟法，思路更有教学性；  
- 在 C/C++ 里注意释放被删节点，避免泄漏。

---

## S — Summary（总结）

### 核心收获

1. 倒数定位问题可以转化为双指针固定间距问题。  
2. `dummy` 是处理链表删除边界的最稳妥手段。  
3. 一趟双指针方案在时间 O(L)、空间 O(1) 下达到很好的工程平衡。  
4. 这道题的模板能迁移到大量链式结构改写任务。  
5. 复杂题往往不是新知识，而是基础模板的稳健组合。

### 延伸阅读

- LeetCode 19（官方）：<https://leetcode.com/problems/remove-nth-node-from-end-of-list/>  
- 力扣中文站：<https://leetcode.cn/problems/remove-nth-node-from-end-of-list/>  
- 相关题：LeetCode 21（合并两个有序链表）、LeetCode 206（反转链表）、LeetCode 25（K 个一组翻转链表）

---

## 行动号召（CTA）

现在就把这份模板默写一遍：  
先写两趟法，再改成一趟双指针，并用 `n=1`、`n=链表长度`、`单节点` 三组边界做自测。你会明显提升链表题稳定性。

---

## 多语言实现（可直接运行）

### Python

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next is not None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next


def from_list(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for x in nums:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = from_list([1, 2, 3, 4, 5])
    print(to_list(remove_nth_from_end(h, 2)))  # [1, 2, 3, 5]
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* new_node(int v) {
    struct ListNode* p = (struct ListNode*)malloc(sizeof(struct ListNode));
    p->val = v;
    p->next = NULL;
    return p;
}

struct ListNode* removeNthFromEnd(struct ListNode* head, int n) {
    struct ListNode dummy = {0, head};
    struct ListNode *fast = &dummy, *slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }

    struct ListNode* del = slow->next;
    slow->next = del->next;
    free(del);
    return dummy.next;
}

void print_list(struct ListNode* head) {
    for (struct ListNode* p = head; p; p = p->next) printf("%d ", p->val);
    printf("\n");
}

void free_list(struct ListNode* head) {
    while (head) {
        struct ListNode* t = head;
        head = head->next;
        free(t);
    }
}

int main() {
    struct ListNode* h1 = new_node(1);
    h1->next = new_node(2);
    h1->next->next = new_node(3);
    h1->next->next->next = new_node(4);
    h1->next->next->next->next = new_node(5);

    h1 = removeNthFromEnd(h1, 2);
    print_list(h1); // 1 2 3 5
    free_list(h1);
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0);
    dummy.next = head;
    ListNode* fast = &dummy;
    ListNode* slow = &dummy;

    for (int i = 0; i < n; ++i) fast = fast->next;
    while (fast->next != nullptr) {
        fast = fast->next;
        slow = slow->next;
    }
    ListNode* del = slow->next;
    slow->next = del->next;
    delete del;
    return dummy.next;
}

ListNode* build(const vector<int>& a) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int x : a) {
        tail->next = new ListNode(x);
        tail = tail->next;
    }
    return dummy.next;
}

void print(ListNode* head) {
    for (ListNode* p = head; p; p = p->next) cout << p->val << " ";
    cout << "\n";
}

void destroy(ListNode* head) {
    while (head) {
        ListNode* t = head;
        head = head->next;
        delete t;
    }
}

int main() {
    ListNode* h = build({1, 2, 3, 4, 5});
    h = removeNthFromEnd(h, 2);
    print(h); // 1 2 3 5
    destroy(h);
    return 0;
}
```

### Go

```go
package main

import "fmt"

type ListNode struct {
	Val  int
	Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	fast, slow := dummy, dummy

	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

func build(nums []int) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for _, x := range nums {
		tail.Next = &ListNode{Val: x}
		tail = tail.Next
	}
	return dummy.Next
}

func printList(head *ListNode) {
	for p := head; p != nil; p = p.Next {
		fmt.Printf("%d ", p.Val)
	}
	fmt.Println()
}

func main() {
	head := build([]int{1, 2, 3, 4, 5})
	head = removeNthFromEnd(head, 2)
	printList(head) // 1 2 3 5
}
```

### Rust（可运行安全版：两趟遍历）

> 说明：为保持代码简洁与所有权安全，Rust 版本采用两趟遍历（同为 O(L) 时间、O(1) 额外空间）。

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

fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut len = 0usize;
    let mut p = head.as_ref();
    while let Some(node) = p {
        len += 1;
        p = node.next.as_ref();
    }

    let idx = len - n as usize; // 要删除的是第 idx(0-based) 个
    let mut dummy = Box::new(ListNode { val: 0, next: head });
    let mut cur = &mut dummy;
    for _ in 0..idx {
        cur = cur.next.as_mut().unwrap();
    }
    let next = cur.next.as_mut().and_then(|node| node.next.take());
    cur.next = next;
    dummy.next
}

fn from_vec(a: Vec<i32>) -> Option<Box<ListNode>> {
    let mut head = None;
    for &x in a.iter().rev() {
        let mut node = Box::new(ListNode::new(x));
        node.next = head;
        head = Some(node);
    }
    head
}

fn to_vec(mut head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut out = Vec::new();
    while let Some(mut node) = head {
        out.push(node.val);
        head = node.next.take();
    }
    out
}

fn main() {
    let head = from_vec(vec![1, 2, 3, 4, 5]);
    let ans = remove_nth_from_end(head, 2);
    println!("{:?}", to_vec(ans)); // [1, 2, 3, 5]
}
```

### JavaScript

```javascript
class ListNode {
  constructor(val = 0, next = null) {
    this.val = val;
    this.next = next;
  }
}

function removeNthFromEnd(head, n) {
  const dummy = new ListNode(0, head);
  let fast = dummy;
  let slow = dummy;

  for (let i = 0; i < n; i++) {
    fast = fast.next;
  }
  while (fast.next !== null) {
    fast = fast.next;
    slow = slow.next;
  }
  slow.next = slow.next.next;
  return dummy.next;
}

function fromArray(arr) {
  const dummy = new ListNode();
  let tail = dummy;
  for (const x of arr) {
    tail.next = new ListNode(x);
    tail = tail.next;
  }
  return dummy.next;
}

function toArray(head) {
  const out = [];
  for (let p = head; p; p = p.next) out.push(p.val);
  return out;
}

const head = fromArray([1, 2, 3, 4, 5]);
console.log(toArray(removeNthFromEnd(head, 2))); // [1,2,3,5]
```
