---
title: "LeetCode 148：排序链表（Sort List）用归并排序稳定拿下 O(n log n)"
date: 2026-02-11T07:59:35+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "归并排序", "快慢指针", "LeetCode 148", "算法题解"]
description: "排序链表的关键是利用链表擅长“拆分+合并”的特性，使用归并排序在 O(n log n) 时间完成升序排序。本文从朴素方案推导到最优解，并附六语言可运行代码。"
keywords: ["Sort List", "排序链表", "LeetCode 148", "归并排序", "快慢指针", "链表排序"]
---

> **副标题 / 摘要**  
> 链表不擅长随机访问，所以直接套数组快排并不合适。排序链表最稳妥的路径是归并排序：先拆成小段，再线性合并，复杂度 `O(n log n)`，并且非常工程化。

- **预计阅读时长**：14~18 分钟  
- **标签**：`链表`、`归并排序`、`快慢指针`、`LeetCode 148`  
- **SEO 关键词**：Sort List, 排序链表, LeetCode 148, merge sort linked list, 链表归并排序  
- **元描述**：从朴素思路到归并排序，系统讲清 LeetCode 148 排序链表，含复杂度、常见坑、工程迁移与 Python/C/C++/Go/Rust/JS 可运行实现。  

---

## 目标读者

- 正在刷链表与排序题的初中级开发者
- 想把“归并排序”为何适合链表讲明白的面试准备者
- 需要在工程中处理链式数据流排序的后端/系统工程师

## 背景 / 动机

这题的难点不在“排序”本身，而在“数据结构选择”：

- 数组常用快排/堆排，依赖随机访问；
- 链表随机访问代价高，但拆链与拼链很便宜；
- 所以要选“顺着链表特性”的算法，而不是生搬硬套数组思维。

一句话：  
**链表排序最匹配的方法是归并排序，而不是快排。**

## 核心概念

- **快慢指针切分**：用 `slow/fast` 找中点，把链表一分为二
- **分治（Divide and Conquer）**：对子链表递归排序
- **有序链表合并**：线性时间把两条有序链表合成一条
- **稳定性**：合并时遇到相等值优先取左链，可保持稳定顺序

---

## A — Algorithm（题目与算法）

### 题目还原

给你链表头节点 `head`，请将其按**升序**排列并返回排序后的链表。

### 输入输出

| 项目 | 说明 |
| --- | --- |
| 输入 | 单链表头节点 `head` |
| 输出 | 排序后的链表头节点（升序） |

### 示例 1

```text
输入: head = [4,2,1,3]
输出: [1,2,3,4]
```

### 示例 2

```text
输入: head = [-1,5,3,4,0]
输出: [-1,0,3,4,5]
```

---

## 思路推导：从朴素到最优

### 朴素方案 1：链表转数组后排序

做法：

1. 遍历链表存入数组；
2. 数组排序；
3. 重新连回链表。

问题：

- 额外空间 `O(n)`；
- 并没有体现链表结构优势；
- 面试官往往会追问“能否更贴合链表做法”。

### 朴素方案 2：链表插入排序

做法：维护有序前缀，每次把新节点插入到正确位置。

问题：

- 最坏时间 `O(n^2)`；
- 数据量稍大就会超时。

### 关键观察

1. 链表可以 O(1) 断开和拼接，适合“拆分 + 合并”；
2. 有序链表合并是 O(n)；
3. 递归深度约 `log n`，整体可达 `O(n log n)`。

### 方法选择：链表归并排序（Top-Down）

- 用快慢指针找到中点并断链；
- 分别排序左右子链；
- 调用“合并两个有序链表”得到结果。

这套方法既满足复杂度要求，也足够稳定和通用。

---

## C — Concepts（核心思想）

### 方法归类

- 分治算法（Divide and Conquer）
- 链表归并排序（Merge Sort on Linked List）
- 双指针 + 哨兵节点（Merge 阶段）

### 概念模型

设 `T(n)` 表示长度 `n` 链表排序开销：

```text
T(n) = 2T(n/2) + O(n)
```

根据主定理：

```text
T(n) = O(n log n)
```

### 正确性要点

1. **分解正确**：快慢指针切分后，左右两段覆盖全部节点且无重叠。  
2. **子问题正确**：递归返回的左右链表各自有序。  
3. **合并正确**：每次选两链头部较小节点拼到结果，最终整体有序。

---

## 实践指南 / 步骤

1. 处理递归终止：空链或单节点链直接返回  
2. 快慢指针找中点，`prev.next = None` 断开链表  
3. 递归排序左半与右半  
4. 合并两条有序链表  
5. 返回合并后新头

Python 可运行示例（最小版本）：

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head

    prev = None
    slow = fast = head
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    prev.next = None

    left = sort_list(head)
    right = sort_list(slow)
    return merge(left, right)


def build(arr: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in arr:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = build([4, 2, 1, 3])
    print(dump(sort_list(h)))  # [1, 2, 3, 4]
```

---

## E — Engineering（工程应用）

### 场景 1：日志代理中的链式缓冲按时间排序（Go）

**背景**：日志代理从多个通道接入事件，暂存在链式缓冲中。  
**为什么适用**：链表不断追加节点，最终批处理前需要按时间排序；归并排序适合链结构。

```go
package main

import "fmt"

type Node struct {
	Ts   int
	Next *Node
}

func merge(a, b *Node) *Node {
	dummy := &Node{}
	tail := dummy
	for a != nil && b != nil {
		if a.Ts <= b.Ts {
			tail.Next = a
			a = a.Next
		} else {
			tail.Next = b
			b = b.Next
		}
		tail = tail.Next
	}
	if a != nil {
		tail.Next = a
	} else {
		tail.Next = b
	}
	return dummy.Next
}

func main() {
	a := &Node{1, &Node{4, nil}}
	b := &Node{2, &Node{3, nil}}
	h := merge(a, b)
	for h != nil {
		fmt.Print(h.Ts, " ")
		h = h.Next
	}
	fmt.Println() // 1 2 3 4
}
```

### 场景 2：内存管理空闲块链按地址整理（C）

**背景**：简化内存分配器会维护空闲块链，按地址排序便于后续合并相邻块。  
**为什么适用**：链表节点仅重连指针，不必额外拷贝结构体。

```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int addr;
    struct Node* next;
};

struct Node* merge(struct Node* a, struct Node* b) {
    struct Node dummy;
    struct Node* tail = &dummy;
    dummy.next = NULL;
    while (a && b) {
        if (a->addr <= b->addr) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

int main(void) {
    struct Node n1 = {100, NULL};
    struct Node n2 = {300, NULL};
    n1.next = &n2;
    struct Node m1 = {200, NULL};
    struct Node m2 = {400, NULL};
    m1.next = &m2;

    struct Node* h = merge(&n1, &m1);
    while (h) {
        printf("%d ", h->addr);
        h = h->next;
    }
    printf("\n"); // 100 200 300 400
    return 0;
}
```

### 场景 3：前端离线任务链按优先级重排（JavaScript）

**背景**：离线队列用链式结构记录待同步任务。  
**为什么适用**：同步前按优先级排序，链表归并可避免大规模数组搬移。

```javascript
function mergeSorted(a, b) {
  const dummy = { p: 0, next: null };
  let tail = dummy;
  while (a && b) {
    if (a.p <= b.p) {
      tail.next = a;
      a = a.next;
    } else {
      tail.next = b;
      b = b.next;
    }
    tail = tail.next;
  }
  tail.next = a || b;
  return dummy.next;
}

const a = { p: 1, next: { p: 5, next: null } };
const b = { p: 2, next: { p: 4, next: null } };
let h = mergeSorted(a, b);
const out = [];
while (h) { out.push(h.p); h = h.next; }
console.log(out); // [1, 2, 4, 5]
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(n log n)`
- 额外空间：
  - 递归版：`O(log n)`（调用栈）
  - 若要求严格常数辅助空间，可用自底向上迭代归并

### 方案对比

| 方案 | 时间复杂度 | 额外空间 | 评价 |
| --- | --- | --- | --- |
| 转数组排序 | `O(n log n)` | `O(n)` | 实现简单，但非链表友好 |
| 插入排序 | `O(n^2)` | `O(1)` | 小规模可用，大规模慢 |
| 递归归并（本解） | `O(n log n)` | `O(log n)` | 代码清晰、面试常用 |
| 迭代归并 | `O(n log n)` | `O(1)` | 更优空间，代码更复杂 |

### 常见错误

- 快慢指针找到中点后忘记断链，导致递归死循环
- 合并时忘记移动 `tail` 或误覆盖 `next`
- 递归边界写成 `if not head`，漏掉单节点情况

### 为什么当前方法是最优/最工程可行

- 时间达到排序类题的标准上限 `O(n log n)`
- 链表重排只改指针，不做元素搬移，工程成本低
- 模板可复用到“排序链表”“合并 k 链表”等一类题

---

## 常见问题与注意事项（FAQ）

### Q1：为什么不推荐快排做链表排序？

快排需要频繁分区与随机访问，链表不擅长；实现复杂且常数大。归并排序更顺链表结构。

### Q2：可以做到稳定排序吗？

可以。合并时相等值优先取左链（`<=`）即可保持稳定性。

### Q3：面试官要求 O(1) 额外空间怎么办？

回答可升级到“迭代自底向上归并排序”，避免递归栈。思路一致，只是工程实现更繁琐。

---

## 最佳实践与建议

- 把“找中点并断链”单独写清楚，避免递归死循环
- 合并函数单测独立验证，再接入整体排序
- 统一使用 `dummy` 简化链表头处理
- 先写递归版拿正确性，再考虑迭代版优化空间

---

## S — Summary（总结）

核心收获：

1. 链表排序首选归并排序，因为链表天然适配“拆分+合并”。
2. 快慢指针 + 断链是这题成败关键，漏一步就会死循环。
3. 合并两个有序链表是可复用基础能力，值得单独背模板。
4. 递归版可读性高；若追求严格常数空间，可进阶迭代版。
5. 该题是连接“链表操作能力”和“分治思维”的典型桥梁题。

推荐延伸阅读：

- LeetCode 21: Merge Two Sorted Lists  
- LeetCode 23: Merge k Sorted Lists  
- LeetCode 147: Insertion Sort List  
- LeetCode 148: Sort List（官方题目）  
- 参考链接：`https://leetcode.com/problems/sort-list/`

---

## 多语言可运行实现

### Python

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        prev = None
        slow = fast = head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        prev.next = None

        left = self.sortList(head)
        right = self.sortList(slow)
        return self.merge(left, right)

    def merge(self, a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy
        while a and b:
            if a.val <= b.val:
                tail.next = a
                a = a.next
            else:
                tail.next = b
                b = b.next
            tail = tail.next
        tail.next = a if a else b
        return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    h = build([4, 2, 1, 3])
    print(dump(Solution().sortList(h)))  # [1, 2, 3, 4]
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
    struct ListNode* n = (struct ListNode*)malloc(sizeof(struct ListNode));
    n->val = v;
    n->next = NULL;
    return n;
}

struct ListNode* merge(struct ListNode* a, struct ListNode* b) {
    struct ListNode dummy;
    struct ListNode* tail = &dummy;
    dummy.next = NULL;
    while (a && b) {
        if (a->val <= b->val) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

struct ListNode* sortList(struct ListNode* head) {
    if (!head || !head->next) return head;

    struct ListNode* prev = NULL;
    struct ListNode* slow = head;
    struct ListNode* fast = head;
    while (fast && fast->next) {
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    prev->next = NULL;

    struct ListNode* left = sortList(head);
    struct ListNode* right = sortList(slow);
    return merge(left, right);
}

struct ListNode* build(const int* a, int n) {
    struct ListNode dummy;
    dummy.next = NULL;
    struct ListNode* tail = &dummy;
    for (int i = 0; i < n; i++) {
        tail->next = new_node(a[i]);
        tail = tail->next;
    }
    return dummy.next;
}

void print_list(struct ListNode* h) {
    while (h) {
        printf("%d", h->val);
        if (h->next) printf(" -> ");
        h = h->next;
    }
    printf("\n");
}

void free_list(struct ListNode* h) {
    while (h) {
        struct ListNode* nxt = h->next;
        free(h);
        h = nxt;
    }
}

int main(void) {
    int a[] = {4, 2, 1, 3};
    struct ListNode* h = build(a, 4);
    h = sortList(h);
    print_list(h); // 1 -> 2 -> 3 -> 4
    free_list(h);
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
    ListNode(int x = 0) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;

        ListNode* prev = nullptr;
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next) {
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        prev->next = nullptr;

        ListNode* left = sortList(head);
        ListNode* right = sortList(slow);
        return merge(left, right);
    }

private:
    ListNode* merge(ListNode* a, ListNode* b) {
        ListNode dummy;
        ListNode* tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) {
                tail->next = a;
                a = a->next;
            } else {
                tail->next = b;
                b = b->next;
            }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }
};

ListNode* build(const vector<int>& a) {
    ListNode dummy;
    ListNode* tail = &dummy;
    for (int v : a) {
        tail->next = new ListNode(v);
        tail = tail->next;
    }
    return dummy.next;
}

void printList(ListNode* h) {
    while (h) {
        cout << h->val;
        if (h->next) cout << " -> ";
        h = h->next;
    }
    cout << "\n";
}

void freeList(ListNode* h) {
    while (h) {
        ListNode* nxt = h->next;
        delete h;
        h = nxt;
    }
}

int main() {
    ListNode* h = build({4, 2, 1, 3});
    h = Solution().sortList(h);
    printList(h); // 1 -> 2 -> 3 -> 4
    freeList(h);
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

func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	var prev *ListNode
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	prev.Next = nil

	left := sortList(head)
	right := sortList(slow)
	return merge(left, right)
}

func merge(a, b *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for a != nil && b != nil {
		if a.Val <= b.Val {
			tail.Next = a
			a = a.Next
		} else {
			tail.Next = b
			b = b.Next
		}
		tail = tail.Next
	}
	if a != nil {
		tail.Next = a
	} else {
		tail.Next = b
	}
	return dummy.Next
}

func build(a []int) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for _, v := range a {
		tail.Next = &ListNode{Val: v}
		tail = tail.Next
	}
	return dummy.Next
}

func printList(h *ListNode) {
	for h != nil {
		fmt.Print(h.Val)
		if h.Next != nil {
			fmt.Print(" -> ")
		}
		h = h.Next
	}
	fmt.Println()
}

func main() {
	h := build([]int{4, 2, 1, 3})
	h = sortList(h)
	printList(h) // 1 -> 2 -> 3 -> 4
}
```

### Rust

```rust
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.as_ref().is_none() || head.as_ref().unwrap().next.is_none() {
        return head;
    }

    let (left, right) = split(head);
    let left = sort_list(left);
    let right = sort_list(right);
    merge(left, right)
}

fn split(mut head: Option<Box<ListNode>>) -> (Option<Box<ListNode>>, Option<Box<ListNode>>) {
    let mut len = 0usize;
    let mut p = head.as_ref();
    while let Some(node) = p {
        len += 1;
        p = node.next.as_ref();
    }
    let mid = len / 2;

    let mut cur = &mut head;
    for _ in 0..(mid - 1) {
        cur = &mut cur.as_mut().unwrap().next;
    }
    let right = cur.as_mut().unwrap().next.take();
    (head, right)
}

fn merge(a: Option<Box<ListNode>>, b: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    match (a, b) {
        (None, r) => r,
        (l, None) => l,
        (Some(mut x), Some(mut y)) => {
            if x.val <= y.val {
                let next = x.next.take();
                x.next = merge(next, Some(y));
                Some(x)
            } else {
                let next = y.next.take();
                y.next = merge(Some(x), next);
                Some(y)
            }
        }
    }
}

fn build(nums: &[i32]) -> Option<Box<ListNode>> {
    let mut head: Option<Box<ListNode>> = None;
    let mut tail = &mut head;
    for &v in nums {
        *tail = Some(Box::new(ListNode::new(v)));
        if let Some(node) = tail {
            tail = &mut node.next;
        }
    }
    head
}

fn dump(mut head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut out = Vec::new();
    while let Some(mut node) = head {
        out.push(node.val);
        head = node.next.take();
    }
    out
}

fn main() {
    let h = build(&[4, 2, 1, 3]);
    let ans = sort_list(h);
    println!("{:?}", dump(ans)); // [1, 2, 3, 4]
}
```

### JavaScript

```javascript
function ListNode(val = 0, next = null) {
  this.val = val;
  this.next = next;
}

function merge(a, b) {
  const dummy = new ListNode();
  let tail = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      tail.next = a;
      a = a.next;
    } else {
      tail.next = b;
      b = b.next;
    }
    tail = tail.next;
  }
  tail.next = a || b;
  return dummy.next;
}

function sortList(head) {
  if (!head || !head.next) return head;

  let prev = null;
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    prev = slow;
    slow = slow.next;
    fast = fast.next.next;
  }
  prev.next = null;

  const left = sortList(head);
  const right = sortList(slow);
  return merge(left, right);
}

function build(arr) {
  const dummy = new ListNode();
  let tail = dummy;
  for (const v of arr) {
    tail.next = new ListNode(v);
    tail = tail.next;
  }
  return dummy.next;
}

function dump(head) {
  const out = [];
  while (head) {
    out.push(head.val);
    head = head.next;
  }
  return out;
}

const h = build([4, 2, 1, 3]);
console.log(dump(sortList(h))); // [1, 2, 3, 4]
```

---

## 行动号召（CTA）

建议你现在马上做两步强化：

1. 不看答案手写一遍“找中点+断链+合并”模板。  
2. 继续练 `LeetCode 23`（合并 k 个有序链表），把归并思想从二路扩展到多路。

把这两题打通后，链表与分治模块会明显稳定很多。
