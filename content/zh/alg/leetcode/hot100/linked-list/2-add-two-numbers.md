---
title: "LeetCode 2：两数相加（Add Two Numbers）链表进位从朴素到最优解"
date: 2026-02-11T07:54:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "进位", "模拟", "LeetCode 2", "算法题解"]
description: "把两个逆序链表表示的非负整数相加，关键是逐位相加与进位传播。本文用 ACERS 结构讲透思路推导、工程类比与多语言可运行实现。"
keywords: ["Add Two Numbers", "两数相加", "链表进位", "LeetCode 2", "dummy node", "carry"]
---

> **副标题 / 摘要**  
> 这题本质是把「小学竖式加法」搬到链表：同位相加、处理进位、走到末尾后可能还要补一个新节点。文章将从朴素思路推到最优单遍解法，并给出工程场景与多语言实现。

- **预计阅读时长**：12~15 分钟  
- **标签**：`链表`、`进位`、`模拟`、`LeetCode 2`  
- **SEO 关键词**：Add Two Numbers, 两数相加, 逆序链表, 进位, LeetCode 2  
- **元描述**：用哨兵节点 + 单遍遍历在 O(max(m,n)) 时间完成两条逆序数字链表求和，附常见坑、工程应用和六语言代码。  

---

## 目标读者

- 刚开始刷链表题，想建立稳定解题模板的同学
- 对「进位」和「边界处理」容易写错的中级开发者
- 希望把算法思维迁移到工程数据流处理的工程师

## 背景 / 动机

看似只是 LeetCode 入门题，但它练的能力非常实用：

- 多输入流同步推进（`l1`、`l2` 两个指针）
- 状态跨轮传播（`carry` 进位）
- 边界完整性（长度不同、最后一位进位）

这三点在工程里非常常见，例如金额分片累加、多源日志计数合并、流式统计补位等。

## 核心概念

- **逆序存储**：个位在链表头部，十位在下一节点，以此类推
- **逐位相加**：每轮只处理一个位，值来自 `x + y + carry`
- **进位传播**：`carry = sum // 10`，当前位 `digit = sum % 10`
- **哨兵节点（dummy）**：避免首次插入时区分“头节点是否为空”

---

## A — Algorithm（题目与算法）

### 题目重述

给你两个**非空**链表，表示两个非负整数。  
数字按**逆序**存储，且每个节点存储一位数字。  
请将两个数相加，并返回同样逆序存储的结果链表。  
题目保证除数字 `0` 外，这两个数都不会以 `0` 开头。

### 输入输出描述

| 项目 | 含义 |
| --- | --- |
| 输入 | 两个链表 `l1`、`l2`，每个节点值在 `0~9` |
| 输出 | 一个新链表，表示 `l1 + l2` 的结果（逆序） |

### 示例 1

```text
输入: l1 = [2,4,3], l2 = [5,6,4]
解释: 342 + 465 = 807
输出: [7,0,8]
```

### 示例 2

```text
输入: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
解释: 9999999 + 9999 = 10009998
输出: [8,9,9,9,0,0,0,1]
```

---

## 思路推导：从朴素到最优

### 朴素思路 1：先转整数再相加

- 把链表转成整数 `n1`、`n2`
- 做 `n1 + n2`
- 再把结果拆位转回链表

问题：

- 在很多语言里会溢出（数字位数很长）
- 额外做了「构造大整数」和「拆整数」两次转换
- 偏离题目本质（链表逐位处理）

### 朴素思路 2：先转数组再按位加

- 把两链表都转数组
- 再从低位到高位相加

问题：

- 需要 O(m+n) 额外空间
- 其实链表已是低位在前，不需要再中转

### 关键观察

- 链表本来就是从个位开始，正好适合「竖式加法」顺序
- 每轮只需要当前两位 + 进位，不依赖更高位
- 因此可以单遍扫描完成

### 方法选择

使用 `dummy + tail` 构建结果链表，循环条件为：

```text
while l1 != null or l2 != null or carry != 0
```

每轮：

1. 取当前位 `x`、`y`（空节点当 0）
2. `sum = x + y + carry`
3. 新节点值 `sum % 10`
4. 更新 `carry = sum // 10`

---

## C — Concepts（核心思想）

### 算法类型归类

- **链表模拟**
- **进位状态机**
- **双指针同步遍历**

### 状态模型

令第 `k` 轮输入位为 `x_k`、`y_k`，进位为 `c_k`，则：

```text
s_k = x_k + y_k + c_k
digit_k = s_k mod 10
c_(k+1) = floor(s_k / 10)
```

其中 `c_k ∈ {0,1}`。  
这个模型就是十进制逐位加法的数学表达。

### 正确性直觉

- 每轮产出的 `digit_k` 就是结果的第 `k` 位
- `carry` 把“超过 9 的部分”准确传给下一轮
- 当两链表都结束但 `carry=1` 时，补一个末尾节点即可

---

## 实践指南 / 步骤

1. 初始化 `dummy` 和 `tail`，`carry = 0`
2. 进入循环：任一链表未结束或仍有 `carry`
3. 读取当前位：`x = l1.val if l1 else 0`，`y = l2.val if l2 else 0`
4. 计算 `sum`，创建新节点 `sum % 10` 接到 `tail.next`
5. 更新 `carry = sum // 10`，移动 `tail` 和输入指针
6. 返回 `dummy.next`

Python 最小可运行示例：

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    carry = 0

    while l1 is not None or l2 is not None or carry:
        x = l1.val if l1 is not None else 0
        y = l2.val if l2 is not None else 0
        s = x + y + carry
        carry = s // 10
        tail.next = ListNode(s % 10)
        tail = tail.next

        if l1 is not None:
            l1 = l1.next
        if l2 is not None:
            l2 = l2.next

    return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for n in nums:
        tail.next = ListNode(n)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head is not None:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    a = build([2, 4, 3])
    b = build([5, 6, 4])
    print(dump(add_two_numbers(a, b)))  # [7, 0, 8]
```

---

## E — Engineering（工程应用）

### 场景 1：财务分片金额逐位合并（Python）

**背景**：部分账务系统会把超长金额做分片存储或传输。  
**为什么适用**：每个分片可看作一位或一组位，核心都是“同位相加 + 进位传播”。

```python
def add_digits(a, b):
    i = j = 0
    carry = 0
    out = []
    while i < len(a) or j < len(b) or carry:
        x = a[i] if i < len(a) else 0
        y = b[j] if j < len(b) else 0
        s = x + y + carry
        out.append(s % 10)
        carry = s // 10
        i += 1
        j += 1
    return out


print(add_digits([2, 4, 3], [5, 6, 4]))  # [7,0,8]
```

### 场景 2：后台服务多源计数流拼接（Go）

**背景**：两个服务分别上报低位优先的计数块。  
**为什么适用**：按位合并后继续上报，内存占用稳定、可流式处理。

```go
package main

import "fmt"

func addDigits(a, b []int) []int {
	i, j, carry := 0, 0, 0
	out := make([]int, 0)
	for i < len(a) || j < len(b) || carry > 0 {
		x, y := 0, 0
		if i < len(a) {
			x = a[i]
			i++
		}
		if j < len(b) {
			y = b[j]
			j++
		}
		s := x + y + carry
		out = append(out, s%10)
		carry = s / 10
	}
	return out
}

func main() {
	fmt.Println(addDigits([]int{9, 9, 9}, []int{1})) // [0 0 0 1]
}
```

### 场景 3：前端离线草稿版本号累加（JavaScript）

**背景**：离线编辑器可能把超长版本号拆位缓存。  
**为什么适用**：浏览器端不依赖大整数库即可安全处理长数字。

```javascript
function addDigits(a, b) {
  let i = 0;
  let j = 0;
  let carry = 0;
  const out = [];

  while (i < a.length || j < b.length || carry) {
    const x = i < a.length ? a[i++] : 0;
    const y = j < b.length ? b[j++] : 0;
    const s = x + y + carry;
    out.push(s % 10);
    carry = Math.floor(s / 10);
  }
  return out;
}

console.log(addDigits([2, 4, 3], [5, 6, 4])); // [7,0,8]
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(max(m, n))`
- 空间复杂度：`O(max(m, n))`（结果链表本身）；额外辅助空间为 `O(1)`

### 替代方案对比

| 方案 | 时间 | 额外空间 | 问题 |
| --- | --- | --- | --- |
| 转整数后相加 | O(m+n) | 取决于大整数实现 | 易溢出或依赖大整数库 |
| 转数组再相加 | O(m+n) | O(m+n) | 不必要中转 |
| 单遍链表模拟（本解） | O(max(m,n)) | O(1) 辅助 | 边界清晰，工程可用 |

### 常见错误思路

- 漏掉循环条件里的 `carry != 0`，导致 `999 + 1` 少一位
- 长度不等时直接访问空指针
- 试图原地复用输入链表，导致代码分支复杂、可读性下降

### 为什么当前方法最优/最工程可行

- 单遍遍历，逻辑直接映射十进制加法
- 不依赖语言的大整数能力
- 边界统一，易测试、易迁移到任意语言

---

## 常见问题与注意事项（FAQ）

### Q1：为什么循环条件必须包含 `carry`？

因为最后一轮可能两链表都走完了，但仍有进位。例如 `5 + 5 = 10`，还需要再输出一位 `1`。

### Q2：可以原地修改 `l1` 或 `l2` 吗？

可以，但会增加分支复杂度，且可能破坏调用方对输入链表的复用预期。面试与工程里更推荐新建结果链表。

### Q3：如果数字是正序存储怎么办？

那是另一题（LeetCode 445），常用栈或递归从高位回卷处理，不同于本题的低位优先模型。

---

## 最佳实践与建议

- 固定模板：`dummy + tail + carry`，不要每次重写分支
- 用 `while l1 or l2 or carry` 一次收敛所有边界
- 写 3 组回归用例：等长、非等长、全进位链
- 把“取值为空视为 0”写成同一段，减少判空散落

---

## S — Summary（总结）

核心收获：

1. 逆序链表求和的本质是十进制逐位加法状态机。
2. `carry` 是跨轮状态，必须进入循环条件统一处理。
3. `dummy` 节点可以显著减少头节点特判，提升稳定性。
4. 该题是链表模拟、双指针与边界管理的入门基石。
5. 方法可直接迁移到工程里的分片数值/流式计数合并问题。

推荐延伸阅读：

- LeetCode 445 `Add Two Numbers II`（正序链表求和）
- LeetCode 21 `Merge Two Sorted Lists`（链表双指针模板）
- LeetCode 206 `Reverse Linked List`（链表基础操作）
- CLRS / 算法导论中关于链表与基本数据结构章节

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
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        carry = 0

        while l1 is not None or l2 is not None or carry:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            s = x + y + carry
            carry = s // 10
            tail.next = ListNode(s % 10)
            tail = tail.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy.next


def build(nums: List[int]) -> Optional[ListNode]:
    d = ListNode()
    t = d
    for v in nums:
        t.next = ListNode(v)
        t = t.next
    return d.next


def dump(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    ans = Solution().addTwoNumbers(build([2, 4, 3]), build([5, 6, 4]))
    print(dump(ans))  # [7, 0, 8]
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

struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    struct ListNode dummy;
    dummy.val = 0;
    dummy.next = NULL;
    struct ListNode* tail = &dummy;
    int carry = 0;

    while (l1 != NULL || l2 != NULL || carry != 0) {
        int x = (l1 != NULL) ? l1->val : 0;
        int y = (l2 != NULL) ? l2->val : 0;
        int s = x + y + carry;
        carry = s / 10;

        tail->next = new_node(s % 10);
        tail = tail->next;

        if (l1 != NULL) l1 = l1->next;
        if (l2 != NULL) l2 = l2->next;
    }

    return dummy.next;
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
    while (h != NULL) {
        printf("%d", h->val);
        if (h->next != NULL) printf(" -> ");
        h = h->next;
    }
    printf("\n");
}

void free_list(struct ListNode* h) {
    while (h != NULL) {
        struct ListNode* nxt = h->next;
        free(h);
        h = nxt;
    }
}

int main(void) {
    int a[] = {2, 4, 3};
    int b[] = {5, 6, 4};
    struct ListNode* l1 = build(a, 3);
    struct ListNode* l2 = build(b, 3);
    struct ListNode* ans = addTwoNumbers(l1, l2);
    print_list(ans); // 7 -> 0 -> 8
    free_list(l1);
    free_list(l2);
    free_list(ans);
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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        int carry = 0;

        while (l1 || l2 || carry) {
            int x = l1 ? l1->val : 0;
            int y = l2 ? l2->val : 0;
            int s = x + y + carry;
            carry = s / 10;
            tail->next = new ListNode(s % 10);
            tail = tail->next;

            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
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
    cout << '\n';
}

void freeList(ListNode* h) {
    while (h) {
        ListNode* nxt = h->next;
        delete h;
        h = nxt;
    }
}

int main() {
    ListNode* l1 = build({2, 4, 3});
    ListNode* l2 = build({5, 6, 4});
    ListNode* ans = Solution().addTwoNumbers(l1, l2);
    printList(ans); // 7 -> 0 -> 8
    freeList(l1);
    freeList(l2);
    freeList(ans);
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

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	carry := 0

	for l1 != nil || l2 != nil || carry != 0 {
		x, y := 0, 0
		if l1 != nil {
			x = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			y = l2.Val
			l2 = l2.Next
		}
		s := x + y + carry
		carry = s / 10
		tail.Next = &ListNode{Val: s % 10}
		tail = tail.Next
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
	l1 := build([]int{2, 4, 3})
	l2 := build([]int{5, 6, 4})
	ans := addTwoNumbers(l1, l2)
	printList(ans) // 7 -> 0 -> 8
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
        ListNode { next: None, val }
    }
}

pub fn add_two_numbers(
    mut l1: Option<Box<ListNode>>,
    mut l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut digits: Vec<i32> = Vec::new();
    let mut carry = 0;

    while l1.is_some() || l2.is_some() || carry > 0 {
        let mut x = 0;
        let mut y = 0;

        if let Some(mut node) = l1 {
            x = node.val;
            l1 = node.next.take();
        } else {
            l1 = None;
        }

        if let Some(mut node) = l2 {
            y = node.val;
            l2 = node.next.take();
        } else {
            l2 = None;
        }

        let s = x + y + carry;
        carry = s / 10;
        digits.push(s % 10);
    }

    let mut head: Option<Box<ListNode>> = None;
    let mut tail = &mut head;
    for d in digits {
        *tail = Some(Box::new(ListNode::new(d)));
        if let Some(node) = tail {
            tail = &mut node.next;
        }
    }
    head
}

fn build(nums: &[i32]) -> Option<Box<ListNode>> {
    let mut head: Option<Box<ListNode>> = None;
    let mut tail = &mut head;
    for &n in nums {
        *tail = Some(Box::new(ListNode::new(n)));
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
    let l1 = build(&[2, 4, 3]);
    let l2 = build(&[5, 6, 4]);
    let ans = add_two_numbers(l1, l2);
    println!("{:?}", dump(ans)); // [7, 0, 8]
}
```

### JavaScript

```javascript
function ListNode(val = 0, next = null) {
  this.val = val;
  this.next = next;
}

function addTwoNumbers(l1, l2) {
  const dummy = new ListNode(0);
  let tail = dummy;
  let carry = 0;

  while (l1 !== null || l2 !== null || carry !== 0) {
    const x = l1 ? l1.val : 0;
    const y = l2 ? l2.val : 0;
    const s = x + y + carry;
    carry = Math.floor(s / 10);

    tail.next = new ListNode(s % 10);
    tail = tail.next;

    if (l1) l1 = l1.next;
    if (l2) l2 = l2.next;
  }

  return dummy.next;
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

const ans = addTwoNumbers(build([2, 4, 3]), build([5, 6, 4]));
console.log(dump(ans)); // [7, 0, 8]
```

---

## 行动号召（CTA）

如果你在写这题时经常卡在边界条件，建议你现在就做两件事：

1. 手写一遍 `while l1 or l2 or carry` 模板，不看答案完成。
2. 再做 LeetCode 445，对比逆序与正序链表加法的差异。

你也可以在下一篇继续挑战：`LeetCode 25` 或 `LeetCode 142`，把链表题的“指针基本功”一次补齐。
