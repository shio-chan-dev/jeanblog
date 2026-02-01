---
title: "Hot100：合并两个有序链表（Merge Two Sorted Lists）哨兵节点归并 ACERS 解析"
date: 2026-02-01T21:40:06+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "双指针", "归并", "哨兵节点", "LeetCode 21"]
description: "用哨兵节点 + 双指针在 O(m+n) 时间把两个升序链表“拼接式”合并为一个升序链表，并对比递归写法与常见坑，附多语言可运行实现（Hot100）。"
keywords: ["Merge Two Sorted Lists", "合并两个有序链表", "归并", "哨兵节点", "双指针", "LeetCode 21", "Hot100", "O(m+n)"]
---

> **副标题 / 摘要**  
> 这是链表版的“归并排序合并步骤”：两条升序链表像两根排好队的队伍，比较头部把更小的节点接到结果尾部即可。本文用 ACERS 结构把哨兵节点迭代写法讲透，并给出递归对照与多语言可运行实现。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`链表`、`归并`、`双指针`  
- **SEO 关键词**：Hot100, Merge Two Sorted Lists, 合并两个有序链表, 归并, 哨兵节点, LeetCode 21  
- **元描述**：哨兵节点 + 双指针 O(m+n) 合并两个升序链表，附递归对比、工程迁移与多语言实现。  

---

## 目标读者

- 正在刷 Hot100 / 准备面试的同学  
- 写链表题经常丢头/断链、希望建立稳定模板的中级开发者  
- 需要在 C/C++/Go/Rust 等语言里熟练做“拼接式合并”的工程师

## 背景 / 动机

“合并两个有序链表”看上去是简单题，但它非常像工程里的真实任务：

- 合并两路已排序的数据流（日志、事件、时间线）  
- 合并两份排序好的列表（缓存片段、分片结果、分页结果）  
- 在 O(1) 额外空间下复用节点，避免额外分配与拷贝

更重要的是：它是很多题的前置技能（如合并 k 个链表、排序链表、分治归并）。  
把这个模板写熟，你后续的链表题会明显更顺。

## 核心概念

- **升序链表**：沿 `next` 方向节点值非递减  
- **拼接式合并（splicing）**：不创建新节点（除哨兵节点外），只重连 `next` 指针把节点接到结果链表  
- **哨兵节点（dummy/sentinel）**：用一个虚拟头简化“结果链表头是谁”的特判  
- **尾指针（tail）**：始终指向结果链表的最后一个节点，方便 O(1) 追加

---

## A — Algorithm（题目与算法）

### 题目还原

给你两个升序链表 `list1` 和 `list2` 的头节点，  
请将它们合并为一个新的 **升序** 链表并返回。  
新链表是通过 **拼接** 给定的两个链表的所有节点组成的。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| list1 | ListNode | 升序链表 1 的头节点（可能为空） |
| list2 | ListNode | 升序链表 2 的头节点（可能为空） |
| 返回 | ListNode | 合并后的升序链表头节点 |

### 示例 1（自拟）

```text
list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4
输出:  1 -> 1 -> 2 -> 3 -> 4 -> 4
```

### 示例 2（自拟）

```text
list1: null
list2: 0 -> 5
输出:  0 -> 5
```

---

## C — Concepts（核心思想）

### 思路推导：从“拉平排序”到“归并拼接”

1. **朴素思路：把节点值拉平到数组再排序**  
   - 遍历两条链表，把值放入数组  
   - 排序后再重建链表  
   缺点：O(m+n) 额外空间；而且“重建节点”不满足“拼接节点”的语义（面试常追问）。

2. **关键观察：两条链表已经分别有序**  
   这跟归并排序的合并步骤一模一样：  
   两个指针分别指向两条链表当前头部，每次把较小者接到结果尾部，再向前移动该指针。

3. **方法选择：哨兵节点 + 双指针（稳定、无特判）**  
   - 用 `dummy` 做结果链表的虚拟头  
   - 用 `tail` 指向结果尾部  
   - 维护 `p1/p2` 指向 list1/list2 当前节点  
   - 每次比较 `p1.val` 与 `p2.val`，把更小的节点接到 `tail.next`

### 方法归类

- **双指针归并（Two-pointer Merge）**  
- **链表原地拼接（In-place Splicing）**  
- **哨兵节点消除边界特判**

### 循环不变量（写对的关键）

在每次循环开始时保持：

- `dummy.next .. tail` 已经是升序的合并结果（包含了已经取走的节点）  
- `p1` 和 `p2` 分别指向两条链表尚未合并的最小节点  

循环一次后，合并结果长度 +1，且仍保持升序。

当一条链表耗尽（p1 或 p2 为 null），把另一条剩余部分直接接到 `tail.next` 即可（因为剩余部分本来就有序且都 ≥ 当前 tail）。

---

## 实践指南 / 步骤

1. 建立哨兵节点 `dummy`，尾指针 `tail = dummy`  
2. 指针 `p1 = list1`，`p2 = list2`  
3. 当 `p1` 与 `p2` 都非空时：  
   - 若 `p1.val <= p2.val`，接上 `p1` 并移动 `p1`  
   - 否则接上 `p2` 并移动 `p2`  
   - 同时移动 `tail` 到新尾部  
4. 退出循环后，把非空的那条剩余链表整体接到 `tail.next`  
5. 返回 `dummy.next`

Python 可运行示例（保存为 `merge_two_lists.py`）：

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    p1, p2 = list1, list2

    while p1 is not None and p2 is not None:
        if p1.val <= p2.val:
            nxt = p1.next
            tail.next = p1
            p1.next = None
            p1 = nxt
        else:
            nxt = p2.next
            tail.next = p2
            p2.next = None
            p2 = nxt
        tail = tail.next

    tail.next = p1 if p1 is not None else p2
    return dummy.next


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
    l1 = from_list([1, 2, 4])
    l2 = from_list([1, 3, 4])
    print(to_list(merge_two_lists(l1, l2)))
    print(to_list(merge_two_lists(None, from_list([0, 5]))))
```

> 注：上面在拼接时把 `p1.next = None` / `p2.next = None` 断开旧指针，是一种“防误用”的工程习惯；不写也能过题，但写上更不容易意外形成长链残留。

---

## E — Engineering（工程应用）

> 归并不仅是算法题，更是“按排序键合并两路数据”的通用能力。  
> 链表版归并对应的是“只改链接关系、不拷贝对象”，对性能/内存更友好。

### 场景 1：合并两路按时间排序的事件流（Go）

**背景**：服务端常把事件按时间排序（trace、审计日志、业务事件）。  
**为什么适用**：两路来源各自已排序，合并时只需线性扫描，适合在线处理。

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
            nxt := a.Next
            tail.Next = a
            a.Next = nil
            a = nxt
        } else {
            nxt := b.Next
            tail.Next = b
            b.Next = nil
            b = nxt
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
    a := &Node{1, &Node{3, &Node{7, nil}}}
    b := &Node{2, &Node{4, &Node{8, nil}}}
    head := merge(a, b)
    for p := head; p != nil; p = p.Next {
        fmt.Print(p.Ts, " ")
    }
    fmt.Println()
}
```

### 场景 2：数据分析里合并两份已排序 ID 列表（Python）

**背景**：离线任务常会拿到两份已排序的 ID（比如两种规则筛选结果）。  
**为什么适用**：线性归并能得到整体排序输出，也能顺便做去重/计数等扩展。

```python
def merge_sorted(a, b):
    i = j = 0
    res = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i]); i += 1
        else:
            res.append(b[j]); j += 1
    res.extend(a[i:])
    res.extend(b[j:])
    return res


if __name__ == "__main__":
    print(merge_sorted([1, 2, 4], [1, 3, 4]))
```

### 场景 3：系统编程中合并按地址排序的空闲块链（C）

**背景**：简化版内存管理中，空闲块可能按地址排序维护，便于后续合并相邻块。  
**为什么适用**：合并两条“已排序空闲链”是归并的直接应用，且可以复用节点不额外分配。

```c
struct Node {
    int addr;
    struct Node* next;
};

struct Node* merge(struct Node* a, struct Node* b) {
    struct Node dummy;
    struct Node* tail = &dummy;
    dummy.next = 0;

    while (a && b) {
        if (a->addr <= b->addr) {
            struct Node* nxt = a->next;
            tail->next = a;
            a->next = 0;
            a = nxt;
        } else {
            struct Node* nxt = b->next;
            tail->next = b;
            b->next = 0;
            b = nxt;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- **时间复杂度**：O(m+n)，每个节点最多被访问/拼接一次  
- **空间复杂度**：O(1)（迭代版，除哨兵外不分配新节点）；递归版为 O(m+n)（调用栈）

### 替代方案对比

| 方法 | 思路 | 额外空间 | 典型问题 |
| --- | --- | --- | --- |
| 拉平排序 | 收集值/节点再排序 | O(m+n) | 破坏“拼接节点”语义；多一次排序 |
| 递归归并 | 选小者为头，递归合并剩余 | O(m+n) 栈 | 深度太大可能栈溢出 |
| **迭代哨兵（本文）** | tail 逐步拼接 | **O(1)** | 要避免断链/丢头 |

### 常见错误与注意事项

1. **忘记把剩余链表接上**：循环退出后必须 `tail.next = p1 or p2`。  
2. **比较值相等时的选择**：用 `<=` 优先选 list1，可保证“稳定”（相同值时保持 list1 在前）。  
3. **误创建新节点**：题目强调“拼接给定节点”，面试官可能追问是否复用节点。  
4. **断链/成环**：在工程里推荐断开被拼接节点的 `next`（如示例），减少“旧链接残留”导致的 bug。

---

## 解释与原理（为什么这么做）

这就是归并排序的合并步骤：  
每次只做局部最优选择（选两者头部较小者），全局结果仍有序，因为：

- 两条链表各自有序 ⇒ 当前头部是该链表剩余部分的最小值  
- 选择两者头部较小者 ⇒ 该值也是所有剩余节点中的最小值  
- 把它接到结果尾部后，结果仍保持非递减  

重复直到某一条链表用尽，另一条剩余部分整体追加即可。

---

## 常见问题与注意事项

1. **list1 或 list2 为空怎么办？**  
   直接返回另一条即可；哨兵模板天然涵盖该情况。

2. **需要处理重复值吗？**  
   需要。用 `<=` 或 `<` 都可以得到有序结果；`<=` 更稳定、更可预测。

3. **为什么不推荐递归？**  
   递归很短，但链表很长时有栈深风险；工程上更偏向迭代模板。

---

## 最佳实践与建议

- 先写哨兵：`dummy` + `tail`，把“头节点是谁”的特判消掉  
- while 循环只做三件事：比较 → 拼接 → 推进指针  
- 退出循环后一行接上剩余链表  
- 在工程代码里，拼接时可选择断开 `next`（防止旧链残留）

---

## S — Summary（总结）

### 核心收获

- 合并两个升序链表 = 归并排序的合并步骤  
- 哨兵节点可以彻底消除“结果头节点”特判  
- 双指针线性扫描即可 O(m+n) 合并完成  
- 迭代法 O(1) 额外空间，更适合工程  

### 小结 / 结论

LeetCode 21 是链表归并的“第一块积木”。  
掌握这题的哨兵迭代模板，你就能顺滑过渡到“合并 k 个链表 / 排序链表 / 分治归并”等更高频的链表题。

### 参考与延伸阅读

- LeetCode 21. Merge Two Sorted Lists  
- LeetCode 23. Merge k Sorted Lists  
- LeetCode 148. Sort List（归并排序 + 链表拆分）

---

## 元信息

- **阅读时长**：10~12 分钟  
- **标签**：Hot100、链表、归并、双指针、LeetCode 21  
- **SEO 关键词**：Hot100, Merge Two Sorted Lists, 合并两个有序链表, 归并, 哨兵节点  
- **元描述**：哨兵节点 + 双指针 O(m+n) 合并两个升序链表，附递归对比与多语言实现。  

---

## 行动号召（CTA）

做完这题后，建议立刻用同一套“归并模板”继续刷两题巩固：

1) LeetCode 23（合并 k 个链表）  
2) LeetCode 148（排序链表）

如果你希望我把 23/148 也按 Hot100 的 ACERS 风格写出来，告诉我即可。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    p1, p2 = list1, list2

    while p1 is not None and p2 is not None:
        if p1.val <= p2.val:
            nxt = p1.next
            tail.next = p1
            p1.next = None
            p1 = nxt
        else:
            nxt = p2.next
            tail.next = p2
            p2.next = None
            p2 = nxt
        tail = tail.next

    tail.next = p1 if p1 is not None else p2
    return dummy.next


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
    l1 = from_list([1, 2, 4])
    l2 = from_list([1, 3, 4])
    print(to_list(mergeTwoLists(l1, l2)))
```

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2) {
    struct ListNode dummy;
    struct ListNode* tail = &dummy;
    dummy.next = NULL;

    while (l1 && l2) {
        if (l1->val <= l2->val) {
            struct ListNode* nxt = l1->next;
            tail->next = l1;
            l1->next = NULL;
            l1 = nxt;
        } else {
            struct ListNode* nxt = l2->next;
            tail->next = l2;
            l2->next = NULL;
            l2 = nxt;
        }
        tail = tail->next;
    }
    tail->next = l1 ? l1 : l2;
    return dummy.next;
}

static struct ListNode* push_back(struct ListNode* tail, int v) {
    struct ListNode* node = (struct ListNode*)malloc(sizeof(struct ListNode));
    node->val = v;
    node->next = NULL;
    tail->next = node;
    return node;
}

static struct ListNode* from_array(const int* a, int n) {
    struct ListNode dummy;
    struct ListNode* tail = &dummy;
    dummy.next = NULL;
    for (int i = 0; i < n; ++i) tail = push_back(tail, a[i]);
    return dummy.next;
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
        printf("%d", p->val);
        if (p->next) printf(" -> ");
    }
    printf("\n");
}

int main(void) {
    int a[] = {1, 2, 4};
    int b[] = {1, 3, 4};
    struct ListNode* l1 = from_array(a, 3);
    struct ListNode* l2 = from_array(b, 3);
    struct ListNode* m = mergeTwoLists(l1, l2);
    print_list(m);
    free_list(m);
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

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            ListNode* nxt = l1->next;
            tail->next = l1;
            l1->next = nullptr;
            l1 = nxt;
        } else {
            ListNode* nxt = l2->next;
            tail->next = l2;
            l2->next = nullptr;
            l2 = nxt;
        }
        tail = tail->next;
    }
    tail->next = l1 ? l1 : l2;
    return dummy.next;
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
        std::cout << p->val << (p->next ? " -> " : "\n");
    }
}

int main() {
    ListNode* l1 = fromVec({1, 2, 4});
    ListNode* l2 = fromVec({1, 3, 4});
    ListNode* m = mergeTwoLists(l1, l2);
    printList(m);
    freeList(m);
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

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    tail := dummy
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            nxt := l1.Next
            tail.Next = l1
            l1.Next = nil
            l1 = nxt
        } else {
            nxt := l2.Next
            tail.Next = l2
            l2.Next = nil
            l2 = nxt
        }
        tail = tail.Next
    }
    if l1 != nil {
        tail.Next = l1
    } else {
        tail.Next = l2
    }
    return dummy.Next
}

func main() {
    l1 := &ListNode{1, &ListNode{2, &ListNode{4, nil}}}
    l2 := &ListNode{1, &ListNode{3, &ListNode{4, nil}}}
    head := mergeTwoLists(l1, l2)
    for p := head; p != nil; p = p.Next {
        fmt.Print(p.Val, " ")
    }
    fmt.Println()
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

pub fn merge_two_lists(
    mut l1: Option<Box<ListNode>>,
    mut l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut dummy = Box::new(ListNode::new(0));
    let mut tail: &mut Box<ListNode> = &mut dummy;

    while l1.is_some() && l2.is_some() {
        let take_l1 = l1.as_ref().unwrap().val <= l2.as_ref().unwrap().val;
        if take_l1 {
            let mut node = l1.take().unwrap();
            l1 = node.next.take();
            tail.next = Some(node);
        } else {
            let mut node = l2.take().unwrap();
            l2 = node.next.take();
            tail.next = Some(node);
        }
        tail = tail.next.as_mut().unwrap();
    }

    tail.next = if l1.is_some() { l1 } else { l2 };
    dummy.next
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
    let l1 = from_vec(&[1, 2, 4]);
    let l2 = from_vec(&[1, 3, 4]);
    let m = merge_two_lists(l1, l2);
    println!("{:?}", to_vec(&m));
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function mergeTwoLists(l1, l2) {
  const dummy = new ListNode(0);
  let tail = dummy;
  let p1 = l1, p2 = l2;

  while (p1 && p2) {
    if (p1.val <= p2.val) {
      const nxt = p1.next;
      tail.next = p1;
      p1.next = null;
      p1 = nxt;
    } else {
      const nxt = p2.next;
      tail.next = p2;
      p2.next = null;
      p2 = nxt;
    }
    tail = tail.next;
  }
  tail.next = p1 ? p1 : p2;
  return dummy.next;
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

const l1 = fromArray([1, 2, 4]);
const l2 = fromArray([1, 3, 4]);
console.log(toArray(mergeTwoLists(l1, l2)));
```
