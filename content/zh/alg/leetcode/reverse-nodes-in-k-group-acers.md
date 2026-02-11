---
title: "LeetCode 25：K 个一组翻转链表（ACERS 全链路解析）"
date: 2026-02-11T07:58:23+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "分组反转", "双指针", "原地算法", "LeetCode 25", "ACERS"]
description: "系统讲透 LeetCode 25：从朴素思路到 O(n)/O(1) 的分组原地反转模板，覆盖边界处理、工程迁移与 Python/C/C++/Go/Rust/JS 多语言实现。"
keywords: ["K 个一组翻转链表", "Reverse Nodes in k-Group", "LeetCode 25", "链表分组反转", "dummy node", "双指针"]
---

> **副标题 / 摘要**  
> 这题难点不在“反转链表”，而在“分组判定 + 组间拼接 + 尾组保留”。本文按 ACERS 结构给出可复用模板，帮助你稳定写出不丢节点、不断链的原地解法。

- **预计阅读时长**：14~18 分钟  
- **标签**：`链表`、`分组反转`、`双指针`、`原地算法`  
- **SEO 关键词**：K 个一组翻转链表, Reverse Nodes in k-Group, LeetCode 25, 链表分组反转  

## 目标读者

- 已掌握 206 反转链表，但在“分段反转”容易错边界的同学
- 准备面试高频链表题，想建立稳定模板的开发者
- 希望把算法套路迁移到工程批处理链式结构的工程师

## 背景 / 动机

在工程里，链式结构常出现“按批处理”的需求：批量回放、批次重排、分段回滚。  
这类问题的共性是：

- 每批有固定规模（k）
- 批内需要变换（本题是反转）
- 尾部不足 k 的残组需保持原序

LeetCode 25 是这个模式的标准训练题。

## 核心概念

- **dummy（哑节点）**：统一处理“头节点参与反转”的情况，避免特判。
- **groupPrev（组前驱）**：永远指向当前待处理组的前一个节点。
- **kth（组尾探针）**：从 `groupPrev` 走 `k` 步，用于判断是否有完整组。
- **groupNext（组后继）**：当前组反转后要接回的后续链表头。
- **循环不变量**：`groupPrev` 左侧部分始终是最终正确结构。

## 思路推导（从朴素到最优）

### 1) 朴素解法：数组化后分组反转

- 把链表节点放进数组
- 每 k 个元素反转，再重建链表

问题：

- 额外空间 O(n)
- 题目要求“交换节点而非改值”，数组化方案不是最优表达

### 2) 关键瓶颈

难点不是“反转动作”，而是：

- 如何稳定找出“完整 k 组”
- 如何把反转后的组重新接回主链
- 如何保证不足 k 的尾部不动

### 3) 关键观察

只要每轮都做到：

1. 先确认完整 k 节点  
2. 只反转这一组  
3. 正确拼回，再推进到下一组

整题就能分治为重复模板，且每节点只访问常数次。

### 4) 方法选择

采用：

- `dummy + groupPrev + kth` 的分组框架
- 组内原地反转，`prev` 初始设为 `groupNext`

最终达到：

- 时间复杂度 O(n)
- 额外空间 O(1)

## A — Algorithm（题目与算法）

### 题目还原

给你链表头节点 `head`，每 `k` 个节点一组翻转，返回修改后的链表。  
若节点总数不是 `k` 的整数倍，最后剩余节点保持原顺序。  
不能只改节点值，必须做真实节点交换（改指针）。

### 输入输出

- 输入：`head: ListNode`，`k: int`
- 输出：分组翻转后的链表头节点

### 示例 1

```text
输入: head = [1,2,3,4,5], k = 2
输出: [2,1,4,3,5]
```

### 示例 2

```text
输入: head = [1,2,3,4,5], k = 3
输出: [3,2,1,4,5]
```

## C — Concepts（核心思想）

### 方法归类

- 链表原地重连（In-place Rewiring）
- 分段处理（Chunk Processing）
- 双指针边界定位（Predecessor + Kth）

### 概念模型

每轮操作前：

- `groupPrev.next` 是当前组首节点
- `groupPrev` 左侧已经是最终结构

每轮操作后：

- 当前组被正确反转并拼回主链
- `groupPrev` 前移到新组尾（反转前组首）

### 关键数据结构

- 单链表节点 `ListNode`
- 常数个辅助指针（`groupPrev/kth/groupNext/cur/prev`）

## 实践指南 / 步骤

1. 初始化：`dummy.next = head`, `groupPrev = dummy`
2. 从 `groupPrev` 走 `k` 步找 `kth`  
   - 找不到：说明不足 k，结束
3. 记录 `groupNext = kth.next`
4. 反转 `[groupPrev.next, kth]` 这一段  
   - 用 `prev = groupNext` 作为反转起点
5. 拼接：
   - `groupPrev.next` 接新组头
   - `groupPrev` 移到新组尾
6. 重复直到剩余不足 k

## 可运行示例（Python）

```python
from typing import Optional, List


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if head is None or k <= 1:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next

        group_next = kth.next
        prev = group_next
        cur = group_prev.next

        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        old_group_head = group_prev.next
        group_prev.next = prev
        group_prev = old_group_head


def build(nums: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    for x in nums:
        tail.next = ListNode(x)
        tail = tail.next
    return dummy.next


def dump(head: Optional[ListNode]) -> List[int]:
    ans = []
    while head:
        ans.append(head.val)
        head = head.next
    return ans


if __name__ == "__main__":
    head = build([1, 2, 3, 4, 5])
    print(dump(reverse_k_group(head, 2)))  # [2, 1, 4, 3, 5]
```

## 解释与原理

为什么 `prev = groupNext` 很关键？

- 反转后原组首会变成组尾，它应该指向下一段
- 把 `prev` 设为 `groupNext`，组尾指针在反转过程中自动就位

这能减少拼接分支，降低“断链/成环/丢节点”风险。

## E — Engineering（工程应用）

### 场景 1：数据清洗任务链按批回放（Python）

**背景**：日志处理流水线需要每 k 个任务逆序重跑。  
**为什么适用**：批内顺序调整、批间顺序保持、尾批不足 k 保留。

```python
def reverse_chunks(tasks, k):
    out = []
    for i in range(0, len(tasks), k):
        chunk = tasks[i:i + k]
        out.extend(reversed(chunk) if len(chunk) == k else chunk)
    return out


if __name__ == "__main__":
    print(reverse_chunks(["t1", "t2", "t3", "t4", "t5"], 2))
```

### 场景 2：高性能包处理链分批反向重排（C++）

**背景**：网络包处理链在回放窗口内按批次逆序重放。  
**为什么适用**：只改链接关系，避免复制数据块。

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> packets = {1, 2, 3, 4, 5, 6, 7};
    int k = 3;
    for (int i = 0; i + k <= (int)packets.size(); i += k) {
        reverse(packets.begin() + i, packets.begin() + i + k);
    }
    for (int x : packets) cout << x << " ";
    cout << "\n";
    return 0;
}
```

### 场景 3：后台任务队列按批重排（Go）

**背景**：微服务消费队列按批处理，失败批次需要局部回放。  
**为什么适用**：可以保证“完整批”重排而不影响残组。

```go
package main

import "fmt"

func reverseChunks(nums []int, k int) []int {
	out := make([]int, 0, len(nums))
	for i := 0; i < len(nums); i += k {
		end := i + k
		if end > len(nums) {
			out = append(out, nums[i:]...)
			break
		}
		for j := end - 1; j >= i; j-- {
			out = append(out, nums[j])
		}
	}
	return out
}

func main() {
	fmt.Println(reverseChunks([]int{1, 2, 3, 4, 5}, 2))
}
```

## 常见问题与注意事项

1. **不足 k 的尾组要反转吗？**  
   不要，按原顺序保留。

2. **为什么不能直接改节点值？**  
   题目要求真实节点交换，核心在指针重连能力。

3. **最容易错在哪里？**  
   在组边界拼接，尤其是反转后新头/新尾接回时。

4. **`k=1` 怎么处理？**  
   直接返回原链表，避免无意义指针操作。

## 最佳实践与建议

- 统一使用 `dummy`，不要手写头节点特判
- 每轮先找 `kth` 再反转，防止先改后发现不足 k
- 反转代码固定模板化，减少即兴改动
- 调试时优先打印每轮的组首、组尾、后继节点

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：**O(n)**  
  每个节点被扫描/重连常数次。
- 空间复杂度：**O(1)**  
  只使用固定数量指针变量。

### 替代方案对比

1. **暴力法（数组重建）**  
   - 优点：实现直观  
   - 缺点：空间 O(n)，不符合“链表指针交换”本质

2. **递归分组反转**  
   - 优点：表达简洁  
   - 缺点：递归栈 O(n/k)，工程中不如迭代稳定

3. **常见错误思路**  
   - 反转前不检查完整 k 组  
   - 反转后忘记接回 `groupNext`  
   - `groupPrev` 推进到错误节点导致死循环

### 为什么当前方案更可工程化

- 边界统一、可模板化复用
- 无额外线性内存
- 对长链输入更稳定，利于线上问题排查

## S — Summary（总结）

### 核心收获

1. 这题本质是“分组驱动的区间反转”，不是单纯反转链表。  
2. `dummy + groupPrev + kth` 是最稳的分组框架。  
3. `prev = groupNext` 能让组尾自动接回，显著降低断链风险。  
4. 先判完整组再反转，是保证尾组不动的关键。  
5. 该模板可迁移到批处理回放、链式任务重排等工程场景。

### 推荐延伸阅读

- LeetCode 206：Reverse Linked List  
- LeetCode 92：Reverse Linked List II  
- LeetCode 24：Swap Nodes in Pairs  
- *Introduction to Algorithms*（链表与指针操作章节）

## 小结 / 结论

这题真正训练的是“链表分段处理框架”，不是单点反转技巧。  
一旦你把 `dummy + 分组判定 + 组内反转 + 拼接推进` 固化成模板，后续同类题会明显降维。

## 参考与延伸阅读

- LeetCode 25 官方题目：Reverse Nodes in k-Group  
- Eric Roberts, *Programming Abstractions in C*（链表指针章节）  
- LeetCode Discuss：Reverse Nodes in k-Group 题解汇总

## 多语言实现（Python / C / C++ / Go / Rust / JS）

### Python

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev = dummy

        while True:
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if kth is None:
                    return dummy.next

            group_next = kth.next
            prev = group_next
            cur = group_prev.next

            while cur != group_next:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt

            old_head = group_prev.next
            group_prev.next = prev
            group_prev = old_head
```

### C

```c
struct ListNode {
    int val;
    struct ListNode *next;
};

struct ListNode* reverseKGroup(struct ListNode* head, int k) {
    if (head == NULL || k <= 1) return head;

    struct ListNode dummy;
    dummy.val = 0;
    dummy.next = head;
    struct ListNode* groupPrev = &dummy;

    while (1) {
        struct ListNode* kth = groupPrev;
        for (int i = 0; i < k; i++) {
            kth = kth->next;
            if (kth == NULL) return dummy.next;
        }

        struct ListNode* groupNext = kth->next;
        struct ListNode* prev = groupNext;
        struct ListNode* cur = groupPrev->next;

        while (cur != groupNext) {
            struct ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }

        struct ListNode* oldHead = groupPrev->next;
        groupPrev->next = prev;
        groupPrev = oldHead;
    }
}
```

### C++

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head || k <= 1) return head;

        ListNode dummy(0, head);
        ListNode* groupPrev = &dummy;

        while (true) {
            ListNode* kth = groupPrev;
            for (int i = 0; i < k; i++) {
                kth = kth->next;
                if (!kth) return dummy.next;
            }

            ListNode* groupNext = kth->next;
            ListNode* prev = groupNext;
            ListNode* cur = groupPrev->next;

            while (cur != groupNext) {
                ListNode* nxt = cur->next;
                cur->next = prev;
                prev = cur;
                cur = nxt;
            }

            ListNode* oldHead = groupPrev->next;
            groupPrev->next = prev;
            groupPrev = oldHead;
        }
    }
};
```

### Go

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil || k <= 1 {
		return head
	}

	dummy := &ListNode{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k; i++ {
			kth = kth.Next
			if kth == nil {
				return dummy.Next
			}
		}

		groupNext := kth.Next
		prev := groupNext
		cur := groupPrev.Next

		for cur != groupNext {
			nxt := cur.Next
			cur.Next = prev
			prev = cur
			cur = nxt
		}

		oldHead := groupPrev.Next
		groupPrev.Next = prev
		groupPrev = oldHead
	}
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

fn has_k(mut node: &Option<Box<ListNode>>, k: i32) -> bool {
    for _ in 0..k {
        match node {
            Some(n) => node = &n.next,
            None => return false,
        }
    }
    true
}

pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    if k <= 1 || !has_k(&head, k) {
        return head;
    }

    let mut curr = head;
    let mut prev: Option<Box<ListNode>> = None;

    for _ in 0..k {
        if let Some(mut node) = curr {
            let next = node.next.take();
            node.next = prev;
            prev = Some(node);
            curr = next;
        }
    }

    let mut tail_ref = &mut prev;
    while let Some(node) = tail_ref {
        if node.next.is_none() {
            node.next = reverse_k_group(curr, k);
            break;
        }
        tail_ref = &mut node.next;
    }

    prev
}
```

### JavaScript

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function reverseKGroup(head, k) {
  if (!head || k <= 1) return head;

  const dummy = new ListNode(0, head);
  let groupPrev = dummy;

  while (true) {
    let kth = groupPrev;
    for (let i = 0; i < k; i++) {
      kth = kth.next;
      if (!kth) return dummy.next;
    }

    const groupNext = kth.next;
    let prev = groupNext;
    let cur = groupPrev.next;

    while (cur !== groupNext) {
      const nxt = cur.next;
      cur.next = prev;
      prev = cur;
      cur = nxt;
    }

    const oldHead = groupPrev.next;
    groupPrev.next = prev;
    groupPrev = oldHead;
  }
}
```

## 元信息

- **阅读时长**：14~18 分钟  
- **标签**：链表、分组反转、双指针、原地算法  
- **SEO 关键词**：K 个一组翻转链表, Reverse Nodes in k-Group, LeetCode 25, 链表分组反转  
- **元描述**：从分组判定到原地拼接，系统讲透 LeetCode 25，并给出六语言可复用模板。

## 行动号召（CTA）

建议你现在连续练这三题，建立“区间反转模板家族”：

1. 206 Reverse Linked List  
2. 92 Reverse Linked List II  
3. 25 Reverse Nodes in k-Group

然后把三题统一成一个“定位边界 + 原地反转 + 回接主链”的个人模板。
