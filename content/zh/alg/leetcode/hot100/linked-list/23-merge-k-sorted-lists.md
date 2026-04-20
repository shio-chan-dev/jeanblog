---
title: "Hot100：合并K个升序链表（Merge k Sorted Lists）分治归并 O(N log k) ACERS 解析"
date: 2026-02-10T17:05:53+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "链表", "分治", "归并", "优先队列", "LeetCode 23"]
description: "把 LeetCode 21 的双链表归并扩展到 k 路：从朴素串行合并到分治归并，时间复杂度降到 O(N log k)，并对比最小堆方案与工程取舍，附多语言实现。"
keywords: ["Merge k Sorted Lists", "合并K个升序链表", "分治归并", "LeetCode 23", "Hot100", "O(N log k)"]
---

> **副标题 / 摘要**  
> 这题本质是“k 路归并”。如果直接一条条串行并入，性能会退化；用分治按二叉树方式两两合并，能把复杂度优化到 O(N log k)。本文按 ACERS 模板把思路推导、工程映射和多语言实现一次讲透。

- **预计阅读时长**：12~16 分钟  
- **标签**：`Hot100`、`链表`、`分治`、`归并`  
- **SEO 关键词**：Merge k Sorted Lists, 合并K个升序链表, 分治归并, LeetCode 23, O(N log k)  
- **元描述**：从串行归并到分治归并，系统讲解 LeetCode 23 的最优复杂度解法与工程实践。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个链表数组 `lists`，每个链表都按升序排列。  
请将所有链表合并到一个升序链表中，并返回合并后的头节点。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| lists | ListNode[] | k 条升序链表，元素可为空 |
| 返回 | ListNode | 合并后的升序链表头节点 |

### 示例 1

```text
输入: lists = [[1,4,5],[1,3,4],[2,6]]
输出: [1,1,2,3,4,4,5,6]
```

### 示例 2

```text
输入: lists = []
输出: []
```

---

## 目标读者

- 正在刷 Hot100，已经掌握 LeetCode 21 的同学
- 想把“双链表归并”升级为“k 路归并模板”的开发者
- 在服务端做多路有序流合并（日志、时间线、分片结果）的工程师

## 背景 / 动机

LeetCode 23 是 LeetCode 21 的自然升级版：

- 21：2 路归并
- 23：k 路归并

核心挑战不在“能不能合并”，而在“如何把复杂度控制在可接受范围”。

如果每次把结果链表继续和下一条链表做串行归并，早期节点会被反复遍历，实际性能很容易退化。

## 核心概念

- **N**：所有链表节点总数
- **k**：链表条数
- **串行归并**：从左到右不断把当前结果和下一条链表合并
- **分治归并**：像归并排序一样，两两合并，按层收敛
- **最小堆方案**：维护 k 个当前头节点，每次弹出最小值并推进其所在链表

---

## C — Concepts（核心思想）

### 思路是怎么推出来的

#### Step 1：先用一个“已经不是 21 题”的最小例子

看这组三条链表：

```text
l1: 1 -> 4
l2: 1 -> 3
l3: 2 -> 6
```

我们其实已经会做“两个有序链表合并”了。
所以这题真正新增的难点不是单次比较规则，而是：

> 该按什么顺序复用 `merge_two`，才能让总工作量别失控？

#### Step 2：先排除最顺手但代价不稳的串行归并

最直接的写法是：

```text
((l1 merge l2) merge l3) merge l4 ...
```

它当然能做对，但有个隐藏问题：

- 结果链表会越来越长
- 早期节点会被反复参与后续归并

如果 `k` 很大，这种“一个大结果不断并入下一条链表”的做法，最坏会逼近 `O(Nk)`。

问题不在“不会合并”，而在“合并顺序不平衡”。

#### Step 3：把问题改写成一个可复用的小子问题

我们希望定义：

```python
solve(l, r)
```

它表示：

> 把 `lists[l]` 到 `lists[r]` 这些链表全部合并后的结果。

一旦这个定义成立，事情就简单了：

- 左半边先合并好
- 右半边先合并好
- 最后再把左右结果做一次 `merge_two`

这说明 23 题其实是在“调度很多次 21 题”。

#### Step 4：为什么要按二叉树方式两两归并？

如果我们按层做两两归并：

```text
(l1,l2) (l3,l4) (l5,l6) ...
```

那么会形成一棵平衡合并树：

- 每一层总共处理的节点数大约都是 `N`
- 层数大约只有 `log k`

也就是说，每个节点只会在约 `log k` 层里被处理，而不是被一个不断变大的结果链表反复拖着走。

#### Step 5：先把递归边界说清楚

当区间里只剩一条链表时：

```python
if l == r:
    return lists[l]
```

因为单条链表本来就是有序的，不需要再处理。

这个 base case 也是整个分治定义能落地的关键。

#### Step 6：再定义“一层分治”到底做什么

把区间从中间切开：

```python
mid = (l + r) // 2
left = solve(l, mid)
right = solve(mid + 1, r)
return merge_two(left, right)
```

注意这里没有新魔法。
局部合并仍然是 21 题模板，变化的只是“谁先和谁合并”。

#### Step 7：慢速走一棵合并树

假设有四条链表：

```text
[l1, l2, l3, l4]
```

分治过程是：

```text
solve(0, 3)
├─ solve(0, 1) -> merge_two(l1, l2)
└─ solve(2, 3) -> merge_two(l3, l4)
```

最后再做：

```text
merge_two(merge(l1,l2), merge(l3,l4))
```

这样不会出现“前面合好的大链表不断被后面小链表反复扫描”的问题。

#### Step 8：把方法压缩成一句话

23 题不是发明新的合并规则，而是把 `merge_two` 按平衡二叉树的顺序复用起来。

### Assemble the Full Code

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(left: int, right: int) -> Optional[ListNode]:
        if left == right:
            return lists[left]
        mid = (left + right) // 2
        return merge_two(solve(left, mid), solve(mid + 1, right))

    return solve(0, len(lists) - 1)


def from_list(arr: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    lists = [from_list([1, 4, 5]), from_list([1, 3, 4]), from_list([2, 6])]
    print(to_list(merge_k_lists(lists)))
```

### Reference Answer

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(l: int, r: int) -> Optional[ListNode]:
        if l == r:
            return lists[l]
        m = (l + r) // 2
        return merge_two(solve(l, m), solve(m + 1, r))

    return solve(0, len(lists) - 1)
```

### 方法归类

- **分治 + 归并**
- **链表拼接（in-place splicing）**
- 与“最小堆 k 路归并”同属最优复杂度级别

### 循环/递归不变量

对于区间 `[l, r]`：

- `mergeRange(lists, l, r)` 返回的是 `lists[l..r]` 全部节点的升序合并结果
- 子问题正确时，`mergeTwo(left, right)` 仍保持升序且不丢节点
## 实践指南 / 步骤

1. 先写好 `mergeTwo`（LeetCode 21 模板）
2. 实现 `mergeRange(l, r)`：
   - `l == r` 直接返回
   - `mid = (l+r)//2`
   - 递归合并左区间和右区间
3. 顶层调用 `mergeRange(0, k-1)`

Python 可运行示例（`merge_k_lists.py`）：

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(left: int, right: int) -> Optional[ListNode]:
        if left == right:
            return lists[left]
        mid = (left + right) // 2
        return merge_two(solve(left, mid), solve(mid + 1, right))

    return solve(0, len(lists) - 1)


def from_list(arr: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    while head:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    lists = [from_list([1, 4, 5]), from_list([1, 3, 4]), from_list([2, 6])]
    print(to_list(merge_k_lists(lists)))
```

---

## 解释与原理（为什么这么做）

分治归并的本质是“按层处理”而不是“按顺序叠加”：

- 串行归并会让前面的节点在后续反复参与比较
- 分治归并让每个节点只在每一层参与一次

因此：

- 每层工作量约 N
- 层数约 log k
- 总计 O(N log k)

---

## E — Engineering（工程应用）

### 场景 1：多分片日志时间线合并（Go）

**背景**：每个分片内日志按时间升序，聚合层需要给出全局升序。  
**为什么适用**：分治归并天然适合批量收敛，且易并行化分层处理。

```go
package main

import "fmt"

type Node struct {
	Ts   int
	Next *Node
}

func mergeTwo(a, b *Node) *Node {
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
	a := &Node{1, &Node{4, &Node{9, nil}}}
	b := &Node{2, &Node{5, nil}}
	for p := mergeTwo(a, b); p != nil; p = p.Next {
		fmt.Print(p.Ts, " ")
	}
	fmt.Println()
}
```

### 场景 2：离线任务合并多路排序结果（Python）

**背景**：多个规则引擎输出各自排序结果，需要统一排序输出。  
**为什么适用**：分治归并可复用既有双路模板，维护成本低。

```python
def merge_sorted_arrays(arrays):
    if not arrays:
        return []

    def merge(a, b):
        i = j = 0
        out = []
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                out.append(a[i]); i += 1
            else:
                out.append(b[j]); j += 1
        out.extend(a[i:])
        out.extend(b[j:])
        return out

    cur = arrays
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                nxt.append(merge(cur[i], cur[i + 1]))
            else:
                nxt.append(cur[i])
        cur = nxt
    return cur[0]

print(merge_sorted_arrays([[1, 4, 5], [1, 3, 4], [2, 6]]))
```

### 场景 3：前端多路已排序卡片流合并（JavaScript）

**背景**：多个来源返回已按时间排序的卡片列表。  
**为什么适用**：在前端直接分治合并可减少后端聚合接口压力。

```javascript
function mergeTwo(a, b) {
  let i = 0;
  let j = 0;
  const out = [];
  while (i < a.length && j < b.length) {
    if (a[i].ts <= b[j].ts) out.push(a[i++]);
    else out.push(b[j++]);
  }
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}

function mergeK(arrays) {
  if (!arrays.length) return [];
  let cur = arrays;
  while (cur.length > 1) {
    const nxt = [];
    for (let i = 0; i < cur.length; i += 2) {
      if (i + 1 < cur.length) nxt.push(mergeTwo(cur[i], cur[i + 1]));
      else nxt.push(cur[i]);
    }
    cur = nxt;
  }
  return cur[0];
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 设总节点数为 N，链表条数为 k
- 分治归并：
  - 时间复杂度：`O(N log k)`
  - 空间复杂度：`O(log k)`（递归栈）

### 替代方案对比

| 方法 | 时间复杂度 | 空间复杂度 | 评价 |
| --- | --- | --- | --- |
| 拉平排序 | O(N log N) | O(N) | 简单但没利用分路有序 |
| 串行归并 | 最坏近似 O(Nk) | O(1) | k 大时性能差 |
| 最小堆 | O(N log k) | O(k) | 在线场景友好 |
| 分治归并 | O(N log k) | O(log k) | 模板统一，工程常用 |

### 常见错误思路

1. 忘记处理空输入 `lists=[]`
2. 串行归并误以为“也是 O(N log k)”
3. `mergeTwo` 写错导致断链或环
4. 递归边界 `l==r` / `l>r` 漏判

### 为什么当前方法最可行

- 与 LeetCode 21 模板复用度最高
- 相比堆方案，代码结构更直观、调试更容易
- 对离线批处理和批量收敛场景非常友好

---

## 常见问题与注意事项

1. **分治和最小堆谁更好？**  
   批量一次性合并常选分治；流式持续读入常选最小堆。

2. **可以全程原地不分配新节点吗？**  
   可以（除哨兵节点外），通过拼接 `next` 实现。

3. **k 非常大怎么办？**  
   优先 O(N log k) 的分治或堆，避免串行归并。

4. **链表里有重复值有影响吗？**  
   无影响，`<=` 策略可保持稳定性。

---

## 最佳实践与建议

- 先把 `mergeTwo` 写成稳定工具函数
- `mergeK` 优先用分治层级收敛，不要串行叠加
- 用奇偶条数和空链表做单测基线
- 工程里记录 `k` 与 `N` 指标，评估分治/堆切换阈值

---

## S — Summary（总结）

- LeetCode 23 本质是 k 路归并，不是“把 21 重复 k 次”
- 分治归并通过平衡合并树，把复杂度降到 O(N log k)
- 最小堆与分治同阶，前者偏在线、后者偏批量
- 掌握这题后，合并类链表题会形成统一模板

### 推荐延伸阅读

- LeetCode 21. Merge Two Sorted Lists
- LeetCode 23. Merge k Sorted Lists
- LeetCode 148. Sort List
- LeetCode 632. Smallest Range Covering Elements from K Lists

---

## 小结 / 结论

这题最核心的收获是“算法结构升级”：
从线性串行思维转向分层收敛思维。  
你在这里学到的不只是一个题解，而是处理多路有序数据的通用工程策略。

---

## 参考与延伸阅读

- https://leetcode.com/problems/merge-k-sorted-lists/
- https://en.cppreference.com/w/cpp/container/priority_queue
- https://docs.python.org/3/library/heapq.html
- https://pkg.go.dev/container/heap

---

## 元信息

- **阅读时长**：12~16 分钟
- **标签**：Hot100、链表、分治、归并
- **SEO 关键词**：Merge k Sorted Lists, LeetCode 23, 分治归并, O(N log k)
- **元描述**：系统讲解 LeetCode 23 的分治归并方案，含复杂度推导、工程场景与多语言实现。

---

## 行动号召（CTA）

建议你立刻做两件事：

1. 把 `mergeTwo` + `mergeK` 抽成可复用模板
2. 再练一道 `merge k arrays` 或 `k-way stream merge` 强化迁移能力

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            nxt = a.next
            tail.next = a
            a.next = None
            a = nxt
        else:
            nxt = b.next
            tail.next = b
            b.next = None
            b = nxt
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None

    def solve(l: int, r: int) -> Optional[ListNode]:
        if l == r:
            return lists[l]
        m = (l + r) // 2
        return merge_two(solve(l, m), solve(m + 1, r))

    return solve(0, len(lists) - 1)
```

```c
#include <stddef.h>

typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

ListNode* mergeTwo(ListNode* a, ListNode* b) {
    ListNode dummy;
    dummy.next = NULL;
    ListNode* tail = &dummy;

    while (a && b) {
        if (a->val <= b->val) {
            ListNode* nxt = a->next;
            tail->next = a;
            a->next = NULL;
            a = nxt;
        } else {
            ListNode* nxt = b->next;
            tail->next = b;
            b->next = NULL;
            b = nxt;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

ListNode* mergeRange(ListNode** lists, int l, int r) {
    if (l > r) return NULL;
    if (l == r) return lists[l];
    int m = l + (r - l) / 2;
    ListNode* left = mergeRange(lists, l, m);
    ListNode* right = mergeRange(lists, m + 1, r);
    return mergeTwo(left, right);
}

ListNode* mergeKLists(ListNode** lists, int listsSize) {
    if (listsSize == 0) return NULL;
    return mergeRange(lists, 0, listsSize - 1);
}
```

```cpp
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};

class Solution {
    ListNode* mergeTwo(ListNode* a, ListNode* b) {
        ListNode dummy;
        ListNode* tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) {
                ListNode* nxt = a->next;
                tail->next = a;
                a->next = nullptr;
                a = nxt;
            } else {
                ListNode* nxt = b->next;
                tail->next = b;
                b->next = nullptr;
                b = nxt;
            }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }

    ListNode* solve(vector<ListNode*>& lists, int l, int r) {
        if (l > r) return nullptr;
        if (l == r) return lists[l];
        int m = l + (r - l) / 2;
        return mergeTwo(solve(lists, l, m), solve(lists, m + 1, r));
    }

public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        return solve(lists, 0, (int)lists.size() - 1);
    }
};
```

```go
package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwo(a, b *ListNode) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for a != nil && b != nil {
		if a.Val <= b.Val {
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

func mergeRange(lists []*ListNode, l, r int) *ListNode {
	if l > r {
		return nil
	}
	if l == r {
		return lists[l]
	}
	m := l + (r-l)/2
	left := mergeRange(lists, l, m)
	right := mergeRange(lists, m+1, r)
	return mergeTwo(left, right)
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	return mergeRange(lists, 0, len(lists)-1)
}
```

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

fn merge_two(a: Option<Box<ListNode>>, b: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    match (a, b) {
        (None, x) => x,
        (x, None) => x,
        (Some(mut na), Some(mut nb)) => {
            if na.val <= nb.val {
                let next = na.next.take();
                na.next = merge_two(next, Some(nb));
                Some(na)
            } else {
                let next = nb.next.take();
                nb.next = merge_two(Some(na), next);
                Some(nb)
            }
        }
    }
}

fn solve(lists: &mut [Option<Box<ListNode>>], l: usize, r: usize) -> Option<Box<ListNode>> {
    if l == r {
        return lists[l].take();
    }
    let m = (l + r) / 2;
    let left = solve(lists, l, m);
    let right = solve(lists, m + 1, r);
    merge_two(left, right)
}

pub fn merge_k_lists(mut lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    if lists.is_empty() {
        return None;
    }
    let n = lists.len();
    solve(&mut lists, 0, n - 1)
}
```

```javascript
function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function mergeTwo(a, b) {
  const dummy = new ListNode(0);
  let tail = dummy;
  while (a && b) {
    if (a.val <= b.val) {
      const nxt = a.next;
      tail.next = a;
      a.next = null;
      a = nxt;
    } else {
      const nxt = b.next;
      tail.next = b;
      b.next = null;
      b = nxt;
    }
    tail = tail.next;
  }
  tail.next = a || b;
  return dummy.next;
}

function mergeRange(lists, l, r) {
  if (l > r) return null;
  if (l === r) return lists[l];
  const m = (l + r) >> 1;
  return mergeTwo(mergeRange(lists, l, m), mergeRange(lists, m + 1, r));
}

function mergeKLists(lists) {
  if (!lists || lists.length === 0) return null;
  return mergeRange(lists, 0, lists.length - 1);
}
```
