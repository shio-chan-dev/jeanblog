---
title: "LeetCode 23：合并 K 个升序链表（最小堆）ACERS 全解析"
date: 2026-02-11T08:00:29+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "优先队列", "最小堆", "分治", "LeetCode 23"]
description: "从顺序两两合并到分治与最小堆，完整讲解 LeetCode 23 合并 K 个升序链表的思路推导、复杂度比较、工程迁移与六语言可运行代码。"
keywords: ["LeetCode 23", "Merge k Sorted Lists", "合并K个升序链表", "最小堆", "优先队列", "分治归并", "链表"]
---

> **副标题 / 摘要**  
> 多路有序链表合并是“归并思想 + 数据结构选择”的代表题。本文从朴素做法推导到最小堆最优解，用 ACERS 框架讲清算法正确性、工程价值与常见坑。

- **预计阅读时长**：14~18 分钟  
- **适用场景标签**：`链表进阶`、`优先队列`、`多路归并`  
- **SEO 关键词**：LeetCode 23, Merge k Sorted Lists, 合并 K 个升序链表, 最小堆, 优先队列  
- **元描述（Meta Description）**：LeetCode 23 全流程题解：朴素法、分治法、最小堆法的对比与取舍，附 Python/C/C++/Go/Rust/JS 可运行实现。

---

## 目标读者

- 正在刷 LeetCode Hot100 / 链表专题的同学  
- 想从“会写归并”升级到“会做多路归并”的中级开发者  
- 在日志、消息、时间线合并场景中需要稳定有序输出的工程师

## 背景 / 动机

题目看起来是链表题，实际上考的是：  
**如何高效地从 K 个“当前候选最小值”里快速挑最小**。

如果你只会“两个链表合并”，直接扩展到 K 路时会遇到两个痛点：

- 每次找最小头节点可能要扫 K 次；
- K 很大时，顺序两两合并会重复扫描大量节点。

所以这题是理解“多路归并复杂度”的关键入口，也是搜索引擎倒排合并、日志管道合并、流式数据聚合的基础模型。

## 核心概念

- **多路归并（K-way Merge）**：同时合并 K 个已排序序列。  
- **最小堆 / 优先队列**：支持高效拿到当前最小元素，插入与弹出都是 `O(log k)`。  
- **哨兵节点（dummy）**：简化链表拼接，避免首节点特判。  
- **总节点数 `N`**：`N = 所有链表长度总和`，复杂度通常写成 `O(N log k)`。

---

## A — Algorithm（题目与算法）

### 题目还原

给你一个链表数组 `lists`，每个链表都已经按升序排列。  
请你将所有链表合并到一个升序链表中，返回合并后的链表。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| `lists` | `List[ListNode]` | 长度为 `k` 的链表数组，每个链表升序 |
| 返回 | `ListNode` | 一个新的升序链表头结点（由原节点拼接） |

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

### 示例 3

```text
输入: lists = [[]]
输出: []
```

### 直观图示（最小堆思路）

```text
初始把每条链表头结点入堆：
heap: [1(a), 1(b), 2(c)]

每次弹最小并接到结果链表尾部，再把它的 next 入堆：
pop 1(a) -> push 4(a)
pop 1(b) -> push 3(b)
pop 2(c) -> push 6(c)
...
直到堆空，结果天然有序
```

---

## C — Concepts（核心思想）

### 思路推导：从朴素到最优

1. **朴素法：收集全部节点值后排序再重建链表**  
   - 时间 `O(N log N)`，空间 `O(N)`；  
   - 没利用“每条链表已排序”的先验条件。

2. **顺序两两合并（线性叠加）**  
   - 先合并 `list1/list2`，再与 `list3` 合并……  
   - 最坏时间接近 `O(Nk)`，当 `k` 大时代价高。

3. **分治归并（像 merge sort）**  
   - 每轮把链表两两配对合并，轮数 `log k`；  
   - 总复杂度 `O(N log k)`，空间可做到 `O(1)` 额外（不计递归栈）。

4. **最小堆（优先队列）主解**  
   - 堆里始终维护“每条链当前头节点”；  
   - 每次弹出全局最小并补入该链下一个节点；  
   - 一共弹出 `N` 次，每次 `O(log k)`，总复杂度 `O(N log k)`。

### 方法归类

- 归并思想（Merge）  
- 堆结构维护多路最小值（Heap / Priority Queue）  
- 链表原地拼接（Pointer Rewire）

### 正确性要点（简版）

不变量：堆中始终包含每条“未耗尽链表”的当前最小候选节点。  

因此：

- 堆顶是所有未处理节点中的全局最小值；
- 每次取出堆顶接到结果尾部都保持结果有序；
- 取出节点后把其后继入堆，不会漏元素。

当堆为空时，所有节点都被按非降序输出，算法正确。

---

## 实践指南 / 步骤

1. 准备哨兵节点 `dummy` 与尾指针 `tail`。  
2. 遍历 `lists`，把非空头结点加入最小堆。  
3. 循环直到堆空：  
   - 弹出最小节点 `node`；  
   - `tail.next = node`，`tail = tail.next`；  
   - 若 `node.next` 非空则入堆。  
4. 返回 `dummy.next`。

Python 可运行演示（最小堆）：

```python
from typing import List, Optional
import heapq


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    heap = []
    for i, node in enumerate(lists):
        if node is not None:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    tail = dummy

    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

---

## E — Engineering（工程应用）

### 场景 1：多分片日志按时间合并（Go）

**背景**：日志系统按分片写入，每个分片内部按时间戳有序。  
**为什么适用**：实时查询时需要把 K 路有序日志输出成一个全局有序流，最小堆是标准解。

```go
package main

import (
	"container/heap"
	"fmt"
)

type Item struct {
	ts   int
	part int
	idx  int
}

type MinHeap []Item

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].ts < h[j].ts }
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func merge(parts [][]int) []int {
	h := &MinHeap{}
	heap.Init(h)
	for p := range parts {
		if len(parts[p]) > 0 {
			heap.Push(h, Item{ts: parts[p][0], part: p, idx: 0})
		}
	}

	ans := []int{}
	for h.Len() > 0 {
		it := heap.Pop(h).(Item)
		ans = append(ans, it.ts)
		ni := it.idx + 1
		if ni < len(parts[it.part]) {
			heap.Push(h, Item{ts: parts[it.part][ni], part: it.part, idx: ni})
		}
	}
	return ans
}

func main() {
	parts := [][]int{{1, 4, 9}, {2, 3, 8}, {5, 6, 7}}
	fmt.Println(merge(parts)) // [1 2 3 4 5 6 7 8 9]
}
```

### 场景 2：数据管道多路排序结果合并（Python）

**背景**：ETL 任务中每个 worker 输出局部有序批次。  
**为什么适用**：把多个有序批次做在线合并可以降低内存峰值，不必一次性全量排序。

```python
import heapq


def merge_sorted_batches(batches):
    heap = []
    for i, arr in enumerate(batches):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    out = []
    while heap:
        val, i, j = heapq.heappop(heap)
        out.append(val)
        nj = j + 1
        if nj < len(batches[i]):
            heapq.heappush(heap, (batches[i][nj], i, nj))
    return out


if __name__ == "__main__":
    print(merge_sorted_batches([[1, 4, 5], [1, 3, 4], [2, 6]]))
```

### 场景 3：前端聚合多个时间线源（JavaScript）

**背景**：前端页面需要把多个服务返回的时间线事件统一展示。  
**为什么适用**：每个来源内部已按时间排序，用最小堆可增量合并并支持流式渲染。

```javascript
class MinHeap {
  constructor() {
    this.a = [];
  }
  size() {
    return this.a.length;
  }
  push(x) {
    this.a.push(x);
    this.up(this.a.length - 1);
  }
  pop() {
    if (this.a.length === 0) return null;
    const top = this.a[0];
    const last = this.a.pop();
    if (this.a.length) {
      this.a[0] = last;
      this.down(0);
    }
    return top;
  }
  up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[p][0] <= this.a[i][0]) break;
      [this.a[p], this.a[i]] = [this.a[i], this.a[p]];
      i = p;
    }
  }
  down(i) {
    const n = this.a.length;
    while (true) {
      let t = i;
      const l = i * 2 + 1;
      const r = i * 2 + 2;
      if (l < n && this.a[l][0] < this.a[t][0]) t = l;
      if (r < n && this.a[r][0] < this.a[t][0]) t = r;
      if (t === i) break;
      [this.a[t], this.a[i]] = [this.a[i], this.a[t]];
      i = t;
    }
  }
}

function mergeStreams(streams) {
  const h = new MinHeap();
  for (let i = 0; i < streams.length; i++) {
    if (streams[i].length) h.push([streams[i][0], i, 0]);
  }
  const ans = [];
  while (h.size()) {
    const [v, i, j] = h.pop();
    ans.push(v);
    if (j + 1 < streams[i].length) h.push([streams[i][j + 1], i, j + 1]);
  }
  return ans;
}

console.log(mergeStreams([[1, 4, 5], [1, 3, 4], [2, 6]]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

设共有 `k` 条链表，总节点数为 `N`：

- 最小堆法：时间 `O(N log k)`，空间 `O(k)`  
- 分治归并：时间 `O(N log k)`，空间 `O(1)` 到 `O(log k)`（视实现）  
- 顺序两两合并：最坏可达 `O(Nk)`  

### 替代方案对比

| 方案 | 时间复杂度 | 空间复杂度 | 适用性 | 问题 |
| --- | --- | --- | --- | --- |
| 全量收集排序 | O(N log N) | O(N) | 实现快 | 浪费“局部有序”信息 |
| 顺序两两合并 | O(Nk) | O(1) | k 很小时可用 | k 大时明显变慢 |
| 分治归并 | O(N log k) | O(log k) | 综合稳定 | 实现稍复杂 |
| 最小堆 | O(N log k) | O(k) | 工程常用，流式友好 | 需掌握堆结构 |

### 常见错误与坑

- 忘记跳过空链表，导致空指针异常；  
- 堆比较只比值，不加“打破平局”的字段，在某些语言会比较对象报错；  
- 拼接节点后忘记推进 `tail`；  
- C/C++ 实现里释放或复用节点不当引入悬垂指针。

### 为什么主解工程可行

- 能边读边出（流式），不要求一次性加载所有数据；  
- 对 `k` 变化适应性强；  
- 在消息系统、日志系统、搜索排序合并中有成熟实践。

---

## 常见问题（FAQ）

### Q1：最小堆和分治都 `O(N log k)`，怎么选？

如果你要流式输出，优先最小堆。  
如果你更偏向“链表归并模板复用”且希望减少堆结构依赖，分治也很好。

### Q2：为什么 Python 堆里要放 `(val, idx, node)`？

`node` 对象不可直接比较，当值相等时需要 `idx` 作为次关键字，避免比较对象时报错。

### Q3：可以不用哨兵节点吗？

可以，但你要额外处理“结果头节点第一次赋值”的分支。哨兵节点会更稳。

---

## 最佳实践与建议

- 固定使用 `dummy + tail` 拼接模板，减少断链风险；  
- 堆元素建议携带来源链编号，便于调试与回溯；  
- 对极端输入（`[]`、`[[]]`、全空链）做单测；  
- 面试时可以先说分治，再给最小堆，体现思路广度与取舍能力。

---

## S — Summary（总结）

### 核心收获

1. 这题核心是“多路最小值维护”，不是普通链表删除/反转。  
2. 最小堆把“每轮找最小”从 `O(k)` 降到 `O(log k)`，整体达到 `O(N log k)`。  
3. 分治归并与最小堆同阶，前者偏模板复用，后者偏流式工程。  
4. 哨兵节点可以大幅降低链表拼接的边界错误率。  
5. 多路归并思想可以直接迁移到日志、消息、搜索等真实系统。

### 延伸阅读

- LeetCode 23（官方）：<https://leetcode.com/problems/merge-k-sorted-lists/>  
- 力扣中文站：<https://leetcode.cn/problems/merge-k-sorted-lists/>  
- 关联题：LeetCode 21（合并两个有序链表）、LeetCode 148（排序链表）、LeetCode 378（有序矩阵中第 K 小）

---

## 行动号召（CTA）

建议立刻做一组对照练习：  
先实现“顺序两两合并”，再实现“最小堆合并”，用同一组随机数据比较耗时。你会更直观地掌握 `log k` 的收益。

---

## 多语言实现（可直接运行）

### Python（最小堆）

```python
from typing import List, Optional
import heapq


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    heap = []
    for i, node in enumerate(lists):
        if node is not None:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    tail = dummy

    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next


def from_list(nums):
    dummy = ListNode(0)
    cur = dummy
    for x in nums:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next


def to_list(head):
    out = []
    while head is not None:
        out.append(head.val)
        head = head.next
    return out


if __name__ == "__main__":
    lists = [from_list([1, 4, 5]), from_list([1, 3, 4]), from_list([2, 6])]
    print(to_list(merge_k_lists(lists)))  # [1, 1, 2, 3, 4, 4, 5, 6]
```

### C（最小堆）

```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct Heap {
    struct ListNode** a;
    int size;
    int cap;
};

void heap_init(struct Heap* h, int cap) {
    h->a = (struct ListNode**)malloc(sizeof(struct ListNode*) * cap);
    h->size = 0;
    h->cap = cap;
}

void heap_swap(struct ListNode** x, struct ListNode** y) {
    struct ListNode* t = *x;
    *x = *y;
    *y = t;
}

void heap_push(struct Heap* h, struct ListNode* node) {
    int i = h->size++;
    h->a[i] = node;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->a[p]->val <= h->a[i]->val) break;
        heap_swap(&h->a[p], &h->a[i]);
        i = p;
    }
}

struct ListNode* heap_pop(struct Heap* h) {
    if (h->size == 0) return NULL;
    struct ListNode* top = h->a[0];
    h->a[0] = h->a[--h->size];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, t = i;
        if (l < h->size && h->a[l]->val < h->a[t]->val) t = l;
        if (r < h->size && h->a[r]->val < h->a[t]->val) t = r;
        if (t == i) break;
        heap_swap(&h->a[t], &h->a[i]);
        i = t;
    }
    return top;
}

struct ListNode* mergeKLists(struct ListNode** lists, int listsSize) {
    struct Heap h;
    heap_init(&h, listsSize > 0 ? listsSize : 1);

    for (int i = 0; i < listsSize; ++i) {
        if (lists[i]) heap_push(&h, lists[i]);
    }

    struct ListNode dummy = {0, NULL};
    struct ListNode* tail = &dummy;

    while (h.size > 0) {
        struct ListNode* node = heap_pop(&h);
        tail->next = node;
        tail = tail->next;
        if (node->next) heap_push(&h, node->next);
    }

    free(h.a);
    return dummy.next;
}

struct ListNode* new_node(int v) {
    struct ListNode* p = (struct ListNode*)malloc(sizeof(struct ListNode));
    p->val = v;
    p->next = NULL;
    return p;
}

struct ListNode* build(int* a, int n) {
    struct ListNode dummy = {0, NULL};
    struct ListNode* tail = &dummy;
    for (int i = 0; i < n; ++i) {
        tail->next = new_node(a[i]);
        tail = tail->next;
    }
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
    int a1[] = {1, 4, 5}, a2[] = {1, 3, 4}, a3[] = {2, 6};
    struct ListNode* lists[3];
    lists[0] = build(a1, 3);
    lists[1] = build(a2, 3);
    lists[2] = build(a3, 2);
    struct ListNode* ans = mergeKLists(lists, 3);
    print_list(ans); // 1 1 2 3 4 4 5 6
    free_list(ans);
    return 0;
}
```

### C++（优先队列）

```cpp
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct Cmp {
    bool operator()(ListNode* a, ListNode* b) const {
        return a->val > b->val;
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Cmp> pq;
    for (auto node : lists) {
        if (node) pq.push(node);
    }

    ListNode dummy(0);
    ListNode* tail = &dummy;
    while (!pq.empty()) {
        auto node = pq.top();
        pq.pop();
        tail->next = node;
        tail = tail->next;
        if (node->next) pq.push(node->next);
    }
    return dummy.next;
}

ListNode* build(const vector<int>& arr) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int x : arr) {
        tail->next = new ListNode(x);
        tail = tail->next;
    }
    return dummy.next;
}

void print(ListNode* head) {
    for (auto p = head; p; p = p->next) cout << p->val << " ";
    cout << "\n";
}

void destroy(ListNode* head) {
    while (head) {
        auto t = head;
        head = head->next;
        delete t;
    }
}

int main() {
    vector<ListNode*> lists = {
        build({1, 4, 5}),
        build({1, 3, 4}),
        build({2, 6}),
    };
    ListNode* ans = mergeKLists(lists);
    print(ans); // 1 1 2 3 4 4 5 6
    destroy(ans);
    return 0;
}
```

### Go（container/heap）

```go
package main

import (
	"container/heap"
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

type NodeHeap []*ListNode

func (h NodeHeap) Len() int            { return len(h) }
func (h NodeHeap) Less(i, j int) bool  { return h[i].Val < h[j].Val }
func (h NodeHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *NodeHeap) Push(x interface{}) { *h = append(*h, x.(*ListNode)) }
func (h *NodeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func mergeKLists(lists []*ListNode) *ListNode {
	h := &NodeHeap{}
	heap.Init(h)
	for _, node := range lists {
		if node != nil {
			heap.Push(h, node)
		}
	}

	dummy := &ListNode{}
	tail := dummy
	for h.Len() > 0 {
		node := heap.Pop(h).(*ListNode)
		tail.Next = node
		tail = tail.Next
		if node.Next != nil {
			heap.Push(h, node.Next)
		}
	}
	return dummy.Next
}

func build(a []int) *ListNode {
	dummy := &ListNode{}
	tail := dummy
	for _, x := range a {
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
	lists := []*ListNode{build([]int{1, 4, 5}), build([]int{1, 3, 4}), build([]int{2, 6})}
	ans := mergeKLists(lists)
	printList(ans) // 1 1 2 3 4 4 5 6
}
```

### Rust（分治归并，可运行）

> 说明：Rust 在 `Option<Box<ListNode>>` 下写堆版会有较多所有权细节。这里给出同阶 `O(N log k)` 的分治实现，工程上同样常用。

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

fn merge_two(a: Option<Box<ListNode>>, b: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    match (a, b) {
        (None, x) => x,
        (x, None) => x,
        (Some(mut x), Some(mut y)) => {
            if x.val <= y.val {
                let next = x.next.take();
                x.next = merge_two(next, Some(y));
                Some(x)
            } else {
                let next = y.next.take();
                y.next = merge_two(Some(x), next);
                Some(y)
            }
        }
    }
}

fn merge_k_lists(mut lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    if lists.is_empty() {
        return None;
    }
    while lists.len() > 1 {
        let mut merged = Vec::new();
        let mut i = 0;
        while i < lists.len() {
            if i + 1 < lists.len() {
                merged.push(merge_two(lists[i].take(), lists[i + 1].take()));
            } else {
                merged.push(lists[i].take());
            }
            i += 2;
        }
        lists = merged;
    }
    lists.pop().flatten()
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
    let lists = vec![from_vec(vec![1, 4, 5]), from_vec(vec![1, 3, 4]), from_vec(vec![2, 6])];
    let ans = merge_k_lists(lists);
    println!("{:?}", to_vec(ans)); // [1, 1, 2, 3, 4, 4, 5, 6]
}
```

### JavaScript（最小堆）

```javascript
class ListNode {
  constructor(val = 0, next = null) {
    this.val = val;
    this.next = next;
  }
}

class MinHeap {
  constructor() {
    this.a = [];
  }
  size() {
    return this.a.length;
  }
  push(x) {
    this.a.push(x);
    this.up(this.a.length - 1);
  }
  pop() {
    if (!this.a.length) return null;
    const top = this.a[0];
    const last = this.a.pop();
    if (this.a.length) {
      this.a[0] = last;
      this.down(0);
    }
    return top;
  }
  up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[p].val <= this.a[i].val) break;
      [this.a[p], this.a[i]] = [this.a[i], this.a[p]];
      i = p;
    }
  }
  down(i) {
    const n = this.a.length;
    while (true) {
      let t = i;
      const l = i * 2 + 1;
      const r = i * 2 + 2;
      if (l < n && this.a[l].val < this.a[t].val) t = l;
      if (r < n && this.a[r].val < this.a[t].val) t = r;
      if (t === i) break;
      [this.a[t], this.a[i]] = [this.a[i], this.a[t]];
      i = t;
    }
  }
}

function mergeKLists(lists) {
  const h = new MinHeap();
  for (const node of lists) {
    if (node) h.push(node);
  }

  const dummy = new ListNode();
  let tail = dummy;
  while (h.size()) {
    const node = h.pop();
    tail.next = node;
    tail = tail.next;
    if (node.next) h.push(node.next);
  }
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
  const ans = [];
  for (let p = head; p; p = p.next) ans.push(p.val);
  return ans;
}

const lists = [fromArray([1, 4, 5]), fromArray([1, 3, 4]), fromArray([2, 6])];
console.log(toArray(mergeKLists(lists))); // [1,1,2,3,4,4,5,6]
```
