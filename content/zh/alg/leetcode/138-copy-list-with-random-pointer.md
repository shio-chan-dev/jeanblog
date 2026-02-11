---
title: "LeetCode 138：随机链表的复制（Copy List with Random Pointer）深拷贝全解析"
date: 2026-02-11T07:59:32+08:00
draft: false
categories: ["LeetCode"]
tags: ["链表", "哈希表", "深拷贝", "随机指针", "LeetCode 138"]
description: "随机链表复制的核心是把“原节点身份”映射到“新节点身份”，再重建 next/random 指针。本文用 ACERS 结构给出思路推导、工程类比、常见坑和多语言可运行实现。"
keywords: ["Copy List with Random Pointer", "随机链表的复制", "深拷贝", "哈希映射", "LeetCode 138"]
---

> **副标题 / 摘要**  
> 这道题的难点不是遍历链表，而是正确复制 `random` 指针所形成的“跨节点引用关系”。本文从朴素思路推导到哈希映射法，讲清为什么它稳定、可维护、易工程落地。

- **预计阅读时长**：12~16 分钟  
- **标签**：`链表`、`深拷贝`、`哈希表`、`随机指针`  
- **SEO 关键词**：LeetCode 138, Copy List with Random Pointer, 随机链表复制, 深拷贝, 哈希映射  
- **元描述**：用两趟遍历 + 映射表完成随机链表深拷贝，系统讲解正确性、复杂度、工程实践与六语言实现。  

---

## 目标读者

- 刷 LeetCode 时对 `random` 指针题目不够稳的开发者
- 想厘清“浅拷贝 vs 深拷贝”差异的同学
- 希望把算法思路迁移到工程对象复制场景的工程师

## 背景 / 动机

普通链表只要复制 `val` 和 `next`，逻辑很直观；  
但随机链表多了一个 `random` 指针，它可能：

- 指向任意节点（前面、后面、自己）
- 也可能是 `null`

这使问题从“线性复制”变成“带额外引用关系的结构复制”。  
工程里常见等价问题：

- 复制工作流节点对象，同时保留跨步骤跳转关系
- 复制缓存对象图，保持对象间引用一致
- 复制会话链，保持回溯/快捷索引引用

## 核心概念

- **浅拷贝（Shallow Copy）**：只复制节点壳，内部引用仍指向旧对象
- **深拷贝（Deep Copy）**：新建完整对象图，所有引用都指向新对象
- **节点身份映射**：`old_node -> new_node`，是重建 `random` 的关键
- **结构等价**：新链表应与旧链表在值与指针关系上同构，但完全不共享节点

---

## A — Algorithm（题目与算法）

### 题目重述

给定一个长度为 `n` 的链表，每个节点有：

- `val`
- `next`
- `random`（可指向任意节点或 `null`）

要求构造该链表的**深拷贝**并返回新头节点。  
新链表中的任何指针都不能指向原链表节点。

### 输入 / 输出表示

题面常用 `[val, random_index]` 表示每个节点：

- `val`：节点值
- `random_index`：`random` 指向的节点下标；若为空则为 `null`

你的函数入参只有 `head`，输出复制链表的头节点。

### 示例 1

```text
输入: [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出: [[7,null],[13,0],[11,4],[10,2],[1,0]]
解释: 输出与输入的“值与引用关系”一致，但节点是全新对象。
```

### 示例 2

```text
输入: [[1,1],[2,1]]
输出: [[1,1],[2,1]]
解释: 第一个节点 random 指向第二个节点，第二个节点 random 指向自己。
```

---

## 思路推导：从朴素到可维护方案

### 朴素误区：边遍历边“即时”处理 random

如果你在第一次遇到节点时就想设置 `new.random`，会遇到问题：

- `random` 目标节点可能还没复制出来
- 需要反复回填，代码分支复杂，容易漏边界

### 关键观察

`random` 无法独立于“节点身份映射”存在。  
只要建立 `old -> new` 的映射，所有指针重建都变成查表操作。

### 方法选择：两趟遍历 + 哈希映射

1. 第一趟：复制每个节点值，建立映射 `map[old] = new`
2. 第二趟：根据映射重建 `next` 与 `random`

这套方案的优点：

- 思路直观，调试成本低
- 正确性好证明
- 在面试和工程里都易维护

---

## C — Concepts（核心思想）

### 算法归类

- **链表遍历**
- **哈希映射（对象身份映射）**
- **图结构复制（特殊图：每节点最多两条出边）**

### 概念模型

把链表看成一张有向图：

- 节点集合：`V`
- 边集合：`E = {next边, random边}`

复制目标是构造同构图 `G'`，满足：

- `val(v') = val(v)`
- `f(next(v)) = next(f(v))`
- `f(random(v)) = random(f(v))`

其中 `f` 就是映射 `old -> new`。

### 正确性要点（简述）

- 第一趟后，对每个旧节点 `u` 都有唯一新节点 `f(u)`
- 第二趟对每条边 `u -> v`，设 `f(u).ptr = f(v)`（`v` 可为空）
- 因为每条 `next/random` 都按映射重连，所以结构完全等价且无旧节点泄漏

---

## 实践指南 / 步骤

1. 特判空链表：`head == null` 直接返回 `null`
2. 第一趟遍历：为每个旧节点创建新节点并存入映射
3. 第二趟遍历：设置每个新节点的 `next` 和 `random`
4. 返回 `map[head]`

Python 可运行示例：

```python
from typing import Optional, List


class Node:
    def __init__(self, x: int, next: Optional["Node"] = None, random: Optional["Node"] = None):
        self.val = x
        self.next = next
        self.random = random


def copy_random_list(head: Optional[Node]) -> Optional[Node]:
    if head is None:
        return None

    mp = {}
    cur = head
    while cur is not None:
        mp[cur] = Node(cur.val)
        cur = cur.next

    cur = head
    while cur is not None:
        mp[cur].next = mp.get(cur.next)
        mp[cur].random = mp.get(cur.random)
        cur = cur.next

    return mp[head]


def build(arr: List[List[Optional[int]]]) -> Optional[Node]:
    if not arr:
        return None
    nodes = [Node(v) for v, _ in arr]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, (_, r) in enumerate(arr):
        nodes[i].random = nodes[r] if r is not None else None
    return nodes[0]


def dump(head: Optional[Node]) -> List[List[Optional[int]]]:
    out = []
    idx = {}
    cur, i = head, 0
    while cur is not None:
        idx[cur] = i
        cur = cur.next
        i += 1
    cur = head
    while cur is not None:
        out.append([cur.val, idx.get(cur.random)])
        cur = cur.next
    return out


if __name__ == "__main__":
    data = [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]
    src = build(data)
    cp = copy_random_list(src)
    print(dump(cp))
```

---

## 代码 / 测试用例 / 测试结果

### 代码要点

- 两趟遍历，第一趟建点，第二趟连边
- `map.get(None) == None`（Python）可减少判空分支

### 测试用例

```text
用例1: []
期望: []

用例2: [[1,null]]
期望: [[1,null]]

用例3: [[1,0]]
期望: [[1,0]]  (自指 random)

用例4: [[7,null],[13,0],[11,4],[10,2],[1,0]]
期望: 同结构复制
```

### 测试结果（示例）

```text
所有测试通过：结构一致；复制链表节点地址与原链表完全不同。
```

---

## E — Engineering（工程应用）

### 场景 1：工作流定义的深拷贝（Python）

**背景**：工作流节点有顺序 `next`，也可能有“跳转节点”引用（类似 `random`）。  
**为什么适用**：复制模板生成新流程时，必须保持跳转关系且不污染原模板。

```python
class Step:
    def __init__(self, name):
        self.name = name
        self.next = None
        self.jump = None


def copy_steps(head):
    if not head:
        return None
    mp = {}
    cur = head
    while cur:
        mp[cur] = Step(cur.name)
        cur = cur.next
    cur = head
    while cur:
        mp[cur].next = mp.get(cur.next)
        mp[cur].jump = mp.get(cur.jump)
        cur = cur.next
    return mp[head]
```

### 场景 2：后台任务链路复制（Go）

**背景**：任务节点线性执行，但允许失败时跳回某补偿节点。  
**为什么适用**：失败跳转关系本质是 `random` 引用，复制时必须一并重建。

```go
package main

import "fmt"

type Task struct {
	Name   string
	Next   *Task
	Backup *Task
}

func copyTasks(head *Task) *Task {
	if head == nil {
		return nil
	}
	mp := map[*Task]*Task{}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur] = &Task{Name: cur.Name}
	}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur].Next = mp[cur.Next]
		mp[cur].Backup = mp[cur.Backup]
	}
	return mp[head]
}

func main() {
	a := &Task{Name: "A"}
	b := &Task{Name: "B"}
	a.Next = b
	b.Backup = b
	cp := copyTasks(a)
	fmt.Println(cp.Name, cp.Next.Name, cp.Next.Backup == cp.Next) // A B true
}
```

### 场景 3：前端编辑器历史链复制（JavaScript）

**背景**：编辑器历史记录通常有线性链，同时保存“快速跳转到某关键版本”的引用。  
**为什么适用**：切换用户会话时复制历史链，防止对象引用串线。

```javascript
class Version {
  constructor(id) {
    this.id = id;
    this.next = null;
    this.jump = null;
  }
}

function copyVersions(head) {
  if (!head) return null;
  const mp = new Map();
  for (let cur = head; cur; cur = cur.next) mp.set(cur, new Version(cur.id));
  for (let cur = head; cur; cur = cur.next) {
    mp.get(cur).next = mp.get(cur.next) || null;
    mp.get(cur).jump = mp.get(cur.jump) || null;
  }
  return mp.get(head);
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(n)`（两次线性遍历）
- 空间复杂度：`O(n)`（映射表）

### 替代方案对比

| 方案 | 时间 | 额外空间 | 评价 |
| --- | --- | --- | --- |
| 哈希映射两趟（本文） | O(n) | O(n) | 最易写、最稳、可维护性高 |
| 交织链表法（原地插入副本再拆分） | O(n) | O(1) | 空间更优，但实现细节更多 |
| 序列化再反序列化 | 通常 > O(n) | 取决于格式 | 工程可用，但不适合面试核心考点 |

### 常见错误思路

- 只复制 `val/next`，漏复制 `random`
- 误把新节点 `random` 指回旧链表
- 在第二趟连边时使用旧节点对象，而不是映射后的新节点
- 忘记处理 `head == null`

### 为什么当前方法工程上更可行

- 逻辑分层清晰（建点与连边分离）
- 调试简单（先看映射规模，再看指针连线）
- 对团队协作更友好，新人也容易快速接手

---

## 常见问题与注意事项（FAQ）

### Q1：为什么这题可看作“图拷贝”？

因为每个节点有两类边：`next` 和 `random`，复制的是节点与边整体关系，而不只是链式顺序。

### Q2：可以一趟遍历完成吗？

理论可做，但代码复杂度与出错率明显升高。面试与工程里更推荐两趟哈希映射版本。

### Q3：必须使用哈希表吗？

不是必须。若追求 O(1) 额外空间，可用交织链表法；但可读性通常不如哈希映射法。

---

## 最佳实践与建议

- 把“复制节点”和“重建指针”分成两个阶段，避免状态混乱
- 映射 key 使用“节点对象身份”，而不是节点值
- 写回归用例覆盖：空链表、自指 random、交叉 random、尾节点 random 为 null
- 打印调试时优先输出 `[val, random_index]`，比看地址更直观

---

## S — Summary（总结）

核心收获：

1. 这题本质是“对象身份映射 + 指针重连”，不是普通线性链表复制。
2. 两趟遍历法把问题拆成“建点”和“连边”，正确性与可维护性都更好。
3. `random` 指针的正确复制依赖 `old -> new` 的全量映射。
4. 哈希映射法是工程上极稳的基线写法，面试表达也最清晰。
5. 理解该题后，可自然迁移到图拷贝、工作流复制、对象图克隆等场景。

推荐延伸阅读：

- LeetCode 133 `Clone Graph`
- LeetCode 146 `LRU Cache`（哈希映射与链表协作）
- LeetCode 21 / 206（链表基本功巩固）
- 《Designing Data-Intensive Applications》对象关系与数据复制章节

---

## 多语言可运行实现

### Python

```python
from typing import Optional


class Node:
    def __init__(self, x: int, next: Optional["Node"] = None, random: Optional["Node"] = None):
        self.val = x
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if head is None:
            return None

        mp = {}
        cur = head
        while cur is not None:
            mp[cur] = Node(cur.val)
            cur = cur.next

        cur = head
        while cur is not None:
            mp[cur].next = mp.get(cur.next)
            mp[cur].random = mp.get(cur.random)
            cur = cur.next

        return mp[head]
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int val;
    struct Node* next;
    struct Node* random;
};

struct Node* new_node(int v) {
    struct Node* n = (struct Node*)malloc(sizeof(struct Node));
    n->val = v;
    n->next = NULL;
    n->random = NULL;
    return n;
}

// 交织链表法：O(n) time, O(1) extra space
struct Node* copyRandomList(struct Node* head) {
    if (head == NULL) return NULL;

    struct Node* cur = head;
    while (cur != NULL) {
        struct Node* cp = new_node(cur->val);
        cp->next = cur->next;
        cur->next = cp;
        cur = cp->next;
    }

    cur = head;
    while (cur != NULL) {
        struct Node* cp = cur->next;
        cp->random = (cur->random != NULL) ? cur->random->next : NULL;
        cur = cp->next;
    }

    struct Node* new_head = head->next;
    cur = head;
    while (cur != NULL) {
        struct Node* cp = cur->next;
        cur->next = cp->next;
        cp->next = (cp->next != NULL) ? cp->next->next : NULL;
        cur = cur->next;
    }
    return new_head;
}

void print_list(struct Node* head) {
    struct Node* arr[128];
    int n = 0;
    for (struct Node* p = head; p != NULL; p = p->next) arr[n++] = p;
    for (int i = 0; i < n; i++) {
        int r = -1;
        for (int j = 0; j < n; j++) {
            if (arr[i]->random == arr[j]) {
                r = j;
                break;
            }
        }
        if (r >= 0) printf("[%d,%d] ", arr[i]->val, r);
        else printf("[%d,null] ", arr[i]->val);
    }
    printf("\n");
}

int main(void) {
    struct Node* a = new_node(1);
    struct Node* b = new_node(2);
    a->next = b;
    a->random = b;
    b->random = b;
    struct Node* cp = copyRandomList(a);
    print_list(cp); // [1,1] [2,1]
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <unordered_map>

using namespace std;

class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) : val(_val), next(nullptr), random(nullptr) {}
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        unordered_map<Node*, Node*> mp;
        for (Node* cur = head; cur; cur = cur->next) {
            mp[cur] = new Node(cur->val);
        }
        for (Node* cur = head; cur; cur = cur->next) {
            mp[cur]->next = cur->next ? mp[cur->next] : nullptr;
            mp[cur]->random = cur->random ? mp[cur->random] : nullptr;
        }
        return mp[head];
    }
};
```

### Go

```go
package main

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	mp := map[*Node]*Node{}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur] = &Node{Val: cur.Val}
	}
	for cur := head; cur != nil; cur = cur.Next {
		mp[cur].Next = mp[cur.Next]
		mp[cur].Random = mp[cur.Random]
	}
	return mp[head]
}
```

### Rust

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    val: i32,
    next: Option<Rc<RefCell<Node>>>,
    random: Option<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(val: i32) -> Self {
        Self { val, next: None, random: None }
    }
}

fn copy_random_list(head: Option<Rc<RefCell<Node>>>) -> Option<Rc<RefCell<Node>>> {
    let start = head.clone()?;
    let mut mp: HashMap<*const RefCell<Node>, Rc<RefCell<Node>>> = HashMap::new();

    let mut cur = head.clone();
    while let Some(node_rc) = cur {
        let ptr = Rc::as_ptr(&node_rc);
        let val = node_rc.borrow().val;
        mp.insert(ptr, Rc::new(RefCell::new(Node::new(val))));
        cur = node_rc.borrow().next.clone();
    }

    cur = head;
    while let Some(node_rc) = cur {
        let old_ptr = Rc::as_ptr(&node_rc);
        let new_node = mp.get(&old_ptr).unwrap().clone();

        let next_old = node_rc.borrow().next.clone();
        let random_old = node_rc.borrow().random.clone();

        {
            let mut nm = new_node.borrow_mut();
            nm.next = next_old
                .as_ref()
                .and_then(|x| mp.get(&Rc::as_ptr(x)).cloned());
            nm.random = random_old
                .as_ref()
                .and_then(|x| mp.get(&Rc::as_ptr(x)).cloned());
        }

        cur = next_old;
    }

    mp.get(&Rc::as_ptr(&start)).cloned()
}

fn main() {
    let n1 = Rc::new(RefCell::new(Node::new(1)));
    let n2 = Rc::new(RefCell::new(Node::new(2)));
    n1.borrow_mut().next = Some(n2.clone());
    n1.borrow_mut().random = Some(n2.clone());
    n2.borrow_mut().random = Some(n2.clone());

    let cp = copy_random_list(Some(n1)).unwrap();
    println!("{}", cp.borrow().val); // 1
}
```

### JavaScript

```javascript
function Node(val, next = null, random = null) {
  this.val = val;
  this.next = next;
  this.random = random;
}

function copyRandomList(head) {
  if (head === null) return null;

  const mp = new Map();
  for (let cur = head; cur !== null; cur = cur.next) {
    mp.set(cur, new Node(cur.val));
  }
  for (let cur = head; cur !== null; cur = cur.next) {
    mp.get(cur).next = cur.next ? mp.get(cur.next) : null;
    mp.get(cur).random = cur.random ? mp.get(cur.random) : null;
  }
  return mp.get(head);
}
```

---

## 行动号召（CTA）

建议你现在立刻做两步巩固：

1. 不看答案手写一次“两趟哈希映射”并通过自测。
2. 再挑战 `LeetCode 133 Clone Graph`，把“身份映射复制”迁移到更一般的图结构。

如果你愿意，我下一篇可以继续写 `LeetCode 146 LRU Cache`，把“哈希 + 链表”从复制问题延伸到缓存淘汰问题。
