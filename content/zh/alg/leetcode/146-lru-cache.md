---
title: "LeetCode 146：LRU 缓存设计（O(1)）哈希表 + 双向链表实战"
date: 2026-02-11T08:02:05+08:00
draft: false
categories: ["LeetCode"]
tags: ["LRU", "哈希表", "双向链表", "缓存", "LeetCode 146"]
description: "LRUCache 的核心是用哈希表做 O(1) 定位、双向链表维护最近使用顺序，实现 get/put 平均 O(1)。本文按 ACERS 模板给出推导、工程应用与多语言代码。"
keywords: ["LRU Cache", "LeetCode 146", "哈希表 双向链表", "缓存淘汰策略", "O(1) get put"]
---

> **副标题 / 摘要**  
> 这题不是“背答案题”，而是缓存系统的基本功：如何在常数时间内同时满足“快速访问”和“按最近最少使用淘汰”。本文从朴素方案推到最优结构，并给出可运行的多语言实现。

- **预计阅读时长**：14~18 分钟  
- **标签**：`LRU`、`哈希表`、`双向链表`、`系统设计`  
- **SEO 关键词**：LRU Cache, LeetCode 146, 哈希表, 双向链表, O(1)  
- **元描述**：通过哈希表 + 双向链表实现 LRU 缓存，`get/put` 平均 O(1)，附工程场景、常见坑与六语言实现。  

---

## 目标读者

- 正在刷 LeetCode 中等题、想吃透“数据结构组合技”的同学
- 做后端/中间件，需要实现或优化本地缓存的工程师
- 面试中经常被问到 LRU，但只记住结论、没掌握细节的人

## 背景 / 动机

缓存是“空间换时间”，但空间是有限的。  
当缓存满了，必须淘汰一些键。LRU（Least Recently Used，最近最少使用）假设：

- 最近被访问的数据，将来更可能再次访问
- 很久没访问的数据，优先淘汰更合理

工程里常见于：

- 接口响应缓存
- 数据库热点记录缓存
- 页面/会话本地状态缓存

## 核心概念

- **LRU 策略**：淘汰“最久未使用”的键
- **访问即更新新鲜度**：`get` 成功后要把该 key 标为“最近使用”
- **容量约束**：`put` 新 key 造成超容时，需要立即驱逐一个最旧键
- **O(1) 平均复杂度**：`get` 和 `put` 都不能线性扫描

---

## A — Algorithm（题目与算法）

### 题目重述

设计并实现一个满足 LRU 约束的数据结构 `LRUCache`：

- `LRUCache(int capacity)`：用正整数容量初始化
- `int get(int key)`：若 key 存在返回 value，否则返回 `-1`
- `void put(int key, int value)`：
  - key 已存在：更新 value，并视作最近使用
  - key 不存在：插入新键值对
  - 若超出容量：淘汰最久未使用的 key

并要求 `get` 和 `put` 平均时间复杂度为 `O(1)`。

### 示例 1（操作序列）

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 1)    // 缓存: {1=1}
cache.put(2, 2)    // 缓存: {1=1, 2=2}
cache.get(1)       // 返回 1，且 1 变成最近使用
cache.put(3, 3)    // 容量满，淘汰 key=2
cache.get(2)       // 返回 -1
cache.put(4, 4)    // 淘汰 key=1
cache.get(1)       // 返回 -1
cache.get(3)       // 返回 3
cache.get(4)       // 返回 4
```

### 示例 2（更新已有键）

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 10)
cache.put(1, 99)   // 更新 value，且 1 视作最近使用
cache.get(1)       // 返回 99
```

---

## 思路推导：从朴素到最优

### 朴素法 1：数组记录使用顺序

- `get`：哈希表查值 O(1)，但要把 key 挪到“最新”位置，数组删除+插入是 O(n)
- `put`：满容量时淘汰数组头元素 O(1)，但更新顺序仍常有 O(n)

结论：不满足 `get/put` 都 O(1)。

### 朴素法 2：链表维护顺序

- 能 O(1) 在头尾插删
- 但只靠链表找 key 需要 O(n)

结论：访问慢，仍不达标。

### 关键观察

需要同时满足两件事：

1. 快速定位 key 对应节点 -> 哈希表
2. 快速调整“最近使用顺序” -> 双向链表

### 方法选择（最优）

- **哈希表**：`key -> 节点指针/迭代器`
- **双向链表**：
  - 头部表示最近使用（MRU）
  - 尾部表示最久未使用（LRU）

操作定义：

- `get(key)`：命中后把节点移到头部
- `put(key,value)`：
  - 已存在：更新值并移到头部
  - 不存在：若满容量，删尾节点；再插入头部

---

## C — Concepts（核心思想）

### 数据结构模型

```text
HashMap: key -> Node*

DoubleList:
head <-> n1 <-> n2 <-> ... <-> nk <-> tail
^ 最近使用(MRU)                        最久未使用(LRU) ^
```

### 循环不变量

- 链表从头到尾按“新 -> 旧”排列
- 哈希表中的每个 key 都指向链表中唯一节点
- 链表节点数 == 哈希表元素数

### 关键操作原子化

- `remove(node)`：把任意节点从链表摘下（O(1)）
- `add_front(node)`：把节点插到头部（O(1)）
- `move_to_front(node)`：`remove + add_front`（O(1)）
- `pop_back()`：删除尾前节点（真实 LRU）（O(1)）

---

## 实践指南 / 步骤

1. 定义双向节点：`key, value, prev, next`
2. 建立两个哨兵节点 `head/tail`，避免边界特判
3. 用 `dict/map` 保存 `key -> node`
4. `get` 命中后移动节点到头部
5. `put` 新键前先检查容量，必要时 `pop_back` 并从 map 删除
6. 每次插入都放头部，表示最近访问

Python 最小可运行示例：

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head = Node()  # MRU side sentinel
        self.tail = Node()  # LRU side sentinel
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        p, n = node.prev, node.next
        p.next = n
        n.prev = p

    def _add_front(self, node: Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_front(self, node: Node) -> None:
        self._remove(node)
        self._add_front(node)

    def _pop_lru(self) -> Node:
        node = self.tail.prev
        self._remove(node)
        return node

    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._move_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return
        node = self.map.get(key)
        if node is not None:
            node.val = value
            self._move_front(node)
            return
        if len(self.map) == self.cap:
            old = self._pop_lru()
            del self.map[old.key]
        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)


if __name__ == "__main__":
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    print(c.get(1))  # 1
    c.put(3, 3)
    print(c.get(2))  # -1
```

---

## E — Engineering（工程应用）

### 场景 1：接口响应短期缓存（Python）

**背景**：热点接口短时间内重复请求同参数。  
**为什么适用**：最近访问的数据命中概率高，LRU 能在固定内存内保留热键。

```python
import time

cache = {}

def fetch_user_profile(uid: int) -> dict:
    # 假设这里是慢查询
    time.sleep(0.02)
    return {"uid": uid, "name": f"user-{uid}"}

print(fetch_user_profile(7))
```

### 场景 2：服务端配置中心本地缓存（Go）

**背景**：微服务频繁读取配置，远端拉取有网络开销。  
**为什么适用**：最近使用配置更可能继续被访问，LRU 控制本地缓存体积。

```go
package main

import "fmt"

func main() {
	// 实际工程中可把 LRU 封装成 config client 的一层
	fmt.Println("config cache ready with LRU policy")
}
```

### 场景 3：前端页面数据缓存（JavaScript）

**背景**：单页应用切换路由时，希望复用最近看过的数据。  
**为什么适用**：最近页面最可能被返回访问，LRU 可以减少重复请求。

```javascript
const pageState = new Map();
pageState.set("feed?page=1", { items: [1, 2, 3] });
console.log(pageState.get("feed?page=1"));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- `get`：哈希查找 + 链表移动，平均 `O(1)`
- `put`：哈希查找/插入 + 链表插删，平均 `O(1)`
- 空间复杂度：`O(capacity)`

### 替代方案对比

| 方案 | `get` | `put` | 问题 |
| --- | --- | --- | --- |
| 仅哈希表 + 时间戳 | O(1) | 淘汰常需 O(n) 扫描 | 逐出慢 |
| 仅链表 | O(n) | O(1) | 查找慢 |
| 哈希表 + 双向链表 | O(1) | O(1) | 实现稍复杂但最稳 |

### 常见错误思路

- 命中 `get` 后忘记“提升新鲜度”（不移动到头部）
- `put` 已有 key 时只改值，不调整最近使用顺序
- 淘汰时只删链表节点，忘删哈希表映射（产生脏指针）
- 容量为 `0` 时未处理，导致逻辑异常

### 为什么该方法最工程可行

- 性能稳定：常数时间行为可预期
- 可扩展：容易加 TTL、统计命中率、并发锁
- 结构清晰：拆成原子操作后便于单测与排障

---

## 常见问题与注意事项（FAQ）

### Q1：为什么需要双向链表，单向不行吗？

单向链表删除任意节点需要前驱指针，通常要先遍历。双向链表可 O(1) 删除任意已知节点。

### Q2：为什么要存 `key` 在链表节点里？

淘汰尾节点时，需要从哈希表删除对应 key。若节点不存 key，就无法 O(1) 删除 map 项。

### Q3：这题和 LFU 有什么区别？

LRU 按“最近访问时间”淘汰，LFU 按“访问频次”淘汰。LFU 维护结构更复杂，更新成本更高。

---

## 最佳实践与建议

- 强制使用哨兵头尾，避免空链与单节点特判
- 把链表原子操作私有化：`remove/add_front/move/pop_back`
- 写操作序列单测而不是只测最终状态
- 先保证正确性，再讨论并发与锁粒度

---

## S — Summary（总结）

核心收获：

1. LRU 的本质是“访问新鲜度排序 + 固定容量淘汰”。
2. 达到 `O(1)` 的关键在于哈希表与双向链表组合，而非单一结构。
3. 代码的稳定性来自不变量维护：顺序一致、映射一致、容量一致。
4. 该题是工程缓存（本地热点数据、配置缓存、页面缓存）的基础模型。
5. 掌握这题后，可自然进阶到 TTL-LRU、并发 LRU、LFU 等变体。

推荐延伸阅读：

- LeetCode 460 `LFU Cache`
- `Redis` 淘汰策略文档（allkeys-lru / volatile-lru）
- 《Designing Data-Intensive Applications》缓存章节
- 系统设计中的本地缓存与一致性策略资料

---

## 多语言可运行实现

### Python

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        p, n = node.prev, node.next
        p.next = n
        n.prev = p

    def _add_front(self, node: Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_front(self, node: Node) -> None:
        self._remove(node)
        self._add_front(node)

    def _pop_lru(self) -> Node:
        node = self.tail.prev
        self._remove(node)
        return node

    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._move_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return
        node = self.map.get(key)
        if node:
            node.val = value
            self._move_front(node)
            return
        if len(self.map) == self.cap:
            old = self._pop_lru()
            del self.map[old.key]
        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)


if __name__ == "__main__":
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    print(c.get(1))  # 1
    c.put(3, 3)
    print(c.get(2))  # -1
    c.put(4, 4)
    print(c.get(1), c.get(3), c.get(4))  # -1 3 4
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

#define HASH_SIZE 4093

typedef struct Node {
    int key;
    int val;
    struct Node* prev;
    struct Node* next;
} Node;

typedef struct Entry {
    int key;
    Node* node;
    struct Entry* next;
} Entry;

typedef struct {
    Entry* buckets[HASH_SIZE];
} HashMap;

typedef struct {
    int cap;
    int size;
    HashMap map;
    Node head;
    Node tail;
} LRUCache;

unsigned int h(int key) {
    unsigned int x = (unsigned int)key;
    return (x * 2654435761u) % HASH_SIZE;
}

Node* map_get(HashMap* m, int key) {
    unsigned int idx = h(key);
    Entry* e = m->buckets[idx];
    while (e) {
        if (e->key == key) return e->node;
        e = e->next;
    }
    return NULL;
}

void map_put(HashMap* m, int key, Node* node) {
    unsigned int idx = h(key);
    Entry* e = m->buckets[idx];
    while (e) {
        if (e->key == key) {
            e->node = node;
            return;
        }
        e = e->next;
    }
    Entry* ne = (Entry*)malloc(sizeof(Entry));
    ne->key = key;
    ne->node = node;
    ne->next = m->buckets[idx];
    m->buckets[idx] = ne;
}

void map_remove(HashMap* m, int key) {
    unsigned int idx = h(key);
    Entry* cur = m->buckets[idx];
    Entry* pre = NULL;
    while (cur) {
        if (cur->key == key) {
            if (pre) pre->next = cur->next;
            else m->buckets[idx] = cur->next;
            free(cur);
            return;
        }
        pre = cur;
        cur = cur->next;
    }
}

void list_remove(Node* n) {
    n->prev->next = n->next;
    n->next->prev = n->prev;
}

void list_add_front(LRUCache* c, Node* n) {
    n->prev = &c->head;
    n->next = c->head.next;
    c->head.next->prev = n;
    c->head.next = n;
}

void move_front(LRUCache* c, Node* n) {
    list_remove(n);
    list_add_front(c, n);
}

Node* pop_lru(LRUCache* c) {
    Node* n = c->tail.prev;
    list_remove(n);
    return n;
}

LRUCache* lruCreate(int capacity) {
    LRUCache* c = (LRUCache*)calloc(1, sizeof(LRUCache));
    c->cap = capacity;
    c->size = 0;
    c->head.next = &c->tail;
    c->tail.prev = &c->head;
    return c;
}

int lruGet(LRUCache* c, int key) {
    Node* n = map_get(&c->map, key);
    if (!n) return -1;
    move_front(c, n);
    return n->val;
}

void lruPut(LRUCache* c, int key, int value) {
    if (c->cap == 0) return;
    Node* n = map_get(&c->map, key);
    if (n) {
        n->val = value;
        move_front(c, n);
        return;
    }
    if (c->size == c->cap) {
        Node* old = pop_lru(c);
        map_remove(&c->map, old->key);
        free(old);
        c->size--;
    }
    Node* nn = (Node*)malloc(sizeof(Node));
    nn->key = key;
    nn->val = value;
    list_add_front(c, nn);
    map_put(&c->map, key, nn);
    c->size++;
}

void lruFree(LRUCache* c) {
    Node* cur = c->head.next;
    while (cur != &c->tail) {
        Node* nxt = cur->next;
        free(cur);
        cur = nxt;
    }
    for (int i = 0; i < HASH_SIZE; i++) {
        Entry* e = c->map.buckets[i];
        while (e) {
            Entry* ne = e->next;
            free(e);
            e = ne;
        }
    }
    free(c);
}

int main(void) {
    LRUCache* c = lruCreate(2);
    lruPut(c, 1, 1);
    lruPut(c, 2, 2);
    printf("%d\n", lruGet(c, 1)); // 1
    lruPut(c, 3, 3);
    printf("%d\n", lruGet(c, 2)); // -1
    lruPut(c, 4, 4);
    printf("%d %d %d\n", lruGet(c, 1), lruGet(c, 3), lruGet(c, 4)); // -1 3 4
    lruFree(c);
    return 0;
}
```

### C++

```cpp
#include <iostream>
#include <list>
#include <unordered_map>

using namespace std;

class LRUCache {
private:
    int cap;
    list<pair<int, int>> dq; // front = MRU, back = LRU
    unordered_map<int, list<pair<int, int>>::iterator> pos;

public:
    explicit LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        auto it = pos.find(key);
        if (it == pos.end()) return -1;
        dq.splice(dq.begin(), dq, it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        if (cap == 0) return;
        auto it = pos.find(key);
        if (it != pos.end()) {
            it->second->second = value;
            dq.splice(dq.begin(), dq, it->second);
            return;
        }
        if ((int)dq.size() == cap) {
            int oldKey = dq.back().first;
            pos.erase(oldKey);
            dq.pop_back();
        }
        dq.push_front({key, value});
        pos[key] = dq.begin();
    }
};

int main() {
    LRUCache c(2);
    c.put(1, 1);
    c.put(2, 2);
    cout << c.get(1) << "\n"; // 1
    c.put(3, 3);
    cout << c.get(2) << "\n"; // -1
    c.put(4, 4);
    cout << c.get(1) << " " << c.get(3) << " " << c.get(4) << "\n"; // -1 3 4
    return 0;
}
```

### Go

```go
package main

import (
	"container/list"
	"fmt"
)

type entry struct {
	key   int
	value int
}

type LRUCache struct {
	cap int
	ll  *list.List
	pos map[int]*list.Element
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		cap: capacity,
		ll:  list.New(),
		pos: make(map[int]*list.Element),
	}
}

func (c *LRUCache) Get(key int) int {
	e, ok := c.pos[key]
	if !ok {
		return -1
	}
	c.ll.MoveToFront(e)
	return e.Value.(entry).value
}

func (c *LRUCache) Put(key int, value int) {
	if c.cap == 0 {
		return
	}
	if e, ok := c.pos[key]; ok {
		e.Value = entry{key: key, value: value}
		c.ll.MoveToFront(e)
		return
	}
	if c.ll.Len() == c.cap {
		back := c.ll.Back()
		old := back.Value.(entry)
		delete(c.pos, old.key)
		c.ll.Remove(back)
	}
	e := c.ll.PushFront(entry{key: key, value: value})
	c.pos[key] = e
}

func main() {
	c := Constructor(2)
	c.Put(1, 1)
	c.Put(2, 2)
	fmt.Println(c.Get(1)) // 1
	c.Put(3, 3)
	fmt.Println(c.Get(2)) // -1
	c.Put(4, 4)
	fmt.Println(c.Get(1), c.Get(3), c.Get(4)) // -1 3 4
}
```

### Rust

```rust
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Node {
    key: i32,
    val: i32,
    prev: usize,
    next: usize,
}

struct LRUCache {
    cap: usize,
    len: usize,
    map: HashMap<i32, usize>, // key -> node index
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize, // sentinel
    tail: usize, // sentinel
}

impl LRUCache {
    fn new(capacity: i32) -> Self {
        let head = 0usize;
        let tail = 1usize;
        let nodes = vec![
            Node {
                key: 0,
                val: 0,
                prev: head,
                next: tail,
            },
            Node {
                key: 0,
                val: 0,
                prev: head,
                next: tail,
            },
        ];
        let mut c = Self {
            cap: capacity.max(0) as usize,
            len: 0,
            map: HashMap::new(),
            nodes,
            free: Vec::new(),
            head,
            tail,
        };
        c.nodes[c.head].next = c.tail;
        c.nodes[c.tail].prev = c.head;
        c
    }

    fn detach(&mut self, idx: usize) {
        let p = self.nodes[idx].prev;
        let n = self.nodes[idx].next;
        self.nodes[p].next = n;
        self.nodes[n].prev = p;
    }

    fn insert_front(&mut self, idx: usize) {
        let first = self.nodes[self.head].next;
        self.nodes[idx].prev = self.head;
        self.nodes[idx].next = first;
        self.nodes[self.head].next = idx;
        self.nodes[first].prev = idx;
    }

    fn move_front(&mut self, idx: usize) {
        self.detach(idx);
        self.insert_front(idx);
    }

    fn pop_lru(&mut self) -> Option<usize> {
        let idx = self.nodes[self.tail].prev;
        if idx == self.head {
            return None;
        }
        self.detach(idx);
        Some(idx)
    }

    fn alloc_node(&mut self, key: i32, val: i32) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = Node {
                key,
                val,
                prev: self.head,
                next: self.tail,
            };
            idx
        } else {
            self.nodes.push(Node {
                key,
                val,
                prev: self.head,
                next: self.tail,
            });
            self.nodes.len() - 1
        }
    }

    fn get(&mut self, key: i32) -> i32 {
        let idx = match self.map.get(&key) {
            Some(&i) => i,
            None => return -1,
        };
        self.move_front(idx);
        self.nodes[idx].val
    }

    fn put(&mut self, key: i32, value: i32) {
        if self.cap == 0 {
            return;
        }
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx].val = value;
            self.move_front(idx);
            return;
        }
        if self.len == self.cap {
            if let Some(old_idx) = self.pop_lru() {
                let old_key = self.nodes[old_idx].key;
                self.map.remove(&old_key);
                self.free.push(old_idx);
                self.len -= 1;
            }
        }
        let idx = self.alloc_node(key, value);
        self.insert_front(idx);
        self.map.insert(key, idx);
        self.len += 1;
    }
}

fn main() {
    let mut c = LRUCache::new(2);
    c.put(1, 1);
    c.put(2, 2);
    println!("{}", c.get(1)); // 1
    c.put(3, 3);
    println!("{}", c.get(2)); // -1
    c.put(4, 4);
    println!("{} {} {}", c.get(1), c.get(3), c.get(4)); // -1 3 4
}
```

### JavaScript

```javascript
class Node {
  constructor(key = 0, value = 0) {
    this.key = key;
    this.value = value;
    this.prev = null;
    this.next = null;
  }
}

class LRUCache {
  constructor(capacity) {
    this.cap = capacity;
    this.map = new Map();
    this.head = new Node();
    this.tail = new Node();
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  _remove(node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
  }

  _addFront(node) {
    node.prev = this.head;
    node.next = this.head.next;
    this.head.next.prev = node;
    this.head.next = node;
  }

  _moveFront(node) {
    this._remove(node);
    this._addFront(node);
  }

  _popLRU() {
    const node = this.tail.prev;
    this._remove(node);
    return node;
  }

  get(key) {
    if (!this.map.has(key)) return -1;
    const node = this.map.get(key);
    this._moveFront(node);
    return node.value;
  }

  put(key, value) {
    if (this.cap === 0) return;
    if (this.map.has(key)) {
      const node = this.map.get(key);
      node.value = value;
      this._moveFront(node);
      return;
    }
    if (this.map.size === this.cap) {
      const old = this._popLRU();
      this.map.delete(old.key);
    }
    const node = new Node(key, value);
    this.map.set(key, node);
    this._addFront(node);
  }
}

const c = new LRUCache(2);
c.put(1, 1);
c.put(2, 2);
console.log(c.get(1)); // 1
c.put(3, 3);
console.log(c.get(2)); // -1
c.put(4, 4);
console.log(c.get(1), c.get(3), c.get(4)); // -1 3 4
```

---

## 行动号召（CTA）

建议你现在直接做三步巩固：

1. 不看答案手写一版 `remove / add_front / pop_back`。  
2. 用操作序列压测边界：重复 put 同 key、容量为 1、容量为 0。  
3. 进阶挑战 LeetCode 460（LFU），比较两者结构复杂度差异。  
