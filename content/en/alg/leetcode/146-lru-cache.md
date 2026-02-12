---
title: "LeetCode 146: LRU Cache Design with O(1) Hash Map + Doubly Linked List"
date: 2026-02-12T13:51:08+08:00
draft: false
categories: ["LeetCode"]
tags: ["LRU", "hash map", "doubly linked list", "cache", "LeetCode 146"]
description: "Implement LRUCache with O(1) average get/put by combining a hash map for lookup and a doubly linked list for recency order. Includes ACERS reasoning, engineering mapping, and runnable multi-language code."
keywords: ["LRU Cache", "LeetCode 146", "hash map doubly linked list", "cache eviction", "O(1) get put"]
---

> **Subtitle / Summary**  
> This is not a memorization question. It is core cache-engineering practice: satisfy fast lookup and least-recently-used eviction at the same time, both in constant average time. We derive the optimal structure from naive approaches and provide runnable implementations.

- **Reading time**: 14-18 min  
- **Tags**: `LRU`, `hash map`, `doubly linked list`, `system design`  
- **SEO keywords**: LRU Cache, LeetCode 146, hash map, doubly linked list, O(1)  
- **Meta description**: Build an LRU cache with hash map + doubly linked list to achieve O(1) average `get/put`, with engineering use cases, pitfalls, and six-language implementations.

---

## Target Readers

- LeetCode learners who want to master data-structure composition
- Backend/middleware engineers implementing local caches
- Interview candidates who know the answer headline but not the invariants

## Background / Motivation

Caching trades space for time, but cache space is limited.  
When full, we must evict keys. LRU (Least Recently Used) assumes:

- Recently accessed data is more likely to be accessed again
- Long-idle data is a better eviction candidate

Real-world examples:

- API response caching
- Database hot-record caching
- Local page/session state caching

## Core Concepts

- **LRU policy**: evict the least recently used key
- **Access refresh**: successful `get` must make key most recently used
- **Capacity constraint**: `put` on new key may trigger immediate eviction
- **O(1) average complexity**: neither `get` nor `put` can do linear scans

---

## A — Algorithm (Problem & Algorithm)

### Problem Restatement

Design and implement `LRUCache`:

- `LRUCache(int capacity)`: initialize with positive capacity
- `int get(int key)`: return value if key exists, else `-1`
- `void put(int key, int value)`:
  - key exists: update value and mark it as most recently used
  - key not exists: insert key-value
  - over capacity: evict least recently used key

Both operations must run in average `O(1)` time.

### Example 1 (Operation Sequence)

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 1)    // cache: {1=1}
cache.put(2, 2)    // cache: {1=1, 2=2}
cache.get(1)       // return 1, and 1 becomes most recent
cache.put(3, 3)    // capacity full, evict key=2
cache.get(2)       // return -1
cache.put(4, 4)    // evict key=1
cache.get(1)       // return -1
cache.get(3)       // return 3
cache.get(4)       // return 4
```

### Example 2 (Update Existing Key)

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 10)
cache.put(1, 99)   // update value, key=1 is refreshed as most recent
cache.get(1)       // return 99
```

---

## Derivation: from Naive to Optimal

### Naive approach 1: array tracks recency order

- `get`: hash lookup O(1), but moving key to newest in array is O(n)
- `put`: eviction can be O(1), but recency updates are often O(n)

Conclusion: cannot guarantee O(1) for both operations.

### Naive approach 2: linked list only

- O(1) insert/delete at ends
- key lookup still O(n)

Conclusion: lookup too slow.

### Key Observation

We need both:

1. Fast key-to-node location -> hash map
2. Fast recency reordering -> doubly linked list

### Method Choice (Optimal)

- **Hash map**: `key -> node pointer/iterator`
- **Doubly linked list**:
  - head side = most recently used (MRU)
  - tail side = least recently used (LRU)

Operations:

- `get(key)`: if hit, move node to front
- `put(key,value)`:
  - exists: update and move to front
  - not exists: if full, remove tail node; then insert at front

---

## C — Concepts (Core Ideas)

### Data Structure Model

```text
HashMap: key -> Node*

DoubleList:
head <-> n1 <-> n2 <-> ... <-> nk <-> tail
^ MRU                                   LRU ^
```

### Invariants

- List order is always newest -> oldest
- Every key in map points to exactly one list node
- `list_size == map_size`

### Atomic Operations

- `remove(node)`: unlink any known node in O(1)
- `add_front(node)`: insert at front in O(1)
- `move_to_front(node)`: `remove + add_front` in O(1)
- `pop_back()`: remove least recent node in O(1)

---

## Practice Guide / Steps

1. Define doubly-linked node with `key, value, prev, next`
2. Create head/tail sentinels to avoid edge-case branches
3. Store `key -> node` in map
4. On `get` hit, move node to front
5. On new `put`, evict from back if full, then insert front
6. Always insert/refresh at front to represent recent usage

Minimal runnable Python example:

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

## E — Engineering (Real-world Scenarios)

### Scenario 1: short-term API response cache (Python)

**Background**: hot API endpoints receive repeated requests with same parameters.  
**Why it fits**: recently accessed keys are likely to be reused, while capacity stays bounded.

```python
import time

cache = {}

def fetch_user_profile(uid: int) -> dict:
    # Simulate slow query
    time.sleep(0.02)
    return {"uid": uid, "name": f"user-{uid}"}

print(fetch_user_profile(7))
```

### Scenario 2: config-center local cache in services (Go)

**Background**: microservices read config frequently; remote fetch has network overhead.  
**Why it fits**: recently used config keys are more likely to be accessed again.

```go
package main

import "fmt"

func main() {
	// In production, LRU can be a layer in your config client.
	fmt.Println("config cache ready with LRU policy")
}
```

### Scenario 3: frontend page-data cache (JavaScript)

**Background**: SPA route switching benefits from reusing recently visited data.  
**Why it fits**: recently viewed pages have higher revisit probability.

```javascript
const pageState = new Map();
pageState.set("feed?page=1", { items: [1, 2, 3] });
console.log(pageState.get("feed?page=1"));
```

---

## R — Reflection (Deep Dive)

### Complexity

- `get`: hash lookup + list move, average `O(1)`
- `put`: hash lookup/insert + list insert/delete, average `O(1)`
- Space: `O(capacity)`

### Alternative Comparison

| Approach | `get` | `put` | Problem |
| --- | --- | --- | --- |
| Hash map + timestamp only | O(1) | eviction often needs O(n) scan | slow eviction |
| Linked list only | O(n) | O(1) | slow lookup |
| Hash map + doubly linked list | O(1) | O(1) | slightly more implementation detail |

### Common Mistakes

- Hit in `get` but forget to refresh recency (not moving to front)
- Existing key in `put` updates value but not recency
- Evict from list but forget to remove map entry (stale pointer)
- Forgetting the `capacity == 0` edge case

### Why this method is production-friendly

- Stable performance and predictable constant-time behavior
- Easy extensibility: TTL, hit-rate metrics, lock wrappers
- Clear invariants and atomic operations for testing/debugging

---

## FAQ

### Q1: Why doubly linked list, not singly linked list?

Removing an arbitrary node in singly linked list needs predecessor lookup, usually O(n). Doubly linked list deletes known nodes in O(1).

### Q2: Why store `key` inside each list node?

When evicting tail node, we must remove its key from map in O(1). Without storing key in node, that step becomes expensive.

### Q3: How is this different from LFU?

LRU evicts by recency; LFU evicts by frequency. LFU requires more complex structures and update logic.

---

## Best Practices

- Always use head/tail sentinels to avoid fragile boundary branches
- Keep atomic list ops private: `remove/add_front/move/pop_back`
- Test operation sequences, not only final state snapshots
- Lock correctness first, then optimize concurrency granularity

---

## S — Summary

Key takeaways:

1. LRU is recency ordering + fixed-capacity eviction.
2. O(1) requires data-structure composition, not a single structure.
3. Stability comes from invariants: order consistency, map consistency, capacity consistency.
4. This model maps directly to practical caches in backend and frontend systems.
5. It is a strong base for TTL-LRU, concurrent LRU, and LFU extensions.

Recommended follow-ups:

- LeetCode 460 `LFU Cache`
- Redis eviction policy docs (`allkeys-lru`, `volatile-lru`)
- *Designing Data-Intensive Applications* caching chapters
- System design materials on local cache consistency strategies

---

## Runnable Multi-language Implementations

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

## CTA

Run these three drills now:

1. Re-implement `remove / add_front / pop_back` without looking at the answer.  
2. Stress-test operation sequences: repeated `put` on same key, capacity `1`, capacity `0`.  
3. Solve LeetCode 460 (LFU) and compare structure complexity with LRU.
