---
title: "Hot100: LRU Cache Hash Table + Doubly Linked List O(1) ACERS Guide"
date: 2026-04-20T09:36:56+08:00
draft: false
url: "/alg/leetcode/hot100/146-lru-cache/"
categories: ["LeetCode"]
tags: ["Hot100", "LRU", "hash table", "doubly linked list", "cache", "LeetCode 146"]
description: "Design an LRU cache where both get and put run in O(1) average time by combining a hash table with a doubly linked list. This ACERS guide derives the data-structure split and includes runnable multi-language implementations."
keywords: ["LRU Cache", "hash table doubly linked list", "O(1) get put", "LeetCode 146", "Hot100", "cache eviction"]
---

> **Subtitle / Summary**  
> LeetCode 146 is not a memorize-the-template problem. It is the core cache-design exercise: how to support fast lookup and fast recency updates at the same time.

- **Reading time**: 14-18 min  
- **Tags**: `Hot100`, `LRU`, `hash table`, `doubly linked list`  
- **SEO keywords**: LRU Cache, hash table + doubly linked list, O(1) get put, LeetCode 146, Hot100  
- **Meta description**: A derivation-first ACERS guide to LRU Cache using a hash table plus doubly linked list, with engineering scenarios, pitfalls, and runnable multi-language code.

---

## A - Algorithm (Problem and Algorithm)

### Problem Restatement

Design a data structure `LRUCache` with:

- `LRUCache(int capacity)` initializes the cache with a positive capacity
- `int get(int key)` returns the value if the key exists, otherwise `-1`
- `void put(int key, int value)` inserts or updates the key

The cache must evict the **least recently used** key when capacity is exceeded, and both `get` and `put` must run in average `O(1)` time.

### Input / Output

| Operation | Input | Output | Meaning |
| --- | --- | --- | --- |
| constructor | `capacity` | cache object | initialize cache |
| `get(key)` | `key` | value or `-1` | lookup + refresh recency |
| `put(key, value)` | `key`, `value` | none | insert/update + maybe evict |

### Example 1

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 1)    // cache: {1=1}
cache.put(2, 2)    // cache: {1=1, 2=2}
cache.get(1)       // returns 1, and 1 becomes most recently used
cache.put(3, 3)    // evicts key 2
cache.get(2)       // returns -1
cache.put(4, 4)    // evicts key 1
cache.get(1)       // returns -1
cache.get(3)       // returns 3
cache.get(4)       // returns 4
```

### Example 2

```text
LRUCache cache = new LRUCache(2)
cache.put(1, 10)
cache.put(1, 99)   // update existing key, still becomes most recent
cache.get(1)       // returns 99
```

---

## Target Readers

- Hot100 learners who want to understand the structure instead of memorizing it
- Backend engineers building local caches or request-result stores
- Interview candidates who know the answer shape but cannot yet justify it

## Background / Motivation

Any cache has two separate concerns:

- how to find data quickly
- how to decide what to evict when the cache is full

LRU assumes that recently used entries are more likely to be used again soon, so older untouched entries should be removed first.
This policy shows up in:

- API response caches
- local metadata caches
- browser-side data reuse
- service configuration hot paths

The difficulty is not the policy itself.
It is making both lookup and recency updates fast enough.

## Core Concepts

- **LRU (Least Recently Used)**: evict the key that has not been used for the longest time
- **Recency update on access**: a successful `get` changes the key's position in the usage order
- **Average `O(1)` operations**: no linear scan on `get` or `put`
- **Structure split**:
  - hash table for fast key lookup
  - doubly linked list for fast order updates

---

## C - Concepts (Core Ideas)

### How To Build The Solution From Scratch

#### Step 1: Start from the operational requirements, not the final template name

An LRU cache must support all of these at once:

- find a key quickly
- mark a key as "most recent" after access
- remove the least recent key when full

So the real question is:

> what state must be maintained so both lookup and recency changes stay constant-time?

#### Step 2: Why one structure alone is not enough

If we use only an array:

- the order is visible
- but moving an accessed entry to the front is usually `O(n)`

If we use only a linked list:

- moving nodes can be fast
- but finding a key is `O(n)`

If we use only a hash table:

- lookup is fast
- but the table does not remember which key is oldest or newest

So the problem is inherently a combination problem.

#### Step 3: Split the responsibilities cleanly

The natural split is:

- **hash table**: `key -> node`
- **doubly linked list**: maintain recency order

In the list:

- front = most recently used (MRU)
- back = least recently used (LRU)

We need a doubly linked list because removing an arbitrary node in O(1) requires direct access to both neighbors.

#### Step 4: Define the atomic list operations first

Once the structure split is clear, the whole design reduces to four atomic operations:

1. `_remove(node)` removes a node from its current position
2. `_add_front(node)` inserts a node right after the head sentinel
3. `_move_front(node)` refreshes recency after access
4. `_pop_lru()` removes the node right before the tail sentinel

Everything else is just orchestration around these operations.

#### Step 5: Ask what `get` really does

`get(key)` is not just lookup.
If the key exists, the key has now been used again.

So a successful `get` must do three things:

1. look up the node from the hash table
2. move that node to the front of the linked list
3. return the stored value

If the key does not exist, return `-1`.

#### Step 6: Ask what `put` really does

`put(key, value)` has two branches:

- key already exists:
  - update the value
  - move the node to the front
- key does not exist:
  - if cache is full, evict the current LRU node
  - create the new node
  - insert it at the front
  - add it to the hash table

This is why the list and the map must stay synchronized at all times.

#### Step 7: Walk one short trace slowly

Capacity = 2:

```text
put(1,1)  -> [1]
put(2,2)  -> [2,1]   front is newest
get(1)    -> [1,2]
put(3,3)  -> evict 2, get [3,1]
```

Notice the split:

- hash table finds `1` in O(1)
- doubly linked list moves `1` to the front in O(1)
- tail-side eviction removes `2` in O(1)

#### Step 8: Reduce the method to one sentence

LeetCode 146 is "use a hash table to find nodes fast and a doubly linked list to reorder and evict nodes fast."

### Assemble the Full Code

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
        self.head = Node()  # MRU-side sentinel
        self.tail = Node()  # LRU-side sentinel
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
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))  # 1
    cache.put(3, 3)
    print(cache.get(2))  # -1
```

### Reference Answer

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
```

### Data Structure Model

- `map[key] = node` lets us locate an entry in average O(1)
- the doubly linked list stores entries in recency order
- `head` and `tail` sentinels remove boundary branches for insertions and removals

### Loop / State Invariant

At all times:

1. every key in the map points to exactly one real list node
2. every real list node belongs to exactly one key in the map
3. nodes are ordered from MRU near `head` to LRU near `tail`

Correctness is mostly about preserving these three facts after each operation.

### Why atomic list operations matter

If `_remove`, `_add_front`, `_move_front`, and `_pop_lru` are correct, then:

- `get` becomes "lookup + move"
- `put` becomes "update or evict + insert"

That decomposition keeps the implementation defendable and debuggable.

## Practice Guide / Steps

1. Build a doubly linked list with `head` and `tail` sentinels.
2. Store `key -> node` in a hash table.
3. Implement `_remove(node)` and `_add_front(node)` first.
4. Implement `_move_front(node)` and `_pop_lru()` from those primitives.
5. In `get`, return `-1` on miss; otherwise move the node to the front.
6. In `put`, update if the key exists; otherwise evict from the tail when full, then insert the new node at the front.

Runnable Python example (`lru_cache.py`):

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

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_front(self, node):
        self._remove(node)
        self._add_front(node)

    def _pop_lru(self):
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
    print(c.get(1))
    c.put(3, 3)
    print(c.get(2))
```

---

## Explanation / Why This Works

The whole design comes from splitting two responsibilities that no single simple structure can satisfy alone:

- the hash table answers "where is this key?"
- the doubly linked list answers "which key is newest or oldest?"

Because nodes can be removed and inserted in O(1), and keys can be located in O(1) on average, the combined operations also stay O(1) on average.

---

## E - Engineering (Real-world Scenarios)

### Scenario 1: short-lived API response cache (Python)

**Background**: a backend service caches recent endpoint results.  
**Why it fits**: lookups and recency refreshes must stay cheap under load.

```python
from collections import OrderedDict


class TinyLRU:
    def __init__(self, cap):
        self.cap = cap
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key, last=False)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data[key] = value
            self.data.move_to_end(key, last=False)
            return
        if len(self.data) == self.cap:
            self.data.popitem(last=True)
        self.data[key] = value
        self.data.move_to_end(key, last=False)
```

### Scenario 2: service configuration hot cache (Go)

**Background**: a service keeps the most recently requested config entries in memory.  
**Why it fits**: fast hits and cheap eviction are more important than perfect historical statistics.

```go
package main

import (
	"container/list"
	"fmt"
)

type entry struct {
	key string
	val string
}

func main() {
	ll := list.New()
	pos := map[string]*list.Element{}
	put := func(key, val string, cap int) {
		if e, ok := pos[key]; ok {
			e.Value = entry{key: key, val: val}
			ll.MoveToFront(e)
			return
		}
		if ll.Len() == cap {
			back := ll.Back()
			delete(pos, back.Value.(entry).key)
			ll.Remove(back)
		}
		pos[key] = ll.PushFront(entry{key: key, val: val})
	}
	put("a", "1", 2)
	put("b", "2", 2)
	put("c", "3", 2)
	fmt.Println(ll.Front().Value)
}
```

### Scenario 3: frontend page-data cache (JavaScript)

**Background**: a browser view keeps recently opened page data to avoid refetching.  
**Why it fits**: the UI needs cheap updates when users revisit one of the recent pages.

```javascript
class TinyLRU {
  constructor(capacity) {
    this.cap = capacity;
    this.map = new Map();
  }

  get(key) {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  put(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    else if (this.map.size === this.cap) {
      const lruKey = this.map.keys().next().value;
      this.map.delete(lruKey);
    }
    this.map.set(key, value);
  }
}
```

---

## R - Reflection (Complexity, Alternatives, Tradeoffs)

### Complexity

- `get`: average `O(1)`
- `put`: average `O(1)`
- Space: `O(capacity)`

### Alternatives

| Method | Lookup | Recency update | Eviction | Notes |
| --- | --- | --- | --- | --- |
| Array only | O(n) | O(n) | O(1) or O(n) | poor for updates |
| Linked list only | O(n) | O(1) | O(1) | poor for lookup |
| Hash table only | O(1) | hard | hard | no order information |
| Hash table + doubly linked list | O(1) avg | O(1) | O(1) | intended solution |

### Common mistakes

1. Forgetting to move a key to the front after `get`.
2. Updating the list but not the hash table, or vice versa.
3. Using a singly linked list and then discovering arbitrary removal is no longer O(1).
4. Forgetting to store `key` inside the node, making eviction cleanup harder.

### Why this is the best practical method

It directly matches the two required capabilities:

- key lookup
- ordered eviction

No simpler structure covers both without sacrificing one of the O(1) requirements.

---

## FAQ and Notes

1. **Why do we need a doubly linked list instead of a singly linked list?**  
   Because removing an arbitrary node in O(1) requires access to its predecessor and successor.

2. **Why must the node store the `key` as well as the `value`?**  
   When evicting from the tail, we must also delete the key from the hash table.

3. **How is this different from LFU?**  
   LRU tracks recency only; LFU tracks usage frequency and usually needs a more complex structure.

---

## Best Practices

- Implement and test the four atomic list operations before `get` and `put`.
- Use sentinels to remove edge-case branches at the list ends.
- Keep the map and list updates in the same logical block.
- Test edge cases: repeated `put` on same key, capacity 1, and capacity 0.

---

## S - Summary

- LRU cache design is fundamentally a two-structure coordination problem.
- The hash table gives O(1) average lookup.
- The doubly linked list gives O(1) recency updates and O(1) tail eviction.
- Once the atomic operations are correct, the whole design becomes straightforward.

### Further Reading

- LeetCode 146. LRU Cache
- LeetCode 460. LFU Cache
- Python `OrderedDict`
- Go `container/list`

---

## Conclusion

If you can explain why a hash table alone fails, why a linked list alone fails, and why the combined structure succeeds, then you understand the cache design, not just the answer.

---

## References

- https://leetcode.com/problems/lru-cache/
- https://docs.python.org/3/library/collections.html#collections.OrderedDict
- https://pkg.go.dev/container/list
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map

---

## Meta Info

- **Reading time**: 14-18 min
- **Tags**: Hot100, LRU, hash table, doubly linked list, LeetCode 146
- **SEO keywords**: LRU Cache, hash table + doubly linked list, O(1) get put, LeetCode 146, Hot100
- **Meta description**: A derivation-first ACERS guide to LRU Cache using hash-table lookup and doubly linked list recency tracking, with runnable multi-language implementations.

---

## CTA

After this article, do three drills:

1. Re-implement `_remove`, `_add_front`, and `_pop_lru` without looking.
2. Test the cache with repeated `put` on the same key.
3. Compare this design with LFU to understand why LFU needs more bookkeeping.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

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
    map: HashMap<i32, usize>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
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
