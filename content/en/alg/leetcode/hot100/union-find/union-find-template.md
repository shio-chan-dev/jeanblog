---
title: "Union-Find Template: Derive find / union / count"
date: 2026-06-30T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Union-Find", "DSU", "graph", "connectivity", "template"]
description: "Derive the Python Union-Find template from a small set-merging task: parent, find, union, count, connected, and path compression."
keywords: ["Union-Find", "DSU", "Disjoint Set Union", "find", "union", "count", "path compression", "connected components", "Hot100"]
---

> **Subtitle / Summary**
> Union-Find is not a pair of function names to memorize first. It solves one concrete problem: decide whether two nodes already belong to the same set, and merge two sets when a connection appears.

- **Reading time**: 10-12 min
- **Tags**: `Hot100`, `Union-Find`, `DSU`, `graph`, `connectivity`
- **SEO keywords**: Union-Find, DSU, Disjoint Set Union, find, union, count, path compression
- **Meta description**: A Python Union-Find template guide that derives parent, find, union, count, connected, path compression, and connected component counting.

---

## A - Algorithm: Start From Set-Merging Pressure

### Tiny task: merge sets and answer connectivity

Suppose we have `5` nodes:

```text
0, 1, 2, 3, 4
```

At the beginning, every node is its own set:

```text
{0}, {1}, {2}, {3}, {4}
```

Now two merge operations happen:

```text
union(0, 1)
union(1, 2)
```

We need to answer three questions:

- Are `0` and `2` in the same set?
- Are `3` and `4` in the same set?
- How many sets are left?

By hand, the sets become:

```text
{0, 1, 2}, {3}, {4}
```

So:

```text
0 and 2 are in the same set
3 and 4 are not in the same set
count = 3
```

This is exactly what Union-Find needs to support:

- `find(x)`: find the representative of the set containing `x`
- `union(a, b)`: merge the sets containing `a` and `b`
- `count`: track how many sets remain

### Why not scan all sets every time?

A direct approach is to store many sets and scan them whenever we need to answer whether two nodes belong together.

That breaks down when there are many nodes, many merges, and many connectivity queries.
The same membership work gets repeated again and again.

Union-Find changes the representation:

> Instead of storing every set as a list of elements, each node points to a parent node. Nodes that eventually reach the same representative belong to the same set.

---

## Target Readers

- Learners who want to memorize and rewrite the basic Union-Find template
- Readers who get stuck on `find / union / count` in connectivity and connected-component problems
- Anyone who has seen LeetCode 547 but wants a standalone DSU template first

## C - Concepts: Grow the Template One Need at a Time

### Step 1: Let every node start as its own set

The current pressure is simple: at the start, we have `n` nodes, and every node is alone.

The smallest state is:

```python
n = 5
parent = list(range(n))
```

Now:

```text
parent = [0, 1, 2, 3, 4]
```

The meaning is:

```text
parent[x] == x
```

This means `x` is currently the representative of its own set.

This version can represent the initial state:

```text
{0}, {1}, {2}, {3}, {4}
```

It still lacks:

- a way to follow `parent` links to find the final representative of a node

### Step 2: Write the first find

Ask one concrete question:

> If `1` has already been attached under `0`, how do we know which set `1` belongs to?

For example:

```text
parent = [0, 0, 2, 3, 4]
```

Here `parent[1] = 0`, so `1` points to `0`.
And `parent[0] = 0`, so `0` is the representative.

Therefore, `find(1)` should return `0`.

The first version is:

```python
def find(x: int) -> int:
    while parent[x] != x:
        x = parent[x]
    return x
```

The `find` loop invariant is:

> At the start of each loop iteration, `x` is still in the original set; moving to `parent[x]` does not change the set, it only moves closer to the representative.

If `parent[x] == x`, we have reached the representative.

Check it:

```python
parent = [0, 0, 2, 3, 4]

assert find(0) == 0
assert find(1) == 0
assert find(2) == 2
```

This version can:

- find the representative of a node's set
- use `find(a) == find(b)` to test whether two nodes are in the same set

It still lacks:

- a way to merge two sets when a connection appears

### Step 3: union merges representatives, not raw nodes

Now process the first merge:

```text
union(0, 1)
```

The current baseline is:

```python
parent = [0, 1, 2, 3, 4]
```

Writing this directly:

```python
parent[1] = 0
```

works only because `1` is currently a representative.

The real operation is:

```text
union(a, b)
```

`a` and `b` may not be representatives.
So before merging, we must find the representatives of their sets:

```python
root_a = find(a)
root_b = find(b)
```

If the representatives are the same, the nodes are already in the same set.
If the representatives are different, attach one representative to the other:

```python
def union(a: int, b: int) -> bool:
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    return True
```

Returning `True / False` tells the caller whether a real merge happened.

Check it:

```python
parent = list(range(5))

assert union(0, 1) is True
assert find(0) == find(1)

assert union(1, 2) is True
assert find(0) == find(2)

assert find(3) != find(4)
```

This version can:

- merge the sets containing two nodes
- test connectivity between two nodes
- avoid merging the same set twice

It still lacks:

- a count of how many sets remain

### Step 4: count decreases only on a real merge

Initially there are `n` nodes and every node is its own set.
So:

```python
count = n
```

Every time `union(a, b)` really merges two different sets, the number of sets decreases by `1`.

But if `a` and `b` are already in the same set, `count` must not change.

The smallest counterexample is:

```text
n = 3
union(0, 1)  # count goes from 3 to 2
union(0, 1)  # duplicate merge, count should stay 2
```

Putting `count` inside an object keeps the state clear:

```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        self.parent[root_b] = root_a
        self.count -= 1
        return True
```

Check duplicate merging:

```python
uf = UnionFind(3)

assert uf.count == 3
assert uf.union(0, 1) is True
assert uf.count == 2
assert uf.union(0, 1) is False
assert uf.count == 2
```

This version can:

- decide whether two nodes are in the same set
- merge two different sets
- maintain the current set count

It still lacks:

- speed when `parent` chains become long

### Step 5: Path compression makes find shorter over time

The current `find` is correct, but it can be slow.

Consider a long chain:

```text
0 -> 1 -> 2 -> 3
```

If every `find(0)` walks from `0` to `3`, later queries repeat the same work.

Path compression says:

> After finding the representative, attach the visited nodes directly to that representative.

The recursive version is short:

```python
def find(self, x: int) -> int:
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]
```

The key line is:

```python
self.parent[x] = self.find(self.parent[x])
```

This does not change which set `x` belongs to.
It only rewires `x` to point directly to the final representative.

Check a chain:

```python
uf = UnionFind(4)
uf.parent = [1, 2, 3, 3]

assert uf.find(0) == 3
assert uf.parent[0] == 3
assert uf.parent[1] == 3
assert uf.parent[2] == 3
```

After path compression, the next `find(0)` reaches the representative faster.

---

## Runnable Example (Python)

Here is the basic Union-Find template.
It includes:

- `parent`
- `find`
- `union`
- `count`
- `connected`
- path compression

```python
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.count = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        self.parent[root_b] = root_a
        self.count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


if __name__ == "__main__":
    uf = UnionFind(5)

    assert uf.count == 5
    assert uf.union(0, 1) is True
    assert uf.union(1, 2) is True

    assert uf.connected(0, 2) is True
    assert uf.connected(3, 4) is False
    assert uf.count == 3

    assert uf.union(0, 2) is False
    assert uf.count == 3
```

If you prefer a function template instead of a class, memorize this shape:

```python
n = 5
parent = list(range(n))
count = n


def find(x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(a: int, b: int) -> bool:
    global count

    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return False

    parent[root_b] = root_a
    count -= 1
    return True


def connected(a: int, b: int) -> bool:
    return find(a) == find(b)
```

---

## Explanation: Why this design works

### parent does not store original graph edges

`parent[x]` does not mean there is an original graph edge between `x` and `parent[x]`.

It only means:

> For set management, `x` points to a node closer to the set representative.

Union-Find maintains set membership, not the original graph shape.

That is why Union-Find is good at answering:

```text
Are a and b connected?
How many connected components are left?
```

But it is not good at answering:

```text
What is the exact path from a to b?
What is the shortest path?
```

### Why must union call find first?

Because `a` and `b` may not be representatives.

If we write:

```python
parent[b] = a
```

we may attach a normal node under another normal node and damage the structure.

The stable pattern is always:

```python
root_a = find(a)
root_b = find(b)
```

Then only connect representatives.

### The count invariant

The `count` invariant is:

> `count` equals the number of current set representatives.

At initialization, every node is a representative, so `count = n`.

Only when `root_a != root_b` do two representatives become one representative.
That is the only time `count -= 1` is correct.

If `root_a == root_b`, the number of representatives did not change, so `count` must not change.

---

## R - Reflection: Complexity, Tradeoffs, and Pitfalls

### Complexity

Let there be `n` nodes and `m` `find/union` operations.

With path compression, Union-Find operations are amortized close to `O(1)`.
More formally, with path compression plus union by rank or union by size, the complexity is:

```text
O(alpha(n))
```

Here `alpha(n)` is the inverse Ackermann function.
For practical input sizes, it behaves like a constant.

This basic template keeps path compression but does not put rank or size into the main line.
The reason is:

- this version is enough for many basic problems
- the template is shorter and easier to rewrite
- rank/size optimizes tree height, but it is not the first idea needed to understand `find / union / count`

### Common mistakes

- Writing `parent[b] = a` inside `union(a, b)` without finding representatives first
- Decreasing `count` even when two nodes are already in the same set
- Making `find(x)` return only `parent[x]` instead of the final representative
- Treating Union-Find as a structure that can recover the exact path between two nodes
- Adding rank or size before the basic template is stable

### When should you use Union-Find?

Use it when:

- connections arrive one by one
- you need to test whether two nodes are connected
- you need to count connected components
- you only care about set membership, not the exact path

Avoid it when:

- you need shortest paths
- you need to output the path from `a` to `b`
- edges can be deleted and connectivity must stay exact online
- the problem is directed strongly connected components

---

## S - Summary

- Union-Find solves set membership and set merging.
- `parent[x] == x` means `x` is a set representative.
- `find(x)` follows `parent` links until it reaches the final representative.
- `union(a, b)` must find the two representatives before merging them.
- `count` decreases only when two different sets are really merged.
- Path compression does not change the answer; it makes later `find` calls faster.

### Further Practice

- LeetCode 547: Number of Provinces
- Number of Islands variants: map 2D coordinates to 1D ids
- Redundant Connection: use Union-Find to detect whether an edge connects nodes already in the same set
- Kruskal minimum spanning tree: sort edges by weight and use Union-Find to detect cycles
