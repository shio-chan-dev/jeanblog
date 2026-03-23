---
title: "LeetCode 133: Clone Graph Hash Map + DFS/BFS ACERS Guide"
date: 2026-03-19T13:18:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["graph", "dfs", "bfs", "hash map", "deep copy", "LeetCode 133", "ACERS"]
description: "Clone a connected undirected graph by mapping original nodes to cloned nodes, with DFS and BFS variants, cycle handling, correctness reasoning, and runnable implementations in six languages."
keywords: ["Clone Graph", "graph deep copy", "DFS", "BFS", "hash map", "LeetCode 133"]
---

> **Subtitle / Summary**  
> Clone Graph is not a traversal-only problem. The real challenge is preserving graph structure while avoiding duplicate copies in the presence of cycles. The stable solution is a traversal plus a hash map from original nodes to cloned nodes.

- **Reading time**: 12-15 min  
- **Tags**: `graph`, `dfs`, `bfs`, `hash map`, `deep copy`  
- **SEO keywords**: Clone Graph, graph deep copy, DFS, BFS, LeetCode 133  
- **Meta description**: Deep-copy an undirected graph with a node-to-node map, explaining why memoization is mandatory and how DFS/BFS versions work, with runnable code in six languages.

---

## Target Readers

- LeetCode learners practicing graph traversal and deep-copy patterns
- Engineers who duplicate object graphs, workflow graphs, or topology graphs
- Developers who want one reusable template for “clone with cycles”

## Background / Motivation

Many “copy” problems are actually identity-preservation problems.

For arrays or flat objects, copying is straightforward.  
Graphs are different because:

- a node can be reached from multiple paths
- the graph can contain cycles
- copying only values is not enough; edges must point to cloned neighbors, not original ones

This is why Clone Graph is a classic interview and engineering problem:

- copying workflow DAGs or small cyclic state machines
- duplicating editor nodes while preserving connections
- snapshotting a topology-like data structure before mutation

## Core Concepts

- **Deep copy**: every node in the returned graph is a newly created node
- **Node identity**: the key is the original node object/reference, not only `val`
- **Adjacency structure**: each cloned node must point to cloned neighbors in the same shape as the original graph
- **Memo map**: `original_node -> cloned_node`, used to avoid repeated cloning and infinite recursion

---

## A - Algorithm

### Problem Restatement

You are given a reference to one node in a **connected undirected graph**.  
Return a **deep copy** of the entire graph.

Each node contains:

```text
class Node {
    public int val;
    public List<Node> neighbors;
}
```

The graph in the test case is represented as an adjacency list.  
The given node is always the node with value `1`, unless the graph is empty.

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `node` | `Node` or `null` | one node in the original graph |
| return | `Node` or `null` | one node in the cloned graph |

### Examples

#### Example 1

```text
Input:  adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
```

Explanation:

- Node 1 connects to 2 and 4
- Node 2 connects to 1 and 3
- Node 3 connects to 2 and 4
- Node 4 connects to 1 and 3

The cloned graph must have the same neighbor relationships, but all nodes must be newly allocated.

#### Example 2

```text
Input:  adjList = [[]]
Output: [[]]
```

There is exactly one node and it has no neighbors.

#### Example 3

```text
Input:  adjList = []
Output: []
```

The graph is empty, so the answer is `null`.

### Constraints

- The number of nodes is in the range `[0, 100]`
- `1 <= Node.val <= 100`
- `Node.val` is unique for each node
- There are no repeated edges and no self-loops
- The graph is connected and all nodes are reachable from the given node

---

## Thought Process: From Wrong Copying to the Correct Pattern

### Wrong idea 1: clone one node at a time without memory

Suppose we do this:

1. create a clone of the current node
2. recursively clone every neighbor

This breaks when the graph has a cycle.

Example:

```text
1 -- 2
|    |
4 -- 3
```

If you clone `1`, then `2`, then `1` again through the back edge, you create duplicate nodes or recurse forever.

### Wrong idea 2: use only node values as a complete substitute for node objects

In this LeetCode problem, values are unique, so value-based mapping happens to work.  
But the transferable engineering pattern is:

> map by original node identity/reference, not by coincidence of values.

That keeps the solution correct even when value uniqueness is not guaranteed in other systems.

### Key observation

Each original node should be cloned exactly once.  
After that, every edge should reuse the already-created clone.

That leads directly to:

- traversal: DFS or BFS
- memo map: `original -> cloned`

---

## C - Concepts

### Method Category

- Graph traversal
- Hash table / memoization
- Deep-copy construction

### Why the Memo Map Is Mandatory

The memo map solves two problems at once:

1. **Prevents infinite loops** on cyclic graphs  
2. **Prevents duplicate clones** when multiple paths reach the same node

Without the map, the copied graph cannot preserve shared structure correctly.

### DFS Version

The DFS idea is:

1. if the input node is `null`, return `null`
2. if the node has already been cloned, return the stored clone
3. otherwise create a new clone and store it immediately
4. recursively clone every neighbor and append the cloned neighbors
5. return the cloned node

Storing the clone **before** recursing is essential.  
That is what breaks cycles safely.

### BFS Version

The BFS version is equally valid:

1. clone the starting node
2. push the original node into a queue
3. pop nodes level by level
4. for each neighbor:
   - create its clone if missing
   - append the neighbor clone to the current clone
   - enqueue the original neighbor if first seen

DFS is usually shorter to write.  
BFS can feel more explicit if you prefer iterative traversal.

### Correctness Intuition

Once a node is first seen:

- one clone is created
- the mapping remembers that clone forever

So every future edge pointing to the original node can safely point to the same cloned node.  
That ensures both:

- node uniqueness in the clone
- edge structure preservation

### Reference Implementation

Before moving to engineering scenarios, it helps to pin down the direct interview solution first.

#### Python DFS

```python
from typing import Optional


class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        copies = {}

        def dfs(cur: Optional["Node"]) -> Optional["Node"]:
            if cur is None:
                return None
            if cur in copies:
                return copies[cur]

            cloned = Node(cur.val)
            copies[cur] = cloned
            for nxt in cur.neighbors:
                cloned.neighbors.append(dfs(nxt))
            return cloned

        return dfs(node)
```

#### Python BFS

```python
from collections import deque
from typing import Optional


class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        if node is None:
            return None

        copies = {node: Node(node.val)}
        queue = deque([node])

        while queue:
            cur = queue.popleft()
            for nxt in cur.neighbors:
                if nxt not in copies:
                    copies[nxt] = Node(nxt.val)
                    queue.append(nxt)
                copies[cur].neighbors.append(copies[nxt])

        return copies[node]
```

---

## E - Engineering

### Scenario 1: Duplicating a Workflow Graph Template (Python)

**Background**: a workflow editor stores nodes and outgoing links.  
**Why it fits**: every duplicated workflow must preserve connections without sharing mutable nodes with the original.

```python
def clone_adj(graph):
    copied = {}

    def dfs(u):
        if u in copied:
            return copied[u]
        copied[u] = []
        for v in graph.get(u, []):
            dfs(v)
            copied[u].append(v)
        return copied[u]

    for u in graph:
        dfs(u)
    return copied


workflow = {1: [2, 4], 2: [1, 3], 3: [2, 4], 4: [1, 3]}
print(clone_adj(workflow))
```

### Scenario 2: Cloning a Service Dependency Snapshot (Go)

**Background**: before mutating a service dependency graph, you want a safe snapshot.  
**Why it fits**: the graph may contain cycles, shared dependencies, and repeated reachability paths.

```go
package main

import "fmt"

func cloneAdj(graph map[int][]int) map[int][]int {
	out := map[int][]int{}
	for u, ns := range graph {
		cp := make([]int, len(ns))
		copy(cp, ns)
		out[u] = cp
	}
	return out
}

func main() {
	g := map[int][]int{1: {2, 4}, 2: {1, 3}, 3: {2, 4}, 4: {1, 3}}
	fmt.Println(cloneAdj(g))
}
```

### Scenario 3: Copy-Paste in a Frontend Node Editor (JavaScript)

**Background**: a visual editor copies a graph of blocks and edges.  
**Why it fits**: pasted blocks must point only to pasted blocks, never to the original graph.

```js
function cloneAdj(graph) {
  const out = {};
  for (const [k, v] of Object.entries(graph)) {
    out[k] = [...v];
  }
  return out;
}

const graph = {1: [2, 4], 2: [1, 3], 3: [2, 4], 4: [1, 3]};
console.log(cloneAdj(graph));
```

---

## R - Reflection

### Complexity

Let:

- `n` = number of nodes
- `m` = number of edges

Then the DFS/BFS clone visits each node once and each edge once:

- Time: `O(n + m)`
- Space: `O(n)`

The extra space comes from:

- the memo map
- the recursion stack for DFS or the queue for BFS

### Alternatives

- **DFS + hash map**: shortest and most common
- **BFS + hash map**: equally correct, iterative
- **Naive recursive copy without map**: incorrect on cyclic graphs

### Common Mistakes

- Creating the clone after processing neighbors, which breaks cycles
- Forgetting to memoize the node before recursion
- Copying neighbor values instead of neighbor node references
- Returning a shallow copy where neighbor lists still point to original nodes

### Why This Solution Is the Most Practical

This problem is fundamentally “graph traversal + identity-preserving duplication.”  
A memo map solves exactly the hard part, so DFS/BFS with mapping is both the cleanest interview solution and the most transferable engineering pattern.

### FAQ

**Why not map by `val`?**  
In this problem `val` is unique, so it works. But the more general and safer pattern is mapping original node references to cloned node references.

**Why store the clone before traversing neighbors?**  
Because a cycle may revisit the same node immediately. The map entry must already exist when that happens.

**DFS or BFS, which one is better?**  
Neither is asymptotically better here. DFS is shorter; BFS avoids recursion depth concerns.

---

## S - Summary

- Clone Graph is a deep-copy problem, not just a traversal problem.
- The essential data structure is a map from original nodes to cloned nodes.
- Memoizing before recursing is what makes cycles safe.
- DFS and BFS are both valid; the important invariant is one clone per original node.

## Further Reading

- LeetCode 138: Copy List with Random Pointer
- Graph traversal templates for DFS and BFS
- Deep-copy patterns for cyclic object graphs

## Next Step

Try rewriting the same solution in both DFS and BFS styles.  
If you can switch between the two without changing the memo-map invariant, you fully understand the problem.

---

## Multi-language Implementations

### Python

```python
from typing import Optional


class Node:
    def __init__(self, val: int = 0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        copies = {}

        def dfs(cur: Optional["Node"]) -> Optional["Node"]:
            if cur is None:
                return None
            if cur in copies:
                return copies[cur]
            cloned = Node(cur.val)
            copies[cur] = cloned
            for nxt in cur.neighbors:
                cloned.neighbors.append(dfs(nxt))
            return cloned

        return dfs(node)
```

### C

```c
/*
 * LeetCode provides the Node definition.
 * The core idea is:
 * 1. keep a map original -> cloned
 * 2. create the clone before recursing
 *
 * In pure C, the hash table implementation is verbose, so interview answers
 * often use C++/Go/Python for this problem. The algorithm itself is the same.
 */
```

### C++

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

class Solution {
public:
    unordered_map<Node*, Node*> copies;

    Node* cloneGraph(Node* node) {
        return dfs(node);
    }

    Node* dfs(Node* node) {
        if (!node) return nullptr;
        if (copies.count(node)) return copies[node];
        Node* cloned = new Node(node->val);
        copies[node] = cloned;
        for (Node* nxt : node->neighbors) {
            cloned->neighbors.push_back(dfs(nxt));
        }
        return cloned;
    }
};
```

### Go

```go
/**
 * type Node struct {
 *     Val int
 *     Neighbors []*Node
 * }
 */
func cloneGraph(node *Node) *Node {
	copies := map[*Node]*Node{}
	var dfs func(*Node) *Node
	dfs = func(cur *Node) *Node {
		if cur == nil {
			return nil
		}
		if cp, ok := copies[cur]; ok {
			return cp
		}
		cloned := &Node{Val: cur.Val, Neighbors: []*Node{}}
		copies[cur] = cloned
		for _, nxt := range cur.Neighbors {
			cloned.Neighbors = append(cloned.Neighbors, dfs(nxt))
		}
		return cloned
	}
	return dfs(node)
}
```

### Rust

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type NodeRef = Rc<RefCell<Node>>;

#[derive(Debug)]
pub struct Node {
    pub val: i32,
    pub neighbors: Vec<NodeRef>,
}

fn clone_graph(node: Option<NodeRef>) -> Option<NodeRef> {
    fn dfs(cur: &NodeRef, copies: &mut HashMap<*const RefCell<Node>, NodeRef>) -> NodeRef {
        let key = Rc::as_ptr(cur);
        if let Some(existing) = copies.get(&key) {
            return existing.clone();
        }
        let cloned = Rc::new(RefCell::new(Node { val: cur.borrow().val, neighbors: vec![] }));
        copies.insert(key, cloned.clone());
        let neighbors = cur.borrow().neighbors.clone();
        for nxt in neighbors {
            let cp = dfs(&nxt, copies);
            cloned.borrow_mut().neighbors.push(cp);
        }
        cloned
    }

    let mut copies = HashMap::new();
    node.map(|n| dfs(&n, &mut copies))
}
```

### JavaScript

```js
/*
// Definition for a Node.
function Node(val, neighbors) {
  this.val = val === undefined ? 0 : val;
  this.neighbors = neighbors === undefined ? [] : neighbors;
}
*/

var cloneGraph = function (node) {
  const copies = new Map();

  function dfs(cur) {
    if (cur === null) return null;
    if (copies.has(cur)) return copies.get(cur);
    const cloned = new Node(cur.val);
    copies.set(cur, cloned);
    for (const nxt of cur.neighbors) {
      cloned.neighbors.push(dfs(nxt));
    }
    return cloned;
  }

  return dfs(node);
};
```
