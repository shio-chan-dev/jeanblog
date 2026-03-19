---
title: "克隆图：哈希表 + DFS/BFS 实现无向图深拷贝（LeetCode 133）"
date: 2026-03-19T13:18:26+08:00
draft: false
categories: ["LeetCode"]
tags: ["图", "DFS", "BFS", "哈希表", "深拷贝", "LeetCode 133"]
description: "用原节点到新节点的映射表实现无向连通图深拷贝，覆盖 DFS 与 BFS 两种写法、环处理、正确性分析与多语言可运行实现。"
keywords: ["Clone Graph", "克隆图", "图深拷贝", "DFS", "BFS", "LeetCode 133"]
---

> **副标题 / 摘要**  
> Clone Graph 不是单纯的图遍历题，而是“带环对象图的深拷贝”题。真正的关键不是能不能走完图，而是如何保证每个原节点只克隆一次，并且所有边都指向克隆图中的新节点。

- **预计阅读时长**：12~15 分钟  
- **标签**：`图`、`DFS`、`BFS`、`哈希表`、`深拷贝`  
- **SEO 关键词**：克隆图, Clone Graph, 图深拷贝, LeetCode 133, DFS, BFS  
- **元描述**：通过“原节点 -> 新节点”映射表实现无向图深拷贝，讲清 DFS/BFS、环处理、复杂度与多语言代码。  

---

## 目标读者

- 刷 LeetCode 图论题、希望掌握“深拷贝 + visited/map”模板的学习者  
- 需要复制对象图、工作流图、拓扑结构的工程师  
- 经常在“图遍历”和“对象复制”之间混淆的开发者  

## 背景 / 动机

很多同学第一次做这题，会把它当成普通遍历题：

- DFS 一遍
- BFS 一遍
- 把值抄过去

但真正难点在于：

- 图里可能有环
- 同一个节点可能从多条路径访问到
- 复制出来的新图，所有邻居必须指向“新节点”，不能混入旧节点引用

所以这题本质上是：

> **带环对象图的深拷贝问题**

这类模式在工程里也很常见：

- 复制流程编排图
- 克隆编辑器里的节点网络
- 复制依赖关系图做快照

## 核心概念

- **深拷贝**：返回的新图里每个节点都必须是新建对象  
- **节点身份**：判断“是不是同一个节点”看的是对象身份 / 引用，不只是 `val`  
- **邻接关系保持**：新图的边结构必须与原图完全一致  
- **映射表**：`原节点 -> 克隆节点`，既防止死循环，也防止重复创建  

---

## A — Algorithm（题目与算法）

### 题目重述

给定一个无向连通图中某个节点 `node` 的引用，请返回这个图的**深拷贝**。

每个节点结构如下：

```text
class Node {
    public int val;
    public List<Node> neighbors;
}
```

题目测试用例使用邻接表表示图。  
如果图不为空，给定节点总是值为 `1` 的节点。

### 输入 / 输出

| 名称 | 类型 | 含义 |
| --- | --- | --- |
| `node` | `Node` 或 `null` | 原图中的某个节点引用 |
| 返回值 | `Node` 或 `null` | 克隆图中的对应节点引用 |

### 示例 1

```text
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
```

解释：

- 节点 1 的邻居是 2、4
- 节点 2 的邻居是 1、3
- 节点 3 的邻居是 2、4
- 节点 4 的邻居是 1、3

复制后图结构完全一样，但所有节点都必须是新对象。

### 示例 2

```text
输入：adjList = [[]]
输出：[[]]
```

只有一个节点，且没有邻居。

### 示例 3

```text
输入：adjList = []
输出：[]
```

空图，返回 `null`。

### 约束

- 节点数范围为 `[0, 100]`
- `1 <= Node.val <= 100`
- 每个节点的 `val` 唯一
- 图中没有重复边，也没有自环
- 图是连通图，所有节点都可从给定节点到达

---

## 思路推导：从错误复制到正确模板

### 错误思路 1：看到一个节点就立刻 new 一个，再递归邻居

如果不记录“这个原节点以前是否复制过”，一旦图中有环，就会出问题。

例如：

```text
1 -- 2
|    |
4 -- 3
```

你从 1 走到 2，再从 2 走回 1，如果没有映射表，就会：

- 重复创建节点 1 的副本
- 或者递归无限循环

### 错误思路 2：只按值复制，不按节点身份复制

在 LeetCode 133 里，节点值恰好唯一，所以“按值建表”碰巧也能过。  
但工程上更稳的模式是：

> 永远按“原节点对象引用”建立映射，而不是依赖值的唯一性。

这样在一般对象图复制场景里也不会翻车。

### 关键观察

每个原节点应该只克隆一次。  
之后任何边只要再指向它，都应该复用之前那份克隆节点。

这就自然导向：

- 遍历：DFS 或 BFS
- 配套哈希表：`原节点 -> 克隆节点`

---

## C — Concepts（核心思想）

### 方法归类

- 图遍历
- 哈希表 / 记忆化
- 深拷贝构造

### 为什么映射表是必须的

映射表同时解决两个问题：

1. **防止有环时无限递归 / 无限入队**
2. **防止多个路径指向同一原节点时被重复克隆**

没有映射表，就无法正确保持“共享结构”。

### DFS 写法

DFS 核心步骤：

1. 若当前节点为空，返回空
2. 若当前节点已克隆过，直接返回映射表里的副本
3. 否则新建一个克隆节点，并立刻写入映射表
4. 递归处理所有邻居，把邻居克隆结果挂到当前克隆节点上
5. 返回当前克隆节点

这里最重要的一步是：

> **先写入映射表，再递归邻居**

这是断开环的关键。

### BFS 写法

BFS 同样成立：

1. 先克隆起点，入队原节点
2. 出队一个原节点时，遍历它的所有邻居
3. 邻居若尚未克隆，则创建并入队
4. 把“邻居的克隆节点”接到“当前克隆节点”的邻居列表中

DFS 更短，BFS 更显式。  
本题两者本质一致。

### 正确性直觉

一旦原节点第一次出现：

- 就会被创建唯一一份克隆节点
- 并放入映射表

从那以后，所有指向这个原节点的边，都能稳定指向同一份克隆节点。  
这样才能同时保证：

- 克隆图节点不重复
- 边关系与原图一致

---

## E — Engineering（工程应用）

### 场景 1：复制工作流模板（Python）

**背景**  
工作流编辑器内部常把流程节点和边表示成图结构。复制一个模板时，必须完整保留边关系，但新模板不能和旧模板共享节点对象。

**为什么适用**  
这与 Clone Graph 完全同构：节点要新建，连接关系要保留，且图中可能有回边。

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

### 场景 2：服务依赖图快照（Go）

**背景**  
对服务依赖图做变更前，往往要先复制一份做模拟或回滚预案。

**为什么适用**  
依赖图可能有共享节点，甚至存在环。若只是浅拷贝，很容易把模拟修改污染到线上图结构。

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

### 场景 3：前端节点编辑器复制粘贴（JavaScript）

**背景**  
流程图 / 思维导图 / 可视化编排器常支持“复制一个子图”。

**为什么适用**  
如果复制后的节点还连回原图，那就是典型浅拷贝 bug；正确做法必须是图深拷贝。

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

## R — Reflection（反思与深入）

### 复杂度分析

设：

- `n` 为节点数
- `m` 为边数

DFS / BFS 版都会：

- 每个节点处理一次
- 每条边遍历一次

因此：

- 时间复杂度：`O(n + m)`
- 空间复杂度：`O(n)`

额外空间主要来自：

- 映射表
- DFS 递归栈或 BFS 队列

### 替代方案对比

- **DFS + 哈希表**：最常见、代码最短
- **BFS + 哈希表**：一样正确，偏显式迭代风格
- **不带映射表的递归复制**：错误，遇环必炸

### 常见错误

- 递归邻居前没有先把当前克隆节点写入映射表
- 把邻居值抄过去，却没连接到“克隆邻居节点”
- 返回的图里混入了原图节点引用
- 把题目误当成“遍历打印图结构”而不是“深拷贝对象图”

### 为什么当前方案最工程可行

这题的难点本来就不是遍历本身，而是：

> 如何在存在环和共享节点的情况下，保证“一原节点只对应一新节点”

映射表正好解决这个核心矛盾，所以 DFS/BFS + 映射表既是最稳的面试解，也是最通用的工程模式。

### FAQ

**1. 能不能按 `val` 建映射？**  
这题里值唯一，所以能过。但更通用、也更安全的写法仍然是按原节点引用建映射。

**2. 为什么必须先 `map[node] = clone` 再处理邻居？**  
因为环可能立刻回到当前节点。如果不先登记，就无法在回边时复用已创建的副本。

**3. DFS 和 BFS 该选哪个？**  
本题都可以。DFS 代码更短，BFS 更适合不想写递归或担心递归深度的场景。

---

## S — Summary（总结）

- Clone Graph 的核心不是遍历，而是“带环对象图的深拷贝”。
- 哈希表 `原节点 -> 克隆节点` 是整题的灵魂。
- 必须先登记当前克隆节点，再递归 / 遍历邻居。
- 只要守住“一原节点只克隆一次”这个不变量，DFS 和 BFS 都能正确完成复制。

## 参考与延伸阅读

- LeetCode 138：Copy List with Random Pointer
- 图遍历模板：DFS / BFS
- 工程中的对象图深拷贝与快照模式

## 下一步建议

试着把同一题分别用 DFS 和 BFS 各写一遍。  
如果你能在两种遍历方式里都保持同一份映射表不变量，这题就真正掌握了。

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

### Python 实现

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

### C 实现

```c
/*
 * LeetCode 会提供 Node 结构定义。
 * 这题在纯 C 里主要麻烦在“原节点 -> 新节点”哈希表实现比较啰嗦，
 * 但核心算法不变：
 * 1. 建映射表
 * 2. 先登记克隆节点
 * 3. 再递归克隆邻居
 */
```

### C++ 实现

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

### Go 实现

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

### Rust 实现

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

### JavaScript 实现

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
