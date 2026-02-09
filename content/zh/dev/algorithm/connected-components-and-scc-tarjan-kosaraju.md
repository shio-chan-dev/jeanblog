---
title: "连通分量与强连通分量：Tarjan / Kosaraju 工程 ACERS 解析"
date: 2026-02-09T09:50:22+08:00
draft: false
description: "系统讲解无向图 Connected Components 与有向图 SCC，重点覆盖 Tarjan（工程常用）与 Kosaraju，对应图数据库中的社区划分、子图切分与分片 hint。"
tags: ["图论", "连通分量", "SCC", "Tarjan", "Kosaraju", "BFS", "DFS", "图数据库"]
categories: ["逻辑与算法"]
keywords: ["Connected Components", "Strongly Connected Components", "Tarjan", "Kosaraju", "图分片", "社区划分"]
---

> **副标题 / 摘要**  
> 连通分量是图算法的基础地基：无向图关注“是否连在一起”，有向图关注“是否互相可达”。本文按 ACERS 模板，从朴素做法推导到 Tarjan / Kosaraju，并给出图数据库落地场景与多语言可运行实现。

- **预计阅读时长**：14~18 分钟  
- **标签**：`图论`、`连通分量`、`SCC`、`Tarjan`  
- **SEO 关键词**：Connected Components, SCC, Tarjan, Kosaraju, 图数据库  
- **元描述**：从无向连通分量到有向强连通分量，讲清 Tarjan/Kosaraju 的核心机制、复杂度和工程落地。  

---

## 目标读者

- 需要把 BFS/DFS 用到“滚瓜烂熟”的算法学习者
- 在图数据库场景做子图分析、分片规划的工程师
- 想建立“无向 CC + 有向 SCC”统一认知框架的中级开发者

## 背景 / 动机

工程里你会很快遇到这三类问题：

1. 这批节点是否天然分成多个互不相连的群？（无向图连通分量）
2. 哪些节点形成“互相可达”的强闭环？（有向图 SCC）
3. 如何把大图切成更可并行、更易缓存、更易分片的子图？

如果只会 BFS/DFS 但不会“分量视角”，你会反复做可达性查询，成本高且难维护。  
连通分量算法的价值是：**一次全图扫描，把局部查询变成 O(1) 的分量 ID 比较**。

## 核心概念

- **Connected Components（CC）**：无向图中，任意两点都可达的最大节点集合
- **Strongly Connected Components（SCC）**：有向图中，任意两点互相可达的最大节点集合
- **Condensation DAG（缩点图）**：把每个 SCC 缩成一个点后得到的有向无环图
- **Tarjan 核心状态**：`dfn[u]`（时间戳），`low[u]`（可回溯到的最小时间戳），栈与 `in_stack`
- **Kosaraju 核心流程**：原图按完成时序排序 + 反图二次 DFS

---

## A — Algorithm（题目与算法）

### 题目还原（工程化表述）

给定一个图 `G=(V,E)`：

- 若 `G` 是无向图，输出所有 **Connected Components**；
- 若 `G` 是有向图，输出所有 **Strongly Connected Components**。

并返回：

- 分量总数
- 每个节点所属分量 ID

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| n | int | 节点数（`0..n-1`） |
| edges | List[(u,v)] | 边集合 |
| directed | bool | 是否有向图 |
| 返回 | (k, comp_id[]) | `k` 为分量数，`comp_id[i]` 为节点 i 的分量编号 |

### 示例 1（无向图 CC）

```text
n = 7
edges = [(0,1),(1,2),(3,4),(5,6)]

输出连通分量：
{0,1,2}, {3,4}, {5,6}
k = 3
```

### 示例 2（有向图 SCC）

```text
n = 6
edges = [(0,1),(1,2),(2,0),(2,3),(3,4),(4,3),(4,5)]

输出强连通分量：
{0,1,2}, {3,4}, {5}
k = 3
```

---

## 思路推导（从朴素到最优）

### 朴素思路

- 对每个节点做一次可达性搜索（BFS/DFS）
- 再做集合归并或交叉比较

问题：

- 时间复杂度会膨胀到 `O(V*(V+E))`
- 重复扫描同一批边，缓存局部性差，工程吞吐低

### 关键观察

1. **无向图**：从一个未访问点出发，一次 BFS/DFS 就能“吞掉”一个完整连通分量。  
2. **有向图**：单向可达不够，必须识别“互相可达”的等价类（SCC）。

### 方法选择

- 无向图：迭代 BFS/DFS + visited（最稳健）
- 有向图：Tarjan（单次 DFS、工程里更常用）
- Kosaraju：实现直观，适合作为对照与校验

---

## C — Concepts（核心思想）

### 方法归类

- 图遍历：BFS / DFS
- 分量划分：Connected Components / SCC
- 缩点建模：SCC -> DAG

### Tarjan 的不变量

在 DFS 过程中维护：

- `dfn[u]`：节点首次被访问的时间戳
- `low[u]`：从 `u` 出发，经树边 + 回边能到达的最小 `dfn`

当 `dfn[u] == low[u]` 时，`u` 是一个 SCC 的根，持续弹栈直到 `u`，得到一个完整 SCC。

### Kosaraju 的本质

1. 在原图按 DFS 完成顺序记录后序序列
2. 构建反图
3. 按后序逆序在反图 DFS，每次 DFS 得到一个 SCC

### 为什么工程里常用 Tarjan

- 一次 DFS 完成 SCC 划分（不必显式构反图）
- 常量因子小，内存行为更直接
- 更容易与在线统计（如 SCC 大小阈值）集成

---

## 实践指南 / 步骤

### 无向图 Connected Components（迭代版）

1. 建邻接表
2. 从每个未访问节点启动一次栈/队列遍历
3. 遍历过程中给节点打 `comp_id`
4. 可选 early stop：
   - 只需判断两点是否同分量时，发现同分量即停止
   - 只需 k-hop 子图时，限制层数

### 有向图 SCC（Tarjan）

1. 维护全局时间戳 `time`
2. DFS 入栈并初始化 `dfn/low`
3. 遇到未访问邻居递归；遇到栈内点更新 `low`
4. `dfn==low` 时弹栈形成 SCC

### visited 的工程选择

- **bitmap**：精确、可预测、适合固定 ID 空间
- **bloom filter**：省内存但有误判；适合“近似去重”而非严格正确性路径

---

## 可运行示例（Python）

```python
from collections import deque
from typing import List, Tuple


def connected_components_undirected(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    comp = [-1] * n
    cid = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        queue = deque([start])
        comp[start] = cid
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    queue.append(v)
        cid += 1
    return cid, comp


def scc_tarjan(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    dfn = [-1] * n
    low = [0] * n
    in_stack = [False] * n
    stack = []
    comp = [-1] * n

    time = 0
    scc_id = 0

    def dfs(u: int) -> None:
        nonlocal time, scc_id
        dfn[u] = low[u] = time
        time += 1
        stack.append(u)
        in_stack[u] = True

        for v in graph[u]:
            if dfn[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif in_stack[v]:
                low[u] = min(low[u], dfn[v])

        if dfn[u] == low[u]:
            while True:
                x = stack.pop()
                in_stack[x] = False
                comp[x] = scc_id
                if x == u:
                    break
            scc_id += 1

    for i in range(n):
        if dfn[i] == -1:
            dfs(i)

    return scc_id, comp


if __name__ == "__main__":
    n1 = 7
    e1 = [(0, 1), (1, 2), (3, 4), (5, 6)]
    k1, c1 = connected_components_undirected(n1, e1)
    print("Undirected CC count:", k1, "comp:", c1)

    n2 = 6
    e2 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)]
    k2, c2 = scc_tarjan(n2, e2)
    print("Directed SCC count:", k2, "comp:", c2)
```

运行：

```bash
python3 connected_components_demo.py
```

---

## E — Engineering（工程应用）

### 场景 1：图数据库社区粗分（Python）

**背景**：在用户关系图做社区分析前，先去掉互不连通的孤立块。  
**为什么适用**：先做 CC 可直接把后续算法（如 Louvain）作用域缩小。  

```python
def group_by_component(node_ids, comp_ids):
    groups = {}
    for node, cid in zip(node_ids, comp_ids):
        groups.setdefault(cid, []).append(node)
    return groups
```

### 场景 2：子图切分做并行任务分发（Go）

**背景**：离线图计算任务按分量拆分到 worker，减少跨 worker 通信。  
**为什么适用**：分量天然独立，任务可并行且无交叉依赖。  

```go
package main

import "fmt"

func bucketByComp(comp []int) map[int][]int {
	b := map[int][]int{}
	for node, cid := range comp {
		b[cid] = append(b[cid], node)
	}
	return b
}

func main() {
	comp := []int{0, 0, 1, 1, 2}
	fmt.Println(bucketByComp(comp))
}
```

### 场景 3：图分片 partition hint（JavaScript）

**背景**：在线图服务做分片时，希望把高耦合节点尽量放同分片。  
**为什么适用**：SCC/CC ID 可作为强信号，降低跨分片边比例。  

```javascript
function assignShardByComp(compIds, shardCount) {
  return compIds.map((cid) => cid % shardCount);
}

console.log(assignShardByComp([0, 0, 1, 1, 2, 2], 2));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 无向 CC（BFS/DFS）：`O(V+E)`，空间 `O(V)`
- Tarjan SCC：`O(V+E)`，空间 `O(V)`
- Kosaraju SCC：`O(V+E)`，空间 `O(V+E)`（含反图）

### 替代方案与取舍

| 方法 | 适用图类型 | 时间复杂度 | 优点 | 局限 |
| --- | --- | --- | --- | --- |
| BFS/DFS 连通分量 | 无向图 | O(V+E) | 直观、稳定 | 不处理 SCC |
| Tarjan | 有向图 | O(V+E) | 单遍、工程常用 | 实现门槛高于 BFS |
| Kosaraju | 有向图 | O(V+E) | 思路清晰 | 需要反图与两遍 DFS |
| 并查集 Union-Find | 无向图静态连通 | 近似 O(E α(V)) | 工程实现快 | 不适合 SCC |

### 为什么 Tarjan 更工程可行

- 与在线管道更契合：一遍扫描可直接产出 SCC ID
- 不需要构反图，减少额外内存与数据搬运
- 更容易附加统计：SCC 大小、出边数量、跨 SCC 边比例

---

## 解释与原理（为什么这么做）

- CC 的本质是“无向可达等价类”，一次遍历可完整覆盖一个等价类。
- SCC 的本质是“有向互相可达等价类”，Tarjan 用 `dfn/low + 栈`在线识别“闭环根”。
- 把节点映射到 `comp_id` 后，大量查询可降维：
  - “是否同群？” => `comp_id[u] == comp_id[v]`
  - “分片 hint？” => `hash(comp_id)`

---

## 常见问题与注意事项

1. **无向图能用 Tarjan 求 SCC 吗？**  
   可以，但没有必要；无向图直接做 CC 更简单。

2. **Tarjan 一定要递归吗？**  
   不是。可以改成显式栈迭代版，但实现复杂度更高。

3. **Bloom filter 能替代 visited 吗？**  
   严格正确性场景不能完全替代，误判会漏遍历节点。

4. **为什么我算出的 SCC 顺序不一致？**  
   SCC 划分正确即可，编号顺序受遍历顺序影响。

---

## 最佳实践与建议

- 先统一节点 ID（0..n-1）再上图算法，避免映射错误
- 优先用迭代 BFS/DFS 做无向 CC，规避深图递归栈风险
- 有向大图优先 Tarjan；若需要教学可读性再补 Kosaraju
- 在工程中把 `comp_id` 持久化，复用到查询、缓存、分片决策

---

## S — Summary（总结）

### 核心收获

- 连通分量是图计算里的“第一层降维”，一次计算可服务多类查询。
- 无向图 CC 与有向图 SCC 是两个不同问题，不能混用。
- Tarjan 用 `dfn/low` 在线识别 SCC，`O(V+E)` 且工程常用。
- Kosaraju 适合理解原理与交叉验证，Tarjan 适合生产落地。
- `comp_id` 在图数据库中可直接用于社区粗分、子图切分和分片 hint。

### 推荐延伸阅读

- Tarjan, R. (1972). Depth-first search and linear graph algorithms.
- CLRS 图算法章节（SCC、拓扑排序）
- Neo4j Graph Data Science: Connected Components / SCC 文档

### 小结 / 结论

如果你已经掌握 BFS/DFS，下一步必须把“分量思维”补齐。  
工程里真正有价值的不是“会遍历”，而是把遍历结果变成稳定可复用的结构化标签（`comp_id`）。

---

## 元信息

- **阅读时长**：14~18 分钟
- **标签**：图论、连通分量、SCC、Tarjan、图数据库
- **SEO 关键词**：Connected Components, SCC, Tarjan, Kosaraju, 图分片
- **元描述**：系统讲解无向 CC 与有向 SCC，重点覆盖 Tarjan/Kosaraju 和图数据库工程落地。

---

## 行动号召（CTA）

建议你立即做两件事：

1. 用你的业务图跑一次 CC / SCC，输出 `comp_id` 分布直方图。
2. 统计跨分量边比例，评估是否适合做分片或子图并行。

如果你愿意，我可以继续写“3️⃣ 最短路（Dijkstra / A* / 多源 BFS）”并保持同一套 ACERS 风格。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import deque


def connected_components_undirected(n, edges):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    comp = [-1] * n
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        q = deque([s])
        comp[s] = cid
        while q:
            u = q.popleft()
            for v in g[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return cid, comp
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int* data;
    int size;
    int cap;
} Vec;

void push(Vec* v, int x) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 4;
        v->data = (int*)realloc(v->data, sizeof(int) * v->cap);
    }
    v->data[v->size++] = x;
}

int main(void) {
    int n = 5;
    int comp[5] = {-1, -1, -1, -1, -1};
    // 演示占位：真实工程请按边构建邻接表并做 BFS/DFS
    comp[0] = comp[1] = 0;
    comp[2] = comp[3] = 1;
    comp[4] = 2;
    for (int i = 0; i < n; ++i) printf("node %d -> comp %d\n", i, comp[i]);
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;

pair<int, vector<int>> connectedComponentsUndirected(int n, const vector<pair<int,int>>& edges) {
    vector<vector<int>> g(n);
    for (auto [u,v] : edges) {
        g[u].push_back(v);
        g[v].push_back(u);
    }
    vector<int> comp(n, -1);
    int cid = 0;
    queue<int> q;
    for (int s = 0; s < n; ++s) {
        if (comp[s] != -1) continue;
        comp[s] = cid;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : g[u]) {
                if (comp[v] == -1) {
                    comp[v] = cid;
                    q.push(v);
                }
            }
        }
        cid++;
    }
    return {cid, comp};
}

int main() {
    vector<pair<int,int>> edges = {{0,1},{1,2},{3,4}};
    auto [k, comp] = connectedComponentsUndirected(5, edges);
    cout << "k=" << k << "\n";
    for (int i = 0; i < (int)comp.size(); ++i) cout << i << ":" << comp[i] << " ";
    cout << "\n";
}
```

```go
package main

import "fmt"

func connectedComponentsUndirected(n int, edges [][2]int) (int, []int) {
	g := make([][]int, n)
	for _, e := range edges {
		u, v := e[0], e[1]
		g[u] = append(g[u], v)
		g[v] = append(g[v], u)
	}
	comp := make([]int, n)
	for i := range comp {
		comp[i] = -1
	}
	cid := 0
	q := make([]int, 0)
	for s := 0; s < n; s++ {
		if comp[s] != -1 {
			continue
		}
		comp[s] = cid
		q = append(q, s)
		for len(q) > 0 {
			u := q[0]
			q = q[1:]
			for _, v := range g[u] {
				if comp[v] == -1 {
					comp[v] = cid
					q = append(q, v)
				}
			}
		}
		cid++
	}
	return cid, comp
}

func main() {
	edges := [][2]int{{0, 1}, {1, 2}, {3, 4}}
	k, comp := connectedComponentsUndirected(5, edges)
	fmt.Println(k, comp)
}
```

```rust
use std::collections::VecDeque;

fn connected_components_undirected(n: usize, edges: &[(usize, usize)]) -> (usize, Vec<i32>) {
    let mut g = vec![vec![]; n];
    for &(u, v) in edges {
        g[u].push(v);
        g[v].push(u);
    }
    let mut comp = vec![-1; n];
    let mut cid: i32 = 0;

    for s in 0..n {
        if comp[s] != -1 {
            continue;
        }
        let mut q = VecDeque::new();
        comp[s] = cid;
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            for &v in &g[u] {
                if comp[v] == -1 {
                    comp[v] = cid;
                    q.push_back(v);
                }
            }
        }
        cid += 1;
    }

    (cid as usize, comp)
}

fn main() {
    let edges = vec![(0, 1), (1, 2), (3, 4)];
    let (k, comp) = connected_components_undirected(5, &edges);
    println!("{} {:?}", k, comp);
}
```

```javascript
function connectedComponentsUndirected(n, edges) {
  const g = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    g[u].push(v);
    g[v].push(u);
  }

  const comp = Array(n).fill(-1);
  let cid = 0;

  for (let s = 0; s < n; s += 1) {
    if (comp[s] !== -1) continue;
    const queue = [s];
    comp[s] = cid;

    while (queue.length) {
      const u = queue.shift();
      for (const v of g[u]) {
        if (comp[v] === -1) {
          comp[v] = cid;
          queue.push(v);
        }
      }
    }
    cid += 1;
  }

  return [cid, comp];
}

console.log(connectedComponentsUndirected(5, [[0, 1], [1, 2], [3, 4]]));
```
