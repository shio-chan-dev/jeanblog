---
title: "最短路径三件套：BFS、Dijkstra、A* 工程实战 ACERS 解析"
date: 2026-02-09T09:48:00+08:00
draft: false
categories: ["逻辑与算法"]
tags: ["图论", "最短路径", "BFS", "Dijkstra", "A*", "双向搜索", "工程实践"]
description: "系统讲透最短路径三件套：无权图 BFS、非负权 Dijkstra、启发式 A*。覆盖多源 BFS、双向搜索、路径裁剪等工程优化，并附可运行代码与多语言模板。"
keywords: ["shortest path", "BFS", "Dijkstra", "A*", "双向 BFS", "双向 Dijkstra", "多源 BFS", "max depth"]
---

> **副标题 / 摘要**  
> 最短路径不是一道题，而是一组“按图条件选算法”的工程能力。本文按 ACERS 结构拆解 **BFS（无权）/ Dijkstra（非负权）/ A*（启发式）**，并给出你在关系图、推荐链路、路径解释里真正会用到的优化模板。

- **预计阅读时长**：14~18 分钟  
- **标签**：`图论`、`最短路径`、`BFS`、`Dijkstra`、`A*`  
- **SEO 关键词**：最短路径, BFS, Dijkstra, A*, 双向搜索, 多源 BFS  
- **元描述**：最短路径三件套工程指南：算法边界、复杂度、可运行代码、优化策略与实战场景。  

---

## 目标读者

- 正在补图算法基础，希望形成可复用工程模板的学习者
- 做社交关系链路、推荐路径、图查询解释的后端/算法工程师
- 对 BFS、Dijkstra、A* 都“知道名字”，但选型和优化还不稳定的开发者

## 背景 / 动机

最短路径问题常见于：

- 社交网络里的最短关系链路（几跳可达）
- 推荐系统里的最小代价路径（多目标折中）
- 可解释系统里的“为什么推荐给你”路径展示

工程里最容易犯的错误是“只会一个算法硬套全部场景”：

1. 用 BFS 跑加权图，结果错但不报错
2. 用 Dijkstra 跑负权边，得到不可靠结果
3. 用 A* 但启发函数不合格，性能退化成 Dijkstra

本质上，最短路径应先做 **图条件分类**，再做算法选型。

## 核心概念

| 算法 | 适用图 | 最优性条件 | 典型复杂度 | 关键词 |
| --- | --- | --- | --- | --- |
| BFS | 无权图 / 等权图 | 按层首次到达即最短边数 | `O(V+E)` | queue, level |
| Dijkstra | 非负权图 | 堆顶弹出的节点距离已最优 | `O((V+E)logV)` | relaxation, min-heap |
| A* | 非负权图 + 启发式 | `h(n)` 可采纳（不高估） | 最坏同 Dijkstra，平均更快 | `f=g+h` |

关键公式：

- **Dijkstra 松弛**：`dist[v] > dist[u] + w(u,v)` 时更新
- **A* 评估函数**：`f(n) = g(n) + h(n)`

其中：
- `g(n)` 是起点到 `n` 的已知代价
- `h(n)` 是 `n` 到终点的启发式估计代价

---

## A — Algorithm（题目与算法）

### 统一题模

给定图 `G=(V,E)`、起点 `s`、终点 `t`，求从 `s` 到 `t` 的最短路径长度与路径本身。  
图可能是无权图，也可能是非负权图。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| graph | 邻接表 | 图结构，`graph[u]` 是邻居或 `(邻居, 权重)` |
| s | 节点ID | 起点 |
| t | 节点ID | 终点 |
| 返回 | 距离 + 路径 | 不可达返回 `INF/null` 或空路径 |

### 示例 1（无权图）

```text
A -> B -> D
A -> C -> D

从 A 到 D 的最短边数 = 2
可行路径: A-B-D 或 A-C-D
```

### 示例 2（非负权图）

```text
A -> B (2)
A -> C (5)
B -> C (1)
B -> D (4)
C -> D (1)

A 到 D 最短代价 = 4
路径: A-B-C-D
```

---

## 思路推导（从朴素到工程可行）

### 朴素思路：枚举所有路径

- DFS 枚举 `s -> t` 所有路径，再取最小
- 在有环图中需要复杂去重，路径数可能指数级

结论：除极小图外不可用。

### 关键观察 1：如果边权都相同，层数就是代价

- 这时最短路径问题退化为“最少边数”
- BFS 按层扩展，第一次到达终点即最优

### 关键观察 2：边权非负时，可用贪心扩展最短前缀

- Dijkstra 每次弹出当前最小 `dist` 节点
- 由于非负权，已经弹出的节点不会被更短路径改写

### 关键观察 3：如果你知道“离目标大概多远”，可减少搜索

- A* 在 Dijkstra 基础上加启发式 `h(n)`
- 让搜索优先靠近目标，减少无关区域扩展

---

## C — Concepts（核心思想）

### 方法归类

- **BFS**：分层遍历 + 最短跳数
- **Dijkstra**：最短路树 + 松弛 + 小根堆
- **A***：最短路 + 启发式 best-first

### 三者关系

1. Dijkstra 可以看作 `h(n)=0` 的 A*
2. BFS 可以看作“所有边权为 1”时的 Dijkstra
3. A* 的性能高度依赖 `h(n)` 质量：
   - 过弱：退化为 Dijkstra
   - 过强且高估：可能失去最优性

### 工程选型矩阵

| 问题特征 | 首选算法 | 备注 |
| --- | --- | --- |
| 无权图 / hop 最短 | BFS | 关系链路、k-hop 搜索 |
| 非负权代价最短 | Dijkstra | 通用稳定，适合服务端 |
| 非负权 + 可设计启发式 | A* | 路网、空间图、解释路径 |
| 存在负权边 | Bellman-Ford/Johnson | 不用 Dijkstra/A* |

---

## 实践指南 / 步骤

### 步骤 1：先判图条件

1. 是否无权或等权？是 -> BFS
2. 是否有负权？有 -> 不能 Dijkstra/A*
3. 是否有可用启发式？有 -> 优先 A*

### 步骤 2：统一路径恢复接口

维护 `parent` 映射：`parent[v] = u`，最终从 `t` 回溯到 `s`。

### 步骤 3：实现可运行模板（Python）

```python
from collections import deque
import heapq
from math import inf


def reconstruct_path(parent, s, t):
    if t not in parent and s != t:
        return []
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs_shortest_path(graph, s, t, max_depth=None):
    """graph[u] = [v1, v2, ...]"""
    q = deque([(s, 0)])
    parent = {s: s}
    visited = {s}

    while q:
        u, d = q.popleft()
        if u == t:
            return d, reconstruct_path(parent, s, t)
        if max_depth is not None and d >= max_depth:
            continue
        for v in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append((v, d + 1))

    return inf, []


def dijkstra_shortest_path(graph, s, t, max_cost=None):
    """graph[u] = [(v, w), ...], w >= 0"""
    dist = {s: 0.0}
    parent = {s: s}
    pq = [(0.0, s)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u, inf):
            continue
        if max_cost is not None and du > max_cost:
            continue
        if u == t:
            return du, reconstruct_path(parent, s, t)
        for v, w in graph.get(u, []):
            nd = du + w
            if nd < dist.get(v, inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return inf, []


def astar_shortest_path(graph, s, t, h):
    """h(u) is admissible heuristic estimate from u to t"""
    g = {s: 0.0}
    parent = {s: s}
    pq = [(h(s), s)]  # (f, node)

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return g[u], reconstruct_path(parent, s, t)

        for v, w in graph.get(u, []):
            ng = g[u] + w
            if ng < g.get(v, inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), v))

    return inf, []


if __name__ == "__main__":
    unweighted = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    print(bfs_shortest_path(unweighted, "A", "D"))  # (2, ['A', 'B', 'D']) or C path

    weighted = {
        "A": [("B", 2), ("C", 5)],
        "B": [("C", 1), ("D", 4)],
        "C": [("D", 1)],
        "D": [],
    }
    print(dijkstra_shortest_path(weighted, "A", "D"))  # (4.0, ['A', 'B', 'C', 'D'])

    heuristic = {"A": 3, "B": 2, "C": 1, "D": 0}
    print(astar_shortest_path(weighted, "A", "D", lambda x: heuristic[x]))
```

---

## E — Engineering（工程应用）

### 场景 1：社交关系最短链路（BFS + 双向 BFS）

**背景**：给定用户 A 和用户 B，查“最短关系链路”用于可解释展示。  
**为什么适用**：无权图，目标是最少跳数，BFS 天然匹配；双向 BFS 进一步降扩展量。

```python
from collections import deque


def bidirectional_bfs(graph, s, t, max_depth=6):
    if s == t:
        return 0

    qa, qb = deque([s]), deque([t])
    da, db = {s: 0}, {t: 0}

    while qa and qb:
        # expand smaller frontier first
        if len(qa) <= len(qb):
            q, dcur, dother = qa, da, db
        else:
            q, dcur, dother = qb, db, da

        u = q.popleft()
        if dcur[u] >= max_depth:
            continue

        for v in graph.get(u, []):
            if v in dcur:
                continue
            dcur[v] = dcur[u] + 1
            if v in dother:
                return dcur[v] + dother[v]
            q.append(v)

    return -1
```

### 场景 2：推荐路径（Dijkstra）

**背景**：边权表示“代价”（时延、风险、惩罚）；要给出最低总代价路径。  
**为什么适用**：非负权图，Dijkstra 稳定且容易服务化。

```go
package main

import (
	"container/heap"
	"fmt"
)

type Edge struct{ To string; W float64 }

type Item struct{ D float64; U string }
type PQ []Item
func (p PQ) Len() int { return len(p) }
func (p PQ) Less(i, j int) bool { return p[i].D < p[j].D }
func (p PQ) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p *PQ) Push(x interface{}) { *p = append(*p, x.(Item)) }
func (p *PQ) Pop() interface{} { old := *p; x := old[len(old)-1]; *p = old[:len(old)-1]; return x }

func dijkstra(g map[string][]Edge, s, t string) float64 {
	const INF = 1e18
	dist := map[string]float64{s: 0}
	pq := &PQ{{0, s}}
	heap.Init(pq)

	for pq.Len() > 0 {
		it := heap.Pop(pq).(Item)
		if it.D != dist[it.U] { continue }
		if it.U == t { return it.D }
		for _, e := range g[it.U] {
			nd := it.D + e.W
			if d, ok := dist[e.To]; !ok || nd < d {
				dist[e.To] = nd
				heap.Push(pq, Item{nd, e.To})
			}
		}
	}
	return INF
}

func main() {
	g := map[string][]Edge{
		"A": {{"B", 2}, {"C", 5}},
		"B": {{"C", 1}, {"D", 4}},
		"C": {{"D", 1}},
	}
	fmt.Println(dijkstra(g, "A", "D")) // 4
}
```

### 场景 3：关系解释路径（A* + 路径裁剪）

**背景**：给用户展示“为什么从 X 推荐到 Y”，希望路径可解释且查询时延可控。  
**为什么适用**：A* 可利用领域先验（相似度距离）减少扩展；配合 `maxDepth` 裁剪控制成本。

```javascript
function astar(graph, s, t, h, maxDepth = 6) {
  const g = new Map([[s, 0]]);
  const pq = [[h(s), 0, s]]; // [f, depth, node]

  while (pq.length) {
    pq.sort((a, b) => a[0] - b[0]);
    const [f, depth, u] = pq.shift();
    if (u === t) return g.get(u);
    if (depth >= maxDepth) continue;

    for (const [v, w] of (graph.get(u) || [])) {
      const ng = g.get(u) + w;
      if (!g.has(v) || ng < g.get(v)) {
        g.set(v, ng);
        pq.push([ng + h(v), depth + 1, v]);
      }
    }
  }
  return Infinity;
}
```

---

## 优化要点（你必须会）

### 1) 多源 BFS

把多个起点同时入队，统一做一轮 BFS。
适用于“离任一兴趣点最近的节点”“批量感染扩散半径”等。

```python
from collections import deque

def multi_source_bfs(graph, sources):
    q = deque(sources)
    dist = {s: 0 for s in sources}
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

### 2) 双向 BFS / 双向 Dijkstra

- 双向 BFS：无权图中通常能显著减少搜索层数
- 双向 Dijkstra：非负权图可降低状态扩展，但实现复杂度更高

### 3) 路径裁剪（max depth / max cost）

在在线服务中，先保证可用延迟，再追求最优覆盖：

- BFS：`max_depth`
- Dijkstra：`max_cost`
- A*：`max_depth + 启发式`

### 4) visited bitmap / bloom

- **bitmap**：准确、内存可控（节点可映射为连续 ID 时优先）
- **bloom**：空间更省但有假阳性，适合“召回型预过滤”，不适合需要严格最优性的主判定链路

---

## R — Reflection（反思与深入）

### 复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 |
| --- | --- | --- |
| BFS | `O(V+E)` | `O(V)` |
| Dijkstra（heap） | `O((V+E)logV)` | `O(V)` |
| A* | 最坏同 Dijkstra | `O(V)` |

### 替代方案与取舍

| 方案 | 适用条件 | 成本 | 何时选 |
| --- | --- | --- | --- |
| Bellman-Ford | 可有负权 | `O(VE)` | 必须支持负权 |
| Floyd-Warshall | 全源最短路 | `O(V^3)` | 小图离线全对查询 |
| 本文三件套 | 高频在线查询 | 低到中 | 大多数工程在线路径问题 |

### 常见错误思路

1. 把 BFS 用在加权图
2. 忽略负权边检查直接上 Dijkstra
3. A* 使用不合理启发式，导致大量无效扩展
4. 过早标记 visited（在加权图里可能错失更优路径）

### 为什么这套最工程可行

- 覆盖最常见图条件（无权 + 非负权 + 有启发式）
- 可以统一接口抽象，业务层只关心“路径查询服务”
- 可与双向搜索、裁剪策略自然组合，便于 SLA 控制

---

## 解释与原理（为什么这么做）

你可以把三者看成同一条演进线：

1. BFS：按层扩展，解决“边数代价一致”
2. Dijkstra：按当前最小真实代价扩展，解决“非负权代价不同”
3. A*：在 Dijkstra 上引入“目标导向”启发式，减少无关扩展

本质区别不是代码写法，而是 **扩展顺序的依据**：

- BFS 依据层数
- Dijkstra 依据 `g`
- A* 依据 `g+h`

---

## 常见问题与注意事项

1. **图不连通怎么办？**  
   返回不可达（`INF` 或空路径），不要强行回溯路径。

2. **Dijkstra 里 visited 何时设置？**  
   推荐在“弹出并确认是当前最优 dist”时再视作确定状态。

3. **A* 的 h(n) 怎么选？**  
   路网常用曼哈顿/欧氏；图推荐可用 embedding 距离下界。必须避免系统性高估。

4. **什么时候用双向搜索？**  
   起终点都明确、图较大且分支因子高时，通常收益明显。

---

## 最佳实践与建议

- 先做图条件校验（是否无权、是否负权、是否有可用启发式）
- 把路径恢复、裁剪、日志埋点做成通用中间层
- 在线服务优先保证 tail latency：可接受时再追求全局最优覆盖
- 大图下优先邻接表 + 节点 ID 压缩 + bitmap visited

---

## S — Summary（总结）

### 核心收获

- BFS、Dijkstra、A* 是最短路径工程三件套，核心是按图条件选型
- 无权图用 BFS，非负权用 Dijkstra，有启发式再上 A*
- 多源、双向、裁剪不是锦上添花，而是线上性能与成本控制的主手段
- A* 的性能上限取决于启发函数质量，差启发会退化
- 统一路径服务接口能显著降低算法切换成本

### 推荐延伸阅读

- LeetCode 127（Word Ladder，双向 BFS）
- LeetCode 743（Network Delay Time，Dijkstra）
- A* 搜索经典：Hart, Nilsson, Raphael (1968)
- 负权场景：Bellman-Ford / Johnson

---

## 元信息

- **阅读时长**：14~18 分钟
- **标签**：图论、最短路径、BFS、Dijkstra、A*、双向搜索
- **SEO 关键词**：最短路径, BFS, Dijkstra, A*, 双向 BFS, 多源 BFS
- **元描述**：最短路径三件套工程指南：算法边界、复杂度、优化策略与可运行代码。

---

## 行动号召（CTA）

下一步建议你用同一套模板做两件事：

1. 把你当前图查询接口改成“算法可插拔”（BFS/Dijkstra/A* 可切换）
2. 加一组线上指标：扩展节点数、平均路径长度、P95 查询时延

如果你愿意，我可以下一篇直接写“负权图最短路（Bellman-Ford/Johnson）工程版”。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
# Dijkstra (non-negative weights), adjacency list
import heapq
from math import inf


def dijkstra(graph, s, t):
    dist = {s: 0.0}
    parent = {s: s}
    pq = [(0.0, s)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u, inf):
            continue
        if u == t:
            break
        for v, w in graph.get(u, []):
            nd = du + w
            if nd < dist.get(v, inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if t not in dist:
        return inf, []
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return dist[t], path
```

```c
// Dijkstra O(V^2) demo for dense/small graphs (non-negative weights)
#include <stdio.h>
#define N 5
#define INF 1000000000

int main(void) {
    int g[N][N] = {
        {0, 2, 5, 0, 0},
        {0, 0, 1, 4, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    int s = 0, t = 3;
    int dist[N], vis[N] = {0};

    for (int i = 0; i < N; i++) dist[i] = INF;
    dist[s] = 0;

    for (int i = 0; i < N; i++) {
        int u = -1;
        for (int j = 0; j < N; j++)
            if (!vis[j] && (u == -1 || dist[j] < dist[u])) u = j;
        if (u == -1 || dist[u] == INF) break;
        vis[u] = 1;
        for (int v = 0; v < N; v++) {
            if (g[u][v] > 0 && dist[v] > dist[u] + g[u][v])
                dist[v] = dist[u] + g[u][v];
        }
    }

    if (dist[t] >= INF) printf("unreachable\n");
    else printf("dist=%d\n", dist[t]);
    return 0;
}
```

```cpp
#include <bits/stdc++.h>
using namespace std;

pair<long long, vector<int>> dijkstra(int n, vector<vector<pair<int,int>>>& g, int s, int t) {
    const long long INF = (1LL<<60);
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;

    dist[s] = 0;
    parent[s] = s;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (du != dist[u]) continue;
        if (u == t) break;
        for (auto [v, w] : g[u]) {
            long long nd = du + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                parent[v] = u;
                pq.push({nd, v});
            }
        }
    }

    if (dist[t] == INF) return {INF, {}};
    vector<int> path;
    for (int x = t; x != s; x = parent[x]) path.push_back(x);
    path.push_back(s);
    reverse(path.begin(), path.end());
    return {dist[t], path};
}
```

```go
package main

import (
	"container/heap"
	"fmt"
)

type Edge struct{ To int; W int64 }
type Item struct{ D int64; U int }
type PQ []Item
func (p PQ) Len() int { return len(p) }
func (p PQ) Less(i, j int) bool { return p[i].D < p[j].D }
func (p PQ) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p *PQ) Push(x interface{}) { *p = append(*p, x.(Item)) }
func (p *PQ) Pop() interface{} { old := *p; x := old[len(old)-1]; *p = old[:len(old)-1]; return x }

func dijkstra(g [][]Edge, s, t int) int64 {
	const INF int64 = 1<<60
	dist := make([]int64, len(g))
	for i := range dist { dist[i] = INF }
	dist[s] = 0

	pq := &PQ{{0, s}}
	heap.Init(pq)

	for pq.Len() > 0 {
		it := heap.Pop(pq).(Item)
		if it.D != dist[it.U] { continue }
		if it.U == t { return it.D }
		for _, e := range g[it.U] {
			nd := it.D + e.W
			if nd < dist[e.To] {
				dist[e.To] = nd
				heap.Push(pq, Item{nd, e.To})
			}
		}
	}
	return INF
}

func main() {
	g := make([][]Edge, 4)
	g[0] = []Edge{{1, 2}, {2, 5}}
	g[1] = []Edge{{2, 1}, {3, 4}}
	g[2] = []Edge{{3, 1}}
	fmt.Println(dijkstra(g, 0, 3)) // 4
}
```

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

fn dijkstra(graph: &Vec<Vec<(usize, i64)>>, s: usize, t: usize) -> i64 {
    let inf = i64::MAX / 4;
    let mut dist = vec![inf; graph.len()];
    let mut pq = BinaryHeap::new();

    dist[s] = 0;
    pq.push((Reverse(0_i64), s));

    while let Some((Reverse(du), u)) = pq.pop() {
        if du != dist[u] { continue; }
        if u == t { return du; }
        for &(v, w) in &graph[u] {
            let nd = du + w;
            if nd < dist[v] {
                dist[v] = nd;
                pq.push((Reverse(nd), v));
            }
        }
    }
    inf
}

fn main() {
    let g = vec![
        vec![(1,2),(2,5)],
        vec![(2,1),(3,4)],
        vec![(3,1)],
        vec![]
    ];
    println!("{}", dijkstra(&g, 0, 3)); // 4
}
```

```javascript
// BFS shortest hops in unweighted graph
function bfsShortest(graph, s, t) {
  const q = [[s, 0]];
  const seen = new Set([s]);

  while (q.length) {
    const [u, d] = q.shift();
    if (u === t) return d;
    for (const v of (graph.get(u) || [])) {
      if (!seen.has(v)) {
        seen.add(v);
        q.push([v, d + 1]);
      }
    }
  }
  return Infinity;
}

const g = new Map([
  ["A", ["B", "C"]],
  ["B", ["D"]],
  ["C", ["D"]],
  ["D", []],
]);
console.log(bfsShortest(g, "A", "D")); // 2
```
