---
title: "k-hop 与可达性查询：BFS 限制、Reachability 索引与 2-hop Labeling ACERS 解析"
subtitle: "从在线搜索到离线索引：在时延、内存、更新成本之间做可解释取舍"
date: 2026-02-09T09:52:17+08:00
draft: false
summary: "围绕 k-hop 与可达性查询，讲清 BFS+hop 限制、传递闭包取舍、以及位图索引/2-hop labeling 的工程落地路径。"
categories: ["逻辑与算法"]
tags: ["图", "BFS", "Reachability", "k-hop", "Transitive Closure", "2-hop labeling", "位图索引"]
description: "围绕 k-hop 与可达性查询，系统讲解 BFS+hop 限制、为何一般不全算传递闭包、以及工程常用的位图索引/2-hop labeling 思路，并给出可运行多语言实现。"
keywords: ["k-hop", "Reachability", "Transitive Closure", "BFS hop limit", "2-hop labeling", "reach index", "bitset"]
readingTime: 15
---

> **副标题 / 摘要**  
> 图查询真正难的不是“能不能搜到”，而是“在时延和内存预算内稳定搜到”。本文把可达性问题拆成三层：**在线 BFS+hop 限制、离线闭包（通常不全算）、索引化查询（2-hop / reach index）**，并给出可直接落地的工程决策模板。

- **预计阅读时长**：12~16 分钟  
- **标签**：`k-hop`、`Reachability`、`BFS`、`位图索引`  
- **SEO 关键词**：k-hop, Reachability, Transitive Closure, 2-hop labeling, reach index  
- **元描述**：从在线 BFS 到索引化可达性查询，讲清 hop 限制、闭包成本与 2-hop/位图索引选型。

---

## 目标读者

- 做图数据库、风控图、依赖分析、调用链排障的工程师
- 需要把“路径是否存在”从题解变成线上能力的人
- 面临“查询多、图大、更新频繁”三难问题的系统设计者

## 背景 / 动机

可达性查询是图系统的基本能力，但工程里有三个现实矛盾：

1. 查询要快：通常是接口内同步执行（毫秒级）
2. 图要大：节点/边数量上百万到上亿
3. 更新要频繁：索引维护成本不能无限上升

因此你不能只盯一个算法，而要按场景分层：

- 在线低延迟：BFS + hop 限制 + early stop
- 离线精确：传递闭包（一般不全算）
- 查询密集：位图索引、2-hop labeling、reach index

## 核心概念

| 概念 | 定义 | 关键代价 |
| --- | --- | --- |
| Reachability | `u -> v` 是否存在路径 | 查询时延 |
| k-hop 查询 | 限制路径长度 `<= k` 的可达集合 | 前沿扩展规模 |
| Transitive Closure | 全部点对可达关系矩阵 | 预计算与存储成本 |
| 2-hop Labeling | 用中转标签判定可达性 | 标签构建与维护复杂度 |
| Reach Index | 面向查询构建的可达性索引族 | 索引体积与更新代价 |

---

## A — Algorithm（题目与算法）

### 题目还原（工程抽象版）

给定有向图 `G=(V,E)`，需要支持两类查询：

1. `reachable(u, v)`: 判断 `u` 是否可达 `v`
2. `k_hop(u, k)`: 返回从 `u` 出发 `k` 跳内可达节点集合

约束：

- 查询需支持 early stop（命中目标、超过 hop、达到预算）
- 不能递归（深图风险），采用迭代版
- 可选：引入索引提高高频查询性能

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| graph | List[List[int]] | 邻接表，节点 ID 为 0..n-1 |
| u, v | int | 起点、目标点 |
| k | int | 最大 hop |
| 返回1 | bool | 是否可达 |
| 返回2 | Set[int] | k-hop 邻域节点集合 |

### 示例 1：k-hop

```text
graph = [
  [1,2],   # 0
  [3],     # 1
  [3,4],   # 2
  [5],     # 3
  [],      # 4
  []       # 5
]
query: k_hop(0, 2)
result: {0,1,2,3,4}
```

### 示例 2：Reachability

```text
query: reachable(0, 5)
result: true
query: reachable(4, 5)
result: false
```

---

## 思路推导（从朴素到工程可用）

### 朴素方案 1：每次查询都全图 BFS

- 正确但不经济
- 对高频查询场景，重复计算过多

### 朴素方案 2：全量传递闭包（TC）

- 查询可 O(1)
- 但构建和存储通常过重（尤其大图 + 频繁更新）

### 关键观察

1. 大多数线上查询只需要局部范围（k-hop）或早停命中
2. 不是所有图都值得全算闭包
3. 索引要按“查询/更新比”选择，而非盲目追求理论最优

### 方法选择

- **在线查询优先**：BFS + hop 限制 + early stop
- **静态图高查询密度**：考虑 reach index（2-hop/位图）
- **动态图高更新频率**：尽量轻索引 + 在线搜索混合

---

## C — Concepts（核心思想）

### 1) BFS + hop 限制

对 `k-hop` 来说，BFS 是天然模型，因为层数就是 hop。

状态定义：`(node, depth)`

剪枝规则：

- `depth == k`：不再扩展邻居
- `node == target`：可达查询立即返回 true
- `visited_budget` 达上限：返回部分结果或降级

### 2) Reachability 与 Transitive Closure

传递闭包可理解为布尔可达矩阵 `R`：

- `R[u][v] = 1` 当且仅当 `u` 可达 `v`

优势：查询极快。  
代价：构建重、存储大、更新贵。

工程结论：**一般不全算**，除非图相对静态且查询密集到足以摊薄成本。

### 3) 位图索引 / 2-hop labeling / reach index

2-hop labeling 的判定形式（有向图可达性）：

- 对每个点 `x`，维护 `L_out(x)` 与 `L_in(x)`
- `u` 可达 `v` 当且仅当 `L_out(u) ∩ L_in(v) != ∅`（并结合自反规则）

优点：查询非常快。  
难点：标签构建和增量维护复杂，且标签体积受图结构影响很大。

工程上常见折中：

- 位图 reach index（压缩存储）
- 分层索引 + 在线 BFS 验证
- landmark/bloom 预过滤 + 精确搜索兜底

### 4) 2-hop labeling 最小手算例子

考虑有向图：

```text
0 -> 1 -> 3
 \\        ^
  -> 2 ----|
```

可构造一个简化标签集合（演示用途）：

- `L_out(0) = {1,2,3}`
- `L_out(1) = {3}`
- `L_out(2) = {3}`
- `L_in(3) = {0,1,2}`

查询 `reachable(0,3)` 时，只需判断：

```text
L_out(0) ∩ L_in(3) = {1,2,3} ∩ {0,1,2} = {1,2} != ∅
```

即可返回可达，而不必在线展开整条搜索前沿。  
这也是 2-hop 在读多写少场景常被采用的原因：把查询代价转移到离线构建。

---

## 实践指南 / 步骤

1. 先量化业务：QPS、P99、图规模、更新频率
2. 实现基线：迭代 BFS + hop 限制 + early stop
3. 压测后再加索引：优先位图索引或轻量 reach index
4. 索引命中失败时，用在线 BFS 做兜底
5. 对严格正确场景，bloom 只能做预过滤，不能单独判定

可运行 Python 示例（`python3 reachability_demo.py`）：

```python
from collections import deque
from typing import List, Set


def bfs_k_hop(graph: List[List[int]], s: int, k: int) -> Set[int]:
    vis = bytearray(len(graph))
    vis[s] = 1
    q = deque([(s, 0)])
    out = {s}

    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = 1
                out.add(v)
                q.append((v, d + 1))

    return out


def reachable_bfs(graph: List[List[int]], s: int, t: int, hop_limit: int | None = None) -> bool:
    vis = bytearray(len(graph))
    vis[s] = 1
    q = deque([(s, 0)])

    while q:
        u, d = q.popleft()
        if u == t:
            return True
        if hop_limit is not None and d == hop_limit:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = 1
                q.append((v, d + 1))

    return False


def transitive_closure_small(graph: List[List[int]]) -> List[int]:
    """小图演示：每个节点一行 bitset（Python int）。"""
    n = len(graph)
    rows = [0] * n
    for u in range(n):
        rows[u] |= 1 << u
        for v in graph[u]:
            rows[u] |= 1 << v

    # Warshall-bitset: if u reaches k, then u also reaches all nodes that k reaches
    for k in range(n):
        mk = 1 << k
        rk = rows[k]
        for u in range(n):
            if rows[u] & mk:
                rows[u] |= rk

    return rows


def reachable_by_tc(rows: List[int], u: int, v: int) -> bool:
    return ((rows[u] >> v) & 1) == 1


if __name__ == "__main__":
    g = [[1, 2], [3], [3, 4], [5], [], []]

    print("k<=2 from 0:", sorted(bfs_k_hop(g, 0, 2)))
    print("reachable 0->5:", reachable_bfs(g, 0, 5))
    print("reachable 4->5:", reachable_bfs(g, 4, 5))

    tc = transitive_closure_small(g)
    print("tc 0->5:", reachable_by_tc(tc, 0, 5))
```

---

## E — Engineering（工程应用）

### 场景 1：风控关系图 k-hop 扩散（Python）

**背景**：从风险种子账户出发，扩散到 `k` 跳账户做实时拦截。  
**为什么适用**：BFS 层级语义与 hop 规则一致，易做预算控制。

```python
from collections import deque

def risk_expand(graph, seeds, k):
    vis = set(seeds)
    q = deque((s, 0) for s in seeds)
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if v not in vis:
                vis.add(v)
                q.append((v, d + 1))
    return vis
```

### 场景 2：服务调用可达性快速判定（Go）

**背景**：排障时判断服务 A 是否经调用链可达服务 B。  
**为什么适用**：reachable 查询命中即停，适合在线诊断接口。

```go
package main

import "fmt"

func Reachable(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	q := []int{s}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		u := q[head]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(Reachable(g, 0, 5))
}
```

### 场景 3：静态依赖图位图索引（C++）

**背景**：构建/编译依赖图更新不频繁，但查询“是否依赖”非常频繁。  
**为什么适用**：位图闭包构建一次后查询 O(1) 位判断。

```cpp
#include <iostream>
#include <vector>

std::vector<unsigned long long> closure6(const std::vector<std::vector<int>>& g) {
    int n = (int)g.size();
    std::vector<unsigned long long> row(n, 0);
    for (int u = 0; u < n; ++u) {
        row[u] |= 1ULL << u;
        for (int v : g[u]) row[u] |= 1ULL << v;
    }
    for (int k = 0; k < n; ++k) {
        unsigned long long mk = 1ULL << k;
        for (int u = 0; u < n; ++u) {
            if (row[u] & mk) row[u] |= row[k];
        }
    }
    return row;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    auto r = closure6(g);
    std::cout << (((r[0] >> 5) & 1ULL) ? "reachable" : "not") << "\n";
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

设查询过程中实际触达子图为 `V'` 节点、`E'` 边：

- 在线 BFS 查询：`O(V' + E')`
- k-hop 查询：最坏仍到 `O(V'+E')`，但通常由 `k` 限制显著缩小
- 全量闭包：
  - 基于 BFS from every node：`O(n*(n+m))`
  - 基于布尔矩阵/位运算优化：仍有较高预计算和存储成本

### 替代方案与取舍

| 方案 | 查询 | 构建 | 更新 | 适用 |
| --- | --- | --- | --- | --- |
| 每次 BFS | 中等 | 无 | 无 | 更新频繁、查询中低频 |
| 全量闭包 | 极快 | 很高 | 很高 | 静态小中图、高查询密度 |
| 2-hop / reach index | 快 | 中高 | 中高 | 查询密集、可容忍离线构建 |
| 轻索引 + BFS 兜底 | 快（平均） | 中等 | 中等 | 大多数线上系统的折中 |

### 常见错误

1. 盲目全算闭包，导致构建和存储不可控
2. 在严格正确场景只用 bloom 直接判可达
3. 无 hop / 预算限制，线上长尾时延失控

### 为什么这套是工程可行解

- BFS + hop 限制是低复杂度、低维护成本基线
- 索引按查询密度渐进引入，不一次性过度设计
- “索引命中 + 搜索兜底”兼顾时延与正确性

---

## 常见问题与注意事项

1. **Reachability 和最短路一样吗？**  
   不一样。Reachability 只关心“是否存在路径”，不关心最短距离。

2. **Transitive Closure 一定不能算吗？**  
   不是。静态图 + 高查询密度时很有价值；只是大多数线上动态图不适合全量维护。

3. **2-hop labeling 一定优于 BFS 吗？**  
   也不是。它对查询友好，但构建/维护更重，适合“读多写少”场景。

---

## 最佳实践与建议

- 先上线可观测的 BFS 基线（含 hop、预算、超时）
- 用真实流量画像决定是否引入 reach index
- 索引设计先追求“可维护”，再追求“理论最优”
- 预留降级路径：索引失效时可回退到 BFS

---

## S — Summary（总结）

### 核心收获

- 可达性查询是“算法 + 系统约束”问题，不是单一最优算法问题
- `k-hop` 查询首选 BFS + hop 限制 + early stop
- Transitive Closure 能快查，但一般不全算，尤其在动态图
- 2-hop labeling / reach index 适合读多写少场景
- 工程最稳解通常是“轻索引 + 在线 BFS 兜底”

### 推荐延伸阅读

- LeetCode 1971（Find if Path Exists in Graph）
- LeetCode 847（Shortest Path Visiting All Nodes，状态搜索扩展）
- 图数据库查询优化文档（Neo4j / JanusGraph 的邻域查询策略）
- Reachability Indexing 经典论文（2-hop labeling / GRAIL）

---

## 元信息

- **阅读时长**：12~16 分钟
- **标签**：Reachability、k-hop、BFS、2-hop labeling
- **SEO 关键词**：Reachability, k-hop, Transitive Closure, 2-hop labeling, reach index
- **元描述**：工程化可达性查询方案：BFS+hop 限制、闭包取舍、位图索引/2-hop labeling 与在线兜底策略。

---

## 行动号召（CTA）

建议你下一步做两件事：

1. 先给现有可达性接口加上 `hop_limit` 与 `visit_budget` 参数
2. 对真实流量做一次“每次 BFS vs 轻量索引+兜底”的 A/B 压测

如果你希望，我可以继续给你写下一篇：
“Reachability 索引落地手册：什么时候选 2-hop、什么时候选 GRAIL、什么时候坚持 BFS”。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import deque


def reachable(graph, s, t):
    vis = [False] * len(graph)
    q = deque([s])
    vis[s] = True
    while q:
        u = q.popleft()
        if u == t:
            return True
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                q.append(v)
    return False


def k_hop(graph, s, k):
    vis = [False] * len(graph)
    q = deque([(s, 0)])
    vis[s] = True
    out = {s}
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                out.add(v)
                q.append((v, d + 1))
    return out
```

```c
#include <stdbool.h>
#include <stdio.h>

#define N 6

bool reachable(int g[N][N], int s, int t) {
    int q[128], head = 0, tail = 0;
    bool vis[N] = {0};
    q[tail++] = s;
    vis[s] = true;
    while (head < tail) {
        int u = q[head++];
        if (u == t) return true;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                q[tail++] = v;
            }
        }
    }
    return false;
}

int main(void) {
    int g[N][N] = {0};
    g[0][1] = g[0][2] = 1;
    g[1][3] = 1;
    g[2][3] = g[2][4] = 1;
    g[3][5] = 1;
    printf("%d\n", reachable(g, 0, 5));
    return 0;
}
```

```cpp
#include <iostream>
#include <queue>
#include <vector>

bool reachable(const std::vector<std::vector<int>>& g, int s, int t) {
    std::vector<char> vis(g.size(), 0);
    std::queue<int> q;
    vis[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (u == t) return true;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                q.push(v);
            }
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    std::cout << reachable(g, 0, 5) << "\n";
}
```

```go
package main

import "fmt"

func reachable(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	q := []int{s}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		u := q[head]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(reachable(g, 0, 5))
}
```

```rust
use std::collections::VecDeque;

fn reachable(graph: &Vec<Vec<usize>>, s: usize, t: usize) -> bool {
    let mut vis = vec![false; graph.len()];
    let mut q = VecDeque::new();
    vis[s] = true;
    q.push_back(s);

    while let Some(u) = q.pop_front() {
        if u == t {
            return true;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                q.push_back(v);
            }
        }
    }
    false
}

fn main() {
    let g = vec![vec![1, 2], vec![3], vec![3, 4], vec![5], vec![], vec![]];
    println!("{}", reachable(&g, 0, 5));
}
```

```javascript
function reachable(graph, s, t) {
  const vis = Array(graph.length).fill(false);
  const q = [s];
  let head = 0;
  vis[s] = true;

  while (head < q.length) {
    const u = q[head++];
    if (u === t) return true;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        q.push(v);
      }
    }
  }
  return false;
}

const g = [[1, 2], [3], [3, 4], [5], [], []];
console.log(reachable(g, 0, 5));
```
