---
title: "BFS / DFS 工程入门：k-hop 查询、子图抽取与路径可达性 ACERS 解析"
date: 2026-02-09T09:44:11+08:00
draft: false
categories: ["逻辑与算法"]
tags: ["图", "BFS", "DFS", "k-hop", "子图抽取", "路径可达性"]
description: "围绕 k-hop 查询、子图抽取、路径可达性三类高频图任务，系统讲清 BFS/DFS 的迭代模板、early stop 剪枝与 visited bitmap/bloom 选型，并附多语言可运行实现。"
keywords: ["BFS", "DFS", "k-hop", "subgraph extraction", "path existence", "visited bitmap", "bloom filter"]
---

> **副标题 / 摘要**  
> BFS / DFS 不是“会写就行”，而是要到工程可用、可控成本、可证明正确。本文按 ACERS 模板，把最常用的三类任务（k-hop 查询、子图抽取、路径可达性）拆成可复用模板：**迭代实现 + early stop + visited 结构选型**。

- **预计阅读时长**：12~16 分钟  
- **标签**：`图`、`BFS`、`DFS`、`k-hop`、`子图抽取`  
- **SEO 关键词**：BFS, DFS, k-hop 查询, 子图抽取, 路径可达性, visited bitmap, bloom filter  
- **元描述**：面向工程场景讲解 BFS/DFS：迭代版避免栈溢出、early stop 降低搜索成本、visited bitmap/bloom 优化内存与判重性能。

---

## 目标读者

- 正在做图数据库、风控关系图、调用链分析的工程师
- 只会“题解式 BFS/DFS”，但还没形成工程模板的同学
- 希望把图遍历写成“稳定、可观测、可扩展”代码的人

## 背景 / 动机

在工程里，BFS/DFS 通常不是一次性离线脚本，而是在线请求的一部分：

- `k-hop` 邻域查询要控制时延
- 子图抽取要控制内存与输出规模
- 路径可达性要快速返回 true/false

如果只停留在教科书递归模板，会很快踩坑：

1. 深图导致递归栈溢出
2. 无剪枝导致无谓扩展
3. visited 结构选错，内存和吞吐同时恶化

所以这篇文章聚焦一个目标：
把 BFS / DFS 升级到“**滚瓜烂熟且能上线**”的程度。

## 核心概念

| 概念 | 作用 | 工程关注点 |
| --- | --- | --- |
| BFS（队列） | 按层扩展、天然支持 hop 层级 | 适合 k-hop、最短边数、层级子图 |
| DFS（栈） | 深入探索、路径存在性高效 | 适合快速可达性判断与深度剪枝 |
| early stop | 提前终止搜索 | 控制 P99 延迟和资源消耗 |
| visited bitmap | 精确判重，内存紧凑 | 需先做节点 ID 压缩 |
| bloom filter | 概率判重/预过滤 | 有假阳性，不能单独用于“严格正确性”场景 |

---

## A — Algorithm（题目与算法）

### 题目还原（LeetCode 风格训练题）

给定一个无权图 `G`（邻接表），起点 `s`，最大跳数 `K`，可选目标点 `t`：

1. 返回从 `s` 出发 `K` 跳内可达节点集合（k-hop 查询）
2. 返回由访问到的节点与边组成的子图（子图抽取）
3. 判断 `s -> t` 是否存在路径（路径可达性）

要求：

- 使用**迭代版** BFS / DFS（不使用递归）
- 支持 early stop（如超过 K 跳、命中目标、命中业务谓词）
- 维护 visited，避免重复扩展

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| graph | List[List[int]] | 邻接表，节点 ID 为 0..n-1 |
| s | int | 起点 |
| K | int | 最大 hop（用于 BFS） |
| t | int | 目标点（用于可达性） |
| 返回1 | Set[int] | K 跳内节点集合 |
| 返回2 | List[Tuple[int,int]] | 抽取的边集合（可选） |
| 返回3 | bool | 是否可达 |

### 示例 1：k-hop 查询

```text
graph = [
  [1,2],    # 0
  [3],      # 1
  [3,4],    # 2
  [5],      # 3
  [],       # 4
  []        # 5
]
s = 0, K = 2

输出节点: {0,1,2,3,4}
```

解释：2 跳内可达 `0(0跳), 1/2(1跳), 3/4(2跳)`，节点 5 需要 3 跳。

### 示例 2：路径可达性

```text
graph 同上
s = 0, t = 5

输出: true
```

---

## 思路推导（从朴素到工程模板）

### 朴素写法：递归 DFS / 无剪枝 BFS

- 递归 DFS 在深图上会触发栈深问题
- BFS 若不限制 hop，可能把整图扫完
- 不做 visited 会指数级重复扩展

### 关键观察

1. 你真正要的通常不是“全图遍历”，而是“满足业务条件的最小遍历”
2. 搜索顺序可以模板化（队列/BFS，栈/DFS），剪枝条件要业务化
3. visited 不是一个固定实现，必须按图规模和正确性要求选

### 方法选择

- `k-hop`：优先 BFS（天然按层）
- 路径存在性：优先迭代 DFS（栈 + 早停）
- 大规模图：ID 压缩 + bitmap；高吞吐弱一致去重可加 bloom 预过滤

---

## C — Concepts（核心思想）

### 方法归类

- 图遍历（Graph Traversal）
- 层序搜索（BFS）
- 深度搜索（DFS）
- 剪枝搜索（Pruned Search）

### 工程不变量

1. `visited[u] = true` 表示节点 `u` 已入队/入栈（或已消费，取决于策略）
2. BFS 中 `(node, depth)` 的 `depth` 不超过 `K`
3. early stop 条件触发后，结果保持业务定义的正确性

### Early Stop 设计模板

- **hop 限制**：`depth == K` 时不再扩展邻居
- **目标命中**：`node == t` 时直接返回
- **预算控制**：访问节点数超阈值就停止并返回部分结果
- **谓词剪枝**：节点属性不满足业务条件时跳过扩展

### visited 结构选型

| 结构 | 正确性 | 内存 | 速度 | 适用场景 |
| --- | --- | --- | --- | --- |
| HashSet | 精确 | 中等偏高 | 快 | 节点 ID 稀疏、动态 ID |
| Bitmap | 精确 | 最省（按位） | 快 | 节点 ID 可压缩为连续整数 |
| Bloom Filter | 近似（有假阳性） | 极省 | 快 | 预过滤、去重加速（容忍误差） |

关键结论：

- **严格正确性任务**（如权限判定、风控命中）不能只用 bloom
- bloom 最稳妥用法是“预过滤 + 精确结构二次确认”

---

## 实践指南 / 步骤

1. 做节点 ID 规范化（必要时压缩到 `0..n-1`）
2. `k-hop` 用 BFS，队列元素带 `depth`
3. 路径可达性用迭代 DFS，栈保存待扩展节点
4. 在循环内部第一时间做 early stop
5. visited 优先 bitmap（可压缩时），否则 HashSet
6. 如果吞吐瓶颈在判重，考虑 bloom 预过滤

Python 可运行示例（`python3 bfs_dfs_demo.py`）：

```python
from collections import deque
from typing import List, Set


class SimpleBloom:
    """演示用 Bloom：只做预过滤，不单独作为正确性依据。"""

    def __init__(self, m: int = 1 << 15):
        self.m = m
        self.bits = bytearray(m // 8 + 1)

    def _idx(self, x: int, salt: int) -> int:
        return hash((x, salt)) & (self.m - 1)

    def _set(self, i: int) -> None:
        self.bits[i >> 3] |= 1 << (i & 7)

    def _get(self, i: int) -> bool:
        return (self.bits[i >> 3] >> (i & 7)) & 1 == 1

    def add(self, x: int) -> None:
        for salt in (17, 31, 73):
            self._set(self._idx(x, salt))

    def maybe_contains(self, x: int) -> bool:
        return all(self._get(self._idx(x, salt)) for salt in (17, 31, 73))


def bfs_k_hop(graph: List[List[int]], s: int, k: int) -> Set[int]:
    n = len(graph)
    visited = bytearray(n)  # bitmap
    q = deque([(s, 0)])
    visited[s] = 1
    result = {s}

    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if not visited[v]:
                visited[v] = 1
                result.add(v)
                q.append((v, d + 1))

    return result


def dfs_path_exists(graph: List[List[int]], s: int, t: int) -> bool:
    n = len(graph)
    visited = bytearray(n)
    stack = [s]
    visited[s] = 1

    while stack:
        u = stack.pop()
        if u == t:  # early stop
            return True
        for v in graph[u]:
            if not visited[v]:
                visited[v] = 1
                stack.append(v)

    return False


def bfs_with_bloom_prefilter(graph: List[List[int]], s: int, limit: int = 100000) -> int:
    """示例：bloom 仅用于减少 set 查询次数，最终仍靠 set 保证正确性。"""
    q = deque([s])
    exact = {s}
    bloom = SimpleBloom()
    bloom.add(s)
    visited_count = 0

    while q and visited_count < limit:
        u = q.popleft()
        visited_count += 1
        for v in graph[u]:
            # bloom says "not seen" => 一定没见过，可直接入队
            if not bloom.maybe_contains(v):
                bloom.add(v)
                exact.add(v)
                q.append(v)
                continue
            # bloom says "maybe seen" => 用精确集合确认
            if v not in exact:
                exact.add(v)
                q.append(v)

    return visited_count


if __name__ == "__main__":
    graph = [
        [1, 2],  # 0
        [3],     # 1
        [3, 4],  # 2
        [5],     # 3
        [],      # 4
        [],      # 5
    ]

    print("k-hop<=2:", sorted(bfs_k_hop(graph, 0, 2)))
    print("path 0->5:", dfs_path_exists(graph, 0, 5))
    print("bloom+exact visits:", bfs_with_bloom_prefilter(graph, 0, limit=100))
```

---

## E — Engineering（工程应用）

### 场景 1：图数据库 k-hop 邻域查询（Python）

**背景**：用户输入种子点，系统要在 N 跳内返回邻域节点。  
**为什么适用**：BFS 天然按层，`depth` 控制直接对应 `k-hop` 业务语义。

```python
from collections import deque

def k_hop_nodes(graph, s, k):
    q = deque([(s, 0)])
    vis = {s}
    out = {s}
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in graph[u]:
            if v not in vis:
                vis.add(v)
                out.add(v)
                q.append((v, d + 1))
    return out
```

### 场景 2：调用链故障回溯（Go）

**背景**：判断服务 A 是否可能在调用图中触达故障服务 B。  
**为什么适用**：迭代 DFS + 目标命中 early stop，通常比全图扫描更快返回。

```go
package main

import "fmt"

func pathExists(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	stack := []int{s}
	vis[s] = true
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				stack = append(stack, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(pathExists(g, 0, 5)) // true
}
```

### 场景 3：关系图在线判重预过滤（C++）

**背景**：高 QPS 场景下，visited 集合查询成为热点。  
**为什么适用**：先用 Bloom 做“可能未见”快速分流，再用精确位图/集合确认，降低平均判重成本。

```cpp
#include <bitset>
#include <iostream>
#include <unordered_set>

struct Bloom {
    static const int M = 1 << 16;
    std::bitset<M> bits;
    int h1(int x) const { return (x * 1315423911u) & (M - 1); }
    int h2(int x) const { return (x * 2654435761u) & (M - 1); }
    void add(int x) { bits.set(h1(x)); bits.set(h2(x)); }
    bool maybe(int x) const { return bits.test(h1(x)) && bits.test(h2(x)); }
};

int main() {
    Bloom b;
    std::unordered_set<int> exact;
    for (int x : {1, 2, 3}) { b.add(x); exact.insert(x); }
    int q = 4;
    if (!b.maybe(q) || exact.find(q) != exact.end()) {
        std::cout << "not visited yet\n";
    }
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

设访问到的子图节点数为 `V'`、边数为 `E'`：

- BFS / DFS 时间复杂度：`O(V' + E')`
- visited 额外空间：
  - HashSet：`O(V')`
  - Bitmap：`O(N)` 位（N 为全图节点上界）
  - Bloom：`O(m)` 位（m 为位数组大小，近似配置）

对 `k-hop` 任务，`V'` 与 `E'` 往往远小于全图，这也是 early stop 的核心收益来源。

### 替代方案与取舍

| 方案 | 优点 | 缺点 | 适用 |
| --- | --- | --- | --- |
| 递归 DFS | 代码短 | 深图栈风险、可控性弱 | 小图离线脚本 |
| 迭代 DFS | 可控、易加 early stop | 需手动维护栈 | 路径存在性/在线判断 |
| BFS | 层次清晰、适合 hop | 内存峰值可能高于 DFS | k-hop / 层级检索 |
| 双向 BFS | 路径查询更快 | 实现复杂度更高 | 稀疏图单点到单点 |

### 常见错误思路

1. **visited 在出队时才标记**：可能导致重复入队，队列膨胀
2. **Bloom 单独当 visited**：假阳性会漏掉本应访问的节点
3. **无预算上限**：线上请求在高出度节点可能触发长尾

### 为什么这是最工程可行方案

- 迭代版规避了递归风险
- early stop 把搜索成本约束在业务边界内
- bitmap / bloom 让 visited 策略可按规模弹性调整

---

## 常见问题与注意事项

1. **BFS 与 DFS 谁更快？**  
   不存在绝对结论。`k-hop` 常用 BFS；目标可达性且目标可能在深层时，DFS 常更快命中。

2. **Bloom 误判会不会影响正确性？**  
   会。如果你用 Bloom 单独判重，假阳性会漏搜索。严格正确场景必须配精确结构二次确认。

3. **visited 应该什么时候置位？**  
   通常在“入队/入栈”时置位，避免同一节点被重复加入容器。

---

## 最佳实践与建议

- 先定义业务停止条件，再写遍历代码
- 默认用迭代版，递归只用于小规模离线工具
- 节点 ID 可压缩时优先 bitmap，兼顾速度与内存
- Bloom 只当预过滤器，不单独承诺正确性
- 给遍历加上访问上限与耗时监控，避免线上雪崩

---

## S — Summary（总结）

### 核心收获

- BFS/DFS 的工程版本核心是：迭代容器、清晰不变量、明确 early stop
- `k-hop` 查询与子图抽取优先 BFS；路径可达性判断优先迭代 DFS
- visited 结构不是固定答案：HashSet、bitmap、bloom 各有边界
- Bloom 有假阳性，适合“预过滤 + 精确确认”的组合，不适合单独强一致判定
- 把搜索预算（hop、节点数、耗时）做成显式参数，才能稳定上线

### 推荐延伸阅读

- LeetCode 200（岛屿数量）：图遍历模板
- LeetCode 127（单词接龙）：BFS + 剪枝
- Graph500 / 图计算基准：大规模图遍历性能思路
- 布隆过滤器经典论文与工程参数估算（误判率与位数组大小）

---

## 元信息

- **阅读时长**：12~16 分钟
- **标签**：图、BFS、DFS、k-hop、子图抽取
- **SEO 关键词**：BFS, DFS, k-hop 查询, 路径可达性, visited bitmap, bloom filter
- **元描述**：面向工程场景的 BFS/DFS 模板：迭代实现、early stop、visited bitmap/bloom 选型与多语言可运行代码。

---

## 行动号召（CTA）

建议你立刻做两步固化：

1. 把你线上一个图查询接口改成“显式 early stop 参数化”（hop、节点预算、时间预算）
2. 用真实数据压测 HashSet vs bitmap（必要时加 bloom 预过滤）并记录吞吐与内存曲线

如果你愿意，我可以再给你下一篇：
“并查集 + BFS/DFS 的图问题选型清单（什么时候该遍历，什么时候该合并）”。

---

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from collections import deque


def bfs_k_hop(graph, s, k):
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


def dfs_path_exists(graph, s, t):
    vis = [False] * len(graph)
    st = [s]
    vis[s] = True
    while st:
        u = st.pop()
        if u == t:
            return True
        for v in graph[u]:
            if not vis[v]:
                vis[v] = True
                st.append(v)
    return False


if __name__ == "__main__":
    g = [[1, 2], [3], [3, 4], [5], [], []]
    print(sorted(bfs_k_hop(g, 0, 2)))
    print(dfs_path_exists(g, 0, 5))
```

```c
#include <stdio.h>
#include <stdbool.h>

#define N 6

void bfs_k_hop(int g[N][N], int s, int k) {
    int q[128][2], head = 0, tail = 0;
    bool vis[N] = {0};
    vis[s] = true;
    q[tail][0] = s; q[tail][1] = 0; tail++;

    while (head < tail) {
        int u = q[head][0], d = q[head][1];
        head++;
        if (d == k) continue;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                q[tail][0] = v; q[tail][1] = d + 1; tail++;
            }
        }
    }

    for (int i = 0; i < N; ++i) if (vis[i]) printf("%d ", i);
    printf("\n");
}

bool dfs_path_exists(int g[N][N], int s, int t) {
    int st[128], top = 0;
    bool vis[N] = {0};
    st[top++] = s;
    vis[s] = true;

    while (top) {
        int u = st[--top];
        if (u == t) return true;
        for (int v = 0; v < N; ++v) {
            if (g[u][v] && !vis[v]) {
                vis[v] = true;
                st[top++] = v;
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

    bfs_k_hop(g, 0, 2);          // 0 1 2 3 4
    printf("%d\n", dfs_path_exists(g, 0, 5)); // 1
    return 0;
}
```

```cpp
#include <iostream>
#include <queue>
#include <vector>

std::vector<int> bfsKHop(const std::vector<std::vector<int>>& g, int s, int k) {
    std::vector<char> vis(g.size(), 0);
    std::queue<std::pair<int, int>> q;
    vis[s] = 1;
    q.push({s, 0});

    while (!q.empty()) {
        auto [u, d] = q.front();
        q.pop();
        if (d == k) continue;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                q.push({v, d + 1});
            }
        }
    }

    std::vector<int> out;
    for (int i = 0; i < (int)g.size(); ++i) if (vis[i]) out.push_back(i);
    return out;
}

bool dfsPathExists(const std::vector<std::vector<int>>& g, int s, int t) {
    std::vector<char> vis(g.size(), 0);
    std::vector<int> st = {s};
    vis[s] = 1;
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        if (u == t) return true;
        for (int v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                st.push_back(v);
            }
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> g = {{1,2},{3},{3,4},{5},{},{}};
    auto nodes = bfsKHop(g, 0, 2);
    for (int x : nodes) std::cout << x << " ";
    std::cout << "\n" << dfsPathExists(g, 0, 5) << "\n";
}
```

```go
package main

import "fmt"

func bfsKHop(graph [][]int, s, k int) []bool {
	vis := make([]bool, len(graph))
	type Node struct{ u, d int }
	q := []Node{{s, 0}}
	vis[s] = true
	for head := 0; head < len(q); head++ {
		cur := q[head]
		if cur.d == k {
			continue
		}
		for _, v := range graph[cur.u] {
			if !vis[v] {
				vis[v] = true
				q = append(q, Node{v, cur.d + 1})
			}
		}
	}
	return vis
}

func dfsPathExists(graph [][]int, s, t int) bool {
	vis := make([]bool, len(graph))
	stack := []int{s}
	vis[s] = true
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if u == t {
			return true
		}
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				stack = append(stack, v)
			}
		}
	}
	return false
}

func main() {
	g := [][]int{{1, 2}, {3}, {3, 4}, {5}, {}, {}}
	fmt.Println(bfsKHop(g, 0, 2))
	fmt.Println(dfsPathExists(g, 0, 5))
}
```

```rust
use std::collections::VecDeque;

fn bfs_k_hop(graph: &Vec<Vec<usize>>, s: usize, k: usize) -> Vec<bool> {
    let mut vis = vec![false; graph.len()];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    vis[s] = true;
    q.push_back((s, 0));

    while let Some((u, d)) = q.pop_front() {
        if d == k {
            continue;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                q.push_back((v, d + 1));
            }
        }
    }
    vis
}

fn dfs_path_exists(graph: &Vec<Vec<usize>>, s: usize, t: usize) -> bool {
    let mut vis = vec![false; graph.len()];
    let mut st = vec![s];
    vis[s] = true;

    while let Some(u) = st.pop() {
        if u == t {
            return true;
        }
        for &v in &graph[u] {
            if !vis[v] {
                vis[v] = true;
                st.push(v);
            }
        }
    }
    false
}

fn main() {
    let graph = vec![vec![1, 2], vec![3], vec![3, 4], vec![5], vec![], vec![]];
    println!("{:?}", bfs_k_hop(&graph, 0, 2));
    println!("{}", dfs_path_exists(&graph, 0, 5));
}
```

```javascript
function bfsKHop(graph, s, k) {
  const vis = Array(graph.length).fill(false);
  const q = [[s, 0]];
  let head = 0;
  vis[s] = true;

  while (head < q.length) {
    const [u, d] = q[head++];
    if (d === k) continue;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        q.push([v, d + 1]);
      }
    }
  }
  return vis;
}

function dfsPathExists(graph, s, t) {
  const vis = Array(graph.length).fill(false);
  const st = [s];
  vis[s] = true;

  while (st.length) {
    const u = st.pop();
    if (u === t) return true;
    for (const v of graph[u]) {
      if (!vis[v]) {
        vis[v] = true;
        st.push(v);
      }
    }
  }
  return false;
}

const g = [[1, 2], [3], [3, 4], [5], [], []];
console.log(bfsKHop(g, 0, 2));
console.log(dfsPathExists(g, 0, 5));
```
