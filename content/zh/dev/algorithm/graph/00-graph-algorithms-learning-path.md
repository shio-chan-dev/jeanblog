---
title: "图算法专题学习路径：从 BFS 到图计算模型"
date: 2026-02-09T10:14:45+08:00
draft: false
categories: ["逻辑与算法"]
tags: ["图算法", "学习路径", "阅读顺序", "图数据库", "工程实践"]
description: "图算法专题导航与推荐阅读顺序，覆盖 BFS/DFS、可达性、最短路、CC/SCC、中心性、PageRank、社区发现、子图匹配、动态图、图分区与图计算模型。"
keywords: ["图算法", "阅读顺序", "学习路径", "Pregel", "GAS", "PageRank", "BFS", "CC", "SCC"]
---

> 这是一页“图算法专题导航”。目标不是把文章堆在一起，而是给你一条从基础遍历到分布式图计算的可执行学习路径。

## 目录现状（已完成专题化）

图算法系列已迁移到：

- `content/zh/dev/algorithm/graph/`

并采用两位数字前缀（`00/10/20...`）做阅读顺序标识，方便：

- 文件系统内按顺序浏览
- 后续增量插入新文章（可保留编号间隔）
- 批量维护时快速定位阶段

## 推荐阅读顺序（按能力建设）

### 第 0 阶段：遍历基本功（先打地基）

1. [BFS / DFS 工程入门：k-hop 查询、子图抽取与路径可达性](./10-bfs-dfs-k-hop-subgraph-path-existence.md)  
2. [最短路径实战：BFS、Dijkstra、A* 的工程化选型](./20-shortest-path-bfs-dijkstra-astar-acers.md)

目标：

- 能稳定写出迭代版图遍历；
- 能解释什么时候用 BFS、什么时候用 Dijkstra/A*；
- 习惯加 `early stop`、`visited`、预算限制。

### 第 1 阶段：可达性与连通结构（图查询核心）

3. [k-hop 与可达性查询：BFS 限制、Reachability 索引与 2-hop Labeling](./30-k-hop-reachability-and-reach-index.md)  
4. [Connected Components 与 SCC：Tarjan / Kosaraju](./40-connected-components-and-scc-tarjan-kosaraju.md)

目标：

- 把“能不能到达”从一次搜索升级成系统能力；
- 理解无向连通与有向强连通是两类不同问题；
- 建立“在线 BFS + 离线索引”的组合思维。

### 第 2 阶段：图分析指标（从可达走向洞察）

5. [图中心性：Degree / Betweenness / Closeness](./50-graph-centrality-degree-betweenness-closeness.md)  
6. [PageRank / Personalized PageRank：节点重要性与增量更新](./60-pagerank-and-personalized-pagerank.md)

目标：

- 能解释“重要性”的不同定义与适用边界；
- 能把中心性与 PageRank 用在推荐/风控/影响力分析；
- 理解“指标正确”和“平台能跑”是两回事。

### 第 3 阶段：结构挖掘与匹配（应用层能力）

7. [子图匹配：VF2、Ullmann 与剪枝](./70-subgraph-matching-vf2-ullmann-and-pruning.md)  
8. [社区发现：Louvain 与 Label Propagation](./80-community-detection-louvain-label-propagation.md)

目标：

- 能做模式识别与规则图匹配；
- 能在“社区质量 vs 速度”之间做工程取舍；
- 理解匹配和聚类的成本曲线差异。

### 第 4 阶段：大规模与动态场景（平台级能力）

9. [动态图与增量计算：增量最短路径、增量 PageRank、连通性维护](./90-dynamic-graph-incremental-computation.md)  
10. [图分区：Edge-cut、Vertex-cut 与 METIS 选型](./100-graph-partitioning-edge-cut-vertex-cut-metis.md)  
11. [图计算模型：Pregel（BSP）与 GAS，PageRank / CC / 并行 BFS 怎么跑](./110-graph-computation-models-pregel-gas-parallel-bfs.md)

目标：

- 能判断何时做全量、何时做增量；
- 能把算法与分区/通信/收敛策略一起设计；
- 能回答“为什么这套图任务在分布式上慢”的根因。

## 两条实操学习节奏

### 节奏 A（2 周冲刺，工程优先）

- Week 1：0~1 阶段（1~4 篇）  
- Week 2：2~4 阶段（5~11 篇）

适合：要尽快把图能力接到业务线的人。

### 节奏 B（4 周稳扎稳打，原理优先）

- Week 1：遍历与最短路（1~2）  
- Week 2：可达与连通（3~4）  
- Week 3：中心性与 PageRank（5~6）  
- Week 4：匹配/社区/动态图/分区/计算模型（7~11）

适合：要做图平台或长期维护图服务的人。

## 专题使用建议

1. 每读完一篇，至少跑一次文中的可运行代码。  
2. 把你自己的业务图带入同一问题框架（输入规模、更新频率、SLA）。  
3. 对每个任务都写“停止条件 + 预算 + 回归基线”，这比多背一个公式更重要。

## 下一步（可选）

如果你继续扩写这个专题，建议按以下方式演进：

1. 先给这 11 篇统一打标签（如 `图算法专题`）  
2. 再新增二级聚合页（基础/分析/平台）  
3. 新增文章时优先使用 `120/130...` 编号，避免重排老文件
