---
title: "Graph Algorithms Learning Path: From BFS to Graph Computation Models"
date: 2026-02-09T10:14:45+08:00
draft: false
categories: ["Logic and Algorithms"]
tags: ["graph algorithms", "learning path", "reading order", "graph databases", "engineering practice"]
description: "A graph algorithms topic guide and recommended reading order covering BFS/DFS, reachability, shortest paths, CC/SCC, centrality, PageRank, community detection, subgraph matching, dynamic graphs, graph partitioning, and graph computation models."
keywords: ["graph algorithms", "reading order", "learning path", "Pregel", "GAS", "PageRank", "BFS", "CC", "SCC"]
---

> This is a "graph algorithms topic navigation" page. The goal is not to stack articles together, but to give you an executable learning path from basic traversal to distributed graph computation.

## Current Directory Status (Topic Structuring Completed)

The graph algorithms series has been migrated to:

- `content/zh/dev/algorithm/graph/`

It also uses two-digit prefixes (`00/10/20...`) to mark reading order, which makes it easier to:

- Browse in sequence within the file system
- Insert new articles incrementally later (while preserving numbering gaps)
- Locate stages quickly during batch maintenance

## Recommended Reading Order (By Capability Building)

### Stage 0: Traversal Fundamentals (Lay the Foundation First)

1. [BFS / DFS Engineering Intro: k-hop Queries, Subgraph Extraction, and Path Reachability](./10-bfs-dfs-k-hop-subgraph-path-existence.md)  
2. [Shortest Path in Practice: Engineering Selection of BFS, Dijkstra, and A*](./20-shortest-path-bfs-dijkstra-astar-acers.md)

Goals:

- Reliably implement iterative graph traversal;
- Explain when to use BFS and when to use Dijkstra/A*;
- Build the habit of adding `early stop`, `visited`, and budget limits.

### Stage 1: Reachability and Connectivity Structure (Core of Graph Querying)

3. [k-hop and Reachability Queries: BFS Constraints, Reachability Indexes, and 2-hop Labeling](./30-k-hop-reachability-and-reach-index.md)  
4. [Connected Components and SCC: Tarjan / Kosaraju](./40-connected-components-and-scc-tarjan-kosaraju.md)

Goals:

- Upgrade "can it reach?" from one-off search to a system capability;
- Understand that undirected connectivity and directed strong connectivity are different problem classes;
- Build a combined mindset of "online BFS + offline index."

### Stage 2: Graph Analytics Metrics (From Reachability to Insight)

5. [Graph Centrality: Degree / Betweenness / Closeness](./50-graph-centrality-degree-betweenness-closeness.md)  
6. [PageRank / Personalized PageRank: Node Importance and Incremental Updates](./60-pagerank-and-personalized-pagerank.md)

Goals:

- Explain different definitions of "importance" and their applicability boundaries;
- Apply centrality and PageRank to recommendation, risk control, and influence analysis;
- Understand that "metric is theoretically correct" and "platform can run it well" are different issues.

### Stage 3: Structure Mining and Matching (Application-Layer Capabilities)

7. [Subgraph Matching: VF2, Ullmann, and Pruning](./70-subgraph-matching-vf2-ullmann-and-pruning.md)  
8. [Community Detection: Louvain and Label Propagation](./80-community-detection-louvain-label-propagation.md)

Goals:

- Perform pattern recognition and rule-graph matching;
- Make engineering tradeoffs between "community quality vs speed";
- Understand cost-curve differences between matching and clustering.

### Stage 4: Large-Scale and Dynamic Scenarios (Platform-Level Capabilities)

9. [Dynamic Graphs and Incremental Computation: Incremental Shortest Path, Incremental PageRank, and Connectivity Maintenance](./90-dynamic-graph-incremental-computation.md)  
10. [Graph Partitioning: Edge-cut, Vertex-cut, and METIS Selection](./100-graph-partitioning-edge-cut-vertex-cut-metis.md)  
11. [Graph Computation Models: Pregel (BSP) and GAS, How to Run PageRank / CC / Parallel BFS](./110-graph-computation-models-pregel-gas-parallel-bfs.md)

Goals:

- Decide when to do full recomputation and when to do incremental updates;
- Co-design algorithms with partitioning/communication/convergence strategies;
- Explain the root causes of "why this graph workload is slow in distributed environments."

## Two Practical Study Rhythms

### Rhythm A (2-Week Sprint, Engineering First)

- Week 1: Stages 0-1 (articles 1-4)  
- Week 2: Stages 2-4 (articles 5-11)

Best for: people who need to connect graph capabilities to business lines quickly.

### Rhythm B (4-Week Steady Path, Principles First)

- Week 1: Traversal and shortest path (1-2)  
- Week 2: Reachability and connectivity (3-4)  
- Week 3: Centrality and PageRank (5-6)  
- Week 4: Matching/community/dynamic graph/partitioning/computation models (7-11)

Best for: people building graph platforms or maintaining graph services long term.

## Recommendations for Using This Series

1. After each article, run at least one runnable code sample from the post.  
2. Bring your own business graph into the same problem frame (input scale, update frequency, SLA).  
3. For every task, write down "stop condition + budget + regression baseline"; this matters more than memorizing one more formula.

## Next Steps (Optional)

If you continue expanding this series, evolve it in this order:

1. First, apply a unified tag across all 11 posts (for example, `graph-algorithms-series`)  
2. Then add second-level aggregation pages (fundamentals/analytics/platform)  
3. For new posts, prioritize `120/130...` numbering to avoid renumbering older files
