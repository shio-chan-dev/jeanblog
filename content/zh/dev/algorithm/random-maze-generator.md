---
title: "随机迷宫生成：深度优先回溯法"
date: 2026-01-24T13:20:24+08:00
draft: false
description: "用深度优先回溯生成迷宫，并提供可运行示例。"
tags: ["算法", "图", "随机", "生成"]
categories: ["逻辑与算法"]
keywords: ["迷宫生成", "DFS", "回溯", "随机"]
---

## 副标题 / 摘要

随机迷宫生成常用深度优先回溯法。本文解释思路并提供可运行实现。

## 目标读者

- 学习图算法的开发者
- 想做程序化生成的工程师
- 算法入门学习者

## 背景 / 动机

迷宫生成是图遍历与随机化的经典结合。  
它能帮助理解 DFS、回溯与边界处理。

## 核心概念

- **网格图**：迷宫格点和通道
- **DFS 回溯**：随机探索与回退
- **墙与通路**：用字符表示结构

## 实践指南 / 步骤

1. **初始化全墙网格**
2. **从起点开始 DFS**
3. **随机选择未访问邻居并打通墙**
4. **回溯直到全部访问完成**

## 可运行示例

```python
import random


def maze(w, h):
    grid = [["#"] * (2 * w + 1) for _ in range(2 * h + 1)]
    visited = [[False] * w for _ in range(h)]

    def carve(x, y):
        visited[y][x] = True
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                grid[y * 2 + 1 + dy][x * 2 + 1 + dx] = " "
                grid[y * 2 + 1][x * 2 + 1] = " "
                grid[ny * 2 + 1][nx * 2 + 1] = " "
                carve(nx, ny)

    carve(0, 0)
    return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    print(maze(6, 4))
```

## 解释与原理

DFS 回溯保证每个格子被访问一次，随机方向带来多样性。  
打通墙壁即可形成迷宫通路。

## 常见问题与注意事项

1. **为什么要用 2n+1 网格？**  
   用墙与通路分离更直观。

2. **迷宫会不会有环？**  
   DFS 生成的是“完美迷宫”，通常无环。

3. **如何控制复杂度？**  
   通过宽高控制规模，DFS 是 O(wh)。

## 最佳实践与建议

- 使用固定随机种子便于测试
- 大规模迷宫注意递归深度
- 可扩展为生成多入口迷宫

## 小结 / 结论

深度优先回溯是迷宫生成的经典方法，简单且效果好。  
它是学习图算法的优秀练习题。

## 参考与延伸阅读

- Maze Generation Algorithms
- Graph Traversal Basics

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：迷宫生成、DFS  
- **SEO 关键词**：随机迷宫, DFS 回溯  
- **元描述**：使用 DFS 回溯生成随机迷宫。

## 行动号召（CTA）

给迷宫加入“解题路径”输出，练习一次图搜索。
