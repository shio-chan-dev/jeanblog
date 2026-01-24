---
title: "Greenfield vs Brownfield：新项目还是老系统？"
date: 2026-01-24T12:47:45+08:00
draft: false
description: "对比新建项目与遗留系统的利弊，并给出选择与落地策略。"
tags: ["工程实践", "架构", "遗留系统", "决策"]
categories: ["工程实践"]
keywords: ["Greenfield", "Brownfield", "新项目", "遗留系统"]
---

## 副标题 / 摘要

新建项目（Greenfield）与遗留系统（Brownfield）各有成本与风险。本文给出选择依据与工程化落地策略。

## 目标读者

- 需要决定“重写还是演进”的团队
- 负责技术决策的负责人
- 维护老系统的工程师

## 背景 / 动机

“重写 vs 演进”是长期争论的话题。  
选择错误会导致成本爆炸或业务停滞。

## 核心概念

- **Greenfield**：从零开始，无历史负担
- **Brownfield**：已有系统上演进
- **迁移成本**：数据、流程、人员
- **风险控制**：业务连续性优先

## 实践指南 / 步骤

1. **评估现有系统核心价值**  
2. **量化重写成本与风险**  
3. **考虑业务连续性**  
4. **选择渐进式替换或并行系统**  
5. **预留回滚通道**

## 可运行示例

```python
# 选择模型：成本与风险的简化比较

def choose(rewrite_cost, evolve_cost, risk):
    return "rewrite" if rewrite_cost + risk < evolve_cost else "evolve"


if __name__ == "__main__":
    print(choose(100, 60, 50))
```

## 解释与原理

Greenfield 的优势是“自由”，但风险是“未知”。  
Brownfield 的优势是“稳定”，但成本是“历史债务”。

## 常见问题与注意事项

1. **重写一定更快吗？**  
   不一定，往往低估了边界与隐性需求。

2. **演进会不会永远修不完？**  
   如果没有清晰边界与目标，会陷入维护泥潭。

3. **如何降低风险？**  
   用并行系统与灰度切换。

## 最佳实践与建议

- 核心业务优先演进
- 非核心模块可重写试点
- 设定里程碑与回滚点

## 小结 / 结论

Greenfield 和 Brownfield 的选择不是偏好问题，而是风险与成本的权衡。  
理性评估比直觉更重要。

## 参考与延伸阅读

- *Working Effectively with Legacy Code*
- *Monolith to Microservices*

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：Greenfield、Brownfield、工程决策  
- **SEO 关键词**：Greenfield, Brownfield  
- **元描述**：对比新项目与遗留系统的选择策略。

## 行动号召（CTA）

列出系统中最该演进的模块，先从可拆分的部分开始。
