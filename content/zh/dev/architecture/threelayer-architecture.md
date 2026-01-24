---
title: "什么是三层架构：职责划分与工程价值"
date: 2026-01-24T12:29:51+08:00
draft: false
description: "解释三层架构的分层职责、优缺点与适用场景。"
tags: ["软件架构", "三层架构", "分层", "可维护性"]
categories: ["软件架构"]
keywords: ["三层架构", "Presentation", "Business", "Data"]
---

## 副标题 / 摘要

三层架构通过分离展示、业务与数据层，降低耦合与维护成本。本文讲清职责边界与工程价值。

## 目标读者

- 负责系统设计与分层的工程师
- 需要规范代码结构的团队
- 想降低耦合与提升维护性的开发者

## 背景 / 动机

业务系统复杂度上升时，代码容易变成“泥球”。  
三层架构提供了一种稳定的分层结构，帮助控制复杂性。

## 核心概念

- **表现层（Presentation）**：UI / API 接口
- **业务层（Business）**：核心业务规则
- **数据层（Data）**：数据库与外部存储

## 实践指南 / 步骤

1. **定义清晰的层边界**  
2. **禁止跨层直连**（表现层不直接访问数据库）  
3. **业务层成为唯一的规则入口**  
4. **数据层只负责持久化**  
5. **用接口隔离依赖**

## 可运行示例

```python
class UserRepository:
    def get(self, user_id):
        return {"id": user_id, "name": "Alice"}


class UserService:
    def __init__(self, repo):
        self.repo = repo

    def profile(self, user_id):
        user = self.repo.get(user_id)
        return {"id": user["id"], "display": user["name"].upper()}


class UserAPI:
    def __init__(self, service):
        self.service = service

    def handle(self, user_id):
        return self.service.profile(user_id)
```

## 解释与原理

三层架构的核心是“职责分离”。  
各层只关心自己的问题，减少依赖与变更扩散。

## 常见问题与注意事项

1. **三层会不会过度设计？**  
   对小项目可能是，但中大型系统很有价值。

2. **业务逻辑放哪一层？**  
   必须在业务层，避免散落到控制器或 DAO。

3. **三层是否影响性能？**  
   一般影响不大，维护性收益更高。

## 最佳实践与建议

- 业务层保持纯粹
- 数据层只负责持久化
- 表现层只做协议转换

## 小结 / 结论

三层架构是最基础的分层模式，能显著提升系统可维护性与可演进性。  
关键在于坚持边界。

## 参考与延伸阅读

- *Clean Architecture*
- *Enterprise Application Architecture Patterns*

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：三层架构、分层、软件设计  
- **SEO 关键词**：三层架构, 分层  
- **元描述**：解释三层架构的分层职责与价值。

## 行动号召（CTA）

画一下你项目的分层图，看看是否存在跨层直连。
