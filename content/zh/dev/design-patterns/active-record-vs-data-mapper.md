---
title: "Active Record vs Data Mapper：差异、优缺点与选型"
date: 2026-01-24T12:40:32+08:00
draft: false
description: "对比 Active Record 与 Data Mapper 两种持久化模式的适用场景与代价。"
tags: ["设计模式", "ORM", "数据建模", "架构"]
categories: ["设计模式"]
keywords: ["Active Record", "Data Mapper", "ORM", "持久化模式"]
---

## 副标题 / 摘要

Active Record 把数据与持久化绑定在一起，Data Mapper 把持久化隔离为独立层。本文对比二者并给出选型建议。

## 目标读者

- 使用 ORM 的后端工程师
- 设计领域模型的开发者
- 需要做架构取舍的团队

## 背景 / 动机

项目变复杂时，持久化模型往往开始“侵入”业务逻辑。  
理解 Active Record 与 Data Mapper 的差异，是避免架构污染的关键。

## 核心概念

- **Active Record**：对象自己保存/加载（数据与持久化耦合）
- **Data Mapper**：持久化逻辑在独立映射层
- **领域模型纯度**：业务模型是否被 ORM 污染

## 实践指南 / 步骤

1. **小型项目可用 Active Record**  
2. **复杂领域建议 Data Mapper**  
3. **明确领域边界，避免 ORM 侵入**  
4. **用 Repository 隔离持久化**  
5. **测试业务逻辑时替换存储层**

## 可运行示例

```python
# Active Record 风格
class UserAR:
    def __init__(self, name):
        self.name = name

    def save(self):
        print("save", self.name)


# Data Mapper 风格
class User:
    def __init__(self, name):
        self.name = name


class UserMapper:
    def save(self, user: User):
        print("save", user.name)


if __name__ == "__main__":
    UserAR("Alice").save()
    UserMapper().save(User("Bob"))
```

## 解释与原理

Active Record 简单直观，但把持久化耦合进领域模型。  
Data Mapper 更复杂，但让业务逻辑更纯粹、更易测试。

## 常见问题与注意事项

1. **Active Record 适合大项目吗？**  
   通常不适合，耦合过深。

2. **Data Mapper 会不会太重？**  
   会增加复杂度，但更利于长期维护。

3. **可以混用吗？**  
   可以，但要有清晰边界。

## 最佳实践与建议

- 早期快速交付可用 Active Record
- 复杂业务优先 Data Mapper + Repository
- 避免 ORM 方法侵入领域模型

## 小结 / 结论

Active Record 简单但耦合高，Data Mapper 复杂但可维护性好。  
选择取决于业务复杂度与演进需求。

## 参考与延伸阅读

- *Patterns of Enterprise Application Architecture*
- Martin Fowler: Active Record / Data Mapper

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：Active Record、Data Mapper、ORM  
- **SEO 关键词**：Active Record, Data Mapper  
- **元描述**：对比两种持久化模式并给出选型建议。

## 行动号召（CTA）

评估一次你的领域模型是否被 ORM 侵入，必要时引入 Repository。
