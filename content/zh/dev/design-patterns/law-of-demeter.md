---
title: "迪米特法则（最少知识原则）：违例与修复示例"
date: 2026-01-24T12:40:32+08:00
draft: false
description: "解释迪米特法则的核心思想，并给出违反与修复示例。"
tags: ["设计模式", "代码设计", "耦合", "可维护性"]
categories: ["设计模式"]
keywords: ["Law of Demeter", "最少知识原则", "耦合"]
---

## 副标题 / 摘要

迪米特法则强调“只和直接朋友说话”。本文用示例说明违规写法，并给出修复方式。

## 目标读者

- 想降低耦合的工程师
- 负责代码评审与重构的开发者
- 需要维护大型系统的团队

## 背景 / 动机

深层链式调用让对象之间依赖过强，改动一个结构就影响一大片。  
迪米特法则就是用来控制这种耦合的。

## 核心概念

- **最少知识原则**：对象只了解直接依赖
- **消息委托**：把内部结构封装在对象内
- **耦合控制**：减少“链式访问”

## 实践指南 / 步骤

1. **识别链式调用**（a.b.c.d）  
2. **让中间对象提供必要方法**  
3. **封装内部结构**  
4. **避免跨层访问内部字段**

## 可运行示例

```python
class Wallet:
    def __init__(self, balance):
        self.balance = balance

    def has_enough(self, amount):
        return self.balance >= amount


class User:
    def __init__(self, wallet):
        self.wallet = wallet

    def can_pay(self, amount):
        return self.wallet.has_enough(amount)


def checkout(user, amount):
    # 违例：user.wallet.balance
    # 修复：user.can_pay
    return user.can_pay(amount)
```

## 解释与原理

通过让 User 暴露 `can_pay` 方法，调用方无需知道 wallet 的内部结构。  
这样 wallet 内部变化时，调用方不需要改动。

## 常见问题与注意事项

1. **链式调用一定不好吗？**  
   在简单场景可以，但深层调用会导致耦合脆弱。

2. **法则会导致方法太多吗？**  
   会增加一些包装方法，但换来稳定性。

3. **如何评估是否需要？**  
   看链路深度与变更频率。

## 最佳实践与建议

- 关注 `a.b.c` 这种链式调用
- 用“委托方法”减少依赖
- 保持对象边界清晰

## 小结 / 结论

迪米特法则的价值是控制耦合与变化传播。  
在复杂系统中，遵守它能显著提升可维护性。

## 参考与延伸阅读

- *Design Patterns*（GoF）
- *The Pragmatic Programmer*

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：迪米特法则、耦合、设计模式  
- **SEO 关键词**：Law of Demeter, 最少知识原则  
- **元描述**：解释迪米特法则并给出修复示例。

## 行动号召（CTA）

在一次代码评审中，专门检查链式调用并提出改进建议。
