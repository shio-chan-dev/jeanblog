---
title: "Saga 与补偿操作：分布式流程的核心区别"
date: 2026-01-24T13:27:25+08:00
draft: false
description: "解释 Saga 与补偿操作的关系，以及在 SOA/微服务中的落地方式。"
tags: ["SOA", "微服务", "Saga", "事务"]
categories: ["SOA与微服务"]
keywords: ["Saga", "补偿操作", "分布式事务"]
---

## 副标题 / 摘要

Saga 是一组本地事务的流程编排，补偿是失败后的回滚手段。本文解释二者关系与工程实践。

## 目标读者

- 设计跨服务流程的工程师
- 需要理解一致性策略的团队
- 架构与技术负责人

## 背景 / 动机

分布式系统不适合强一致长事务。  
Saga 通过补偿机制实现“最终一致”。

## 核心概念

- **Saga**：多个本地事务组成的流程
- **补偿操作**：失败后执行的逆操作
- **编排/协作**：流程驱动方式

## 实践指南 / 步骤

1. **为每个步骤设计补偿动作**
2. **明确补偿是否可逆与可重复**
3. **记录流程状态与执行日志**
4. **处理部分失败与重试**

## 可运行示例

```python
# 订单流程：创建 -> 扣库存 -> 失败补偿

state = []


def step(name):
    state.append(name)


def compensate(name):
    print("compensate:", name)


def run():
    try:
        step("create_order")
        step("reserve_stock")
        raise RuntimeError("fail")
    except Exception:
        while state:
            compensate(state.pop())


if __name__ == "__main__":
    run()
```

## 解释与原理

Saga 描述的是完整流程，而补偿是其中一部分“逆向操作”。  
没有补偿，Saga 就无法在失败时回滚。

## 常见问题与注意事项

1. **补偿一定能完全回滚吗？**  
   不一定，需要业务设计支持。

2. **补偿能重复执行吗？**  
   必须幂等，否则会产生二次错误。

3. **谁来协调 Saga？**  
   可以用编排器或基于事件的协作。

## 最佳实践与建议

- 设计幂等补偿
- 记录状态以便恢复
- 对失败路径做演练

## 小结 / 结论

Saga 是流程，补偿是回退手段。  
理解二者关系是设计分布式一致性的关键。

## 参考与延伸阅读

- Saga Pattern
- Microservices Patterns

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：Saga、补偿事务  
- **SEO 关键词**：Saga, 补偿操作  
- **元描述**：解释 Saga 与补偿操作的区别与联系。

## 行动号召（CTA）

为你的核心流程补齐补偿逻辑，并验证幂等性。
