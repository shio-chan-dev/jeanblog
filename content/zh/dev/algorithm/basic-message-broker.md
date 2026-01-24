---
title: "手写一个基础消息代理：发布、订阅与队列"
date: 2026-01-24T13:20:24+08:00
draft: false
description: "用最小实现解释消息代理的核心概念与工作流程。"
tags: ["算法", "分布式", "消息队列", "系统设计"]
categories: ["逻辑与算法"]
keywords: ["消息代理", "发布订阅", "队列", "Broker"]
---

## 副标题 / 摘要

消息代理是分布式系统的关键组件。本文用最小实现展示发布/订阅的基本机制。

## 目标读者

- 学习分布式系统的开发者
- 需要理解消息队列原理的人
- 关注解耦与可靠性的工程师

## 背景 / 动机

直接调用会导致系统耦合，消息代理提供异步解耦与缓冲能力。  
理解基础模型有助于正确使用 Kafka、RabbitMQ 等系统。

## 核心概念

- **Broker**：中间消息代理
- **主题（Topic）**：消息分类通道
- **订阅者（Subscriber）**：消费消息的客户端

## 实践指南 / 步骤

1. **定义主题与订阅者列表**
2. **发布消息时通知订阅者**
3. **支持多订阅者并发消费**
4. **加入简单缓存/重放机制（可选）**

## 可运行示例

```python
class Broker:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, fn):
        self.subscribers.setdefault(topic, []).append(fn)

    def publish(self, topic, msg):
        for fn in self.subscribers.get(topic, []):
            fn(msg)


if __name__ == "__main__":
    broker = Broker()
    broker.subscribe("order", lambda m: print("A got", m))
    broker.subscribe("order", lambda m: print("B got", m))

    broker.publish("order", {"id": 1})
```

## 解释与原理

消息代理把“生产者”与“消费者”解耦，通过主题进行路由。  
核心优势是异步化、缓冲与扩展性。

## 常见问题与注意事项

1. **这个示例可靠吗？**  
   不可靠，仅用于理解模型。

2. **如何保证消息不丢？**  
   需要持久化、ACK 和重试机制。

3. **如何支持消费进度？**  
   需要 offset 记录与消费组管理。

## 最佳实践与建议

- 关键业务消息要持久化
- 引入幂等与去重策略
- 对堆积消息做监控告警

## 小结 / 结论

基础消息代理的核心是“主题 + 发布订阅”。  
理解它能帮助你更有效使用成熟消息系统。

## 参考与延伸阅读

- Kafka 核心概念
- RabbitMQ AMQP 指南

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：消息代理、发布订阅  
- **SEO 关键词**：消息代理, Broker  
- **元描述**：解释消息代理的核心机制。

## 行动号召（CTA）

用这个最小模型模拟一次“订单创建 → 库存扣减”的消息流。
