---
title: "如何在不可靠协议上构建可靠通信：重传、确认与顺序"
date: 2026-01-24T11:06:00+08:00
draft: false
description: "用确认、超时与重传在不可靠网络上实现可靠通信，并给出简化的可运行示例。"
tags: ["网络基础", "可靠传输", "协议设计", "超时重传"]
categories: ["网络基础"]
keywords: ["可靠传输", "ARQ", "ACK", "重传", "超时"]
---

## 副标题 / 摘要

不可靠协议上构建可靠通信的核心是：确认、超时、重传与顺序控制。本文给出工程要点与简化实现示例。

## 目标读者

- 需要理解可靠传输机制的后端工程师
- 设计自定义协议的开发者
- 对网络底层原理有兴趣的同学

## 背景 / 动机

UDP 等不可靠协议不保证送达、不保证顺序。  
但许多业务需要可靠性：日志上报、订单同步、状态更新等。  
因此需要在应用层补齐可靠性能力。

## 核心概念

- **ACK 确认**：接收方回执
- **超时重传**：超时未确认就重发
- **序列号**：保证顺序与去重
- **窗口机制**：提升吞吐（Stop-and-Wait / Sliding Window）

## 实践指南 / 步骤

1. **每个消息加序列号**  
2. **接收端发送 ACK**  
3. **发送端设置超时重传**  
4. **去重与乱序处理**  
5. **必要时加入滑动窗口**

## 可运行示例

下面用“丢包概率 + 重试”模拟可靠发送：

```python
import random
import time


def unreliable_send(loss_rate: float) -> bool:
    return random.random() > loss_rate


def send_reliable(data: str, loss_rate=0.3, timeout=0.1, max_retry=10):
    for attempt in range(1, max_retry + 1):
        ok = unreliable_send(loss_rate)
        if ok:
            return attempt
        time.sleep(timeout)
    return None


if __name__ == "__main__":
    tries = send_reliable("hello", loss_rate=0.4)
    print("delivered after", tries, "tries")
```

## 解释与原理

可靠传输的本质是“在不可靠通道上建立协议保障”。  
ACK 表示已收到，超时重传保证最终送达，序列号避免重复与乱序。

## 常见问题与注意事项

1. **重传会不会放大拥塞？**  
   会，需要配合退避算法与窗口控制。

2. **只靠重传能保证顺序吗？**  
   不能，还要序列号与乱序缓存。

3. **为什么 TCP 要慢启动？**  
   为了避免重传导致网络拥塞雪崩。

## 最佳实践与建议

- 可靠性机制需要与拥塞控制配合
- 序列号是必需的元数据
- 超时要基于 RTT 动态调整

## 小结 / 结论

可靠通信不是“传输一定成功”，而是“最终一致”。  
通过 ACK、超时、重传与顺序控制，可以在不可靠协议上实现可靠传输。

## 参考与延伸阅读

- ARQ 协议（Stop-and-Wait / Go-Back-N / Selective Repeat）
- TCP 可靠传输机制
- QUIC 设计

## 元信息

- **阅读时长**：8~10 分钟  
- **标签**：可靠传输、协议设计、超时重传  
- **SEO 关键词**：ACK, 重传, ARQ, 可靠通信  
- **元描述**：解释如何在不可靠协议之上构建可靠通信。

## 行动号召（CTA）

试着在一个 UDP 小项目里加上 ACK 和重传，你会更直观理解可靠传输。
