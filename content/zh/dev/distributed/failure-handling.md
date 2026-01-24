---
title: "分布式系统如何处理故障：超时、重试与降级"
date: 2026-01-24T13:08:42+08:00
draft: false
description: "总结分布式系统常见故障处理策略，并给出工程落地建议。"
tags: ["分布式系统", "容错", "可靠性", "架构"]
categories: ["分布式系统"]
keywords: ["故障处理", "超时", "重试", "降级"]
---

## 副标题 / 摘要

故障是分布式系统的常态。本文介绍超时、重试、熔断与降级等核心策略。

## 目标读者

- 负责服务稳定性的后端工程师
- 需要设计容错机制的团队
- 关注可靠性的技术负责人

## 背景 / 动机

网络抖动、依赖失败、资源耗尽都会导致故障。  
没有系统化的策略，故障会扩散为雪崩。

## 核心概念

- **超时**：避免无限等待
- **重试**：恢复短暂故障
- **熔断**：快速失败，保护系统
- **降级**：保核心功能

## 实践指南 / 步骤

1. **为所有外部调用设置超时**
2. **重试加入退避与上限**
3. **引入熔断器阻止雪崩**
4. **为关键路径准备降级策略**

## 可运行示例

```python
class CircuitBreaker:
    def __init__(self, threshold=3):
        self.failures = 0
        self.threshold = threshold
        self.open = False

    def call(self, fn):
        if self.open:
            return "fallback"
        try:
            result = fn()
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open = True
            return "fallback"


def unstable():
    raise RuntimeError("fail")


if __name__ == "__main__":
    cb = CircuitBreaker()
    for _ in range(4):
        print(cb.call(unstable))
```

## 解释与原理

超时与重试解决“短暂故障”，熔断防止“持续故障扩散”。  
降级保证系统在失败时仍能提供核心价值。

## 常见问题与注意事项

1. **重试会放大流量吗？**  
   会，所以必须限流与退避。

2. **熔断后如何恢复？**  
   需要半开状态或定期探测。

3. **降级等于停服吗？**  
   不等于，降级是保住核心功能。

## 最佳实践与建议

- 关键依赖设置超时与熔断
- 业务上定义清晰降级策略
- 用可观测性指标验证策略有效性

## 小结 / 结论

分布式故障无法避免，但可以被控制。  
系统化的超时、重试、熔断与降级是必备手段。

## 参考与延伸阅读

- Release It!
- Resilience Patterns

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：容错、熔断、降级  
- **SEO 关键词**：分布式故障处理, 熔断  
- **元描述**：总结分布式系统常见故障处理策略。

## 行动号召（CTA）

为你最关键的依赖服务补上超时与熔断配置，并验证效果。
