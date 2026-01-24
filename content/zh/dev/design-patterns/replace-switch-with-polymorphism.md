---
title: "用多态替换 switch：让代码更符合开闭原则"
date: 2026-01-24T15:42:47+08:00
draft: false
description: "通过策略模式消除 switch 语句，降低分支扩展成本。"
tags: ["设计模式", "面向对象", "重构", "开闭原则"]
categories: ["设计模式"]
keywords: ["多态", "switch", "策略模式", "开闭原则"]
---

## 副标题 / 摘要

switch 往往会不断膨胀。本文用策略模式把分支逻辑拆分成可扩展的多态结构。

## 目标读者

- 需要重构条件分支的工程师
- 关注可维护性的团队
- 学习设计模式的开发者

## 背景 / 动机

分支逻辑一旦增长，switch 会变成维护噩梦。  
多态可以把“选择逻辑”变成“可扩展结构”。

## 核心概念

- **策略模式**：把算法封装为对象
- **开闭原则**：对扩展开放，对修改关闭
- **多态分发**：用对象替代条件分支

## 实践指南 / 步骤

1. **识别 switch 的分支类型**
2. **为每个分支定义策略类**
3. **用工厂或映射选择策略**
4. **新增分支只新增类**

## 可运行示例

```python
class Formatter:
    def format(self, text):
        raise NotImplementedError


class FailFormatter(Formatter):
    def format(self, text):
        return "error"


class OkFormatter(Formatter):
    def format(self, text):
        return text + text


def get_formatter(response):
    return {"FAIL": FailFormatter(), "OK": OkFormatter()}.get(response)


if __name__ == "__main__":
    f = get_formatter("OK")
    print(f.format("hi"))
```

## 解释与原理

switch 把逻辑集中在一处，扩展时必须修改旧代码。  
多态把分支拆成独立类，新增规则只需新增类。

## 常见问题与注意事项

1. **小分支是否需要多态？**  
   不一定，只有分支频繁扩展时值得。

2. **工厂会不会复杂？**  
   可以用映射表降低复杂度。

3. **如何保证一致性？**  
   用接口与测试约束策略行为。

## 最佳实践与建议

- 分支频繁变化时用多态
- 用映射表管理策略选择
- 为策略编写单元测试

## 小结 / 结论

多态替换 switch 能显著降低扩展成本。  
当分支持续增长时，这是更稳健的结构选择。

## 参考与延伸阅读

- Head First Design Patterns
- Refactoring: Replace Conditional with Polymorphism

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：多态、重构  
- **SEO 关键词**：switch 重构, 策略模式  
- **元描述**：用多态替换 switch 的重构思路。

## 行动号召（CTA）

找一个增长中的 switch 语句，试着用策略模式重构。
