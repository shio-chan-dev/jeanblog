---
title: "为什么组合优于继承：灵活性、可测试性与演进成本"
date: 2026-01-24T12:40:32+08:00
draft: false
description: "解释“组合优于继承”的工程原因，并给出可运行示例与落地步骤。"
tags: ["设计模式", "面向对象", "组合", "继承", "可维护性"]
categories: ["设计模式"]
keywords: ["Composition", "Inheritance", "组合", "继承"]
---

## 副标题 / 摘要

继承容易让系统变脆，组合让系统更灵活。本文解释为什么组合更适合工程演进，并给出实用示例。

## 目标读者

- 写面向对象代码的工程师
- 负责模块演进与重构的开发者
- 做代码评审与架构设计的团队

## 背景 / 动机

继承会把父类的实现细节暴露给子类，容易导致“脆弱基类问题”。  
组合通过“把能力作为对象注入”来降低耦合，更易测试与替换。

## 核心概念

- **继承（Inheritance）**：is-a 关系，强耦合
- **组合（Composition）**：has-a 关系，弱耦合
- **脆弱基类问题**：父类改动导致子类行为改变

## 实践指南 / 步骤

1. **优先建接口，延后继承**  
2. **把可变行为抽成组件**  
3. **用组合注入行为**  
4. **通过依赖替换实现测试**  
5. **只在“稳定共性”时使用继承**

## 可运行示例

```python
class Logger:
    def log(self, msg: str) -> None:
        print(msg)


class FileSaver:
    def save(self, data: str) -> None:
        print("save", data)


class ReportService:
    def __init__(self, logger: Logger, saver: FileSaver):
        self.logger = logger
        self.saver = saver

    def run(self, data: str) -> None:
        self.logger.log("start")
        self.saver.save(data)
        self.logger.log("done")


if __name__ == "__main__":
    svc = ReportService(Logger(), FileSaver())
    svc.run("report")
```

## 解释与原理

组合让行为可替换（如 Logger 可以换成 Mock）。  
继承则把依赖固定在父类上，一旦父类变化，子类难以控制影响。

## 常见问题与注意事项

1. **继承是不是完全不好？**  
   不是，稳定领域模型可以使用继承。

2. **组合会不会更啰嗦？**  
   会多一些对象，但换来可测试性与灵活性。

3. **什么时候适合继承？**  
   当“is-a”关系稳定且不会频繁变更。

## 最佳实践与建议

- 可变行为优先组合
- 继承只用于稳定抽象
- 把依赖注入作为默认模式

## 小结 / 结论

组合减少耦合、提升可测试性，是更适合工程演进的方式。  
继承应谨慎使用，只在稳定抽象场景下采用。

## 参考与延伸阅读

- *Design Patterns*（GoF）
- *Refactoring*（Martin Fowler）

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：组合、继承、设计模式  
- **SEO 关键词**：Composition, Inheritance  
- **元描述**：解释为何组合优于继承，并给出工程实践。

## 行动号召（CTA）

找一个继承层级复杂的模块，尝试用组合重构，你会看到可维护性提升。
