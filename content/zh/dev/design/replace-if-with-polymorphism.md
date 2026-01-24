---
title: "用多态替换 if：把流程判断变成对象职责"
date: 2026-01-24T15:42:47+08:00
draft: false
description: "通过对象职责拆分，消除重复 if 判断。"
tags: ["设计", "面向对象", "重构", "可维护性"]
categories: ["设计"]
keywords: ["多态", "if 重构", "职责拆分"]
---

## 副标题 / 摘要

大量 if 判断会让代码难以维护。本文通过职责拆分与多态消除条件分支。

## 目标读者

- 需要重构遗留代码的工程师
- 关注可维护性的团队
- 学习设计原则的开发者

## 背景 / 动机

if 分支常常是“规则塞在一起”的信号。  
当规则变化时，分支会持续膨胀。

## 核心概念

- **职责拆分**：让对象承担自己的规则
- **多态**：用对象替代条件判断
- **空对象**：避免 null 判断

## 实践指南 / 步骤

1. **识别 if 判断的业务规则**
2. **为规则创建对象或策略**
3. **用空对象替代 null 分支**
4. **把规则拆成可测试单元**

## 可运行示例

```python
class Foo:
    def do(self, file):
        return f"process {file}"


class NullFoo(Foo):
    def do(self, file):
        return ""


def get_foo(repo, key):
    return repo.get(key, NullFoo())


if __name__ == "__main__":
    repo = {"a.xml": Foo()}
    foo = get_foo(repo, "a.xml")
    print(foo.do("a.xml"))
```

## 解释与原理

把“有/无对象”的判断交给对象本身（Null Object），  
可减少 if 分支并提升可读性。

## 常见问题与注意事项

1. **是否一定要多态？**  
   小规模逻辑可以保留 if。

2. **空对象会隐藏错误吗？**  
   需要确保业务允许“空行为”。

3. **如何避免过度设计？**  
   在规则增长时再引入多态。

## 最佳实践与建议

- 规则多变时采用多态
- 用空对象减少 null 判断
- 用测试覆盖规则变更

## 小结 / 结论

多态能让规则扩展更清晰，减少 if 分支带来的复杂度。  
在复杂业务逻辑中尤为有效。

## 参考与延伸阅读

- Refactoring: Replace Conditional with Polymorphism
- Null Object Pattern

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：多态、重构  
- **SEO 关键词**：if 重构, 多态  
- **元描述**：通过多态替换 if 的重构思路。

## 行动号召（CTA）

挑一个 if/else 链，尝试用对象职责拆分进行重构。
