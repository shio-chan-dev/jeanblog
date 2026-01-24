---
title: "手写一个最小的垃圾回收器：标记-清除模型"
date: 2026-01-24T13:20:24+08:00
draft: false
description: "用最小示例解释标记-清除 GC 的核心思路。"
tags: ["算法", "系统", "内存", "GC"]
categories: ["逻辑与算法"]
keywords: ["垃圾回收", "标记清除", "GC"]
---

## 副标题 / 摘要

标记-清除是最基础的垃圾回收模型。本文用简化示例解释“可达性”与回收过程。

## 目标读者

- 想理解 GC 原理的开发者
- 系统编程与语言设计学习者
- 关注内存管理的工程师

## 背景 / 动机

手动内存管理容易出错，而 GC 通过“可达性”自动回收对象。  
理解基础模型有助于调试内存问题。

## 核心概念

- **根集合（Roots）**：直接可达的对象
- **可达性**：从根出发可访问到的对象
- **标记-清除**：标记存活对象，清除不可达对象

## 实践指南 / 步骤

1. **构建对象图与引用关系**
2. **从根集合进行标记遍历**
3. **清除未被标记的对象**
4. **输出回收结果**

## 可运行示例

```python
class Obj:
    def __init__(self, name):
        self.name = name
        self.refs = []
        self.marked = False


def mark(obj):
    if obj.marked:
        return
    obj.marked = True
    for r in obj.refs:
        mark(r)


def sweep(heap):
    return [o for o in heap if o.marked]


if __name__ == "__main__":
    a = Obj("a")
    b = Obj("b")
    c = Obj("c")
    a.refs.append(b)
    heap = [a, b, c]

    roots = [a]
    for r in roots:
        mark(r)

    heap = sweep(heap)
    print([o.name for o in heap])  # c 被回收
```

## 解释与原理

GC 的核心是假设“不可达对象可以回收”。  
标记阶段找到存活对象，清除阶段释放其他对象。

## 常见问题与注意事项

1. **循环引用怎么办？**  
   标记-清除能正确处理循环引用。

2. **为什么会有停顿？**  
   标记阶段可能需要遍历整个对象图。

3. **有没有更高级的 GC？**  
   有，分代、增量、并发等优化。

## 最佳实践与建议

- 理解可达性有助于定位泄漏
- 在生产系统关注 GC 暂停时间
- 对对象生命周期进行监控

## 小结 / 结论

标记-清除是 GC 的基础模型，理解它就能理解更复杂的 GC 设计。  
这是系统与语言设计中的关键概念。

## 参考与延伸阅读

- Garbage Collection Handbook
- JVM GC 文档

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：垃圾回收、内存管理  
- **SEO 关键词**：标记清除, GC  
- **元描述**：用示例解释标记-清除垃圾回收。

## 行动号召（CTA）

画出你项目中的对象生命周期图，看看哪些对象容易被遗忘。
