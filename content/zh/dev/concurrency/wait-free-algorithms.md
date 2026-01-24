---
title: "什么是 Wait-Free 算法：并发中的最高进度保证"
date: 2026-01-24T13:08:42+08:00
draft: false
description: "解释 wait-free 的含义，与 lock-free/obstruction-free 的区别，并给出实践理解。"
tags: ["并发", "无锁", "算法", "性能"]
categories: ["并发"]
keywords: ["Wait-Free", "Lock-Free", "并发算法", "进度保证"]
---

## 副标题 / 摘要

Wait-free 表示“每个线程都能在有限步内完成操作”。本文对比 wait-free 与 lock-free，并解释适用场景。

## 目标读者

- 关注并发正确性与性能的工程师
- 学习无锁算法的开发者
- 需要理解进度保证的架构师

## 背景 / 动机

在高并发系统里，阻塞可能导致长尾延迟。  
Wait-free 提供最强的进度保证，但实现成本也最高。

## 核心概念

- **Wait-free**：每个线程都有完成上界
- **Lock-free**：整体有进展，但可能单线程饥饿
- **Obstruction-free**：无干扰时可完成

## 实践指南 / 步骤

1. **先评估是否需要最强保证**
2. **优先使用成熟的无锁数据结构**
3. **对关键路径进行延迟测量**
4. **用限制争用的设计降低复杂度**

## 可运行示例

```python
# “每线程独立槽位”示例：写入无需等待其他线程
from concurrent.futures import ThreadPoolExecutor


def write_slot(slots, idx, value):
    slots[idx] = value


if __name__ == "__main__":
    slots = [None] * 4
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i in range(4):
            ex.submit(write_slot, slots, i, i * 10)
    print(slots)
```

## 解释与原理

Wait-free 的关键是“每个线程都不依赖别人完成”。  
上例中每个线程写自己的槽位，不会等待锁或其他线程。

## 常见问题与注意事项

1. **Wait-free 就一定更快吗？**  
   不一定，实现复杂度和常数成本更高。

2. **什么时候需要 wait-free？**  
   低延迟和强实时约束场景。

3. **能否直接改造现有锁？**  
   通常很难，需要重新设计数据结构。

## 最佳实践与建议

- 先用 lock-free 评估收益
- 使用成熟库避免自研错误
- 对性能收益做 A/B 测试

## 小结 / 结论

Wait-free 是最高进度保证，但成本高、实现复杂。  
在低延迟系统中值得考虑，普通系统优先 lock-free。

## 参考与延伸阅读

- The Art of Multiprocessor Programming
- Herlihy & Shavit 无锁算法

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：无锁算法、进度保证  
- **SEO 关键词**：Wait-Free, Lock-Free  
- **元描述**：解释 wait-free 与 lock-free 的区别与取舍。

## 行动号召（CTA）

选一个关键并发模块，评估它是否需要 wait-free 的进度保证。
