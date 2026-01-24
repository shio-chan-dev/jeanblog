---
title: "用队列实现栈：单队列旋转法"
date: 2026-01-24T13:20:24+08:00
draft: false
description: "解释如何用队列模拟栈，并给出可运行实现与复杂度分析。"
tags: ["数据结构", "栈", "队列", "算法"]
categories: ["逻辑与算法"]
keywords: ["用队列实现栈", "单队列", "旋转"]
---

## 副标题 / 摘要

只有队列（FIFO）时，也能实现栈（LIFO）。本文展示单队列旋转法，并分析复杂度与边界。

## 目标读者

- 刷题与面试准备的开发者
- 需要理解数据结构转换的人
- 初中级算法学习者

## 背景 / 动机

栈的后进先出与队列的先进先出相反。  
“旋转队列”可以把最新元素移动到队头，从而模拟栈顶。

## 核心概念

- **队列（FIFO）**：先进先出
- **栈（LIFO）**：后进先出
- **队列旋转**：把新元素转到队首

## 实践指南 / 步骤

1. **入栈**：将元素入队
2. **旋转队列**：把队首依次出队再入队，直到新元素位于队首
3. **出栈**：直接出队
4. **取栈顶**：查看队首

## 可运行示例

```python
from collections import deque


class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q


if __name__ == "__main__":
    s = MyStack()
    s.push(1)
    s.push(2)
    print(s.top())
    print(s.pop())
    print(s.empty())
```

## 解释与原理

每次 push 后旋转队列，使新元素位于队首。  
这样 pop 就相当于弹出“栈顶”。

## 常见问题与注意事项

1. **复杂度如何？**  
   push 为 O(n)，pop 为 O(1)。

2. **能否用两个队列？**  
   可以，但逻辑更复杂，不一定更快。

3. **适合高频 push 的场景吗？**  
   不适合，push 成本较高。

## 最佳实践与建议

- 如果 push 频繁，考虑双队列优化
- 对空栈操作做好异常处理
- 明确时间复杂度在文档中说明

## 小结 / 结论

单队列旋转法简单直观，但 push 成本较高。  
它是理解 FIFO/LIFO 互换的经典练习。

## 参考与延伸阅读

- LeetCode 225
- 数据结构教材章节

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：栈、队列、旋转  
- **SEO 关键词**：用队列实现栈, 单队列  
- **元描述**：解释用队列实现栈的单队列方法。

## 行动号召（CTA）

对比双队列与单队列的实现，写一份复杂度对比表。
