---
title: "为什么需要并发：吞吐、延迟与资源利用率"
date: 2026-01-24T11:11:28+08:00
draft: false
description: "从 I/O 等待、吞吐与响应时间出发解释并发的必要性，并给出可运行示例与落地建议。"
tags: ["并发", "性能", "吞吐", "延迟", "系统设计"]
categories: ["并发"]
keywords: ["Concurrency", "并发", "吞吐", "延迟", "I/O"]
---

## 副标题 / 摘要

并发不是为了“更快”，而是为了更好地利用等待时间。本文解释并发的价值，并给出工程实践的判断与示例。

## 目标读者

- 希望提升系统吞吐的后端工程师
- 经常处理 I/O 等待的开发者
- 需要做性能与架构决策的技术负责人

## 背景 / 动机

很多程序在等待 I/O（磁盘、网络、数据库）时并不占用 CPU。  
并发让 CPU 在等待期间去做别的事，从而提高吞吐、降低整体延迟。

## 核心概念

- **吞吐（Throughput）**：单位时间内处理的请求数量
- **延迟（Latency）**：单个请求完成的时间
- **I/O 等待**：CPU 空闲但任务阻塞
- **并行与并发**：并行是同时执行，并发是交错执行

## 实践指南 / 步骤

1. **判断瓶颈是否来自 I/O**  
2. **优先使用异步或多线程处理 I/O**  
3. **对 CPU 计算使用并行**  
4. **限制并发度**，避免过度上下文切换  
5. **监控吞吐与尾延迟**

## 可运行示例

下面对比串行与并发请求：

```python
import time
import threading


def io_task(i):
    time.sleep(0.2)
    return i


def serial(n):
    start = time.time()
    for i in range(n):
        io_task(i)
    return time.time() - start


def concurrent(n):
    start = time.time()
    threads = []
    for i in range(n):
        t = threading.Thread(target=io_task, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return time.time() - start


if __name__ == "__main__":
    print("serial:", serial(5))
    print("concurrent:", concurrent(5))
```

## 解释与原理

串行执行时，5 个 I/O 等待累加。  
并发执行时，等待时间重叠，整体耗时接近单个 I/O 的时间。

## 常见问题与注意事项

1. **并发一定更快吗？**  
   不一定。CPU 密集任务过多并发会导致切换成本升高。

2. **并发是不是等于并行？**  
   不等。单核也可以并发，但不能并行。

3. **并发过多会怎样？**  
   可能导致争用、排队和资源枯竭。

## 最佳实践与建议

- I/O 密集任务优先并发
- CPU 密集任务优先并行（多进程/多核）
- 控制并发度，避免资源放大

## 小结 / 结论

并发的价值在于提高资源利用率和吞吐，并非一味追求速度。  
只有在 I/O 等待明显时，并发才有明显收益。

## 参考与延伸阅读

- *The Art of Multiprocessor Programming*
- Linux 上下文切换与调度
- Go / Java / Python 并发模型对比

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：并发、吞吐、I/O  
- **SEO 关键词**：Concurrency, I/O, Throughput  
- **元描述**：解释为什么需要并发，并给出工程落地建议。

## 行动号召（CTA）

挑一个 I/O 密集接口，试试引入并发或异步，观察吞吐变化。
