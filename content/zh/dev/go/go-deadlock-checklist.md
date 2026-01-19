---
title: "Go 死锁排查 Checklist：从报错到定位的实用手册"
date: 2026-01-19T12:34:37+08:00
description: "整理一套可执行的 Go 死锁排查清单，覆盖日志、堆栈、协程状态与常见根因快速定位。"
tags: ["go", "concurrency", "deadlock", "debug", "checklist"]
categories: ["go", "concurrency"]
draft: false
---

### **标题**

Go 死锁排查 Checklist：从报错到定位的实用手册

---

### **副标题 / 摘要**

一页式清单，帮助你在看到 `all goroutines are asleep - deadlock!` 时，
快速定位是哪一类等待造成卡死。

---

### **目标读者**

* **初学者**：首次遇到 deadlock，不知道从哪下手。
* **中级开发者**：需要可复用的排查流程，缩短定位时间。
* **团队负责人**：希望沉淀成团队规范，避免重复踩坑。

---

### **背景 / 动机**

死锁往往发生在高并发与多协作场景，复现难、定位慢。
有一份稳定的排查清单，可以把“凭直觉猜”变成“按步骤验证”。

---

### **核心概念**

* **deadlock 报错**：所有 goroutine 都在等待，程序无法推进。
* **堆栈定位**：栈上出现 `<-ch` / `ch <-` / `mu.Lock()` / `wg.Wait()`。
* **依赖闭环**：等待关系形成环，导致无人能继续执行。

---

### **实践指南 / 步骤**

1️⃣ **确认报错与堆栈是否完整**

* 记录 `fatal error: all goroutines are asleep - deadlock!` 后的完整堆栈。
* 优先关注 main goroutine 的等待点。

2️⃣ **分类定位阻塞类型**

* channel：`<-ch` / `ch <-`
* WaitGroup：`wg.Wait()`
* Mutex：`mu.Lock()` / `RWMutex` 的读写锁等待

3️⃣ **检查等待关系是否闭环**

* A 等 B，B 等 C，C 再等 A
* 多锁场景优先看锁顺序是否一致

4️⃣ **核对计数与配对关系**

* WaitGroup：Add 与 Done 是否等量
* channel：发送者/接收者是否配对

5️⃣ **复现与最小化**

* 抽取最小可复现场景
* 去掉无关逻辑，集中复现死锁点

---

### **可运行示例**

下面示例演示如何主动打印 goroutine 栈（用于非 runtime deadlock 的卡死场景）：

```go
package main

import (
	"fmt"
	"runtime"
	"time"
)

func main() {
	go func() {
		for {
			time.Sleep(2 * time.Second)
			buf := make([]byte, 1<<16)
			n := runtime.Stack(buf, true)
			fmt.Println(string(buf[:n]))
		}
	}()

	select {} // 模拟永久阻塞
}
```

---

### **解释与原理**

* **堆栈是最重要的证据**：deadlock 报错后，堆栈就是“案发现场”。
* **分类比盲查更快**：先确定是 channel、WaitGroup 还是 mutex，再去找配对关系。
* **最小化复现**：能把问题从复杂业务中剥离出来，减少误判。

---

### **常见问题与注意事项**

* **Q：没有 deadlock 报错，但程序卡住了？**  
  A：可能是 goroutine 没全部阻塞，需用 `runtime.Stack` 或 pprof 排查。
* **Q：加缓冲能解决吗？**  
  A：缓冲只是延后阻塞，闭环仍在。
* **Q：WaitGroup 为什么最常见？**  
  A：Add 在主协程，Done 在子协程，最容易遗漏。

---

### **最佳实践与建议**

* **先对齐收发，再考虑优化**：无缓冲 channel 必须保证收发存在。
* **写清楚 Done 责任**：谁 Add 谁确保 Done。
* **统一锁顺序**：多锁场景的顺序必须固定。
* **为协程设计退出路径**：防止永远等待。

---

### **小结 / 结论**

死锁排查的关键是：确认等待点、分类阻塞类型、查找依赖闭环。
按清单执行，基本能在短时间内定位根因。

---

### **参考与延伸阅读**

* 📘 [Go 官方并发教程](https://go.dev/doc/effective_go#concurrency)
* 📗 [Go blog: Pipelines and cancellation](https://go.dev/blog/pipelines)
* 🧩 [runtime.Stack 文档](https://pkg.go.dev/runtime#Stack)

---

### **元信息**

* **阅读时长**：约 6 分钟
* **标签**：Go、并发、死锁、排查、Checklist
* **SEO 关键词**：Go 死锁排查、deadlock、goroutine 堆栈、WaitGroup、channel
* **元描述**：一页式 Go 死锁排查清单，覆盖堆栈分析、等待分类与闭环定位方法。

---

### **行动号召（CTA）**

如果你有一段死锁堆栈或可复现代码，发出来，我可以帮你做具体定位。
