---
title: "让 FastAPI 异步真正‘不卡’：asyncio.create_task + to_thread 并发实践（含 MySQL 写入）"
date: 2025-11-19
draft: false
tags: ["FastAPI", "asyncio", "并发", "线程池", "MySQL", "Python"]
keywords: ["FastAPI 异步", "asyncio to_thread", "create_task 并发", "MySQL 连接池", "事件循环阻塞", "Python 并发"]
description: "把同步重活丢给线程、把可并行的子流程拆出来并发执行，让 FastAPI WebSocket/HTTP 服务在高并发文件处理场景下保持流畅与可靠。"
---

# 让 FastAPI 异步真正“不卡”：asyncio.create_task + to_thread 并发实践（含 MySQL 写入）

**副标题 / 摘要**

把同步重活丢给线程、把可并行的子流程拆出来并发执行，让你的 FastAPI WebSocket/HTTP 服务在高并发文件处理场景下保持流畅与可靠。适合需要在事件循环中混合 CPU 计算与阻塞 I/O 的工程团队。

---

## 目标读者

- 中级后端工程师、服务端架构师
- 正在用 FastAPI/asyncio 落地异步工作流、混合 I/O/CPU 任务的开发者

---

## 背景 / 动机

常见痛点：

- 在异步服务里不小心执行了同步 CPU/数据库操作，单个请求“卡住”事件循环，导致同一 worker 上的其它请求/WebSocket 心跳/进度推送都被拖慢。
- CPU/数据库步骤彼此本无强依赖，却被串行放到一条链上，整体时延被“关键路径”拖长。

目标：

- 不改变外部行为的前提下，消除事件循环阻塞。
- 让独立步骤并发执行，缩短关键路径。

---

## 核心概念

- 线程（Thread）：同一进程内共享内存，切换开销低；CPython 受 GIL 限制，纯 Python CPU 计算难并行，但适合并发等待阻塞 I/O。
- 进程（Process）：独立内存、无 GIL 约束，CPU 计算可多核并行；切换/通信成本更高，参数/结果需可序列化。
- 异步（async/await）：单线程事件循环的协作式调度；只有在 await 时让出控制权，同步阻塞会“卡死”循环。
- asyncio.to_thread：把同步函数放到后台线程，释放事件循环；不等于多核加速，但对阻塞 I/O 有实效。
- asyncio.create_task：并发启动一个协程，让它和当前协程重叠运行；用于编排并发，而非解除阻塞。

---

## 实践指南 / 步骤

1) 识别阻塞点（示例项目）

- CPU 构树/展平/序列化：`HeaderTree.from_documents`、`flatten_dfs`、`FlatHeaderTree.to_dict`
- 同步 MySQL 写入：`file_tree_table.upsert_tree`

2) 用 to_thread 包裹同步重活（释放事件循环）

- 在 `build_file_tree` 中，将 CPU/DB 步骤放入 `await asyncio.to_thread(...)`。

3) 并发编排，缩短关键路径

- 在 `full_pipeline_async`：在 split 后立即 `create_task(build_file_tree(...))`，并发执行图片/表格处理、重组、存储；返回前再 `await` 构树结果。

4) 可选：事件屏障与互斥

- 如需“保证某步骤不早于构树完成”，用 `asyncio.Event`。
- 多协程修改共享状态，用 `asyncio.Lock` 保护原子更新。

5) 观测与参数

- MySQL 连接池每进程默认较小（示例为 2），必要时调大。
- Uvicorn `workers` 控制进程数，提升隔离与吞吐。

---

## 可运行示例

非阻塞构树与持久化（替换 `build_file_tree` 内部）：

```python
import asyncio
from typing import Dict, List
from langchain_core.documents import Document
from repositories.file_tree_table import file_tree_table

async def build_file_tree(self, file_id: str, docs: List[Document]) -> Dict:
    self._update_progress("creating_tree", 0, 100, f"开始创建文件树结构 file_id={file_id}")

    tree = await asyncio.to_thread(self.tree_peocessor.from_documents, docs)
    flat_tree = await asyncio.to_thread(tree.flatten_dfs)
    tree_json = await asyncio.to_thread(flat_tree.to_dict)

    collection_name = f"md_{file_id}"
    await asyncio.to_thread(file_tree_table.upsert_tree, file_id, collection_name, tree_json)

    self._update_progress("creating_tree", 100, 100, "创建文件树结构完成")
    return tree_json
```

并发编排（在 `full_pipeline_async` 中让构树与后续步骤重叠）：

```python
build_task = asyncio.create_task(self.build_file_tree(task_status.file_id, split_documents))

processed_documents = await self.process_content_blocks(split_documents)
reorganized_docs = await self.reorganize_documents(processed_documents)
await self.store_to_vectorstore(qdrant_storage, split_documents, reorganized_docs,
                                collection_name, force_recreate)

directory = await build_task  # 返回前汇合，确保目录树与写库都已完成
```

事件屏障（保证“任何步骤不早于构树完成”）：

```python
# __init__
self._tree_ready = asyncio.Event()

# build_file_tree 末尾
self._tree_ready.set()

# 需要保证顺序的位置（如返回前或某一步末尾）
await self._tree_ready.wait()
```

互斥保护共享状态（避免交错写）：

```python
# __init__
self._state_lock = asyncio.Lock()

# 修改共享状态
async with self._state_lock:
    self.progress_state["stages"]["reorganizing"]["current"] = x
```

---

## 解释与原理

- 为什么 to_thread 有效：同步 CPU/DB 会占住事件循环；丢到线程后，事件循环空闲，可继续调度其它协程（WebSocket 心跳/进度、其它文件的步骤）。对 CPU 计算不一定更快（受 GIL），但“服务不卡”。
- 为什么 create_task：把“只在末尾需要”的构树步骤并发启动，缩短关键路径；最后再等待结果即可保证一致性。
- 替代方案与取舍：
  - 多进程（ProcessPoolExecutor）：能加速纯 CPU，但要可序列化、成本更高、代码改动更大。
  - 原生异步数据库（aiomysql/asyncmy）：从根上避免阻塞 I/O，但需要重写仓储层与连接管理。
  - 仅提高 workers：能隔离阻塞影响，但单 worker 内依旧会阻塞；治标不治本。

---

## 常见问题与注意事项

- to_thread 不能强杀线程：取消 await 不会停止后台函数执行；对幂等与可重入要有准备。
- GIL 限制：纯 Python CPU 计算用线程不提速；若要加速，考虑多进程或释放 GIL 的实现。
- 数据库连接池：高并发 upsert 会排队；按压力调大连接池。
- 线程安全：不要复用同一连接/游标到多个线程；每次获取新连接更安全。
- 错误传播：线程内抛出的异常会在 await 处重新抛出，注意日志与兜底。
- 任务生命周期：用 `_current_tasks` 跟踪 `create_task`，统一取消与清理。
- 资源清理：WebSocket/文件句柄/HTTP 会话要在 finally 里关闭。

---

## 最佳实践与建议

- 把“阻塞 I/O”优先放到线程；把“纯 CPU”优先放到进程。
- 把“只在收尾需要的步骤”并发启动，最后汇合等待。
- 用事件屏障控制顺序，用锁保护共享状态的原子更新。
- 观测优先：为每个阶段打点记录耗时与排队，基于数据调参（线程池大小、连接池、workers）。
- 失败即早停：任一分支失败/取消，及时取消其他协程并清理。

---

## 小结 / 结论

通过 `asyncio.to_thread` 解除事件循环阻塞、通过 `asyncio.create_task` 并发独立子流程，你可以在不改业务语义的前提下，显著提升 FastAPI 异步服务在重 I/O/轻 CPU 混合场景下的平滑度与吞吐。

下一步建议：

- 将构树与写库改为非阻塞并发。
- 根据观测调大数据库连接池与线程池规模。
- 评估是否将重 CPU 步骤迁移到多进程或释放 GIL 的库。

---

## 参考与延伸阅读

- Python 官方文档：asyncio（to_thread, create_task, TaskGroup）
- FastAPI 官方：Concurrency and async/await
- mysql-connector-python 文档：连接池与线程使用
- The GIL and its effects on Python multithreading

---

## 元信息

- 阅读时长：8–12 分钟
- 标签：FastAPI、asyncio、并发、线程池、数据库、性能优化
- SEO 关键词：FastAPI 异步、asyncio to_thread、create_task 并发、MySQL 连接池、事件循环阻塞
- 元描述：在 FastAPI 异步服务中用 asyncio.to_thread 解除事件循环阻塞，并用 create_task 并发独立子流程，含可复制代码片段与工程实践建议。

---

## 行动号召（CTA）

- 试一试：把你的构树/写库步骤替换为 to_thread，并在 split 后用 create_task 并发跑；观察吞吐与延迟变化。
- 如果需要，我可以帮你把上述改动直接打补丁到你的仓库里，并提供一版可回滚的差异。

