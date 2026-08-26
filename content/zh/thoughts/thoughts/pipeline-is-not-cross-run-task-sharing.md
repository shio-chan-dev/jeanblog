---
title: "管线只负责一次运行：文档索引系统不该默认共享跨 Run 节点"
subtitle: "同一文件限制写入并发，在管线内并行，在运行间复用已完成产物"
date: 2026-08-26
summary: "一次 RAG 与知识图谱独立运行模式的架构讨论，让我重新分清了 Pipeline、Run、动态 Target 和跨请求 Singleflight。我的结论是：文档索引系统更适合让每次 Run 独立拥有状态和节点，同一文件只保留一个活动 Run，并通过持久化产物复用结果，而不是共享正在执行的节点任务。"
tags: ["Pipeline", "工作流", "文档索引", "GraphRAG", "LangGraph", "并发控制", "架构设计"]
categories: ["thoughts"]
keywords: ["Pipeline 架构", "文档索引管线", "GraphRAG", "LangGraph", "并发边界", "跨 Run 节点复用", "Singleflight", "动态 Target"]
readingTime: "约 12 分钟"
draft: false
---

> 核心结论：Pipeline 适合组织一次复杂运行内部的步骤、依赖、状态和并行关系，但不应该顺手承担跨 Run 共享正在执行任务的职责。对于会共同写入切片、向量索引和知识图谱的文档处理系统，我更倾向于采用“同一文件单活动 Run、不同文件并发、单次 Run 内按 DAG 并行、后续 Run 复用已完成产物”的模型。

## 目标读者

- 正在设计 OCR、切片、Embedding、RAG 或知识图谱处理链的后端工程师
- 使用 GraphRAG、LangGraph 或自研 DAG Runner 的开发者
- 正在权衡独立运行、共享节点、任务队列和动态目标协调器的架构师
- 遇到取消、续跑、计费和任务所有权越来越复杂问题的团队

## 背景：一个看起来很简单的复用问题

问题来自一条很常见的文档处理管线。

一个文件既可以建立 RAG 索引，也可以抽取知识图谱。两种模式都依赖相同的前置处理：

```text
rag:      prepare -> enrich

graph:    prepare -> graph_extract

parallel: prepare -> enrich
                  \-> graph_extract
```

这里的 `prepare` 可能包括文件转换、OCR、目录识别和切片。它既昂贵，又会被不同模式使用。

一个很自然的想法随之出现：

> 如果 RAG 和 Graph 同时启动，能不能把 `prepare` 做成节点级 `get_or_create`，让两个 Run 共同等待同一个任务？

从减少重复计算的角度看，这个想法很漂亮。但真正落地后，问题很快从“如何复用一次计算”扩展成了：

- 谁拥有这个节点任务？
- 哪个 Run 负责记录 Token？
- 取消 RAG 时，Graph 还在等待，节点能不能停？
- 执行节点的 Worker 挂了以后，谁来接管？
- 租约过期与真实外部写入不一致时，以哪个状态为准？
- 一个节点失败时，两个 Run 都失败，还是只有发起者失败？

这次讨论让我意识到，真正混在一起的不是几段代码，而是四种不同的系统模型。

## 先分清四个概念

### 1. Pipeline：一次运行内部的编排定义

Pipeline 回答的是：

```text
这次任务包含哪些步骤？
它们有什么依赖关系？
哪些步骤可以并行？
失败后从哪里恢复？
```

例如：

```text
prepare
   ├─ enrich
   └─ graph_extract
```

它描述的是一张执行图，而不是一项后台永久能力。

### 2. Run：一次具体调用

Run 是 Pipeline 的一次实例化。它通常拥有自己的：

- `run_id`
- 配置快照
- 运行状态
- 节点状态
- 错误与耗时
- Token 统计
- 取消和续跑语义

同一张 Pipeline 可以运行很多次，但每个 Run 都应该能独立解释“这一次发生了什么”。

### 3. Artifact：节点完成后留下的业务产物

文档处理节点并不只返回内存对象，还会留下长期产物，例如：

- OCR 文本
- 文档目录和切片
- 向量索引
- 实体、关系和图谱快照
- 抽取结果及其版本信息

这些产物可以被后续 Run 复用。它们和“某个正在运行的异步任务”不是同一件事。

### 4. Target：系统期望资源最终具备的能力

动态 Target 关注的不是某次 Run，而是文件的期望状态：

```text
期望能力：prepared + rag_indexed + graph_indexed
当前能力：prepared + rag_indexed
缺失能力：graph_indexed
```

协调器看到差异后，会持续创建任务补齐状态。这更像后台索引平台，而不是普通的“用户启动一次任务”。

## 我原本以为：节点级 get_or_create 会是最自然的复用

如果只看正常路径，跨 Run 共享节点很有吸引力：

```text
RAG Run -----\
              >--- shared prepare ---> 各自继续
Graph Run ---/
```

它似乎同时满足三个目标：

1. RAG 和 Graph 可以独立启动。
2. 两种模式可以同时运行。
3. 重叠节点只计算一次。

问题在于，这三个目标叠加以后，所谓的 `get_or_create` 已经不再是一个简单查询。它实际上需要一套分布式执行协调协议：

```text
claim ownership
-> renew lease
-> wait for remote worker
-> recover expired execution
-> project result into multiple runs
-> resolve cancellation and billing ownership
```

这等于在原有 Pipeline Runner 下面，又实现了一层节点级 Runner。

它不是绝对错误。只是它解决的问题，比“避免重复执行 prepare”大得多。如果业务并不真正要求同一文件的多个模式同时运行，这层复杂度就很难证明值得。

## 实际比较：四个维度决定架构边界

### 1. 并发边界：按系统、文件还是节点划分

“系统支持并发”并不等于“同一个文件的所有任务都必须并发”。

对于文档索引，更自然的边界通常是：

```text
文件 A 与文件 B                    可以并发
同一 Run 内无依赖的 RAG 与 Graph 节点 可以并发
文件 A 的两个独立写入型 Run          不并发
```

这样并没有牺牲整个系统的吞吐量，只是把同一份业务资源限制为单写者。

相反，如果直接允许两个 Run 同时写同一文件的切片、向量库和图谱，就必须明确覆盖、版本、幂等和回滚规则。真正困难的不是 CPU 并发，而是共享副作用。

我的判断是：**并发边界应该优先跟随业务资源的写入边界，而不是跟随 API 上有几个 mode。**

### 2. 状态所有权：一个状态最好只有一个拥有者

独立 Run 模型很容易解释：

```text
RAG Run 拥有 prepare 和 enrich 的状态
Graph Run 拥有 prepare 和 graph_extract 的状态
```

跨 Run 共享节点后，节点执行状态不再属于任何一个 Run。它必须成为第三种实体：

```text
Pipeline Run
Node Execution
Business Artifact
```

然后再维护三者之间的映射。

如果系统本来就要建设通用分布式任务平台，这个抽象可能合理。但如果只是为两个文档处理模式复用一个前置步骤，它很容易成为维护负担。

我的判断是：**能让 Run 直接拥有节点，就不要提前引入独立的节点执行所有权。**

### 3. 取消与失败：共享执行会把局部操作变成协商问题

假设 RAG 和 Graph 正在共同等待 `prepare`：

- 用户取消 RAG，`prepare` 是否继续？
- Graph 随后也取消，谁负责真正停止底层任务？
- `prepare` 已经写了一半切片，能否由新 Worker 接管？
- RAG 超时，但 Graph 愿意继续等待，节点状态如何投影？

只要共享正在执行的任务，就必须区分“执行拥有者”和“等待者”。取消也不能再简单地等于 `task.cancel()`。

而同文件单活动 Run 的语义很直接：

```text
取消 Run
-> 停止这次 Run 尚未完成的节点
-> 持久化终态
-> 下次根据真实产物判断复用或重做
```

我的判断是：**如果用户需要清晰的单次任务取消和审计，独立 Run 比共享执行更自然。**

### 4. 复用方式：复用完成产物通常比复用进行中任务稳定

这里最容易混淆的是两种 `get_or_create`。

第一种是 Run 级幂等：

```text
相同文件、相同 mode 已经在运行
-> 返回已有 Run
```

第二种是 Artifact 级复用：

```text
当前文件版本已有有效 prepare 产物
-> 读取产物并把当前节点标记为 reused/success

没有有效产物
-> 当前 Run 自己执行并创建产物
```

我认为这两种都值得保留。

真正应该谨慎的是第三种：

```text
另一个 Run 正在执行 prepare
-> 当前 Run 加入等待
-> 多个 Run 共享任务、所有权和结果
```

完成产物已经有稳定事实可以检查；进行中任务只有暂时状态，还伴随进程存活、网络中断和部分写入。二者的可靠性基础完全不同。

## GraphRAG 和 LangGraph 给我的启发

GraphRAG 使用 Pipeline，是因为建立图谱索引本身就是多阶段数据加工：加载文档、切片、抽取图谱、检测社区、生成报告和建立向量表示。官方索引架构显式展示了这些 Workflow 的依赖关系。

我查看的 GraphRAG Runner 会为一次索引创建 `PipelineRunContext`，其中包含 storage、cache、state、callbacks 和 stats，然后运行这次 Pipeline 中的 Workflow。它还通过 LLM Cache 复用相同输入的模型结果。

这里值得借鉴的是：

```text
Pipeline 组织一次索引
Context 承载本次运行状态
Storage 保存长期业务产物
Cache 减少重复模型调用
```

我没有在这条 Runner 路径中看到“自动把两个独立 Run 的相同 Workflow 合并成一个共享执行”的语义。

LangGraph 强调的则是 State、Nodes 和 Edges。节点并行发生在同一次图运行可以同时推进的步骤内；持久化和 durable execution 服务于该图运行的恢复。对于同一线程收到并发请求，LangGraph 的部署文档另外提供 `enqueue`、`reject`、`interrupt` 和 `rollback` 等策略。

这说明两个问题被刻意分开了：

```text
图内部如何执行       -> Graph/Pipeline Runner
多个运行如何竞争资源 -> 并发策略
```

这也是我认为最值得保留的边界。

## 最适合文档索引的方案

综合这次讨论，我更倾向于下面这套规则：

```text
同一文件版本只允许一个 active Run

相同文件 + 相同 mode
-> 返回已有 Run，保持触发幂等

相同文件 + 不同 mode
-> 拒绝新请求，提示当前文件正在处理

不同文件
-> 正常并发

同时需要 RAG 和 Graph
-> 显式启动 parallel Pipeline

后续启动其他 mode
-> 复用版本匹配的已完成产物
```

可以用很短的伪代码表达入口规则：

```python
async def trigger(file_id, mode):
    active_run = await find_active_run(file_id)

    if active_run is None:
        return await create_run(file_id, mode)

    if active_run.mode == mode:
        return active_run

    raise FileAlreadyProcessing(active_run.mode)
```

节点层只处理产物复用：

```python
async def run_prepare(file_version):
    artifact = await find_valid_prepare_artifact(file_version)
    if artifact is not None:
        return reused(artifact)

    return await execute_and_persist_prepare(file_version)
```

这两个函数的简单之处，来自一个重要约束：正常情况下不会有另一个 Run 正在同时写同一文件版本。

## 这种方案适合什么

它适合以下场景：

- 节点会写共享切片、索引、数据库或知识图谱
- 单次处理昂贵，需要记录耗时和 Token
- 用户关心一次任务的启动、失败、取消和续跑
- 同一文件多模式同时启动不是核心需求
- 系统仍然需要让不同文件高并发处理
- 一次 Pipeline 内部仍有值得并行的独立节点

这并不是“低并发方案”。它只是把并发控制在不会争写同一业务资源的位置。

## 它不适合什么

### 1. 不适合所有请求都必须执行的批处理平台

如果每个请求都不能丢，冲突时直接拒绝就不够，需要 `enqueue`，并补充排队状态、取消排队任务、优先级和过期策略。

### 2. 不适合用户只关心最终能力的持续协调系统

如果产品语义是“保证文件最终具备 RAG、Graph、OCR 等能力”，动态 Target 更合适。此时用户修改的是期望状态，后台协调器持续补齐差异，具体创建几次 Job 只是内部实现。

### 3. 不适合大量相同的纯计算请求

如果任务没有外部副作用，输入和输出都不可变，例如计算文件哈希或执行确定性的昂贵推理，那么跨请求 Singleflight 很有价值：第一个请求执行，其他请求等待同一结果。

### 4. 不适合输出完全隔离的任务

同一文件同时做翻译、病毒扫描和预览图生成，如果三者没有共享写入，可以直接作为独立 Run 并行。文件级互斥反而会降低吞吐。

## 四个具体例子

### 例子一：RAG 和 Graph 写共享文档产物

```text
第一次启动 RAG：
prepare -> 实际执行
enrich  -> 实际执行

RAG 完成后启动 Graph：
prepare       -> 复用有效切片
graph_extract -> 实际执行
```

适合单文件单活动 Run，加 Artifact 级复用。

### 例子二：搜索平台持续补齐能力

```text
期望：ocr=true, rag=true, graph=true
实际：ocr=true, rag=true, graph=false
动作：后台自动创建 graph job
```

适合动态 Target 协调器。用户关心文件何时可用于查询，而不是某次 Run 的编号。

### 例子三：100 个请求计算相同文件哈希

```text
request 1 --\
request 2 ----> one hash task -> same immutable result
request 3 --/
```

适合跨请求 Singleflight，因为没有复杂副作用、计费和取消归属。

### 例子四：同一文件生成三种隔离输出

```text
translate -> translation/file-A.json
preview   -> preview/file-A/
scan      -> scan/file-A.json
```

适合独立 Run 直接并行，因为三者不争用同一份写入资源。

## 和我当前工作流的边界

我现在会把职责分成四层：

- Mode 负责选择一张完整 DAG
- Run 负责一次执行的状态、取消、续跑和审计
- Workflow 负责图中的业务步骤
- Artifact 负责跨 Run 的长期结果复用

如果以后新增一种文件处理方式，我会先判断：

1. 它是否仍属于同一种文件处理生命周期？如果是，新增 Workflow 或 Mode。
2. 它的输入、状态和产物是否完全不同？如果是，考虑独立 Pipeline。
3. 它是否只是另一项期望能力？只有产品转向持续索引平台时，才引入动态 Target。
4. 它是否真的需要共享进行中的计算？只有高频、昂贵、无副作用且结果不可变时，才考虑 Singleflight。

这套边界保留了 Pipeline 的扩展能力，也避免它逐渐变成什么都负责的全局调度器。

## 一个最小的架构判断方法

遇到类似问题时，我会依次问五个问题：

1. 两个任务是否写同一份业务资源？
2. 用户关心单次任务，还是只关心最终能力？
3. 取消一个请求时，另一个请求是否应该继续？
4. 已完成产物是否足以支持后续复用？
5. 同一输入的并发请求是否高频到值得承担分布式所有权成本？

可以快速映射成下面的选择表：

| 场景 | 推荐模型 |
| --- | --- |
| 共享写入，单次任务需要审计 | 同一资源单活动 Run |
| 每个请求都必须执行 | 同一资源队列 `enqueue` |
| 只关心最终能力 | 动态 Target 协调器 |
| 大量相同、无副作用的昂贵计算 | 跨请求 Singleflight |
| 输出完全隔离 | 独立 Run 并行 |

## 常见问题与注意事项

### 单文件单活动 Run 会不会降低并发？

它只限制同一文件的竞争写入。不同文件仍然可以并发，单次 Run 内无依赖的节点也可以并行。是否影响吞吐，应看真实流量中“同一文件同时触发多个模式”的比例，而不是只看理论并发数。

### 相同 mode 重复点击应该报错吗？

不一定。更友好的做法是返回已有 Run，让触发接口保持幂等。不同 mode 才返回冲突，提示用户等待或改用 `parallel`。

### 已完成产物如何判断有效？

至少应考虑：

```text
file identity/version
+ workflow name
+ workflow implementation or schema version
```

不能只看“目录存在”或“数据库有记录”，否则代码、Prompt 或抽取结构升级后可能误用旧结果。

### Worker 崩溃后还需要所有权机制吗？

仍需要判断 Run 是否失联，以及是否允许续跑，但它是 Run 级恢复问题。恢复时再根据真实 Artifact 判断节点是复用还是重做，不必为每个节点建立长期租约和等待者关系。

### 什么时候应该升级成节点级共享执行？

只有当监控数据证明大量成本来自相同文件、相同版本、相同 Workflow 的并发重复执行，并且这些节点的副作用、取消、计费和恢复语义都能清楚定义时，才值得重新评估。不要因为理论上能省一次 `prepare` 就先支付整套分布式协调成本。

## 最佳实践与建议

- 先定义业务资源的写入边界，再定义并发边界。
- 让一次 Run 完整拥有自己的节点状态。
- 把 DAG 内并行和跨 Run 并发分开设计。
- 相同 mode 的重复触发返回已有 Run。
- 不同 mode 冲突时先使用 `reject`；确有排队需求再引入 `enqueue`。
- 同时需要 RAG 和 Graph 时使用显式 `parallel` DAG。
- 跨 Run 优先复用已完成且版本匹配的 Artifact。
- 只有纯计算、高重复率场景才优先考虑共享进行中任务。
- 不要让 Cache、Artifact、Node Execution 和 Pipeline Run 变成同一个概念。

## 小结

这次架构讨论最后让我确认了一件事：**Pipeline 的价值是让一次复杂处理变得可组合、可观测、可恢复，而不是让所有 Run 自动共享执行。**

对文档索引系统来说，一个更稳妥的默认模型是：

```text
不同文件并发
同一文件单活动 Run
单次 Run 内按 DAG 并行
后续 Run 复用已完成产物
```

如果未来产品真的转向“持续保证每个文件具备一组索引能力”，再把动态 Target 协调器放到 Pipeline 上面；如果未来监控真的证明存在大量相同纯计算，再引入 Singleflight。不要在需求尚未出现时，让一个简单的节点复用问题提前演化成分布式任务所有权系统。

## 参考与延伸阅读

- [Microsoft GraphRAG: Indexing Architecture](https://microsoft.github.io/graphrag/index/architecture/)
- [Microsoft GraphRAG: `run_pipeline.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/run/run_pipeline.py)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangSmith Deployment: Double texting and concurrent run strategies](https://docs.langchain.com/langsmith/double-texting)
- [从参数直传到 Pipeline：一次可复现、可观测的数据处理管线改造实践]({{< ref "/dev/python/from-direct-params-to-config-driven-etl-pipeline.md" >}})

## 元信息

- 文章类型：架构选型 / 工作流评估
- 核心主题：Pipeline、Run、Artifact、动态 Target、Singleflight
- 适用范围：文档处理、RAG 索引、知识图谱抽取及其他共享写入型数据管线
- 结论强度：有条件推荐；前提是同一资源多模式并发不是核心产品需求

## 行动建议

如果你也在设计类似系统，可以先画出两张图：一张是单次 Run 内的 DAG，另一张是多个 Run 对业务存储的读写关系。前一张决定 Pipeline 如何编排，后一张才决定是否需要互斥、排队或共享执行。不要用第一张图替代第二张图的判断。
