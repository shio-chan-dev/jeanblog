---
title: "从零实现一个最小 Pipeline Runner"
subtitle: "普通函数调用够用时不要上 pipeline；当执行过程必须被观察时，runner 才有价值"
date: 2026-05-29T00:00:00+08:00
draft: true
description: "从普通函数调用开始，逐步推导一个最小顺序 Pipeline Runner：mode 解析、handler 执行、共享 context、StepRecord 和 RunRecord。"
summary: "这篇文章从两个普通函数调用开始，不预设 pipeline 结论，而是通过 run/step observation、失败记录和重复执行策略的压力，推导出一个最小可运行的顺序 Pipeline Runner。"
tags: ["Pipeline", "Python", "工程实践", "可观察性", "教程"]
categories: ["工程实践"]
keywords: ["pipeline runner", "python pipeline", "workflow runner", "step observation", "run record", "工程实践"]
readingTime: "约 12 分钟"
---

> 源码仓库：<https://github.com/shio-chan-dev/nano-pipeline-runner>
>
> 本文对应仓库里的 v0 checkpoint：最小顺序 Pipeline Runner。注意：如果仓库仍是 private，公开发布前需要先把仓库改成 public，否则读者无法访问。

## 目标读者

这篇文章适合下面几类读者：

- 会写 Python 函数、字典、异常处理和简单 `dataclass` 的开发者
- 听过 pipeline / workflow runner，但不清楚它和普通函数调用差别在哪里的人
- 想把“多个步骤按顺序执行”变成可观察、可测试结构的人

读完以后，你应该能实现并解释这个最小链路：

```text
trigger(mode, context)
-> resolve workflow names
-> run handlers in order
-> record each executed step
-> return one run record
```

这里的重点不是证明“普通函数调用不好”。恰恰相反，如果普通函数调用已经足够，就不要引入 pipeline。本文只在一个具体压力出现后继续往前走：**调用方不仅要最终结果，还要知道这次 run 里每个 step 的状态、结果和错误。**

## 背景：什么时候普通函数调用不够

假设我们要处理一个文件。业务上看起来像一个动作：

```text
process_file(file_id)
```

但里面其实有两个步骤：

```text
prepare -> enrich
```

最直接的代码是这样：

```python
def prepare(file_id):
    return {"file_id": file_id, "prepared": True}


def enrich(prepared_file):
    return {"saw_prepared": prepared_file["prepared"]}


file_id = "file-1"
prepared_file = prepare(file_id)
enrich_result = enrich(prepared_file)

assert prepared_file == {"file_id": "file-1", "prepared": True}
assert enrich_result == {"saw_prepared": True}
```

这段代码没有问题。如果调用方只关心 `enrich_result`，到这里就应该停下来。

真正的问题出现在调用方想问这些问题时：

```text
这次运行整体成功还是失败？
prepare 跑了吗？
enrich 跑了吗？
哪个 step 失败了？
失败前已经完成了哪些 step？
```

普通函数调用可以抛异常，但它不会自然返回一个这样的运行观察：

```python
{
    "status": "failed",
    "steps": [
        {"name": "prepare", "status": "success"},
        {"name": "enrich", "status": "failed", "error": "enrich failed"},
    ],
}
```

这才是引入 pipeline runner 的起点：不是“函数不扩展”，而是**执行过程没有被表示成数据**。

## 先用普通函数包一层

在引入 runner 之前，先做最简单的改进：

```python
def run_standard(file_id):
    prepared_file = prepare(file_id)
    return enrich(prepared_file)


assert run_standard("file-1") == {"saw_prepared": True}
```

这个包装函数仍然不是 pipeline。它只是把两个调用藏到一个业务函数里。

如果调用方只需要最终结果，`run_standard(file_id)` 就是正确的设计。不要因为代码里有两个步骤，就自动上 pipeline。

它仍然缺少的是运行观察：

```text
没有 run status
没有 step list
没有 failed step record
```

## 给一个流程返回运行观察

下一步只解决一个问题：让 `run_standard()` 返回 run record。

```python
def run_standard(file_id):
    record = {
        "name": "standard",
        "status": "running",
        "steps": [],
    }

    prepared_file = prepare(file_id)
    record["steps"].append(
        {
            "name": "prepare",
            "status": "success",
            "result": prepared_file,
        }
    )

    enrich_result = enrich(prepared_file)
    record["steps"].append(
        {
            "name": "enrich",
            "status": "success",
            "result": enrich_result,
        }
    )

    record["status"] = "success"
    return record


record = run_standard("file-1")

assert record["status"] == "success"
assert [step["name"] for step in record["steps"]] == ["prepare", "enrich"]
assert [step["status"] for step in record["steps"]] == ["success", "success"]
assert record["steps"][1]["result"] == {"saw_prepared": True}
```

现在成功路径可观察了。但失败路径仍然有问题：如果 `enrich` 抛异常，函数会直接退出，调用方拿不到 `enrich` 的 failed step。

所以异常必须在 step 边界被捕获：

```python
def run_standard(file_id):
    record = {
        "name": "standard",
        "status": "running",
        "steps": [],
    }

    try:
        prepared_file = prepare(file_id)
    except Exception as exc:
        record["steps"].append(
            {
                "name": "prepare",
                "status": "failed",
                "error": str(exc),
            }
        )
        record["status"] = "failed"
        return record

    record["steps"].append(
        {
            "name": "prepare",
            "status": "success",
            "result": prepared_file,
        }
    )

    try:
        enrich_result = enrich(prepared_file)
    except Exception as exc:
        record["steps"].append(
            {
                "name": "enrich",
                "status": "failed",
                "error": str(exc),
            }
        )
        record["status"] = "failed"
        return record

    record["steps"].append(
        {
            "name": "enrich",
            "status": "success",
            "result": enrich_result,
        }
    )
    record["status"] = "success"
    return record
```

到这里为止，仍然不一定需要 pipeline。一个 observed flow 直接写成这样也可以。

## 真正的压力：第二个流程也要同样的观察策略

现在出现第二个流程：

```text
core -> prepare
```

它也要返回同样的 run/step observation。直接写当然可以：

```python
def run_core(file_id):
    record = {
        "name": "core",
        "status": "running",
        "steps": [],
    }

    try:
        prepared_file = prepare(file_id)
    except Exception as exc:
        record["steps"].append(
            {
                "name": "prepare",
                "status": "failed",
                "error": str(exc),
            }
        )
        record["status"] = "failed"
        return record

    record["steps"].append(
        {
            "name": "prepare",
            "status": "success",
            "result": prepared_file,
        }
    )
    record["status"] = "success"
    return record
```

问题不是 `prepare` 被调用了两次。问题是两个函数开始复制同一套执行策略：

- 创建 run record
- 按顺序执行 step
- handler 成功后追加 success step
- handler 失败后追加 failed step
- 第一个失败后停止后续步骤
- 最后标记 run success / failed

这时候 pipeline runner 才有价值：把**执行策略**从每个业务流程里抽出来。

## 统一 handler 入参：引入 context

要用一个循环执行多个 step，每个 handler 需要有统一的调用形状。

现在的问题是：

```python
prepared_file = prepare(file_id)
enrich_result = enrich(prepared_file)
```

`prepare` 接收 `file_id`，`enrich` 接收 `prepared_file`。如果 runner 要理解每一步的特殊入参，它又会退化成业务编排函数。

所以我们引入一个共享运行状态：`context`。

```python
def prepare(context):
    context["prepared"] = True
    return {"prepared": True}


def enrich(context):
    return {"saw_prepared": context["prepared"]}


context = {"file_id": "file-1"}
prepare_result = prepare(context)
enrich_result = enrich(context)

assert prepare_result == {"prepared": True}
assert enrich_result == {"saw_prepared": True}
assert context == {"file_id": "file-1", "prepared": True}
```

这里要注意一个边界：`context` 是一个 run 内共享状态，不是配置对象。runner 不应该复制它，否则 handler 修改过的状态就无法被后续 handler 和调用方观察到。

## 用 mode 表达调用方想跑哪个流程

两个 observed flow 的差异现在只剩下 step list：

```python
mode_workflows = {
    "core": ["prepare"],
    "standard": ["prepare", "enrich"],
}
```

`mode` 是调用方语言，workflow name 是 runner 内部语言。我们先加一个 resolver：

```python
def resolve_workflows(mode):
    workflows = mode_workflows.get(mode)
    if workflows is None:
        raise ValueError(f"unknown mode: {mode}")
    return workflows


assert resolve_workflows("standard") == ["prepare", "enrich"]
assert resolve_workflows("core") == ["prepare"]

try:
    resolve_workflows("extract")
except ValueError as exc:
    assert str(exc) == "unknown mode: extract"
else:
    raise AssertionError("unknown mode should fail")
```

unknown mode 必须在任何 handler 运行前失败。否则一个无效请求可能已经产生副作用。

## 用 handlers 表把名字变成函数

workflow name 本身不能执行。我们还需要把名字解析成 handler：

```python
handlers = {
    "prepare": prepare,
    "enrich": enrich,
}


def resolve_handler(workflow_name):
    handler = handlers.get(workflow_name)
    if handler is None:
        raise ValueError(f"missing handler: {workflow_name}")
    return handler
```

然后就可以写出第一个通用 executor：

```python
def run_observed(mode, context):
    workflow_names = resolve_workflows(mode)
    record = {
        "mode": mode,
        "workflows": list(workflow_names),
        "context": context,
        "status": "running",
        "steps": [],
    }

    for workflow_name in workflow_names:
        handler = resolve_handler(workflow_name)
        try:
            result = handler(record["context"])
        except Exception as exc:
            record["steps"].append(
                {
                    "name": workflow_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            record["status"] = "failed"
            return record
        record["steps"].append(
            {
                "name": workflow_name,
                "status": "success",
                "result": result,
            }
        )

    record["status"] = "success"
    return record
```

检查两个流程是否共用同一套执行策略：

```python
standard = run_observed("standard", {"file_id": "file-1"})
core = run_observed("core", {"file_id": "file-1"})

assert standard["status"] == "success"
assert standard["workflows"] == ["prepare", "enrich"]
assert [step["name"] for step in standard["steps"]] == ["prepare", "enrich"]

assert core["status"] == "success"
assert core["workflows"] == ["prepare"]
assert [step["name"] for step in core["steps"]] == ["prepare"]
```

到这里，pipeline 的核心已经出现：

```text
mode -> workflow names -> handler calls -> observed run
```

## 把观察结果固定成 record

手写字典能跑，但字典形状已经变成契约了：

```python
{
    "name": workflow_name,
    "status": "success",
    "result": result,
}
```

这时应该把它命名为结构：

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    name: str
    status: str
    result: Any = None
    error: str | None = None


@dataclass
class RunRecord:
    mode: str
    workflows: list[str]
    context: dict[str, Any]
    status: str = "queued"
    steps: list[StepRecord] = field(default_factory=list)
```

`StepRecord` 描述一个 step 的结果。`RunRecord` 描述一次 trigger 的整体结果。

这里 `result` 放在 `StepRecord` 上，而不是 `RunRecord` 上，是因为 v0 的 run 是步骤集合，不一定有唯一最终结果。每个 step 的返回值属于那个 step；整个 run 的主要结果是 `status` 和 `steps`。

## 最终 API：PipelineRunner.trigger

最后把全局表和 helper 函数收进一个对象：

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

WorkflowHandler = Callable[[dict[str, Any]], Any]


@dataclass
class StepRecord:
    name: str
    status: str
    result: Any = None
    error: str | None = None


@dataclass
class RunRecord:
    mode: str
    workflows: list[str]
    context: dict[str, Any]
    status: str = "queued"
    steps: list[StepRecord] = field(default_factory=list)


class PipelineRunner:
    def __init__(
        self,
        *,
        mode_workflows: Mapping[str, list[str]],
        handlers: Mapping[str, WorkflowHandler],
    ) -> None:
        self._mode_workflows = dict(mode_workflows)
        self._handlers = dict(handlers)

    def trigger(self, mode: str, context: dict[str, Any]) -> RunRecord:
        workflow_names = self._resolve_workflows(mode)
        record = RunRecord(
            mode=mode,
            workflows=list(workflow_names),
            context=context,
            status="running",
        )

        for workflow_name in workflow_names:
            handler = self._resolve_handler(workflow_name)
            try:
                result = handler(record.context)
            except Exception as exc:
                record.steps.append(
                    StepRecord(
                        name=workflow_name,
                        status="failed",
                        error=str(exc),
                    )
                )
                record.status = "failed"
                return record
            record.steps.append(
                StepRecord(
                    name=workflow_name,
                    status="success",
                    result=result,
                )
            )

        record.status = "success"
        return record

    def _resolve_workflows(self, mode: str) -> list[str]:
        workflows = self._mode_workflows.get(mode)
        if workflows is None:
            raise ValueError(f"unknown mode: {mode}")
        return workflows

    def _resolve_handler(self, workflow_name: str) -> WorkflowHandler:
        handler = self._handlers.get(workflow_name)
        if handler is None:
            raise ValueError(f"missing handler: {workflow_name}")
        return handler
```

这个 checkpoint 没有增加新行为，只是把前面已经推导出来的结构收束成 API。

构造函数会复制 `mode_workflows` 和 `handlers` 两张配置表，让 runner 拥有稳定配置。这里和 `context` 不一样：配置是 runner 的设置，`context` 是一次 run 的共享状态。

## 可运行示例

```python
events = []


def prepare(context):
    events.append(("prepare", dict(context)))
    context["prepared"] = True
    return {"prepared": True}


def enrich(context):
    events.append(("enrich", dict(context)))
    return {"saw_prepared": context["prepared"]}


runner = PipelineRunner(
    mode_workflows={"standard": ["prepare", "enrich"]},
    handlers={"prepare": prepare, "enrich": enrich},
)

context = {"file_id": "file-1"}
record = runner.trigger("standard", context)

assert record.status == "success"
assert record.mode == "standard"
assert record.workflows == ["prepare", "enrich"]
assert record.context is context
assert context == {"file_id": "file-1", "prepared": True}
assert [step.name for step in record.steps] == ["prepare", "enrich"]
assert [step.status for step in record.steps] == ["success", "success"]
assert record.steps[1].result == {"saw_prepared": True}
assert events == [
    ("prepare", {"file_id": "file-1"}),
    ("enrich", {"file_id": "file-1", "prepared": True}),
]
```

失败路径也要被锁住：

```python
def failing_enrich(_context):
    raise RuntimeError("enrich failed")


failing_runner = PipelineRunner(
    mode_workflows={"standard": ["prepare", "enrich"]},
    handlers={"prepare": prepare, "enrich": failing_enrich},
)
failed = failing_runner.trigger("standard", {"file_id": "file-1"})

assert failed.status == "failed"
assert [step.status for step in failed.steps] == ["success", "failed"]
assert failed.steps[1].error == "enrich failed"
```

unknown mode 必须在 handler 运行前失败：

```python
events = []


def prepare_for_unknown(_context):
    events.append("prepare")


unknown_runner = PipelineRunner(
    mode_workflows={"standard": ["prepare"]},
    handlers={"prepare": prepare_for_unknown},
)

try:
    unknown_runner.trigger("extract", {"file_id": "file-1"})
except ValueError as exc:
    assert str(exc) == "unknown mode: extract"
else:
    raise AssertionError("unknown mode should fail")

assert events == []
```

## 这个 v0 能做什么

现在我们有了一个最小顺序 pipeline runner：

```text
trigger(mode, context)
-> resolve workflow names
-> run handlers in order
-> record each executed step
-> stop after first failed step
-> return RunRecord
```

它能保证：

- known mode 按配置顺序运行
- unknown mode 不运行任何 handler
- 每个执行过的 workflow 都有 `StepRecord`
- handler 共享同一个 `context`
- step 失败会记录错误并停止后续步骤

它刻意不包含：

- `run_id`
- 持久化
- 队列
- resume / cancel
- DAG / graph execution
- 并行 ready wave
- API projection

这些不是永远不需要，而是 v0 的压力还没有要求它们。

## 常见问题

### 为什么不直接写一个 `process_file()`？

如果只需要最终结果，就应该直接写 `process_file()`。

runner 的价值在于调用方要观察执行过程：run status、step status、step result、step error。如果不需要这些，pipeline 是额外机器。

### 为什么要用 `context`？

因为 generic runner 不应该理解每对 step 之间的特殊入参。

`prepare(file_id)` 和 `enrich(prepared_file)` 很清楚，但它把 handoff 固定在调用代码里。统一成 `handler(context)` 后，runner 只负责执行策略，业务状态通过同一个 `context` 传递。

### 为什么 `context` 不复制？

因为它是一次 run 的共享状态。`prepare` 写入的内容要被 `enrich` 读到，也要被调用方观察到。

配置表可以复制，`context` 不应该复制。

### 为什么 `result` 放在 `StepRecord` 上？

因为每个 step 都可能有自己的返回值。v0 的 run 不一定有唯一最终返回值，所以整个 `RunRecord` 只记录整体状态和所有 step。

如果你的业务只想通过 `context` 传结果，也可以去掉 `StepRecord.result`。那是一个更极简的变体。

### 这个和 DAG / workflow engine 有什么区别？

这个 v0 只能表达线性顺序：

```text
prepare -> enrich
```

它不能表达：

```text
prepare
  -> enrich
  -> extract
```

也就是说，它还不能表示两个 downstream step 在同一个 dependency 成功后同时 ready。这个问题属于 v1：把 workflow list 替换成 dependency graph。

## 验证方式

源码仓库里已经有测试，可以这样跑：

```bash
git clone https://github.com/shio-chan-dev/nano-pipeline-runner.git
cd nano-pipeline-runner
PYTHONPATH=src python3 -m unittest discover -s tests
```

v0 至少应该覆盖：

- happy path：`standard -> prepare -> enrich` 按顺序运行
- shared context：`record.context is context`
- step records：成功 step 都可见
- failure path：`enrich` 失败会记录 failed step 并标记 run failed
- invalid mode：unknown mode 在任何 handler 运行前失败

## 小结

这篇文章推导出来的不是一个完整 workflow engine，而是一个最小 pipeline runner。

它的核心判断是：

```text
如果只要最终结果，用普通函数。
如果执行过程也要变成可观察数据，再引入 runner。
```

v0 的价值是把顺序执行、失败边界、共享上下文和 step observation 固定下来。下一步才是 v1：当 workflow list 无法表达 chain / parallel 的不同依赖形状时，再引入 graph plan 和 ready wave。

源码在这里：<https://github.com/shio-chan-dev/nano-pipeline-runner>。
