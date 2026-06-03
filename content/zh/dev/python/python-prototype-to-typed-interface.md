---
title: "Python 从 dict/list 原型到稳定接口：什么时候该整理类型？"
date: 2026-06-03
draft: false
---

# Python 从 dict/list 原型到稳定接口：什么时候该整理类型？

**副标题：** 验证想法时先用普通变量和 `dict/list` 跑通行为；模型稳定后，再把它们整理成函数参数、dataclass、helper 和类型标注。

**标签：** Python / 原型验证 / 类型标注 / 接口设计 / Pipeline
**适读人群：** 正在从脚本式代码过渡到可维护模块的 Python 开发者
**阅读时间：** 7 min

---

## 背景：为什么一开始不用把类型设计得很完整？

写一个新功能时，经常会卡在这个问题上：

```text
我应该一开始就写 dataclass、class、Mapping、Callable 吗？
还是先用普通 dict/list 把行为跑通？
```

答案不是固定的，但有一个很实用的判断：

```text
模型还不确定时，先用 dict/list 验证行为；
模型稳定后，再把稳定的东西整理成接口。
```

这不是偷懒，而是在降低错误抽象的成本。

比如你想验证一个 graph pipeline runner，最早可以只写：

```python
graph_plans = {
    "parallel": {
        "prepare": (),
        "enrich": ("prepare",),
        "extract": ("prepare",),
    }
}
```

这个变量一开始不是“最终架构”，只是为了回答一个问题：

```text
dependency map 能不能表达 prepare 之后 enrich 和 extract 同时 ready？
```

如果这个问题都没验证清楚，提前设计完整类层次反而会遮住真正的行为压力。

---

## 第一阶段：用普通变量验证模型

最早的脚本里，很多东西看起来像“常量”：

```python
graph_plans = {
    "parallel": {
        "prepare": (),
        "enrich": ("prepare",),
        "extract": ("prepare",),
    }
}

handlers = {
    "prepare": prepare,
    "enrich": enrich,
    "extract": extract,
}

context = {"file_id": "file-1"}
```

它们的作用不是为了长期存在，而是帮助你验证核心行为：

- `graph_plans` 能不能表达依赖关系？
- `handlers` 能不能把 workflow name 解析成函数？
- `context` 能不能承载一次运行的共享状态？
- `ready_nodes(record)` 能不能算出下一波可运行节点？
- 失败传播能不能把下游节点标记为 skipped？

这一阶段最重要的是让代码跑起来，并用断言锁住行为：

```python
record = trigger_graph("parallel", {"file_id": "file-1"})

assert record.status == "success"
assert record.waves == [["prepare"], ["enrich", "extract"]]
```

如果断言能说明关键行为，原型就有价值。

---

## 第二阶段：普通变量会暴露稳定输入

当行为验证通过后，你会发现某些变量反复出现，而且角色越来越清楚。

例如：

```python
graph_plans
handlers
mode
context
dependencies
record
failed_node
```

这时它们不再只是脚本变量，而是在暴露未来接口。

可以这样对应：

| 原型里的变量 | 稳定后的接口位置 |
| --- | --- |
| `mode` | `trigger(mode, context)` |
| `context` | `trigger(mode, context)` |
| `graph_plans` | `GraphPipelineRunner(graph_plans=...)` |
| `handlers` | `GraphPipelineRunner(handlers=...)` |
| `dependencies` | `_validate_graph(dependencies)` |
| `record` | `_ready_nodes(record)` |
| `failed_node` | `_skip_dependents(record, failed_node=...)` |

这就是一个很重要的信号：

> 原型里的稳定变量，往往就是未来函数或类的参数。

所以类型标注不是凭空设计出来的。它是在原型验证后，对稳定输入做命名和约束。

---

## 第三阶段：把行为整理成 helper

原型里最早可能只有一段脚本：

```python
for name, node in record.nodes.items():
    if node.status != "queued":
        continue
    if all(
        record.nodes[dependency].status == "success"
        for dependency in node.dependencies
    ):
        ready.append(name)
```

当这段逻辑变成稳定规则后，就可以提成 helper：

```python
def ready_nodes(record):
    ready = []
    for name, node in record.nodes.items():
        if node.status != "queued":
            continue
        if all(
            record.nodes[dependency].status == "success"
            for dependency in node.dependencies
        ):
            ready.append(name)
    return ready
```

提取 helper 的判断标准不是“代码长了”，而是：

```text
这段逻辑已经有稳定名字和稳定职责。
```

比如 `ready_nodes(record)` 的职责就很明确：

```text
从当前 run record 中找出所有 queued 且依赖已成功的节点。
```

这比把所有逻辑都塞进 `trigger_graph` 更容易读，也更容易测试。

---

## 第四阶段：把状态整理成 dataclass

刚开始你也可以用普通 dict 表示节点状态：

```python
node = {
    "name": "enrich",
    "dependencies": ["prepare"],
    "status": "queued",
    "result": None,
    "error": None,
}
```

但当字段稳定后，`dataclass` 会更合适：

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class GraphNodeRecord:
    name: str
    dependencies: list[str]
    status: str = "queued"
    result: Any = None
    error: str | None = None
```

这个变化的意义是：

- `name` 和 `dependencies` 是构造时必须提供的事实。
- `status`、`result`、`error` 是运行中变化的状态。
- 字段结构稳定后，用 dataclass 比裸 dict 更清楚。

也就是说，dataclass 不是为了“高级”，而是为了把已经稳定的状态模型写清楚。

---

## 第五阶段：把脚本输入整理成类的参数

当行为、状态和 helper 都稳定后，就可以把脚本整理成类：

```python
from collections.abc import Callable, Mapping
from typing import Any

WorkflowHandler = Callable[[dict[str, Any]], Any]


class GraphPipelineRunner:
    def __init__(
        self,
        *,
        graph_plans: Mapping[str, Mapping[str, tuple[str, ...]]],
        handlers: Mapping[str, WorkflowHandler],
    ) -> None:
        self._graph_plans = dict(graph_plans)
        self._handlers = dict(handlers)
```

这时你会发现，前面那些普通变量已经自然变成了构造参数：

```python
runner = GraphPipelineRunner(
    graph_plans=graph_plans,
    handlers=handlers,
)
```

这一步不是“重写”，而是把已经验证过的模型装进稳定边界：

```text
graph_plans 是配置输入；
handlers 是 workflow name 到函数的表；
trigger(mode, context) 是一次运行入口。
```

类型标注的作用也变清楚了：

```text
Mapping 表示只读配置输入；
WorkflowHandler 表示 step 函数形状；
dict[str, Any] 表示共享运行上下文。
```

---

## 那生产代码是不是也应该先从 dict/list 开始？

不一定。

如果是在探索模型、写教程、做小实验，先用普通结构很合理：

```text
dict/list 原型
-> 行为断言
-> helper
-> dataclass
-> runner class
-> 类型标注
```

但如果边界已经很明确，比如：

- 这是公开 API。
- 要跨多个模块复用。
- 要持久化。
- 要给别人调用。
- 已经有稳定业务协议。

那就可以更早使用：

- `dataclass`
- `TypedDict`
- `Protocol`
- `Mapping`
- `Callable` type alias

因为这时你不是在猜模型，而是在保护接口。

所以原则不是“永远先裸 dict”，而是：

```text
不确定模型时，先验证行为；
边界稳定时，尽早表达接口。
```

---

## 常见误区

### 误区 1：一开始不用类型，就是代码不专业

不是。原型阶段的目标是验证模型，过早整理类型可能会把错误模型固定下来。

### 误区 2：跑通后不用再整理

也不对。原型跑通只说明行为可行，不代表接口清楚。稳定后要把输入、状态和 helper 命名出来。

### 误区 3：类型标注是为了让代码变复杂

类型标注应该降低理解成本。如果一个类型标注让读者更困惑，要么时机太早，要么抽象层级不对。

### 误区 4：脚本里的常量和最终接口没有关系

很多最终接口都是从脚本变量里长出来的。反复出现、职责清楚、被多个函数需要的变量，通常就是未来参数。

---

## 小结

一个健康的演进过程通常是：

```text
用 dict/list 验证想法
-> 用断言锁住行为
-> 给稳定逻辑起 helper 名字
-> 给稳定状态建 dataclass
-> 把稳定输入收敛成函数/类参数
-> 用类型标注表达接口能力
```

一句话记忆：

> 先证明模型能工作，再给稳定的东西命名；类型不是起点，而是稳定接口的表达。
