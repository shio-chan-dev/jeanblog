---
title: "函数命名不是取名字：它是在描述系统边界"
date: 2026-06-03
draft: false
description: "用一个小型 graph pipeline runner 解释 get、resolve、validate、ready、skip 等函数命名如何表达行为、失败语义和副作用。"
tags: ["Python", "命名", "可读性", "接口设计", "Pipeline"]
categories: ["Python"]
keywords: ["Python 函数命名", "代码可读性", "接口设计", "resolve 命名", "pipeline runner"]
---

# 函数命名不是取名字：它是在描述系统边界

**副标题：** 好的函数名不只是“好听”。它应该让读者看出：这个函数在做读取、
解析、校验、筛选、创建，还是修改状态。

**适读人群：** 正在写 Python service、runner、pipeline、SDK 或业务模块的开发者
**阅读时间：** 8 min

---

## 背景：为什么 `get_dependencies` 不是总比 `resolve_graph` 直观？

看一个小型 graph pipeline runner：

```python
def trigger(self, mode, context):
    dependencies = self._resolve_graph(mode)
    self._validate_graph(dependencies)
    record = self._create_graph_record(mode, dependencies, context)

    while True:
        ready_nodes = self._ready_nodes(record)
        if not ready_nodes:
            break

        for node_name in ready_nodes:
            ...
```

刚开始可能会问：

```text
为什么不叫 get_dependencies？
为什么不叫 get_handler？
为什么 ready_nodes 没有动词？
为什么 skip_dependents 一看就像会改状态？
```

这些不是风格洁癖。函数命名背后其实是在表达系统边界：

```text
mode 是外部语言，graph 是内部执行结构。
handler name 是配置语言，handler 是可调用对象。
ready node 是运行时状态，不是普通列表。
skip dependents 是副作用，不是查询。
```

这篇文章用一个 pipeline runner 作为例子，讲清楚函数命名应该如何暴露设计意图。

---

## 核心观点：函数名应该说明“行为语义”

一个函数名至少应该回答其中一个问题：

```text
它是在读取，还是解析？
它会不会失败？
它会不会修改传入对象？
它返回的是 bool，还是集合？
它处理的是上游 dependency，还是下游 dependent？
```

所以函数命名不是简单地把变量名套进动词：

```python
def get_graph(...): ...
def process_data(...): ...
def handle_items(...): ...
```

这些名字能跑，但语义弱。更好的名字会把行为说清楚：

```python
def resolve_graph(mode): ...
def validate_graph(dependencies): ...
def ready_nodes(record): ...
def mark_dependents_skipped(record, failed_node): ...
```

---

## `get` 和 `resolve` 的区别

`get` 通常表示普通读取：

```python
def get_status(record):
    return record.status
```

它给人的预期是：

```text
这里大概率只是读一个字段或从容器里拿一个值。
```

但在 pipeline runner 里，`mode` 不是普通 key。它是调用方给出的业务语言：

```python
mode = "parallel"
```

runner 要把它转换成内部执行图：

```python
{
    "prepare": (),
    "enrich": ("prepare",),
    "extract": ("prepare",),
}
```

所以这个函数更适合叫：

```python
def resolve_graph(self, mode):
    dependencies = self._graph_plans.get(mode)
    if dependencies is None:
        raise ValueError(f"unknown mode: {mode}")
    return dependencies
```

`resolve_graph` 这个名字表达了三件事：

```text
输入是外部名字 mode。
输出是内部 graph dependencies。
找不到会失败。
```

如果叫 `get_dependencies`，读者只能看到“拿 dependencies”，看不出这里有外部语言到内部结构的解析边界。

同理：

```python
def resolve_handler(self, node_name):
    handler = self._handlers.get(node_name)
    if handler is None:
        raise ValueError(f"missing handler: {node_name}")
    return handler
```

这也不是单纯 `get_handler`。它表达的是：

```text
node name -> callable handler
如果配置缺失，执行不能继续。
```

---

## `validate` 表达“检查，不修改”

图执行前必须检查结构：

```python
def validate_graph(dependencies):
    for name, depends_on in dependencies.items():
        for dependency in depends_on:
            if dependency not in dependencies:
                raise ValueError(f"unknown dependency for {name}: {dependency}")

    visiting = set()
    visited = set()

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"cycle detected at {name}")
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in dependencies:
        visit(name)
```

`validate_graph` 这个名字给读者的预期是：

```text
它检查输入是否有效。
它不运行 handler。
它不创建 run record。
它不修改 graph。
不合法就抛错。
```

这比 `check_graph` 更明确一点，也比 `prepare_graph` 更安全。`prepare_graph`
可能暗示它会补全、转换或修改 graph。

命名里只要带了 `validate`，最好保持这个边界：

```text
只检查，不修复。
```

如果函数会修复或创建缺失内容，名字更接近：

```python
ensure_graph(...)
normalize_graph(...)
```

---

## 返回 bool 的函数像一个问题

如果函数返回 `bool`，名字最好像一个问题：

```python
def is_ready(node): ...
def has_failed_dependency(node): ...
def can_run(node): ...
```

这样的调用读起来很自然：

```python
if has_failed_dependency(node):
    ...
```

不要写得像数据名：

```python
def failed_dependency(node): ...
```

读者会犹豫：

```text
它返回 failed dependency 对象？
还是返回 True/False？
```

如果你确实返回对象或列表，就把名字写成对象或集合：

```python
def failed_dependencies(node): ...
```

---

## 返回集合时，可以用名词短语

不是所有函数都必须硬塞动词。

比如：

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

这个函数返回的是当前 ready 的 node names：

```python
["prepare"]
```

或者：

```python
["enrich", "extract"]
```

叫 `ready_nodes` 比 `get_ready_nodes` 更干净。因为这里的重点不是“get 这个动作”，而是返回值本身：

```text
当前 ready 的节点集合。
```

如果想更精确，也可以叫：

```python
ready_node_names(record)
```

因为它返回的是 name，不是 `GraphNodeRecord` 对象。

这个命名上的小区别很实用。下面这段代码：

```python
for node_name in ready_nodes(record):
    node = record.nodes[node_name]
```

如果函数实际返回 node 对象，就会出错。所以名字最好和返回值形状一致。

---

## 有副作用的函数，名字必须带动作

看这个失败传播函数：

```python
def skip_dependents(record, failed_node):
    changed = True
    while changed:
        changed = False
        for node in record.nodes.values():
            if node.status != "queued":
                continue
            if any(
                record.nodes[dependency].status in {"failed", "skipped"}
                for dependency in node.dependencies
            ):
                node.status = "skipped"
                node.error = f"dependency failed: {failed_node}"
                changed = True
```

它不是查询 dependent。它会修改 `record`：

```text
queued -> skipped
error  -> dependency failed: ...
```

所以不能叫：

```python
dependents(record, failed_node)
```

更好的名字是：

```python
skip_dependents(record, failed_node)
```

如果想让副作用更明显，可以再直接一点：

```python
mark_dependents_skipped(record, failed_node)
```

这类名字虽然长，但读者一眼就知道：

```text
这个函数会改状态。
```

命名原则很简单：

```text
查询函数可以像名词。
修改函数必须像动作。
```

---

## `dependency` 和 `dependent` 不能混

图代码里最容易混的是这两个词：

```text
dependency = 我依赖的上游
dependent  = 依赖我的下游
```

在这个 graph plan 里：

```python
graph_plans = {
    "parallel": {
        "prepare": (),
        "enrich": ("prepare",),
        "extract": ("prepare",),
    },
}
```

对 `enrich` 来说：

```text
prepare 是 dependency
enrich 是 prepare 的 dependent
```

所以字段名应该是：

```python
node.dependencies
```

而失败传播函数应该叫：

```python
skip_dependents(record, failed_node)
```

因为失败节点影响的是它的下游节点。

如果把这两个词混用，代码很快会变难读：

```python
for dependent in node.dependencies:
    ...
```

这句在语义上就是错的。读者不知道你到底在看上游还是下游。

---

## `create`、`build`、`project` 分别在说什么？

这些动词也常见，但语气不同。

### `create`

`create` 更像创建一个运行时对象：

```python
def create_graph_record(mode, dependencies, context):
    return GraphRunRecord(
        mode=mode,
        context=context,
        nodes={
            name: GraphNodeRecord(name=name, dependencies=list(depends_on))
            for name, depends_on in dependencies.items()
        },
        status="running",
    )
```

它的重点是：

```text
创建一个新的 record。
```

### `build`

`build` 更像从已有数据组装一个派生结构：

```python
def build_dependents_index(dependencies):
    dependents = {name: [] for name in dependencies}
    for name, depends_on in dependencies.items():
        for dependency in depends_on:
            dependents[dependency].append(name)
    return dependents
```

这里不是单纯创建对象，而是在根据 dependencies 组装反向索引。

### `project`

`project` 常用于把内部状态投影成外部视图：

```python
def project_run_status(record):
    return {
        "mode": record.mode,
        "status": record.status,
        "waves": record.waves,
    }
```

它表达的是：

```text
内部 record -> 对外响应/展示视图。
```

如果写成 `get_response`，语义就弱很多。

---

## 让主流程读起来像业务流程

好的 helper 命名最终服务于主流程。

目标不是把所有函数都写短，而是让核心入口读起来像一条清楚的执行叙事：

```python
def trigger(self, mode, context):
    dependencies = self._resolve_graph(mode)
    self._validate_graph(dependencies)
    record = self._create_graph_record(mode, dependencies, context)

    while True:
        ready_nodes = self._ready_nodes(record)
        if not ready_nodes:
            break

        record.waves.append(ready_nodes)

        for node_name in ready_nodes:
            node = record.nodes[node_name]
            node.status = "running"
            handler = self._resolve_handler(node_name)
            try:
                node.result = handler(record.context)
            except Exception as exc:
                node.status = "failed"
                node.error = str(exc)
                self._mark_dependents_skipped(record, failed_node=node_name)
                continue
            node.status = "success"

    self._project_final_status(record)
    return record
```

这段代码里的名字不是装饰。它们在告诉读者：

```text
resolve: 外部名字转内部对象，可能失败
validate: 检查结构，不执行
create: 创建运行记录
ready: 当前状态下可以运行
resolve_handler: 名字转 callable
mark: 修改下游状态
project: 根据节点状态得出 run status
```

这就是命名的真正价值：降低读者在脑子里补全语义的成本。

---

## 常见问题

### 1. 函数名是不是都应该用动词？

多数函数应该偏动词或动词短语，因为函数代表行为。

但返回集合的函数可以是名词短语：

```python
ready_nodes(record)
failed_nodes(record)
downstream_nodes(graph, node)
```

返回 bool 的函数则更适合问题式命名：

```python
is_ready(node)
has_failed_dependency(node)
can_run(node)
```

### 2. `get_` 是不是不能用？

不是。真正只是读取字段或容器值时，`get_` 可以用。

但如果函数包含解析、失败语义、转换边界，就不要用太轻的 `get_`：

```python
resolve_graph(mode)
resolve_handler(node_name)
```

会比：

```python
get_graph(mode)
get_handler(node_name)
```

更准确。

### 3. 名字长一点是不是不好？

不一定。名字长但准确，通常比短但含糊更好。

比如：

```python
mark_dependents_skipped(record, failed_node)
```

比：

```python
update(record)
```

长很多，但读者不需要打开函数体就知道它在做什么。

---

## 最佳实践

- 只是读字段或容器值，可以用 `get_`。
- 外部名字解析成内部对象，用 `resolve_`。
- 检查输入是否合法，用 `validate_`。
- 返回 bool，用 `is_`、`has_`、`can_`。
- 返回集合，可以用清楚的名词短语。
- 会修改状态，名字里必须出现动作。
- 区分 `dependency` 和 `dependent`。
- 让主入口函数读起来像业务流程。
- 不要滥用 `Manager`、`Processor`、`Helper`、`Util` 这类泛名。

---

## 小结

函数命名不是最后的润色，而是设计的一部分。

好的名字能告诉读者：

```text
这个函数依赖什么输入？
它做什么决定？
它会不会失败？
它会不会修改状态？
它返回的是状态、集合、对象，还是外部视图？
```

写命名时可以问自己一句：

```text
读者只看这个名字，能不能猜到它的边界和副作用？
```

如果不能，通常不是名字不够漂亮，而是语义还没有被说清楚。

## 参考与延伸阅读

- Python `collections.abc` 中的 `Mapping`、`Sequence`、`Callable`
- Clean Code 中关于命名的章节
- Code Complete 中关于子程序命名与抽象层次的讨论

## 行动号召

找一段你最近写过的业务代码，只看函数名，不看函数体。问三个问题：

```text
哪些函数会修改状态？
哪些函数会失败？
哪些函数只是查询？
```

如果答案不明显，先从命名开始重构。
