---
title: "Python Callable 类型标注：怎么描述一个 handler 函数？"
date: 2026-06-03
draft: false
---

# Python Callable 类型标注：怎么描述一个 handler 函数？

**副标题：** `Callable[[dict[str, Any]], Any]` 不是一个新类，而是在类型层面说明：
这个参数是一个“能被调用的对象”，它接收什么参数，返回什么结果。

**标签：** Python / 类型标注 / Callable / Type Alias / Pipeline
**适读人群：** 正在写回调、handler、pipeline step、插件机制的 Python 开发者
**阅读时间：** 6 min

---

## 背景：为什么 handler 类型看起来这么长？

在 pipeline runner 里，经常会有这种 handler 表：

```python
handlers = {
    "prepare": prepare,
    "enrich": enrich,
}
```

每个 handler 都接收同一个运行时上下文：

```python
def prepare(context: dict[str, Any]) -> dict[str, bool]:
    context["prepared"] = True
    return {"prepared": True}
```

如果直接给 handler 表写类型，会变成：

```python
from collections.abc import Callable, Mapping
from typing import Any

handlers: Mapping[str, Callable[[dict[str, Any]], Any]]
```

这一串不难，但读起来吵。它把“workflow name 到 workflow handler 的映射”写成了一段类型语法。

所以常见写法是先起一个类型别名：

```python
WorkflowHandler = Callable[[dict[str, Any]], Any]
```

然后再写：

```python
handlers: Mapping[str, WorkflowHandler]
```

这篇文章就讲清楚：`Callable[[dict[str, Any]], Any]` 到底是什么意思，以及为什么要给它起名。

---

## Callable 的语法：参数列表在前，返回值在后

`Callable` 的基本形状是：

```python
Callable[[参数类型列表], 返回值类型]
```

所以：

```python
Callable[[dict[str, Any]], Any]
```

表示：

```text
这是一个可以调用的对象；
它接收 1 个参数：dict[str, Any]；
它返回：Any。
```

对应函数形状是：

```python
def handler(context: dict[str, Any]) -> Any:
    ...
```

再看几个更小的例子：

```python
Callable[[int, int], int]
```

表示：

```python
def add(a: int, b: int) -> int:
    ...
```

```python
Callable[[str], bool]
```

表示：

```python
def is_valid(name: str) -> bool:
    ...
```

```python
Callable[[], None]
```

表示：

```python
def cleanup() -> None:
    ...
```

注意第一层方括号里还有一层方括号：

```python
Callable[[dict[str, Any]], Any]
```

里面的 `[dict[str, Any]]` 是参数类型列表。这里列表里只有一个元素，所以表示这个 callable 只接收一个参数。

---

## Type Alias：给函数形状起一个业务名字

这句：

```python
WorkflowHandler = Callable[[dict[str, Any]], Any]
```

是在定义类型别名。它不是新类，不会改变运行时行为。

它只是说：

```text
WorkflowHandler 这个名字，代表一种函数形状：
接收 context，返回任意结果。
```

不用别名时：

```python
class GraphPipelineRunner:
    def __init__(
        self,
        *,
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
    ) -> None:
        self._handlers = dict(handlers)
```

用别名后：

```python
class GraphPipelineRunner:
    def __init__(
        self,
        *,
        handlers: Mapping[str, WorkflowHandler],
    ) -> None:
        self._handlers = dict(handlers)
```

第二种读起来更接近业务语义：

```text
handlers 是 workflow name 到 workflow handler 的映射。
```

而不是每次都让读者解析一遍：

```python
Callable[[dict[str, Any]], Any]
```

---

## 在 pipeline runner 里的完整例子

```python
from collections.abc import Callable, Mapping
from typing import Any

WorkflowHandler = Callable[[dict[str, Any]], Any]


def prepare(context: dict[str, Any]) -> dict[str, bool]:
    context["prepared"] = True
    return {"prepared": True}


def enrich(context: dict[str, Any]) -> dict[str, bool]:
    return {"saw_prepared": context["prepared"]}


handlers: Mapping[str, WorkflowHandler] = {
    "prepare": prepare,
    "enrich": enrich,
}


def resolve_handler(name: str) -> WorkflowHandler:
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"missing handler: {name}")
    return handler
```

这里 `prepare` 和 `enrich` 都符合 `WorkflowHandler`：

- 它们都能被调用。
- 它们都接收一个 `dict[str, Any]`。
- 它们都返回某个值。

runner 调用时就很清楚：

```python
context = {"file_id": "file-1"}
handler = resolve_handler("prepare")
result = handler(context)

assert result == {"prepared": True}
assert context == {"file_id": "file-1", "prepared": True}
```

---

## Callable 描述的是“可调用对象”，不只是 def 函数

`Callable` 不要求对象一定是用 `def` 定义的普通函数。只要它能被调用，并且调用形状匹配，就可以。

普通函数可以：

```python
def prepare(context: dict[str, Any]) -> dict[str, bool]:
    context["prepared"] = True
    return {"prepared": True}
```

lambda 也可以：

```python
handler: WorkflowHandler = lambda context: {"file_id": context["file_id"]}
```

实现了 `__call__` 的对象也可以：

```python
class PrepareHandler:
    def __call__(self, context: dict[str, Any]) -> dict[str, bool]:
        context["prepared"] = True
        return {"prepared": True}


handler: WorkflowHandler = PrepareHandler()
```

所以 `Callable` 的重点不是“函数类型”，而是：

```text
这个对象能不能按指定参数和返回值形状被调用。
```

---

## 为什么返回值用 Any？

在这个 runner 例子里：

```python
WorkflowHandler = Callable[[dict[str, Any]], Any]
```

返回值写成 `Any`，是因为不同 step 可能产出不同结构：

```python
def prepare(context: dict[str, Any]) -> dict[str, bool]:
    ...


def extract(context: dict[str, Any]) -> dict[str, str]:
    ...
```

runner 只负责记录：

```python
node.result = handler(record.context)
```

它不解释每个业务结果的具体字段。结果具体是什么，由业务 handler 自己决定。

如果你的系统要求所有 handler 返回统一结构，也可以把 `Any` 换成更具体的类型：

```python
StepResult = dict[str, object]
WorkflowHandler = Callable[[dict[str, Any]], StepResult]
```

这会让接口更严格，但也会要求所有 handler 遵守同一种返回格式。

---

## 什么时候值得起类型别名？

不是所有 `Callable` 都需要别名。下面这种一次性参数，直接写也可以：

```python
def retry(operation: Callable[[], None]) -> None:
    operation()
```

但如果这个 callable 有明确业务角色，或者会在多个地方重复出现，就值得起名：

```python
WorkflowHandler = Callable[[dict[str, Any]], Any]
```

因为它表达的不只是类型，还有业务概念：

```text
这是 pipeline step 的处理函数。
```

常见适合起别名的场景：

- pipeline step handler
- event handler
- validation function
- callback
- middleware
- plugin hook

---

## 常见误区

### 误区 1：WorkflowHandler 是一个运行时基类

不是。它只是类型别名：

```python
WorkflowHandler = Callable[[dict[str, Any]], Any]
```

它不会创建新类，也不能拿来做这种判断：

```python
isinstance(prepare, WorkflowHandler)  # 不应该这样用
```

### 误区 2：Callable 会自动检查参数

不会。类型标注主要给读者和类型检查器看。Python 运行时不会因为你写了
`Callable[[dict[str, Any]], Any]` 就自动检查函数签名。

### 误区 3：Any 表示随便写都好

`Any` 表示 runner 不关心返回值的具体类型，不表示业务代码可以没有约定。如果某个系统需要稳定响应格式，就应该把返回值类型收紧。

---

## 小结

`Callable[[dict[str, Any]], Any]` 的读法是：

```text
一个可调用对象；
接收一个 dict[str, Any] 参数；
返回任意类型。
```

`WorkflowHandler = Callable[[dict[str, Any]], Any]` 的意义是给这个函数形状起一个业务名字：

```text
这是 workflow step 的 handler。
```

一句话记忆：

> `Callable` 描述函数形状，类型别名给这个形状一个业务名字。
