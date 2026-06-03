---
title: "Python 类型标注里为什么用 Mapping、Sequence 这类抽象接口？"
date: 2026-06-03
draft: false
---

# Python 类型标注里为什么用 Mapping、Sequence 这类抽象接口？

**副标题：** 当一个函数只读取配置，不修改调用方传入的字典时，用
`Mapping` 比直接写 `dict` 更能表达接口意图；同理，`Iterable`、`Sequence`
和 `Callable` 也在表达“我需要什么能力”。

**标签：** Python / 类型标注 / Mapping / Sequence / Callable / 接口设计
**适读人群：** 正在写 Python 配置对象、runner、pipeline、SDK 参数的开发者
**阅读时间：** 7 min

---

## 背景：为什么不是所有字典参数都写成 dict？

写 Python 配置型代码时，经常会有这种参数：

```python
graph_plans = {
    "parallel": {
        "prepare": (),
        "enrich": ("prepare",),
        "extract": ("prepare",),
    }
}
```

如果一个 runner 只是读取它：

```python
runner = GraphPipelineRunner(graph_plans=graph_plans, handlers=handlers)
```

那么类型标注可以写成：

```python
from collections.abc import Mapping


def __init__(
    self,
    *,
    graph_plans: Mapping[str, Mapping[str, tuple[str, ...]]],
    handlers: Mapping[str, WorkflowHandler],
) -> None:
    ...
```

刚开始看会觉得复杂：为什么不直接写 `dict`？

核心答案是：

> `Mapping` 表达“我只需要按 key 读取这个对象”，`dict` 表达“我要求它就是一个
> dict”。

---

## 核心概念：Mapping、dict、MutableMapping 的区别

可以把三者这样理解：

| 类型 | 表达的接口意图 | 典型场景 |
| --- | --- | --- |
| `Mapping[K, V]` | 只读映射，只要求能按 key 取值、遍历 | 配置输入、只读参数、函数不负责修改 |
| `dict[K, V]` | 明确要求普通字典 | 需要 dict 的具体行为，或项目内简单脚本 |
| `MutableMapping[K, V]` | 可变映射，允许写入、删除 | 函数会修改调用方传入的映射 |

注意：`Mapping` 不是运行时冻结。它只是类型语义，告诉读者和类型检查器：

```text
这个函数不应该修改传入的映射。
```

如果调用方传入普通 dict，外部仍然可以修改它：

```python
plans = {"parallel": {"prepare": ()}}
runner = GraphPipelineRunner(graph_plans=plans, handlers={})

plans["new"] = {}
```

所以如果对象需要稳定配置，构造函数里通常会复制一份：

```python
self._graph_plans = dict(graph_plans)
```

这至少避免 runner 直接持有调用方传入的外层 dict。

---

## 例子：为什么 graph_plans 适合 Mapping？

看一个简化的 graph runner 构造函数：

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

拆开这个类型：

```python
Mapping[
    str,                          # mode: "parallel"
    Mapping[str, tuple[str, ...]]  # node -> dependencies
]
```

对应数据是：

```python
{
    "parallel": {
        "prepare": (),
        "enrich": ("prepare",),
        "extract": ("prepare",),
    }
}
```

这里 runner 对 `graph_plans` 的需求只有两个：

1. 根据 mode 查到 graph plan。
2. 根据 node name 查到 dependencies。

它不需要对调用方传进来的对象执行：

```python
graph_plans["new_mode"] = ...
del graph_plans["parallel"]
```

所以 `Mapping` 比 `dict` 更准确。

---

## 这类类型的共同目的：表达“需要的能力”

`Mapping` 不是孤立的。Python 标准库里还有一组类似的抽象集合类型：

```python
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
```

它们的共同目的不是让代码看起来更“高级”，而是把函数真正需要的能力说清楚。

### Iterable：我只需要遍历

```python
from collections.abc import Iterable


def total(values: Iterable[int]) -> int:
    return sum(values)
```

这个函数只做一件事：

```python
for value in values:
    ...
```

所以它可以接收 list、tuple、set、generator。它不承诺能索引，也不承诺能取长度。

### Sequence：我需要顺序、索引和长度

```python
from collections.abc import Sequence


def first(values: Sequence[int]) -> int:
    return values[0]
```

这里就不能只写 `Iterable`，因为函数需要 `values[0]`。`Sequence` 表达的是：

```text
这个对象有顺序，可以 len(...)，也可以按下标读取。
```

list 和 tuple 都是常见的 sequence。

### Mapping：我按 key 读，不修改

```python
from collections.abc import Mapping


def get_mode(plans: Mapping[str, list[str]], mode: str) -> list[str]:
    return plans[mode]
```

这个函数只依赖 key lookup，不需要写入：

```python
plans[mode]
```

所以 `Mapping` 比 `dict` 更能表达“只读映射输入”。

### MutableMapping：我会写入这个映射

```python
from collections.abc import MutableMapping


def mark_seen(cache: MutableMapping[str, bool], key: str) -> None:
    cache[key] = True
```

这里就不应该写 `Mapping`，因为函数明确会修改传入对象。

`MutableMapping` 告诉调用方：

```text
传进来的对象会被这个函数写入。
```

### Callable：我需要一个可调用对象

```python
from collections.abc import Callable
from typing import Any


def run(handler: Callable[[dict[str, Any]], object], context: dict[str, Any]):
    return handler(context)
```

这里参数不一定非得是普通函数。只要它能像函数一样被调用，并接受一个
`dict[str, Any]`，就满足接口。

### Iterator：我会消耗它

```python
from collections.abc import Iterator


def consume_one(items: Iterator[int]) -> int:
    return next(items)
```

`Iterator` 比 `Iterable` 更具体。它表示这个对象本身就是迭代器，调用
`next(...)` 会推进它的内部状态。读者看到这个类型，就应该意识到：

```text
这个参数可能会被消耗，读过的元素不会再回来。
```

这些类型的共同读法是：

```text
dict/list/function 是具体实现；
Mapping/Sequence/Callable 是能力要求。
```

---

## 为什么不是所有地方都应该用 Mapping？

`Mapping` 是接口设计上的表达，不是越多越好。

如果你写的是学习代码、小脚本，或者团队更重视直观，直接写 `dict` 完全可以：

```python
graph_plans: dict[str, dict[str, tuple[str, ...]]]
```

它的好处是阅读成本低。尤其是初学阶段，`dict` 比 `Mapping` 更容易理解。

但当代码变成公共接口、库、runner、pipeline 配置入口时，`Mapping` 更适合，因为它表达的是能力，而不是具体实现：

```text
你给我任何像 dict 一样可读取的对象都行；
我不会依赖它必须是 dict。
```

---

## 实践建议

可以按这个规则选：

```text
只读取 key/value，不修改传入对象 -> Mapping
要修改这个对象 -> MutableMapping 或 dict
明确需要普通 dict 行为 -> dict
初学教程为了降低理解成本 -> dict 也可以
```

更一般地说，函数参数应该尽量标注它真正需要的最小能力：

```text
只遍历 -> Iterable
要索引和长度 -> Sequence
按 key 读取 -> Mapping
要写入 key/value -> MutableMapping
要调用 -> Callable
要消耗 next(...) -> Iterator
```

再加一条配置类代码里的经验：

```python
def __init__(self, config: Mapping[str, str]) -> None:
    self._config = dict(config)
```

参数用 `Mapping` 表达“只读输入”，内部用 `dict(...)` 复制成自己的状态。这样调用方传什么可读映射都可以，runner 也不会直接依赖调用方的外层对象。

---

## 常见误区

### 误区 1：Mapping 会让 dict 真的不可变

不会。`Mapping` 是类型标注，不会改变对象运行时行为。

```python
data: Mapping[str, int] = {"a": 1}
```

这只是告诉类型检查器和读代码的人：这里应该按只读方式使用。

### 误区 2：用了 Mapping 就不需要复制

不一定。`Mapping` 只表达接口语义。如果你需要 runner 的配置稳定，仍然应该复制：

```python
self._graph_plans = dict(graph_plans)
```

如果内部嵌套结构也可能被外部修改，还需要更深层的复制或冻结。不过不要一开始就复杂化，先看是否真的有这个风险。

### 误区 3：Mapping 一定比 dict 高级

不是。`Mapping` 更抽象，但不一定更适合所有场景。教程、脚本、短代码里用 `dict` 可能更清楚。

---

## 小结

`Mapping` 最适合用在“配置输入”这类场景：

```text
调用方提供一份 key/value 配置；
当前对象只读取它；
当前对象不应该修改调用方的数据。
```

一句话记忆：

> 参数只读，用 `Mapping` 表达接口意图；内部要稳定，就复制成自己的
> `dict`。
