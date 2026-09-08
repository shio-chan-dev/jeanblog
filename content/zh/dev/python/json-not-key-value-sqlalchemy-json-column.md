---
title: "JSON 不只是键值对：Python 列表如何映射到 SQLAlchemy JSON 列"
subtitle: "从 tuple 领域值对象到 JSON 数组，弄清每一层的数据形状和转换边界"
date: 2026-09-08T10:30:00+08:00
draft: false
categories: ["Python"]
tags: ["Python", "JSON", "SQLAlchemy", "ORM", "dataclass"]
summary: "JSON 不只有键值对，还可以表示数组、字符串、数字和布尔值。本文用关键词订阅示例解释 Python tuple、list、dict 与 SQLAlchemy JSON 列之间的映射关系。"
description: "从一个关键词订阅的 Repository 示例出发，解释 JSON 对象与数组、Python list 与 tuple 的转换边界，以及 SQLAlchemy JSON 列的持久化和原地修改注意事项。"
keywords: ["JSON 数组", "SQLAlchemy JSON", "Python list tuple", "ORM 类型映射", "dataclass"]
readingTime: "约 8 分钟"
---

## 先说结论

JSON 不是“只能写键值对”的格式。键值对对应 JSON 的 **object**，而下面这些也都是合法 JSON：

```json
{"name": "张三"}
["变压器", "电缆"]
"变压器"
123
true
null
```

在 SQLAlchemy 中：

```python
keywords = Column(JSON, nullable=False)
```

表示这一列可以保存 JSON 值。给它一个 Python `list`，保存的就是 JSON 数组；给它一个 Python `dict`，保存的就是 JSON 对象。

本文要解释的转换链是：

```text
领域模型 tuple[str, ...]
        -> Python list[str]
        -> 数据库 JSON 数组
```

## 真实场景：领域模型为什么用 tuple？

假设关键词订阅的领域模型这样定义：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordCriteria:
    keywords: tuple[str, ...]
```

这里使用 `tuple` 有明确含义：关键词集合在一次业务操作中是不可变的。领域层不希望某个调用方悄悄执行：

```python
criteria.keywords.append("电缆")
```

但数据库里的 JSON 数组通常会由 Python `list` 表示，因为 JSON 标准中的数组就是有序元素序列。于是 Repository 需要负责一次明确的形状转换。

## JSON 的两种容易混淆的形状

### JSON object：键值对

```json
{
  "enabled": true,
  "items": ["变压器", "电缆"]
}
```

Python 中通常对应：

```python
{
    "enabled": True,
    "items": ["变压器", "电缆"],
}
```

它适合表达一个有命名字段的配置。

### JSON array：有序列表

```json
["变压器", "电缆"]
```

Python 中通常对应：

```python
["变压器", "电缆"]
```

它适合表达单纯的列表。当前 V1 的旧表只保存关键词，所以 `keywords` 是 JSON 数组；如果以后把启用状态和关键词放到一个配置列中，就可以改成 JSON object：

```python
keywords_config = {
    "enabled": True,
    "items": ["变压器", "电缆"],
}
```

两种形状都合法，关键是提前冻结列的结构，不要让同一列有时存数组、有时存对象。

## Repository 中的双向转换

写入数据库时，把领域层的元组转换成列表：

```python
row.keywords = list(subscription.criteria.keywords)
```

假设领域对象中的值是：

```python
("变压器", "电缆")
```

`list()` 得到：

```python
["变压器", "电缆"]
```

SQLAlchemy 的 `JSON` 类型会把这个 Python 列表交给 JSON 序列化器，数据库中得到：

```json
["变压器", "电缆"]
```

读取时做反向转换：

```python
criteria = KeywordCriteria(tuple(row.keywords))
```

完整的数据流是：

```text
写入：tuple -> list -> JSON array
读取：JSON array -> list -> tuple
```

这不是重复转换，而是两个边界各自使用最合适的数据结构：领域层保护不可变性，持久化层遵循 JSON 数组形状。

## 一个可以运行的最小例子

下面的例子使用 SQLite 临时数据库，演示 SQLAlchemy 如何保存和读取 JSON 数组：

```python
from dataclasses import dataclass

from sqlalchemy import JSON, Column, Integer, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session


@dataclass(frozen=True)
class KeywordCriteria:
    keywords: tuple[str, ...]


class Base(DeclarativeBase):
    pass


class KeywordSubscriptionORM(Base):
    __tablename__ = "keyword_subscription"

    id = Column(Integer, primary_key=True)
    keywords = Column(JSON, nullable=False)


engine = create_engine("sqlite+pysqlite:///:memory:")
Base.metadata.create_all(engine)

criteria = KeywordCriteria(("变压器", "电缆"))

with Session(engine) as session:
    row = KeywordSubscriptionORM(
        keywords=list(criteria.keywords),
    )
    session.add(row)
    session.commit()

with Session(engine) as session:
    row = session.scalar(select(KeywordSubscriptionORM))
    restored = KeywordCriteria(tuple(row.keywords))
    print(row.keywords)       # ['变压器', '电缆']
    print(restored.keywords)  # ('变压器', '电缆')
```

运行前安装 SQLAlchemy：

```bash
pip install sqlalchemy
python demo.py
```

注意：SQLite 的 JSON 支持与 PostgreSQL、MySQL 的原生 JSON 类型并不完全相同。这个例子只验证 Python 值与 ORM 映射，生产环境仍要按实际数据库测试查询和索引行为。

## 为什么不手动调用 `json.dumps()`？

下面这种写法通常是错误的：

```python
import json

row.keywords = json.dumps(["变压器", "电缆"])
```

这会把 Python 列表先变成一个 **字符串**：

```python
'["变压器", "电缆"]'
```

随后 SQLAlchemy 可能把这个字符串作为 JSON 字符串保存，而不是 JSON 数组。读取出来的值就可能是字符串，不能直接按列表处理，也会影响 JSON 路径查询。

正确做法是把 Python 容器直接交给 `JSON` 列：

```python
row.keywords = ["变压器", "电缆"]
```

SQLAlchemy 负责数据库驱动需要的序列化。只有在你要把数据发送到文件、日志或 HTTP 文本中时，才需要显式调用 `json.dumps()`。

## `JSON` 列与 `model_dump(mode="json")` 不是一回事

这两个操作都可能出现“转成 JSON”的说法，但边界不同：

```text
Pydantic model_dump(mode="json")
    把 API 模型转换成 JSON 友好的 Python 字典

SQLAlchemy JSON 列
    把 Python dict/list 等值交给数据库驱动保存为 JSON
```

例如 API 响应中：

```python
data.model_dump(mode="json")
```

主要是把 `datetime` 等值转换成可输出的字符串。Repository 中：

```python
row.keywords = list(criteria.keywords)
```

主要是把领域模型的值映射成持久化列所约定的数组形状。两者不应混为一层，也不需要为了“已经是 JSON”而重复编码。

## 一个容易踩到的坑：原地修改

下面的代码看起来合理：

```python
row.keywords.append("开关柜")
```

但普通的 SQLAlchemy `JSON` 类型不一定能可靠追踪嵌套列表的原地修改。更稳妥的写法是重新赋值：

```python
row.keywords = [*row.keywords, "开关柜"]
```

或者显式使用 SQLAlchemy 的可变类型扩展：

```python
from sqlalchemy.ext.mutable import MutableList

keywords = Column(MutableList.as_mutable(JSON), nullable=False)
```

两种方案不要混用成隐式约定。对于关键词配置这种规模很小、每次保存都要经过领域校验的值，先在领域层生成新元组，再整体替换 JSON 列，通常更简单：

```python
next_criteria = KeywordCriteria((*criteria.keywords, "开关柜"))
row.keywords = list(next_criteria.keywords)
```

## 最佳实践

1. 先确定 JSON 列是 object 还是 array，并固定结构。
2. 领域层可以使用 `tuple` 表达不可变值，Repository 显式转换成 `list`。
3. 不要把 `json.dumps()` 的结果再交给 SQLAlchemy `JSON` 列。
4. 读取 JSON 数组后转换回领域层需要的结构，不要让 JSON 形状泄露到领域规则中。
5. JSON 嵌套值尽量整体替换；如果需要原地修改，使用 `MutableList` 或 `MutableDict` 并补测试。
6. SQLite 临时测试通过，不代表 PostgreSQL 或 MySQL 的 JSON 查询、索引和约束已经验证。

## 小结

JSON 不只是键值对：键值对是 object，列表是 array，二者都是标准 JSON。

在关键词订阅这个例子中，最清晰的边界是：

```text
KeywordCriteria(tuple[str, ...])
    -> list[str]
    -> Column(JSON) 中的 JSON array
```

`list()` 的作用不是把“普通值变成 JSON 类型”，而是把领域层的不可变序列转换成持久化层约定的数组容器。只要把这个边界写清楚，JSON、ORM 和领域模型就不会互相污染。

## 参考与延伸阅读

- [Python `json` 标准库文档](https://docs.python.org/3/library/json.html)
- [SQLAlchemy JSON 类型文档](https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.JSON)
- [SQLAlchemy Mutable 扩展文档](https://docs.sqlalchemy.org/en/20/orm/extensions/mutable.html)
