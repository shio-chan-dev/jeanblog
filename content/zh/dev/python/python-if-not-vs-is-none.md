---
title: "Python 中 if not value 与 is None 的区别"
date: 2026-09-08
draft: false
description: "从逐行解析文档的空行过滤代码出发，解释 Python 中 if not value 与 value is None 的语义差异、适用边界和常见误区。"
tags: ["Python", "真值判断", "None", "空字符串", "代码可读性"]
categories: ["Python"]
keywords: ["Python if not", "Python is None", "Python 空字符串判断", "Python 真值判断", "None 判断"]
---

# Python 中 if not value 与 is None 的区别

**副标题：** `if not line` 判断的是“这个值是否为假值”，`if line is None`
判断的是“这个值是否就是 `None`”。两者可能在某些输入上得到相同结果，但表达的业务语义并不相同。

**适读人群：** 正在学习 Python 条件判断，或需要处理文本、接口参数和可选字段的开发者

**阅读时间：** 6 分钟

---

## 从文档解析中的空行过滤说起

逐行解析文档时，经常会看到下面的代码：

```python
for raw_line in lines:
    line = raw_line.strip()
    if not line:
        continue

    # 继续处理非空内容
```

这里为什么使用：

```python
if not line:
```

而不是：

```python
if line is None:
```

关键不在于哪种写法更简短，而在于程序此时要识别的是**空字符串**，不是
**缺失值 `None`**。

## 先跟踪 line 是怎么产生的

这一段代码的执行顺序是：

```python
raw_line = "   "
line = raw_line.strip()
```

`str.strip()` 会删除字符串首尾的空白字符，并返回一个字符串：

```python
assert "  投标人资格要求  ".strip() == "投标人资格要求"
assert "   ".strip() == ""
assert "".strip() == ""
```

只包含空格的行经过 `strip()` 后会变成空字符串 `""`，不会变成 `None`。
因此后面的条件实际需要回答：

> `line` 是不是一个空字符串？

空字符串在 Python 的条件判断中属于假值，所以：

```python
assert not ""
```

`if not line` 条件成立，`continue` 跳过当前循环，空行便不会进入后续解析逻辑。

## `if not line` 判断的是真值

Python 会在 `if`、`while` 和布尔运算中对对象进行真值测试。常见的假值包括：

```python
assert not None
assert not False
assert not 0
assert not 0.0
assert not ""
assert not []
assert not {}
assert not set()
```

所以：

```python
if not value:
```

表达的是：

> 如果 `value` 是任意一种假值，就执行这个分支。

它不是专门判断 `None`，也不是专门判断空字符串。

在文档解析示例里，类型标注 `lines: list[str]` 和前面的 `strip()` 已经把
`line` 的类型限定为 `str`。对字符串而言，假值只有 `""`，因此
`if not line` 在这里可以准确表达“跳过空行”。

## `if line is None` 判断的是对象身份

`None` 是 Python 用来表示“没有值”或“值尚未提供”的单例对象。推荐使用
`is None` 判断一个变量是否正是这个对象：

```python
value = None

assert value is None
```

空字符串不是 `None`：

```python
line = ""

assert line is not None
assert not line
```

因此，把原代码改成下面这样会漏掉空行：

```python
line = "   ".strip()

if line is None:
    print("跳过")

# line 是 ""，条件不成立，程序不会跳过
```

`is None` 回答的是“值是否缺失”，而不是“字符串是否为空”。

## 空值与缺失值不是一回事

这两个判断的选择，本质上取决于业务上是否需要区分不同状态。

| 程序意图 | 推荐写法 |
| --- | --- |
| 跳过空字符串 | `if not line:` |
| 只接受严格的空字符串，且变量可能是其他类型 | `if line == "":` |
| 判断调用方是否没有提供值 | `if value is None:` |
| 区分 `None`、`""`、`0` 和 `False` | 分别显式判断 |
| 把所有假值统一视为“无内容” | `if not value:` |

例如，一个可选备注字段可能需要区分三种状态：

```python
note = None  # 没有提供备注
note = ""    # 明确提供了空备注
note = "待复核"  # 提供了实际内容
```

如果业务需要保留这种区别，就不能用一个 `if not note` 把前两种状态合并。

## 当输入既可能是 None，也可能是字符串

假设外部数据允许传入 `None`，类型应明确写成 `str | None`。此时必须先处理
`None`，再调用字符串方法：

```python
def normalize_line(raw_line: str | None) -> str | None:
    if raw_line is None:
        return None

    line = raw_line.strip()
    if not line:
        return None

    return line
```

顺序不能反过来：

```python
raw_line = None
line = raw_line.strip()  # AttributeError
```

这里的两个条件承担不同职责：

```text
raw_line is None：输入值不存在
not line：清理后的字符串为空
```

如果调用方需要区分“没有输入”和“输入了一行空白”，就不应该都返回 `None`，
而应为两种状态设计不同的返回结果。

## 最容易出现的问题：变量不只有字符串

`if not value` 很方便，但当变量可能包含多种类型时，它可能把有意义的数据一起过滤掉。

### 数字 0 可能是有效值

```python
quantity = 0

if not quantity:
    print("没有数量")
```

这段代码把合法的数量 `0` 当成了缺失值。如果 `None` 才代表“未填写”，应该写：

```python
if quantity is None:
    print("没有填写数量")
```

### False 可能是明确选择

```python
mandatory = False

if not mandatory:
    print("没有填写是否硬性要求")
```

这里无法区分“明确选择否”和“没有填写”。若字段类型是 `bool | None`，应显式判断：

```python
if mandatory is None:
    print("没有填写")
elif mandatory:
    print("硬性要求")
else:
    print("非硬性要求")
```

所以，使用 `if not value` 前应先问：

> 这个变量的所有假值，是否都可以被当成同一种状态？

只有答案为“是”时，合并判断才是安全的。

## 一个可运行的完整示例

下面的函数接收字符串列表，去掉首尾空白并过滤空行：

```python
def collect_non_empty_lines(lines: list[str]) -> list[str]:
    result: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        result.append(line)

    return result


sample_lines = [
    "第一标段\n",
    "   ",
    "  投标人须具备安全生产许可证  ",
    "",
]

parsed_lines = collect_non_empty_lines(sample_lines)

assert parsed_lines == [
    "第一标段",
    "投标人须具备安全生产许可证",
]

print(parsed_lines)
```

运行后输出：

```text
['第一标段', '投标人须具备安全生产许可证']
```

在这个函数中，`if not line` 是合适的，因为函数契约已经确定输入是
`list[str]`，而且业务目标就是过滤清理后的空字符串。

## 选择时记住两句话

```text
if not value：所有假值都进入分支。
if value is None：只有缺失值 None 进入分支。
```

不要根据代码长短选择，而要根据数据类型和业务语义选择。

对于本文开头的文档解析代码，推导过程很明确：

```text
raw_line 是字符串
    -> strip() 返回字符串
    -> 空白行变成 ""
    -> 当前目标是过滤空字符串
    -> 使用 if not line
```

如果以后允许输入 `None`，就应修改类型契约，并在调用 `strip()` 之前显式使用
`is None` 处理它，而不是简单替换原有条件。

## 参考资料

- [Python 文档：Truth Value Testing](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)
- [Python 文档：Comparisons](https://docs.python.org/3/reference/expressions.html#comparisons)
- [PEP 8：与 None 等单例比较](https://peps.python.org/pep-0008/#programming-recommendations)

可以尝试把完整示例中的输入类型改成 `list[str | None]`，先观察直接调用
`strip()` 时的错误，再为 `None` 和空字符串分别补上符合业务含义的处理分支。
