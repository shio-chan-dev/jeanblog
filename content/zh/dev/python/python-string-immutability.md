---
title: "Python 字符串为什么不可变：从 replace 到变量重绑定"
date: 2026-08-28
draft: false
description: "用可运行示例解释 Python str 的不可变性，区分字符串对象不变与变量重新赋值，并说明 replace、切片、+= 和循环拼接的真实行为。"
tags: ["Python", "字符串", "不可变对象", "str", "性能"]
categories: ["Python"]
keywords: ["Python 字符串不可变", "Python immutable string", "str replace", "变量重绑定", "字符串拼接"]
---

# Python 字符串为什么不可变：从 replace 到变量重绑定

**副标题：** `replace` 看起来修改了字符串，实际发生的是创建结果并让变量改指向它。理解这一区别，才能看懂字符串方法、别名行为和循环拼接的成本。

**适读人群：** 刚开始学习 Python 字符串、变量与对象关系的读者

**阅读时间：** 6 分钟

---

## 从一段括号消除代码说起

下面的函数反复删除成对的括号，直到字符串不再变化：

```python
def is_valid_by_elimination(s: str) -> bool:
    while True:
        previous = s
        s = s.replace("()", "").replace("[]", "").replace("{}", "")

        if len(s) == len(previous):
            return s == ""
```

这里容易产生一个疑问：

```python
previous = s
s = s.replace("()", "")
```

既然 `previous` 和 `s` 原来指向同一个字符串，第二行为什么不会同时改变
`previous`？

答案是：Python 字符串是不可变对象。`replace` 没有修改原字符串，而是计算出一个
结果，随后赋值语句让变量 `s` 改为指向这个结果。`previous` 仍然指向原字符串。

## “不可变”到底是什么意思

字符串不可变，指的是：

> 一个 `str` 对象创建以后，它所表示的字符序列不能被原地修改。

例如：

```python
text = "cat"
text[0] = "b"
```

Python 会抛出异常：

```text
TypeError: 'str' object does not support item assignment
```

不能把已有字符串中的 `c` 原地改成 `b`。如果需要 `"bat"`，必须得到另一个字符串值：

```python
text = "cat"
changed = "b" + text[1:]

assert text == "cat"
assert changed == "bat"
```

原来的 `text` 没有发生变化。

## 对象不变，不代表变量不能重新赋值

变量和对象需要分开理解：

```text
变量：保存对对象的引用，可以重新指向别的对象
对象：真正的数据；字符串对象自身不能被修改
```

看这个例子：

```python
text = "hello"
text = text.upper()

assert text == "HELLO"
```

表面上看，`text` 从小写变成了大写。实际过程是：

```text
1. text 指向字符串值 "hello"
2. text.upper() 计算出字符串值 "HELLO"
3. 赋值让 text 改为指向 "HELLO"
4. "hello" 本身从未被修改
```

因此，不可变限制的是字符串对象，不是变量名。

## replace 为什么不会修改原字符串

`str.replace` 返回替换后的结果。调用它本身不会改变原字符串：

```python
source = "red-green-red"
result = source.replace("red", "blue")

assert source == "red-green-red"
assert result == "blue-green-blue"
```

如果忽略返回值，看起来就会像是什么都没有发生：

```python
text = "hello"
text.replace("h", "H")

assert text == "hello"
```

要保留替换结果，需要接收返回值：

```python
text = text.replace("h", "H")

assert text == "Hello"
```

`upper`、`lower`、`strip`、切片和字符串拼接也遵循相同的值语义：它们不会原地改写
已有字符串。

## 回到 previous 和 s

把括号例子缩小：

```python
s = "()[]"
previous = s
s = s.replace("()", "")

assert previous == "()[]"
assert s == "[]"
```

赋值 `previous = s` 没有复制字符串内容。开始时，两个变量都引用相同的字符串值：

```text
previous ----+
             +----> "()[]"
s -----------+
```

执行 `s = s.replace("()", "")` 后，只有 `s` 被重新赋值：

```text
previous ----------> "()[]"
s -----------------> "[]"
```

所以后续可以比较新旧长度：

```python
if len(s) == len(previous):
```

如果长度相同，说明这一轮没有删除任何配对括号，继续循环也不会产生新结果。

## 字符串与列表有什么不同

列表是可变对象。多个变量引用同一个列表时，通过任意一个变量原地修改列表，其他
变量都能看到变化：

```python
letters = ["c", "a", "t"]
alias = letters
letters[0] = "b"

assert letters == ["b", "a", "t"]
assert alias == ["b", "a", "t"]
```

字符串不允许这种原地修改：

```python
word = "cat"
alias = word
word = "b" + word[1:]

assert word == "bat"
assert alias == "cat"
```

区别不在于“赋值是否有效”，而在于操作是修改原对象，还是让变量指向另一个值。

## `+=` 也不是原地修改字符串

下面的代码看起来像是在原字符串后追加内容：

```python
text = "Py"
alias = text
text += "thon"

assert text == "Python"
assert alias == "Py"
```

对于字符串，`+=` 的效果仍然是计算拼接结果并重新绑定 `text`。它不会把 `alias`
所引用的字符串原地扩展。

这一点和列表不同：

```python
items = [1, 2]
alias = items
items += [3]

assert items == [1, 2, 3]
assert alias == [1, 2, 3]
```

列表的 `+=` 会原地扩展现有列表，因此别名也能看到变化。

## 不可变性带来的实际影响

### 1. 字符串可以安全地作为字典键

字典键需要稳定的哈希值。字符串内容不能在放入字典后突然改变，因此可以作为键：

```python
counts = {"python": 3}

assert counts["python"] == 3
```

### 2. 多个变量共享字符串时不会互相修改

函数接收字符串后，不可能通过字符串操作原地改变调用方持有的那个值。函数只能返回
另一个字符串，由调用方决定是否接收。

### 3. 反复拼接可能产生额外成本

因为字符串不能原地增长，在循环中反复拼接较大的文本可能反复创建中间结果：

```python
result = ""
for part in parts:
    result += part
```

构建大量文本时，通常先收集片段，再一次合并：

```python
result = "".join(parts)
```

这不仅表达得更直接，也避免依赖解释器对连续字符串拼接的特定优化。

## 常见误区

### 误区一：赋值改变了字符串对象

```python
name = "Ada"
name = "Grace"
```

这里改变的是 `name` 的指向，不是把原来的 `"Ada"` 对象改写成 `"Grace"`。

### 误区二：字符串方法会自动保存结果

```python
name.strip()
```

如果没有接收返回值，`name` 仍然指向原来的字符串。通常应该写成：

```python
name = name.strip()
```

### 误区三：可以只靠 `id` 判断语义

Python 实现可能复用某些字符串对象，也可能让没有实际变化的操作返回原对象。因此，
不要把 `id` 是否变化当成字符串 API 的语义保证。真正稳定的规则是：任何字符串操作
都不能改变原字符串的字符内容。

## 完整可运行示例

```python
def is_valid_by_elimination(s: str) -> bool:
    while True:
        previous = s
        s = s.replace("()", "").replace("[]", "").replace("{}", "")

        if len(s) == len(previous):
            return s == ""


source = "red-green-red"
result = source.replace("red", "blue")

assert source == "red-green-red"
assert result == "blue-green-blue"

assert is_valid_by_elimination("{[()]}") is True
assert is_valid_by_elimination("([)]") is False
assert is_valid_by_elimination("") is True
```

## 小结

记住一句话即可：

> Python 字符串对象不能被原地修改，但变量随时可以重新指向另一个字符串。

因此：

- `replace`、`upper`、`strip`、切片和拼接产生结果，不修改原字符串；
- `s = s.replace(...)` 改变的是变量 `s` 的指向；
- 其他变量仍然可以保留对旧字符串的引用；
- 大量字符串片段通常使用 `"".join(parts)` 合并。

理解“对象不可变”和“变量重新绑定”的区别后，括号消除代码中的 `previous = s` 就不再
神秘：`previous` 留住旧值，`s` 接收本轮计算后的值，两者可以安全地进行比较。

## 参考与延伸阅读

- [Python 文本序列类型 str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)
- [Python 数据模型](https://docs.python.org/3/reference/datamodel.html)
