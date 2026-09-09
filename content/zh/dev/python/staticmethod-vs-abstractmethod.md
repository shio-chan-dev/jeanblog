---
title: "Python 中 @staticmethod 和 @abstractmethod 的区别"
date: 2026-09-09
draft: false
categories:
  - Python
tags:
  - Python
  - 面向对象
  - 设计
description: "理解 @staticmethod 与 @abstractmethod 解决的问题，掌握实例方法、静态方法、类方法和抽象方法的选择方式。"
---

在阅读 Repository 或 Service 代码时，经常会同时看到 `@staticmethod` 和 `@abstractmethod`。它们都写在方法上方，很容易被误认为是同一类装饰器。

实际上，两者解决的是不同问题：

```text
@staticmethod    解决方法如何被调用
@abstractmethod  约束子类必须提供什么行为
```

理解这个区别后，方法应该放在类中、模块中，还是抽象基类中，会清楚很多。

## `@staticmethod`：不依赖实例状态的方法

普通实例方法默认接收 `self`：

```python
class KeywordMatcher:
    def contains(self, text: str, keyword: str) -> bool:
        return keyword.casefold() in text.casefold()
```

如果方法体不读取或修改 `self`，可以写成静态方法：

```python
class KeywordMatcher:
    @staticmethod
    def contains(text: str, keyword: str) -> bool:
        return keyword.casefold() in text.casefold()
```

调用时不需要创建实例：

```python
KeywordMatcher.contains("变压器采购项目", "变压器")
```

静态方法没有自动传入的 `self`，因此不能访问实例属性：

```python
class KeywordMatcher:
    def __init__(self, case_sensitive: bool):
        self.case_sensitive = case_sensitive

    @staticmethod
    def contains(text: str, keyword: str) -> bool:
        # 这里不能访问 self.case_sensitive
        return keyword in text
```

如果函数只是通用计算，并不属于某个类的概念，直接定义为模块级函数通常更自然：

```python
def contains(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()
```

所以“不需要 `self`”是使用 `@staticmethod` 的重要条件，但还要问一句：这个行为是否真的属于该类？如果答案是否定的，就不必为了使用装饰器而把函数塞进类里。

## `@abstractmethod`：要求子类实现行为

抽象方法关注的不是调用形式，而是类的契约：

```python
from abc import ABC, abstractmethod


class KeywordSubscriptionRepository(ABC):
    @abstractmethod
    async def get_by_user(self, user_id: str):
        """根据用户读取关键词配置。"""
        ...
```

这个基类声明：任何具体 Repository 都必须提供 `get_by_user()`。

```python
class SqlKeywordSubscriptionRepository(KeywordSubscriptionRepository):
    async def get_by_user(self, user_id: str):
        # 这里实现真实数据库查询
        ...
```

如果子类没有实现抽象方法，就不能实例化：

```python
class IncompleteRepository(KeywordSubscriptionRepository):
    pass


IncompleteRepository()  # TypeError
```

`@abstractmethod` 的价值在于尽早暴露接口不完整的问题。它让基类成为一份可检查的契约，而不是只靠注释约定方法名称。

## 两者可以组合

如果一个行为既属于接口契约，又不应该依赖实例状态，可以组合使用：

```python
from abc import ABC, abstractmethod


class TextParser(ABC):
    @staticmethod
    @abstractmethod
    def supports(extension: str) -> bool:
        ...
```

子类必须提供一个静态的 `supports()`：

```python
class PdfParser(TextParser):
    @staticmethod
    def supports(extension: str) -> bool:
        return extension.lower() == ".pdf"
```

这种组合并不常见。只有当“这是类的接口”与“它不需要实例状态”同时成立时才使用。

## 四种方法怎么选

可以先问方法需要什么上下文：

| 方法形式 | 自动获得的对象 | 适合的情况 |
| --- | --- | --- |
| 实例方法 | `self` | 需要读取或修改实例状态 |
| 类方法 | `cls` | 需要访问类本身，或提供替代构造方法 |
| 静态方法 | 无 | 属于类的概念，但只依赖参数 |
| 抽象方法 | 取决于组合形式 | 要求具体子类实现某项行为 |

注意，`@abstractmethod` 不是与实例方法、静态方法、类方法并列的调用形式。它是一个“必须实现”的约束，可以修饰普通方法、类方法或静态方法。

## 放回 Repository 场景

关键词 Repository 的接口通常这样写：

```python
class BaseKeywordSubscriptionRepository(ABC):
    @abstractmethod
    async def get_by_user(self, user_id: str):
        ...

    @abstractmethod
    async def save(self, subscription):
        ...
```

这里使用 `@abstractmethod`，因为不同实现都必须支持“读取”和“保存”这两个行为。方法是实例方法，是因为具体实现可能持有数据库会话工厂、连接配置或其他依赖。

而关键词匹配中的纯计算可以直接写成模块级函数：

```python
def contains_keyword(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()
```

如果项目决定把它作为解析器类的一部分，也可以使用 `@staticmethod`，但这不会让它自动变成接口，也不会要求子类实现它。

## 常见误区

### 误区一：没有 `self` 就必须使用 `@staticmethod`

不需要 `self` 只说明它可以是静态方法。若它不属于某个类，模块级函数更清晰。

### 误区二：`@abstractmethod` 会自动生成实现

抽象方法只定义契约，不提供具体业务逻辑。数据库查询、HTTP 调用和匹配规则仍然要由具体实现完成。

### 误区三：把 Repository 方法写成静态方法

Repository 未来经常需要数据库会话、配置或事务依赖。过早使用 `@staticmethod` 会失去实例注入这些依赖的空间，普通实例方法更合适。

## 一个简单判断顺序

写方法前按这个顺序判断：

1. 方法需要读取或修改 `self` 吗？需要就使用实例方法。
2. 方法需要访问 `cls` 或创建当前类的实例吗？需要就考虑 `@classmethod`。
3. 方法属于某个类，但只依赖参数吗？可以使用 `@staticmethod`。
4. 具体实现是否必须由子类提供？是的话增加 `@abstractmethod`。
5. 如果它只是独立的纯函数，直接放在模块中。

这套判断可以避免为了“看起来面向对象”而增加不必要的类，也能避免把接口约束误解成调用方式。

## 小结

`@staticmethod` 说明方法不需要实例上下文；`@abstractmethod` 说明具体子类必须实现某项能力。前者决定调用形式，后者定义继承契约，两者可以组合但不能互相替代。

下一次看到装饰器时，可以先分别问两个问题：这个方法需要谁的状态？这个行为是否必须由子类实现？答案通常就能确定正确的写法。
