---
title: "LeetCode 155：最小栈，如何让最小值跟着栈一起变化"
date: 2026-08-21
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "栈", "设计", "辅助栈", "LeetCode 155"]
description: "从弹出当前最小值后必须恢复历史最小值的压力出发，逐步构建 LeetCode 155 Min Stack。"
keywords: ["LeetCode 155", "Min Stack", "最小栈", "辅助栈", "Hot100"]
---

## 题目要求

设计一个栈 `MinStack`，支持以下操作：

- `MinStack()`：初始化栈。
- `push(value)`：把 `value` 压入栈顶。
- `pop()`：删除栈顶元素。
- `top()`：返回栈顶元素。
- `getMin()`：返回栈中的最小元素。

题目要求每个操作的时间复杂度都是 `O(1)`。

`pop`、`top` 和 `getMin` 只会在栈非空时调用，因此不需要为这些方法设计额外的空栈返回值。

### 示例

```text
输入操作：
["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"]

输入参数：
[[], [-2], [0], [-3], [], [], [], []]

输出：
[null, null, null, null, -3, null, 0, -2]
```

对应的执行过程是：

```python
min_stack = MinStack()
min_stack.push(-2)
min_stack.push(0)
min_stack.push(-3)
min_stack.getMin()  # -3
min_stack.pop()
min_stack.top()     # 0
min_stack.getMin()  # -2
```

### 约束

- `-2^31 <= value <= 2^31 - 1`
- 最多调用 `3 * 10^4` 次 `push`、`pop`、`top` 和 `getMin`
- `pop`、`top` 和 `getMin` 调用时栈一定非空

## Step 1：弹出最小值后，前一个最小值从哪里回来

普通栈已经能保存值，并按照后进先出的顺序执行 `push`、`pop` 和 `top`。当前 baseline 是：

```text
只保存栈中的元素；栈顶就是最后压入、还没有弹出的元素。
```

这个 baseline 能直接回答 `top()`，却不能直接回答 `getMin()`。最小值可能位于栈底或中间，并不一定等于栈顶。

更关键的问题出现在 `pop()` 之后。逐步观察官方示例：

| 操作 | 操作后的栈 | 当前最小值 | 返回值 |
| --- | --- | ---: | ---: |
| 初始化 | `[]` | 无 | `null` |
| `push(-2)` | `[-2]` | `-2` | `null` |
| `push(0)` | `[-2, 0]` | `-2` | `null` |
| `push(-3)` | `[-2, 0, -3]` | `-3` | `null` |
| `getMin()` | `[-2, 0, -3]` | `-3` | `-3` |
| `pop()` | `[-2, 0]` | `-2` | `null` |
| `top()` | `[-2, 0]` | `-2` | `0` |
| `getMin()` | `[-2, 0]` | `-2` | `-2` |

`-3` 被弹出以后，当前最小值必须立即恢复为 `-2`。如果只用一个变量记录“到目前为止见过的最小值”，它会停留在已经离开栈的 `-3`，无法知道应该恢复成什么。

重复最小值还会带来另一种压力：

| 操作 | 操作后的栈 | 当前最小值 |
| --- | --- | ---: |
| `push(2)` | `[2]` | `2` |
| `push(1)` | `[2, 1]` | `1` |
| `push(1)` | `[2, 1, 1]` | `1` |
| `pop()` | `[2, 1]` | `1` |

弹出一个 `1` 后，另一个 `1` 仍在栈中，所以当前最小值不能恢复为 `2`。

这两个过程共同说明：每一个栈状态都对应自己的当前最小值。`push` 会进入一个新状态，`pop` 会回到前一个状态；最小值必须与这种状态变化保持一致。

现在这一版能做到：

- 逐步解释 `top()` 与 `getMin()` 返回的不是同一类信息。
- 解释弹出唯一最小值后为什么要恢复历史最小值。
- 解释重复最小值为什么不能在一次弹出后消失。

它还缺：

- 一个真正实现五个接口的可运行类。
- 一种能在栈变化后正确计算当前最小值的具体方法。

## Step 2：先在需要时扫描整个栈

当前 baseline 已经知道每个栈状态都有自己的最小值，但还没有可运行的类。先追求行为正确：用一个列表保存栈中的全部值，`push`、`pop` 和 `top` 直接使用列表的尾部。

当 `getMin()` 被调用时，扫描当前列表并返回其中的最小值。这样即使刚刚弹出了唯一最小值，也能从剩余元素中重新找到正确答案。

这与 Step 1 的结论并不冲突：

- 仅靠一个“当前最小值”变量，无法在弹出它以后用 `O(1)` 时间恢复历史最小值。
- 如果允许扫描剩余的全部元素，就可以重新计算最小值，只是时间复杂度变成 `O(n)`。

把这个 baseline 写成完整类：

```python
class MinStackScan:
    def __init__(self):
        self.values = []

    def push(self, value: int) -> None:
        self.values.append(value)

    def pop(self) -> None:
        self.values.pop()

    def top(self) -> int:
        return self.values[-1]

    def getMin(self) -> int:
        return min(self.values)


# 官方操作序列。
stack = MinStackScan()
stack.push(-2)
stack.push(0)
stack.push(-3)
assert stack.getMin() == -3
stack.pop()
assert stack.top() == 0
assert stack.getMin() == -2

# 重复最小值：弹出一个 1 后，另一个 1 仍然有效。
duplicates = MinStackScan()
duplicates.push(2)
duplicates.push(1)
duplicates.push(1)
assert duplicates.getMin() == 1
duplicates.pop()
assert duplicates.getMin() == 1

# 递增序列的最小值位于栈底，递减序列的最小值位于栈顶。
increasing = MinStackScan()
for value in [1, 2, 3]:
    increasing.push(value)
assert increasing.top() == 3
assert increasing.getMin() == 1

decreasing = MinStackScan()
for value in [3, 2, 1]:
    decreasing.push(value)
assert decreasing.top() == 1
assert decreasing.getMin() == 1
```

这个版本五个接口的行为都正确，但复杂度并不都满足题目要求：

| 操作 | 时间复杂度 | 原因 |
| --- | --- | --- |
| `MinStackScan()` | `O(1)` | 只创建空列表 |
| `push(value)` | 均摊 `O(1)` | 在列表尾部追加 |
| `pop()` | `O(1)` | 删除列表尾部 |
| `top()` | `O(1)` | 读取列表尾部 |
| `getMin()` | `O(n)` | 扫描当前全部元素 |

如果连续调用很多次 `getMin()`，每次都会重新比较相同的元素。题目最多执行 `3 * 10^4` 次操作，并且明确要求每个方法都是 `O(1)`，所以这个版本还不能作为最终答案。

现在这一版能做到：

- 完整实现题目要求的五个接口。
- 在弹出唯一最小值或一个重复最小值后返回正确结果。
- 作为后续优化可以直接对照的正确 baseline。

它还缺：

- 让 `getMin()` 不再扫描整个栈。
- 在 `push` 和 `pop` 发生时同步保存恢复最小值所需的信息。

## Step 3：让最小值与每一层栈状态同步

当前 baseline 的结果正确，但 `getMin()` 每次都会扫描 `self.values`。这些比较其实可以在 `push` 时提前完成，因为压入新值后，新的最小值只有两种可能：

```text
新状态的最小值 = min(前一个状态的最小值, 新压入的值)
```

增加一个与值栈等长的列表 `min_values`。`min_values[i]` 保存 `values[0:i+1]` 的最小值，也就是值栈深度为 `i + 1` 时的当前最小值。

官方序列中的两个列表会同步变化：

| 操作 | `values` | `min_values` |
| --- | --- | --- |
| `push(-2)` | `[-2]` | `[-2]` |
| `push(0)` | `[-2, 0]` | `[-2, -2]` |
| `push(-3)` | `[-2, 0, -3]` | `[-2, -2, -3]` |
| `pop()` | `[-2, 0]` | `[-2, -2]` |

两个列表的末尾始终属于同一个栈状态。因此：

- `push` 必须同时追加值和新的当前最小值。
- `pop` 必须同时删除两个列表的末尾。
- `top` 读取 `values[-1]`。
- `getMin` 直接读取 `min_values[-1]`。

重复最小值也会被逐层保存。压入 `2, 1, 1` 后，`min_values` 是 `[2, 1, 1]`；弹出一次只会删除最后一个 `1`，前一个 `1` 仍然留在栈顶。

把上一版线性扫描替换为同步状态：

```python
import random


class MinStack:
    def __init__(self):
        self.values = []
        self.min_values = []

    def push(self, value: int) -> None:
        self.values.append(value)

        if self.min_values:
            value = min(value, self.min_values[-1])
        self.min_values.append(value)

    def pop(self) -> None:
        self.values.pop()
        self.min_values.pop()

    def top(self) -> int:
        return self.values[-1]

    def getMin(self) -> int:
        return self.min_values[-1]


# 官方操作序列。
stack = MinStack()
stack.push(-2)
stack.push(0)
stack.push(-3)
assert stack.getMin() == -3
stack.pop()
assert stack.top() == 0
assert stack.getMin() == -2

# 重复最小值必须逐层保留。
duplicates = MinStack()
duplicates.push(2)
duplicates.push(1)
duplicates.push(1)
assert duplicates.min_values == [2, 1, 1]
duplicates.pop()
assert duplicates.getMin() == 1

# 用固定种子生成 5,000 次合法操作，与 Step 2 baseline 对照。
rng = random.Random(155)
expected = MinStackScan()
actual = MinStack()

for _ in range(5_000):
    if not expected.values or rng.random() < 0.55:
        value = rng.randint(-100, 100)
        expected.push(value)
        actual.push(value)
    else:
        operation = rng.choice(["pop", "top", "getMin"])

        if operation == "pop":
            expected.pop()
            actual.pop()
        elif operation == "top":
            assert actual.top() == expected.top()
        else:
            assert actual.getMin() == expected.getMin()

    assert len(actual.values) == len(actual.min_values)
    if actual.values:
        assert actual.min_values[-1] == min(actual.values)
        assert actual.top() == expected.top()
        assert actual.getMin() == expected.getMin()
```

### 为什么同步列表能恢复历史最小值

每次操作前后都保持下面的不变量：

```text
len(values) == len(min_values)

当 values 非空时：
min_values[-1] == min(values)
```

当 `values` 为空时，`min_values` 也必须为空。题目保证此时不会调用 `top()` 或 `getMin()`，因此不需要读取不存在的末尾元素。

第一次 `push` 时，唯一的值自然也是最小值。之后每次 `push(value)`，新前缀只比旧前缀多一个 `value`，所以比较旧最小值与新值就能得到新最小值。

`pop()` 同时删除两个列表的末尾，相当于一起回到前一个栈深度。前一个深度的最小值早已保存在新的 `min_values[-1]`，不需要重新扫描 `values`。

### 复杂度

在 Python 列表实现中，`push` 和列表尾部 `pop` 的时间复杂度是均摊 `O(1)`，`top` 和 `getMin` 是最坏 `O(1)`。按题目的栈操作模型，四个接口都满足常数时间要求。两个列表各保存与栈深度同阶的元素，空间复杂度是 `O(n)`。

现在这一版能做到：

- 正确实现题目要求的全部接口。
- 在 `O(1)` 时间内返回当前最小值。
- 弹出唯一或重复最小值后立即恢复正确的历史状态。
- 通过固定样例和 5,000 次随机合法操作对照。

算法部分已经完整。下一步只需要教程一致性检查和独立全文审核。
