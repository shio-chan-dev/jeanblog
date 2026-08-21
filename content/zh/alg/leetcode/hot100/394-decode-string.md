---
title: "LeetCode 394：字符串解码，如何保存并恢复嵌套上下文"
date: 2026-08-21
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "栈", "字符串", "递归", "LeetCode 394"]
description: "从嵌套编码会打断外层解码的压力出发，逐步构建 LeetCode 394 Decode String。"
keywords: ["LeetCode 394", "Decode String", "字符串解码", "嵌套上下文", "Hot100"]
---

## 题目要求

给定一个编码字符串 `s`，返回它解码后的字符串。

编码规则是：

```text
k[encoded_string]
```

方括号中的 `encoded_string` 需要连续重复 `k` 次，其中 `k` 是正整数。编码可以嵌套，也可以与普通小写字母相邻。

题目保证：

- 输入字符串始终有效，方括号完整配对且没有多余空格。
- 原始文本不包含数字，数字只表示重复次数。
- 不会出现 `3a` 或 `2[4]` 这类不符合编码规则的输入。
- 解码后的字符串长度不会超过 `10^5`。

### 示例

| 输入 | 输出 |
| --- | --- |
| `"3[a]2[bc]"` | `"aaabcbc"` |
| `"3[a2[c]]"` | `"accaccacc"` |
| `"2[abc]3[cd]ef"` | `"abcabccdcdcdef"` |

### 约束

- `1 <= s.length <= 30`
- `s` 只包含小写英文字母、数字和 `[]`
- 所有重复次数都在 `[1, 300]` 范围内

LeetCode 提供的方法签名是：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        pass
```

## Step 1：进入内层以后，外层信息去了哪里

先从没有嵌套的输入开始：

```text
3[a]
```

读到 `3` 后知道下一段需要重复三次；读完方括号中的 `a`，得到：

```text
a * 3 = aaa
```

当前 baseline 是：

```text
读出一个重复次数，再收集后续方括号中的文本，遇到 ] 时执行重复。
```

这个 baseline 可以处理一层 `3[a]`，却会在 `3[a2[c]]` 上中断。外层已经读到重复次数 `3` 和普通字母 `a`，此时又遇到了内层编码 `2[c]`。如果直接把当前次数改成 `2`、当前文本改成 `c`，外层的 `3` 和 `a` 就丢失了。

逐层展开这个输入：

| 阶段 | 当前处理的层 | 当前层已知内容 | 必须暂时保留的信息 |
| --- | --- | --- | --- |
| 读到外层 `3[` | 外层 | 重复 `3` 次 | 无 |
| 外层读到 `a` | 外层 | 文本前缀是 `a` | 无 |
| 读到内层 `2[` | 内层 | 准备重复 `2` 次 | 外层文本 `a`、外层次数 `3` |
| 内层读到 `c]` | 内层完成 | `c * 2 = cc` | 恢复外层文本 `a`、外层次数 `3` |
| 回到外层 | 外层 | `a + cc = acc` | 外层次数仍是 `3` |
| 读到外层 `]` | 外层完成 | `acc * 3` | 无 |

最终得到：

```text
3[a2[c]]
-> 3[a + cc]
-> 3[acc]
-> accaccacc
```

因此，进入内层之前，不能覆盖尚未完成的外层信息。至少需要暂时保留两项内容：

- 外层已经解码出的文本。
- 内层完成后，外层整体还要重复多少次。

相邻但不嵌套的 `3[a]2[bc]` 不会同时保留两层状态：先完整得到 `aaa`，再独立解码 `2[bc]` 得到 `bcbc`，最后拼接为 `aaabcbc`。真正制造状态保存压力的是“一个编码尚未完成时又进入另一个编码”。

现在这一版能做到：

- 区分单层、相邻和嵌套编码。
- 逐层解释 `3[a2[c]]` 为什么得到 `accaccacc`。
- 明确进入内层前必须保留的外层文本和重复次数。

它还缺：

- 一个可以执行进入内层、返回外层和继续读取的具体算法。
- 对多位重复次数、普通文本后缀和更深嵌套的运行检查。

## Step 2：先让每一层递归完成自己的解码

当前 baseline 已经知道进入内层时需要保留外层上下文，但还没有定义“内层何时完成、外层从哪里继续”。

把问题缩小到一个括号层：从某个位置开始读取，直到遇到属于当前层的 `]`。这个过程需要返回两项结果：

```text
当前层解码后的文本
右括号之后的下一个读取位置
```

外层看到 `[` 时，让下一层从 `[` 后面开始处理；下一层遇到自己的 `]` 后返回，外层把内层文本重复指定次数，再从返回的位置继续读取。

对 `3[a2[c]]` 追踪读取位置：

| 读取位置 | 当前字符 | 当前层操作 | 下一位置或返回结果 |
| ---: | --- | --- | --- |
| 0 | `3` | 顶层累计次数 `3` | 1 |
| 1 | `[` | 进入外层编码，从位置 2 读取 | 等待返回 |
| 2 | `a` | 外层文本加入 `a` | 3 |
| 3 | `2` | 外层累计内层次数 `2` | 4 |
| 4 | `[` | 进入内层编码，从位置 5 读取 | 等待返回 |
| 5 | `c` | 内层文本加入 `c` | 6 |
| 6 | `]` | 内层返回 `("c", 7)` | 外层得到 `c * 2` |
| 7 | `]` | 外层返回 `("acc", 8)` | 顶层得到 `acc * 3` |
| 8 | 字符串末尾 | 顶层返回 `"accaccacc"` | 完成 |

把这个层级契约写成递归 baseline：

```python
class RecursiveDecoder:
    def decodeString(self, s: str) -> str:
        decoded, _ = self._decode_layer(s, 0)
        return decoded

    def _decode_layer(self, s: str, index: int) -> tuple[str, int]:
        parts = []
        repeat = 0

        while index < len(s):
            char = s[index]

            if char.isdigit():
                repeat = repeat * 10 + int(char)
                index += 1
            elif char == "[":
                nested, index = self._decode_layer(s, index + 1)
                parts.append(nested * repeat)
                repeat = 0
            elif char == "]":
                return "".join(parts), index + 1
            else:
                parts.append(char)
                index += 1

        return "".join(parts), index


decoder = RecursiveDecoder()

# 官方示例。
assert decoder.decodeString("3[a]2[bc]") == "aaabcbc"
assert decoder.decodeString("3[a2[c]]") == "accaccacc"
assert decoder.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"

# 多位重复次数、普通文本和更深嵌套。
assert decoder.decodeString("12[a]") == "a" * 12
assert decoder.decodeString("abc3[cd]xyz") == "abccdcdcdxyz"
assert decoder.decodeString("2[a2[b2[c]]]") == "abccbccabccbcc"
```

这里的 `repeat` 只属于当前层：连续读到多个数字时，通过 `repeat * 10 + int(char)` 组成完整次数。当前层进入下一层后，递归调用会拥有自己的 `parts` 和 `repeat`；返回后，当前层原来的局部状态仍然存在。

遇到 `]` 时必须返回 `index + 1`，因为调用者下一步应该读取右括号之后的字符，而不是再次处理同一个 `]`。顶层没有对应的右括号，所以会在 `index == len(s)` 时返回最终结果。

### 正确性与复杂度

每一层只消费属于自己的普通字符、次数和直接子层。子层返回的文本已经完成解码，因此当前层只需按自己的次数重复并拼接。输入保证括号合法，所以每次进入子层都能在对应的 `]` 处返回，顶层最终会到达字符串末尾。

设编码字符串长度为 `n`，最终输出长度为 `L`，最大嵌套深度为 `d`。每个输入字符只负责一次分支判断，但重复和拼接必须实际构造输出。更准确地说，时间复杂度是 `O(n + W)`，其中 `W` 是所有层在整个运行期间构造的字符串字符总量，最坏可写为 `O(n + dL)`。

空间不能直接用累计构造量 `W` 表示，因为先前的临时字符串可能已经释放。递归调用帧占 `O(d)`；各层的 `parts`、索引等状态占 `O(n + d)`。再设 `P` 为某一时刻仍同时存活的解码文本和临时副本字符总量，则峰值空间是 `O(n + d + P)`；由于每一层同时保留的文本长度都不超过最终输出长度，`P` 最坏为 `O(dL)`，所以包含返回结果在内的空间上界是 `O(n + dL)`。

现在这一版能做到：

- 正确解码单层、相邻和嵌套编码。
- 正确累计多位重复次数。
- 在子层结束后返回外层应该继续读取的位置。
- 通过官方示例、普通后缀和深层嵌套检查。

它还缺：

- 把每次进入内层时保存的外层文本和次数直接展示出来。
- 用显式状态替代语言自动保存的递归调用上下文。

## Step 3：把递归保存的上下文放到显式栈中

递归 baseline 已经能够正确解码，但外层状态由语言的调用栈自动保存。每次递归进入下一层时，调用帧实际保留了当前层已经读出的文本、重复次数和继续读取的位置；返回时再恢复这些信息。

现在改成从左到右扫描一次字符串，循环本身会负责继续读取的位置。因此，遇到 `[` 时只需保存：

```text
(外层已经解码出的文本, 当前重复次数)
```

后进入的内层必须先完成，才能恢复外层，这正好符合后进先出的顺序。用 `contexts` 保存这些尚未恢复的外层上下文：

- 读到数字：逐位累计 `repeat_count`。
- 读到 `[`：把当前文本和次数压入 `contexts`，再重置当前层状态。
- 读到 `]`：弹出最近的外层上下文，把当前层文本重复后接回外层。
- 读到字母：追加到当前层文本。

为了避免 Python 不可变字符串在逐字符追加时反复复制，`current_parts` 保存当前层的文本片段；需要完成一层时才用 `"".join(current_parts)` 合并。它与递归版的 `parts` 含义相同，只是现在由循环主动保存和恢复。

对 `3[a2[c]]` 追踪栈状态：

| 当前字符 | `repeat_count` | `contexts`（栈底到栈顶） | 当前层文本 | 操作 |
| --- | ---: | --- | --- | --- |
| `3` | `3` | `[]` | `""` | 累计外层次数 |
| `[` | `0` | `[("", 3)]` | `""` | 保存外层并重置 |
| `a` | `0` | `[("", 3)]` | `"a"` | 追加普通字符 |
| `2` | `2` | `[("", 3)]` | `"a"` | 累计内层次数 |
| `[` | `0` | `[("", 3), ("a", 2)]` | `""` | 保存当前外层并重置 |
| `c` | `0` | `[("", 3), ("a", 2)]` | `"c"` | 追加普通字符 |
| `]` | `0` | `[("", 3)]` | `"acc"` | 恢复 `a`，接上 `c * 2` |
| `]` | `0` | `[]` | `"accaccacc"` | 恢复空前缀，接上 `acc * 3` |

整个扫描过程中保持三个约束：

1. `current_parts` 按顺序保存当前未完成层已经解码出的文本。
2. `repeat_count` 保存最近连续数字组成的完整次数；遇到 `[` 后立即归零。
3. `contexts` 从栈底到栈顶保存由外到内、尚未恢复的上下文，栈顶永远是当前 `]` 应该恢复的一层。

把这四种字符分支写进最终实现：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        contexts: list[tuple[list[str], int]] = []
        current_parts: list[str] = []
        repeat_count = 0

        for char in s:
            if char.isdigit():
                repeat_count = repeat_count * 10 + int(char)
            elif char == "[":
                contexts.append((current_parts, repeat_count))
                current_parts = []
                repeat_count = 0
            elif char == "]":
                nested_text = "".join(current_parts)
                outer_parts, count = contexts.pop()
                outer_parts.append(nested_text * count)
                current_parts = outer_parts
            else:
                current_parts.append(char)

        return "".join(current_parts)
```

先检查固定边界：

```python
solution = Solution()

# 官方示例。
assert solution.decodeString("3[a]2[bc]") == "aaabcbc"
assert solution.decodeString("3[a2[c]]") == "accaccacc"
assert solution.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"

# 多位重复次数、普通文本和更深嵌套。
assert solution.decodeString("12[a]") == "a" * 12
assert solution.decodeString("abc3[cd]xyz") == "abccdcdcdxyz"
assert solution.decodeString("2[a2[b2[c]]]") == "abccbccabccbcc"
```

再让随机生成器同时给出合法编码及其期望结果，并与 Step 2 的递归 baseline 对照。下面的检查接在前两个完整代码块之后运行：

```python
import random


random_generator = random.Random(394)


def generate_valid_case(depth: int = 0) -> tuple[str, str]:
    encoded_parts = []
    decoded_parts = []

    for _ in range(random_generator.randint(1, 3)):
        if depth < 3 and random_generator.random() < 0.5:
            nested_encoded, nested_decoded = generate_valid_case(depth + 1)
            repeat = random_generator.randint(1, 12)
            encoded_parts.append(f"{repeat}[{nested_encoded}]")
            decoded_parts.append(nested_decoded * repeat)
        else:
            text = "".join(
                random_generator.choice("abc")
                for _ in range(random_generator.randint(1, 3))
            )
            encoded_parts.append(text)
            decoded_parts.append(text)

    return "".join(encoded_parts), "".join(decoded_parts)


baseline = RecursiveDecoder()
checked = 0

while checked < 2_000:
    encoded, expected = generate_valid_case()
    if len(encoded) > 30 or len(expected) > 100_000:
        continue

    assert baseline.decodeString(encoded) == expected
    assert solution.decodeString(encoded) == expected
    checked += 1
```

随机检查不是用两个实现互相证明正确：生成器直接构造 `expected`，先提供独立期望值；递归版和显式栈版都必须与它一致。两种实现再对照，可以额外暴露进入或恢复层级时的差异。

### 正确性与复杂度

读到 `[` 时，算法完整保存尚未完成的外层状态；读到匹配的 `]` 时，根据栈的后进先出性质恢复最近一层。当前层已经完全解码，所以 `nested_text * count` 正是这一段编码的结果；把它追加回外层以后，三个约束继续成立。输入括号合法，扫描结束时所有上下文都已恢复，`current_parts` 因而表示整个字符串的解码结果。

设输入长度为 `n`、最终输出长度为 `L`、最大嵌套深度为 `d`。扫描分支本身是 `O(n)`；合并和重复必须实际构造解码文本。若 `W` 表示所有层在整个运行期间构造的字符总量，时间复杂度是 `O(n + W)`，最坏为 `O(n + dL)`。栈深度是 `O(d)`，所有活动片段列表及其引用是 `O(n)`，同时存活的解码文本和临时副本是 `O(L)`，因此包含返回结果在内的峰值空间是 `O(n + d + L)`。

现在这一版能做到：

- 用显式栈保存并恢复任意嵌套层级的外层上下文。
- 正确处理相邻编码、多位重复次数、普通前后缀和深层嵌套。
- 通过 6 个固定断言和 2,000 组合法随机编码检查。
- 把 20、155 和 394 串成“匹配最近状态、同步历史状态、保存嵌套上下文”的基础栈小闭环。

它还缺：

- 独立审核对推导连续性、代码正确性和复杂度结论的最终确认。
