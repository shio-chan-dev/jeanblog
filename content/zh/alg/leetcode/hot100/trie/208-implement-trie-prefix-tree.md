---
title: "LeetCode 208：实现 Trie（前缀树）模板题解析"
date: 2026-06-24T17:38:48+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Trie", "前缀树", "字典树", "字符串", "LeetCode 208"]
description: "LeetCode 208 Implement Trie 是 Trie 模板题，核心是按固定接口实现 insert、search 和 startsWith。"
keywords: ["LeetCode 208", "Implement Trie", "Prefix Tree", "Trie", "前缀树", "字典树", "startsWith"]
---

> **副标题 / 摘要**
> 208 题的难点不在算法变化，而在把 Trie 模板按平台接口写稳：`insert` 建路径，`search` 查完整单词，`startsWith` 只查前缀路径。

- **预计阅读时长**：8~10 分钟
- **标签**：`Hot100`、`Trie`、`前缀树`、`LeetCode 208`
- **SEO 关键词**：LeetCode 208, Implement Trie, Prefix Tree, startsWith
- **元描述**：从接口要求出发实现 LeetCode 208，讲清 Trie 节点、children、is_end、insert/search/startsWith 的区别。

---

## A — Algorithm（题目与算法）

### 先看最小操作压力

208 题最关键的操作序列是：

```text
Trie trie = new Trie()
trie.insert("apple")
trie.search("apple")    -> true
trie.search("app")      -> false
trie.startsWith("app")  -> true
trie.insert("app")
trie.search("app")      -> true
```

这个例子说明：

- `app` 可以是 `apple` 的前缀
- 但只有插入过 `app` 后，`search("app")` 才能返回 `True`

所以这题不是“路径存在就算命中”。
我们必须同时维护：

- 路径是否存在
- 这条路径是否刚好是完整单词

### 题目接口

设计一个 Trie，也叫前缀树，支持三个操作：

- `insert(word)`：插入一个单词
- `search(word)`：判断完整单词是否已经插入
- `startsWith(prefix)`：判断是否存在某个已插入单词以 `prefix` 开头

平台要求使用固定类名和方法签名：

```python
class Trie:
    def __init__(self):
        ...

    def insert(self, word: str) -> None:
        ...

    def search(self, word: str) -> bool:
        ...

    def startsWith(self, prefix: str) -> bool:
        ...
```

---

## 目标读者

- 已经写过 Trie 模板，准备提交 LeetCode 208 的学习者
- 会用哈希表查单词，但不清楚前缀树接口差异的人
- 想固定 `insert/search/startsWith` 三个方法边界的人

## 背景 / 动机

LeetCode 208 是典型模板题。
它不要求你做复杂剪枝，也不要求统计前缀数量。
它只检查你是否真正理解这三个问题：

- 插入时如何创建缺失路径？
- 完整单词查询为什么不能只看路径？
- 前缀查询为什么不用检查 `is_end`？

---

## C — Concepts（核心思想）

### Step 1：固定节点结构

当前节点需要知道两件事：

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

- `children`：下一字符到下一节点的映射
- `is_end`：从根走到当前节点形成的字符串，是否是完整单词

题目只要求基本 Trie，所以不需要额外字段。

### Step 2：`insert` 建出缺失路径

插入 `word` 时，从根节点开始扫描字符。

当前循环 invariant：

> 处理第 `i` 个字符前，`node` 指向 `word[:i]` 对应的节点，且这条路径已经存在。

如果下一字符不存在，就创建：

```python
if ch not in node.children:
    node.children[ch] = TrieNode()
node = node.children[ch]
```

循环结束后，`node` 指向整个 `word` 的末尾节点。
这时设置：

```python
node.is_end = True
```

### Step 3：把路径查找抽成内部方法

`search` 和 `startsWith` 都要走一段字符串路径。
不同点只在走完之后怎么判断。

所以可以写一个内部方法：

```python
def _find_node(self, s: str):
    node = self.root
    for ch in s:
        if ch not in node.children:
            return None
        node = node.children[ch]
    return node
```

查询循环 invariant：

> 处理第 `i` 个字符前，`node` 指向 `s[:i]` 对应的节点。

如果某个字符不存在，路径断开，直接返回 `None`。

### Step 4：`search` 查完整单词

`search(word)` 需要两个条件都满足：

- `word` 的路径存在
- 末尾节点的 `is_end` 是 `True`

所以：

```python
node = self._find_node(word)
return node is not None and node.is_end
```

只插入 `apple` 时，`app` 的路径存在，但 `app` 末尾节点没有被标记成完整单词。
因此 `search("app")` 应该返回 `False`。

### Step 5：`startsWith` 只查前缀路径

`startsWith(prefix)` 只关心路径是否存在：

```python
return self._find_node(prefix) is not None
```

它不检查 `is_end`。
因为 `prefix` 不一定要是一个完整单词，只要能作为某个已插入单词的开头即可。

---

## 可运行示例（Python）

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_node(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None


if __name__ == "__main__":
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.startsWith("app") is True
    trie.insert("app")
    assert trie.search("app") is True
```

### Reference Answer

提交到 LeetCode 时，可以直接使用同一套实现：

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_node(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
```

这里没有引入新逻辑，只是把上面的可运行示例整理成提交形态。

---

## 解释与原理

### 为什么 LeetCode 208 是模板题？

因为三个方法刚好覆盖了 Trie 的三个基础动作：

| 方法 | 本质 | 是否检查 `is_end` |
| --- | --- | --- |
| `insert` | 创建路径并标记结束 | 最后设置为 `True` |
| `search` | 查完整单词路径 | 是 |
| `startsWith` | 查前缀路径 | 否 |

如果这三个方法能写稳，后续 Trie 题通常只是扩展字段：

- 统计经过某个前缀的单词数
- 删除单词时维护计数
- DFS 搜索时用 Trie 剪枝

### 为什么不用哈希集合？

哈希集合可以让 `search(word)` 很快。
但 `startsWith(prefix)` 会变麻烦：

- 要么扫描所有单词
- 要么为所有前缀额外建集合

Trie 把公共前缀天然合并在同一条路径里，所以前缀查询只需要沿字符走一遍。

### `search` 和 `startsWith` 的最小反例

只插入：

```text
apple
```

然后查询：

```text
search("app")     -> False
startsWith("app") -> True
```

这个反例能检查你有没有正确使用 `is_end`。

---

## R — Reflection（反思与深入）

### 复杂度分析

设输入字符串长度为 `L`：

- `insert`：时间 `O(L)`
- `search`：时间 `O(L)`
- `startsWith`：时间 `O(L)`
- 空间：总共 `O(total_chars)`，更准确地说是不同前缀节点数

### 常见错误

- `search` 只判断路径存在，导致 `search("app")` 错误返回 `True`
- `startsWith` 错误检查 `is_end`，导致合法前缀返回 `False`
- 插入时遇到已有字符也重新建节点，破坏之前插入的单词
- 忘记在 `insert` 结束后设置 `is_end`

### 什么时候用数组 children？

如果字符集固定为小写英文，可以写：

```python
self.children = [None] * 26
```

然后用 `idx = ord(ch) - ord("a")` 找子节点。
数组版本常数更小，但模板学习阶段容易被下标细节干扰。
这篇选择 `dict`，因为它最直接表达“字符 -> 子节点”。

---

## S — Summary（总结）

- LeetCode 208 就是 Trie 基础模板题
- `insert` 的核心 invariant 是：`node` 始终指向已处理前缀的节点，并创建缺失路径
- `search` 必须检查路径存在且 `is_end == True`
- `startsWith` 只检查前缀路径存在，不检查 `is_end`
- 写熟 208 后，再做 Trie + DFS 或前缀统计题会顺很多

### 推荐延伸阅读

- Trie 模板：从节点字段到插入查询 invariant
- 单词搜索 II：Trie 用于剪掉不可能的搜索分支
- 前缀计数题：在节点上增加 `pass_count` 或 `end_count`
