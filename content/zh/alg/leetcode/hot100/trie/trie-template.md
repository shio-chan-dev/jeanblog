---
title: "Trie 模板：从节点字段到插入查询 invariant"
date: 2026-06-24T17:38:48+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Trie", "前缀树", "字典树", "字符串", "模板"]
description: "从零搭一个 Trie 模板，重点理解每个节点存什么、children 怎么走、is_end 为什么需要，以及 insert/search 的循环 invariant。"
keywords: ["Trie", "前缀树", "字典树", "children", "is_end", "字符串模板", "Hot100"]
---

> **副标题 / 摘要**
> Trie 的重点不是背代码，而是理解“一个节点代表一个前缀”。只要这个模型稳定，插入、完整单词查询和前缀查询都会变成同一个循环。

- **预计阅读时长**：8~10 分钟
- **标签**：`Hot100`、`Trie`、`前缀树`、`字典树`
- **SEO 关键词**：Trie, 前缀树, 字典树, children, is_end
- **元描述**：用 Python 写一个最小 Trie 模板，讲清节点字段、children 走法、结束标记和循环 invariant。

---

## A — Algorithm（从一个小任务开始）

### 小任务：同时回答完整单词和前缀

假设已经插入：

```text
app
apple
bat
```

现在要问：

- `app` 是不是完整单词？
- `ap` 是不是某个单词的前缀？
- `apply` 是否存在？

这个小任务暴露了两个缺口：

- 只用哈希集合，可以快速判断完整单词，但不能自然回答前缀问题
- 只看路径存在，又会把 `app` 和 `apple` 的前缀关系混成一件事

Trie 要解决的就是：让很多字符串共享公共前缀，同时还能区分“前缀存在”和“完整单词存在”。

### 从压力反推要支持什么

我们先不管任何题目接口，只定义一个自己的模板：

- `insert(word)`：把一个单词插入 Trie
- `search(word)`：判断完整单词是否存在
- `starts_with(prefix)`：判断是否存在以 `prefix` 开头的单词

### 最小结构图

插入 `app` 和 `apple` 后，结构可以想成：

```text
root
 └─ a
    └─ p
       └─ p  [end]
          └─ l
             └─ e  [end]
```

这里最重要的是：

- 节点本身不是存整个单词
- 从 `root` 走到某个节点的路径，才组成一个前缀
- `[end]` 表示这条路径刚好也是一个完整单词

---

## 目标读者

- 第一次接触 Trie，想先理解结构本身的人
- 做字符串前缀匹配题时，总是分不清 `search` 和 `startsWith` 的人
- 想把 LeetCode 208 之前的模板先写稳的人

## 背景 / 动机

如果只判断一个单词是否出现，哈希表已经够用。
但当问题变成“有没有某个前缀”时，哈希表就不够自然了。

Trie 的价值就在这里：它不是把每个字符串孤立存起来，而是把公共前缀压到同一条路径上。

---

## C — Concepts（核心思想）

### Step 1：每个节点到底存什么？

一个 Trie 节点只需要两个字段：

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

字段含义：

- `children`：从当前前缀继续往下走，可选的下一个字符
- `is_end`：从根走到当前节点形成的字符串，是否是一个完整单词

注意，节点里通常不需要存当前字符。
因为当前字符已经体现在“父节点通过哪条边走到我”这件事里。

### Step 2：children 怎么走？

假设已经在表示前缀 `ap` 的节点上，下一字符是 `p`：

```python
node = node.children["p"]
```

这一步的语义是：

> 当前前缀从 `ap` 变成 `app`。

如果 `children` 里没有 `"p"`，说明当前 Trie 里没有这条路径。
插入时需要创建节点，查询时直接失败。

### Step 3：为什么需要 is_end？

只看路径存在不够。

插入 `apple` 后，路径 `a -> p -> p` 一定存在。
但这不代表 `app` 被插入过。

所以：

- `starts_with("app")` 只需要路径存在
- `search("app")` 还需要最后节点的 `is_end == True`

这就是 `is_end` 的作用：区分“只是前缀”和“完整单词”。

### Step 4：插入的循环 invariant

插入 `word` 时，循环处理到第 `i` 个字符之前，保持这个 invariant：

> `node` 指向 `word[:i]` 这个前缀对应的节点；并且 `word[:i]` 的路径已经存在。

处理当前字符 `ch = word[i]`：

```python
if ch not in node.children:
    node.children[ch] = TrieNode()
node = node.children[ch]
```

循环结束后：

> `node` 指向整个 `word` 对应的节点。

这时把 `node.is_end = True`，表示这个路径不只是前缀，而是一个完整单词。

### Step 5：查询的循环 invariant

查询 `word` 或 `prefix` 时，循环处理到第 `i` 个字符之前，保持这个 invariant：

> `node` 指向查询串 `s[:i]` 这个前缀对应的节点；这个前缀路径已经被找到。

处理当前字符：

```python
if ch not in node.children:
    return None
node = node.children[ch]
```

只要某一步找不到字符，就说明路径断了。
如果所有字符都走完，就返回最后节点。

于是我们可以抽出一个 `_find_node(s)`：

- 找不到路径：返回 `None`
- 找到路径：返回最后一个字符对应的节点

`search` 和 `starts_with` 的区别只在最后一步判断：

```python
node = self._find_node(word)
return node is not None and node.is_end
```

```python
return self._find_node(prefix) is not None
```

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

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None


if __name__ == "__main__":
    trie = Trie()
    trie.insert("app")
    trie.insert("apple")

    assert trie.search("app") is True
    assert trie.search("ap") is False
    assert trie.starts_with("ap") is True
    assert trie.search("apply") is False
```

---

## 解释与原理

### `root` 表示什么？

`root` 表示空前缀 `""`。
所有单词都从空前缀开始，一字符一字符往下走。

这也是为什么插入和查询都从：

```python
node = self.root
```

开始。

### `children` 为什么用 dict？

Python 里用 `dict` 最直接：

- key 是下一个字符
- value 是对应的子节点
- 查找和插入平均 `O(1)`

如果题目明确只包含小写英文字母，也可以用长度为 `26` 的数组。
数组版本常数更小，但代码更容易被下标转换细节打断。
学习模板时，`dict` 更清楚。

### `is_end` 只在插入末尾设置

插入 `apple` 时，不应该把 `a`、`ap`、`app`、`appl` 都标成完整单词。
只有循环走完整个 `word` 后，当前节点才对应完整单词。

如果之后再插入 `app`，会走到同一个 `app` 节点，并把它的 `is_end` 改成 `True`。

---

## R — Reflection（反思与深入）

### 复杂度分析

设字符串长度为 `L`：

- `insert`：时间 `O(L)`，最多新增 `L` 个节点
- `search`：时间 `O(L)`，空间 `O(1)`
- `starts_with`：时间 `O(L)`，空间 `O(1)`

总空间取决于所有单词的不同前缀数量。
共享前缀越多，Trie 相比直接存全部字符串越能复用节点。

### 常见错误

- 只写 `children`，忘记 `is_end`，导致前缀被误判成完整单词
- 查询时遇到缺失字符还继续走，导致空指针错误
- 插入时每次都新建节点，覆盖已有公共前缀
- 把 `starts_with` 写成必须检查 `is_end`，导致短前缀查询失败

### 模板记忆方式

Trie 可以压缩成三句话：

- 节点表示“某个前缀走到这里”
- `children[ch]` 表示“追加字符 `ch` 后到哪个节点”
- `is_end` 表示“这个前缀是否刚好是完整单词”

---

## S — Summary（总结）

- Trie 的核心是路径，不是单个节点里的字符
- `children` 负责继续往下走，`is_end` 负责区分前缀和完整单词
- 插入和查询的循环 invariant 都是：`node` 始终指向当前已处理前缀对应的节点
- 理解这个模板后，LeetCode 208 只是把方法名换成题目要求的接口

### 推荐延伸阅读

- `208. Implement Trie (Prefix Tree)`：把这个模板套进固定类名和方法签名
- 单词搜索 II：Trie + DFS 剪枝
- 前缀统计类题目：在节点上扩展计数字段
