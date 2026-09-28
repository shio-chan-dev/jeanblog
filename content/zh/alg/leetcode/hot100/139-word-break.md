---
title: "LeetCode 139：单词拆分，从前缀是否可达开始"
date: 2026-09-22T12:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "动态规划", "字符串", "前缀可达性", "LeetCode 139"]
---

## 题目要求

### 输入输出

- 输入：字符串 `s` 和字符串列表 `wordDict`
- `wordDict` 中的每个元素都是一个可以使用的单词；同一个单词可以重复使用
- 输出：如果 `s` 可以被拆成一个或多个字典单词，返回 `True`；否则返回 `False`
- 拆分必须覆盖整个字符串，不能跳过字符，也不能改变字符顺序

### 示例

```text
输入：s = "leetcode", wordDict = ["leet", "code"]
输出：True
解释："leetcode" 可以拆成 "leet" + "code"
```

```text
输入：s = "applepenapple", wordDict = ["apple", "pen"]
输出：True
解释："apple" + "pen" + "apple"；同一个单词可以重复使用
```

```text
输入：s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出：False
解释：前面可以拼出一些片段，但最后不能覆盖完整字符串
```

### 约束

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` 和 `wordDict[i]` 只包含小写英文字母
- `wordDict` 中的单词互不相同

## Step 1：先让“已完成的前缀”变得可见

先只问一个具体问题：对于 `s = "leetcode"`，在确认整串能否拆开之前，我们能不能先记录“已经拆完了哪一段”？

### 压力：答案依赖中间边界

这个例子的合法拆分是：

```text
"leetcode" = "leet" + "code"
```

如果我们刚确认了 `"leet"`，那就知道下一个待处理位置是下标 `4`。但此时还没有处理 `"code"`，所以不能把“前缀已经可行”和“整串已经可行”混成一个布尔值。

### 当前 baseline

现在手里只有题目要求：最终要判断整串，直觉上可以尝试不同的切分方式，但还没有一个地方保存“某个前缀已经完成”的结果。

### 这个 baseline 在哪里断掉？

如果不记录中间边界，就无法表达下面这个状态：

```text
前 4 个字符 "leet"：已经可以拆分
剩余的 "code"：还没有处理
```

我们需要把“前缀是否可行”单独记下来。这里的前缀用一个边界表示：边界 `i` 对应 `s[:i]`，因此 `i = 4` 表示前四个字符，而不是第 4 个字符本身。

### 在当前版本中加入一个状态表

先只建立状态，不急着匹配字典单词：

```python
s = "leetcode"

# reachable[i] 表示 s[:i] 是否已经被完整拆分
reachable = [False] * (len(s) + 1)

# 空前缀不需要切任何单词，因此它是一个可行起点
reachable[0] = True
```

这里多出来的一个位置很重要：`s` 有 `8` 个字符，但我们要记录 `0..8` 共 `9` 个边界。`reachable[0]` 表示空前缀 `s[:0]`，`reachable[8]` 才代表整串 `s[:8]`。

### 检查这一处变化

当前版本还没有把任何字典单词接上去，所以唯一已知的可达边界应该是 `0`：

```python
assert len(reachable) == len(s) + 1
assert reachable[0] is True
assert all(flag is False for flag in reachable[1:])
print(reachable)
# [True, False, False, False, False, False, False, False, False]
```

### 冻结当前 checkpoint

现在这个版本能做到：

- 用 `reachable[i]` 表示“前缀 `s[:i]` 是否已经完整拆分”；
- 把空前缀作为后续扩展的可行起点；
- 区分“某个前缀可行”和“整串已经有答案”。

它还缺：

- 如何从一个已经可达的边界出发，匹配一个字典单词；
- 匹配成功后，如何把新的结束边界标记为可达。

下一步只处理一个起点的单词扩展，不会同时跳到所有边界。

## Step 2：先只走出第一个词

### 压力：空前缀已经可达，但还没有新的边界

Task 1 只告诉我们 `reachable[0]` 是真。回到 `"leetcode"`：如果从边界 `0` 开始尝试不同的结尾，什么时候可以把下一个边界标记为真？

### 当前 baseline

上一版有一张边界表：

```text
reachable[0] = True
reachable[1..8] = False
```

它能表示起点，却还没有使用题目给出的字典。

### 这个 baseline 在哪里断掉？

没有“候选片段是否是字典单词”的判断，就无法把 `0` 推进到 `4`。但这一步先只研究一个起点；如果现在就遍历所有起点，后面很难看清到底是哪一条规则完成了扩展。

### 在上一版中加入一次单层扩展

把字典写成集合，当前只从 `start = 0` 尝试结尾：

```python
s = "leetcode"
word_set = {"leet", "code"}

# reachable[i] 表示 s[:i] 是否已经被完整拆分
reachable = [False] * (len(s) + 1)
reachable[0] = True

start = 0
for end in range(start + 1, len(s) + 1):
    piece = s[start:end]
    if reachable[start] and piece in word_set:
        reachable[end] = True
```

这里发生的只有一件事：当 `s[start:end]` 是字典单词，并且 `start` 本身已经可达，就把 `end` 标为可达。对当前例子，`end = 4` 时片段是 `"leet"`，因此得到 `reachable[4] = True`。

### 检查这一处变化

```python
assert reachable[0] is True
assert reachable[4] is True      # 0 -> 4："leet"
assert reachable[8] is False     # 4 -> 8 还没有扫描
print(reachable)
# [True, False, False, False, True, False, False, False, False]
```

`reachable[8]` 仍然是 `False` 并不是答案为假，而是这版代码故意只处理了固定起点 `0`；它还没有从新到达的边界 `4` 继续尝试 `"code"`。

### 冻结当前 checkpoint

现在这个版本能做到：

- 从一个已经可达的边界出发枚举候选结尾；
- 用字典匹配结果把 `0` 推进到 `4`；
- 保持“前缀可达”和“整串可达”是两个不同状态。

它还缺：

- 从 `4` 等后续边界继续扩展；
- 把同一条规则应用到所有可能的 `start`，从而覆盖多词拆分。

下一步会只增加“扫描所有可达起点”的外层规则，并以此得到首个完整正确版本。

## Step 3：把同一规则交给所有可达边界

### 压力：`reachable[4]` 已经为真，但答案仍停在半路

Task 2 已经找到 `"leet"`，却因为代码固定了 `start = 0`，没有机会再尝试 `"code"`。对于多词拆分，新的可达边界必须成为下一轮的起点。

### 当前 baseline

上一版只做这一层：

```text
start = 0
从 0 枚举 end
```

内层的片段匹配规则本身没有问题，缺的是对后续 `start` 的扫描。

### 这个 baseline 在哪里断掉？

如果永远只从 `0` 出发，`reachable[4]` 虽然已经被标记，却永远不会参与下一次判断。因此 `"leetcode"` 的第二个词 `"code"` 被漏掉，`reachable[8]` 仍然是 `False`。

### 在上一版中只增加外层边界扫描

把固定的 `start = 0` 换成所有边界的循环；内层“片段命中字典就标记终点”的规则原样保留：

```python
def can_break(s: str, word_set: set[str]) -> bool:
    # reachable[i] 表示 s[:i] 是否已经被完整拆分
    reachable = [False] * (len(s) + 1)
    reachable[0] = True

    for start in range(len(s)):
        if not reachable[start]:
            continue

        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if piece in word_set:
                reachable[end] = True

    return reachable[len(s)]
```

这里唯一新增的算法规则是外层 `start` 扫描：只有已经可达的边界才继续尝试。`end` 循环和 `piece in word_set` 的判断，都是从 Task 2 原样带过来的。

### 先追踪 `"leetcode"` 的边界

```text
start = 0  -> 命中 "leet"  -> reachable[4] = True
start = 1,2,3 -> 不可达，跳过
start = 4  -> 命中 "code"  -> reachable[8] = True
```

因此最终返回 `reachable[8]`，而不是只看第一次扩展的结果。

### 检查这一处变化

```python
assert can_break("leetcode", {"leet", "code"}) is True
assert can_break("applepenapple", {"apple", "pen"}) is True
assert can_break(
    "catsandog",
    {"cats", "dog", "sand", "and", "cat"},
) is False
print("three examples passed")
```

这三个断言分别覆盖：两段连接、单词重复使用，以及前缀看似可行但无法覆盖终点的情况。此时这是第一个能对完整输入返回正确布尔值的版本；`word_set` 暂时作为已经准备好的集合传入，列表输入的最终包装留到最后。

### 冻结当前 checkpoint

现在这个版本能做到：

- 从每个已达边界继续尝试后续字典单词；
- 用同一条转移规则覆盖多段拆分；
- 对三个题目示例返回正确的布尔结果。

它还缺：

- 当前对每个起点都尝试到字符串末尾，仍会检查不可能超过字典最长单词的片段；
- 还没有把 `wordDict` 列表、LeetCode 的 `Solution` 包装和复杂度说明收束起来。

下一步只优化候选结尾的范围，不改变 `reachable` 的含义或转移规则。

## Step 4：只扫描可能的词长

### 压力：正确版本仍在检查不可能的片段

Task 3 已经能得到正确答案，但从某个 `start` 出发时，内层循环一直尝试到字符串末尾。例如 `word_set = {"leet", "code"}` 的最长单词长度是 `4`，从 `start = 0` 开始时，长度 `5`、`6`、直到 `8` 的片段不可能命中字典。

### 当前 baseline

上一版对每个可达 `start` 都执行：

```python
for end in range(start + 1, len(s) + 1):
    piece = s[start:end]
```

可达性转移已经正确，浪费只发生在候选范围上。

### 这个 baseline 在哪里断掉？

题目保证字典中的单词长度有限。若候选片段长度超过字典中最长单词，它一定不会命中 `word_set`，继续检查只会增加扫描次数，不会产生新的可达边界。

### 在上一版中只收窄 `end` 的上界

先从字典得到最长单词长度，再让每个起点最多向后看这么远：

```python
def can_break(s: str, word_set: set[str]) -> bool:
    # reachable[i] 表示 s[:i] 是否已经被完整拆分
    reachable = [False] * (len(s) + 1)
    reachable[0] = True
    max_word_len = max(len(word) for word in word_set)

    for start in range(len(s)):
        if not reachable[start]:
            continue

        end_limit = min(len(s), start + max_word_len)
        for end in range(start + 1, end_limit + 1):
            piece = s[start:end]
            if piece in word_set:
                reachable[end] = True

    return reachable[len(s)]
```

对 `word_set = {"leet", "code"}`，`max_word_len` 是 `4`；当 `start = 0` 时，`end` 只会取 `1..4`，刚好保留可能命中的最长片段。这里没有改变“命中单词就标记终点”的规则，只减少了不可能的候选。

### 检查这一处变化

```python
assert can_break("leetcode", {"leet", "code"}) is True
assert can_break("applepenapple", {"apple", "pen"}) is True
assert can_break(
    "catsandog",
    {"cats", "dog", "sand", "and", "cat"},
) is False

# 合法单词长度刚好等于最长长度时，不能把上界少算一位
assert can_break("abcdef", {"abc", "def"}) is True
print("bounded scan examples passed")
```

自检时用同一组输入分别运行 Task 3 的未限制版本和这一版，结果完全一致；`start = 0`、最长词长为 `4` 的范围追踪也确认没有检查长度 `5..8` 的片段。

### 冻结当前 checkpoint

现在这个版本能做到：

- 保持 Task 3 的全部可达性结果；
- 对每个起点只检查不超过最长字典词长度的候选；
- 在不改变状态转移的前提下减少无意义扫描。

它还缺：

- 把 `wordDict` 列表转换、LeetCode `Solution.wordBreak` 方法和最终测试整理成可直接提交的形态；
- 给出完整的不变量、正确性和复杂度说明。

下一步只做最终平台包装与证明收束，不再引入新的算法逻辑。

## Step 5：交付最终可提交版本

### 压力：核心逻辑还不是 LeetCode 的提交接口

Task 4 的 `can_break` 已经得到正确结果，但它接收的是已经准备好的 `word_set`。题目提交接口给的是 `wordDict` 列表，平台还要求把代码放进 `Solution.wordBreak`。

### 当前 baseline

上一版已经确定了全部算法部件：

- `reachable[0] = True`；
- 扫描可达 `start`；
- 只检查最长词长以内的 `end`；
- 命中字典片段就标记 `reachable[end]`。

### 这个 baseline 在哪里断掉？

它还不能直接复制到 LeetCode 的方法签名中，也没有把“为什么返回值正确”写成可检查的不变量。此处不再寻找新策略，只完成交付形态和证明。

### 只加入平台包装，不改变循环

下面的循环与 Task 4 完全相同；新增的只有 `class Solution`、`wordBreak` 方法，以及把列表转换成集合：

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        word_set = set(wordDict)
        reachable = [False] * (len(s) + 1)
        reachable[0] = True
        max_word_len = max(len(word) for word in word_set)

        for start in range(len(s)):
            if not reachable[start]:
                continue

            end_limit = min(len(s), start + max_word_len)
            for end in range(start + 1, end_limit + 1):
                piece = s[start:end]
                if piece in word_set:
                    reachable[end] = True

        return reachable[len(s)]
```

### 正确性：`reachable` 保持什么不变量？

处理到任意边界 `i` 时，`reachable[i]` 表示且仅表示：前缀 `s[:i]` 可以由字典单词完整拼出。

- 初始时 `reachable[0] = True`，空前缀不需要选择任何单词。
- 如果 `reachable[start]` 为真，并且 `s[start:end]` 在字典中，那么把这个合法单词接上去，就得到一个合法的 `s[:end]`，所以可以设置 `reachable[end] = True`。
- 反过来，任意一个合法拆分的最后一个单词都对应某个 `start:end` 区间；它的长度不超过 `max_word_len`，因此会被当前循环检查到。只有当前缀 `s[:start]` 已经可达时，这个区间才会被接上。

所以循环结束后，`reachable[len(s)]` 为真，当且仅当整串可以被完整拆分。

### 复杂度

设 `n = len(s)`，`L` 是字典中的最长单词长度，`C` 是所有字典单词长度之和。

- 最多检查 `O(nL)` 个候选区间。
- 这份 Python 实现会为每个候选创建切片并计算字符串哈希；把这些操作成本算进去，时间复杂度为 `O(nL^2 + C)`。如果把区间视为常数时间的字符串视图，转移数量则是 `O(nL)`。
- `reachable` 需要 `O(n)` 额外空间，`word_set` 需要 `O(C)` 存储。

### 回归检查

```python
solver = Solution()
assert solver.wordBreak("leetcode", ["leet", "code"]) is True
assert solver.wordBreak("applepenapple", ["apple", "pen"]) is True
assert solver.wordBreak(
    "catsandog",
    ["cats", "dog", "sand", "and", "cat"],
) is False
assert solver.wordBreak("abcdef", ["abc", "def"]) is True
assert solver.wordBreak("a", ["b"]) is False
assert solver.wordBreak("", ["a"]) is True
print("Solution.wordBreak regression tests passed")
```

### 冻结最终 checkpoint

现在这个版本能做到：

- 直接使用 LeetCode 139 的 `Solution.wordBreak` 接口；
- 复用已经逐步验证过的前缀可达性状态和最长词长边界；
- 用不变量解释每次状态更新，并通过回归断言覆盖可拆分、重复单词、不可拆分和边界情况。

这篇教程的代码链已经闭合：最终包装没有加入任何前面没有出现的新算法逻辑。

## 小结

- 先把“前缀是否已经完整拆分”写成边界状态 `reachable[i]`。
- 空前缀 `reachable[0]` 是所有合法拆分的起点。
- 从可达边界接上一个字典词，就把它的结束边界标为可达。
- 外层扫描所有可达边界，才能覆盖多词拆分。
- 最长词长只负责收窄候选范围，不改变状态转移。
