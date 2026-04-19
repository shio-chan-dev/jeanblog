---
title: "Hot100：分割回文串（Palindrome Partitioning）回溯 + 回文预处理 ACERS 解析"
date: 2026-04-19T00:09:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "字符串", "回文", "DP", "LeetCode 131"]
description: "围绕 LeetCode 131 分割回文串，讲清“切分枚举 + 回文区间预处理”的完整推导过程，帮助你把字符串分割型回溯真正写稳。"
keywords: ["Palindrome Partitioning", "分割回文串", "回溯", "回文预处理", "DP", "LeetCode 131", "Hot100"]
---

> **副标题 / 摘要**
> `131. 分割回文串` 的难点不是递归本身，而是两个问题要同时想清楚：当前切到了哪里，以及一个候选片段是不是回文。把这两件事拆开，你就能得到“回溯枚举 + 回文表预处理”的稳定写法。

- **预计阅读时长**：15~18 分钟
- **标签**：`Hot100`、`回溯`、`字符串`、`回文`、`DP`
- **SEO 关键词**：Palindrome Partitioning, 分割回文串, 回溯, 回文预处理, DP
- **元描述**：通过 LeetCode 131 理解字符串分割型回溯与回文区间预处理，掌握“先判合法片段，再递归切后缀”的题目模型。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个字符串 `s`，请把它切分成若干个子串，使得每个子串都是回文串，并返回所有可能的切分方案。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| s | `string` | 只包含小写英文字母的字符串 |
| 返回 | `string[][]` | 所有合法的回文切分方案 |

### 示例 1

```text
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

### 示例 2

```text
输入：s = "a"
输出：[["a"]]
```

### 约束

- `1 <= s.length <= 16`
- `s` 仅由小写英文字母组成

---

## 目标读者

- 已经学过 `78/90` 这类数组回溯，准备进入“字符串切分型回溯”的学习者
- 会写递归，但总把“切分位置”和“合法性检查”搅在一起的开发者
- 想掌握“先预处理合法区间，再枚举切分”的工程迁移套路的读者

## 背景 / 动机

这题非常适合帮你建立一种更通用的模型：

> 先判断哪些区间是合法片段，再递归地把整个串切完。

如果你只是把它看成“回文题”，容易只盯着 `isPalindrome()`。
但如果你把它看成“字符串分割题”，就会更清楚它的结构：

- `start` 表示当前要从哪里继续切
- `path` 表示已经切出来的片段
- `[start, end]` 是当前想尝试的新片段
- 只有当这个片段合法时，才递归求解剩余后缀

这套结构不只用于回文，很多“词典切分 / 模板切分 / 规则切分”问题都能直接迁移。

## 核心概念

- **`start`**：当前还没切分的后缀起点
- **`path`**：当前已经切出的片段序列
- **回文区间**：`s[l:r+1]` 是否是回文串
- **区间预处理**：先用 `dp[l][r]` 记录每个子串是否合法
- **切分型回溯**：当前切一刀，递归处理剩余后缀

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先用最小例子看“切分”到底在做什么

看 `s = "aab"`。

从左往右切，你会发现只有两种合法方案：

- `["a", "a", "b"]`
- `["aa", "b"]`

这里最重要的观察不是“答案有哪些”，而是：

- 每次你都在当前起点选一个结尾
- 只要当前片段合法，就把问题递归交给后面的后缀

#### Step 2：当前部分答案最少要记住什么？

既然我们是在“逐段切分字符串”，就需要一个状态记录已经切出来的片段。
这就是 `path`。

```python
path = []
```

`path` 的含义是：

- 当前递归分支已经切好的所有片段
- 还不是最终答案全集

#### Step 3：剩余子问题应该怎样命名？

当 `path` 已经固定后，剩下的问题就是：

> 从下标 `start` 开始，把后缀 `s[start:]` 继续切完。

所以最自然的递归签名是：

```python
def dfs(start: int) -> None:
    ...
```

这和数组题里的 `startIndex` 很像，但语义已经变成“下一个待切分的字符位置”。

#### Step 4：什么时候说明一条路径已经完成？

当 `start == len(s)` 时，说明整个字符串都已经被合法切完了。

```python
if start == len(s):
    res.append(path.copy())
    return
```

此时 `path` 就是一整套完整切分方案。

#### Step 5：当前层有哪些候选动作？

从 `start` 出发，你可以把当前片段的结尾放在 `start, start+1, ... , n-1` 的任意位置。

```python
for end in range(start, len(s)):
    ...
```

每个 `end` 都对应一个候选片段 `s[start:end+1]`。

#### Step 6：怎样判断一个候选片段是不是回文？

最直观的方法是写一个辅助函数：

```python
def is_pal(left: int, right: int) -> bool:
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

只要 `is_pal(start, end)` 为真，就可以把这段加入 `path`。

#### Step 7：为什么只靠现查回文还不够稳？

如果你在 DFS 过程中反复检查回文，同一段子串可能会被检查很多次。
例如 `s = "aaaaa..."` 时，很多重叠区间会被重复判断。

由于题目长度虽然只有 `16`，但这类题的更稳定思路是：

- 把“区间是否合法”这件事单独预处理好
- DFS 时只负责枚举切分，不再重复做字符比较

#### Step 8：怎样预处理所有回文区间？

定义 `pal[i][j]` 表示 `s[i:j+1]` 是否为回文。

状态转移很直接：

- 两端字符不同，一定不是回文
- 两端字符相同，并且中间子串也是回文，就成立

```python
for i in range(n - 1, -1, -1):
    for j in range(i, n):
        if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
            pal[i][j] = True
```

倒序枚举 `i`，是为了让 `pal[i + 1][j - 1]` 在使用前已经算好。

#### Step 9：有了回文表以后，回溯怎么写？

DFS 里不再现算回文，只看这段区间是否合法：

```python
if pal[start][end]:
    path.append(s[start:end + 1])
    dfs(end + 1)
    path.pop()
```

这时整道题已经被拆成两个职责清晰的模块：

- `pal` 负责回答“这段能不能切”
- `dfs` 负责回答“接下来怎么继续切”

#### Step 10：慢速走一条分支，看状态如何流动

还是看 `s = "aab"`。

开始时：

- `start = 0`
- `path = []`

尝试 `end = 0`：

- `s[0:1] = "a"`，是回文
- `path = ["a"]`
- 递归 `dfs(1)`

在 `dfs(1)` 里：

- `end = 1`，片段 `"a"` 合法
- `path = ["a", "a"]`
- 递归 `dfs(2)`

然后：

- `end = 2`，片段 `"b"` 合法
- `path = ["a", "a", "b"]`
- `dfs(3)` 命中终点，收集一组答案

回到起点后，还会尝试：

- `end = 1`，片段 `"aa"` 合法
- 所以又得到 `["aa", "b"]`

### Assemble the Full Code

下面把“回文表预处理 + 回溯枚举”拼成第一版完整代码。
这版代码可以直接运行。

```python
from typing import List


def partition_palindrome(s: str) -> List[List[str]]:
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                pal[i][j] = True

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not pal[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(partition_palindrome("aab"))
    print(partition_palindrome("a"))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                    pal[i][j] = True

        res: List[List[str]] = []
        path: List[str] = []

        def dfs(start: int) -> None:
            if start == n:
                res.append(path.copy())
                return
            for end in range(start, n):
                if not pal[start][end]:
                    continue
                path.append(s[start : end + 1])
                dfs(end + 1)
                path.pop()

        dfs(0)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它可以正式描述成：

- 回溯
- 字符串切分型搜索
- 动态规划预处理合法区间

但更值得固定下来的模型是：

- `dfs(start)` 负责切剩余后缀
- `pal[start][end]` 负责判断当前片段能不能切
- 每次只做一件事：挑一个合法片段，然后递归处理后面的部分

---

## E — Engineering（工程应用）

### 场景 1：词典约束切分候选生成（Python）

**背景**：搜索或推荐系统要把一个字符串切成若干合法词片段，供后续召回或纠错模块尝试。  
**为什么适用**：这和本题的结构完全一致，区别只是“回文”换成了“在词典里合法”。

```python
from typing import List, Set


def all_segmentations(s: str, vocab: Set[str]) -> List[List[str]]:
    n = len(s)
    valid = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            valid[i][j] = s[i : j + 1] in vocab

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not valid[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res


print(all_segmentations("applepen", {"apple", "app", "le", "pen"}))
```

### 场景 2：规则片段切分引擎（Go）

**背景**：日志清洗或协议解析时，先预处理哪些区间满足规则，再枚举所有可行切分方案。  
**为什么适用**：本题教的不是“回文”本身，而是“先算合法区间表，再 DFS 切分”。

```go
package main

import "fmt"

func partitions(s string, valid map[string]bool) [][]string {
	n := len(s)
	ok := make([][]bool, n)
	for i := 0; i < n; i++ {
		ok[i] = make([]bool, n)
		for j := i; j < n; j++ {
			ok[i][j] = valid[s[i:j+1]]
		}
	}

	res := make([][]string, 0)
	path := make([]string, 0, n)

	var dfs func(int)
	dfs = func(start int) {
		if start == n {
			snapshot := append([]string(nil), path...)
			res = append(res, snapshot)
			return
		}
		for end := start; end < n; end++ {
			if !ok[start][end] {
				continue
			}
			path = append(path, s[start:end+1])
			dfs(end + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}

func main() {
	fmt.Println(partitions("abc", map[string]bool{"a": true, "ab": true, "bc": true, "c": true}))
}
```

### 场景 3：前端文本玩法切分（JavaScript）

**背景**：做文字谜题、输入法实验或内容交互时，前端需要实时展示多种合法切分方案。  
**为什么适用**：只要“某段是否合法”能被预处理成布尔表，后面的 DFS 框架就完全一样。

```javascript
function segmentations(s, allow) {
  const n = s.length;
  const ok = Array.from({ length: n }, () => Array(n).fill(false));
  for (let i = 0; i < n; i += 1) {
    for (let j = i; j < n; j += 1) {
      ok[i][j] = allow.has(s.slice(i, j + 1));
    }
  }

  const res = [];
  const path = [];

  function dfs(start) {
    if (start === n) {
      res.push([...path]);
      return;
    }
    for (let end = start; end < n; end += 1) {
      if (!ok[start][end]) continue;
      path.push(s.slice(start, end + 1));
      dfs(end + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}

console.log(segmentations("level", new Set(["l", "e", "v", "eve", "level"])));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 回文预处理时间复杂度：`O(n^2)`
- DFS 枚举最坏时间复杂度：`O(n * 2^(n-1))`
  - 最坏情况下，几乎每个切割位置都可以切
  - 复制路径和切片会带来额外的线性成本
- 空间复杂度：`O(n^2 + n)`
  - `O(n^2)` 来自回文表
  - `O(n)` 来自递归栈和当前路径
  - 若计入输出，空间还要再加上所有答案本身

### 和几种常见思路对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 回溯 + 回文表预处理 | 先算所有合法区间，再 DFS | 结构清晰，重复判断少 | 需要多一个 `O(n^2)` 表 |
| 回溯 + 现查回文 | 每次切片时双指针判断 | 代码短，容易起步 | 重复判断多，思路不够稳 |
| 只做 DP 不回溯 | 只判断能否切分 | 适合判存在性 | 不能枚举全部方案 |

### 这题最容易错的地方

- 把 `start` 理解成“当前片段长度”，导致递归语义混乱
- 没有先判合法片段，就一股脑往下递归
- 回文表遍历顺序写反，导致 `pal[i + 1][j - 1]` 还没算出来
- 用 `path.append(...)` 之后忘记 `pop()`，后面的切分被污染

## 常见问题与注意事项

### 为什么这里适合先做回文预处理？

因为 DFS 会多次访问重叠区间。
把“`s[i:j+1]` 是否回文”提前算成表后，DFS 每次只需要 `O(1)` 查询，不必重复双指针比较。

### 这题是不是也能不用 DP？

能。

如果你只是想先写出一个能过的小数据版本，可以在 DFS 中现查回文。
但从模板稳定性和后续迁移角度，预处理合法区间是更值得掌握的写法。

### 这题和 `139. 单词拆分` 的关系是什么？

二者结构非常接近：

- 都是“从起点开始切一个合法片段”
- 都需要判断区间是否合法

区别在于：

- `131` 要枚举**所有方案**
- `139` 常常只关心**是否可行**

## 最佳实践与建议

- 遇到“字符串切分 + 局部合法性判断”时，优先考虑“合法区间预处理 + DFS”
- 先把 `dfs(start)` 的语义说清楚，再决定状态表长什么样
- 任何切分题都先问自己：当前片段合法条件能不能单独缓存
- 画出 `"aab"` 这类小例子的递归树，比直接背代码可靠得多

---

## S — Summary（总结）

- `131. 分割回文串` 的本质是“字符串切分型回溯”
- `start` 表示当前待切分后缀的起点，`path` 表示已经切出的片段
- 把回文判定预处理成 `pal[i][j]` 后，DFS 就能专注于枚举切分路径
- 这题最有迁移价值的不是“回文”本身，而是“先算合法区间，再递归枚举”
- 学会这题后，很多字符串切分和区间合法性搜索题都会明显顺手

### 推荐延伸阅读

- `132. 分割回文串 II`：从“枚举所有方案”转到“求最少切分次数”
- `139. 单词拆分`：同样是合法区间切分，但目标变成可达性判断
- `78. 子集`：理解基础回溯树结构
- `17. 电话号码的字母组合`：固定层数的字符串 DFS

### 行动建议

做完这题以后，不妨再回头看一遍“字符串切分”类问题。
如果你已经能把“合法区间表”和“后缀 DFS”这两层职责拆清楚，很多看起来不同的题其实已经归到同一套模板里了。

---

## 多语言实现

### Python

```python
from typing import List


def partition_palindrome(s: str) -> List[List[str]]:
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                pal[i][j] = True

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not pal[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char*** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static char* clone_str(const char* s) {
    size_t len = strlen(s);
    char* copy = malloc(len + 1);
    memcpy(copy, s, len + 1);
    return copy;
}

static char* clone_range(const char* s, int left, int right) {
    int len = right - left + 1;
    char* copy = malloc(len + 1);
    memcpy(copy, s + left, len);
    copy[len] = '\0';
    return copy;
}

static void push_result(Result* res, char** path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(char**) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    char** row = malloc(sizeof(char*) * path_size);
    for (int i = 0; i < path_size; ++i) {
        row[i] = clone_str(path[i]);
    }
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(const char* s, int n, int start, bool* pal, char** path, int path_size, Result* res) {
    if (start == n) {
        push_result(res, path, path_size);
        return;
    }
    for (int end = start; end < n; ++end) {
        if (!pal[start * n + end]) {
            continue;
        }
        path[path_size] = clone_range(s, start, end);
        dfs(s, n, end + 1, pal, path, path_size + 1, res);
        free(path[path_size]);
    }
}

char*** partition(char* s, int* returnSize, int** returnColumnSizes) {
    int n = (int)strlen(s);
    bool* pal = calloc(n * n, sizeof(bool));
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            if (s[i] == s[j] && (j - i <= 2 || pal[(i + 1) * n + (j - 1)])) {
                pal[i * n + j] = true;
            }
        }
    }

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(char**) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    char** path = malloc(sizeof(char*) * n);
    dfs(s, n, 0, pal, path, 0, &res);

    free(path);
    free(pal);
    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.data;
}
```

### C++

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<string>> partition(string s) {
        int n = (int)s.size();
        vector<vector<bool>> pal(n, vector<bool>(n, false));
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (s[i] == s[j] && (j - i <= 2 || pal[i + 1][j - 1])) {
                    pal[i][j] = true;
                }
            }
        }

        vector<vector<string>> res;
        vector<string> path;
        dfs(s, 0, pal, path, res);
        return res;
    }

private:
    void dfs(const string& s, int start, const vector<vector<bool>>& pal, vector<string>& path, vector<vector<string>>& res) {
        if (start == (int)s.size()) {
            res.push_back(path);
            return;
        }
        for (int end = start; end < (int)s.size(); ++end) {
            if (!pal[start][end]) {
                continue;
            }
            path.push_back(s.substr(start, end - start + 1));
            dfs(s, end + 1, pal, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func partition(s string) [][]string {
	n := len(s)
	pal := make([][]bool, n)
	for i := 0; i < n; i++ {
		pal[i] = make([]bool, n)
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] && (j-i <= 2 || pal[i+1][j-1]) {
				pal[i][j] = true
			}
		}
	}

	res := make([][]string, 0)
	path := make([]string, 0, n)

	var dfs func(int)
	dfs = func(start int) {
		if start == n {
			snapshot := append([]string(nil), path...)
			res = append(res, snapshot)
			return
		}
		for end := start; end < n; end++ {
			if !pal[start][end] {
				continue
			}
			path = append(path, s[start:end+1])
			dfs(end + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}
```

### Rust

```rust
impl Solution {
    pub fn partition(s: String) -> Vec<Vec<String>> {
        let bytes = s.as_bytes();
        let n = bytes.len();
        let mut pal = vec![vec![false; n]; n];

        for i in (0..n).rev() {
            for j in i..n {
                if bytes[i] == bytes[j] && (j - i <= 2 || pal[i + 1][j - 1]) {
                    pal[i][j] = true;
                }
            }
        }

        fn dfs(bytes: &[u8], start: usize, pal: &Vec<Vec<bool>>, path: &mut Vec<String>, res: &mut Vec<Vec<String>>) {
            if start == bytes.len() {
                res.push(path.clone());
                return;
            }
            for end in start..bytes.len() {
                if !pal[start][end] {
                    continue;
                }
                path.push(String::from_utf8(bytes[start..=end].to_vec()).unwrap());
                dfs(bytes, end + 1, pal, path, res);
                path.pop();
            }
        }

        let mut res: Vec<Vec<String>> = Vec::new();
        let mut path: Vec<String> = Vec::new();
        dfs(bytes, 0, &pal, &mut path, &mut res);
        res
    }
}
```

### JavaScript

```javascript
/**
 * @param {string} s
 * @return {string[][]}
 */
var partition = function (s) {
  const n = s.length;
  const pal = Array.from({ length: n }, () => Array(n).fill(false));

  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = i; j < n; j += 1) {
      if (s[i] === s[j] && (j - i <= 2 || pal[i + 1][j - 1])) {
        pal[i][j] = true;
      }
    }
  }

  const res = [];
  const path = [];

  function dfs(start) {
    if (start === n) {
      res.push([...path]);
      return;
    }
    for (let end = start; end < n; end += 1) {
      if (!pal[start][end]) continue;
      path.push(s.slice(start, end + 1));
      dfs(end + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
};
```
