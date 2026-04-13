---
title: "Hot100：电话号码的字母组合（Letter Combinations of a Phone Number）固定层数 DFS ACERS 解析"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "字符串", "DFS", "电话键盘", "LeetCode 17"]
description: "围绕 LeetCode 17 讲清固定层数 DFS、数字到字母映射与多语言实现。"
keywords: ["Letter Combinations of a Phone Number", "电话号码的字母组合", "回溯", "DFS", "LeetCode 17", "Hot100"]
---

> **副标题 / 摘要**
> 这题表面上像字符串题，实质上是一个非常标准的固定层数回溯模型：第 `k` 层只处理第 `k` 个数字，从其映射字母里选一个，直到路径长度等于输入长度。

- **预计阅读时长**：10~12 分钟
- **标签**：`Hot100`、`回溯`、`字符串`、`DFS`
- **SEO 关键词**：Letter Combinations of a Phone Number, 电话号码的字母组合, 回溯, DFS
- **元描述**：用 LeetCode 17 建立固定层数 DFS 模板，理解字符映射、路径长度终止与多语言实现。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个仅由数字 `2` 到 `9` 组成的字符串 `digits`，返回它能表示的所有字母组合。
答案顺序不限。数字与字母的映射与电话按键一致，`1` 不对应任何字母。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| digits | string | 由 `2` 到 `9` 组成的数字串 |
| 返回 | string[] | 所有可能的字母组合 |

### 示例 1

```text
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### 示例 2

```text
输入：digits = "2"
输出：["a","b","c"]
```

### 提示

- `1 <= digits.length <= 4`
- `digits[i]` 是范围 `['2', '9']` 内的一个数字

---

## 目标读者

- 已经掌握 `78 / 46`，准备看另一类回溯树形态的学习者
- 想把“每层处理一个位置”这种 DFS 模型固定下来的开发者
- 需要做编码扩展、短串生成、候选串组合的工程师

## 背景 / 动机

这题和子集、排列都不太一样。

- 子集题：每层决定“要不要继续选后面的元素”
- 排列题：每层决定“当前位置放哪个未使用元素”
- 本题：每层对应一个固定数字位置，只能从该数字映射的字母中选一个

因此它非常适合训练“固定深度 DFS”：

- 递归层数由输入长度决定
- 每一层的候选集由当前字符直接决定
- 路径长度等于输入长度时结束

这类模型在字典枚举、编码扩展、模板字符串生成中很常见。

## 核心概念

- **数字映射表**：`2 -> abc`, `3 -> def`, ..., `9 -> wxyz`
- **固定层数 DFS**：第 `index` 层只处理 `digits[index]`
- **叶子条件**：`index == len(digits)` 时得到一个完整答案
- **路径构建**：每层向路径追加一个字符，返回时撤销

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先从最小例子看搜索树长什么样

看 `digits = "23"`。

这道题最自然的理解方式不是“字符串回溯”，而是：

- 第 0 层处理数字 `2`
- 第 1 层处理数字 `3`

因为电话键盘映射是固定的：

- `2 -> abc`
- `3 -> def`

所以搜索树会长成这样：

```text
[]
|- a
|  |- ad
|  |- ae
|  |- af
|- b
|  |- bd
|  |- be
|  |- bf
|- c
   |- cd
   |- ce
   |- cf
```

这个例子直接告诉我们：

- 每一层处理的是一个固定位置
- 当前层的候选集由当前数字决定

#### Step 2：我们首先需要什么固定信息？

既然每一层都要把数字映射成字母，那就先要准备一张映射表。
这是整道题最基础的静态信息。

```python
mapping = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}
```

#### Step 3：当前部分答案最少要记住什么？

如果我们在逐位构造一个字符串，就必须保存当前已经选出来的字符。
这就是 `path`。

```python
path = []
```

这里的 `path` 表示：

- 当前递归分支已经选中的字符序列
- 它最终会被拼成一个完整字符串

#### Step 4：递归真正要解决的子问题是什么？

当已经处理完前面若干位之后，剩下的问题就是：

> 现在轮到 `digits[index]`，我要从它对应的字母里挑一个，然后去处理下一位。

所以这里真正控制层数的变量是 `index`。

```python
def dfs(index: int) -> None:
    ...
```

#### Step 5：什么时候说明一条路径已经完成？

当 `index == len(digits)` 时，说明每一位数字都已经处理过了。
这时 `path` 正好组成一个完整答案。

```python
if index == len(digits):
    res.append("".join(path))
    return
```

这里收集答案时要 `join`，因为最终返回值是字符串数组，不是字符数组。

#### Step 6：当前层有哪些可选字符？

当前层的候选字母完全由 `digits[index]` 决定。

```python
letters = mapping[digits[index]]
for ch in letters:
    ...
```

这也是它和子集、排列最大的区别：

- 没有 `startIndex`
- 没有 `used[]`
- 候选集不是全局的，而是由当前位置决定的

#### Step 7：选中一个字符后，状态怎么推进？

如果当前选择 `ch`，就把它追加到 `path`，然后去处理下一位数字。

```python
path.append(ch)
dfs(index + 1)
```

这里的 `index + 1` 非常直观：

- 当前位已经填完
- 下一层去填下一位

#### Step 8：递归回来之后要撤销什么？

回溯回来以后，要把刚才选的字符弹出去，这样当前层才能继续试下一个候选字母。

```python
path.pop()
```

因为这里只有一个可变状态 `path`，所以撤销动作也更干净。

#### Step 9：慢速走一条分支

还是看 `digits = "23"`。

开始时：

- `path = []`
- `index = 0`

第 0 层处理数字 `2`，先选 `a`：

- `path = ["a"]`
- 递归到 `dfs(1)`

第 1 层处理数字 `3`，先选 `d`：

- `path = ["a", "d"]`
- 递归到 `dfs(2)`

现在 `index == len(digits)`，说明所有位置都填完了，收集 `"ad"`。
然后回溯：

- `pop()` 掉 `d`
- 继续尝试 `e`、`f`

等 `a` 这一支走完，再回到第 0 层尝试 `b`、`c`。
这就是整棵树的完整工作方式。

### Assemble the Full Code

把前面的碎片拼起来，第一版完整代码就是：

```python
from typing import List


def letter_combinations(digits: str) -> List[str]:
    if not digits:
        return []

    mapping = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res: List[str] = []
    path: List[str] = []

    def dfs(index: int) -> None:
        if index == len(digits):
            res.append("".join(path))
            return

        for ch in mapping[digits[index]]:
            path.append(ch)
            dfs(index + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(letter_combinations("23"))
    print(letter_combinations("2"))
```

### Reference Answer

整理成 LeetCode 提交版，可以写成：

```python
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        mapping = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        res: List[str] = []
        path: List[str] = []

        def dfs(index: int) -> None:
            if index == len(digits):
                res.append("".join(path))
                return

            for ch in mapping[digits[index]]:
                path.append(ch)
                dfs(index + 1)
                path.pop()

        dfs(0)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 固定层数 DFS
- 回溯
- 位置驱动的字符串构造

但真正需要固定下来的不是名词，而是这三个点：

- `index` 表示当前正在处理第几位
- 当前层候选集来自 `mapping[digits[index]]`
- 填完一位就递归到下一位，回来后撤销刚才选的字符

---

## E — Engineering（工程应用）

### 场景 1：短码候选串生成（Python）

**背景**：给定一串数字编码，系统要生成候选短串给后续检索使用。
**为什么适用**：每个数字位都映射到固定字符集，完全同构。

```python
def expand(code, mapping):
    if not code:
        return []
    res = [""]
    for ch in code:
        res = [prefix + c for prefix in res for c in mapping[ch]]
    return res


print(expand("23", {"2": "abc", "3": "def"}))
```

### 场景 2：服务标签编码展开（Go）

**背景**：一些内部编码系统用短数字串映射成多个标签组合候选。
**为什么适用**：每一位都有固定候选集合，适合按位 DFS。

```go
package main

import "fmt"

func expand(code string, mp map[byte]string) []string {
	res := []string{""}
	for i := 0; i < len(code); i++ {
		next := make([]string, 0)
		for _, prefix := range res {
			for _, ch := range mp[code[i]] {
				next = append(next, prefix+string(ch))
			}
		}
		res = next
	}
	return res
}

func main() {
	fmt.Println(expand("23", map[byte]string{'2': "abc", '3': "def"}))
}
```

### 场景 3：前端手机号助记提示（JavaScript）

**背景**：前端输入组件想给短数字串展示可记忆字符候选。
**为什么适用**：按位展开即可得到所有候选串。

```javascript
function expand(code, mapping) {
  let res = [""];
  for (const digit of code) {
    const next = [];
    for (const prefix of res) {
      for (const ch of mapping[digit]) {
        next.push(prefix + ch);
      }
    }
    res = next;
  }
  return res;
}

console.log(expand("23", { 2: "abc", 3: "def" }));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 设输入长度为 `n`
- 最坏情况下每位最多对应 4 个字母
- 时间复杂度可写作 `O(4^n * n)`
  生成每个字符串时最终都要拼接出长度为 `n` 的结果
- 递归栈空间为 `O(n)`，若计入输出则与答案规模同阶

### 替代方案对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| DFS 回溯 | 一位一位向下构造 | 模板清晰，最适合后续迁移 | 需要理解递归层含义 |
| BFS 队列 | 逐层扩展已有字符串 | 迭代直观 | 对更复杂搜索题迁移性较弱 |

### 常见错误

- 把这一题也写成 `startIndex` 组合模板
- 叶子条件写错成 `len(path) == len(mapping[digits[index]])`
- 忘记当前层对应的是“数字位置”，不是“字符位置”

## 常见问题与注意事项

### 这题为什么不需要 `used[]`

因为每层选的不是“还没用过的元素”，而是“当前这个数字能映射出的一个字母”。
层和层之间天然按位置推进，不存在重复使用同一数字槽位的问题。

### 它和排列题的共性是什么

共性在于：

- 都是走到叶子再收集答案
- 都要维护一条路径

差别在于：

- 排列题的候选集来自“未使用元素”
- 本题候选集来自“当前数字的映射字符”

## 最佳实践与建议

- 优先把这题理解为“固定深度树”，而不是“字符串拼接题”
- 路径尽量用字符数组维护，叶子时再 `join`
- 看到“每一位都有若干候选字符”时，优先联想到这题模板
- 学完这题再做 `39. 组合总和`，更容易看懂剪枝

---

## S — Summary（总结）

- 这题是固定层数 DFS 的标准模板题
- 每一层只处理一个数字位置，候选来自映射表
- 叶子条件是“所有数字位都处理完”
- 学会这题后，很多字符串枚举 / 编码扩展问题都能快速迁移

### 推荐延伸阅读

- `78. 子集`：组合型回溯
- `46. 全排列`：状态型回溯
- `39. 组合总和`：可重复选 + 剪枝
- `22. 括号生成`：固定长度构造 + 约束剪枝

### 行动建议

如果你今天已经做了 `78` 和 `46`，这题是第三题非常合适。
它能帮你把“回溯树不一定都长一样”这件事真正建立起来。

---

## 多语言实现

### Python

```python
from typing import List


def letter_combinations(digits: str) -> List[str]:
    if not digits:
        return []

    mapping = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res: List[str] = []
    path: List[str] = []

    def dfs(index: int) -> None:
        if index == len(digits):
            res.append("".join(path))
            return
        for ch in mapping[digits[index]]:
            path.append(ch)
            dfs(index + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdlib.h>
#include <string.h>

static const char* map_table[10] = {
    "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
};

typedef struct {
    char** data;
    int size;
    int capacity;
} Result;

static void push_result(Result* res, const char* path) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(char*) * res->capacity);
    }
    res->data[res->size] = strdup(path);
    res->size += 1;
}

static void dfs(const char* digits, int n, int index, char* path, Result* res) {
    if (index == n) {
        path[index] = '\0';
        push_result(res, path);
        return;
    }
    const char* letters = map_table[digits[index] - '0'];
    for (int i = 0; letters[i] != '\0'; ++i) {
        path[index] = letters[i];
        dfs(digits, n, index + 1, path, res);
    }
}

char** letterCombinations(char* digits, int* returnSize) {
    if (digits[0] == '\0') {
        *returnSize = 0;
        return NULL;
    }

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(char*) * res.capacity);

    int n = (int)strlen(digits);
    char path[5];
    dfs(digits, n, 0, path, &res);

    *returnSize = res.size;
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
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        vector<string> mp = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> res;
        string path;
        dfs(digits, 0, mp, path, res);
        return res;
    }

private:
    void dfs(const string& digits, int index, const vector<string>& mp, string& path, vector<string>& res) {
        if (index == (int)digits.size()) {
            res.push_back(path);
            return;
        }
        const string& letters = mp[digits[index] - '0'];
        for (char ch : letters) {
            path.push_back(ch);
            dfs(digits, index + 1, mp, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func letterCombinations(digits string) []string {
	if digits == "" {
		return []string{}
	}

	mp := map[byte]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}

	res := make([]string, 0)
	path := make([]byte, 0, len(digits))

	var dfs func(int)
	dfs = func(index int) {
		if index == len(digits) {
			res = append(res, string(path))
			return
		}
		for i := 0; i < len(mp[digits[index]]); i++ {
			path = append(path, mp[digits[index]][i])
			dfs(index + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}
```

### Rust

```rust
fn letter_combinations(digits: String) -> Vec<String> {
    if digits.is_empty() {
        return vec![];
    }

    let mp = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];
    let chars: Vec<char> = digits.chars().collect();
    let mut res = Vec::new();
    let mut path = String::new();

    fn dfs(chars: &[char], index: usize, mp: &[&str; 10], path: &mut String, res: &mut Vec<String>) {
        if index == chars.len() {
            res.push(path.clone());
            return;
        }
        let letters = mp[chars[index].to_digit(10).unwrap() as usize];
        for ch in letters.chars() {
            path.push(ch);
            dfs(chars, index + 1, mp, path, res);
            path.pop();
        }
    }

    dfs(&chars, 0, &mp, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function letterCombinations(digits) {
  if (digits.length === 0) return [];

  const mapping = {
    2: "abc",
    3: "def",
    4: "ghi",
    5: "jkl",
    6: "mno",
    7: "pqrs",
    8: "tuv",
    9: "wxyz",
  };

  const res = [];
  const path = [];

  function dfs(index) {
    if (index === digits.length) {
      res.push(path.join(""));
      return;
    }
    for (const ch of mapping[digits[index]]) {
      path.push(ch);
      dfs(index + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}
```
