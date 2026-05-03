---
title: "LeetCode 216：组合总和 III（回溯 / 固定长度搜索）ACERS 解析"
date: 2026-04-17T14:31:14+08:00
draft: false
categories: ["LeetCode"]
tags: ["回溯", "组合总和 III", "固定长度", "剪枝", "DFS", "LeetCode 216"]
description: "围绕 LeetCode 216 讲清固定长度组合搜索、1..9 有界候选集与只能使用一次的回溯写法，并从题目本身一步一步推出最终模板。"
keywords: ["Combination Sum III", "组合总和 III", "回溯", "固定长度", "k 个数", "LeetCode 216"]
---

> **副标题 / 摘要**  
> `216. 组合总和 III` 给回溯再加了一条关键约束：不仅总和要命中，组合长度也必须恰好等于 `k`。这会把问题变成一个很干净的“固定长度组合搜索”。

- **预计阅读时长**：12~15 分钟  
- **标签**：`回溯`、`固定长度`、`剪枝`、`组合搜索`  
- **SEO 关键词**：Combination Sum III, 组合总和 III, 固定长度回溯, k 个数, 剪枝, LeetCode 216  
- **元描述**：从题目本身推导 LeetCode 216 的稳定解法，真正理解固定长度 `k`、有界候选集 `1..9` 与只能使用一次的回溯模型。  

---

## A — Algorithm（题目与算法）

### 题目还原

找出所有满足条件的组合，使得：

- 从 `1` 到 `9` 中选数
- 一共恰好选 `k` 个数
- 这些数的和等于 `n`
- 每个数最多只能使用一次

返回所有可能的合法组合。  
答案中不能包含重复组合，组合顺序不重要。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| k | int | 需要选出的数字个数 |
| n | int | 目标和 |
| 返回 | int[][] | 所有长度为 `k` 且和为 `n` 的合法组合 |

### 示例 1

```text
输入：k = 3, n = 7
输出：[[1,2,4]]
```

### 示例 2

```text
输入：k = 3, n = 9
输出：[[1,2,6],[1,3,5],[2,3,4]]
```

### 示例 3

```text
输入：k = 4, n = 1
输出：[]
```

### 约束

- `2 <= k <= 9`
- `1 <= n <= 60`

---

## 目标读者

- 已经理解 `39`、`40`，现在准备继续叠加“长度必须恰好是 `k`”这个新约束的学习者
- 会写基本 DFS，但经常把“目标和”与“固定长度”混在一起判断的读者
- 想掌握固定长度组合搜索模板的开发者

## 背景 / 动机

这道题很适合作为 `39` 和 `40` 之后的下一题，因为它保留了回溯骨架，但搜索空间进一步收紧了：

- 候选集固定为 `1..9`
- 每个数最多只能使用一次
- 组合长度必须恰好等于 `k`

最后这条非常关键。

在 `39` 里，只要总和命中就算成功。  
但在 `216` 里，即使总和提早命中，如果长度不对，也不能收集答案；  
反过来，即使已经选满 `k` 个数，如果总和还没命中，也必须失败返回。

所以这题必须同时看两件事：

- 还差多少和
- 当前已经选了多少个数

## 核心概念

- **`path`**：当前已经选中的数字
- **`remain`**：距离目标和 `n` 还差多少
- **`start`**：当前层允许尝试的最小数字
- **固定长度 `k`**：只有 `len(path) == k` 时，才有资格判断是不是完整答案
- **只能用一次**：选了 `x` 之后，下一层必须从 `x + 1` 开始
- **有界候选集**：候选值只可能是 `1..9`

---

## C — Concepts（核心思想）

### 这道题要怎么一步一步推出来？

#### Step 1：先看一个最能体现“固定长度”约束的小例子

看 `k = 3`，`n = 7`。

答案只有：

- `[1,2,4]`

这个最小例子立刻告诉我们：

- 数字必须互不重复
- 顺序不重要
- 和必须等于 `7`
- 长度必须恰好是 `3`

所以虽然 `[7]` 的和也等于 `7`，它依然是错误答案，因为长度不对。

#### Step 2：部分答案至少要记住什么？

我们还是在逐步构造一个组合，所以当前分支必须被显式保存下来：

```python
path = []
```

`path` 同时承担两层含义：

- 当前已经选了哪些数字
- 当前已经占用了多少个位置

#### Step 3：下一层要解决的“小问题”是什么？

和前面几道题一样，直接维护“还差多少”是最清晰的：

```python
def dfs(start: int, remain: int) -> None:
    ...
```

这样递归状态就变成：

- `start`：下一层最小能选到哪个数字
- `remain`：当前组合还差多少才能凑到 `n`
- `path`：已经选了多少个数

#### Step 4：什么时候说明一条分支已经完整？

这就是它和 `39` 最不一样的地方。

分支结构上“已经选满”的条件是：

```python
if len(path) == k:
    ...
```

在这一刻：

- 如果 `remain == 0`，说明既满足长度又满足和，可以收集答案
- 否则说明虽然选满了，但不合法，直接返回

```python
if len(path) == k:
    if remain == 0:
        res.append(path.copy())
    return
```

#### Step 5：这一层有哪些可选项？

这题不是从输入数组里选，而是固定从 `1..9` 中选。

所以当前层的枚举是：

```python
for x in range(start, 10):
    ...
```

这一步天然带来三个好处：

- 数字严格递增
- 每个数最多使用一次
- 组合不会因为顺序不同而重复

#### Step 6：选中一个数之后，状态怎样推进？

如果当前选择了 `x`，下一层就必须从 `x + 1` 开始。

```python
path.append(x)
dfs(x + 1, remain - x)
path.pop()
```

这和 `40` 的“每个位置只能使用一次”本质一样，只不过这里候选集本身就是 `1..9`。

#### Step 7：这题有哪些天然剪枝？

因为我们是从小到大枚举，一旦当前 `x > remain`，后面的值只会更大：

```python
if x > remain:
    break
```

另外，长度条件本身也是一种非常强的剪枝：

```python
if len(path) == k:
    ...
```

它保证任何分支都不会无限扩展下去。

#### Step 8：慢速走一条分支，看看状态怎么变化

看 `k = 3`，`n = 9`。

开始时：

- `path = []`
- `remain = 9`
- `start = 1`

选择 `1`：

- `path = [1]`
- `remain = 8`
- 下一层从 `2` 开始

选择 `2`：

- `path = [1,2]`
- `remain = 6`
- 下一层从 `3` 开始

选择 `6`：

- `path = [1,2,6]`
- `remain = 0`
- `len(path) == 3`

所以 `[1,2,6]` 被收集。

这一条分支已经把整套方法都展示出来了：

- 递增选择避免重复
- `remain` 管目标和
- `len(path)` 管固定长度

### Assemble the Full Code

先把已经解释过的状态和规则拼成一个完整、可运行的版本：

```python
from typing import List


def combination_sum_iii(k: int, n: int) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if len(path) == k:
            if remain == 0:
                res.append(path.copy())
            return

        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, n)
    return res


if __name__ == "__main__":
    print(combination_sum_iii(3, 7))
    print(combination_sum_iii(3, 9))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成更贴近提交环境的版本：

```python
from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int, remain: int) -> None:
            if len(path) == k:
                if remain == 0:
                    res.append(path.copy())
                return

            for x in range(start, 10):
                if x > remain:
                    break
                path.append(x)
                dfs(x + 1, remain - x)
                path.pop()

        dfs(1, n)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 回溯
- 固定长度组合搜索
- 只能使用一次的选择模型
- 在有界候选集 `1..9` 上做搜索

但不要从模板名字开始背。  
应该先从题目事实出发：

- 总和要命中
- 长度必须恰好为 `k`
- 数字递增且不重复

这些条件一旦说清楚，代码就自然会长成现在这个样子。

---

## E — Engineering（工程应用）

### 场景 1：固定名额预算包选择（Python）

**背景**：一个分析流程需要从一批一次性额度中恰好选 `k` 个，且总预算必须命中目标值。  
**为什么适用**：每个额度只能用一次，结果长度固定，总和必须精确相等。

```python
def choose_credits(k, target):
    res = []
    path = []

    def dfs(start, remain):
        if len(path) == k:
            if remain == 0:
                res.append(path[:])
            return
        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, target)
    return res


print(choose_credits(3, 9))
```

### 场景 2：固定槽位资源挑选（Go）

**背景**：后端分配器需要从一个很小的资源编号范围里选出恰好 `k` 个不同编号，并让总权重命中目标。  
**为什么适用**：候选集很小、每个值只能使用一次、并且结果长度固定。

```go
package main

import "fmt"

func chooseResources(k, target int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if len(path) == k {
			if remain == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}
		for x := start; x <= 9; x++ {
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(x+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(1, target)
	return res
}

func main() {
	fmt.Println(chooseResources(3, 7))
}
```

### 场景 3：前端固定数量筹码规划器（JavaScript）

**背景**：前端规划器要求用户恰好选 `k` 个筹码，每个筹码权重唯一，目标是达到某个积分值。  
**为什么适用**：值域固定、每个值只能使用一次，而且数量必须精确。

```javascript
function combinationSum3(k, n) {
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (path.length === k) {
      if (remain === 0) res.push([...path]);
      return;
    }

    for (let x = start; x <= 9; x += 1) {
      if (x > remain) break;
      path.push(x);
      dfs(x + 1, remain - x);
      path.pop();
    }
  }

  dfs(1, n);
  return res;
}

console.log(combinationSum3(3, 9));
```

---

## R — Reflection（反思与深入）

### 正确性直觉

这套写法之所以成立，是因为下面这些不变量一直被维护：

- 搜索始终从小到大递增，所以不会出现顺序重复
- `dfs(x + 1, ...)` 保证每个数字最多使用一次
- `remain` 始终表示当前路径还差多少和
- `len(path) == k` 明确限制了组合长度
- 因为候选值递增，`x > remain` 时可以安全结束当前层

### 复杂度分析

这题的候选集固定就是 `1..9`，所以整体搜索空间其实很小。

一种比较清晰的描述方式是：

- 它在 9 个数里搜索固定长度为 `k` 的组合
- 可以写成 `O(C(9, k) * k)` 量级，再加上收集答案时的复制开销
- 递归深度是 `O(k)`

相比一般的组合总和题，它更小的原因是：

- 候选集是固定且很小的
- 值没有重复
- 长度必须固定

### 常见问题 / FAQ

#### 为什么已经 `remain == 0` 了，还要检查 `len(path) == k`？

因为这题不是只看总和，还要求必须恰好选 `k` 个数。  
总和命中了，但长度不对，依然不能算答案。

#### 为什么这里不需要像 `40` 那样写去重条件？

因为候选值天然就是 `1,2,3,...,9`，本身没有重复值。  
没有重复输入，自然也不需要做“同层去重”。

#### 为什么递归时传的是 `x + 1`？

因为每个数最多只能使用一次，而且下一层必须继续向更大的数字推进，才能保持组合顺序。

### 常见错误

- 只要 `remain == 0` 就收集答案，忘了同时检查长度是否等于 `k`
- 递归时还写成 `dfs(x, ...)`，错误允许重复使用当前值
- 把这题误写成 `39` 那种“无限复用”的目标和问题
- 明明候选集固定很小，却把状态设计得过于复杂

## Best Practices

- 题目一旦出现“恰好选 `k` 个数”，就要立刻把长度条件写进 base case
- 这里的 `1..9` 不是实现细节，而是题目结构的一部分
- 最适合把 `216` 和 `39`、`40` 放在一起对照，理解每条约束是怎么映射成代码变化的
- 在有界候选集里，优先使用递增顺序带来的天然剪枝

## 参考与延伸阅读

- 官方题目：<https://leetcode.cn/problems/combination-sum-iii/>
- 对照题：`39. 组合总和`、`40. 组合总和 II`
- 相关固定长度组合题：`77. 组合`

---

## S — Summary（总结）

- `216` 的本质是固定长度组合搜索，不只是目标和搜索
- `len(path) == k` 是答案条件的一部分，不是额外补丁
- `dfs(x + 1, remain - x)` 同时表达了“只能用一次”和“保持组合顺序”
- 候选集固定为 `1..9`，让这题比一般组合总和题更小也更适合练模板

### 建议下一步练习

- 把 `39`、`40`、`216` 连着重写一遍，对比 base case 和递归参数的变化
- 然后继续做 `90. 子集 II` 或 `77. 组合`，巩固去重和固定长度两条思路

### CTA

试着不用“这是一道标准回溯题”来解释它。  
如果你能把它说成“从 `1..9` 里递增地选出恰好 `k` 个不同数字，让总和等于 `n`”，那这道题的代码通常就已经在你脑子里成型了。

---

## Multi-Language Implementations

### Python

```python
from typing import List


def combination_sum_iii(k: int, n: int) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if len(path) == k:
            if remain == 0:
                res.append(path.copy())
            return

        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, n)
    return res
```

### C

```c
#include <stdlib.h>

typedef struct {
    int** rows;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static void push_result(Result* res, int* path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->rows = realloc(res->rows, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }

    int* row = malloc(sizeof(int) * path_size);
    for (int i = 0; i < path_size; ++i) {
        row[i] = path[i];
    }

    res->rows[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(int k, int start, int remain, int* path, int path_size, Result* res) {
    if (path_size == k) {
        if (remain == 0) {
            push_result(res, path, path_size);
        }
        return;
    }

    for (int x = start; x <= 9; ++x) {
        if (x > remain) {
            break;
        }
        path[path_size] = x;
        dfs(k, x + 1, remain - x, path, path_size + 1, res);
    }
}

int** combinationSum3(int k, int n, int* returnSize, int** returnColumnSizes) {
    Result res = {0};
    res.capacity = 16;
    res.rows = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int path[9];
    dfs(k, 1, n, path, 0, &res);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.rows;
}
```

### C++

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(k, 1, n, path, res);
        return res;
    }

private:
    void dfs(int k, int start, int remain,
             vector<int>& path, vector<vector<int>>& res) {
        if (static_cast<int>(path.size()) == k) {
            if (remain == 0) {
                res.push_back(path);
            }
            return;
        }

        for (int x = start; x <= 9; ++x) {
            if (x > remain) {
                break;
            }
            path.push_back(x);
            dfs(k, x + 1, remain - x, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func combinationSum3(k int, n int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if len(path) == k {
			if remain == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}

		for x := start; x <= 9; x++ {
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(x+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(1, n)
	return res
}
```

### Rust

```rust
fn combination_sum_3(k: i32, n: i32) -> Vec<Vec<i32>> {
    fn dfs(
        k: usize,
        start: i32,
        remain: i32,
        path: &mut Vec<i32>,
        res: &mut Vec<Vec<i32>>,
    ) {
        if path.len() == k {
            if remain == 0 {
                res.push(path.clone());
            }
            return;
        }

        for x in start..=9 {
            if x > remain {
                break;
            }
            path.push(x);
            dfs(k, x + 1, remain - x, path, res);
            path.pop();
        }
    }

    let mut res = Vec::new();
    let mut path = Vec::new();
    dfs(k as usize, 1, n, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function combinationSum3(k, n) {
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (path.length === k) {
      if (remain === 0) res.push([...path]);
      return;
    }

    for (let x = start; x <= 9; x += 1) {
      if (x > remain) break;
      path.push(x);
      dfs(x + 1, remain - x);
      path.pop();
    }
  }

  dfs(1, n);
  return res;
}
```
