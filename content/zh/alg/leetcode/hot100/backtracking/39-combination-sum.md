---
title: "Hot100：组合总和（回溯 / 剪枝 / 可重复选取）ACERS 解析"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "组合", "剪枝", "DFS", "LeetCode 39"]
description: "围绕 LeetCode 39 讲清组合型回溯、可重复选取、remain 状态与排序剪枝，并用一步一步构建代码的方式把模板真正学会。"
keywords: ["Combination Sum", "组合总和", "回溯", "剪枝", "remain", "LeetCode 39", "Hot100"]
---

> **副标题 / 摘要**
> 组合总和是 Hot100 回溯阶段里第一道真正把“组合模板 + 目标约束 + 排序剪枝”揉在一起的题。你不应该直接背结论，而应该顺着问题一步步推出：为什么要有 `path`，为什么要有 `remain`，为什么下一层仍然从 `i` 开始。

- **预计阅读时长**：14~16 分钟
- **标签**：`Hot100`、`回溯`、`组合`、`剪枝`  
- **SEO 关键词**：Combination Sum, 组合总和, 回溯, 剪枝, remain, DFS
- **元描述**：通过 LeetCode 39 建立组合型回溯加剪枝模板，理解可重复选取、remain 语义与排序后的安全剪枝。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个无重复元素的整数数组 `candidates` 和一个目标整数 `target`，
找出 `candidates` 中所有可以使数字和为 `target` 的不同组合，并以列表形式返回。

你可以无限次选取同一个候选数字。
如果两个组合中某个数字出现次数不同，则它们被视为不同组合。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| candidates | int[] | 无重复元素的候选数组 |
| target | int | 目标和 |
| 返回 | int[][] | 所有和为 `target` 的不同组合 |

### 示例 1

```text
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
```

### 示例 2

```text
输入：candidates = [2,3,5], target = 8
输出：[[2,2,2,2],[2,3,3],[3,5]]
```

### 示例 3

```text
输入：candidates = [2], target = 1
输出：[]
```

### 约束

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- `candidates` 中所有元素互不相同
- `1 <= target <= 40`
- 官方保证满足条件的不同组合数少于 `150`

---

## 目标读者

- 已经做过 `78. 子集`，现在准备学习“带目标约束”的回溯写法
- 知道回溯大概长什么样，但一碰到“数字可重复选取”就容易写错边界的学习者
- 想把这题真正写成模板，而不是只记住 `dfs(i, remain - x)` 这一行的开发者

## 背景 / 动机

`39. 组合总和` 很适合作为回溯第二阶段的训练题，因为它让你同时处理三件事：

- 它仍然是组合问题，所以顺序不重要
- 同一个数字可以重复选取，所以边界不能照搬子集
- 有 `target` 这个目标值，所以排序以后可以做安全剪枝

很多人第一次做这题时，代码也许能凑出来，但脑子里并没有稳定模型。
真正应该建立的是这条思路链：

- 我正在构造哪一部分答案
- 我离目标还差多少
- 下一层应该从哪里继续
- 什么情况下可以立刻停掉后续搜索

这四件事想清楚，`39`、`40`、`216` 这一串题就会顺很多。

## 核心概念

- **`path`**：当前已经选进组合里的数字
- **`remain`**：距离 `target` 还差多少
- **`startIndex` / `start`**：当前层允许从哪个位置开始选，保证组合顺序不重复
- **可重复选取**：如果当前选了 `candidates[i]`，下一层仍然从 `i` 开始
- **排序剪枝**：如果升序数组里的当前值已经大于 `remain`，后面的值也不可能成功

---

## C — Concepts（核心思想）

### 先看朴素做法暴露出的难点

如果你一开始只是想“把所有可能都试一遍，再筛出和等于 `target` 的组合”，很快就会遇到两个问题：

- 你没有一个稳定的方式避免 `[2,2,3]` 和 `[2,3,2]` 这种顺序重复
- 你没有一个便宜的条件尽早停下无效分支

这就是为什么这题不能只用“暴力枚举”来理解。
真正稳定的模型是：

- 用 `path` 表示当前正在构造的组合
- 用 `remain` 表示还差多少
- 用 `start` 保证后续只从当前及其右侧继续选
- 用排序后的 `break` 把不可能成功的分支整段砍掉

### 这道题是怎么一步一步推出来的

#### Step 1：先从一个最小但不平凡的例子开始

先看 `candidates = [2,3,6,7], target = 7`。

我们真正要做的不是“瞬间想出整套回溯模板”，而是先把目标说清楚：

- 如果先选 `2`，那还差 `5`
- 如果再选 `2`，那还差 `3`
- 如果再选 `3`，刚好命中目标，得到 `[2,2,3]`
- 如果一开始直接选 `7`，也能得到 `[7]`

这个例子马上告诉我们两件事：

- 题目关心的是组合，不是排列
- 我们需要持续跟踪“还差多少”

#### Step 2：当前部分答案最少要记住什么？

既然我们是在“逐步构造一个组合”，那当前部分答案就必须被显式保存下来。
这就是 `path` 的来源。

```python
path = []
```

`path` 的含义很明确：

- 它不是最终答案全集
- 它只是当前这一条递归路径上已经选中的数字

#### Step 3：怎样知道自己离目标还差多少？

如果每次都重新计算 `sum(path)`，逻辑会更散，也更不利于剪枝。
更稳定的办法是直接把“剩余目标”作为状态传下去。

```python
def dfs(start: int, remain: int) -> None:
    ...
```

这里的 `remain` 表示：

- 当前路径还差多少才能凑到 `target`
- 每选一个数，就把这个数从 `remain` 里减掉

#### Step 4：怎样避免把同一组数按不同顺序重复统计？

因为这是组合问题，所以 `[2,2,3]` 和 `[2,3,2]` 只能算一组。
最稳定的控制方式是：每一层只能从 `start` 开始往右枚举。

```python
for i in range(start, len(candidates)):
    x = candidates[i]
```

这样做的含义是：

- 当前层之前的数字，不允许回头再选
- 组合按非递减顺序构造，自然不会产生顺序重复

#### Step 5：什么时候说明一条路径已经成功？

当 `remain == 0` 时，说明当前 `path` 的和已经正好等于 `target`。
这时就可以收集答案。

```python
if remain == 0:
    res.append(path.copy())
    return
```

这里必须用 `path.copy()`，因为 `path` 之后还会继续被修改。

#### Step 6：为什么要先排序？

排序不是为了去重，而是为了让剪枝变得安全。
只要数组是升序的，一旦当前值已经大于 `remain`，后面的值只会更大。

```python
candidates.sort()
```

这一步之后，我们才有资格在循环里写 `break`。

#### Step 7：当前候选值过大时该怎么办？

因为已经排序，所以一旦 `x > remain`，当前层后面的值都不用再看了。

```python
if x > remain:
    break
```

这里是 `break`，不是 `continue`。
`continue` 只会跳过当前数字，但后面的数字更大，根本不可能成功。

#### Step 8：选中一个数之后，状态要怎样推进？

当我们选择 `x` 时，要做三件事：

1. 把 `x` 放进 `path`
2. 递归求解更小的问题
3. 返回后撤销这次选择

```python
path.append(x)
dfs(i, remain - x)
path.pop()
```

这里最关键的是 `dfs(i, remain - x)` 里的 `i`：

- 传 `i`：表示当前数字还能继续选
- 传 `i + 1`：就变成“每个数字只能用一次”的另一道题了

#### Step 9：慢速走一条分支，看看状态怎么流动

还是用 `candidates = [2,3,6,7], target = 7`。

开始时：

- `path = []`
- `remain = 7`
- `start = 0`

选择 `2`：

- `path = [2]`
- `remain = 5`
- 下一层仍然从索引 `0` 开始，因为 `2` 还能继续用

再次选择 `2`：

- `path = [2,2]`
- `remain = 3`

再看当前层：

- 先试 `2`，得到 `remain = 1`
- 此时下一个候选 `2` 已经大于 `1`，当前层直接 `break`
- 回退到 `path = [2,2]`

接着试 `3`：

- `path = [2,2,3]`
- `remain = 0`
- 收集答案 `[2,2,3]`

然后回溯，继续尝试其他分支。
这条路径把这题最重要的三件事都展示出来了：

- 为什么要传 `remain`
- 为什么下一层仍然从 `i` 开始
- 为什么排序后可以直接 `break`

### Assemble the Full Code

下面把前面的碎片拼成第一版完整代码。
这版代码已经是可直接运行、可打印示例结果的版本。

```python
from typing import List


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:
                break

            path.append(x)
            dfs(i, remain - x)
            path.pop()

    dfs(0, target)
    return res


if __name__ == "__main__":
    print(combination_sum([2, 3, 6, 7], 7))
    print(combination_sum([2, 3, 5], 8))
```

### Reference Answer

如果你要提交到 LeetCode，可以把它整理成更贴近提交环境的版本：

```python
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int, remain: int) -> None:
            if remain == 0:
                res.append(path.copy())
                return

            for i in range(start, len(candidates)):
                x = candidates[i]
                if x > remain:
                    break

                path.append(x)
                dfs(i, remain - x)
                path.pop()

        dfs(0, target)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 回溯
- 组合型搜索
- 排序后剪枝
- 同一候选值可重复选取

但顺序很重要。
不是先背“这是回溯 + 剪枝”，而是先从题目事实出发，把这些状态和规则一步步推出来，最后你才会发现：这其实就是一个很稳定的回溯模板。

---

## E — Engineering（工程应用）

### 场景 1：预算凑整组合（Python）

**背景**：若干固定面额可以重复使用，想找出所有刚好凑满预算的组合。
**为什么适用**：本质就是“候选值可重复选取，目标和固定”的搜索。

```python
def fill_budget(costs, target):
    costs = sorted(costs)
    res = []
    path = []

    def dfs(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(costs)):
            if costs[i] > remain:
                break
            path.append(costs[i])
            dfs(i, remain - costs[i])
            path.pop()

    dfs(0, target)
    return res


print(fill_budget([2, 3, 5], 8))
```

### 场景 2：资源包容量拼装（Go）

**背景**：后端服务需要从若干可重复使用的容量包中拼出满足目标容量的所有方案。
**为什么适用**：容量包可重复选，总容量必须精确命中目标。

```go
package main

import (
	"fmt"
	"sort"
)

func fill(capacities []int, target int) [][]int {
	sort.Ints(capacities)
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if remain == 0 {
			res = append(res, append([]int(nil), path...))
			return
		}
		for i := start; i < len(capacities); i++ {
			if capacities[i] > remain {
				break
			}
			path = append(path, capacities[i])
			dfs(i, remain-capacities[i])
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}

func main() {
	fmt.Println(fill([]int{2, 3, 5}, 8))
}
```

### 场景 3：套餐价格拼装器（JavaScript）

**背景**：前端套餐配置器希望列出所有价格刚好命中预算的组合方案。
**为什么适用**：价格项可以重复选，且顺序无关。

```javascript
function combinationSum(candidates, target) {
  candidates.sort((a, b) => a - b);
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (remain === 0) {
      res.push([...path]);
      return;
    }
    for (let i = start; i < candidates.length; i += 1) {
      const x = candidates[i];
      if (x > remain) break;
      path.push(x);
      dfs(i, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}

console.log(combinationSum([2, 3, 5], 8));
```

---

## R — Reflection（反思与深入）

### 正确性直觉

这套写法正确，主要因为三条不变量一直被维护：

- `path` 中的数字顺序非递减，所以不会把同一组合按不同顺序重复统计
- `remain` 始终等于“距离目标还差多少”，当它变成 `0` 时当前路径必然是合法答案
- 排序后只要 `x > remain`，后面的值也都不可能成功，直接 `break` 不会漏解

### 复杂度分析

令：

- `n = len(candidates)`
- `m = min(candidates)`
- `d = target / m`

那么递归深度最多是 `d`，因为每次至少会让 `remain` 减少 `m`。
一个宽松但常用的上界是：

- 时间复杂度：`O(n^d)`，其中 `d = target / min(candidates)`
- 递归额外空间：`O(d)`，不包含答案输出本身

这不是一道拥有整齐闭式复杂度的题，实际运行时间会强烈依赖剪枝效果和有效答案数量。

### 常见问题 / FAQ

#### 为什么下一层传 `i`，不是 `i + 1`？

因为题目允许重复使用当前数字。
传 `i + 1` 会把“可重复选取”误写成“每个数字只能用一次”。

#### 为什么这里能 `break`，不是 `continue`？

因为数组已经排序。
如果当前值都大于 `remain` 了，后面只会更大，没有必要继续看。

#### 为什么不直接维护 `sum(path)`？

也可以，但 `remain` 更直接：

- 判断完成只要看 `remain == 0`
- 判断非法或剪枝只要比较 `x > remain`
- 不需要反复计算路径和

### 常见错误

- 忘记先排序，却仍然使用 `if x > remain: break`
- 收集答案时写成 `res.append(path)`，导致后续回溯把历史答案一起改掉
- 递归参数写成 `dfs(i + 1, remain - x)`，结果把题目做成了 `40. Combination Sum II` 风格
- 把这题当排列题写，从索引 `0` 反复枚举，导致顺序重复

## Best Practices

- 写代码前先问自己：`path` 表示什么，`remain` 表示什么，`start` 控制什么
- 对“可重复选取”的题，先检查递归参数是不是应该继续传当前下标
- 只在“排序已经完成”的前提下使用 `break` 剪枝
- 如果今天刚学完 `78. 子集`，一定要把这题和 `78` 对照着写一遍

## 参考与延伸阅读

- 官方题目：<https://leetcode.cn/problems/combination-sum/>
- 推荐下一题：`40. Combination Sum II`
- 推荐对照题：`78. 子集`、`46. 全排列`

---

## S — Summary（总结）

- 这题的本质是组合型回溯，不是排列型回溯
- `remain` 让“命中目标”和“剪枝条件”都变得非常清楚
- `dfs(i, remain - x)` 里的 `i` 正是“可重复选取”的代码体现
- 排序后使用 `if x > remain: break`，是这题最稳定也最重要的剪枝

### 建议下一步

- `78. 子集`：回头复盘 `startIndex` 为什么能去掉顺序重复
- `46. 全排列`：感受“组合型回溯”和“排列型回溯”的边界差异
- `40. Combination Sum II`：学习“不能重复选 + 有重复值时如何去重”
- `216. Combination Sum III`：继续练目标约束搜索

### CTA

今天如果只练一道回溯题，就把这题从空白开始再手写一遍。
重点不要背最终代码，而是要求自己一步步说出：为什么需要 `path`、`remain`、`start` 和排序剪枝。

---

## Multi-Language Implementations

### Python

```python
from typing import List


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:
                break
            path.append(x)
            dfs(i, remain - x)
            path.pop()

    dfs(0, target)
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

static int cmp_int(const void* a, const void* b) {
    return (*(const int*)a) - (*(const int*)b);
}

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

static void dfs(int* candidates, int candidates_size, int start, int remain,
                int* path, int path_size, Result* res) {
    if (remain == 0) {
        push_result(res, path, path_size);
        return;
    }

    for (int i = start; i < candidates_size; ++i) {
        int x = candidates[i];
        if (x > remain) {
            break;
        }
        path[path_size] = x;
        dfs(candidates, candidates_size, i, remain - x, path, path_size + 1, res);
    }
}

int** combinationSum(int* candidates, int candidatesSize, int target,
                     int* returnSize, int** returnColumnSizes) {
    qsort(candidates, candidatesSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.rows = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * (target + 1));
    dfs(candidates, candidatesSize, 0, target, path, 0, &res);
    free(path);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.rows;
}
```

### C++

```cpp
#include <algorithm>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res;
        vector<int> path;
        dfs(candidates, 0, target, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& candidates, int start, int remain,
             vector<int>& path, vector<vector<int>>& res) {
        if (remain == 0) {
            res.push_back(path);
            return;
        }

        for (int i = start; i < static_cast<int>(candidates.size()); ++i) {
            int x = candidates[i];
            if (x > remain) {
                break;
            }
            path.push_back(x);
            dfs(candidates, i, remain - x, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

import "sort"

func combinationSum(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if remain == 0 {
			res = append(res, append([]int(nil), path...))
			return
		}

		for i := start; i < len(candidates); i++ {
			x := candidates[i]
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(i, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}
```

### Rust

```rust
fn combination_sum(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(
        candidates: &[i32],
        start: usize,
        remain: i32,
        path: &mut Vec<i32>,
        res: &mut Vec<Vec<i32>>,
    ) {
        if remain == 0 {
            res.push(path.clone());
            return;
        }

        for i in start..candidates.len() {
            let x = candidates[i];
            if x > remain {
                break;
            }
            path.push(x);
            dfs(candidates, i, remain - x, path, res);
            path.pop();
        }
    }

    candidates.sort_unstable();
    let mut res = Vec::new();
    let mut path = Vec::new();
    dfs(&candidates, 0, target, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function combinationSum(candidates, target) {
  candidates.sort((a, b) => a - b);
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (remain === 0) {
      res.push([...path]);
      return;
    }

    for (let i = start; i < candidates.length; i += 1) {
      const x = candidates[i];
      if (x > remain) break;
      path.push(x);
      dfs(i, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}
```
