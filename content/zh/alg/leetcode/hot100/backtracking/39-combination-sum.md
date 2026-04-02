---
title: "Hot100：组合总和（Combination Sum）回溯剪枝 / 可重复选取 ACERS 解析"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "组合", "剪枝", "DFS", "LeetCode 39"]
description: "围绕 LeetCode 39 讲清可重复选取、排序剪枝与组合型回溯实现。"
keywords: ["Combination Sum", "组合总和", "回溯", "剪枝", "DFS", "LeetCode 39", "Hot100"]
---

> **副标题 / 摘要**  
> 组合总和是回溯专题里第一道真正把“组合模板 + 目标约束 + 剪枝”揉在一起的题。你要学会的不只是枚举，而是怎样用排序和剩余值 `remain` 把搜索树收紧。

- **预计阅读时长**：12~15 分钟  
- **标签**：`Hot100`、`回溯`、`组合`、`剪枝`  
- **SEO 关键词**：Combination Sum, 组合总和, 回溯, 剪枝, DFS  
- **元描述**：通过 LeetCode 39 建立组合型回溯加剪枝模板，理解可重复选取、排序与 remain 约束。  

---

## 目标读者

- 已经做过 `78. 子集`，准备把回溯模板升级到“带约束搜索”的学习者
- 想搞清楚“同一个数可以重复使用”时递归边界怎么写的开发者
- 需要做资源打包、预算组合、规格拼装类组合搜索的工程师

## 背景 / 动机

这题是很多人真正开始理解“回溯不是暴力乱搜”的分水岭。

因为它同时有三件事：

- 仍然是组合问题，所以要保持顺序无关
- 候选数字可以重复使用
- 目标和 `target` 给了你天然剪枝条件

如果你只会硬搜，代码虽然也许能过，但模板不稳定。  
而一旦你把“排序 + remain + 从 `i` 开始递归”的逻辑想清楚，这一类题都会顺很多。

## 核心概念

- **`path`**：当前正在尝试的一组组合
- **`remain`**：当前还差多少才能凑到目标值
- **从 `i` 继续递归**：表示当前数字可以重复使用
- **排序剪枝**：若 `candidates[i] > remain`，后面的数更大，可直接停止

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个无重复元素的整数数组 `candidates` 和一个目标值 `target`，  
找出所有和为 `target` 的不同组合。

同一个候选数字可以被重复选取。  
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

### 提示

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- `candidates` 的所有元素互不相同
- `1 <= target <= 40`
- 官方保证满足条件的不同组合数少于 `150`

---

## C — Concepts（核心思想）

### 这题为什么仍然是“组合型”回溯

虽然可以重复选数字，但顺序仍然不重要。  
`[2,2,3]` 和 `[2,3,2]` 表示的是同一个组合，不应该重复统计。

所以边界设计仍然要坚持组合型写法：

- 本层从 `startIndex` 开始枚举
- 下一层仍从当前 `i` 开始，而不是 `i + 1`

这里的差别恰好对应“当前数字能否重复使用”：

- `dfs(i + 1, ...)`：下次不能再选自己
- `dfs(i, ...)`：下次还可以继续选自己

### 为什么先排序

排序不是为了去重，而是为了剪枝。

一旦数组升序：

- 如果当前数已经大于 `remain`
- 后面的数只会更大
- 那么整段循环都可以直接 `break`

### 最稳定的模板

```text
sort(candidates)

dfs(start, remain):
    if remain == 0:
        收集答案
        return
    for i in [start .. n-1]:
        if candidates[i] > remain:
            break
        选 candidates[i]
        dfs(i, remain - candidates[i])
        撤销 candidates[i]
```

---

## 实践指南 / 步骤

1. 先对 `candidates` 排序
2. 维护答案数组 `ans`、路径数组 `path`
3. 进入 `dfs(startIndex, remain)` 后，先判断是否已经凑满
4. 遍历当前可选候选数
5. 如果当前数已经大于 `remain`，直接停止后续枚举
6. 选中当前数，递归时仍传 `i`，因为允许重复使用
7. 递归返回后撤销选择

## 可运行示例（Python）

```python
from typing import List


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    ans: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            ans.append(path.copy())
            return
        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:
                break
            path.append(x)
            dfs(i, remain - x)
            path.pop()

    dfs(0, target)
    return ans


if __name__ == "__main__":
    print(combination_sum([2, 3, 6, 7], 7))
    print(combination_sum([2, 3, 5], 8))
```

## 解释与原理

### 为什么递归时传 `i` 而不是 `i + 1`

因为题目明确允许同一个数字重复选取。  
如果传 `i + 1`，当前数字只能用一次，就把题目做成了另一题。

### 为什么 `remain` 很重要

`remain` 让搜索过程具备清晰的“剩余目标”语义。  
你不用每次都重新计算路径和，也更容易写出剪枝条件。

### 这题最关键的剪枝是什么

排序后，若 `candidates[i] > remain`，后续更大的数字也不可能成功。  
这就是最稳定、最便宜的一刀剪枝。

---

## E — Engineering（工程应用）

### 场景 1：预算组合搜索（Python）

**背景**：已知若干固定成本项，想找出所有能刚好凑满预算的选项组合。  
**为什么适用**：本质就是“候选值可重复使用、目标和固定”的组合搜索。

```python
def fill_budget(costs, target):
    costs = sorted(costs)
    ans = []

    def dfs(start, remain, path):
        if remain == 0:
            ans.append(path[:])
            return
        for i in range(start, len(costs)):
            if costs[i] > remain:
                break
            path.append(costs[i])
            dfs(i, remain - costs[i], path)
            path.pop()

    dfs(0, target, [])
    return ans


print(fill_budget([2, 3, 5], 8))
```

### 场景 2：资源包规格拼装（Go）

**背景**：后台服务要从若干规格包中拼出满足目标容量的组合方案。  
**为什么适用**：规格包可重复选，且总容量必须命中目标。

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

### 场景 3：前端套餐拼装器（JavaScript）

**背景**：前端配置器要列出若干可行套餐，使价格正好命中用户预算。  
**为什么适用**：组合无序、可重复选项、目标和约束，完全同构。

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
      if (candidates[i] > remain) break;
      path.push(candidates[i]);
      dfs(i, remain - candidates[i]);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}
```

---

## R — Reflection（反思与深入）

### 复杂度分析

这题没有一个像 `2^n` 或 `n!` 那样整齐的固定答案。  
它的搜索规模取决于：

- 候选值大小
- `target` 大小
- 有多少条路径会被剪掉

因此更稳妥的表述是：

- 时间复杂度：最坏情况下呈指数级，且明显依赖输出规模
- 递归深度：最多约为 `target / min(candidates)`
- 若计入答案，空间也受输出规模影响

### 替代方案对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 回溯 + 剪枝 | 枚举组合并提前停止不可能分支 | 最适合输出全部组合 | 最坏情况仍可能很大 |
| 动态规划 | 更适合判断是否可达或计数 | 对存在性/计数类问题友好 | 不适合直接恢复全部组合 |

### 常见错误

- 递归传 `i + 1`，误把题目做成“每个数只能用一次”
- 不排序就写 `break` 剪枝，逻辑不成立
- 用路径和反复求和，导致代码又慢又乱

## 常见问题与注意事项

### 为什么这题和子集题都属于组合型回溯

因为顺序不重要。  
我们关心的是“选了哪些数、各选了几次”，而不是它们进入路径的排列顺序。

### 和 `40. 组合总和 II` 的差别是什么

`39` 允许同一个数重复使用；  
`40` 通常每个位置只能使用一次，而且要处理输入里可能出现的重复数字。  
两题模板相近，但边界和判重逻辑不同。

## 最佳实践与建议

- 先排序，再谈剪枝
- 用 `remain` 表示目标约束，比每次求和清晰得多
- 遇到“可重复选”时，先想递归边界是不是该继续传 `i`
- 写这类题时，把“为什么这里能 break”说出来，代码会更稳

---

## S — Summary（总结）

- 组合总和是组合型回溯加剪枝的经典模板题
- 排序的主要价值是让 `x > remain` 的剪枝成立
- 允许重复选取时，递归边界要继续传当前下标 `i`
- 这题写熟后，再做 `40 / 216 / 377` 一类题会轻松很多

### 推荐延伸阅读

- `78. 子集`：组合型回溯起点
- `17. 电话号码的字母组合`：固定层数 DFS
- `40. 组合总和 II`：单次使用 + 判重
- `216. 组合总和 III`：固定长度 + 固定和

### 行动建议

如果你今天按 `78 -> 46 -> 17 -> 39` 的顺序学，这题正好是第四题。  
做到这里，你的回溯模板已经从“骨架”升级到“能带约束搜索”的水平了。

---

## 多语言实现

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
    int** data;
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
        res->data = realloc(res->data, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    int* row = malloc(sizeof(int) * path_size);
    for (int i = 0; i < path_size; ++i) row[i] = path[i];
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(int* candidates, int n, int start, int remain, int* path, int depth, Result* res) {
    if (remain == 0) {
        push_result(res, path, depth);
        return;
    }
    for (int i = start; i < n; ++i) {
        if (candidates[i] > remain) break;
        path[depth] = candidates[i];
        dfs(candidates, n, i, remain - candidates[i], path, depth + 1, res);
    }
}

int** combinationSum(int* candidates, int candidatesSize, int target, int* returnSize, int** returnColumnSizes) {
    qsort(candidates, candidatesSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int path[40];
    dfs(candidates, candidatesSize, 0, target, path, 0, &res);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.data;
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
    void dfs(const vector<int>& candidates, int start, int remain, vector<int>& path, vector<vector<int>>& res) {
        if (remain == 0) {
            res.push_back(path);
            return;
        }
        for (int i = start; i < (int)candidates.size(); ++i) {
            if (candidates[i] > remain) break;
            path.push_back(candidates[i]);
            dfs(candidates, i, remain - candidates[i], path, res);
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
			snapshot := append([]int(nil), path...)
			res = append(res, snapshot)
			return
		}
		for i := start; i < len(candidates); i++ {
			if candidates[i] > remain {
				break
			}
			path = append(path, candidates[i])
			dfs(i, remain-candidates[i])
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
    candidates.sort();

    fn dfs(candidates: &[i32], start: usize, remain: i32, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if remain == 0 {
            res.push(path.clone());
            return;
        }
        for i in start..candidates.len() {
            if candidates[i] > remain {
                break;
            }
            path.push(candidates[i]);
            dfs(candidates, i, remain - candidates[i], path, res);
            path.pop();
        }
    }

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
      if (candidates[i] > remain) break;
      path.push(candidates[i]);
      dfs(i, remain - candidates[i]);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}
```
