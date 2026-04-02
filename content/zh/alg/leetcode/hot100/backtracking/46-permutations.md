---
title: "Hot100：全排列（Permutations）used[] 状态回溯模板 ACERS 解析"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "全排列", "DFS", "used数组", "LeetCode 46"]
description: "围绕 LeetCode 46 全排列，讲清 used[] 状态控制、叶子收集答案与多语言实现。"
keywords: ["Permutations", "全排列", "回溯", "used", "DFS", "LeetCode 46", "Hot100"]
---

> **副标题 / 摘要**  
> 如果说子集题教你“组合型回溯”的骨架，那么全排列题教你的就是“状态型回溯”的核心：当前位置要选一个还没用过的元素，直到路径长度等于 `n` 才收集答案。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`回溯`、`全排列`、`DFS`  
- **SEO 关键词**：Permutations, 全排列, 回溯, used, DFS  
- **元描述**：通过 LeetCode 46 固定排列型回溯模板，重点理解 used[]、叶子收集与多语言实现。  

---

## 目标读者

- 已经做完 `78. 子集`，准备进入排列型回溯的学习者
- 会写递归，但状态恢复经常出错的开发者
- 需要枚举任务执行顺序、测试顺序或操作序列的工程师

## 背景 / 动机

排列问题和组合问题最本质的区别是：

- 组合只关心“选了哪些元素”
- 排列还关心“这些元素出现的顺序”

所以在全排列里，`[1,2,3]` 和 `[1,3,2]` 是两个不同答案。  
这意味着你不能再靠 `startIndex` 只向后看，而必须显式记录“哪些元素已经用过”。

LeetCode 46 的价值就在这里：它把“状态恢复”这件事讲得非常干净。

## 核心概念

- **`path`**：当前构造中的排列
- **`used[i]`**：`nums[i]` 是否已经被当前路径使用
- **叶子收集答案**：只有当路径长度等于 `nums.length` 时，才得到一个完整排列
- **状态撤销**：递归返回时同时撤销 `path` 和 `used`

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个不含重复数字的数组 `nums`，返回它的所有可能全排列。  
答案顺序不限。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 不含重复元素的整数数组 |
| 返回 | int[][] | 所有可能的全排列 |

### 示例 1

```text
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

### 示例 2

```text
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

### 示例 3

```text
输入：nums = [1]
输出：[[1]]
```

### 提示

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- `nums` 中所有整数互不相同

---

## C — Concepts（核心思想）

### 从子集题到全排列题，模板哪里变了

`78. 子集` 的关键是 `startIndex`，因为组合不关心顺序。  
但全排列不同，每一层都可以从“所有还没用过的元素”里选一个，所以：

- 不能使用 `startIndex`
- 必须维护一个 `used[]`
- 只有走到叶子时才收集答案

### 搜索树模型

以 `nums = [1,2,3]` 为例：

```text
[]
|- [1]
|  |- [1,2]
|  |  |- [1,2,3]
|  |- [1,3]
|     |- [1,3,2]
|- [2]
|- [3]
```

和子集题不同的是，中间节点不是完整排列。  
只有路径长度到达 `n`，才说明所有位置都填满了。

### 最稳定的模板

```text
dfs():
    if path 长度 == n:
        收集答案
        return
    for i in [0 .. n-1]:
        if used[i]:
            continue
        选 nums[i]
        used[i] = true
        dfs()
        used[i] = false
        撤销 nums[i]
```

---

## 实践指南 / 步骤

1. 准备答案数组 `ans`、路径数组 `path`、布尔数组 `used`
2. 进入 `dfs` 后先看路径是否已经凑满
3. 遍历所有下标，如果当前元素没被使用过，就加入路径
4. 递归进入下一层
5. 返回时恢复 `used[i]` 和 `path`

## 可运行示例（Python）

```python
from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    ans: List[List[int]] = []
    path: List[int] = []
    used = [False] * len(nums)

    def dfs() -> None:
        if len(path) == len(nums):
            ans.append(path.copy())
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return ans


if __name__ == "__main__":
    print(permute([1, 2, 3]))
```

## 解释与原理

### 为什么不能用 `startIndex`

因为排列要求顺序不同也算不同答案。  
如果你只允许下一层从后面开始选，那么 `[2,1]` 这种排列永远不可能出现。

### 为什么要在叶子收集答案

排列要求每个位置都确定下来。  
中间状态如 `[1,2]` 还只是一个前缀，不是完整排列。

### 这题真正训练的是什么

不是“写一个 DFS”，而是训练你对下面两个状态的同步恢复：

- 路径状态：`path.append` / `path.pop`
- 使用状态：`used[i] = True` / `used[i] = False`

---

## E — Engineering（工程应用）

### 场景 1：任务执行顺序枚举（Python）

**背景**：离线调度器要比较几个任务不同执行顺序带来的结果差异。  
**为什么适用**：任务顺序不同就可能产生不同效果，天然是排列问题。

```python
def orders(tasks):
    if not tasks:
        return [[]]
    res = []
    for i, task in enumerate(tasks):
        for rest in orders(tasks[:i] + tasks[i + 1:]):
            res.append([task] + rest)
    return res


print(orders(["fetch", "score", "notify"]))
```

### 场景 2：接口回归顺序试跑（Go）

**背景**：同一组接口按不同调用顺序可能触发不同缓存 / 状态路径。  
**为什么适用**：要验证顺序敏感性时，排列枚举最直接。

```go
package main

import "fmt"

func permute(items []string) [][]string {
	if len(items) == 0 {
		return [][]string{{}}
	}
	res := make([][]string, 0)
	for i, item := range items {
		rest := append([]string{}, items[:i]...)
		rest = append(rest, items[i+1:]...)
		for _, tail := range permute(rest) {
			res = append(res, append([]string{item}, tail...))
		}
	}
	return res
}

func main() {
	fmt.Println(permute([]string{"login", "query", "logout"}))
}
```

### 场景 3：前端动画播放顺序枚举（JavaScript）

**背景**：UI 原型阶段想尝试多个动画步骤的排列顺序。  
**为什么适用**：顺序变化直接产生不同体验。

```javascript
function permute(items) {
  if (items.length === 0) return [[]];
  const res = [];
  for (let i = 0; i < items.length; i += 1) {
    const rest = items.slice(0, i).concat(items.slice(i + 1));
    for (const tail of permute(rest)) {
      res.push([items[i], ...tail]);
    }
  }
  return res;
}

console.log(permute(["fade", "scale", "slide"]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(n * n!)`
- 递归栈和 `used` 辅助空间：`O(n)`
- 若计入输出，整体空间同样受 `n!` 级答案规模影响

### 和子集题的对比

| 题目 | 本质 | 收集时机 | 关键状态 |
| --- | --- | --- | --- |
| 78 子集 | 组合 | 每个节点 | `startIndex` |
| 46 全排列 | 排列 | 叶子节点 | `used[]` |

### 常见错误

- 忘记恢复 `used[i]`
- 把收集答案写在递归入口，导致拿到半成品路径
- 以为 `startIndex` 也能解决排列，结果漏掉顺序不同的答案

## 常见问题与注意事项

### `used[]` 能不能省掉

在无重复元素的全排列里，最稳定的写法就是 `used[]`。  
也可以用“交换到当前位置”的原地回溯，但可读性和迁移性通常不如 `used[]` 版本。

### 下一步该学什么

做完这题后，很适合去做：

- `17. 电话号码的字母组合`：固定层数 DFS
- `39. 组合总和`：组合型回溯 + 剪枝

## 最佳实践与建议

- 排列题优先想“每个位置填什么”
- 组合题和排列题不要混用同一套边界思维
- 恢复现场时，路径状态和辅助状态要成对撤销
- 画一棵三层搜索树，比死记代码更可靠

---

## S — Summary（总结）

- 全排列题的关键不在递归本身，而在 `used[]` 状态控制
- 排列题只在叶子收集答案，因为只有叶子才是完整结果
- 与子集题相比，全排列更强调“状态恢复”
- 这题写熟之后，很多顺序型搜索题都会变得容易很多

### 推荐延伸阅读

- `78. 子集`：组合型回溯模板
- `17. 电话号码的字母组合`：固定层数 DFS
- `47. 全排列 II`：加入重复元素后的判重技巧
- `51. N 皇后`：更复杂的状态约束搜索

### 行动建议

如果你今天已经做完了 `78. 子集`，这题就该是第二题。  
把 `startIndex` 和 `used[]` 的差异说清楚，你的回溯理解会扎实很多。

---

## 多语言实现

### Python

```python
from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []
    used = [False] * len(nums)

    def dfs() -> None:
        if len(path) == len(nums):
            res.append(path.copy())
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return res
```

### C

```c
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    int** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static void push_result(Result* res, int* path, int n) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    int* row = malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) row[i] = path[i];
    res->data[res->size] = row;
    res->col_sizes[res->size] = n;
    res->size += 1;
}

static void dfs(int* nums, int n, bool* used, int* path, int depth, Result* res) {
    if (depth == n) {
        push_result(res, path, n);
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (used[i]) continue;
        used[i] = true;
        path[depth] = nums[i];
        dfs(nums, n, used, path, depth + 1, res);
        used[i] = false;
    }
}

int** permute(int* nums, int nums_size, int* return_size, int** return_column_sizes) {
    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    bool* used = calloc(nums_size, sizeof(bool));
    int* path = malloc(sizeof(int) * nums_size);
    dfs(nums, nums_size, used, path, 0, &res);

    free(used);
    free(path);
    *return_size = res.size;
    *return_column_sizes = res.col_sizes;
    return res.data;
}
```

### C++

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> path;
        vector<int> used(nums.size(), 0);
        dfs(nums, used, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& nums, vector<int>& used, vector<int>& path, vector<vector<int>>& res) {
        if ((int)path.size() == (int)nums.size()) {
            res.push_back(path);
            return;
        }
        for (int i = 0; i < (int)nums.size(); ++i) {
            if (used[i]) continue;
            used[i] = 1;
            path.push_back(nums[i]);
            dfs(nums, used, path, res);
            path.pop_back();
            used[i] = 0;
        }
    }
};
```

### Go

```go
package main

func permute(nums []int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0, len(nums))
	used := make([]bool, len(nums))

	var dfs func()
	dfs = func() {
		if len(path) == len(nums) {
			snapshot := append([]int(nil), path...)
			res = append(res, snapshot)
			return
		}
		for i, x := range nums {
			if used[i] {
				continue
			}
			used[i] = true
			path = append(path, x)
			dfs()
			path = path[:len(path)-1]
			used[i] = false
		}
	}

	dfs()
	return res
}
```

### Rust

```rust
fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
    fn dfs(nums: &[i32], used: &mut [bool], path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if path.len() == nums.len() {
            res.push(path.clone());
            return;
        }
        for i in 0..nums.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            path.push(nums[i]);
            dfs(nums, used, path, res);
            path.pop();
            used[i] = false;
        }
    }

    let mut res = Vec::new();
    let mut path = Vec::new();
    let mut used = vec![false; nums.len()];
    dfs(&nums, &mut used, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function permute(nums) {
  const res = [];
  const path = [];
  const used = new Array(nums.length).fill(false);

  function dfs() {
    if (path.length === nums.length) {
      res.push([...path]);
      return;
    }
    for (let i = 0; i < nums.length; i += 1) {
      if (used[i]) continue;
      used[i] = true;
      path.push(nums[i]);
      dfs();
      path.pop();
      used[i] = false;
    }
  }

  dfs();
  return res;
}
```
