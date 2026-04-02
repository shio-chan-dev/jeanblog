---
title: "Hot100：子集（Subsets）回溯枚举 / startIndex 模板 ACERS 解析"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "子集", "DFS", "startIndex", "LeetCode 78"]
description: "围绕 LeetCode 78 子集，讲清回溯树、startIndex 边界与多语言可运行实现。"
keywords: ["Subsets", "子集", "回溯", "startIndex", "幂集", "LeetCode 78", "Hot100"]
---

> **副标题 / 摘要**  
> 子集是 Hot100 回溯专题里最适合打地基的一题。真正要固定下来的不是“把答案都列出来”，而是 `path`、`startIndex` 和“每个节点都是答案”这三个核心不变式。

- **预计阅读时长**：10~12 分钟  
- **标签**：`Hot100`、`回溯`、`子集`、`DFS`  
- **SEO 关键词**：Subsets, 子集, 回溯, startIndex, 幂集  
- **元描述**：用 LeetCode 78 子集建立最稳定的回溯模板，含工程场景、复杂度分析与多语言实现。  

---

## 目标读者

- 刚进入 Hot100 回溯专题、想先把模板打稳的学习者
- 能写 DFS，但还没真正理解“组合”和“排列”区别的开发者
- 希望把枚举思路迁移到配置组合、策略试跑场景的工程师

## 背景 / 动机

“列出所有可能组合”在工程里并不少见。  
比如功能开关组合试跑、权限策略候选集生成、前端筛选项预设等，本质上都在做“从若干候选元素里列出所有选择结果”。

这类问题最容易犯的错有两个：

- 把组合写成排列，导致重复答案
- 把“什么时候收集答案”放错位置，导致漏解

LeetCode 78 的价值就在于：它约束足够简单，没有重复元素，也不要求复杂剪枝，适合你先把回溯树的骨架搭稳。

## 核心概念

- **`path`**：当前递归路径上已经选中的元素
- **`startIndex`**：下一层从哪里开始选，保证组合不会倒序重复
- **前序收集答案**：子集题里，每个节点本身就是一个合法答案
- **回溯撤销**：递归返回后，要把刚加入 `path` 的元素弹出

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个元素互不相同的整数数组 `nums`，返回它的所有可能子集。  
结果中不能包含重复子集，返回顺序不限。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | int[] | 元素互不相同的整数数组 |
| 返回 | int[][] | 所有可能的子集 |

### 示例 1

```text
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

### 示例 2

```text
输入：nums = [0]
输出：[[],[0]]
```

### 提示

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中所有元素互不相同

---

## C — Concepts（核心思想）

### 为什么它是回溯入门题

这题没有“目标和”、没有“判重数组”、也没有棋盘边界。  
你只需要想清楚一件事：

> 从当前位置开始，我可以选后面的任意一个元素，然后把选择继续向下展开。

因此它特别适合先把回溯模板中的三个角色固定下来：

1. `path` 负责保存当前决策
2. `startIndex` 负责限定下一层的可选范围
3. `ans.append(path.copy())` 负责在每个节点收集答案

### 搜索树该怎么理解

以 `nums = [1,2,3]` 为例：

```text
[]
|- [1]
|  |- [1,2]
|  |  |- [1,2,3]
|  |- [1,3]
|- [2]
|  |- [2,3]
|- [3]
```

这棵树里的每个节点都表示“当前已经选出的一个子集”，  
所以空集、单元素子集、双元素子集、全集都应该被记录。

### 方法类型

回溯 + 组合枚举。

### 最稳定的模板

```text
dfs(start):
    先收集当前 path
    for i in [start .. n-1]:
        选 nums[i]
        dfs(i + 1)
        撤销 nums[i]
```

这里的 `i + 1` 很关键，它表示“后面的层不能再回头选前面的元素”，  
因此 `[1,2]` 会被生成一次，但 `[2,1]` 不会再出现。

---

## 实践指南 / 步骤

1. 准备结果数组 `ans` 和路径数组 `path`
2. 定义 `dfs(startIndex)`
3. 每次进入 `dfs`，先把当前 `path` 复制到答案中
4. 从 `startIndex` 开始枚举候选元素
5. 选中一个元素后递归到下一层，下一层从 `i + 1` 开始
6. 递归返回后弹出元素，恢复现场

## 可运行示例（Python）

```python
from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    ans: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        ans.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return ans


if __name__ == "__main__":
    print(subsets([1, 2, 3]))
    print(subsets([0]))
```

运行方式示例：

```bash
python3 subsets.py
```

## 解释与原理

### 为什么每个节点都要收集

因为子集题没有“必须选满 k 个”或者“必须凑成 target”这种终点条件。  
只要当前 `path` 合法，它就是一个答案。

### 为什么必须复制 `path`

`path` 是同一个可变数组，会不断 append / pop。  
如果直接把它本身塞进结果数组，后续修改会把之前答案一起改掉。

### 为什么要用 `startIndex`

如果没有 `startIndex`，每层都从 `0` 开始枚举，你得到的就不再是组合，而是排列式重复结果。  
子集题要的是“是否选择某个元素”，不是“元素出现顺序”。

---

## E — Engineering（工程应用）

### 场景 1：功能开关组合试跑（Python）

**背景**：你有几组灰度开关，想生成所有候选开关组合做小流量验证。  
**为什么适用**：这和“列出所有子集”完全同构。

```python
def all_toggle_sets(toggles):
    ans = [[]]
    for name in toggles:
        ans += [old + [name] for old in ans]
    return ans


print(all_toggle_sets(["new-ui", "cache-v2", "risk-guard"]))
```

### 场景 2：策略模块候选集生成（Go）

**背景**：后台风控系统要枚举不同策略模块组合，离线评估命中效果。  
**为什么适用**：每个模块可选或不选，天然就是子集问题。

```go
package main

import "fmt"

func subsets(items []string) [][]string {
	res := [][]string{{}}
	for _, item := range items {
		size := len(res)
		for i := 0; i < size; i++ {
			next := append([]string{}, res[i]...)
			next = append(next, item)
			res = append(res, next)
		}
	}
	return res
}

func main() {
	fmt.Println(subsets([]string{"ruleA", "ruleB", "ruleC"}))
}
```

### 场景 3：前端筛选预设生成（JavaScript）

**背景**：前端页面要预生成若干筛选器组合，做演示或回归测试。  
**为什么适用**：筛选项的开关组合本质上就是幂集。

```javascript
function subsets(items) {
  const res = [[]];
  for (const item of items) {
    const size = res.length;
    for (let i = 0; i < size; i += 1) {
      res.push([...res[i], item]);
    }
  }
  return res;
}

console.log(subsets(["tag", "price", "stock"]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：`O(n * 2^n)`  
  子集总数是 `2^n`，复制路径的总成本与答案规模同阶。
- 空间复杂度：递归栈 `O(n)`，若计入输出则为 `O(n * 2^n)`

### 替代方案对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 回溯 | 路径递归展开 | 模板统一，最适合迁移到后续题 | 需要理解搜索树 |
| 位运算 | 用二进制位表示选或不选 | 写法短，适合离线枚举 | 可读性弱，不利于迁移到复杂回溯 |
| 迭代扩展 | 对已有答案批量加新元素 | 简洁直观 | 对“剪枝 / 约束”类题扩展性较弱 |

### 常见错误

- 只在叶子节点收集答案，漏掉大量合法子集
- 把 `path` 直接 append 到结果里，导致结果被后续修改污染
- 下一层仍从 `0` 枚举，得到重复顺序结果

## 常见问题与注意事项

### 子集为什么不需要 `used[]`

因为元素是否能再选，不是靠“当前层之前有没有用过”控制，  
而是靠 `startIndex` 保证后面的层只向后看。

### 什么时候该从这题升级到下一题

当你能稳定回答下面四个问题，就可以继续做 `46 全排列`：

- `path` 表示什么
- 为什么每个节点都收集答案
- `startIndex` 为什么是 `i + 1`
- 回溯时撤销了什么状态

## 最佳实践与建议

- 把“组合类回溯”统一写成 `dfs(startIndex)` 模板
- 收集答案时一律复制路径，不要共享可变数组
- 先画搜索树，再写代码，能明显降低出错率
- 学完这题后，立刻衔接 `46 / 17 / 39`，模板差异最清楚

---

## S — Summary（总结）

- 子集题是回溯模板里最适合打地基的一题
- `startIndex` 决定这是组合，不是排列
- 子集题的答案收集时机是“每个节点”，不是“只在叶子”
- 学会这题后，后续的组合、剪枝、固定层数 DFS 都更容易迁移

### 推荐延伸阅读

- `46. 全排列`：加入 `used[]`，理解排列型回溯
- `39. 组合总和`：加入目标值与剪枝
- `90. 子集 II`：处理重复元素时的层内判重
- `77. 组合`：固定长度组合的经典模板

### 行动建议

今天如果你准备正式进入回溯专题，先把这题写到能脱稿，再去做 `46. 全排列`。  
这比一开始就上复杂剪枝题更稳。

---

## 多语言实现

### Python

```python
from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

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

static void dfs(int* nums, int nums_size, int start, int* path, int path_size, Result* res) {
    push_result(res, path, path_size);
    for (int i = start; i < nums_size; ++i) {
        path[path_size] = nums[i];
        dfs(nums, nums_size, i + 1, path, path_size + 1, res);
    }
}

int** subsets(int* nums, int nums_size, int* return_size, int** return_column_sizes) {
    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * nums_size);
    dfs(nums, nums_size, 0, path, 0, &res);
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
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(nums, 0, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& nums, int start, vector<int>& path, vector<vector<int>>& res) {
        res.push_back(path);
        for (int i = start; i < (int)nums.size(); ++i) {
            path.push_back(nums[i]);
            dfs(nums, i + 1, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func subsets(nums []int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]int(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(nums); i++ {
			path = append(path, nums[i])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}
```

### Rust

```rust
fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    fn dfs(nums: &[i32], start: usize, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        res.push(path.clone());
        for i in start..nums.len() {
            path.push(nums[i]);
            dfs(nums, i + 1, path, res);
            path.pop();
        }
    }

    let mut res = Vec::new();
    let mut path = Vec::new();
    dfs(&nums, 0, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function subsets(nums) {
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < nums.length; i += 1) {
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}
```
