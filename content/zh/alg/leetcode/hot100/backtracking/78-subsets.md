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

### 这道题是怎么一步一步推出来的

#### Step 1：先从一个最小但不平凡的例子开始

看 `nums = [1,2,3]`。

不要一上来就问“怎样一次性生成所有子集”，先换成更小的问题：

- 当前已经选了一些数
- 下一步还能从哪里继续选
- 什么时候当前路径本身就已经是一个答案

对 `[1,2,3]` 来说，搜索树会是：

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

这个例子最重要的观察是：

- 每个节点本身就是一个合法子集
- 选了 `1` 之后，下一层不能再回头把 `1` 前面的元素重新枚举

#### Step 2：当前部分答案最少要记住什么？

既然我们是在“逐步构造一个子集”，那就必须有一个状态来保存当前已经选了哪些数。
这就是 `path`。

```python
path = []
```

`path` 的含义是：

- 它只表示当前递归分支上的选择
- 它不是最终答案全集

#### Step 3：怎样避免把同一组元素按不同顺序重复生成？

子集是组合问题，不关心顺序。
所以 `[1,2]` 和 `[2,1]` 只能算一个答案。

这就要求我们给每一层一个边界：只能从当前位置往后选。
这就是 `startIndex` 的来源。

```python
def dfs(start: int) -> None:
    ...
```

这里的 `start` 表示：

- 当前层允许从哪个下标开始继续枚举
- 更前面的元素不再回头考虑

#### Step 4：什么时候应该收集答案？

这题和排列、目标和类题不一样。
它没有“必须选满”或“必须凑成 target”这种终点约束。

只要当前 `path` 合法，它就是一个子集。
所以一进入 `dfs`，就应该先收集当前路径。

```python
res.append(path.copy())
```

这里一定要 `copy()`，因为 `path` 后面还会继续变化。

#### Step 5：当前层有哪些可选动作？

当前层只需要从 `start` 开始往后看，把每个还没处理的元素依次拿出来试一下。

```python
for i in range(start, len(nums)):
    ...
```

这个边界保证了：

- 子集按下标递增的顺序构造
- 不会出现 `[2,1]` 这种倒序重复

#### Step 6：选中一个元素后，状态怎么推进？

如果当前选择 `nums[i]`，就把它压进 `path`，然后递归处理后面的元素。

```python
path.append(nums[i])
dfs(i + 1)
```

这里的 `i + 1` 非常关键：

- 当前元素已经决定“选了”
- 下一层不能再从自己或更前面重新开始

#### Step 7：递归回来之后要撤销什么？

回溯的核心就是：选完、递归、撤销。
所以返回时必须把刚加入的元素弹出去。

```python
path.pop()
```

这样循环才能继续尝试同一层的下一个候选值。

#### Step 8：慢速走一条分支

还是看 `nums = [1,2,3]`。

开始时：

- `path = []`
- `start = 0`

一进入 `dfs(0)`：

- 先收集 `[]`

选择 `1`：

- `path = [1]`
- 进入 `dfs(1)`
- 先收集 `[1]`

在这一层再选 `2`：

- `path = [1,2]`
- 进入 `dfs(2)`
- 先收集 `[1,2]`

继续选 `3`：

- `path = [1,2,3]`
- 进入 `dfs(3)`
- 收集 `[1,2,3]`

然后一路 `pop()` 回来，再去尝试 `[1,3]`、`[2]`、`[2,3]`、`[3]`。
整道题其实就是在重复这一个模式。

### Assemble the Full Code

下面把上面的碎片拼成第一版完整代码。
这版代码可以直接运行验证结果。

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


if __name__ == "__main__":
    print(subsets([1, 2, 3]))
    print(subsets([0]))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成更贴近提交环境的版本：

```python
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
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

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 回溯
- 组合型搜索
- `startIndex` 边界控制

但更重要的不是名字，而是这三个不变式：

- `path` 表示当前已经选了什么
- `startIndex` 表示当前层从哪里开始继续选
- 每个节点本身都是答案，所以收集发生在递归最前面

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
