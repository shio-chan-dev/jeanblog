---
title: "LeetCode 40：组合总和 II（回溯 / 同层去重 / 只能用一次）ACERS 解析"
date: 2026-04-17T14:31:11+08:00
draft: false
categories: ["LeetCode"]
tags: ["回溯", "组合总和 II", "去重", "剪枝", "DFS", "LeetCode 40"]
description: "围绕 LeetCode 40 讲清“每个位置只能用一次”与“同层去重”的真正含义，并用一步一步推导的方式建立稳定回溯模板。"
keywords: ["Combination Sum II", "组合总和 II", "回溯", "同层去重", "剪枝", "LeetCode 40"]
---

> **副标题 / 摘要**  
> 如果说 `39. 组合总和` 教你“当前值还能继续用”，那 `40. 组合总和 II` 教你的就是下一层升级：数组里会出现重复值、每个位置只能用一次、去重必须发生在正确的树层上。

- **预计阅读时长**：14~16 分钟  
- **标签**：`回溯`、`去重`、`剪枝`、`组合搜索`  
- **SEO 关键词**：Combination Sum II, 组合总和 II, 回溯, 同层去重, 剪枝, LeetCode 40  
- **元描述**：从题目本身一步一步推出 LeetCode 40 的稳定解法，真正理解排序、`i + 1` 递归和“同层去重”规则。  

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个候选数组 `candidates` 和一个目标值 `target`，  
找出所有和为 `target` 的不同组合，并以列表形式返回。

`candidates` 中的每个数字在每个组合里 **最多只能使用一次**。  
最终答案中不能包含重复组合。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| candidates | int[] | 候选数组，允许出现重复值 |
| target | int | 目标和 |
| 返回 | int[][] | 所有和为 `target` 的不同组合 |

### 示例 1

```text
输入：candidates = [10,1,2,7,6,1,5], target = 8
输出：
[
  [1,1,6],
  [1,2,5],
  [1,7],
  [2,6]
]
```

### 示例 2

```text
输入：candidates = [2,5,2,1,2], target = 5
输出：
[
  [1,2,2],
  [5]
]
```

### 约束

- `1 <= candidates.length <= 100`
- `1 <= candidates[i] <= 50`
- `1 <= target <= 30`

---

## 目标读者

- 已经做完 `39. 组合总和`，想继续搞清“不能重复选”之后模板怎么变的学习者
- 会写回溯框架，但一碰到输入里有重复值就容易写乱去重逻辑的读者
- 想真正掌握“同层去重”而不是硬背一句 `if i > start and ...` 的开发者

## 背景 / 动机

这道题是 `39` 的自然下一题，因为它同时改了两条规则：

- 输入数组本身可能有重复值
- 每个位置最多只能被用一次

这就意味着，单纯“把所有可能都搜出来再去重”的写法会产生大量重复分支。

比如排序后数组开头是 `[1,1,2,...]` 时：

- 如果把第一个 `1` 作为当前层的第一个选择，后面可能得到 `[1,2]`
- 如果把第二个 `1` 作为当前层的第一个选择，也会得到同样的 `[1,2]`

所以这题真正要解决的，不只是“怎么搜”，而是：

- 什么样的重复分支应该跳过？
- 什么样的相同数值在更深一层仍然必须保留？

这两个问题想明白，`40` 才算真正做会。

## 核心概念

- **`path`**：当前正在构造的组合
- **`remain`**：距离目标和还差多少
- **`start` / `startIndex`**：当前层允许从哪个位置开始枚举
- **只能用一次**：选了 `candidates[i]` 之后，下一层必须从 `i + 1` 开始
- **同层去重**：若 `i > start` 且 `candidates[i] == candidates[i - 1]`，则跳过
- **排序剪枝**：若当前值已经大于 `remain`，后续更大的值也不用再看

---

## C — Concepts（核心思想）

### 这道题要怎么从题目一步步推出解法？

#### Step 1：先看一个最能暴露“重复分支”问题的小例子

看 `candidates = [1,1,2]`，`target = 3`。

它真正的合法组合只有：

- `[1,2]`

但如果你不控制重复分支，就很容易把 `[1,2]` 生成两次：

- 一次从第一个 `1` 开始
- 一次从第二个 `1` 开始

这说明两件事：

- 它仍然是组合问题，不是排列问题
- 输入中的重复值会在搜索树同一层制造重复分支

#### Step 2：部分答案至少要记住什么？

既然我们是在逐步构造一个组合，就必须把当前已经选中的数字存下来。

```python
path = []
```

`path` 表示：

- 当前这一条递归路径上已经决定好的数字
- 不是最终答案全集

#### Step 3：下一层要解决的“小问题”是什么？

和 `39` 一样，最稳定的做法仍然是直接维护“还差多少”：

```python
def dfs(start: int, remain: int) -> None:
    ...
```

这里的 `remain` 表示：

- 当前路径离 `target` 还差多少
- 每选一个数，就把这个值从 `remain` 里减掉

#### Step 4：为什么还需要 `startIndex`？

因为它仍然是组合问题。

如果下一层允许回头枚举更早的位置，那么 `[1,2]` 和 `[2,1]` 这种顺序重复就会重新出现。

所以每一层都只能从 `start` 开始往右枚举：

```python
for i in range(start, len(candidates)):
    x = candidates[i]
```

#### Step 5：和 `39` 相比，最关键的变化是什么？

最大的变化是：当前值不能再重复使用。

所以选了 `candidates[i]` 之后，下一层必须从 `i + 1` 开始，而不是 `i`。

```python
path.append(x)
dfs(i + 1, remain - x)
path.pop()
```

这一行就是“每个位置最多用一次”的代码表达。

#### Step 6：为什么这题一定要先排序？

排序有两个作用，而且都非常关键：

1. 把相同值放到一起，便于去重
2. 让 `if x > remain: break` 成为安全剪枝

```python
candidates.sort()
```

如果不排序：

- 重复值不相邻，去重条件不好写
- 当前值过大，也不能推出后面一定更大

#### Step 7：怎样只跳过“真的重复”的那种分支？

这题的核心规则就是：

```python
if i > start and candidates[i] == candidates[i - 1]:
    continue
```

它的意思要拆开看：

- `i > start`：说明这不是当前层第一个被尝试的数
- `candidates[i] == candidates[i - 1]`：说明它和前一个数值相同

所以它真正表达的是：

- 如果两个相同值在竞争“当前层的同一个位置”，那只试第一个
- 但如果某个相同值已经在更深层被选进 `path`，那么后面的相同值在下一层仍然可能合法

这就是为什么这题叫“同层去重”，而不是“全局去重”。

#### Step 8：什么时候收集答案？什么时候可以提前停？

收集答案的条件仍然是：

```python
if remain == 0:
    res.append(path.copy())
    return
```

排序后的安全剪枝是：

```python
if x > remain:
    break
```

这里一定是 `break`，不是 `continue`，因为后面的值只会更大。

#### Step 9：慢速走一条合法分支，再看一条被跳过的重复分支

把官方示例 1 排序后，得到：

```text
[1,1,2,5,6,7,10], target = 8
```

在第 0 层：

- 先尝试索引 `0` 的 `1`
- 当循环走到索引 `1` 的 `1` 时，因为 `i > start` 且当前值等于前一个值，所以这条分支会被跳过

但是在已经选中第一个 `1` 的那条分支里：

- `path = [1]`
- 下一层从索引 `1` 开始
- 此时再选第二个 `1` 是合法的，因为它不再是同层竞争关系

这正是 `[1,1,6]` 还能被保留下来的原因。

### Assemble the Full Code

先把所有已经解释过的碎片拼成一个可运行版本：

```python
from typing import List


def combination_sum_ii(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if i > start and x == candidates[i - 1]:
                continue
            if x > remain:
                break

            path.append(x)
            dfs(i + 1, remain - x)
            path.pop()

    dfs(0, target)
    return res


if __name__ == "__main__":
    print(combination_sum_ii([10, 1, 2, 7, 6, 1, 5], 8))
    print(combination_sum_ii([2, 5, 2, 1, 2], 5))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成下面这种提交版：

```python
from typing import List


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int, remain: int) -> None:
            if remain == 0:
                res.append(path.copy())
                return

            for i in range(start, len(candidates)):
                x = candidates[i]
                if i > start and x == candidates[i - 1]:
                    continue
                if x > remain:
                    break

                path.append(x)
                dfs(i + 1, remain - x)
                path.pop()

        dfs(0, target)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 回溯
- 组合型搜索
- 同层去重
- 每个位置最多选一次
- 排序后剪枝

但顺序仍然很重要：

- 先看题目事实
- 再确定状态和转移
- 最后才给它贴上模板标签

---

## E — Engineering（工程应用）

### 场景 1：实体优惠券组合（Python）

**背景**：一个结算系统拿到一组优惠券卡片，不同卡片可能面值相同。  
**为什么适用**：每张实体卡只能用一次，但相同面值的组合不能重复展示。

```python
def coupon_bundles(coupons, target):
    coupons.sort()
    res = []
    path = []

    def dfs(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(coupons)):
            x = coupons[i]
            if i > start and coupons[i] == coupons[i - 1]:
                continue
            if x > remain:
                break
            path.append(x)
            dfs(i + 1, remain - x)
            path.pop()

    dfs(0, target)
    return res


print(coupon_bundles([10, 1, 2, 7, 6, 1, 5], 8))
```

### 场景 2：一次性库存批次拼装（Go）

**背景**：后端服务需要从若干实体库存批次中拼出目标数量，不同批次可能大小相同。  
**为什么适用**：每个批次只能消耗一次，但同值批次不应该让结果出现重复组合。

```go
package main

import (
	"fmt"
	"sort"
)

func assemble(lots []int, target int) [][]int {
	sort.Ints(lots)
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if remain == 0 {
			res = append(res, append([]int(nil), path...))
			return
		}
		for i := start; i < len(lots); i++ {
			x := lots[i]
			if i > start && lots[i] == lots[i-1] {
				continue
			}
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(i+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}

func main() {
	fmt.Println(assemble([]int{2, 5, 2, 1, 2}, 5))
}
```

### 场景 3：前端重复价位方案规划器（JavaScript）

**背景**：前端配置器里有一批一次性选项，不同选项可能价格一样。  
**为什么适用**：每个选项只能选一次，UI 又不能把同样的价格组合重复列出来。

```javascript
function combinationSum2(candidates, target) {
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
      if (i > start && candidates[i] === candidates[i - 1]) continue;
      if (x > remain) break;
      path.push(x);
      dfs(i + 1, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}

console.log(combinationSum2([2, 5, 2, 1, 2], 5));
```

---

## R — Reflection（反思与深入）

### 正确性直觉

这套写法成立，依赖以下几个不变量：

- `path` 中的元素总是按索引递增选入，所以不会产生顺序重复
- `remain` 始终表示“当前路径还差多少”
- `dfs(i + 1, ...)` 保证每个输入位置最多只被使用一次
- 同层去重规则会砍掉重复兄弟分支，但不会误杀更深层的合法方案
- 排序后 `x > remain` 可以安全结束当前层

### 复杂度分析

令 `n = len(candidates)`。

- 排序复杂度：`O(n log n)`
- 搜索树本质接近子集枚举，因为每个位置只能用一次
- 常见宽松上界：`O(2^n * n)`，其中额外的 `n` 主要来自收集答案时复制路径
- 递归额外空间：`O(n)`，不包含输出答案本身

实际运行时间很依赖：

- 重复值有多少
- 去重规则能砍掉多少分支
- 排序剪枝能提前停掉多少搜索

### 常见问题 / FAQ

#### 为什么下一层传 `i + 1`，不是 `i`？

因为每个输入位置最多只能用一次。  
如果传 `i`，就会把题目写回 `39. Combination Sum` 那种“可重复选”的模型。

#### 为什么去重条件是 `i > start`，不是 `i > 0`？

因为我们只想去掉“同一层里值相同的兄弟分支”。

如果你写成 `i > 0`，就可能把 `[1,1,6]` 这种本来合法的更深层选择错误跳掉。

#### 为什么一定要先排序？

因为排序之后，这两个规则才成立：

- `if i > start and candidates[i] == candidates[i - 1]: continue`
- `if x > remain: break`

不排序的话，重复值不相邻，后面的值也不一定更大。

### 常见错误

- 递归时还写成 `dfs(i, remain - x)`，误把题目做成可重复选
- 把去重写成 `if i > 0 and ...`，误删合法解
- 没排序就直接用去重和 `break` 剪枝
- 收集答案时直接 `append(path)`，导致回溯时历史答案被一起改掉

## Best Practices

- 一定把这题和 `39` 放在一起对照，尤其是 `i` 与 `i + 1` 的差别
- 写去重条件前，先明确它是“同层去重”还是“全局去重”
- 任何依赖顺序的剪枝和去重，都先问自己：数组排好序了吗
- 在回溯题里，尽量让 `path`、`remain`、`start` 的语义长期保持一致

## 参考与延伸阅读

- 官方题目：<https://leetcode.cn/problems/combination-sum-ii/>
- 前一道推荐复盘：`39. 组合总和`
- 下一道推荐：`216. 组合总和 III`
- 相关去重题：`90. 子集 II`

---

## S — Summary（总结）

- `40` 相比 `39` 的真正升级点，不只是“不能重复选”，而是“不能重复选 + 正确的去重层级”
- `dfs(i + 1, remain - x)` 就是在代码层面表达“每个位置最多使用一次”
- `if i > start and candidates[i] == candidates[i - 1]: continue` 只会砍掉重复兄弟分支
- 排序是这题去重和剪枝的共同前提

### 建议下一步练习

- 把 `39` 和 `40` 放在一起重写一遍，彻底吃透 `i` 与 `i + 1`
- 然后接着做 `216`，继续叠加“固定长度 `k`”这个新约束

### CTA

不要只背去重条件。  
拿 `[1,1,2]` 这个最小例子，自己把前两层搜索树画出来，再解释一遍：为什么同层第二个 `1` 要跳过，但更深层第二个 `1` 仍然可能合法。

---

## Multi-Language Implementations

### Python

```python
from typing import List


def combination_sum_ii(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if i > start and x == candidates[i - 1]:
                continue
            if x > remain:
                break
            path.append(x)
            dfs(i + 1, remain - x)
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
        if (i > start && candidates[i] == candidates[i - 1]) {
            continue;
        }
        if (x > remain) {
            break;
        }
        path[path_size] = x;
        dfs(candidates, candidates_size, i + 1, remain - x, path, path_size + 1, res);
    }
}

int** combinationSum2(int* candidates, int candidatesSize, int target,
                      int* returnSize, int** returnColumnSizes) {
    qsort(candidates, candidatesSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.rows = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * candidatesSize);
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
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
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
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (x > remain) {
                break;
            }
            path.push_back(x);
            dfs(candidates, i + 1, remain - x, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

import "sort"

func combinationSum2(candidates []int, target int) [][]int {
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
			if i > start && candidates[i] == candidates[i-1] {
				continue
			}
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(i+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}
```

### Rust

```rust
fn combination_sum_2(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
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
            if i > start && candidates[i] == candidates[i - 1] {
                continue;
            }
            if x > remain {
                break;
            }
            path.push(x);
            dfs(candidates, i + 1, remain - x, path, res);
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
function combinationSum2(candidates, target) {
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
      if (i > start && candidates[i] === candidates[i - 1]) continue;
      if (x > remain) break;
      path.push(x);
      dfs(i + 1, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}
```
