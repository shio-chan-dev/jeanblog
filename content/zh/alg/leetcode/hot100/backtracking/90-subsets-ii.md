---
title: "Hot100：子集 II（Subsets II）排序 + 层内去重回溯 ACERS 解析"
date: 2026-04-19T00:09:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "子集", "去重", "排序", "LeetCode 90"]
description: "围绕 LeetCode 90 子集 II，讲清排序、层内去重与组合型回溯的真正边界，帮助你从 78 子集平滑升级到含重复元素的版本。"
keywords: ["Subsets II", "子集 II", "回溯", "层内去重", "排序", "LeetCode 90", "Hot100"]
---

> **副标题 / 摘要**
> `90. 子集 II` 是 `78. 子集` 的升级版。真正新增的难点不是“还能不能继续回溯”，而是“遇到重复元素时，怎样只跳过重复分支，而不误伤合法答案”。

- **预计阅读时长**：12~15 分钟
- **标签**：`Hot100`、`回溯`、`子集`、`去重`、`排序`
- **SEO 关键词**：Subsets II, 子集 II, 回溯, 层内去重, 排序
- **元描述**：通过 LeetCode 90 理解排序后层内去重的稳定写法，建立含重复元素时的组合型回溯模板。

---

## A — Algorithm（题目与算法）

### 题目还原

给定一个可能包含重复元素的整数数组 `nums`，请返回它的所有可能子集（幂集）。

结果集 **不能包含重复子集**，返回顺序不限。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| nums | `int[]` | 可能包含重复元素的整数数组 |
| 返回 | `int[][]` | 所有不重复的子集 |

### 示例 1

```text
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

### 示例 2

```text
输入：nums = [0]
输出：[[],[0]]
```

### 约束

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`

---

## 目标读者

- 已经做过 `78. 子集`，但还没掌握“重复元素判重边界”的学习者
- 会写基础回溯，但一碰到去重就想上 `set` 暴力兜底的开发者
- 希望把“排序 + 同层跳过重复值”固定成稳定模板的 Hot100 刷题读者

## 背景 / 动机

在 `78. 子集` 里，数组元素互不相同，所以你只需要处理“选 / 不选”。
但到了这题，重复元素一出现，搜索树里就会长出许多**值不同位置相同**的重复分支。

例如 `nums = [1,2,2]`：

- 第一个 `2` 可以产生 `[2]`
- 第二个 `2` 也会再产生一个 `[2]`

如果你不控制好分支边界，答案里就会出现重复子集。

这道题的核心价值是让你真正建立下面这条链路：

- 组合问题仍然用 `startIndex`
- 先排序，把相同值放到一起
- 去重不是“全局不准再用”，而是“同一层里，相同值只展开第一条分支”

## 核心概念

- **`path`**：当前递归路径上已经选中的元素
- **`startIndex`**：当前层从哪个下标开始继续枚举
- **排序**：把相同元素放在一起，去重条件才能稳定成立
- **层内去重**：同一层遇到相同值，后面的重复值直接跳过
- **前序收集答案**：子集问题里，每个节点都是一个合法答案

---

## C — Concepts（核心思想）

### 这道题是怎么一步一步推出来的

#### Step 1：先用最小但不平凡的例子看重复是怎么产生的

看 `nums = [1,2,2]`。

如果你还按 `78. 子集` 的思路无脑展开，会得到两条会撞车的分支：

- 第一层选第一个 `2`，得到 `[2]`
- 第一层跳过第一个 `2`，再选第二个 `2`，也得到 `[2]`

问题已经暴露出来了：

- 我们不是不会生成子集
- 我们是不知道**哪些分支其实表示同一个值选择**

#### Step 2：当前部分答案最少要记住什么？

和 `78. 子集` 一样，仍然需要一个状态保存“当前这条分支已经选了什么”。
这就是 `path`。

```python
path = []
```

`path` 只表示当前分支，不是最终答案全集。

#### Step 3：为什么这题必须先排序？

如果相同的数散落在数组不同位置，你就很难写出稳定的去重条件。
先排序后，相同值会挨在一起，我们才能判断“当前值是不是本层前一个值的重复展开”。

```python
nums.sort()
```

例如 `[2,1,2]` 排序后变成 `[1,2,2]`，重复值被放到相邻位置，判重条件马上清晰很多。

#### Step 4：这题为什么仍然是 `startIndex` 模板？

虽然有重复元素，但它本质上还是组合问题，不关心顺序。
所以我们仍然要限制“下一层只能从当前位置往后选”，避免 `[1,2]` 和 `[2,1]` 这种顺序重复。

```python
def dfs(start: int) -> None:
    ...
```

这里的 `start` 表示：

- 当前层允许从哪个下标开始继续选
- 更前面的元素不再回头考虑

#### Step 5：什么时候该收集答案？

这点和 `78. 子集` 完全一样。

只要当前 `path` 合法，它就是一个子集，所以一进入递归就应该先收集。

```python
res.append(path.copy())
```

不是只在叶子收集，也不是长度固定后才收集。

#### Step 6：当前层有哪些候选动作？

当前层要从 `start` 开始，依次尝试每个可选元素。

```python
for i in range(start, len(nums)):
    ...
```

到这里为止，代码和 `78. 子集` 还几乎一样。
真正的升级发生在下一步。

#### Step 7：怎样只跳过重复分支，而不误删合法答案？

关键判断是：

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

这句的含义不是“相同值永远不能再用”，而是：

- **如果当前值和前一个值相同**
- **并且它们处在同一层枚举里**
- 那么这个值作为“本层新开的分支”已经展开过了

所以可以直接跳过。

注意这里必须是 `i > start`，不是 `i > 0`。
因为我们只想做**层内去重**，不是全局禁用重复值。

#### Step 8：选中一个元素后，状态如何推进？

当前值通过判重以后，就进入标准回溯三件套：

```python
path.append(nums[i])
dfs(i + 1)
path.pop()
```

这里仍然是 `i + 1`，因为子集问题里每个位置最多使用一次。

#### Step 9：慢速走一条分支，看“同层跳过”到底在跳什么

还是看排序后的 `[1,2,2]`。

开始时：

- `path = []`
- `start = 0`

进入 `dfs(0)`：

- 先收集 `[]`

第一层选 `1`：

- `path = [1]`
- 进入 `dfs(1)`，收集 `[1]`

在这一层选第一个 `2`：

- `path = [1,2]`
- 进入 `dfs(2)`，收集 `[1,2]`

再选第二个 `2`：

- `path = [1,2,2]`
- 进入 `dfs(3)`，收集 `[1,2,2]`

回溯到第一层后，再看第一层的第二个 `2`：

- 这时 `i = 2`
- `start = 0`
- `nums[2] == nums[1]`

说明“以 `2` 作为第一层新分支”这件事已经做过一次了，因此跳过。

#### Step 10：把规则整理成一句能复用的话

这题可以直接记成：

> 组合型回溯先排序；同一层里，如果当前值等于前一个值，就跳过当前分支。

这也是很多“含重复元素的组合 / 子集 / 排列”题最常见的判重起点。

### Assemble the Full Code

先把上面推出来的碎片拼成第一版完整代码。
这版代码可以直接运行验证。

```python
from typing import List


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(subsets_with_dup([1, 2, 2]))
    print(subsets_with_dup([0]))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import List


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它的正式名字可以叫：

- 回溯
- 组合型搜索
- 排序 + 层内去重

但真正要固定下来的不是名词，而是这四个不变式：

- `path` 表示当前已经选了什么
- `startIndex` 保证这是组合而不是排列
- 子集题依然在每个节点收集答案
- 判重条件只针对同一层的重复展开

---

## E — Engineering（工程应用）

### 场景 1：SKU 筛选条件去重组合（Python）

**背景**：电商后台从多个来源汇总筛选条件时，可能得到重复标签，例如两个上游都传来了 `red`。  
**为什么适用**：你仍然需要生成所有候选筛选组合，但不能让重复标签造成重复预设。

```python
from typing import List


def unique_filter_sets(tags: List[str]) -> List[List[str]]:
    tags.sort()
    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(tags)):
            if i > start and tags[i] == tags[i - 1]:
                continue
            path.append(tags[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


print(unique_filter_sets(["red", "red", "xl"]))
```

### 场景 2：权限标签候选包去重生成（Go）

**背景**：权限系统把多个服务返回的角色标签合并后，可能出现重复角色名。  
**为什么适用**：要枚举候选权限包做离线验证时，本质上就是“含重复元素的子集去重”。

```go
package main

import (
	"fmt"
	"sort"
)

func bundles(tags []string) [][]string {
	sort.Strings(tags)
	res := make([][]string, 0)
	path := make([]string, 0, len(tags))

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]string(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(tags); i++ {
			if i > start && tags[i] == tags[i-1] {
				continue
			}
			path = append(path, tags[i])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}

func main() {
	fmt.Println(bundles([]string{"read", "read", "write"}))
}
```

### 场景 3：前端多选预设面板去重（JavaScript）

**背景**：配置中心返回的可选项里有重复值，前端要预生成所有多选预设用于回归测试。  
**为什么适用**：如果不做层内去重，UI 会拿到大量语义相同的预设。

```javascript
function uniquePresets(items) {
  const sorted = [...items].sort();
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < sorted.length; i += 1) {
      if (i > start && sorted[i] === sorted[i - 1]) continue;
      path.push(sorted[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}

console.log(uniquePresets(["tag", "tag", "price"]));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 时间复杂度：最坏情况下仍是 `O(n * 2^n)`
  - 去重不会改变指数级枚举上界
  - 复制路径的成本与答案规模同阶
- 空间复杂度：递归栈 `O(n)`
  - 若计入输出，整体空间同样受答案规模影响

### 和几种常见写法对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 排序 + 层内去重 | 回溯时直接跳过重复分支 | 模板稳定、最适合继续做 `40/47/90` | 需要真正理解“同层”含义 |
| 暴力生成后放 `set` 去重 | 先全枚举，再把结果转成集合 | 好想到 | 浪费搜索、代码重、迁移性差 |
| 位运算 + 集合去重 | 每个位置选或不选后再 dedupe | 写法短 | 对重复控制不直观，不利于后续复杂题 |

### 这题最容易错的地方

- 把判重写成 `i > 0`，结果误伤跨层合法选择
- 忘了先排序，导致相同值不相邻，判重条件失效
- 以为“重复值不能再选”，把 `[2,2]` 这种合法子集也删掉

## 常见问题与注意事项

### 为什么一定是 `i > start`，不能写成 `i > 0`？

因为我们要跳过的是“同一层里重复开出来的新分支”。

如果写成 `i > 0`，那么在更深层递归里，合法的第二个 `2` 也会被错误跳过，像 `[2,2]` 这样的答案就没了。

### 这题和 `78. 子集` 的差别到底只有哪一行？

从模板角度看，几乎就是这一行：

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

但这一行背后其实新增了两个前提：

- 必须先排序
- 必须理解“层内去重”而不是“全局禁用重复值”

### 什么时候该想到这一套判重规则？

当你看到：

- 数组里可能有重复值
- 问题要求返回不重复组合 / 子集 / 排列

就应该优先检查：

- 要不要先排序
- 去重是发生在“同层”还是“同枝”

## 最佳实践与建议

- 先把 `78. 子集` 的 `dfs(start)` 模板写熟，再接这题
- 去重题先想“重复答案是在哪一层长出来的”，不要先想 `set`
- 只要涉及“相同元素相邻判重”，排序几乎都是第一步
- 用小例子手动画出两条撞车分支，理解会比死背公式快得多

---

## S — Summary（总结）

- `90. 子集 II` 的本质仍然是组合型回溯，不是新题型
- 真正新增的关键是：**排序后做层内去重**
- 判重条件写成 `i > start and nums[i] == nums[i - 1]`，是因为我们只跳过同层重复分支
- 子集题的收集时机没有变，仍然是“每个节点都是答案”
- 学会这题后，`40. 组合总和 II`、`47. 全排列 II` 会顺很多

### 推荐延伸阅读

- `78. 子集`：不含重复元素的基础版本
- `39. 组合总和`：理解组合型回溯与目标约束
- `40. 组合总和 II`：重复元素 + 目标和 + 去重
- `47. 全排列 II`：排列型问题里的重复元素判重

### 行动建议

现在最值得做的事，不是继续背更多题，而是把 `78` 和 `90` 并排写一遍。
如果你能明确说出“只多了哪一个条件、为什么必须排序、为什么是层内去重”，回溯这一章就真正站稳了。

---

## 多语言实现

### Python

```python
from typing import List


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
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
    return *(const int*)a - *(const int*)b;
}

static void push_result(Result* res, int* path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    int* row = malloc(sizeof(int) * path_size);
    for (int i = 0; i < path_size; ++i) {
        row[i] = path[i];
    }
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(int* nums, int n, int start, int* path, int path_size, Result* res) {
    push_result(res, path, path_size);
    for (int i = start; i < n; ++i) {
        if (i > start && nums[i] == nums[i - 1]) {
            continue;
        }
        path[path_size] = nums[i];
        dfs(nums, n, i + 1, path, path_size + 1, res);
    }
}

int** subsetsWithDup(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    qsort(nums, numsSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * numsSize);
    dfs(nums, numsSize, 0, path, 0, &res);
    free(path);

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
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        vector<int> path;
        dfs(nums, 0, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& nums, int start, vector<int>& path, vector<vector<int>>& res) {
        res.push_back(path);
        for (int i = start; i < (int)nums.size(); ++i) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
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

import "sort"

func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	res := make([][]int, 0)
	path := make([]int, 0, len(nums))

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]int(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(nums); i++ {
			if i > start && nums[i] == nums[i-1] {
				continue
			}
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
impl Solution {
    pub fn subsets_with_dup(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort();
        let mut res: Vec<Vec<i32>> = Vec::new();
        let mut path: Vec<i32> = Vec::new();

        fn dfs(nums: &Vec<i32>, start: usize, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
            res.push(path.clone());
            for i in start..nums.len() {
                if i > start && nums[i] == nums[i - 1] {
                    continue;
                }
                path.push(nums[i]);
                dfs(nums, i + 1, path, res);
                path.pop();
            }
        }

        dfs(&nums, 0, &mut path, &mut res);
        res
    }
}
```

### JavaScript

```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsetsWithDup = function (nums) {
  nums.sort((a, b) => a - b);
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < nums.length; i += 1) {
      if (i > start && nums[i] === nums[i - 1]) continue;
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
};
```
