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

### 这道题是怎么一步一步推出来的

#### Step 1：先从最小例子看“填位置”这件事

看 `nums = [1,2,3]`。

全排列最自然的想法不是“记住一个模板”，而是把问题看成：

- 第 1 个位置放什么
- 第 2 个位置放什么
- 第 3 个位置放什么

如果第 1 个位置先放 `1`，那后面就只能在 `2` 和 `3` 里选。
如果接着放 `2`，最后只剩下 `3`，于是得到 `[1,2,3]`。

这和子集题很不一样：

- 子集题关心“选了哪些元素”
- 排列题还关心“这些元素按什么顺序出现”

#### Step 2：当前部分答案最少要记住什么？

既然我们是在逐个位置构造排列，就得有一个状态表示“当前已经填好的前缀”。
这就是 `path`。

```python
path = []
```

`path` 的含义是：

- 当前递归分支已经选好的元素顺序
- 还不是最终答案全集

#### Step 3：为什么还需要额外状态？

光有 `path` 还不够，因为你还得知道哪些数已经被放进去了。
否则当前层根本不知道哪些数字还能继续选。

这就是 `used[i]` 的来源。

```python
used = [False] * len(nums)
```

它表示：

- `used[i] = True`：`nums[i]` 已经在当前排列里
- `used[i] = False`：当前层仍然可以选它

#### Step 4：递归真正要解决的子问题是什么？

当 `path` 已经固定了一部分之后，剩下的问题就是：

> 再选一个还没使用过的数，放到下一个位置上。

所以这里不需要 `startIndex`，因为每一层都必须重新扫描整个数组，只跳过已经用过的元素。

```python
def dfs() -> None:
    ...
```

#### Step 5：什么时候说明一条路径已经完整？

当 `path` 的长度已经和 `nums` 一样长时，说明每个位置都填满了。
这时当前路径才是一个完整排列。

```python
if len(path) == len(nums):
    res.append(path.copy())
    return
```

这里也必须 `copy()`，因为 `path` 后面还会继续被修改。

#### Step 6：当前层有哪些可选动作？

当前层要做的事很简单：扫描整个数组，把所有还没使用的元素都试一遍。

```python
for i, x in enumerate(nums):
    if used[i]:
        continue
```

和子集题不同，这里不能只看右边。
因为排列允许 `[2,1,3]`、`[3,1,2]` 这种顺序。

#### Step 7：选中一个元素后，状态怎么变化？

如果当前选择 `x`，就要同步更新两个状态：

```python
used[i] = True
path.append(x)
```

然后递归求解“剩余位置怎么填”。

```python
dfs()
```

#### Step 8：递归回来之后要撤销什么？

回溯的关键不是“会递归”，而是递归回来以后要把现场完整恢复。

```python
path.pop()
used[i] = False
```

这里必须同时撤销：

- `path` 里的末尾元素
- `used[i]` 的占用状态

少撤一个，后面的分支都会被污染。

#### Step 9：慢速走一条分支

还是看 `nums = [1,2,3]`。

开始时：

- `path = []`
- `used = [False, False, False]`

选 `1`：

- `path = [1]`
- `used = [True, False, False]`

在下一层选 `2`：

- `path = [1,2]`
- `used = [True, True, False]`

再选 `3`：

- `path = [1,2,3]`
- `used = [True, True, True]`

这时路径长度已经等于 `3`，所以收集 `[1,2,3]`。
然后开始回溯：

- `pop()` 掉 `3`
- 把 `used[2]` 改回 `False`

返回到 `path = [1,2]` 这一层，看看还有没有别的选择。
没有，就继续撤销 `2`。
这样才能再去尝试 `[1,3,2]`、`[2,1,3]` 等其他分支。

### Assemble the Full Code

下面把刚才推出来的碎片拼成第一版完整代码。
这版代码已经可以直接运行。

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


if __name__ == "__main__":
    print(permute([1, 2, 3]))
```

### Reference Answer

如果你要直接提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
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

### 我们刚刚搭出来的到底是什么方法？

它的正式名字是：

- 回溯
- 排列型搜索
- `used[]` 状态控制

但这题真正要固定下来的不是名词，而是两件事：

- 每一层都从“所有未使用元素”里挑一个
- 递归返回时同时恢复 `path` 和 `used`

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
