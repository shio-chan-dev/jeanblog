---
title: "Hot100：N 皇后（N-Queens）列 / 对角线约束回溯 ACERS 解析"
date: 2026-04-19T00:09:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "约束搜索", "棋盘", "N皇后", "LeetCode 51"]
description: "围绕 LeetCode 51 N 皇后，讲清“按行放置 + 列/主对角线/副对角线占用集”的稳定写法，帮助你建立约束搜索题的核心模型。"
keywords: ["N-Queens", "N 皇后", "回溯", "约束搜索", "对角线", "LeetCode 51", "Hot100"]
---

> **副标题 / 摘要**
> `51. N 皇后` 是 Hot100 回溯专题里很关键的一道约束搜索题。你需要学会的不是“在棋盘上乱试”，而是“按行放置，并用列 / 主对角线 / 副对角线三组状态在 O(1) 时间判断冲突”。

- **预计阅读时长**：16~20 分钟
- **标签**：`Hot100`、`回溯`、`约束搜索`、`棋盘`、`N皇后`
- **SEO 关键词**：N-Queens, N 皇后, 回溯, 约束搜索, 对角线
- **元描述**：通过 LeetCode 51 固定 N 皇后约束搜索模板，重点理解按行放置、列与双对角线判冲突，以及结果棋盘构造。

---

## A — Algorithm（题目与算法）

### 题目还原

`n` 皇后问题要求在一个 `n x n` 的棋盘上放置 `n` 个皇后，使得任意两个皇后都不能互相攻击。

给定整数 `n`，请返回所有不同的放置方案。
每个方案都用一个字符串数组表示，其中：

- `'Q'` 表示皇后
- `'.'` 表示空位

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| n | `int` | 棋盘大小，也是皇后数量 |
| 返回 | `string[][]` | 所有不同的棋盘方案 |

### 示例 1

```text
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：4 皇后一共有两种不同解法
```

### 示例 2

```text
输入：n = 1
输出：[["Q"]]
```

### 约束

- `1 <= n <= 9`

---

## 目标读者

- 已经掌握基础回溯，但还没做过“多重约束同时判冲突”的学习者
- 碰到棋盘类搜索就想直接二维数组硬搜、导致代码又慢又乱的开发者
- 想把“按层放置 + 占用状态 O(1) 检查”迁移到其他约束问题的 Hot100 刷题读者

## 背景 / 动机

`51. N 皇后` 是一道非常典型的“状态约束搜索”题。

和 `78 / 90 / 131` 这类题不同，它不是只要决定“切不切 / 选不选”就够了，而是每放一个皇后，都会同时引入多组约束：

- 这一列不能再放
- 这一条主对角线不能再放
- 这一条副对角线不能再放

所以这题的真正价值不只是棋盘本身，而是让你掌握一种很常见的搜索套路：

> 按固定顺序放置对象，并把冲突检查变成 O(1) 的状态查询。

这类模型在排班、布局、资源放置、组合优化原型里都很常见。

## 核心概念

- **按行递归**：每一层只决定当前这一行把皇后放在哪一列
- **`queens[row] = col`**：记录当前部分答案
- **合法性检查**：先写“扫描前面所有行”的朴素版，再优化成 O(1) 状态查询
- **叶子收集答案**：只有放满 `n` 行，当前布局才是完整解

---

## C — Concepts（核心思想）

### 这道题要怎样从“能写”一步一步长成“写对”？

#### Step 1：先写出最粗糙的按行 DFS 骨架

先别急着想所有约束，只先回答一个问题：

> 递归的每一层到底代表什么？

因为每一行最终都要放一个皇后，所以最自然的定义就是：`dfs(row)` 表示“现在要处理第 `row` 行”。

```python
def dfs(row: int) -> None:
    if row == n:
        return

    for col in range(n):
        dfs(row + 1)
```

这版代码当然还不能用，但它先把子问题钉死了：

- 一层递归只负责一行
- 当前层的选择是“这一行放哪一列”

#### Step 2：再把“当前做了什么选择”记下来

光有骨架还不够，我们还得让代码记住：前面几行分别把皇后放到了哪一列。

```python
queens = [-1] * n

def dfs(row: int) -> None:
    if row == n:
        return

    for col in range(n):
        queens[row] = col
        dfs(row + 1)
        queens[row] = -1
```

现在这版虽然还不正确，但读者已经能看懂回溯的主线了：

- `queens[row] = col` 表示“第 `row` 行选第 `col` 列”
- 递归下去是“继续决定下一行”
- 回来后恢复成 `-1`，因为还要试别的列

#### Step 3：什么时候这条分支才算一个完整答案？

当 `row == n` 时，说明 `0 .. n-1` 这些行都已经完成选择了。
这时不能只是 `return`，而要把 `queens` 转成棋盘字符串收集起来。

```python
def build_board() -> List[str]:
    board = []
    for col in queens:
        board.append("." * col + "Q" + "." * (n - col - 1))
    return board

def dfs(row: int) -> None:
    if row == n:
        res.append(build_board())
        return
```

到这里我们已经知道：

- 递归层表示哪一行
- `queens` 表示当前部分答案
- 叶子节点要做什么

#### Step 4：现在代码哪里还不对？缺合法性判断

如果直接这么搜，它会把非法棋盘也收进去。
比如两行都放在同一列，或者放在同一条斜线上，它也照样继续递归。

所以现在缺的不是“更多框架”，而是一个最朴素的合法性判断：

```python
def is_valid(row: int, col: int) -> bool:
    for prev_row in range(row):
        prev_col = queens[prev_row]
        if prev_col == col:
            return False
        if abs(prev_row - row) == abs(prev_col - col):
            return False
    return True
```

再把它塞回 DFS：

```python
for col in range(n):
    if not is_valid(row, col):
        continue
    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

这一版已经是完整、正确、可运行的回溯代码了，只是还不够快。

#### Step 5：先把“对角线冲突”这件事彻底说清楚

很多人卡的不是回溯，而是这里突然冒出来的 `diag1 / diag2`。
所以先不要谈数组，先只看棋盘。

对一个格子 `(row, col)` 来说，皇后除了攻击整列，还会攻击两组斜线：

- 左上到右下这一组
- 右上到左下这一组

如果两个格子在同一条斜线上，那么它们“行差的绝对值”和“列差的绝对值”一定相等。

所以朴素版里这句：

```python
if abs(prev_row - row) == abs(prev_col - col):
    return False
```

本质上就是在判断：

> 当前格子和前面某个皇后，是不是落在同一条斜线上？

先把这个几何关系看懂，后面的状态优化才不会显得像凭空出现。

#### Step 6：这版已经能跑，但慢在哪里？

慢在 `is_valid()`。

每试一个 `(row, col)`，都要把前面 `0 .. row-1` 的皇后重新扫一遍。
题目数据不大时还能过，但这说明我们每次都在重复做同一类检查。

最先可以缓存的，是“某一列是否已经被占用”：

```python
cols = [False] * n
```

这样同列冲突就不用扫了，只看一次：

```python
if cols[col]:
    continue
```

这就是优化方向的起点：

- 朴素版靠“扫描旧状态”
- 优化版靠“直接查询占用状态”

#### Step 7：再引出两组 diagonal arrays，它们到底在存什么？

现在才适合定义 `diag1` 和 `diag2`，因为你已经知道它们只是 `is_valid()` 的替代品。

棋盘上的每个格子 `(row, col)`，同时属于两类斜线中的各一条：

- **主对角线**：左上到右下，同一条线上的格子满足 `row - col` 相同
- **副对角线**：右上到左下，同一条线上的格子满足 `row + col` 相同

所以我们可以分别记录“这条斜线有没有皇后”：

```python
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

这里数组里存的不是“某个格子”，而是：

- `diag1[i]`：编号为 `i` 的主对角线是否已被占用
- `diag2[i]`：编号为 `i` 的副对角线是否已被占用

因为 `row - col` 可能为负数，所以主对角线要平移一下：

```python
d1 = row - col + n - 1
d2 = row + col
```

顺序一定要记成：

- 先观察棋盘上的两组斜线
- 再发现它们分别满足 `row - col` 和 `row + col`
- 最后才把这两个量压成数组下标

#### Step 8：用 O(1) 状态检查替换朴素 `is_valid()`

现在我们就可以把“扫描前面所有行”替换掉了。

先准备三组占用状态：

```python
cols = [False] * n
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

然后在枚举 `col` 时，直接判断：

```python
for col in range(n):
    d1 = row - col + n - 1
    d2 = row + col
    if cols[col] or diag1[d1] or diag2[d2]:
        continue

    queens[row] = col
    cols[col] = True
    diag1[d1] = True
    diag2[d2] = True

    dfs(row + 1)

    queens[row] = -1
    cols[col] = False
    diag1[d1] = False
    diag2[d2] = False
```

这一步就是最终优化版真正长出来的地方：

- `queens` 仍然是主状态
- `cols / diag1 / diag2` 是为了把合法性判断降到 O(1) 的辅助状态
- choose / recurse / undo 的骨架没有变，只是检查方式变快了

#### Step 9：慢速走一条分支，看状态是怎么一起变化的

仍然看 `n = 4`，假设第 `0` 行先放在第 `1` 列。

此时会发生四件事：

- `queens[0] = 1`
- `cols[1] = True`
- `diag1[0 - 1 + 3] = diag1[2] = True`
- `diag2[0 + 1] = diag2[1] = True`

接着看第 `1` 行：

- `col = 1` 会因为 `cols[1]` 被挡掉
- 某些列会因为 `diag1` 被挡掉
- 某些列会因为 `diag2` 被挡掉

只有三组状态都没被占用的列，才能继续往下递归。

所以整道题的最终节奏其实非常统一：

- 先决定当前行放哪一列
- 再看三组状态是否允许
- 选择后同步更新主状态和辅助状态
- 递归回来后，把这几组状态一起撤销

### Assemble the Full Code

下面把这些碎片拼成第一版完整代码。
这版代码可以直接运行，输出所有棋盘方案。

```python
from typing import List


def solve_n_queens(n: int) -> List[List[str]]:
    res: List[List[str]] = []
    queens = [-1] * n
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def build_board() -> List[str]:
        board: List[str] = []
        for col in queens:
            board.append("." * col + "Q" + "." * (n - col - 1))
        return board

    def dfs(row: int) -> None:
        if row == n:
            res.append(build_board())
            return

        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col
            if cols[col] or diag1[d1] or diag2[d2]:
                continue

            queens[row] = col
            cols[col] = diag1[d1] = diag2[d2] = True
            dfs(row + 1)
            cols[col] = diag1[d1] = diag2[d2] = False
            queens[row] = -1

    dfs(0)
    return res


if __name__ == "__main__":
    print(solve_n_queens(4))
    print(solve_n_queens(1))
```

### Reference Answer

如果你要提交到 LeetCode，可以整理成下面这种形式：

```python
from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res: List[List[str]] = []
        queens = [-1] * n
        cols = [False] * n
        diag1 = [False] * (2 * n - 1)
        diag2 = [False] * (2 * n - 1)

        def build_board() -> List[str]:
            board: List[str] = []
            for col in queens:
                board.append("." * col + "Q" + "." * (n - col - 1))
            return board

        def dfs(row: int) -> None:
            if row == n:
                res.append(build_board())
                return

            for col in range(n):
                d1 = row - col + n - 1
                d2 = row + col
                if cols[col] or diag1[d1] or diag2[d2]:
                    continue

                queens[row] = col
                cols[col] = diag1[d1] = diag2[d2] = True
                dfs(row + 1)
                cols[col] = diag1[d1] = diag2[d2] = False
                queens[row] = -1

        dfs(0)
        return res
```

### 我们刚刚搭出来的到底是什么方法？

它可以正式叫：

- 回溯
- 约束搜索
- 按层放置 + 占用状态剪枝

但真正值得记住的是这套模型：

- 每层递归只处理一行
- `queens` 记录主状态
- `cols / diag1 / diag2` 负责 O(1) 冲突判断
- 只有放满所有行，当前布局才是答案

---

## E — Engineering（工程应用）

### 场景 1：谜题 / 棋盘生成器原型（Python）

**背景**：做棋盘类产品、约束谜题或教学可视化时，需要快速生成所有合法布局。  
**为什么适用**：N 皇后本身就是一个经典的约束布局生成问题。

```python
from typing import List


def layouts(n: int) -> List[List[int]]:
    res: List[List[int]] = []
    pos = [-1] * n
    cols = [False] * n
    d1 = [False] * (2 * n - 1)
    d2 = [False] * (2 * n - 1)

    def dfs(row: int) -> None:
        if row == n:
            res.append(pos.copy())
            return
        for col in range(n):
            a = row - col + n - 1
            b = row + col
            if cols[col] or d1[a] or d2[b]:
                continue
            pos[row] = col
            cols[col] = d1[a] = d2[b] = True
            dfs(row + 1)
            cols[col] = d1[a] = d2[b] = False
            pos[row] = -1

    dfs(0)
    return res


print(layouts(4))
```

### 场景 2：二维设备布点原型（Go）

**背景**：在一个抽象网格上按行布置设备，要求同列和两类斜向干扰范围都不能冲突。  
**为什么适用**：这和 N 皇后的约束结构同构，核心都是“多组占用状态 O(1) 判冲突”。

```go
package main

import "fmt"

func placeDevices(n int) [][]int {
	res := make([][]int, 0)
	pos := make([]int, n)
	for i := range pos {
		pos[i] = -1
	}
	cols := make([]bool, n)
	d1 := make([]bool, 2*n-1)
	d2 := make([]bool, 2*n-1)

	var dfs func(int)
	dfs = func(row int) {
		if row == n {
			snapshot := append([]int(nil), pos...)
			res = append(res, snapshot)
			return
		}
		for col := 0; col < n; col++ {
			a := row - col + n - 1
			b := row + col
			if cols[col] || d1[a] || d2[b] {
				continue
			}
			pos[row] = col
			cols[col], d1[a], d2[b] = true, true, true
			dfs(row + 1)
			cols[col], d1[a], d2[b] = false, false, false
			pos[row] = -1
		}
	}

	dfs(0)
	return res
}

func main() {
	fmt.Println(placeDevices(4))
}
```

### 场景 3：前端约束布局预览（JavaScript）

**背景**：前端原型工具要在网格上展示若干组件的所有无冲突布局。  
**为什么适用**：只要约束可抽象成列 / 两组斜线冲突，N 皇后的状态模板就能直接复用。

```javascript
function placements(n) {
  const res = [];
  const pos = new Array(n).fill(-1);
  const cols = new Array(n).fill(false);
  const d1 = new Array(2 * n - 1).fill(false);
  const d2 = new Array(2 * n - 1).fill(false);

  function dfs(row) {
    if (row === n) {
      res.push([...pos]);
      return;
    }
    for (let col = 0; col < n; col += 1) {
      const a = row - col + n - 1;
      const b = row + col;
      if (cols[col] || d1[a] || d2[b]) continue;
      pos[row] = col;
      cols[col] = d1[a] = d2[b] = true;
      dfs(row + 1);
      cols[col] = d1[a] = d2[b] = false;
      pos[row] = -1;
    }
  }

  dfs(0);
  return res;
}

console.log(placements(4));
```

---

## R — Reflection（反思与深入）

### 复杂度分析

- 搜索时间复杂度：通常写作 `O(n!)`
  - 每一行都在尝试若干列
  - 剪枝会大幅减少实际搜索量，但上界仍可按排列级别理解
- 辅助空间复杂度：`O(n)`
  - `queens`、递归栈、列和对角线状态都与 `n` 同阶
- 若计入答案构造：
  - 每个解生成棋盘需要 `O(n^2)`
  - 总输出成本还要乘上解的数量

### 和几种常见写法对比

| 方法 | 思路 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 列 + 对角线布尔数组 | O(1) 判冲突 | 直观、稳定、最适合教学与迁移 | 代码稍长 |
| 每次扫描棋盘检查冲突 | 放一个点就回头扫 | 好想到 | 判冲突太慢，代码也更乱 |
| 位运算 / 位掩码优化 | 用位压缩列与对角线 | 很快，适合竞赛优化 | 学习门槛更高，不适合第一次建立模型 |

### 这题最容易错的地方

- 把搜索写成“枚举任意空格”，导致状态空间失控
- 主对角线和副对角线下标映射写错
- 更新了 `cols` 却忘了同步更新 / 撤销对角线状态
- 收集答案时直接存 `queens`，但没有转换成棋盘字符串

## 常见问题与注意事项

### 为什么按行放置就足够了？

因为每行最终必须恰好放一个皇后。
一旦你把“层”固定成“行”，行约束就被结构本身消掉了，剩下只需要检查列和对角线。

### 为什么主对角线是 `row - col + n - 1`？

因为主对角线上所有格子的 `row - col` 都相等，但这个值可能是负数。
加上 `n - 1` 之后，就能稳定映射到 `0 .. 2n-2` 的数组下标范围。

### 这题能不能再优化？

能。

如果你追求更高性能，可以用位运算把：

- 列占用
- 主对角线占用
- 副对角线占用

压成整数掩码。
但第一次做这题，更建议先把布尔数组版彻底写稳。

## 最佳实践与建议

- 遇到棋盘约束题，优先问自己能不能“按行 / 按列”固定一维递归
- 把冲突检测尽量变成数组或位掩码的 O(1) 查询
- 主状态和辅助状态要成对更新、成对撤销
- 先把 `n = 4` 的两组答案跑通，再去做任何位运算优化

---

## S — Summary（总结）

- `51. N 皇后` 的核心不是“棋盘”，而是“约束搜索”
- 只要按行递归，每层问题就会缩成“这一行选哪一列”
- `cols / diag1 / diag2` 让冲突判断从扫棋盘降到 O(1)
- 收集答案时不要直接交 `queens`，要把它转换成字符串棋盘
- 这题是很多布局、排班、组合优化原型题的基础模板

### 推荐延伸阅读

- `52. N 皇后 II`：同样的搜索框架，但只统计方案数
- `46. 全排列`：理解“主状态 + 辅助状态”如何成对恢复
- `37. 解数独`：更复杂的多约束棋盘搜索
- 位运算版 N 皇后：进一步理解状态压缩优化

### 行动建议

如果你刚做完这题，下一步最有价值的不是直接背位运算版本，而是自己重新写一遍：

- 为什么按行递归
- 为什么是三组占用状态
- 为什么撤销状态时要一组不漏

把这三件事说清楚，你就真正进入约束搜索这一类题了。

---

## 多语言实现

### Python

```python
from typing import List


def solve_n_queens(n: int) -> List[List[str]]:
    res: List[List[str]] = []
    queens = [-1] * n
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def build_board() -> List[str]:
        board: List[str] = []
        for col in queens:
            board.append("." * col + "Q" + "." * (n - col - 1))
        return board

    def dfs(row: int) -> None:
        if row == n:
            res.append(build_board())
            return
        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col
            if cols[col] or diag1[d1] or diag2[d2]:
                continue
            queens[row] = col
            cols[col] = diag1[d1] = diag2[d2] = True
            dfs(row + 1)
            cols[col] = diag1[d1] = diag2[d2] = False
            queens[row] = -1

    dfs(0)
    return res
```

### C

```c
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    char*** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static void push_result(Result* res, int* queens, int n) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(char**) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }

    char** board = malloc(sizeof(char*) * n);
    for (int row = 0; row < n; ++row) {
        board[row] = malloc(n + 1);
        for (int col = 0; col < n; ++col) {
            board[row][col] = '.';
        }
        board[row][n] = '\0';
        board[row][queens[row]] = 'Q';
    }

    res->data[res->size] = board;
    res->col_sizes[res->size] = n;
    res->size += 1;
}

static void dfs(int n, int row, int* queens, bool* cols, bool* diag1, bool* diag2, Result* res) {
    if (row == n) {
        push_result(res, queens, n);
        return;
    }

    for (int col = 0; col < n; ++col) {
        int d1 = row - col + n - 1;
        int d2 = row + col;
        if (cols[col] || diag1[d1] || diag2[d2]) {
            continue;
        }

        queens[row] = col;
        cols[col] = diag1[d1] = diag2[d2] = true;
        dfs(n, row + 1, queens, cols, diag1, diag2, res);
        cols[col] = diag1[d1] = diag2[d2] = false;
        queens[row] = -1;
    }
}

char*** solveNQueens(int n, int* returnSize, int** returnColumnSizes) {
    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(char**) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* queens = malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        queens[i] = -1;
    }
    bool* cols = calloc(n, sizeof(bool));
    bool* diag1 = calloc(2 * n - 1, sizeof(bool));
    bool* diag2 = calloc(2 * n - 1, sizeof(bool));

    dfs(n, 0, queens, cols, diag1, diag2, &res);

    free(queens);
    free(cols);
    free(diag1);
    free(diag2);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
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
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<int> queens(n, -1);
        vector<int> cols(n, 0);
        vector<int> diag1(2 * n - 1, 0);
        vector<int> diag2(2 * n - 1, 0);
        dfs(0, n, queens, cols, diag1, diag2, res);
        return res;
    }

private:
    vector<string> buildBoard(const vector<int>& queens, int n) {
        vector<string> board(n, string(n, '.'));
        for (int row = 0; row < n; ++row) {
            board[row][queens[row]] = 'Q';
        }
        return board;
    }

    void dfs(int row, int n, vector<int>& queens, vector<int>& cols, vector<int>& diag1, vector<int>& diag2, vector<vector<string>>& res) {
        if (row == n) {
            res.push_back(buildBoard(queens, n));
            return;
        }
        for (int col = 0; col < n; ++col) {
            int d1 = row - col + n - 1;
            int d2 = row + col;
            if (cols[col] || diag1[d1] || diag2[d2]) {
                continue;
            }
            queens[row] = col;
            cols[col] = diag1[d1] = diag2[d2] = 1;
            dfs(row + 1, n, queens, cols, diag1, diag2, res);
            cols[col] = diag1[d1] = diag2[d2] = 0;
            queens[row] = -1;
        }
    }
};
```

### Go

```go
package main

func solveNQueens(n int) [][]string {
	res := make([][]string, 0)
	queens := make([]int, n)
	for i := range queens {
		queens[i] = -1
	}
	cols := make([]bool, n)
	diag1 := make([]bool, 2*n-1)
	diag2 := make([]bool, 2*n-1)

	buildBoard := func() []string {
		board := make([]string, n)
		for row, col := range queens {
			bytes := make([]byte, n)
			for i := range bytes {
				bytes[i] = '.'
			}
			bytes[col] = 'Q'
			board[row] = string(bytes)
		}
		return board
	}

	var dfs func(int)
	dfs = func(row int) {
		if row == n {
			res = append(res, buildBoard())
			return
		}
		for col := 0; col < n; col++ {
			d1 := row - col + n - 1
			d2 := row + col
			if cols[col] || diag1[d1] || diag2[d2] {
				continue
			}
			queens[row] = col
			cols[col], diag1[d1], diag2[d2] = true, true, true
			dfs(row + 1)
			cols[col], diag1[d1], diag2[d2] = false, false, false
			queens[row] = -1
		}
	}

	dfs(0)
	return res
}
```

### Rust

```rust
impl Solution {
    pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut res: Vec<Vec<String>> = Vec::new();
        let mut queens = vec![usize::MAX; n];
        let mut cols = vec![false; n];
        let mut diag1 = vec![false; 2 * n - 1];
        let mut diag2 = vec![false; 2 * n - 1];

        fn build_board(queens: &[usize]) -> Vec<String> {
            let n = queens.len();
            let mut board = Vec::with_capacity(n);
            for &col in queens.iter() {
                let mut row = vec![b'.'; n];
                row[col] = b'Q';
                board.push(String::from_utf8(row).unwrap());
            }
            board
        }

        fn dfs(
            row: usize,
            n: usize,
            queens: &mut Vec<usize>,
            cols: &mut Vec<bool>,
            diag1: &mut Vec<bool>,
            diag2: &mut Vec<bool>,
            res: &mut Vec<Vec<String>>,
        ) {
            if row == n {
                res.push(build_board(queens));
                return;
            }
            for col in 0..n {
                let d1 = row + n - 1 - col;
                let d2 = row + col;
                if cols[col] || diag1[d1] || diag2[d2] {
                    continue;
                }
                queens[row] = col;
                cols[col] = true;
                diag1[d1] = true;
                diag2[d2] = true;
                dfs(row + 1, n, queens, cols, diag1, diag2, res);
                cols[col] = false;
                diag1[d1] = false;
                diag2[d2] = false;
            }
        }

        dfs(0, n, &mut queens, &mut cols, &mut diag1, &mut diag2, &mut res);
        res
    }
}
```

### JavaScript

```javascript
/**
 * @param {number} n
 * @return {string[][]}
 */
var solveNQueens = function (n) {
  const res = [];
  const queens = new Array(n).fill(-1);
  const cols = new Array(n).fill(false);
  const diag1 = new Array(2 * n - 1).fill(false);
  const diag2 = new Array(2 * n - 1).fill(false);

  function buildBoard() {
    return queens.map((col) => ".".repeat(col) + "Q" + ".".repeat(n - col - 1));
  }

  function dfs(row) {
    if (row === n) {
      res.push(buildBoard());
      return;
    }
    for (let col = 0; col < n; col += 1) {
      const d1 = row - col + n - 1;
      const d2 = row + col;
      if (cols[col] || diag1[d1] || diag2[d2]) continue;
      queens[row] = col;
      cols[col] = diag1[d1] = diag2[d2] = true;
      dfs(row + 1);
      cols[col] = diag1[d1] = diag2[d2] = false;
      queens[row] = -1;
    }
  }

  dfs(0);
  return res;
};
```
