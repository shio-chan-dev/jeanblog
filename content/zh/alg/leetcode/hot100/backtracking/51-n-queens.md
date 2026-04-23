---
title: "Hot100：N 皇后严格增量构建教程"
date: 2026-04-19T00:09:28+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "约束搜索", "棋盘", "N皇后", "LeetCode 51"]
description: "用严格增量构建的方式讲解 LeetCode 51 N 皇后：从最小例子、最小 DFS 骨架，到第一版正确解，再到列与对角线状态优化。"
keywords: ["N-Queens", "N 皇后", "回溯", "约束搜索", "对角线", "LeetCode 51", "Hot100"]
---

`51. N 皇后` 最适合用“代码一步一步长出来”的方式学，而不是直接看一个已经想好的模板答案。
这篇教程只保留教学主线：先从最小例子暴露冲突，再写最小 DFS 骨架，先得到第一版正确解，然后一步步优化到列 / 对角线状态版。

## 题目

`n` 皇后问题要求在一个 `n x n` 的棋盘上放置 `n` 个皇后，使得任意两个皇后都不能互相攻击。

给定整数 `n`，返回所有不同的放置方案。
每个方案都用一个字符串数组表示，其中：

- `'Q'` 表示皇后
- `'.'` 表示空位

### 示例 1

```text
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

### 示例 2

```text
输入：n = 1
输出：[["Q"]]
```

### 约束

- `1 <= n <= 9`

---

## 一步一步把代码逼出来

### Step 1：先看最小能暴露冲突的例子

先只问一个很具体的问题：

> 放下一个皇后以后，到底麻烦在哪里？

看 `n = 4`。
假设第 `0` 行把皇后放在第 `1` 列。

那么后面的行就不再是“随便找一个空格”了，而是立刻多出三类限制：

- 第 `1` 列不能再放
- 一条左上到右下的对角线不能再放
- 一条右上到左下的对角线不能再放

这个最小例子已经说明了问题的本质：

- 我们要按某种固定顺序放置皇后
- 每做一个选择，都会封锁后面的若干位置

这一版还没有代码。

现在这一版能做到：

- 看清楚题目的冲突结构

但它还缺：

- 一个明确的子问题定义
- 一个可以落代码的搜索骨架

### Step 2：先把更小的子问题定义清楚，再写最小 DFS 骨架

接下来只解决一个问题：

> 如果前 `row` 行都已经处理好了，剩下的问题是什么？

剩下的问题就是：

> 给第 `row` 行选一个列，然后把同样的问题交给第 `row + 1` 行

所以一层递归最自然的定义就是：

- `dfs(row)` 表示“现在要给第 `row` 行选位置”

在这个定义下，先加最小骨架：

```python
def dfs(row: int) -> None:
    if row == n:
        return

    for col in range(n):
        dfs(row + 1)
```

这是第一版真正的代码。

现在这一版能做到：

- 把搜索层和“行”绑定起来
- 把每层的动作固定成“当前行选哪一列”

但它还缺：

- 前面选过什么，还没有被记录
- 到达叶子以后，也还不会收集答案
- 也还没有合法性判断

### Step 3：先补一个状态，记录当前已经选了什么

现在只解决下一个问题：

> 当前这一层选了哪一列，代码要存到哪里？

先新增一个状态：

```python
queens = [-1] * n
```

它表示：

- `queens[row] = col`：第 `row` 行选了第 `col` 列
- `-1`：这一行还没选

然后把上一版 DFS 里的循环体替换成：

```python
for col in range(n):
    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

这时回溯的主骨架已经长出来了：

- choose
- recurse
- undo

现在这一版能做到：

- 记录当前路径上的选择
- 在试完一个列之后恢复现场

但它还缺：

- 到叶子时怎样把答案收起来
- 当前选择是不是合法，还完全没检查

### Step 4：补上“什么时候算完整答案”

接下来只解决一个问题：

> 什么时候这条分支已经是一个完整解？

当 `row == n` 时，说明 `0 .. n-1` 这些行都已经完成选择了。
所以这时不能只是 `return`，而要把 `queens` 转成题目要求的棋盘字符串。

先新增一个辅助函数：

```python
def build_board() -> List[str]:
    board = []
    for col in queens:
        board.append("." * col + "Q" + "." * (n - col - 1))
    return board
```

然后把上一版 base case 替换成：

```python
if row == n:
    res.append(build_board())
    return
```

现在这一版能做到：

- 知道什么叫“走到叶子”
- 把当前 `queens` 转成题目需要的答案格式

但它还缺：

- 所有非法布局也会被收进去

### Step 5：补上第一版正确的合法性判断

现在最急需解决的问题是：

> `(row, col)` 这个位置到底能不能放？

最朴素但正确的办法是：

- 只跟前面已经放好的皇后逐个比较

先新增一个合法性函数：

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

然后把上一版循环体替换成：

```python
for col in range(n):
    if not is_valid(row, col):
        continue

    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

这一版非常重要，因为它已经是第一版**完整且正确**的代码了。
虽然还不快，但已经能求出正确答案。

现在这一版能做到：

- 过滤掉同列冲突
- 过滤掉对角线冲突
- 正确生成所有解

但它还缺：

- 每次判断都要扫前面所有行，效率不高

### Step 6：先别一下子全优化，先只把列优化掉

下一步只解决一个更小的问题：

> 朴素版里，最容易先缓存掉的是哪一部分重复工作？

答案是列冲突。
我们先不碰对角线，只先把列这部分从扫描变成 O(1)。

先新增一个状态：

```python
cols = [False] * n
```

它表示：

- `cols[col] == True`：这一列已经被占用

因为列判断已经不需要再扫了，所以把原来的 `is_valid()` 拆成“只检查对角线”的版本。

新增：

```python
def is_valid_diagonal(row: int, col: int) -> bool:
    for prev_row in range(row):
        prev_col = queens[prev_row]
        if abs(prev_row - row) == abs(prev_col - col):
            return False
    return True
```

然后把上一版循环体替换成这个中间版：

```python
for col in range(n):
    if cols[col]:
        continue
    if not is_valid_diagonal(row, col):
        continue

    queens[row] = col
    cols[col] = True

    dfs(row + 1)

    cols[col] = False
    queens[row] = -1
```

这一步不能省。
因为它让读者真的看到：代码不是“从朴素版一下跳到最终版”，而是先快一点，再继续快。

现在这一版能做到：

- 列冲突已经是 O(1) 判断
- 对角线冲突还保留扫描版

但它还缺：

- 对角线还没有变成 O(1)

### Step 7：先定义对角线状态到底在存什么

现在只解决这个问题：

> 如果列能缓存，那对角线到底该缓存什么？

这里先不要急着上代码，要先把对象说清楚。

棋盘上：

- 同一条**主对角线**上的格子，`row - col` 相同
- 同一条**副对角线**上的格子，`row + col` 相同

所以我们真正要记录的不是“某个格子”，而是：

- 某条主对角线有没有皇后
- 某条副对角线有没有皇后

这时才新增两组状态：

```python
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

它们表示：

- `diag1[i]`：编号为 `i` 的主对角线是否被占用
- `diag2[i]`：编号为 `i` 的副对角线是否被占用

再新增下标映射：

```python
d1 = row - col + n - 1
d2 = row + col
```

这里 `+ n - 1` 只是因为 `row - col` 可能是负数。

现在这一版能做到：

- 明确知道对角线状态数组在存什么
- 明确知道 `(row, col)` 怎样映射到两条对角线编号

但它还缺：

- 这些状态还没有真正接回 DFS

### Step 8：把对角线扫描也替换成 O(1) 状态判断

现在终于可以解决最后一个缺口：

> 怎样把“只优化列”的中间版推进到最终优化版？

在上一版中间代码的基础上，把合法性判断和 choose / undo 部分替换成：

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

注意这一步到底改了什么：

- `cols[col]` 这一部分完全继承自上一版
- 只是把“扫描对角线”替换成了 `diag1[d1]` 和 `diag2[d2]`
- choose / undo 时，也从维护一组列状态，升级成同时维护三组状态

现在这一版能做到：

- 列冲突 O(1)
- 主对角线冲突 O(1)
- 副对角线冲突 O(1)
- 保留和前面完全一致的 row-based DFS 骨架

但它还缺：

- 没有本质缺口了，这就是最终逻辑

### Step 9：慢速走一条分支，看状态是怎么一起变化的

还是看 `n = 4`。
假设第 `0` 行先放在第 `1` 列。

那么会同时发生四件事：

- `queens[0] = 1`
- `cols[1] = True`
- `diag1[0 - 1 + 3] = diag1[2] = True`
- `diag2[0 + 1] = diag2[1] = True`

接着来到第 `1` 行：

- `col = 1` 会被 `cols[1]` 直接挡掉
- 有些列会被 `diag1[d1]` 挡掉
- 有些列会被 `diag2[d2]` 挡掉

只有三组状态都允许的位置，才会继续往下递归。

所以最终这题的节奏其实非常稳定：

- 当前行尝试一列
- 查看三组占用状态
- 合法就一起更新主状态和辅助状态
- 递归回来后再一起撤销

走到这里，代码其实已经完整且可运行了：

```python
from typing import List


def solve_n_queens(n: int) -> List[List[str]]:
    res: List[List[str]] = []
    queens = [-1] * n
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def build_board() -> List[str]:
        board = []
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
            cols[col] = True
            diag1[d1] = True
            diag2[d2] = True

            dfs(row + 1)

            queens[row] = -1
            cols[col] = False
            diag1[d1] = False
            diag2[d2] = False

    dfs(0)
    return res
```
