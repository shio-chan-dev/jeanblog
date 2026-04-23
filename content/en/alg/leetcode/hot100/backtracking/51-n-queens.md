---
title: "Hot100: N-Queens Incremental Build Tutorial"
date: 2026-04-19T14:49:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "constraint search", "board", "N-Queens", "LeetCode 51"]
description: "A teaching-first incremental tutorial for LeetCode 51 that grows the N-Queens solution from a tiny example and a minimal DFS into column and diagonal state optimization."
keywords: ["N-Queens", "backtracking", "constraint search", "diagonals", "LeetCode 51", "Hot100"]
---

`51. N-Queens` is best learned by watching the code grow one small step at a time.
This tutorial keeps only the teaching path: tiny example, first DFS skeleton, first correct version, column-only optimization, then full diagonal-state optimization.

## Problem

The `n`-queens puzzle asks us to place `n` queens on an `n x n` chessboard so that no two queens attack each other.

Given an integer `n`, return all distinct solutions.
Each solution is represented as a list of strings where:

- `'Q'` means a queen
- `'.'` means an empty cell

### Example 1

```text
input: n = 4
output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

### Example 2

```text
input: n = 1
output: [["Q"]]
```

### Constraints

- `1 <= n <= 9`

---

## Build It Step by Step

### Step 1: Start from the smallest example that exposes the conflict

First ask one concrete question:

> what actually becomes difficult after we place one queen?

Use `n = 4`.
Suppose row `0` places a queen at column `1`.

Then the next rows do **not** get to choose from “all empty cells”.
They immediately inherit three restrictions:

- column `1` is blocked
- one top-left to bottom-right diagonal is blocked
- one top-right to bottom-left diagonal is blocked

That tiny example already tells us the real structure of the problem:

- we will place queens in some fixed order
- each choice blocks future positions

This step does not add code yet.
It gives us the evidence that the recursion should be about **one row at a time**, not about scanning arbitrary board cells.

Current version can do:

- identify the shape of the search space

It still lacks:

- a named subproblem
- any code structure

### Step 2: Define the smaller subproblem and write the smallest DFS skeleton

Now solve one specific problem:

> if rows `0 .. row-1` have already been handled, what remains?

The remaining work is:

> choose a column for row `row`, then solve the same problem for row `row + 1`

So one recursion layer should mean:

- `dfs(row)` = “decide where to place the queen for row `row`”

Add this minimal skeleton:

```python
def dfs(row: int) -> None:
    if row == n:
        return

    for col in range(n):
        dfs(row + 1)
```

This is the first real code.

Current version can do:

- express the row-based search structure
- show that one layer corresponds to one row

It still lacks:

- any record of what columns were chosen
- any way to collect an answer
- any legality check

### Step 3: Add the first state variable for the partial answer

Now solve the next problem:

> if we choose a column in one row, where do we store that choice?

Add one state:

```python
queens = [-1] * n
```

Here:

- `queens[row] = col` means row `row` chooses column `col`
- `-1` means that row is still unset

In the previous DFS skeleton, replace the loop body with:

```python
for col in range(n):
    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

Now the code has the real backtracking rhythm:

- choose
- recurse
- undo

Current version can do:

- remember one full path of choices
- undo one choice before trying the next one

It still lacks:

- any leaf action when a full placement is reached
- any legality check

### Step 4: Add the completion rule and build the board

Next question:

> when does one branch become a complete answer?

When `row == n`, every row already has a chosen column.
So now we need one helper that converts `queens` into the required board strings.

Add:

```python
def build_board() -> List[str]:
    board = []
    for col in queens:
        board.append("." * col + "Q" + "." * (n - col - 1))
    return board
```

Then replace the base case:

```python
if row == n:
    res.append(build_board())
    return
```

At this point the code knows:

- what the partial state means
- when a branch is complete
- how to materialize one answer

Current version can do:

- collect a full board if the choices happen to be legal

It still lacks:

- a rule that filters out illegal placements

### Step 5: Add the first correct legality check

Now solve the most urgent missing problem:

> how do we know whether `(row, col)` is legal?

The simplest correct answer is:

- compare it with every queen already placed in rows `0 .. row-1`

Add:

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

Now go back to the loop and replace it with:

```python
for col in range(n):
    if not is_valid(row, col):
        continue

    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

This is the first complete correct version.
It is not optimized, but it is already a valid solver.

Current version can do:

- generate all correct answers
- reject same-column conflicts
- reject diagonal conflicts

It still lacks:

- efficient checking

### Step 6: Keep the diagonal scan for now, but optimize columns first

Now ask a narrower optimization question:

> which repeated check is the easiest to remove first?

The easiest one is the column check.
We do **not** have to optimize everything at once.

Add one new state:

```python
cols = [False] * n
```

This stores:

- `cols[col] == True` if that column is already occupied

Because column checking no longer needs the full `is_valid()`, split the old helper into a diagonal-only check.

Add:

```python
def is_valid_diagonal(row: int, col: int) -> bool:
    for prev_row in range(row):
        prev_col = queens[prev_row]
        if abs(prev_row - row) == abs(prev_col - col):
            return False
    return True
```

Now replace the previous loop with this middle version:

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

This middle version matters.
It proves the final optimized answer does not appear in one jump.

Current version can do:

- check columns in O(1)
- still check diagonals by scanning previous rows

It still lacks:

- O(1) diagonal checks

### Step 7: Define what the diagonal helper arrays should store

Now solve the next specific problem:

> if columns can be cached, what exactly should be cached for diagonals?

First stabilize the concept before the arrays:

- cells on the same **main diagonal** share the same `row - col`
- cells on the same **anti-diagonal** share the same `row + col`

So the helper arrays should not store cells.
They should store whether one diagonal line is already occupied.

Add:

```python
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

Their meanings are:

- `diag1[i]`: main diagonal `i` is occupied
- `diag2[i]`: anti-diagonal `i` is occupied

Now add the index mapping:

```python
d1 = row - col + n - 1
d2 = row + col
```

The `+ n - 1` is only there because `row - col` can be negative.

Current version can do:

- name exactly what information the diagonal helpers should store

It still lacks:

- wiring those helpers into the DFS loop

### Step 8: Replace the diagonal scan with full O(1) state checks

Now we can remove the last scan-based part.

In the previous middle version, replace the legality section and the choose/undo section with:

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

This is the final optimized version.

Notice what changed relative to the middle version:

- `cols[col]` stayed exactly the same
- the diagonal scan was replaced by `diag1[d1]` and `diag2[d2]`
- choose and undo now update three helper states instead of one

Current version can do:

- check columns in O(1)
- check both diagonal families in O(1)
- keep the same row-based DFS skeleton as all earlier versions

It still lacks:

- nothing essential; this is the finished logic

### Step 9: Walk one branch slowly

Still use `n = 4`.
Suppose row `0` chooses column `1`.

Then these facts become true together:

- `queens[0] = 1`
- `cols[1] = True`
- `diag1[0 - 1 + 3] = diag1[2] = True`
- `diag2[0 + 1] = diag2[1] = True`

Now move to row `1`.

When the loop tries candidates:

- `col = 1` fails immediately because `cols[1]` is already true
- some candidates fail because `diag1[d1]` is true
- some candidates fail because `diag2[d2]` is true

Only columns that survive all three tests continue.

That is the whole final rhythm:

- choose a column for the current row
- check stored occupancy state
- recurse
- undo everything before the next choice

At this point, the code is already complete and runnable:

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
