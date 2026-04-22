---
title: "Hot100: N-Queens (Columns / Diagonals Backtracking ACERS Guide)"
date: 2026-04-19T14:49:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "constraint search", "board", "N-Queens", "LeetCode 51"]
description: "A practical guide to LeetCode 51 that builds the N-Queens solution from scratch using row-by-row placement and O(1) conflict checks for columns and diagonals."
keywords: ["N-Queens", "backtracking", "constraint search", "diagonals", "LeetCode 51", "Hot100"]
---

> **Subtitle / Summary**
> `51. N-Queens` is one of the key Hot100 constraint-search problems. The real lesson is not “try random cells on a board”, but “place queens row by row and turn conflict checks into O(1) state lookups”.

- **Reading time**: 16-20 min
- **Tags**: `Hot100`, `backtracking`, `constraint search`, `board`, `N-Queens`
- **SEO keywords**: N-Queens, backtracking, constraint search, diagonals, LeetCode 51
- **Meta description**: Learn LeetCode 51 by building row-based constraint search with columns and two diagonal state arrays, plus runnable multi-language solutions.

---

## A — Algorithm

### Problem Restatement

The `n`-queens puzzle asks us to place `n` queens on an `n x n` chessboard so that no two queens attack each other.

Given an integer `n`, return all distinct solutions.
Each solution is represented as a list of strings where:

- `'Q'` means a queen
- `'.'` means an empty cell

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| n | int | board size and number of queens |
| return | `List[List[str]]` | all valid board configurations |

### Example 1

```text
input: n = 4
output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
explanation: there are two distinct valid solutions for the 4-queens puzzle
```

### Example 2

```text
input: n = 1
output: [["Q"]]
```

### Constraints

- `1 <= n <= 9`

---

## Target Readers

- learners who know basic DFS but have not yet built a true multi-constraint search template
- developers whose first instinct on board problems is still “scan the whole board every time”
- readers who want a reusable model for row-by-row placement with O(1) legality checks

## Background / Motivation

`51. N-Queens` is a classic constraint-search problem.

Unlike simpler backtracking tasks, every new queen introduces several simultaneous restrictions:

- its column is blocked
- one main diagonal is blocked
- one anti-diagonal is blocked

That is why this problem matters.
It teaches a general pattern:

> place objects in a fixed order, and make conflict checks constant-time through explicit state

This pattern transfers well to scheduling, layout generation, resource placement, and other constrained search tasks.

## Core Concepts

- **Row-based DFS**: each layer decides only where the current row should place its queen
- **`queens[row] = col`**: the partial answer we are building
- **Legality checking**: first write a scanning version, then upgrade it to O(1) state lookups
- **Leaf-only collection**: only a fully filled board is a valid answer

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Write the roughest row-based DFS skeleton first

Before worrying about every constraint, answer one smaller question:

> what should one recursion layer mean?

Because each row must eventually contain exactly one queen, the cleanest choice is: `dfs(row)` means “we are deciding where to place the queen for row `row`”.

```python
def dfs(row: int) -> None:
    if row == n:
        return

    for col in range(n):
        dfs(row + 1)
```

This version is obviously incomplete, but it locks in the real subproblem:

- one recursion layer handles one row
- the choice in that layer is which column to use

#### Step 2: Record what choice the current layer made

Now the code needs a place to remember where earlier rows put their queens.

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

This is still not correct yet, but the backtracking structure is finally visible:

- `queens[row] = col` means “row `row` chooses column `col`”
- recurse means “go decide the next row”
- restoring `-1` means “undo this choice before trying another column”

#### Step 3: When does one branch become a full answer?

When `row == n`, rows `0 .. n-1` have all been assigned.
At that point we should not just `return`; we should convert `queens` into a board and collect it.

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

At this stage the reader already knows:

- what the recursion index means
- what the partial answer means
- what the leaf node must do

#### Step 4: What is still missing? A legality check

If we stop here, the search will also collect illegal boards.
Two queens may land in the same column or on the same diagonal, and this code would still continue.

So the next missing piece is not more framework.
It is the simplest possible legality check:

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

Then place it back into DFS:

```python
for col in range(n):
    if not is_valid(row, col):
        continue
    queens[row] = col
    dfs(row + 1)
    queens[row] = -1
```

This version is already complete, correct, and runnable.
It is just not optimized yet.

#### Step 5: Explain diagonal conflict before talking about diagonal arrays

This is where many readers get lost.
`diag1` and `diag2` should not appear before the diagonal idea itself is stable.

For one board cell `(row, col)`, a queen attacks not only the whole column, but also two slanted line families:

- the top-left to bottom-right family
- the top-right to bottom-left family

If two cells lie on the same diagonal, then the absolute row difference equals the absolute column difference.

That means this line in the naive checker:

```python
if abs(prev_row - row) == abs(prev_col - col):
    return False
```

is really answering:

> does `(row, col)` lie on the same diagonal as a previously placed queen?

Once that geometric fact is clear, the later state arrays feel earned instead of magical.

#### Step 6: This works, so where is it slow?

The bottleneck is `is_valid()`.

Every time we try one candidate `(row, col)`, we scan all previous rows again.
That means we keep recomputing the same facts over and over.

The first fact we can cache is column occupancy:

```python
cols = [False] * n
```

Then column conflict becomes one direct lookup:

```python
if cols[col]:
    continue
```

This is the direction of the optimization:

- the naive version scans old state
- the optimized version queries occupancy state directly

#### Step 7: Now introduce the two diagonal arrays and what they store

Only now is it the right time to define `diag1` and `diag2`, because now they have a clear job:
replace the diagonal part of `is_valid()`.

Every cell `(row, col)` belongs to one line from each diagonal family:

- **main diagonal**: top-left to bottom-right, where all cells share the same `row - col`
- **anti-diagonal**: top-right to bottom-left, where all cells share the same `row + col`

So we can store whether one diagonal line is already occupied:

```python
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

These arrays do not store board cells.
They store line occupancy:

- `diag1[i]` means main diagonal `i` is occupied
- `diag2[i]` means anti-diagonal `i` is occupied

Because `row - col` can be negative, we shift it into a valid array range:

```python
d1 = row - col + n - 1
d2 = row + col
```

The order matters:

- first observe the two diagonal families on the board
- then notice the `row - col` / `row + col` rules
- only then compress them into array indices

#### Step 8: Replace the naive checker with O(1) state lookups

Now we can swap out the scan-based checker entirely.

First prepare the three occupancy structures:

```python
cols = [False] * n
diag1 = [False] * (2 * n - 1)
diag2 = [False] * (2 * n - 1)
```

Then use them directly inside the DFS loop:

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

This is where the optimized version grows naturally out of the earlier one:

- `queens` is still the main state
- `cols / diag1 / diag2` are helper states for O(1) legality checks
- the choose / recurse / undo skeleton stays exactly the same

#### Step 9: Walk one branch slowly and watch the states move together

Still use `n = 4`, and suppose row `0` chooses column `1`.

Then four facts change together:

- `queens[0] = 1`
- `cols[1] = True`
- `diag1[0 - 1 + 3] = diag1[2] = True`
- `diag2[0 + 1] = diag2[1] = True`

Now look at row `1`:

- `col = 1` is blocked by `cols[1]`
- some columns are blocked by `diag1`
- some columns are blocked by `diag2`

Only columns that survive all three checks can continue.

That means the final rhythm of the whole solution is very stable:

- choose one column for the current row
- check three occupancy states
- update the main state and helper states together
- undo all of them together on return

### Assemble the Full Code

Now assemble the full runnable version.

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

For the LeetCode submission form:

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

### What method did we just build?

Formally, it is:

- backtracking
- constraint search
- row-by-row placement with explicit occupancy state

But the most important reusable model is:

- one layer handles exactly one row
- `queens` stores the main placement state
- `cols`, `diag1`, and `diag2` make legality checks O(1)
- only a fully placed board is collected

---

## E — Engineering

### Scenario 1: Puzzle or board-layout prototype (Python)

**Background**: a puzzle generator or teaching demo needs all valid non-conflicting placements on a small grid.  
**Why this fits**: N-Queens is itself a classic constrained layout generation task.

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

### Scenario 2: Grid device-placement prototype (Go)

**Background**: a system prototype places one device per row on a grid, with column and diagonal-style interference constraints.  
**Why this fits**: the structure is the same as N-Queens: row-based placement with multiple constant-time constraints.

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

### Scenario 3: Frontend constrained layout preview (JavaScript)

**Background**: a frontend design tool wants to preview all non-conflicting placements for one widget per row in a small grid.  
**Why this fits**: once the constraints can be encoded as columns and two diagonal-like sets, the same template applies.

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

## R — Reflection

### Complexity Analysis

- Search time complexity: often summarized as `O(n!)`
  - each row tries candidate columns
  - pruning removes many branches in practice, but the upper bound is still permutation-like
- Auxiliary space: `O(n)`
  - `queens`, recursion depth, and occupancy states are all linear in `n`
- If board construction is counted:
  - each solution costs `O(n^2)` to materialize as strings
  - total output cost also depends on the number of valid solutions

### Comparison with other approaches

| Method | Idea | Advantage | Limitation |
| --- | --- | --- | --- |
| Column + diagonal occupancy arrays | O(1) legality checks | stable and readable | slightly longer setup |
| Scan the board every time | test attacks by walking rows/diagonals | easy to invent | slower and messier |
| Bitmask optimization | compress state into integers | very fast | worse teaching fit for a first pass |

### Common Mistakes

- searching arbitrary board cells instead of fixing one row per layer
- getting diagonal index formulas wrong
- updating columns but forgetting to update or undo diagonals
- storing `queens` directly instead of converting it into board strings

## Common Questions and Pitfalls

### Why is row-by-row placement enough?

Because each row must contain exactly one queen.
Once the recursion layer is defined as a row, the row constraint disappears from the search logic and only columns plus diagonals remain.

### Why is the main diagonal index `row - col + n - 1`?

All cells on one main diagonal share `row - col`, but that value can be negative.
Adding `n - 1` shifts it safely into the array range `0 .. 2n-2`.

### Can this problem be optimized further?

Yes.

A bitmask version can compress:

- columns
- main diagonals
- anti-diagonals

into integers for faster search.
But the boolean-array version is the right one to stabilize first.

## Best Practices and Recommendations

- on grid-constraint problems, first ask whether recursion can be fixed by row or by column
- convert repeated legality checks into array or bitmask lookups whenever possible
- restore the main state and helper states together after recursion
- get the `n = 4` version fully stable before touching bitmask optimizations

---

## S — Summary

- `51. N-Queens` is fundamentally a constraint-search template
- once recursion is row-based, each layer becomes “which column is legal here?”
- `cols`, `diag1`, and `diag2` reduce conflict checks to O(1)
- the final answer is a string board, not just a raw column-position array
- this pattern transfers well to many layout and placement problems

### Recommended Follow-Up Reading

- `52. N-Queens II`: same search structure, but count only
- `46. Permutations`: another example of main state plus helper-state recovery
- `37. Sudoku Solver`: a richer multi-constraint board search
- bitmask N-Queens: the next optimization step after the boolean-array version is stable

### Action Step

Before learning the bitmask version, rewrite the boolean-array version from memory once.
If you can explain why the recursion is row-based and why three occupancy states are enough, the core model is already solid.

---

## Multi-Language Implementations

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
                queens[row] = usize::MAX;
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
