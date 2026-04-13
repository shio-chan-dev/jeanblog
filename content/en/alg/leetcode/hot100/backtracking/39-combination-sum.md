---
title: "Hot100: Combination Sum (Backtracking / Pruning ACERS Guide)"
date: 2026-04-08T16:22:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "combination sum", "pruning", "DFS", "LeetCode 39"]
description: "A practical guide to LeetCode 39 that builds combination backtracking from scratch, explains remain-based state, and shows why sorting makes pruning safe."
keywords: ["Combination Sum", "backtracking", "pruning", "remain", "DFS", "LeetCode 39", "Hot100"]
---

> **Subtitle / Summary**  
> Combination Sum is the first Hot100 backtracking problem that really mixes three ideas at once: combination-style search, a running target, and safe pruning after sorting. The point is not to jump straight to the template, but to build it step by step from the problem itself.

- **Reading time**: 14-16 min  
- **Tags**: `Hot100`, `backtracking`, `combination sum`, `pruning`  
- **SEO keywords**: Combination Sum, backtracking, pruning, remain, DFS  
- **Meta description**: Learn LeetCode 39 by building the solution from scratch, using `path`, `remain`, repeated candidate reuse, and sorted pruning.  

---

## A — Algorithm

### Problem Restatement

Given an array of distinct integers `candidates` and a target integer `target`,  
return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`.

The same number may be chosen from `candidates` an unlimited number of times.  
Two combinations are different if the frequency of at least one chosen number differs.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| candidates | int[] | distinct candidate values |
| target | int | target sum |
| return | `List[List[int]]` | all unique combinations summing to `target` |

### Example 1

```text
input: candidates = [2,3,6,7], target = 7
output: [[2,2,3],[7]]
```

### Example 2

```text
input: candidates = [2,3,5], target = 8
output: [[2,2,2,2],[2,3,3],[3,5]]
```

### Example 3

```text
input: candidates = [2], target = 1
output: []
```

### Constraints

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- all elements of `candidates` are distinct
- `1 <= target <= 40`
- the number of unique combinations is guaranteed to be less than `150`

---

## Target Readers

- Hot100 learners who already finished `78. Subsets` and now want the next backtracking upgrade
- Developers who understand recursion in general but still get confused when one candidate may be reused
- Readers who want a stable template they can transfer to `40` and `216`, not just one memorized answer

## Background / Motivation

`39. Combination Sum` is a good second backtracking problem because it forces you to manage three things at the same time:

- it is still a combination problem, so order must not create duplicates
- the same number may be chosen multiple times
- the target value gives us a natural pruning boundary

That combination is exactly where many learners stop “following the pattern” and start needing a real model.

The model we want is:

- what partial answer are we building
- how far are we from the target
- where may the next layer continue
- when can a whole suffix of the search be cut off immediately

Once that model is stable, Combination Sum stops feeling special and starts feeling reusable.

## Core Concepts

- **`path`**: the numbers currently chosen in the active combination
- **`remain`**: how much is still needed to reach `target`
- **`start` / `startIndex`**: the first index allowed in the current layer
- **Reusable candidates**: after choosing `candidates[i]`, the next call still starts from `i`
- **Sorted pruning**: once a sorted value is greater than `remain`, later values are impossible too

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from a tiny but non-trivial example

Take `candidates = [2,3,6,7]` and `target = 7`.

Instead of asking “what is the final algorithm?”, ask the smaller question:

- if I choose `2` first, I still need `5`
- if I choose `2` again, I still need `3`
- if I then choose `3`, I hit the target and get `[2,2,3]`
- if I choose `7` first, I also get one full answer: `[7]`

This tiny example already reveals two facts:

- the problem is about combinations, not permutations
- we need to keep track of how much is still missing

It also shows why a stable boundary is necessary:

- `[2,2,3]` and `[2,3,2]` must not both appear
- once a candidate is already larger than the remaining target, later candidates are useless too after sorting

#### Step 2: What must the partial answer remember?

If we are building one combination gradually, we need to store the numbers already chosen.  
That is why we need `path`.

```python
path = []
```

`path` is not the full answer set.  
It is only the current branch of the search tree.

#### Step 3: How do we know how far we still are from the target?

We could recompute `sum(path)` every time, but that is a worse teaching model and a worse pruning model.  
It is cleaner to carry the remaining amount directly.

```python
def dfs(start: int, remain: int) -> None:
    ...
```

Here `remain` means:

- how much the current path still needs
- what value will shrink after each choice

#### Step 4: How do we prevent order-based duplicates?

Because this is a combination problem, `[2,2,3]` and `[2,3,2]` must not both appear.  
The stable way to control that is: in each layer, only enumerate from `start` to the right.

```python
for i in range(start, len(candidates)):
    x = candidates[i]
```

That rule means:

- do not go back to earlier indices
- build combinations in non-decreasing order

#### Step 5: When is one branch complete?

When `remain == 0`, the current `path` sums exactly to `target`.  
That is when we collect one answer.

```python
if remain == 0:
    res.append(path.copy())
    return
```

The `.copy()` matters because `path` will keep changing during backtracking.

#### Step 6: Why sort first?

Sorting is not needed for de-duplication here.  
It is needed so pruning becomes safe.

```python
candidates.sort()
```

Once the array is sorted, “current value too large” implies “every later value is too large too.”

#### Step 7: What do we do when the current value is already too large?

After sorting, if `x > remain`, there is no reason to continue scanning the rest of the layer.

```python
if x > remain:
    break
```

This must be `break`, not `continue`.  
Later values are even larger.

#### Step 8: What changes after we choose one number?

After choosing `x`, we do three things:

1. add it to `path`
2. solve the smaller problem
3. undo the choice afterward

```python
path.append(x)
dfs(i, remain - x)
path.pop()
```

The most important detail is the recursive call:

- `dfs(i, remain - x)` means the same candidate may still be reused
- `dfs(i + 1, remain - x)` would incorrectly turn this into a “use once” problem

#### Step 9: Walk one branch slowly

Use the same example: `candidates = [2,3,6,7], target = 7`.

Start:

- `path = []`
- `remain = 7`
- `start = 0`

Choose `2`:

- `path = [2]`
- `remain = 5`
- the next layer still starts at index `0`

Choose `2` again:

- `path = [2,2]`
- `remain = 3`

Now inside that call:

- try `2` first and get `remain = 1`
- at that point the next candidate is already larger than `1`, so the layer breaks
- backtrack to `path = [2,2]`

Then try `3`:

- `path = [2,2,3]`
- `remain = 0`
- collect `[2,2,3]`

That single branch shows the whole method:

- why `remain` is useful
- why the next call keeps `i`
- why sorting makes `break` safe

### Assemble the Full Code

Now we combine the fragments above into the first complete working solution.  
This version is runnable as a normal script.

```python
from typing import List


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:
                break

            path.append(x)
            dfs(i, remain - x)
            path.pop()

    dfs(0, target)
    return res


if __name__ == "__main__":
    print(combination_sum([2, 3, 6, 7], 7))
    print(combination_sum([2, 3, 5], 8))
```

### Reference Answer

For LeetCode submission style, the same logic becomes:

```python
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int, remain: int) -> None:
            if remain == 0:
                res.append(path.copy())
                return

            for i in range(start, len(candidates)):
                x = candidates[i]
                if x > remain:
                    break

                path.append(x)
                dfs(i, remain - x)
                path.pop()

        dfs(0, target)
        return res
```

### What technique did we just build?

Its formal name is:

- backtracking
- combination-style search
- pruning after sorting
- reusable candidate selection

But the order matters.  
You should not start from the label.  
You should start from the problem facts and let those facts force the state and the transitions into place.

---

## E — Engineering

### Scenario 1: exact budget combinations (Python)

**Background**: a budgeting tool wants every combination of allowed amounts that exactly reaches one budget.  
**Why it fits**: values may be reused, order does not matter, and the total must match exactly.

```python
def fill_budget(costs, target):
    costs = sorted(costs)
    res = []
    path = []

    def dfs(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(costs)):
            if costs[i] > remain:
                break
            path.append(costs[i])
            dfs(i, remain - costs[i])
            path.pop()

    dfs(0, target)
    return res


print(fill_budget([2, 3, 5], 8))
```

### Scenario 2: capacity pack assembly (Go)

**Background**: a backend system wants all ways to assemble a target capacity from reusable pack sizes.  
**Why it fits**: pack sizes can repeat, order is irrelevant, and the exact total matters.

```go
package main

import (
	"fmt"
	"sort"
)

func fill(capacities []int, target int) [][]int {
	sort.Ints(capacities)
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if remain == 0 {
			res = append(res, append([]int(nil), path...))
			return
		}
		for i := start; i < len(capacities); i++ {
			if capacities[i] > remain {
				break
			}
			path = append(path, capacities[i])
			dfs(i, remain-capacities[i])
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}

func main() {
	fmt.Println(fill([]int{2, 3, 5}, 8))
}
```

### Scenario 3: pricing bundle builder (JavaScript)

**Background**: a frontend bundle configurator wants every pricing bundle that lands exactly on the user budget.  
**Why it fits**: prices may be reused and bundle order is irrelevant.

```javascript
function combinationSum(candidates, target) {
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
      if (x > remain) break;
      path.push(x);
      dfs(i, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}

console.log(combinationSum([2, 3, 5], 8));
```

---

## R — Reflection

### Correctness intuition

This solution works because three invariants stay true throughout the search:

- the numbers in `path` are built in non-decreasing index order, so order-based duplicates never appear
- `remain` always means “how much is still needed,” so `remain == 0` is exactly the success condition
- after sorting, if `x > remain`, every later candidate also fails, so breaking the loop is safe

### Complexity

Let:

- `n = len(candidates)`
- `m = min(candidates)`
- `d = target / m`

Then the recursion depth is at most `d`, because every step reduces `remain` by at least `m`.  
A standard loose upper bound is:

- Time complexity: `O(n^d)`, where `d = target / min(candidates)`
- Extra recursion space: `O(d)`, excluding the output itself

This problem does not have a neat fixed closed-form count like `2^n` or `n!`.  
Actual runtime depends heavily on pruning and on how many valid combinations exist.

### FAQ

#### Why do we recurse with `i` instead of `i + 1`?

Because the current candidate may still be reused.  
Using `i + 1` would incorrectly change the problem into a “choose each number at most once” variant.

#### Why can we `break` instead of `continue`?

Because the array is sorted.  
If the current value is already too large, later values are even larger.

#### Why not keep computing `sum(path)`?

You can, but `remain` is a cleaner teaching and coding model:

- completion is `remain == 0`
- pruning is `x > remain`
- there is no repeated summation work in the recursion

### Common mistakes

- using `if x > remain: break` without sorting first
- appending `path` directly instead of a copy
- recursing with `i + 1` and accidentally solving a different problem
- restarting every layer from index `0` and turning combinations into permutation-like duplicates

## Best Practices

- Before coding, say out loud what `path`, `remain`, and `start` each mean
- In “reuse allowed” problems, check the recursive index before you trust the solution
- Only use `break` pruning after sorting is done
- If you just studied `78. Subsets`, rewrite this one side by side and compare the role of `start`

## Further Reading

- Official problem: <https://leetcode.cn/problems/combination-sum/>
- Recommended next problem: `40. Combination Sum II`
- Good comparison problems: `78. Subsets`, `46. Permutations`

---

## S — Summary

- This is still combination-style backtracking, not permutation-style backtracking
- `remain` makes both completion and pruning conditions explicit
- `dfs(i, remain - x)` is the concrete expression of “this candidate may be reused”
- sorting plus `if x > remain: break` is the most stable pruning rule in this problem

### Suggested Next Problems

- `78. Subsets`: review why `startIndex` removes order duplicates
- `46. Permutations`: compare combination-style and permutation-style backtracking
- `40. Combination Sum II`: learn “no reuse” plus duplicate handling
- `216. Combination Sum III`: keep practicing target-constrained search

### CTA

If you study only one backtracking problem today, rewrite this one once from a blank page.  
Do not start from the final code. Start by forcing yourself to explain why you need `path`, `remain`, `start`, and sorted pruning.

---

## Multi-Language Implementations

### Python

```python
from typing import List


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return

        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:
                break
            path.append(x)
            dfs(i, remain - x)
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
        if (x > remain) {
            break;
        }
        path[path_size] = x;
        dfs(candidates, candidates_size, i, remain - x, path, path_size + 1, res);
    }
}

int** combinationSum(int* candidates, int candidatesSize, int target,
                     int* returnSize, int** returnColumnSizes) {
    qsort(candidates, candidatesSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.rows = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * (target + 1));
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
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
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
            if (x > remain) {
                break;
            }
            path.push_back(x);
            dfs(candidates, i, remain - x, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

import "sort"

func combinationSum(candidates []int, target int) [][]int {
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
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(i, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(0, target)
	return res
}
```

### Rust

```rust
fn combination_sum(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
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
            if x > remain {
                break;
            }
            path.push(x);
            dfs(candidates, i, remain - x, path, res);
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
function combinationSum(candidates, target) {
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
      if (x > remain) break;
      path.push(x);
      dfs(i, remain - x);
      path.pop();
    }
  }

  dfs(0, target);
  return res;
}
```
