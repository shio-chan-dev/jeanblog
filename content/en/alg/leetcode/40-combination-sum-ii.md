---
title: "LeetCode 40: Combination Sum II (Backtracking / Same-Layer Dedup ACERS Guide)"
date: 2026-04-17T14:31:10+08:00
draft: false
categories: ["LeetCode"]
tags: ["backtracking", "combination sum ii", "dedup", "pruning", "DFS", "LeetCode 40"]
description: "Learn LeetCode 40 by deriving same-layer duplicate skipping, use-once recursion, and sorted pruning from the problem itself."
keywords: ["Combination Sum II", "backtracking", "same-layer dedup", "pruning", "use once", "LeetCode 40"]
---

> **Subtitle / Summary**  
> If `39. Combination Sum` teaches "reuse is allowed", then `40. Combination Sum II` teaches the next real upgrade: duplicate values exist, each number may be used at most once, and de-duplication must happen at the correct tree level.

- **Reading time**: 14-16 min  
- **Tags**: `backtracking`, `dedup`, `pruning`, `combination search`  
- **SEO keywords**: Combination Sum II, backtracking, same-layer dedup, pruning, LeetCode 40  
- **Meta description**: Build the stable solution for LeetCode 40 from scratch by understanding sorted pruning, use-once recursion, and the same-layer duplicate skip rule.  

---

## A — Algorithm

### Problem Restatement

Given a collection of candidate numbers `candidates` and a target integer `target`,  
return all unique combinations where the chosen numbers sum to `target`.

Each number in `candidates` may be used **at most once** in each combination.  
The answer set must not contain duplicate combinations.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| candidates | int[] | candidate values; duplicates may exist |
| target | int | target sum |
| return | `List[List[int]]` | all unique combinations summing to `target` |

### Example 1

```text
input: candidates = [10,1,2,7,6,1,5], target = 8
output:
[
  [1,1,6],
  [1,2,5],
  [1,7],
  [2,6]
]
```

### Example 2

```text
input: candidates = [2,5,2,1,2], target = 5
output:
[
  [1,2,2],
  [5]
]
```

### Constraints

- `1 <= candidates.length <= 100`
- `1 <= candidates[i] <= 50`
- `1 <= target <= 30`

---

## Target Readers

- Learners who already finished `39. Combination Sum` and now want to understand what changes when reuse is forbidden
- Readers who know backtracking syntax but still get duplicate handling wrong
- Developers who want one stable mental rule for "skip duplicates at the same layer"

## Background / Motivation

This problem is the natural follow-up to `39` because it changes two rules at the same time:

- the input itself may contain duplicate values
- each input position may be used only once

That means a naive "just try everything" solution will often generate the same value-combination multiple times.

For example, if the sorted array starts with `[1,1,2,...]`, then:

- choosing the first `1` as the first element may lead to `[1,2]`
- choosing the second `1` as the first element may lead to the same `[1,2]`

So the real question is not only "how do we search?", but also:

- when is a duplicate branch genuinely redundant?
- when must we still allow the same value deeper in the tree?

That distinction is the heart of this problem.

## Core Concepts

- **`path`**: the current combination being built
- **`remain`**: how much sum is still needed
- **`start` / `startIndex`**: the first index allowed in the current layer
- **Use-once recursion**: after choosing `candidates[i]`, the next call must start from `i + 1`
- **Same-layer dedup**: if `i > start` and `candidates[i] == candidates[i - 1]`, skip it
- **Sorted pruning**: once a sorted value is greater than `remain`, later values also fail

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the smallest example that reveals the duplicate problem

Take `candidates = [1,1,2]` and `target = 3`.

The only valid value-combination is:

- `[1,2]`

But if we treat the two `1`s as independent first choices without any control, we can generate `[1,2]` twice:

- first branch starts from the first `1`
- second branch starts from the second `1`

That already tells us:

- this is still a combination problem, not a permutation problem
- duplicate values in the input can create duplicate branches in the search tree

#### Step 2: What state must the partial answer remember?

We are still building one combination gradually, so we need:

```python
path = []
```

`path` means:

- the numbers already chosen on the current recursion branch
- not the full answer list

#### Step 3: What is the smaller subproblem after a few choices are fixed?

Just like `39`, it is cleaner to carry the remaining target directly:

```python
def dfs(start: int, remain: int) -> None:
    ...
```

Here `remain` means:

- how much the current path still needs to reach `target`
- what value shrinks after each choice

#### Step 4: Why do we still need `start`?

Because this is still a combination problem.

If we allow the next layer to go back to earlier indices, then order-based duplicates come back:

- `[1,2]`
- `[2,1]`

So each layer must only enumerate from `start` to the right:

```python
for i in range(start, len(candidates)):
    x = candidates[i]
```

#### Step 5: What changes from `39` when one number may be used only once?

This is the first major difference.

After choosing `candidates[i]`, the next layer must start at `i + 1`, not `i`.

```python
path.append(x)
dfs(i + 1, remain - x)
path.pop()
```

That one line captures the "use each position at most once" rule.

#### Step 6: Why sort first?

Sorting gives us two benefits at once:

1. equal values become adjacent, so duplicate skipping becomes easy
2. pruning with `break` becomes safe

```python
candidates.sort()
```

Without sorting:

- duplicate values are not adjacent
- `if x > remain: break` is not justified

#### Step 7: How do we skip only the truly redundant duplicate branches?

This is the key rule of the whole problem:

```python
if i > start and candidates[i] == candidates[i - 1]:
    continue
```

Read it carefully:

- `i > start` means "this is not the first candidate tried in the current layer"
- `candidates[i] == candidates[i - 1]` means "this candidate has the same value as the previous one"

So the rule says:

- if two equal values compete to be the *same position* in the combination, only try the first one
- but if one equal value was already chosen on a deeper branch, the next equal value may still be valid there

That is why this is called **same-layer dedup**, not global dedup.

#### Step 8: When is one branch complete, and when can we stop early?

Completion is still:

```python
if remain == 0:
    res.append(path.copy())
    return
```

And pruning after sorting is:

```python
if x > remain:
    break
```

This must be `break`, not `continue`, because later values are even larger.

#### Step 9: Walk one branch and one skipped branch slowly

Use the official first example after sorting:

```text
[1,1,2,5,6,7,10], target = 8
```

Start at layer 0:

- choose index `0`, value `1`
- later, when the loop reaches index `1`, value `1`, the condition `i > start and candidates[i] == candidates[i - 1]` is true
- so the second `1` is skipped **at the same layer**

But inside the branch where the first `1` was already chosen:

- `path = [1]`
- next layer starts at index `1`
- now choosing the second `1` is valid, because it is not competing for the same tree layer anymore

That is exactly how `[1,1,6]` remains allowed while duplicate top-level branches disappear.

### Assemble the Full Code

Before the final submission-style solution, here is the assembled standalone version:

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

For LeetCode submission style, the same logic becomes:

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

### What technique did we just build?

Its formal name is:

- backtracking
- combination-style search
- same-layer duplicate skipping
- use-once candidate selection
- pruning after sorting

But the order still matters:

- first observe the problem facts
- then decide the state and transitions
- only then attach the template label

---

## E — Engineering

### Scenario 1: physical coupon bundle selection (Python)

**Background**: a finance tool has a list of coupon cards, and different cards may share the same face value.  
**Why it fits**: each physical card can be used once, and value-duplicate bundles must not be reported twice.

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
            if i > start and x == coupons[i - 1]:
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

### Scenario 2: one-time inventory lot assembly (Go)

**Background**: a backend service must choose physical lots to fill one target amount. Some lots share the same size.  
**Why it fits**: each lot can be consumed once, but equal-value lots should not create duplicate result bundles.

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

### Scenario 3: duplicate-priced option planner (JavaScript)

**Background**: a frontend configurator lists option prices, and different options may share the same price.  
**Why it fits**: each option may be selected once, and the UI should not show the same price-bundle twice.

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

## R — Reflection

### Correctness intuition

This solution works because the following invariants stay true:

- `path` always uses indices in increasing order, so order-based duplicates never appear
- `remain` always equals the missing sum needed by the current path
- `dfs(i + 1, ...)` ensures each input position is used at most once
- the same-layer skip rule prevents equal values from starting redundant sibling branches
- after sorting, `x > remain` safely ends the rest of the layer

### Complexity

Let `n = len(candidates)`.

- Sorting costs `O(n log n)`
- The search tree is subset-like because each position is used at most once
- A standard loose bound is `O(2^n * n)` time, where the extra `n` comes from copying paths into the answer
- Extra recursion space is `O(n)`, excluding the output itself

Actual runtime depends strongly on:

- how effective duplicate skipping is
- how often sorted pruning stops branches early
- how many valid combinations exist

### FAQ

#### Why do we recurse with `i + 1` instead of `i`?

Because each input position may be used at most once.  
Using `i` would incorrectly allow reuse and turn the logic back toward `39. Combination Sum`.

#### Why is the duplicate rule `i > start`, not `i > 0`?

Because dedup is only supposed to remove equal candidates that compete in the **same layer**.

If you write `i > 0`, you may accidentally suppress valid deeper choices such as the second `1` inside `[1,1,6]`.

#### Why must we sort first?

Because sorting is what makes both of these rules valid:

- `if i > start and candidates[i] == candidates[i - 1]: continue`
- `if x > remain: break`

Without sorting, equal values are not adjacent and later values are not guaranteed to be larger.

### Common mistakes

- recursing with `i` and accidentally allowing reuse
- writing the skip rule as `if i > 0 and ...`, which removes valid answers
- forgetting to sort before dedup and pruning
- appending `path` directly instead of a copy

## Best Practices

- Compare this problem directly against `39`, and make the index difference explicit
- Say out loud whether a duplicate rule is "same-layer" or "global" before coding it
- Sort first whenever your pruning or dedup logic depends on order
- Keep `path`, `remain`, and `start` meanings stable across all backtracking problems

## Further Reading

- Official problem: <https://leetcode.com/problems/combination-sum-ii/>
- Direct predecessor: `39. Combination Sum`
- Good next step: `216. Combination Sum III`
- Related duplicate-control problem: `90. Subsets II`

---

## S — Summary

- The real upgrade from `39` to `40` is not just "use once"; it is "use once plus correct duplicate control"
- `dfs(i + 1, remain - x)` is the code-level expression of "each position at most once"
- `if i > start and candidates[i] == candidates[i - 1]: continue` removes only redundant sibling branches
- Sorting is the foundation for both dedup and pruning

### Suggested next practice

- redo `39` and `40` side by side to internalize the difference between `i` and `i + 1`
- then solve `216` to add one more constraint: fixed path length `k`

### CTA

Do not just memorize the skip rule.  
Take `[1,1,2]`, draw the first two layers of the search tree by hand, and explain why one duplicate `1` must be skipped at the same layer but still allowed on a deeper branch.

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
