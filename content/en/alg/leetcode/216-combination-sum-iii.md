---
title: "LeetCode 216: Combination Sum III (Backtracking / Fixed-Length Search ACERS Guide)"
date: 2026-04-17T14:31:12+08:00
draft: false
categories: ["LeetCode"]
tags: ["backtracking", "combination sum iii", "fixed length", "pruning", "DFS", "LeetCode 216"]
description: "Learn LeetCode 216 by deriving fixed-length combination search, use-once recursion, and bounded 1..9 pruning from the problem itself."
keywords: ["Combination Sum III", "backtracking", "fixed length", "k numbers", "pruning", "LeetCode 216"]
---

> **Subtitle / Summary**  
> `216. Combination Sum III` is where backtracking adds one more important constraint: not only must the sum match, but the combination length must be exactly `k`. That turns the problem into a clean fixed-length combination search over the bounded range `1..9`.

- **Reading time**: 12-15 min  
- **Tags**: `backtracking`, `fixed length`, `pruning`, `combination search`  
- **SEO keywords**: Combination Sum III, fixed-length backtracking, k numbers, pruning, LeetCode 216  
- **Meta description**: Build the stable solution for LeetCode 216 from scratch by understanding exact-length backtracking, sorted pruning, and the bounded candidate set 1..9.  

---

## A — Algorithm

### Problem Restatement

Find all valid combinations of `k` numbers that add up to `n`, subject to these rules:

- only numbers `1` through `9` may be used
- each number may be used **at most once**

Return all valid combinations.  
The answer must not contain duplicate combinations, and the order of combinations does not matter.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| k | int | required number of chosen values |
| n | int | target sum |
| return | `List[List[int]]` | all valid length-`k` combinations summing to `n` |

### Example 1

```text
input: k = 3, n = 7
output: [[1,2,4]]
```

### Example 2

```text
input: k = 3, n = 9
output: [[1,2,6],[1,3,5],[2,3,4]]
```

### Example 3

```text
input: k = 4, n = 1
output: []
```

### Constraints

- `2 <= k <= 9`
- `1 <= n <= 60`

---

## Target Readers

- Learners who already understand `39` and `40`, and now want to add the "exactly `k` numbers" constraint
- Readers who are comfortable with basic DFS but still mix up "target sum" and "required path length"
- Developers who want a stable pattern for fixed-size combination search

## Background / Motivation

This problem is a strong follow-up to `39` and `40` because it keeps the backtracking skeleton but changes the search space:

- the candidate range is fixed: `1..9`
- each number is used at most once
- the answer must contain **exactly** `k` numbers

That last rule matters a lot.

In `39`, reaching the target sum is enough.  
In `216`, reaching the sum too early is not enough if the path length is wrong, and reaching length `k` is not enough if the sum is still wrong.

So the model must now track two completion dimensions:

- how much sum is left
- how many positions are already filled

## Core Concepts

- **`path`**: the numbers currently chosen
- **`remain`**: how much sum is still needed
- **`start`**: the next smallest allowed candidate in `1..9`
- **Fixed length `k`**: the branch is complete only when `len(path) == k`
- **Use-once recursion**: after choosing `x`, the next call starts from `x + 1`
- **Bounded search space**: only digits `1..9` are considered

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the smallest example that shows the fixed-length rule

Take `k = 3`, `n = 7`.

The answer is:

- `[1,2,4]`

This tiny example already reveals the real shape of the task:

- the numbers must be distinct
- order does not matter
- the sum must be `7`
- the combination length must be exactly `3`

So `[7]` is invalid even though the sum matches.  
The length is wrong.

#### Step 2: What does the partial answer need to remember?

We still need the current branch:

```python
path = []
```

`path` means:

- the numbers already chosen
- how many slots are already filled

#### Step 3: What remains to solve after a few choices are fixed?

As before, carrying the remaining target directly gives the cleanest model:

```python
def dfs(start: int, remain: int) -> None:
    ...
```

Now the recursive state is interpreted as:

- `start`: the next smallest number we may try
- `remain`: how much sum is still needed
- `path`: how many numbers we already picked

#### Step 4: When is one branch complete?

This is the first big difference from `39`.

A branch is structurally complete when:

```python
if len(path) == k:
    ...
```

At that moment:

- if `remain == 0`, we collect the answer
- otherwise the branch fails

```python
if len(path) == k:
    if remain == 0:
        res.append(path.copy())
    return
```

#### Step 5: What choices are available next?

The candidates are not taken from an input array anymore.  
They are the fixed numbers `1..9`.

So a layer enumerates like this:

```python
for x in range(start, 10):
    ...
```

This already guarantees:

- numbers are increasing
- each number is used at most once
- duplicate combinations never appear

#### Step 6: What changes after choosing one number?

If we choose `x`, the next layer must start from `x + 1` because the current number may not be reused.

```python
path.append(x)
dfs(x + 1, remain - x)
path.pop()
```

This is the same "use once" idea we learned in `40`, but here it is even simpler because the candidate values are exactly `1..9`.

#### Step 7: What pruning is naturally available?

Because `x` increases from small to large, once `x > remain`, later values are also too large:

```python
if x > remain:
    break
```

There is also a length-based early return:

```python
if len(path) == k:
    ...
```

That stops any branch from growing beyond the required size.

#### Step 8: Walk one branch slowly

Use `k = 3`, `n = 9`.

Start:

- `path = []`
- `remain = 9`
- `start = 1`

Choose `1`:

- `path = [1]`
- `remain = 8`
- next start is `2`

Choose `2`:

- `path = [1,2]`
- `remain = 6`
- next start is `3`

Choose `6`:

- `path = [1,2,6]`
- `remain = 0`
- `len(path) == 3`

So `[1,2,6]` is collected.

This one branch already shows the full method:

- increasing choices avoid duplicates
- `remain` tracks the target
- `len(path)` enforces the exact number count

### Assemble the Full Code

Before the final submission-style version, here is the assembled standalone implementation:

```python
from typing import List


def combination_sum_iii(k: int, n: int) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if len(path) == k:
            if remain == 0:
                res.append(path.copy())
            return

        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, n)
    return res


if __name__ == "__main__":
    print(combination_sum_iii(3, 7))
    print(combination_sum_iii(3, 9))
```

### Reference Answer

For LeetCode submission style:

```python
from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int, remain: int) -> None:
            if len(path) == k:
                if remain == 0:
                    res.append(path.copy())
                return

            for x in range(start, 10):
                if x > remain:
                    break
                path.append(x)
                dfs(x + 1, remain - x)
                path.pop()

        dfs(1, n)
        return res
```

### What technique did we just build?

Its formal name is:

- backtracking
- fixed-length combination search
- use-once selection
- bounded search over `1..9`

Again, do not start from the label.  
Start from the problem facts:

- target sum
- exact path length
- distinct increasing choices

Then the template becomes obvious.

---

## E — Engineering

### Scenario 1: fixed-size budget package selection (Python)

**Background**: an analytics workflow must choose exactly `k` one-time credits whose total cost equals a target.  
**Why it fits**: every credit can be used once, the bundle size is fixed, and the sum must match exactly.

```python
def choose_credits(k, target):
    res = []
    path = []

    def dfs(start, remain):
        if len(path) == k:
            if remain == 0:
                res.append(path[:])
            return
        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, target)
    return res


print(choose_credits(3, 9))
```

### Scenario 2: exact-slot resource picks (Go)

**Background**: a backend allocator must pick exactly `k` distinct resource IDs from a small bounded range and hit one target score.  
**Why it fits**: the candidate set is tiny, each ID is used once, and the result size is fixed.

```go
package main

import "fmt"

func chooseResources(k, target int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if len(path) == k {
			if remain == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}
		for x := start; x <= 9; x++ {
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(x+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(1, target)
	return res
}

func main() {
	fmt.Println(chooseResources(3, 7))
}
```

### Scenario 3: front-end fixed-chip planner (JavaScript)

**Background**: a front-end planner lets the user pick exactly `k` chips, each with a unique weight from `1..9`, to hit one score.  
**Why it fits**: values are bounded, each chip is used once, and result length is fixed.

```javascript
function combinationSum3(k, n) {
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (path.length === k) {
      if (remain === 0) res.push([...path]);
      return;
    }

    for (let x = start; x <= 9; x += 1) {
      if (x > remain) break;
      path.push(x);
      dfs(x + 1, remain - x);
      path.pop();
    }
  }

  dfs(1, n);
  return res;
}

console.log(combinationSum3(3, 9));
```

---

## R — Reflection

### Correctness intuition

This solution works because:

- the search always moves from small to large numbers, so order-based duplicates never appear
- `dfs(x + 1, ...)` guarantees each number is used at most once
- `remain` always tracks the missing sum for the current path
- `len(path) == k` enforces the exact required size
- since the candidates are increasing, `x > remain` safely stops the rest of the layer

### Complexity

The candidate set is fixed to `1..9`, so the total search space is small.

A useful way to describe it is:

- the algorithm explores fixed-length combinations from at most 9 numbers
- a natural bound is `O(C(9, k) * k)` time, plus answer-copy costs
- recursion depth is `O(k)`

In practice, this problem is much smaller than general combination-sum variants because:

- the candidate domain is tiny
- values are unique
- the required length is fixed

### FAQ

#### Why do we need `len(path) == k` if `remain == 0` already means success?

Because in this problem, matching the sum is not enough by itself.  
The answer must also use exactly `k` numbers.

#### Why is there no duplicate-skip rule like in `40`?

Because the candidate values are already unique: `1,2,3,...,9`.  
There are no repeated input values to deduplicate.

#### Why does the recursive call use `x + 1`?

Because each number may be used at most once, and later choices must stay strictly larger to preserve combination order.

### Common mistakes

- collecting when `remain == 0` without checking `len(path) == k`
- recursing with `x` instead of `x + 1`
- treating the task as an unlimited-reuse problem like `39`
- forgetting that the candidate range is fixed and overcomplicating the state

## Best Practices

- When a problem says "choose exactly `k` numbers", make the path length part of the completion rule immediately
- Keep the candidate range explicit; here `1..9` is not an implementation detail, it is part of the problem structure
- Compare `216` directly against `39` and `40` to understand which constraints create which code changes
- In bounded-range problems, use the order of candidates to justify pruning cleanly

## Further Reading

- Official problem: <https://leetcode.com/problems/combination-sum-iii/>
- Direct comparisons: `39. Combination Sum`, `40. Combination Sum II`
- Related fixed-length combination problem: `77. Combinations`

---

## S — Summary

- `216` is a fixed-length combination search, not just a target-sum search
- `len(path) == k` is part of the success condition, not a side check
- `dfs(x + 1, remain - x)` captures "use once" and "keep order" at the same time
- The bounded range `1..9` makes the problem smaller and the pruning cleaner

### Suggested next practice

- rewrite `39`, `40`, and `216` back to back and compare what changes in the recursive call and base case
- then move to `90. Subsets II` or `77. Combinations` to reinforce duplicate control and fixed-length thinking

### CTA

Try explaining this problem without saying "it is a standard backtracking problem."  
If you can describe it as "pick exactly `k` increasing numbers from `1..9` so their sum becomes `n`", then the code will usually fall into place naturally.

---

## Multi-Language Implementations

### Python

```python
from typing import List


def combination_sum_iii(k: int, n: int) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int, remain: int) -> None:
        if len(path) == k:
            if remain == 0:
                res.append(path.copy())
            return

        for x in range(start, 10):
            if x > remain:
                break
            path.append(x)
            dfs(x + 1, remain - x)
            path.pop()

    dfs(1, n)
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

static void dfs(int k, int start, int remain, int* path, int path_size, Result* res) {
    if (path_size == k) {
        if (remain == 0) {
            push_result(res, path, path_size);
        }
        return;
    }

    for (int x = start; x <= 9; ++x) {
        if (x > remain) {
            break;
        }
        path[path_size] = x;
        dfs(k, x + 1, remain - x, path, path_size + 1, res);
    }
}

int** combinationSum3(int k, int n, int* returnSize, int** returnColumnSizes) {
    Result res = {0};
    res.capacity = 16;
    res.rows = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int path[9];
    dfs(k, 1, n, path, 0, &res);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.rows;
}
```

### C++

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(k, 1, n, path, res);
        return res;
    }

private:
    void dfs(int k, int start, int remain,
             vector<int>& path, vector<vector<int>>& res) {
        if (static_cast<int>(path.size()) == k) {
            if (remain == 0) {
                res.push_back(path);
            }
            return;
        }

        for (int x = start; x <= 9; ++x) {
            if (x > remain) {
                break;
            }
            path.push_back(x);
            dfs(k, x + 1, remain - x, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func combinationSum3(k int, n int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int, int)
	dfs = func(start, remain int) {
		if len(path) == k {
			if remain == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}

		for x := start; x <= 9; x++ {
			if x > remain {
				break
			}
			path = append(path, x)
			dfs(x+1, remain-x)
			path = path[:len(path)-1]
		}
	}

	dfs(1, n)
	return res
}
```

### Rust

```rust
fn combination_sum_3(k: i32, n: i32) -> Vec<Vec<i32>> {
    fn dfs(
        k: usize,
        start: i32,
        remain: i32,
        path: &mut Vec<i32>,
        res: &mut Vec<Vec<i32>>,
    ) {
        if path.len() == k {
            if remain == 0 {
                res.push(path.clone());
            }
            return;
        }

        for x in start..=9 {
            if x > remain {
                break;
            }
            path.push(x);
            dfs(k, x + 1, remain - x, path, res);
            path.pop();
        }
    }

    let mut res = Vec::new();
    let mut path = Vec::new();
    dfs(k as usize, 1, n, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function combinationSum3(k, n) {
  const res = [];
  const path = [];

  function dfs(start, remain) {
    if (path.length === k) {
      if (remain === 0) res.push([...path]);
      return;
    }

    for (let x = start; x <= 9; x += 1) {
      if (x > remain) break;
      path.push(x);
      dfs(x + 1, remain - x);
      path.pop();
    }
  }

  dfs(1, n);
  return res;
}
```
