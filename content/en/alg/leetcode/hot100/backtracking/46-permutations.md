---
title: "Hot100: Permutations (used[] Backtracking ACERS Guide)"
date: 2026-04-03T17:54:22+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "permutations", "DFS", "used array", "LeetCode 46"]
description: "A practical guide to LeetCode 46 covering used[] state tracking, leaf-only collection, and runnable multi-language implementations."
keywords: ["Permutations", "backtracking", "used array", "DFS", "LeetCode 46", "Hot100"]
---

> **Subtitle / Summary**
> If Subsets teaches the skeleton of combination-style backtracking, Permutations teaches the core of state-based backtracking: at each position, choose one unused element, continue until the path length reaches `n`, and only then collect the answer.

- **Reading time**: 10-12 min
- **Tags**: `Hot100`, `backtracking`, `permutations`, `DFS`
- **SEO keywords**: Permutations, backtracking, used array, DFS, LeetCode 46
- **Meta description**: Learn the stable permutation backtracking template for LeetCode 46, with state recovery, engineering analogies, and runnable multi-language solutions.

---

## A — Algorithm

### Problem Restatement

Given an array `nums` of distinct integers, return all possible permutations.
The answer may be returned in any order.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | array of distinct integers |
| return | `List[List[int]]` | all possible permutations |

### Example 1

```text
input: nums = [1,2,3]
output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

### Example 2

```text
input: nums = [0,1]
output: [[0,1],[1,0]]
```

### Example 3

```text
input: nums = [1]
output: [[1]]
```

### Constraints

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- all integers in `nums` are distinct

---

## Target Readers

- Hot100 learners who have already finished `78. Subsets` and want the next backtracking template
- Developers who understand recursion but still make mistakes when restoring state
- Engineers who need to enumerate execution orders, test sequences, or ordering-sensitive plans

## Background / Motivation

The key difference between combinations and permutations is simple:

- combinations care about which elements are chosen
- permutations also care about the order of those elements

So in this problem, `[1,2,3]` and `[1,3,2]` are different valid answers.
That immediately changes the template:

- `startIndex` is no longer enough
- every layer must be able to consider all positions again
- we need explicit state to record which elements are already used

That is exactly why LeetCode 46 is a foundational backtracking problem.
It forces you to reason clearly about state selection and state recovery.

## Core Concepts

- **`path`**: the permutation currently being built
- **`used[i]`**: whether `nums[i]` has already been placed in the current path
- **Leaf-only collection**: only when `path.length == nums.length` do we have a full permutation
- **State recovery**: on return, both `path` and `used[i]` must be restored

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from a tiny example

Take `nums = [1, 2, 3]`.

Instead of asking "how do I generate all permutations?", ask the smaller question:

> How do I build one permutation one slot at a time?

That shift makes the recursion much easier to design.

#### Step 2: Decide what state a partial answer needs

While building one permutation, we need three pieces of state:

- `path`: the numbers already chosen
- `used[i]`: whether `nums[i]` is already inside `path`
- `res`: where to store finished permutations

So the setup is:

```python
res = []
path = []
used = [False] * len(nums)
```

At the beginning:

- `path = []`
- every element in `used` is `False`

That means we have not chosen anything yet.

#### Step 3: Define the recursive subproblem

The recursive question is:

> Given the current partial permutation in `path`, what unused number can I place next?

That becomes a DFS function:

```python
def dfs() -> None:
    ...
```

It can directly read and modify `res`, `path`, and `used`.

#### Step 4: Define the stopping condition

When is one permutation complete?

When `path` has the same length as `nums`.

```python
if len(path) == len(nums):
    res.append(path.copy())
    return
```

Two details matter here:

- we collect only at leaf nodes, because only then is the permutation complete
- we must append `path.copy()` instead of `path`, because `path` will keep changing during backtracking

#### Step 5: List the available choices

At any point, the next number can be any element that is not already used.

```python
for i, x in enumerate(nums):
    if used[i]:
        continue
```

So every DFS level scans the full array, but only unused numbers are eligible.

#### Step 6: Make one choice

If `x` is unused, choose it by updating the state:

```python
used[i] = True
path.append(x)
```

Now the partial permutation is one element longer.

#### Step 7: Recurse on the smaller problem

After choosing the next number, solve the rest of the problem:

> Fill the remaining positions.

```python
dfs()
```

#### Step 8: Undo the choice

After exploring every permutation that starts with that choice, restore the old state so the loop can try the next option.

```python
path.pop()
used[i] = False
```

This is the key backtracking pattern:

```text
choose
recurse
undo
```

#### Step 9: Walk one branch slowly

For `nums = [1, 2, 3]`:

Start:

- `path = []`
- `used = [False, False, False]`

Choose `1`:

- `path = [1]`
- `used = [True, False, False]`

Choose `2`:

- `path = [1, 2]`
- `used = [True, True, False]`

Choose `3`:

- `path = [1, 2, 3]`
- `used = [True, True, True]`

Now `len(path) == len(nums)`, so save `[1, 2, 3]`.

Then backtrack:

- remove `3`
- mark `3` unused

Return to `path = [1, 2]` and try other choices.
There are none, so backtrack again:

- remove `2`
- mark `2` unused

Now we are back at `path = [1]`, and can try `3` instead.
That produces `[1, 3, 2]`.

The full search keeps repeating that same pattern until every branch has been explored.

### Assemble the Full Code

Now combine the fragments above into the first complete working solution.

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

### Reference Answer

For LeetCode submission style, the same logic becomes:

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

### What mental model should stick?

If you want to derive this from scratch during an interview or while coding, ask:

1. Can I build the answer one position at a time?
2. What does the partial answer look like?
3. When is the partial answer complete?
4. What choices are available next?
5. What state must I undo before trying the next choice?

For permutations, the answers are:

- yes, fill one slot at a time
- the partial answer is `path`
- it is complete when `len(path) == len(nums)`
- the next choice is any unused number
- after recursion, undo with both `path.pop()` and `used[i] = False`

That is the full model.
The code is only a direct translation of that reasoning.

---

## E — Engineering

### Scenario 1: task execution order enumeration (Python)

**Background**: an offline scheduler wants to compare how different task orders affect the final result.
**Why it fits**: when order changes behavior, the search space is permutation-shaped.

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

### Scenario 2: API regression order testing (Go)

**Background**: the same set of API calls may trigger different cache or state paths when called in different orders.
**Why it fits**: validating order sensitivity is directly a permutation problem.

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

### Scenario 3: animation order exploration (JavaScript)

**Background**: during UI prototyping, a team wants to try several orders of animation steps.
**Why it fits**: different step orders produce different user experiences.

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

## R — Reflection

### Complexity

- Time complexity: `O(n * n!)`
- Auxiliary recursion space: `O(n)`
- If output is counted, total space is dominated by the `n!` answer set

### Comparison with Subsets

| Problem | Nature | When to collect | Key state |
| --- | --- | --- | --- |
| `78. Subsets` | combinations | every node | `startIndex` |
| `46. Permutations` | permutations | leaf nodes only | `used[]` |

### Common mistakes

- forgetting to restore `used[i]`
- collecting answers at DFS entry and accidentally storing incomplete prefixes
- trying to solve permutations with `startIndex` and missing order variants

## Best Practices

- Think of this as “fill the next position” rather than “pick the next number”
- Restore `path` and `used` as a pair
- Draw a 3-level search tree before coding if the recursion feels abstract
- Keep the distinction between combination templates and permutation templates explicit in your notes

---

## S — Summary

- The real lesson of LeetCode 46 is state control through `used[]`
- Permutations collect answers only at leaf nodes because only leaves represent full results
- Compared with Subsets, this problem is less about boundaries and more about state recovery
- Once this template is stable, many order-sensitive search problems become much easier

### Suggested Next Problems

- `78. Subsets`: combination-style backtracking template
- `17. Letter Combinations of a Phone Number`: fixed-depth DFS
- `47. Permutations II`: permutations with duplicate handling
- `51. N-Queens`: state-heavy constrained search

### CTA

After reading this, try rewriting the template once from memory and explain out loud why `used[]` is necessary.
That one habit will make the distinction between combinations and permutations much harder to forget.

---

## Multi-Language Implementations

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
