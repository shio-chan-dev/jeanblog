---
title: "Hot100: Subsets (Backtracking / startIndex ACERS Guide)"
date: 2026-04-02T13:48:57+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "subsets", "DFS", "startIndex", "LeetCode 78"]
description: "A practical guide to LeetCode 78 covering subset backtracking, the startIndex invariant, and runnable multi-language implementations."
keywords: ["Subsets", "backtracking", "startIndex", "power set", "LeetCode 78", "Hot100"]
---

> **Subtitle / Summary**
> Subsets is the cleanest entry point into Hot100 backtracking. The main thing to stabilize is not “enumerate everything”, but the three invariants behind the template: `path`, `startIndex`, and “every node is already a valid answer”.

- **Reading time**: 10-12 min
- **Tags**: `Hot100`, `backtracking`, `subsets`, `DFS`
- **SEO keywords**: Subsets, backtracking, startIndex, power set, LeetCode 78
- **Meta description**: Learn the stable backtracking template for LeetCode 78, with engineering analogies, pitfalls, and runnable multi-language solutions.

---

## Target Readers

- Hot100 learners starting the backtracking block today
- Developers who can write DFS but still mix up combinations and permutations
- Engineers who need to enumerate feature sets, candidate policies, or configuration bundles

## Background / Motivation

Many “real” problems reduce to a subset model:

- which feature flags should be enabled together
- which rules should be combined into one experiment
- which filters should be included in a saved preset

What makes LeetCode 78 valuable is that the problem is deliberately simple:

- all numbers are distinct
- there is no target sum
- there is no duplicate-removal complication

That simplicity lets you focus on the template itself before adding pruning, fixed lengths, or duplicate handling.

## Core Concepts

- **`path`**: the current chosen elements on the recursion path
- **`startIndex`**: the first candidate index allowed in the current layer
- **Preorder collection**: in the subsets problem, every node in the search tree is already one valid answer
- **Backtrack undo**: after recursion returns, remove the last chosen element

---

## A — Algorithm

### Problem Restatement

Given an integer array `nums` whose elements are all distinct, return all possible subsets (the power set).

The solution set must not contain duplicate subsets, and the answer order does not matter.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | array of distinct integers |
| return | `List[List[int]]` | all possible subsets |

### Example 1

```text
input: nums = [1,2,3]
output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

### Example 2

```text
input: nums = [0]
output: [[],[0]]
```

### Constraints

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- all elements of `nums` are distinct

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from a tiny example

Take `nums = [1,2,3]`.

Do not ask “how do I generate the whole power set?” yet.
Ask the smaller question:

- I have already chosen some elements
- where may I continue choosing from
- when is the current path already a valid answer by itself

That produces a tree like this:

```text
[]
|- [1]
|  |- [1,2]
|  |  |- [1,2,3]
|  |- [1,3]
|- [2]
|  |- [2,3]
|- [3]
```

This single example already shows the two key facts:

- every node is a valid subset
- once you choose `1`, later layers must not go back and build the same subset in a different order

#### Step 2: What must the partial answer remember?

If we are building one subset gradually, we need a state that stores the chosen elements so far.
That is why we need `path`.

```python
path = []
```

`path` means:

- the current branch of the search tree
- not the full answer set

#### Step 3: How do we avoid order-based duplicates?

Subsets are combination-style results, so `[1,2]` and `[2,1]` must not appear separately.
The stable way to prevent that is to restrict each layer to start from one boundary index.

```python
def dfs(start: int) -> None:
    ...
```

Here `start` means:

- the first index the current layer is allowed to use
- do not go back to earlier elements

#### Step 4: When do we collect one answer?

In this problem, the current path is already a valid subset as soon as it exists.
There is no target sum and no required fixed length.

So collection happens at the beginning of each DFS call:

```python
res.append(path.copy())
```

The `.copy()` matters because `path` will keep changing later.

#### Step 5: What choices are available next?

At the current layer, we only need to iterate from `start` to the end.

```python
for i in range(start, len(nums)):
    ...
```

That boundary is what makes the logic subset-style instead of permutation-style.

#### Step 6: What changes after choosing one element?

If we choose `nums[i]`, we add it to the current path and recurse on the suffix to the right.

```python
path.append(nums[i])
dfs(i + 1)
```

The `i + 1` is the critical move:

- the current element is already decided
- later layers must continue to the right only

#### Step 7: What must be undone?

After recursion returns, we restore the old state by removing the last chosen element.

```python
path.pop()
```

That lets the loop try the next candidate at the same depth.

#### Step 8: Walk one branch slowly

Still using `nums = [1,2,3]`:

Start:

- `path = []`
- `start = 0`

Enter `dfs(0)`:

- collect `[]`

Choose `1`:

- `path = [1]`
- call `dfs(1)`
- collect `[1]`

Inside that call, choose `2`:

- `path = [1,2]`
- call `dfs(2)`
- collect `[1,2]`

Then choose `3`:

- `path = [1,2,3]`
- call `dfs(3)`
- collect `[1,2,3]`

Then backtrack with `pop()`, return to `[1,2]`, and keep exploring the next branches.
The full solution is just this pattern repeated across the tree.

### Assemble the Full Code

Now combine the fragments into the first complete working implementation.

```python
from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(subsets([1, 2, 3]))
    print(subsets([0]))
```

### Reference Answer

For LeetCode submission style, the same logic becomes:

```python
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())
            for i in range(start, len(nums)):
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res
```

### What method did we just build?

Its formal name is:

- backtracking
- combination-style search
- `startIndex` boundary control

But the more important part is the invariant set:

- `path` means “what has already been chosen”
- `start` means “where the current layer may continue”
- every node is already a valid answer, so collection happens before deeper recursion

---

## E — Engineering

### Scenario 1: feature-flag bundle generation (Python)

**Background**: enumerate all possible flag bundles for offline experiment planning.
**Why it fits**: each flag is either included or not included, which is exactly a subset model.

```python
def all_flag_sets(flags):
    res = [[]]
    for flag in flags:
        res += [old + [flag] for old in res]
    return res


print(all_flag_sets(["new-ui", "cache-v2", "risk-guard"]))
```

### Scenario 2: policy-module candidate sets (Go)

**Background**: a backend risk system wants to test all combinations of several rule modules.
**Why it fits**: “pick any subset of modules” is the same combinational space.

```go
package main

import "fmt"

func subsets(items []string) [][]string {
	res := [][]string{{}}
	for _, item := range items {
		size := len(res)
		for i := 0; i < size; i++ {
			next := append([]string{}, res[i]...)
			next = append(next, item)
			res = append(res, next)
		}
	}
	return res
}

func main() {
	fmt.Println(subsets([]string{"ruleA", "ruleB", "ruleC"}))
}
```

### Scenario 3: saved filter preset generation (JavaScript)

**Background**: a frontend app wants to precompute filter presets for demos or regression coverage.
**Why it fits**: each filter can be enabled or disabled, so the full set of presets is a power set.

```javascript
function subsets(items) {
  const res = [[]];
  for (const item of items) {
    const size = res.length;
    for (let i = 0; i < size; i += 1) {
      res.push([...res[i], item]);
    }
  }
  return res;
}

console.log(subsets(["tag", "price", "stock"]));
```

---

## R — Reflection

### Complexity

- Time complexity: `O(n * 2^n)`
- Auxiliary recursion space: `O(n)`
- Output size: `O(n * 2^n)` in total

### Alternatives

| Method | Idea | Strength | Weakness |
| --- | --- | --- | --- |
| Backtracking | grow a path layer by layer | best template for later problems | requires a clear tree model |
| Bitmask | one bit means choose / skip | short and compact | less intuitive for future pruning problems |
| Iterative expansion | extend existing subsets by one new item | elegant for this one problem | less reusable when constraints become complex |

### Common mistakes

- collecting only at leaf nodes and missing most subsets
- appending `path` directly instead of a copy
- restarting each layer from index `0` and accidentally generating permutation-like duplicates

## Best Practices

- Treat this as the base template for “combination-style” backtracking
- Always make a snapshot when storing `path`
- Ask yourself four questions while coding:
  `What does path mean?`
  `Why collect here?`
  `Where does this layer start?`
  `What exactly is undone on return?`

---

## S — Summary

- LeetCode 78 is the cleanest problem for building the backtracking skeleton
- `startIndex` is what makes this combinations/subsets logic, not permutations logic
- In subsets, every node is a valid answer, so collection happens before deeper recursion
- Once this template is stable, problems like permutations, combination sum, and subsets with duplicates become much easier

### Suggested Next Problems

- `46. Permutations`: add `used[]` and learn permutation-style backtracking
- `17. Letter Combinations of a Phone Number`: fixed-depth DFS
- `39. Combination Sum`: add pruning and repeated use of the same candidate
- `90. Subsets II`: handle duplicates cleanly

### CTA

If this is your first backtracking problem today, write it once from memory after reading.
That is the fastest way to make the template stick.

---

## Multi-Language Implementations

### Python

```python
from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static void push_result(Result* res, int* path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    int* row = malloc(sizeof(int) * path_size);
    for (int i = 0; i < path_size; ++i) row[i] = path[i];
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(int* nums, int nums_size, int start, int* path, int path_size, Result* res) {
    push_result(res, path, path_size);
    for (int i = start; i < nums_size; ++i) {
        path[path_size] = nums[i];
        dfs(nums, nums_size, i + 1, path, path_size + 1, res);
    }
}

int** subsets(int* nums, int nums_size, int* return_size, int** return_column_sizes) {
    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * nums_size);
    dfs(nums, nums_size, 0, path, 0, &res);
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
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(nums, 0, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& nums, int start, vector<int>& path, vector<vector<int>>& res) {
        res.push_back(path);
        for (int i = start; i < (int)nums.size(); ++i) {
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

func subsets(nums []int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]int(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(nums); i++ {
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
fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    fn dfs(nums: &[i32], start: usize, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        res.push(path.clone());
        for i in start..nums.len() {
            path.push(nums[i]);
            dfs(nums, i + 1, path, res);
            path.pop();
        }
    }

    let mut res = Vec::new();
    let mut path = Vec::new();
    dfs(&nums, 0, &mut path, &mut res);
    res
}
```

### JavaScript

```javascript
function subsets(nums) {
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < nums.length; i += 1) {
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}
```
