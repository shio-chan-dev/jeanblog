---
title: "Hot100: Subsets II (Sorting / Layer-Level Dedup ACERS Guide)"
date: 2026-04-19T14:49:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "subsets", "deduplication", "sorting", "LeetCode 90"]
description: "A practical guide to LeetCode 90 that builds duplicate-aware subset backtracking from scratch and explains why sorting plus layer-level dedup is the stable solution."
keywords: ["Subsets II", "backtracking", "sorting", "layer-level dedup", "LeetCode 90", "Hot100"]
---

> **Subtitle / Summary**
> `90. Subsets II` is the natural upgrade after `78. Subsets`. The new difficulty is not backtracking itself, but how to skip duplicate branches without deleting valid answers such as `[2,2]`.

- **Reading time**: 12-15 min
- **Tags**: `Hot100`, `backtracking`, `subsets`, `deduplication`, `sorting`
- **SEO keywords**: Subsets II, backtracking, sorting, layer-level dedup, LeetCode 90
- **Meta description**: Learn LeetCode 90 by building the duplicate-aware subsets template from scratch, using sorting and one stable layer-level dedup rule.

---

## A — Algorithm

### Problem Restatement

Given an integer array `nums` that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets, and the answer order does not matter.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | array of integers that may contain duplicates |
| return | `List[List[int]]` | all unique subsets |

### Example 1

```text
input: nums = [1,2,2]
output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```

### Example 2

```text
input: nums = [0]
output: [[],[0]]
```

### Constraints

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`

---

## Target Readers

- Hot100 learners who already understand `78. Subsets` but not duplicate handling yet
- Developers who can write basic DFS but still reach for a `set` as soon as duplicates appear
- Readers who want one stable template for “combination-style search with duplicate values”

## Background / Motivation

In `78. Subsets`, all values are distinct, so the search tree is clean.
In `90. Subsets II`, duplicate values create duplicate branches immediately.

For `nums = [1,2,2]`, the first layer can produce:

- one branch starting from the first `2`
- another branch starting from the second `2`

Both create the same subset `[2]`.

That is the real problem here:

- subsets are still combination-style answers
- but equal values in different positions can generate the same branch meaning

The stable reasoning chain is:

- this is still a `startIndex` combination template
- sort first so equal values become adjacent
- dedup is not “ban equal values globally”
- dedup is “within the same layer, only expand the first branch for that value”

## Core Concepts

- **`path`**: the chosen elements on the current recursion branch
- **`startIndex`**: the first candidate index allowed in the current layer
- **Sorting**: makes equal values adjacent so one dedup rule becomes reliable
- **Layer-level dedup**: in one layer, skip later equal values after the first branch is already expanded
- **Preorder collection**: in subsets problems, every node is already a valid answer

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from the smallest example that exposes duplication

Take `nums = [1,2,2]`.

If you reuse the `78. Subsets` code with no changes, two different branches collide:

- first layer chooses the first `2`, producing `[2]`
- first layer skips that `2` and chooses the second `2`, also producing `[2]`

So the issue is not “how do I generate subsets?”.
The issue is “which branches mean the same value choice in the same layer?”

#### Step 2: What must the partial answer remember?

Just like in `78. Subsets`, we still need one state for the current chosen elements.
That is `path`.

```python
path = []
```

`path` means the current DFS branch, not the full answer set.

#### Step 3: Why must we sort first?

Without sorting, equal values may be far apart, and the dedup rule becomes unstable.
After sorting, equal values become adjacent, so “same value as the previous candidate” is meaningful.

```python
nums.sort()
```

For example, `[2,1,2]` becomes `[1,2,2]`.
Now the duplicate pair is visible to the DFS layer.

#### Step 4: Why is this still a `startIndex` problem?

Even with duplicates, the problem is still subset generation, which is combination-style.
Order must not create separate answers.

So we still define the recursion by a suffix boundary:

```python
def dfs(start: int) -> None:
    ...
```

That means:

- only continue from `start` to the right
- do not go back to earlier indices

#### Step 5: When do we collect one answer?

Exactly like `78. Subsets`, the current path is already a valid subset.

So collection still happens at the beginning of each DFS call:

```python
res.append(path.copy())
```

This remains a subsets problem, not a “leaf-only” problem.

#### Step 6: What choices are available next?

At the current layer, enumerate candidates from `start` to the end:

```python
for i in range(start, len(nums)):
    ...
```

Up to this point, the code still looks almost identical to `78. Subsets`.
The upgrade comes from one extra rule.

#### Step 7: How do we skip duplicate branches without deleting valid answers?

The stable condition is:

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

This does **not** mean “equal values are forbidden forever”.
It means:

- if the current value equals the previous one
- and both are being considered in the same DFS layer
- then the previous equal value has already opened the only necessary branch for this layer

So we skip the later duplicate branch.

The `i > start` part is the key.
It makes the rule layer-level, not global.

#### Step 8: What changes after we choose one value?

Once the current value passes the dedup check, the rest is standard subset backtracking:

```python
path.append(nums[i])
dfs(i + 1)
path.pop()
```

We still recurse with `i + 1` because each position can be used at most once.

#### Step 9: Walk one branch slowly

Still using sorted `nums = [1,2,2]`:

Start:

- `path = []`
- `start = 0`

Enter `dfs(0)`:

- collect `[]`

Choose `1`:

- `path = [1]`
- call `dfs(1)`
- collect `[1]`

Inside that layer, choose the first `2`:

- `path = [1,2]`
- call `dfs(2)`
- collect `[1,2]`

Then choose the second `2`:

- `path = [1,2,2]`
- call `dfs(3)`
- collect `[1,2,2]`

Now go back to the top layer and inspect the second `2` there:

- `i = 2`
- `start = 0`
- `nums[2] == nums[1]`

That means “open a first-layer branch with value `2`” has already been done.
So this branch is skipped.

#### Step 10: Reduce the whole idea to one reusable sentence

The reusable template is:

> For combination-style search with duplicates, sort first, and in each layer skip later equal values with `if i > start and nums[i] == nums[i - 1]`.

### Assemble the Full Code

Now combine those fragments into the first full runnable version.

```python
from typing import List


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(subsets_with_dup([1, 2, 2]))
    print(subsets_with_dup([0]))
```

### Reference Answer

If you want the LeetCode submission form, the same logic becomes:

```python
from typing import List


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res: List[List[int]] = []
        path: List[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res
```

### What method did we just build?

Formally, this is:

- backtracking
- combination-style search
- sorting plus layer-level deduplication

But the more important invariants are:

- `path` stores the current chosen elements
- `startIndex` keeps the problem combination-style
- subsets are still collected at every node
- dedup must only apply within the current layer

---

## E — Engineering

### Scenario 1: Deduplicated filter presets (Python)

**Background**: a product backend merges filter values from multiple upstreams and may receive duplicate labels.  
**Why this fits**: you still want all preset combinations, but duplicated source values must not create duplicated presets.

```python
from typing import List


def unique_filter_sets(tags: List[str]) -> List[List[str]]:
    tags.sort()
    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(tags)):
            if i > start and tags[i] == tags[i - 1]:
                continue
            path.append(tags[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


print(unique_filter_sets(["red", "red", "xl"]))
```

### Scenario 2: Role bundle generation with duplicate inputs (Go)

**Background**: a permission system aggregates role labels from multiple services, and duplicates can appear after the merge.  
**Why this fits**: generating all candidate role bundles is exactly “subsets with duplicates removed at the source-value level”.

```go
package main

import (
	"fmt"
	"sort"
)

func bundles(tags []string) [][]string {
	sort.Strings(tags)
	res := make([][]string, 0)
	path := make([]string, 0, len(tags))

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]string(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(tags); i++ {
			if i > start && tags[i] == tags[i-1] {
				continue
			}
			path = append(path, tags[i])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}

func main() {
	fmt.Println(bundles([]string{"read", "read", "write"}))
}
```

### Scenario 3: Frontend preset generation with duplicate options (JavaScript)

**Background**: a UI testing tool receives duplicated option values from config data and wants every semantic multi-select preset once.  
**Why this fits**: the data model is still subset generation, but duplicate branches must be suppressed.

```javascript
function uniquePresets(items) {
  const sorted = [...items].sort();
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < sorted.length; i += 1) {
      if (i > start && sorted[i] === sorted[i - 1]) continue;
      path.push(sorted[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}

console.log(uniquePresets(["tag", "tag", "price"]));
```

---

## R — Reflection

### Complexity Analysis

- Time complexity: worst-case `O(n * 2^n)`
  - dedup reduces repeated branches in practice
  - but the upper bound is still driven by subset enumeration
- Space complexity: `O(n)` recursion stack
  - if output is counted, total space also depends on the full answer size

### Comparison with other approaches

| Method | Idea | Advantage | Limitation |
| --- | --- | --- | --- |
| Sort + layer-level dedup | skip repeated branches during DFS | stable and transferable | requires real understanding of “same layer” |
| Generate all subsets then use a `set` | brute-force first, dedup later | easy to think of | wasteful and not reusable |
| Bitmask + dedup set | enumerate by mask, dedup afterward | compact code | poor teaching value for later backtracking problems |

### Common Mistakes

- writing the condition as `i > 0`, which wrongly kills valid deeper branches
- forgetting to sort before dedup
- thinking duplicate values can never both appear, which incorrectly removes valid answers like `[2,2]`

## Common Questions and Pitfalls

### Why must it be `i > start` instead of `i > 0`?

Because the goal is to skip duplicate **branches in the current layer**.

If you write `i > 0`, then deeper recursive calls also skip equal values incorrectly, and valid answers like `[2,2]` disappear.

### Is the difference from `78. Subsets` really just one line?

From the template point of view, almost yes:

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

But that line only works because two ideas are already correct:

- the array is sorted first
- dedup is understood as layer-level, not global

### When should I think of this pattern?

When the input:

- may contain duplicates
- and the output must contain unique combinations or unique subsets

then you should immediately ask:

- do I need to sort first?
- is the dedup rule layer-level or branch-level?

## Best Practices and Recommendations

- solidify `78. Subsets` before learning this upgrade
- diagnose duplicate branches by drawing a tiny collision example first
- when values are compared to neighbors, sorting is usually the first move
- prefer one correct DFS dedup rule over “brute force + set cleanup”

---

## S — Summary

- `90. Subsets II` is still a combination backtracking problem, not a brand-new category
- the real upgrade is sorting plus layer-level deduplication
- the condition `i > start and nums[i] == nums[i - 1]` works because it skips only same-layer duplicate branches
- subsets are still collected at every node
- once this pattern is stable, `40. Combination Sum II` and `47. Permutations II` become much easier

### Recommended Follow-Up Reading

- `78. Subsets`: the distinct-values base template
- `39. Combination Sum`: combination-style backtracking with a target
- `40. Combination Sum II`: duplicates plus target plus dedup
- `47. Permutations II`: duplicate handling in permutation-style search

### Action Step

Write `78` and `90` side by side once.
If you can explain exactly why sorting is required and why the dedup rule is layer-level rather than global, this part of backtracking is already stable.

---

## Multi-Language Implementations

### Python

```python
from typing import List


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res: List[List[int]] = []
    path: List[int] = []

    def dfs(start: int) -> None:
        res.append(path.copy())
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdlib.h>

typedef struct {
    int** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static int cmp_int(const void* a, const void* b) {
    return *(const int*)a - *(const int*)b;
}

static void push_result(Result* res, int* path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(int*) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    int* row = malloc(sizeof(int) * path_size);
    for (int i = 0; i < path_size; ++i) {
        row[i] = path[i];
    }
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(int* nums, int n, int start, int* path, int path_size, Result* res) {
    push_result(res, path, path_size);
    for (int i = start; i < n; ++i) {
        if (i > start && nums[i] == nums[i - 1]) {
            continue;
        }
        path[path_size] = nums[i];
        dfs(nums, n, i + 1, path, path_size + 1, res);
    }
}

int** subsetsWithDup(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    qsort(nums, numsSize, sizeof(int), cmp_int);

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(int*) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    int* path = malloc(sizeof(int) * numsSize);
    dfs(nums, numsSize, 0, path, 0, &res);
    free(path);

    *returnSize = res.size;
    *returnColumnSizes = res.col_sizes;
    return res.data;
}
```

### C++

```cpp
#include <algorithm>
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        vector<int> path;
        dfs(nums, 0, path, res);
        return res;
    }

private:
    void dfs(const vector<int>& nums, int start, vector<int>& path, vector<vector<int>>& res) {
        res.push_back(path);
        for (int i = start; i < (int)nums.size(); ++i) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
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

import "sort"

func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	res := make([][]int, 0)
	path := make([]int, 0, len(nums))

	var dfs func(int)
	dfs = func(start int) {
		snapshot := append([]int(nil), path...)
		res = append(res, snapshot)
		for i := start; i < len(nums); i++ {
			if i > start && nums[i] == nums[i-1] {
				continue
			}
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
impl Solution {
    pub fn subsets_with_dup(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort();
        let mut res: Vec<Vec<i32>> = Vec::new();
        let mut path: Vec<i32> = Vec::new();

        fn dfs(nums: &Vec<i32>, start: usize, path: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
            res.push(path.clone());
            for i in start..nums.len() {
                if i > start && nums[i] == nums[i - 1] {
                    continue;
                }
                path.push(nums[i]);
                dfs(nums, i + 1, path, res);
                path.pop();
            }
        }

        dfs(&nums, 0, &mut path, &mut res);
        res
    }
}
```

### JavaScript

```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsetsWithDup = function (nums) {
  nums.sort((a, b) => a - b);
  const res = [];
  const path = [];

  function dfs(start) {
    res.push([...path]);
    for (let i = start; i < nums.length; i += 1) {
      if (i > start && nums[i] === nums[i - 1]) continue;
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
};
```
