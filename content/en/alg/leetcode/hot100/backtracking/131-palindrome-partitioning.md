---
title: "Hot100: Palindrome Partitioning (Backtracking / Palindrome Table ACERS Guide)"
date: 2026-04-19T14:49:56+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "backtracking", "string", "palindrome", "DP", "LeetCode 131"]
description: "A practical guide to LeetCode 131 that builds string-partition backtracking from scratch and shows why a palindrome table makes the search cleaner and more reusable."
keywords: ["Palindrome Partitioning", "backtracking", "palindrome DP", "string partition", "LeetCode 131", "Hot100"]
---

> **Subtitle / Summary**
> `131. Palindrome Partitioning` is not hard because of recursion alone. The real challenge is separating two concerns clearly: where the next cut starts, and whether the current substring is a valid palindrome.

- **Reading time**: 15-18 min
- **Tags**: `Hot100`, `backtracking`, `string`, `palindrome`, `DP`
- **SEO keywords**: Palindrome Partitioning, backtracking, palindrome DP, string partition, LeetCode 131
- **Meta description**: Learn LeetCode 131 by building the solution from scratch, using suffix-based DFS plus a precomputed palindrome table.

---

## A — Algorithm

### Problem Restatement

Given a string `s`, partition it so that every substring in the partition is a palindrome.
Return all possible palindrome partitionings of `s`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| s | string | lowercase English string |
| return | `List[List[str]]` | all valid palindrome partitions |

### Example 1

```text
input: s = "aab"
output: [["a","a","b"],["aa","b"]]
```

### Example 2

```text
input: s = "a"
output: [["a"]]
```

### Constraints

- `1 <= s.length <= 16`
- `s` consists of lowercase English letters only

---

## Target Readers

- learners who already know array-based backtracking and now want string partition search
- developers who mix up “where the cut starts” and “whether this substring is legal”
- readers who want a reusable model for “precompute valid intervals, then DFS the suffix”

## Background / Motivation

This problem is a very good training ground for a common search pattern:

> first decide which intervals are valid pieces, then recursively partition the whole string using only valid pieces

If you look at it only as a “palindrome problem”, you may focus too much on `is_palindrome`.
But structurally, it is a string partition problem:

- `start` tells us where the remaining suffix begins
- `path` stores the pieces already chosen
- `[start, end]` is the next candidate piece
- only valid pieces may be appended before recursing

That model transfers well beyond palindromes, including dictionary segmentation and rule-based text decomposition.

## Core Concepts

- **`start`**: the first index of the suffix not yet partitioned
- **`path`**: the partition pieces already chosen
- **Palindrome interval**: whether `s[l:r+1]` is a palindrome
- **Interval precomputation**: use `pal[l][r]` to cache legal substrings
- **Partition-style DFS**: choose one valid piece, then recurse on the suffix behind it

---

## C — Concepts

### How To Build The Solution From Scratch

#### Step 1: Start from a tiny example

Take `s = "aab"`.

The valid answers are:

- `["a", "a", "b"]`
- `["aa", "b"]`

The key observation is not just what the answers are.
It is the structure:

- at index `start`, choose one ending position `end`
- if `s[start:end+1]` is valid, recurse on the suffix after `end`

#### Step 2: What must the partial answer remember?

Because we are building one partition gradually, we need one state for the pieces already chosen.
That is `path`.

```python
path = []
```

It represents the current DFS branch, not the final full answer set.

#### Step 3: How should we name the remaining subproblem?

Once `path` is fixed, the only question left is:

> how do we partition the suffix starting at index `start`?

So the recursion signature becomes:

```python
def dfs(start: int) -> None:
    ...
```

This is similar in spirit to `startIndex` in array backtracking, but now the meaning is “where the remaining suffix begins”.

#### Step 4: When is one partition complete?

When `start == len(s)`, the whole string has already been split into valid pieces.

```python
if start == len(s):
    res.append(path.copy())
    return
```

That is the leaf condition.

#### Step 5: What choices are available next?

At the current `start`, every `end` from `start` to `n - 1` is a candidate:

```python
for end in range(start, len(s)):
    ...
```

Each candidate means “take substring `s[start:end+1]` as the next piece”.

#### Step 6: How do we check whether a candidate piece is a palindrome?

The most direct helper is a two-pointer check:

```python
def is_pal(left: int, right: int) -> bool:
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

Then DFS could test each candidate before recursing.

#### Step 7: Why is repeated palindrome checking the weak point?

If DFS checks palindromes on demand, the same substring may be tested many times.
That is acceptable for this constraint size, but not the cleanest reusable model.

The cleaner split of responsibilities is:

- precompute which intervals are valid palindromes
- let DFS focus only on partition enumeration

#### Step 8: How do we precompute all palindrome intervals?

Define `pal[i][j]` as whether `s[i:j+1]` is a palindrome.

The transition is:

- if `s[i] != s[j]`, then the interval is not a palindrome
- otherwise it is valid when the interval is short enough or the inner interval is already a palindrome

```python
for i in range(n - 1, -1, -1):
    for j in range(i, n):
        if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
            pal[i][j] = True
```

We iterate `i` backward so `pal[i + 1][j - 1]` is ready before we need it.

#### Step 9: What does DFS look like after the table is ready?

Now the DFS only asks whether the interval is legal:

```python
if pal[start][end]:
    path.append(s[start:end + 1])
    dfs(end + 1)
    path.pop()
```

At that point the problem has been cleanly separated into:

- `pal`: interval legality
- `dfs`: partition enumeration

#### Step 10: Walk one branch slowly

Still using `s = "aab"`:

Start:

- `start = 0`
- `path = []`

Try `end = 0`:

- substring `"a"` is a palindrome
- `path = ["a"]`
- recurse on `dfs(1)`

Inside `dfs(1)`:

- `end = 1`, substring `"a"` is valid
- `path = ["a", "a"]`
- recurse on `dfs(2)`

Inside `dfs(2)`:

- `end = 2`, substring `"b"` is valid
- `path = ["a", "a", "b"]`
- recurse on `dfs(3)`

Now `start == len(s)`, so collect one answer.

Back at the top level, `end = 1` gives `"aa"`, which also leads to a valid solution `["aa", "b"]`.

### Assemble the Full Code

Now combine palindrome precomputation and DFS into one runnable implementation.

```python
from typing import List


def partition_palindrome(s: str) -> List[List[str]]:
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                pal[i][j] = True

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not pal[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res


if __name__ == "__main__":
    print(partition_palindrome("aab"))
    print(partition_palindrome("a"))
```

### Reference Answer

For the LeetCode submission form:

```python
from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                    pal[i][j] = True

        res: List[List[str]] = []
        path: List[str] = []

        def dfs(start: int) -> None:
            if start == n:
                res.append(path.copy())
                return
            for end in range(start, n):
                if not pal[start][end]:
                    continue
                path.append(s[start : end + 1])
                dfs(end + 1)
                path.pop()

        dfs(0)
        return res
```

### What method did we just build?

Formally, it is:

- backtracking
- string partition search
- interval DP precomputation

But the more reusable model is:

- `dfs(start)` means “partition the remaining suffix”
- `pal[start][end]` answers whether the next piece is legal
- DFS chooses one legal piece at a time and recurses on the rest

---

## E — Engineering

### Scenario 1: Dictionary-constrained segmentation (Python)

**Background**: a search or NLP prototype wants to split a string into all valid token sequences from a small vocabulary.  
**Why this fits**: the structure is the same as this problem, except “palindrome” is replaced by “present in a dictionary”.

```python
from typing import List, Set


def all_segmentations(s: str, vocab: Set[str]) -> List[List[str]]:
    n = len(s)
    valid = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            valid[i][j] = s[i : j + 1] in vocab

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not valid[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res


print(all_segmentations("applepen", {"apple", "app", "le", "pen"}))
```

### Scenario 2: Rule-based segment planner (Go)

**Background**: a parser or log-cleaning pipeline first decides which intervals satisfy a rule, then enumerates all valid full decompositions.  
**Why this fits**: the reusable pattern is “precompute legal intervals, then DFS the suffix”.

```go
package main

import "fmt"

func partitions(s string, valid map[string]bool) [][]string {
	n := len(s)
	ok := make([][]bool, n)
	for i := 0; i < n; i++ {
		ok[i] = make([]bool, n)
		for j := i; j < n; j++ {
			ok[i][j] = valid[s[i:j+1]]
		}
	}

	res := make([][]string, 0)
	path := make([]string, 0, n)

	var dfs func(int)
	dfs = func(start int) {
		if start == n {
			snapshot := append([]string(nil), path...)
			res = append(res, snapshot)
			return
		}
		for end := start; end < n; end++ {
			if !ok[start][end] {
				continue
			}
			path = append(path, s[start:end+1])
			dfs(end + 1)
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}

func main() {
	fmt.Println(partitions("abc", map[string]bool{"a": true, "ab": true, "bc": true, "c": true}))
}
```

### Scenario 3: Frontend interactive text segmentation (JavaScript)

**Background**: a text puzzle or input experiment wants to show all valid ways to split the current input.  
**Why this fits**: as long as validity can be cached per interval, the DFS layer is the same.

```javascript
function segmentations(s, allow) {
  const n = s.length;
  const ok = Array.from({ length: n }, () => Array(n).fill(false));
  for (let i = 0; i < n; i += 1) {
    for (let j = i; j < n; j += 1) {
      ok[i][j] = allow.has(s.slice(i, j + 1));
    }
  }

  const res = [];
  const path = [];

  function dfs(start) {
    if (start === n) {
      res.push([...path]);
      return;
    }
    for (let end = start; end < n; end += 1) {
      if (!ok[start][end]) continue;
      path.push(s.slice(start, end + 1));
      dfs(end + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
}

console.log(segmentations("level", new Set(["l", "e", "v", "eve", "level"])));
```

---

## R — Reflection

### Complexity Analysis

- Palindrome-table preprocessing: `O(n^2)`
- DFS enumeration: worst-case about `O(n * 2^(n-1))`
  - many cut positions may be legal
  - copying paths and slicing strings adds linear costs along the way
- Auxiliary space: `O(n^2 + n)`
  - `O(n^2)` for the palindrome table
  - `O(n)` for recursion depth and the current path
  - plus output space if all returned answers are counted

### Comparison with other approaches

| Method | Idea | Advantage | Limitation |
| --- | --- | --- | --- |
| DFS + palindrome table | precompute legal intervals, then partition | clean responsibility split | needs an `O(n^2)` table |
| DFS + on-demand palindrome check | validate each candidate with two pointers | short to start with | repeats interval checks |
| DP only | compute feasibility or min cuts | good for decision problems | does not enumerate all partitions |

### Common Mistakes

- treating `start` as a length instead of a suffix index
- recursing before verifying whether the current substring is valid
- filling the palindrome table in the wrong order
- forgetting to `pop()` after appending a chosen substring

## Common Questions and Pitfalls

### Why is a palindrome table worth building first?

Because DFS revisits overlapping intervals.
Once `pal[i][j]` is available, each legality check becomes `O(1)` and the DFS stays focused on partitioning.

### Can this problem be solved without DP?

Yes.

You can do DFS and check each substring with two pointers on demand.
That version is valid, but the table-based version is a cleaner template for future interval-partition problems.

### How is this related to `139. Word Break`?

They share the same structural idea:

- choose one valid substring from the current start
- recurse or transition on the suffix behind it

The difference is:

- `131` asks for all full partitions
- `139` often asks only whether at least one valid decomposition exists

## Best Practices and Recommendations

- whenever you see “string partition + local validity”, ask whether legal intervals can be cached first
- define `dfs(start)` clearly before choosing any optimization
- separate “is this piece legal?” from “how do I continue after choosing it?”
- draw the recursion on `"aab"` once; it reveals the template much better than memorizing code

---

## S — Summary

- `131. Palindrome Partitioning` is fundamentally a string-partition backtracking problem
- `start` means the suffix boundary, and `path` stores the pieces already chosen
- once legality is precomputed as `pal[i][j]`, DFS can focus entirely on enumeration
- the high-value transferable idea is “precompute valid intervals, then DFS the suffix”
- this pattern applies to many partition and segmentation problems beyond palindromes

### Recommended Follow-Up Reading

- `132. Palindrome Partitioning II`: optimize from “enumerate all” to “min cuts”
- `139. Word Break`: interval-validity partitioning for reachability
- `78. Subsets`: the base backtracking tree model
- `17. Letter Combinations of a Phone Number`: fixed-depth string DFS

### Action Step

After finishing this problem, revisit it as a partition template instead of only a palindrome question.
Once you can explain the roles of `pal` and `dfs(start)` separately, the structure becomes much more reusable.

---

## Multi-Language Implementations

### Python

```python
from typing import List


def partition_palindrome(s: str) -> List[List[str]]:
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                pal[i][j] = True

    res: List[List[str]] = []
    path: List[str] = []

    def dfs(start: int) -> None:
        if start == n:
            res.append(path.copy())
            return
        for end in range(start, n):
            if not pal[start][end]:
                continue
            path.append(s[start : end + 1])
            dfs(end + 1)
            path.pop()

    dfs(0)
    return res
```

### C

```c
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char*** data;
    int* col_sizes;
    int size;
    int capacity;
} Result;

static char* clone_str(const char* s) {
    size_t len = strlen(s);
    char* copy = malloc(len + 1);
    memcpy(copy, s, len + 1);
    return copy;
}

static char* clone_range(const char* s, int left, int right) {
    int len = right - left + 1;
    char* copy = malloc(len + 1);
    memcpy(copy, s + left, len);
    copy[len] = '\0';
    return copy;
}

static void push_result(Result* res, char** path, int path_size) {
    if (res->size == res->capacity) {
        res->capacity *= 2;
        res->data = realloc(res->data, sizeof(char**) * res->capacity);
        res->col_sizes = realloc(res->col_sizes, sizeof(int) * res->capacity);
    }
    char** row = malloc(sizeof(char*) * path_size);
    for (int i = 0; i < path_size; ++i) {
        row[i] = clone_str(path[i]);
    }
    res->data[res->size] = row;
    res->col_sizes[res->size] = path_size;
    res->size += 1;
}

static void dfs(const char* s, int n, int start, bool* pal, char** path, int path_size, Result* res) {
    if (start == n) {
        push_result(res, path, path_size);
        return;
    }
    for (int end = start; end < n; ++end) {
        if (!pal[start * n + end]) {
            continue;
        }
        path[path_size] = clone_range(s, start, end);
        dfs(s, n, end + 1, pal, path, path_size + 1, res);
        free(path[path_size]);
    }
}

char*** partition(char* s, int* returnSize, int** returnColumnSizes) {
    int n = (int)strlen(s);
    bool* pal = calloc(n * n, sizeof(bool));
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            if (s[i] == s[j] && (j - i <= 2 || pal[(i + 1) * n + (j - 1)])) {
                pal[i * n + j] = true;
            }
        }
    }

    Result res = {0};
    res.capacity = 16;
    res.data = malloc(sizeof(char**) * res.capacity);
    res.col_sizes = malloc(sizeof(int) * res.capacity);

    char** path = malloc(sizeof(char*) * n);
    dfs(s, n, 0, pal, path, 0, &res);

    free(path);
    free(pal);
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
    vector<vector<string>> partition(string s) {
        int n = (int)s.size();
        vector<vector<bool>> pal(n, vector<bool>(n, false));
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (s[i] == s[j] && (j - i <= 2 || pal[i + 1][j - 1])) {
                    pal[i][j] = true;
                }
            }
        }

        vector<vector<string>> res;
        vector<string> path;
        dfs(s, 0, pal, path, res);
        return res;
    }

private:
    void dfs(const string& s, int start, const vector<vector<bool>>& pal, vector<string>& path, vector<vector<string>>& res) {
        if (start == (int)s.size()) {
            res.push_back(path);
            return;
        }
        for (int end = start; end < (int)s.size(); ++end) {
            if (!pal[start][end]) {
                continue;
            }
            path.push_back(s.substr(start, end - start + 1));
            dfs(s, end + 1, pal, path, res);
            path.pop_back();
        }
    }
};
```

### Go

```go
package main

func partition(s string) [][]string {
	n := len(s)
	pal := make([][]bool, n)
	for i := 0; i < n; i++ {
		pal[i] = make([]bool, n)
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] && (j-i <= 2 || pal[i+1][j-1]) {
				pal[i][j] = true
			}
		}
	}

	res := make([][]string, 0)
	path := make([]string, 0, n)

	var dfs func(int)
	dfs = func(start int) {
		if start == n {
			snapshot := append([]string(nil), path...)
			res = append(res, snapshot)
			return
		}
		for end := start; end < n; end++ {
			if !pal[start][end] {
				continue
			}
			path = append(path, s[start:end+1])
			dfs(end + 1)
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
    pub fn partition(s: String) -> Vec<Vec<String>> {
        let bytes = s.as_bytes();
        let n = bytes.len();
        let mut pal = vec![vec![false; n]; n];

        for i in (0..n).rev() {
            for j in i..n {
                if bytes[i] == bytes[j] && (j - i <= 2 || pal[i + 1][j - 1]) {
                    pal[i][j] = true;
                }
            }
        }

        fn dfs(bytes: &[u8], start: usize, pal: &Vec<Vec<bool>>, path: &mut Vec<String>, res: &mut Vec<Vec<String>>) {
            if start == bytes.len() {
                res.push(path.clone());
                return;
            }
            for end in start..bytes.len() {
                if !pal[start][end] {
                    continue;
                }
                path.push(String::from_utf8(bytes[start..=end].to_vec()).unwrap());
                dfs(bytes, end + 1, pal, path, res);
                path.pop();
            }
        }

        let mut res: Vec<Vec<String>> = Vec::new();
        let mut path: Vec<String> = Vec::new();
        dfs(bytes, 0, &pal, &mut path, &mut res);
        res
    }
}
```

### JavaScript

```javascript
/**
 * @param {string} s
 * @return {string[][]}
 */
var partition = function (s) {
  const n = s.length;
  const pal = Array.from({ length: n }, () => Array(n).fill(false));

  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = i; j < n; j += 1) {
      if (s[i] === s[j] && (j - i <= 2 || pal[i + 1][j - 1])) {
        pal[i][j] = true;
      }
    }
  }

  const res = [];
  const path = [];

  function dfs(start) {
    if (start === n) {
      res.push([...path]);
      return;
    }
    for (let end = start; end < n; end += 1) {
      if (!pal[start][end]) continue;
      path.push(s.slice(start, end + 1));
      dfs(end + 1);
      path.pop();
    }
  }

  dfs(0);
  return res;
};
```
