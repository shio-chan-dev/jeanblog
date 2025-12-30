---
title: "Data Structures Basics: Number of Good Pairs (Hash Counting ACERS)"
date: 2025-12-30T11:40:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["hash-table", "counting", "array", "good-pairs", "LeetCode 1512", "ACERS"]
description: "Solve Good Pairs with one-pass hash counting, plus engineering scenarios, complexity, and multi-language solutions."
keywords: ["Good Pairs", "hash map", "frequency", "counting", "LeetCode 1512"]
---

> **Subtitle / Abstract**
> A basic counting problem: use frequency + combinations to drop O(n^2) to O(n). Includes engineering use cases and portable implementations.

- **Reading time**: 8-10 minutes
- **Tags**: `hash-table`, `counting`, `array`
- **SEO keywords**: Good Pairs, hash map, frequency
- **Meta description**: Hash counting solution for Good Pairs with complexity and code.

---

## Target readers

- Beginners learning hash tables and counting
- Engineers who want to map interview patterns to real stats tasks
- Interview prep for basic counting models

## Background / Motivation

Counting equal pairs is a classic problem. A double loop is O(n^2). With frequency counting, you can solve it in linear time and scale to large data.

## A - Algorithm (Problem and approach)

### Problem

Given an integer array `nums`, a pair `(i, j)` is a **good pair** if `nums[i] == nums[j]` and `i < j`. Return the number of good pairs.

### Input/Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | integer array |
| return | int | number of good pairs |

### Examples

| nums | output | notes |
| --- | --- | --- |
| [1, 2, 3, 1, 1, 3] | 4 | (0,3) (0,4) (3,4) (2,5) |
| [1, 1, 1, 1] | 6 | C(4,2) = 6 |
| [1, 2, 3] | 0 | no duplicates |

Simple intuition:

```
Value 1 appears 3 times -> C(3,2)=3
Value 3 appears 2 times -> C(2,2)=1
Total = 4
```

## C - Concepts (Core ideas)

- **Frequency count**: count occurrences of each value
- **Combinations**: if value appears `c` times, pairs = `c*(c-1)/2`
- **Hash table**: O(1) average update

Key formula:

```
For each value v with count c:
Pairs = c * (c - 1) / 2
```

One-pass model:

```
ans += count[nums[i]]
count[nums[i]] += 1
```

## Practical steps

1. Initialize `count` and `ans = 0`
2. For each element `x`, add `count[x]` to `ans`
3. Increment `count[x]`

## E - Engineering (Real-world usage)

### Scenario 1: Data quality scoring (Python)

Duplicate-pair score for a column:

```python
def duplicate_pair_score(values):
    count = {}
    score = 0
    for v in values:
        score += count.get(v, 0)
        count[v] = count.get(v, 0) + 1
    return score

print(duplicate_pair_score(["A", "B", "A", "C", "A"]))
```

### Scenario 2: Batch task dedup weight (Go)

```go
package main

import "fmt"

func goodPairs(ids []int) int {
	count := map[int]int{}
	ans := 0
	for _, id := range ids {
		ans += count[id]
		count[id]++
	}
	return ans
}

func main() {
	fmt.Println(goodPairs([]int{7, 7, 8, 9, 7}))
}
```

### Scenario 3: Frontend duplicate warning (JS)

```javascript
function goodPairs(items) {
  const count = new Map();
  let ans = 0;
  for (const x of items) {
    ans += count.get(x) || 0;
    count.set(x, (count.get(x) || 0) + 1);
  }
  return ans;
}

console.log(goodPairs(["u1", "u2", "u1", "u1"]));
```

## R - Reflection

### Complexity

- Time: O(n)
- Space: O(n)

### Alternatives

| Approach | Time | Space | Notes |
| --- | --- | --- | --- |
| double loop | O(n^2) | O(1) | simple but slow |
| sort and group | O(n log n) | O(1) | changes order |
| hash count (one pass) | O(n) | O(n) | fastest in practice |

### Pitfalls

- Add `count[x]` before incrementing to avoid self-pairing
- Use 64-bit integers if counts are large
- Pre-size hash map if possible

## S - Summary

- Good pairs equal combinations of equal values
- Hash counting drops O(n^2) to O(n)
- One-pass counting is clean and safe
- This model transfers to dedup stats, quality scoring, and logs

### Conclusion

Good pairs are a deceptively simple counting problem. Once you master hash counting, many similar tasks become trivial.

## References

- https://leetcode.com/problems/number-of-good-pairs/
- https://en.wikipedia.org/wiki/Combination
- https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
- https://en.cppreference.com/w/cpp/container/unordered_map
- https://doc.rust-lang.org/std/collections/struct.HashMap.html

## Call to Action (CTA)

Use this counting model as a base and adapt it to three-sum variants or grouped stats. Share your approach in comments.

## Multi-language reference implementations

```python
from typing import List


def num_identical_pairs(nums: List[int]) -> int:
    count = {}
    ans = 0
    for x in nums:
        ans += count.get(x, 0)
        count[x] = count.get(x, 0) + 1
    return ans


if __name__ == "__main__":
    print(num_identical_pairs([1, 2, 3, 1, 1, 3]))
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int key;
    int count;
    int used;
} Entry;

static unsigned hash_int(int key) {
    return (uint32_t)key * 2654435761u;
}

static int find_slot(Entry *table, int cap, int key, int *found) {
    unsigned mask = (unsigned)cap - 1u;
    unsigned idx = hash_int(key) & mask;
    while (table[idx].used && table[idx].key != key) {
        idx = (idx + 1u) & mask;
    }
    *found = table[idx].used && table[idx].key == key;
    return (int)idx;
}

long long num_identical_pairs(const int *nums, int n) {
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 2) cap = 2;
    Entry *table = (Entry *)calloc((size_t)cap, sizeof(Entry));
    if (!table) return 0;

    long long ans = 0;
    for (int i = 0; i < n; ++i) {
        int found = 0;
        int pos = find_slot(table, cap, nums[i], &found);
        if (found) {
            ans += table[pos].count;
            table[pos].count += 1;
        } else {
            table[pos].used = 1;
            table[pos].key = nums[i];
            table[pos].count = 1;
        }
    }
    free(table);
    return ans;
}

int main(void) {
    int nums[] = {1, 2, 3, 1, 1, 3};
    printf("%lld\n", num_identical_pairs(nums, 6));
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

long long num_identical_pairs(const std::vector<int> &nums) {
    std::unordered_map<int, long long> count;
    long long ans = 0;
    for (int x : nums) {
        ans += count[x];
        count[x] += 1;
    }
    return ans;
}

int main() {
    std::vector<int> nums{1, 2, 3, 1, 1, 3};
    std::cout << num_identical_pairs(nums) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func numIdenticalPairs(nums []int) int64 {
    count := map[int]int64{}
    var ans int64 = 0
    for _, x := range nums {
        ans += count[x]
        count[x]++
    }
    return ans
}

func main() {
    fmt.Println(numIdenticalPairs([]int{1, 2, 3, 1, 1, 3}))
}
```

```rust
use std::collections::HashMap;

fn num_identical_pairs(nums: &[i32]) -> i64 {
    let mut count: HashMap<i32, i64> = HashMap::new();
    let mut ans: i64 = 0;
    for &x in nums {
        let c = *count.get(&x).unwrap_or(&0);
        ans += c;
        count.insert(x, c + 1);
    }
    ans
}

fn main() {
    let nums = vec![1, 2, 3, 1, 1, 3];
    println!("{}", num_identical_pairs(&nums));
}
```

```javascript
function numIdenticalPairs(nums) {
  const count = new Map();
  let ans = 0;
  for (const x of nums) {
    ans += count.get(x) || 0;
    count.set(x, (count.get(x) || 0) + 1);
  }
  return ans;
}

console.log(numIdenticalPairs([1, 2, 3, 1, 1, 3]));
```
