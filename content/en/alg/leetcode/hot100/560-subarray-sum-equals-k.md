---
title: "Hot100: Subarray Sum Equals K Prefix Sum + Hash Map ACERS Guide"
date: 2026-02-09T13:19:45+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "prefix sum", "hash map", "subarray", "counting", "LeetCode 560"]
description: "Count subarrays whose sum equals k in O(n) using prefix sum and frequency hash map, with engineering scenarios, pitfalls, and runnable multi-language implementations."
keywords: ["Subarray Sum Equals K", "prefix sum", "hash map", "O(n)", "Hot100", "LeetCode 560"]
---

> **Subtitle / Summary**  
> This is Hot100 article #1 for the series: Subarray Sum Equals K. We reduce the naive O(n^2) approach to O(n) with prefix sum plus a frequency hash map, then map the same pattern to real engineering scenarios.

- **Reading time**: 12-15 min  
- **Tags**: `Hot100`, `prefix sum`, `hash map`  
- **SEO keywords**: Subarray Sum Equals K, prefix sum, hash map, O(n), Hot100  
- **Meta description**: O(n) counting of subarrays with sum k using prefix sum + hash map, with complexity analysis and runnable multi-language code.

---

## Target Readers

- Hot100 learners who want stable reusable templates
- Intermediate engineers who want to transfer counting patterns to real data pipelines
- Interview prep readers who want to master prefix sum + hash map

## Background / Motivation

"Count subarrays whose sum equals k" is one of the most classic counting problems.
It appears in log analytics, risk threshold hits, and transaction sequence statistics.
The two-loop brute force method is straightforward, but slows down quickly as input grows.
So we need an O(n) method that scales.

## Core Concepts (Must Understand)

- **Subarray**: a continuous non-empty segment in an array
- **Prefix sum**: cumulative sum from start to a position
- **Difference relation**: if `prefix[r] - prefix[l-1] = k`, then `nums[l..r]` sums to `k`
- **Frequency hash map**: count how many times each prefix sum has appeared

---

## A - Algorithm

### Problem Restatement

Given an integer array `nums` and an integer `k`, return the total number of **subarrays** whose sum equals `k`.
A subarray must be continuous and non-empty.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| nums | int[] | integer array |
| k | int | target sum |
| return | int | number of subarrays with sum `k` |

### Example 1

```text
nums = [1, 1, 1], k = 2
```

Valid subarrays are `[1,1]` at indices `(0..1)` and `(1..2)`.  
**Output**: `2`

### Example 2

```text
nums = [1, 2, 3], k = 3
```

Valid subarrays are `[1,2]` and `[3]`.  
**Output**: `2`

---

## C - Concepts

### Method Category

**Prefix sum + frequency hash map**, a standard counting pattern.

### Key Formula

Define prefix sum as:

```text
prefix[0] = 0
prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
```

Then subarray sum:

```text
sum(l..r) = prefix[r+1] - prefix[l]
```

To make it equal `k`, we need:

```text
prefix[l] = prefix[r+1] - k
```

### Core Idea

Scan from left to right with running sum `s`.
At each element:

1. Count how many previous prefix sums equal `s - k`
2. Add that count to answer
3. Insert current `s` into frequency map

This order ("count first, then insert") prevents missing valid subarrays.

---

## Practical Guide / Steps

1. Initialize `s = 0`, `ans = 0`, `count = {0: 1}`
2. For each element `x` in `nums`:
   - `s += x`
   - `ans += count.get(s - k, 0)`
   - `count[s] = count.get(s, 0) + 1`
3. Return `ans`

---

## Runnable Example (Python)

```python
from typing import List


def subarray_sum(nums: List[int], k: int) -> int:
    count = {0: 1}
    ans = 0
    s = 0
    for x in nums:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


if __name__ == "__main__":
    print(subarray_sum([1, 1, 1], 2))
    print(subarray_sum([1, 2, 3], 3))
```

Run:

```bash
python3 demo.py
```

---

## Explanation / Why This Works

The hard part is continuity: subarrays must be continuous.
Prefix sum turns "continuous range sum" into "difference of two prefix sums".
So counting subarrays becomes counting how many previous prefix sums match `s - k`.

This is also why sliding window is unreliable here:
if negative numbers exist, window monotonicity breaks and common window rules fail.

---

## E - Engineering

### Scenario 1: transaction stream threshold hit counting (Python)

**Background**: count how many contiguous day ranges have net amount exactly `k`.  
**Why it fits**: amounts can be positive/negative; sliding window is not stable.

```python
def count_exact_k(amounts, k):
    count = {0: 1}
    s = 0
    ans = 0
    for x in amounts:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


print(count_exact_k([3, -1, 2, 1, -2, 4], 3))
```

### Scenario 2: service monitoring replay window counting (Go)

**Background**: count contiguous windows where error count sum equals `k` during offline replay.  
**Why it fits**: large log arrays need O(n) throughput.

```go
package main

import "fmt"

func countExactK(nums []int, k int) int {
	count := map[int]int{0: 1}
	sum := 0
	ans := 0
	for _, x := range nums {
		sum += x
		ans += count[sum-k]
		count[sum]++
	}
	return ans
}

func main() {
	fmt.Println(countExactK([]int{1, 2, 3, -2, 2}, 3))
}
```

### Scenario 3: front-end cart threshold hint (JavaScript)

**Background**: count contiguous product price blocks that exactly hit promotion threshold `k`.  
**Why it fits**: lightweight in-browser counting without backend round trip.

```javascript
function countExactK(nums, k) {
  const count = new Map();
  count.set(0, 1);
  let sum = 0;
  let ans = 0;
  for (const x of nums) {
    sum += x;
    ans += count.get(sum - k) || 0;
    count.set(sum, (count.get(sum) || 0) + 1);
  }
  return ans;
}

console.log(countExactK([5, -1, 2, 4, -2], 4));
```

---

## R - Reflection

### Complexity

- Time: `O(n)`
- Space: `O(n)`

### Alternatives and Tradeoffs

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| Brute force double loop | O(n^2) | O(1) | Simple but slow |
| Prefix sum + hash map | O(n) | O(n) | Current method, practical optimum |
| Sorted prefix / tree structure | O(n log n) | O(n) | Useful in related variants, but heavier |

### Common Wrong Ideas

- **Sliding window**: only reliable for non-negative constraints
- **Missing `count[0] = 1`**: loses subarrays starting at index 0
- **32-bit accumulation risk**: use 64-bit where overflow is possible

### Why This Is Optimal

You must inspect each element at least once, so lower bound is `O(n)`.
Hash map gives amortized `O(1)` lookup/insert per step, reaching that bound.

---

## FAQ and Notes

1. **What if array contains negatives?**  
   Prefix-sum counting handles negatives naturally and remains correct.  
   Counterexample for sliding window: `nums = [1, -1, 1], k = 1`.  
   Correct answer is 3 (`[1]`, `[1,-1,1]`, `[1]`), but positive-window rules miss cases.

2. **Can large k overflow integer sum?**  
   Use 64-bit running sum in languages where `int` may overflow.

3. **Is subarray the same as subsequence?**  
   No. Subarray is continuous. Subsequence is not.

---

## Best Practices

- Keep this as a fixed template: prefix sum + frequency map
- Always initialize `count[0] = 1`
- Prefer 64-bit for running sum on large values
- Add tests for negatives, all zeros, and `k = 0`

---

## S - Summary

- Continuous-range sum counting can be converted to prefix-sum difference counting.
- Frequency hash map reduces counting from `O(n^2)` to `O(n)`.
- Sliding window is not generally correct when negatives are present.
- `count[0] = 1` is a critical correctness detail.
- This pattern transfers well to logs, transactions, and monitoring streams.

### Recommended Further Reading

- LeetCode 560 - Subarray Sum Equals K
- Prefix Sum data structure patterns
- Hash-map frequency counting templates
- Sliding window applicability conditions

---

## Conclusion

The value of this problem is not a one-off trick.
It is a reusable counting model.
Once internalized, you can solve many "continuous range count" problems quickly and safely.

---

## References

- https://leetcode.com/problems/subarray-sum-equals-k/
- https://cp-algorithms.com/data_structures/prefix_sum.html
- https://en.cppreference.com/w/cpp/container/unordered_map
- https://doc.rust-lang.org/std/collections/struct.HashMap.html

---

## Meta Info

- **Reading time**: 12-15 min
- **Tags**: Hot100, prefix sum, hash map, counting
- **SEO keywords**: Subarray Sum Equals K, prefix sum, hash map, O(n)
- **Meta description**: Count subarrays with sum k in O(n) using prefix sum + hash map, with engineering mapping and multi-language code.

---

## Call To Action (CTA)

If you are doing Hot100, do not just memorize answers.
Write each problem as "pattern + engineering mapping" and keep a reusable template set.
Share your own variant in comments if you adapt this pattern to a production case.

---

## Multi-language Implementations (Python / C / C++ / Go / Rust / JS)

```python
from typing import List


def subarray_sum(nums: List[int], k: int) -> int:
    count = {0: 1}
    ans = 0
    s = 0
    for x in nums:
        s += x
        ans += count.get(s - k, 0)
        count[s] = count.get(s, 0) + 1
    return ans


if __name__ == "__main__":
    print(subarray_sum([1, 1, 1], 2))
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long key;
    int val;
    int used;
} Entry;

static unsigned long long hash_ll(long long x) {
    return (unsigned long long)x * 11400714819323198485ull;
}

static int find_slot(Entry *table, int cap, long long key, int *found) {
    unsigned long long mask = (unsigned long long)cap - 1ull;
    unsigned long long idx = hash_ll(key) & mask;
    while (table[idx].used && table[idx].key != key) {
        idx = (idx + 1ull) & mask;
    }
    *found = table[idx].used && table[idx].key == key;
    return (int)idx;
}

int subarray_sum(const int *nums, int n, int k) {
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    if (cap < 2) cap = 2;
    Entry *table = (Entry *)calloc((size_t)cap, sizeof(Entry));
    if (!table) return 0;

    long long sum = 0;
    int ans = 0;
    int found = 0;
    int pos = find_slot(table, cap, 0, &found);
    table[pos].used = 1;
    table[pos].key = 0;
    table[pos].val = 1;

    for (int i = 0; i < n; ++i) {
        sum += nums[i];
        pos = find_slot(table, cap, sum - k, &found);
        if (found) ans += table[pos].val;
        pos = find_slot(table, cap, sum, &found);
        if (found) {
            table[pos].val += 1;
        } else {
            table[pos].used = 1;
            table[pos].key = sum;
            table[pos].val = 1;
        }
    }
    free(table);
    return ans;
}

int main(void) {
    int nums[] = {1, 1, 1};
    printf("%d\n", subarray_sum(nums, 3, 2));
    return 0;
}
```

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>

int subarraySum(const std::vector<int> &nums, int k) {
    std::unordered_map<long long, int> count;
    count[0] = 1;
    long long sum = 0;
    int ans = 0;
    for (int x : nums) {
        sum += x;
        auto it = count.find(sum - k);
        if (it != count.end()) {
            ans += it->second;
        }
        count[sum] += 1;
    }
    return ans;
}

int main() {
    std::vector<int> nums{1, 1, 1};
    std::cout << subarraySum(nums, 2) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func subarraySum(nums []int, k int) int {
	count := map[int]int{0: 1}
	sum := 0
	ans := 0
	for _, x := range nums {
		sum += x
		ans += count[sum-k]
		count[sum]++
	}
	return ans
}

func main() {
	fmt.Println(subarraySum([]int{1, 1, 1}, 2))
}
```

```rust
use std::collections::HashMap;

fn subarray_sum(nums: &[i32], k: i32) -> i32 {
    let mut count: HashMap<i64, i32> = HashMap::new();
    count.insert(0, 1);
    let mut sum: i64 = 0;
    let mut ans: i32 = 0;
    for &x in nums {
        sum += x as i64;
        if let Some(v) = count.get(&(sum - k as i64)) {
            ans += *v;
        }
        *count.entry(sum).or_insert(0) += 1;
    }
    ans
}

fn main() {
    let nums = vec![1, 1, 1];
    println!("{}", subarray_sum(&nums, 2));
}
```

```javascript
function subarraySum(nums, k) {
  const count = new Map();
  count.set(0, 1);
  let sum = 0;
  let ans = 0;
  for (const x of nums) {
    sum += x;
    ans += count.get(sum - k) || 0;
    count.set(sum, (count.get(sum) || 0) + 1);
  }
  return ans;
}

console.log(subarraySum([1, 1, 1], 2));
```
