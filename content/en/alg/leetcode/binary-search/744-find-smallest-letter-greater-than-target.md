---
title: "LeetCode 744: Find Smallest Letter Greater Than Target Upper-Bound ACERS Guide"
date: 2026-03-18T13:49:55+08:00
draft: false
categories: ["LeetCode"]
tags: ["binary search", "upper bound", "sorted array", "characters", "LeetCode 744", "ACERS"]
description: "Find the smallest character strictly greater than a target in a sorted circular array by using upper-bound binary search and wrap-around logic, with pitfalls, engineering mappings, and runnable implementations in six languages."
keywords: ["Find Smallest Letter Greater Than Target", "upper bound", "binary search", "circular array", "LeetCode 744"]
---

> **Subtitle / Summary**  
> This problem is a textbook upper-bound search with one extra twist: wrap-around. Once you can find the first character `> target`, the rest is just handling the “no answer inside the array” case by returning the first element.

- **Reading time**: 10-12 min  
- **Tags**: `binary search`, `upper bound`, `characters`, `wrap-around`  
- **SEO keywords**: Find Smallest Letter Greater Than Target, upper bound, LeetCode 744  
- **Meta description**: Use upper-bound binary search and wrap-around handling to solve LeetCode 744, with correctness reasoning, pitfalls, engineering scenarios, and runnable code in six languages.

## Target Readers

- Learners who already know lower bound and want to master upper bound
- Engineers who search the next greater value in a sorted cyclic list
- Interview candidates practicing boundary-style binary search

## Background / Motivation

At first glance, this looks like a character problem.  
It is actually a boundary problem:

- find the first element strictly greater than `target`
- if it does not exist, wrap to the first element

That exact shape appears in many systems:

- next version after the current version
- next shard or route after the current key
- next allowed symbol in a cyclic ordered set

So the real concept is not “letters.”  
It is **upper bound plus wrap-around**.

## Core Concepts

- **Upper bound**: first index `i` such that `letters[i] > target`
- **Wrap-around**: if no such index exists, answer `letters[0]`
- **Sorted array**: gives the monotonic rule required by binary search

## A - Algorithm

### Problem Restatement

You are given a sorted list of lowercase letters `letters` and a target character `target`.

Return the smallest character in `letters` that is strictly greater than `target`.

The array is considered circular, so:

- if every character is `<= target`
- return the first character in the array

### Input / Output

| Name | Type | Meaning |
| --- | --- | --- |
| `letters` | `char[]` | sorted lowercase letters |
| `target` | `char` | current character |
| return | `char` | smallest letter strictly greater than `target` |

### Example 1

```text
letters = ['c', 'f', 'j']
target  = 'a'
output  = 'c'
```

### Example 2

```text
letters = ['c', 'f', 'j']
target  = 'c'
output  = 'f'
```

### Example 3

```text
letters = ['c', 'f', 'j']
target  = 'j'
output  = 'c'
```

## Thought Process: From Scan to Upper Bound

The direct approach is:

1. scan left to right
2. return the first character `> target`
3. if none exists, return the first character

That is `O(n)`.

Because `letters` is sorted, the predicate

```text
letters[i] > target
```

changes from `false` to `true` exactly once.  
That means binary search can find the first `true` in `O(log n)`.

## C - Concepts

### Method Category

- Binary search
- Upper-bound boundary search
- Circular fallback logic

### Why Strictly Greater Matters

This problem is not asking for:

- first `>= target`

It asks for:

- first `> target`

That one symbol difference decides whether you need lower bound or upper bound.

### Stable Algorithm

1. run upper-bound binary search to find the first index where `letters[i] > target`
2. if the index is inside the array, return `letters[index]`
3. otherwise return `letters[0]`

### Boundary Invariant

Using `[l, r)`:

- if `letters[mid] > target`, keep the answer in the left half
- otherwise move right

At the end, `l` is the first valid index, or `len(letters)` if no valid index exists.

## E - Engineering

### Scenario 1: Next Version Selection (Python)

**Background**: versions are kept in sorted order and you need the next greater version marker.  
**Why it fits**: the answer is an upper bound in a cyclic list.

```python
def next_greater(items, target):
    l, r = 0, len(items)
    while l < r:
        mid = (l + r) // 2
        if items[mid] > target:
            r = mid
        else:
            l = mid + 1
    return items[l] if l < len(items) else items[0]


print(next_greater(["c", "f", "j"], "a"))
print(next_greater(["c", "f", "j"], "j"))
```

### Scenario 2: Sorted Cyclic Routing Table (Go)

**Background**: routes are stored in sorted order and requests move to the next available slot.  
**Why it fits**: if the current route is at the end, the system wraps to the first slot.

```go
package main

import "fmt"

func nextGreater(items []byte, target byte) byte {
	l, r := 0, len(items)
	for l < r {
		mid := l + (r-l)/2
		if items[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	if l < len(items) {
		return items[l]
	}
	return items[0]
}

func main() {
	fmt.Printf("%c\n", nextGreater([]byte{'c', 'f', 'j'}, 'c'))
	fmt.Printf("%c\n", nextGreater([]byte{'c', 'f', 'j'}, 'j'))
}
```

### Scenario 3: Frontend Keyboard Hinting (JavaScript)

**Background**: a UI suggests the next available sorted symbol after the current input.  
**Why it fits**: strict-next plus wrap-around is the same model.

```js
function nextGreatestLetter(letters, target) {
  let l = 0, r = letters.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (letters[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l < letters.length ? letters[l] : letters[0];
}

console.log(nextGreatestLetter(["c", "f", "j"], "c")); // f
console.log(nextGreatestLetter(["c", "f", "j"], "j")); // c
```

## R - Reflection

### Complexity

- Time: `O(log n)`
- Space: `O(1)`

### Alternatives

- **Linear scan**: easy, but slower on large arrays
- **Modulo tricks without binary search**: not useful because the key challenge is still finding the upper bound

### Common Mistakes

- Using `>=` instead of `>`, which returns the wrong answer when `target` exists
- Forgetting the wrap-around case
- Returning `letters[l - 1]` or another neighbor without proving it

### Why This Is the Best Practical Method

The array is sorted, the condition is monotonic, and the fallback is simple.  
That makes upper-bound binary search plus one wrap-around check the cleanest and most reusable solution.

## S - Summary

- This is a pure upper-bound problem with a cyclic fallback.
- The correct search condition is `letters[i] > target`, not `>=`.
- If no valid index exists, the answer is the first element.
- The same pattern appears in version routing, cyclic lookup, and next-greater selection problems.

## Further Reading

- LeetCode 35: Search Insert Position
- LeetCode 34: Find First and Last Position of Element in Sorted Array
- Standard-library upper-bound helpers such as `bisect_right` and `upper_bound`

## Multi-language Implementations

### Python

```python
from typing import List


def next_greatest_letter(letters: List[str], target: str) -> str:
    l, r = 0, len(letters)
    while l < r:
        mid = (l + r) // 2
        if letters[mid] > target:
            r = mid
        else:
            l = mid + 1
    return letters[l] if l < len(letters) else letters[0]


if __name__ == "__main__":
    print(next_greatest_letter(["c", "f", "j"], "a"))
    print(next_greatest_letter(["c", "f", "j"], "c"))
    print(next_greatest_letter(["c", "f", "j"], "j"))
```

### C

```c
#include <stdio.h>

char nextGreatestLetter(char *letters, int n, char target) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (letters[mid] > target) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l < n ? letters[l] : letters[0];
}

int main(void) {
    char letters[] = {'c', 'f', 'j'};
    printf("%c\n", nextGreatestLetter(letters, 3, 'a'));
    printf("%c\n", nextGreatestLetter(letters, 3, 'c'));
    printf("%c\n", nextGreatestLetter(letters, 3, 'j'));
    return 0;
}
```

### C++

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

char nextGreatestLetter(const vector<char>& letters, char target) {
    auto it = upper_bound(letters.begin(), letters.end(), target);
    return it == letters.end() ? letters[0] : *it;
}

int main() {
    vector<char> letters{'c', 'f', 'j'};
    cout << nextGreatestLetter(letters, 'a') << "\n";
    cout << nextGreatestLetter(letters, 'c') << "\n";
    cout << nextGreatestLetter(letters, 'j') << "\n";
    return 0;
}
```

### Go

```go
package main

import "fmt"

func nextGreatestLetter(letters []byte, target byte) byte {
	l, r := 0, len(letters)
	for l < r {
		mid := l + (r-l)/2
		if letters[mid] > target {
			r = mid
		} else {
			l = mid + 1
		}
	}
	if l < len(letters) {
		return letters[l]
	}
	return letters[0]
}

func main() {
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'a'))
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'c'))
	fmt.Printf("%c\n", nextGreatestLetter([]byte{'c', 'f', 'j'}, 'j'))
}
```

### Rust

```rust
fn next_greatest_letter(letters: &[char], target: char) -> char {
    let (mut l, mut r) = (0usize, letters.len());
    while l < r {
        let mid = l + (r - l) / 2;
        if letters[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    if l < letters.len() { letters[l] } else { letters[0] }
}

fn main() {
    let letters = vec!['c', 'f', 'j'];
    println!("{}", next_greatest_letter(&letters, 'a'));
    println!("{}", next_greatest_letter(&letters, 'c'));
    println!("{}", next_greatest_letter(&letters, 'j'));
}
```

### JavaScript

```js
function nextGreatestLetter(letters, target) {
  let l = 0, r = letters.length;
  while (l < r) {
    const mid = (l + r) >> 1;
    if (letters[mid] > target) r = mid;
    else l = mid + 1;
  }
  return l < letters.length ? letters[l] : letters[0];
}

console.log(nextGreatestLetter(["c", "f", "j"], "a"));
console.log(nextGreatestLetter(["c", "f", "j"], "c"));
console.log(nextGreatestLetter(["c", "f", "j"], "j"));
```
