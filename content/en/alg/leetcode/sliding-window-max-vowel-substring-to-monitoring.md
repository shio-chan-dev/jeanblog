---
title: "LeetCode 1456: Maximum Vowels in a Substring (ACERS Guide)"
date: 2026-01-20T13:40:45+08:00
draft: false
categories: ["LeetCode"]
tags: ["sliding window", "string", "fixed window", "LeetCode 1456", "ACERS"]
description: "Use a fixed-size sliding window to compute the maximum number of vowels in O(n). Includes engineering scenarios and multi-language implementations."
keywords: ["Maximum Number of Vowels", "Sliding Window", "Fixed Window", "O(n)", "LeetCode 1456"]
---

> **Subtitle / Summary**  
> A standard fixed-window counting problem. This ACERS guide explains the sliding-window model, engineering use cases, and runnable multi-language solutions.

- **Reading time**: 10–12 min  
- **Tags**: `sliding window`, `string`, `fixed window`  
- **SEO keywords**: Maximum Number of Vowels, Sliding Window, Fixed Window  
- **Meta description**: Fixed-window sliding count for maximum vowels with engineering applications.

---

## Target Readers

- LeetCode learners who want stable templates  
- Engineers working on windowed metrics  
- Anyone building real-time counters

## Background / Motivation

Many engineering tasks ask: “What is the maximum count in any fixed-length window?”  
Recomputing every window is O(nk). Sliding window updates in O(1) per step, giving O(n).

## Core Concepts

- **Fixed sliding window**: length `k`, move right one step each time  
- **Incremental update**: add incoming item, remove outgoing item  
- **Condition counting**: count only items matching a predicate

---

## A — Algorithm

### Problem Restatement

Given a string `s` and an integer `k`, return the maximum number of vowels in any substring of length `k`.

### Input / Output

| Name | Type | Description |
| --- | --- | --- |
| s | string | lowercase letters |
| k | int | window length |
| return | int | maximum vowels in any length-k window |

### Example 1

```text
s = "abciiidef", k = 3
output = 3
```

### Example 2

```text
s = "aeiou", k = 2
output = 2
```

---

## C — Concepts

### Method Type

**Fixed sliding window + predicate counting**.

### Key Model

Maintain `cnt` for the current window:

```text
cnt = cnt + isVowel(s[i]) - isVowel(s[i-k])
```

Each step updates in O(1).

---

## Practical Steps

1. Count vowels in the first window  
2. Set `ans = cnt`  
3. Slide: add `s[i]`, remove `s[i-k]`  
4. Update `ans` with max  
5. Return `ans`

---

## Runnable Example (Python)

```python
def max_vowels(s: str, k: int) -> int:
    vowels = set("aeiou")
    cnt = sum(1 for c in s[:k] if c in vowels)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in vowels:
            cnt += 1
        if s[i - k] in vowels:
            cnt -= 1
        if cnt > ans:
            ans = cnt
    return ans


if __name__ == "__main__":
    print(max_vowels("abciiidef", 3))
```

Run:

```bash
python3 demo.py
```

---

## Explanation & Trade-offs

Because the window size is fixed, each move only changes two characters.  
This makes O(1) updates possible and avoids O(nk) recomputation.

---

## E — Engineering

### Scenario 1: Error Peak per Window (Go)

**Background**: Compute the maximum errors in any k-minute window.  
**Why**: Same fixed-window counting model.

```go
package main

import "fmt"

func maxErrors(flags []int, k int) int {
	cnt, ans := 0, 0
	for i, x := range flags {
		if x == 1 {
			cnt++
		}
		if i >= k && flags[i-k] == 1 {
			cnt--
		}
		if i >= k-1 && cnt > ans {
			ans = cnt
		}
	}
	return ans
}

func main() {
	fmt.Println(maxErrors([]int{0, 1, 1, 0, 1, 0, 1}, 3))
}
```

### Scenario 2: Text Feature Peak (Python)

**Background**: Count maximum keyword occurrences in any k-length window.  
**Why**: Predicate can be replaced with any condition.

```python
def max_keyword(text, k, keywords):
    s = list(text)
    cnt = sum(1 for c in s[:k] if c in keywords)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in keywords:
            cnt += 1
        if s[i - k] in keywords:
            cnt -= 1
        ans = max(ans, cnt)
    return ans


print(max_keyword("happyxxsadxxhappy", 5, set("hs")))
```

### Scenario 3: Frontend Live Highlight (JavaScript)

**Background**: Highlight max sensitive chars in the latest k input.  
**Why**: O(1) update per keystroke.

```javascript
function maxFlag(chars, k, flagSet) {
  let cnt = 0;
  for (let i = 0; i < Math.min(k, chars.length); i += 1) {
    if (flagSet.has(chars[i])) cnt += 1;
  }
  let ans = cnt;
  for (let i = k; i < chars.length; i += 1) {
    if (flagSet.has(chars[i])) cnt += 1;
    if (flagSet.has(chars[i - k])) cnt -= 1;
    ans = Math.max(ans, cnt);
  }
  return ans;
}

console.log(maxFlag("abciiidef", 3, new Set(["a", "e", "i", "o", "u"])));
```

---

## R — Reflection

### Complexity

- Time: O(n)  
- Space: O(1)

### Alternatives

| Method | Time | Space | Notes |
| --- | --- | --- | --- |
| Brute force | O(nk) | O(1) | Too slow for large data |
| Prefix sum | O(n) | O(n) | Extra memory |
| Sliding window | O(n) | O(1) | Best balance |

### Common Pitfalls

- Updating result before the window is formed  
- Forgetting to remove the outgoing element  
- Inconsistent vowel predicate

### Why this is optimal

You must inspect each character at least once.  
Sliding window achieves that lower bound with O(1) updates.

---

## S — Summary

- Fixed-window counting is a reusable template  
- Sliding window reduces O(nk) to O(n)  
- The predicate can represent any engineering condition  
- Ideal for streaming or online stats

### Further Reading

- LeetCode 1456  
- Sliding Window Pattern  
- Prefix sum vs window

---

## Conclusion

This problem is less about vowels and more about a fixed-window counting model.  
Once memorized, it translates directly to monitoring and analytics.

---

## References

- https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/
- https://en.wikipedia.org/wiki/Sliding_window_protocol

---

## Metadata

- **Reading time**: 10–12 min  
- **Tags**: sliding window, string, fixed window  
- **SEO**: Maximum Number of Vowels, Sliding Window  
- **Meta description**: O(n) fixed-window max vowels with engineering use cases.

---

## Call to Action

Try rewriting one of your monitoring metrics as a fixed-window count.  
If you want, share your use case and I can help translate it.

---

## Multi-language Reference (Python / C / C++ / Go / Rust / JS)

```python
def max_vowels(s: str, k: int) -> int:
    vowels = set("aeiou")
    cnt = sum(1 for c in s[:k] if c in vowels)
    ans = cnt
    for i in range(k, len(s)):
        if s[i] in vowels:
            cnt += 1
        if s[i - k] in vowels:
            cnt -= 1
        ans = max(ans, cnt)
    return ans


if __name__ == "__main__":
    print(max_vowels("abciiidef", 3))
```

```c
#include <stdio.h>
#include <string.h>

static int is_vowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

int max_vowels(const char *s, int k) {
    int cnt = 0;
    int ans = 0;
    int n = (int)strlen(s);
    for (int i = 0; i < n; ++i) {
        if (is_vowel(s[i])) cnt++;
        if (i >= k && is_vowel(s[i - k])) cnt--;
        if (i >= k - 1 && cnt > ans) ans = cnt;
    }
    return ans;
}

int main(void) {
    printf("%d\n", max_vowels("abciiidef", 3));
    return 0;
}
```

```cpp
#include <iostream>
#include <string>

static bool isVowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

int maxVowels(const std::string &s, int k) {
    int cnt = 0, ans = 0;
    for (int i = 0; i < (int)s.size(); ++i) {
        if (isVowel(s[i])) cnt++;
        if (i >= k && isVowel(s[i - k])) cnt--;
        if (i >= k - 1 && cnt > ans) ans = cnt;
    }
    return ans;
}

int main() {
    std::cout << maxVowels("abciiidef", 3) << "\n";
    return 0;
}
```

```go
package main

import "fmt"

func isVowel(c byte) bool {
	return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
}

func maxVowels(s string, k int) int {
	cnt, ans := 0, 0
	for i := 0; i < len(s); i++ {
		if isVowel(s[i]) {
			cnt++
		}
		if i >= k && isVowel(s[i-k]) {
			cnt--
		}
		if i >= k-1 && cnt > ans {
			ans = cnt
		}
	}
	return ans
}

func main() {
	fmt.Println(maxVowels("abciiidef", 3))
}
```

```rust
fn is_vowel(c: u8) -> bool {
    c == b'a' || c == b'e' || c == b'i' || c == b'o' || c == b'u'
}

fn max_vowels(s: &str, k: usize) -> i32 {
    let bytes = s.as_bytes();
    let mut cnt: i32 = 0;
    let mut ans: i32 = 0;
    for i in 0..bytes.len() {
        if is_vowel(bytes[i]) { cnt += 1; }
        if i >= k && is_vowel(bytes[i - k]) { cnt -= 1; }
        if i + 1 >= k && cnt > ans { ans = cnt; }
    }
    ans
}

fn main() {
    println!("{}", max_vowels("abciiidef", 3));
}
```

```javascript
function maxVowels(s, k) {
  const isVowel = (c) => "aeiou".includes(c);
  let cnt = 0;
  let ans = 0;
  for (let i = 0; i < s.length; i += 1) {
    if (isVowel(s[i])) cnt += 1;
    if (i >= k && isVowel(s[i - k])) cnt -= 1;
    if (i >= k - 1 && cnt > ans) ans = cnt;
  }
  return ans;
}

console.log(maxVowels("abciiidef", 3));
```
