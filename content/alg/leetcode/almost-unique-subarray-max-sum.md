---
title: "固定长度子数组 + 至少 m 个不同元素：几乎唯一子数组的最大和"
date: 2025-12-04T10:40:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["滑动窗口", "哈希表", "数组", "双指针", "面试高频题"]
description: "给定整数数组 nums 和正整数 m、k，要求在所有长度为 k、且至少包含 m 个不同元素的子数组中，找到最大和；若不存在返回 0。本文用滑动窗口 + 计数哈希表推导 O(n) 解法，并提供多语言实现与工程应用示例。"
keywords: ["几乎唯一子数组", "滑动窗口", "distinct elements", "子数组最大和", "算法题解析"]
---

> **副标题 / 摘要**  
> 一道看似麻烦的子数组题：长度必须固定为 k，元素种类又要至少 m 个，还要在满足约束下让子数组和最大。本文通过「固定窗口滑动 + 计数哈希表」，构造 O(n) 级别的简洁算法，并给出多语言实现与工程实践案例。

- **预计阅读时长**：12~15 分钟  
- **适用场景标签**：`滑动窗口进阶`、`distinct 计数`、`子数组最大和`  
- **SEO 关键词**：almost unique subarray, at least m distinct, sliding window, subarray max sum  

---

## 目标读者与背景

**目标读者**

- 已经掌握基础滑动窗口（如「最长无重复子串」）的刷题同学
- 后端 / 数据分析工程师，需要在数组或数据流上做实时统计
- 准备中高级面试，希望写出更工程化解法的开发者

**问题背景 / 动机**

许多业务都有类似需求：

- 推荐系统：固定长度的推荐位里，既要保证足够多的不同品类，又希望整体评分尽量高；
- 监控系统：在最近的固定时间窗口里，要求至少有 m 个不同指标处于活跃状态；
- 行为分析：在 k 次连续行为中，至少访问 m 个不同页面，且总价值最大。

本题正是这类需求的抽象版，非常适合用来练习**滑动窗口 + 计数哈希表**的组合技。

---

## A — Algorithm（题目与算法）

### 题目重述

> 给定整数数组 `nums`，正整数 `m` 和 `k`。  
> 如果一个长度为 `k` 的子数组中**至少包含 `m` 个不同的元素**，则称其为“几乎唯一子数组（almost unique subarray）”。  
> 请在所有几乎唯一子数组中，找到**元素和的最大值**；如果不存在这样的子数组，则返回 `0`。

**输入**

- `nums`: 整数数组，长度为 `n`
- `m`: 至少需要包含的不同元素数量
- `k`: 子数组长度，`1 ≤ k ≤ n`

**输出**

- 整数：所有符合条件的子数组的**最大和**，若不存在则为 0

### 示例 1

```text
nums = [1, 2, 1, 2, 3]
m    = 2
k    = 3
```

所有长度为 3 的子数组为：

1. `[1, 2, 1]`，不同元素集合 `{1, 2}`，个数 2 ≥ m=2，和为 4
2. `[2, 1, 2]`，不同元素 `{1, 2}`，个数 2 ≥ 2，和为 5
3. `[1, 2, 3]`，不同元素 `{1, 2, 3}`，个数 3 ≥ 2，和为 6

所有满足条件的子数组中，最大和为 `6`。

**输出**：`6`

### 示例 2

```text
nums = [5, 5, 5, 5]
m    = 2
k    = 2
```

所有长度为 2 的子数组：

- `[5, 5]`, `[5, 5]`, `[5, 5]`  
  不同元素集合都是 `{5}`，只有 1 个元素 < m=2，不满足条件。

不存在几乎唯一子数组，因此：

**输出**：`0`

---

## C — Concepts（核心思想）

### 核心思想：固定窗口滑动 + 哈希表计数

把题目拆解一下：

1. 子数组长度必须是 **固定的 `k`**；
2. 子数组中不同的元素个数必须 **至少为 `m`**；
3. 在所有满足条件的窗口中，选**和最大的**。

这三个条件分别对应：

- 固定长度窗口 → **固定长度滑动窗口（fixed-size sliding window）**；
- 不同元素个数 → **窗口内去重计数**，适合用哈希表 / 计数器；
- 最大和 → 维护一个滑动的窗口和（sum）。

### 维护的三个核心状态

在窗口滑动过程中，需要维护：

1. `window_sum`：当前窗口内所有元素的总和；
2. `cnt[x]`：当前窗口内，元素 `x` 出现的次数；
3. `distinct`：当前窗口内 **不同元素的个数**，也就是满足 `cnt[x] > 0` 的元素数。

窗口每向右移动 1 个元素（下标 i）时：

1. 把 `nums[i]` 加入窗口：
   - `window_sum += nums[i]`
   - `cnt[nums[i]]++`
   - 若 `cnt[nums[i]]` 从 0 变到 1，则 `distinct++`
2. 如果当前窗口长度超过 `k`：
   - 移除左端元素 `nums[i-k]`：
     - `window_sum -= nums[i-k]`
     - `cnt[nums[i-k]]--`
     - 若 `cnt[nums[i-k]]` 从 1 变到 0，则 `distinct--`
3. 当 `i >= k-1` 时，窗口长度已经是 k：
   - 若 `distinct >= m`，则用 `window_sum` 更新答案。

### 算法类型

- 方法：**滑动窗口 + 哈希表计数**
- 窗口类型：**固定长度 k**
- 特点：一次遍历，同时维护「和」与「不同元素个数」两个指标

---

## 实践指南 / 步骤

可以按以下步骤从 0 到 1 实现该算法：

1. **初始化窗口状态**
   - `window_sum = 0`
   - `distinct = 0`
   - 空哈希表 `cnt`
   - 答案 `ans = 0`

2. **从左到右遍历数组**
   - 每次把 `nums[i]` 纳入窗口：
     - 更新 `window_sum` 和 `cnt`
     - 如果这个数是第一次出现，则 `distinct++`

3. **按需收缩窗口**
   - 当 `i >= k` 时，窗口中元素个数为 `k+1`，超出 1 个：
     - 移除左端 `nums[i-k]`
     - 对应更新 `window_sum`、`cnt` 和 `distinct`

4. **检查是否符合“几乎唯一子数组”条件**
   - 当 `i >= k-1`（窗口长度刚好为 k）且 `distinct >= m` 时：
     - 使用 `window_sum` 更新 `ans`

5. **返回结果**
   - 遍历结束后，若从未满足条件则 `ans` 仍为 0，直接返回

整个过程只需一次线性扫描，时间 O(n)，空间则来自哈希表中存储的不同元素数量。

---

## E — Engineering（工程应用）

下面给三个贴近实际的场景，分别使用 Python、Go、JavaScript 代码示例。

### 场景 1：推荐系统中的“多样性约束窗口评分”（Python）

**背景**  
推荐系统往往希望在固定长度的推荐位中：

- 覆盖足够多的品类（多样性，多样性差会导致用户疲劳）；
- 同时保证内容质量（总得分尽可能高）。

可以把：

- 每个位置的推荐内容评分（或 CTR 预估）当作 `nums[i]`；
- 不同内容品类 ID 当作 `nums[i]` 的另一维属性（这里简化为直接用值区别）；
- `k` 为推荐位长度，`m` 为至少要覆盖的不同品类数。

**为何适用**

需要在固定长度窗口中同时考虑：

- 不同元素数量（多样性）；
- 总和（质量）。

正好就是这道题的抽象。

**示例代码**

```python
from collections import defaultdict
from typing import List


def max_sum_almost_unique(nums: List[int], m: int, k: int) -> int:
    n = len(nums)
    if k > n:
        return 0

    cnt = defaultdict(int)
    distinct = 0
    window_sum = 0
    ans = 0

    for i, x in enumerate(nums):
        window_sum += x
        if cnt[x] == 0:
            distinct += 1
        cnt[x] += 1

        if i >= k:
            y = nums[i - k]
            window_sum -= y
            cnt[y] -= 1
            if cnt[y] == 0:
                distinct -= 1

        if i >= k - 1 and distinct >= m:
            ans = max(ans, window_sum)

    return ans


if __name__ == "__main__":
    print(max_sum_almost_unique([1, 2, 1, 2, 3], 2, 3))  # 6
```

---

### 场景 2：监控 / APM 中的“多指标活跃窗口”（Go）

**背景**  
在监控系统中，你可能希望最近的 `k` 条样本中：

- 至少有 `m` 个不同指标处于活跃状态（比如不同业务线、不同接口）；
- 并且这些样本的某种累积分值（如错误次数、延迟）尽可能大。

**为什么适合用该算法**

- 样本是按时间顺序到达的 → 非常适合滑动窗口；
- 需要同时判断“指标多样性”与“数值总和” → 正好对应 `distinct` 和 `window_sum`。

**示例代码（Go）**

```go
package main

import "fmt"

func maxSumAlmostUnique(nums []int, m, k int) int64 {
	n := len(nums)
	if k > n {
		return 0
	}

	cnt := make(map[int]int)
	distinct := 0
	var windowSum int64
	var ans int64

	for i := 0; i < n; i++ {
		x := nums[i]
		windowSum += int64(x)
		if cnt[x] == 0 {
			distinct++
		}
		cnt[x]++

		if i >= k {
			y := nums[i-k]
			windowSum -= int64(y)
			cnt[y]--
			if cnt[y] == 0 {
				distinct--
			}
		}

		if i >= k-1 && distinct >= m {
			if windowSum > ans {
				ans = windowSum
			}
		}
	}
	return ans
}

func main() {
	fmt.Println(maxSumAlmostUnique([]int{1, 2, 1, 2, 3}, 2, 3)) // 6
}
```

---

### 场景 3：前端行为分析中的“多样化点击序列”（JavaScript）

**背景**  
你在前端收集用户点击 ID 序列 `nums`，想分析：

- 在任意长度为 `k` 的连续点击中，至少要点击过 `m` 个不同元素；
- 并希望在满足多样性的前提下，总“价值”最大（比如每次点击对应一个权重）。

这可以直接在前端运行的脚本中完成，辅助埋点分析或可视化。

```js
function maxSumAlmostUnique(nums, m, k) {
  if (k > nums.length) return 0;

  const cnt = new Map();
  let distinct = 0;
  let windowSum = 0;
  let ans = 0;

  for (let i = 0; i < nums.length; i++) {
    const x = nums[i];
    windowSum += x;
    if (!cnt.has(x) || cnt.get(x) === 0) distinct++;
    cnt.set(x, (cnt.get(x) || 0) + 1);

    if (i >= k) {
      const y = nums[i - k];
      windowSum -= y;
      cnt.set(y, cnt.get(y) - 1);
      if (cnt.get(y) === 0) distinct--;
    }

    if (i >= k - 1 && distinct >= m) {
      ans = Math.max(ans, windowSum);
    }
  }

  return ans;
}

console.log(maxSumAlmostUnique([1, 2, 1, 2, 3], 2, 3)); // 6
```

---

## R — Reflection（反思与深入）

### 时间与空间复杂度

- **时间复杂度**：O(n)  
  每个元素最多进入和离开窗口各一次，哈希表操作均摊 O(1)。

- **空间复杂度**：O(U)  
  其中 U 为窗口内可能出现的不同元素个数（哈希表大小）。  
  在绝大多数场景下远小于 `n`。

---

### 替代方案与常见错误

**1. 暴力法（枚举所有子数组）**

- 对每个起点 i，构造子数组 `nums[i..i+k-1]`，用集合统计不同元素个数并求和；
- 每个窗口 O(k)，总窗口数 ~O(n) → 总复杂度 O(n·k)，大数据很容易超时。

**2. 排序 + 双指针（错误思路）**

- 有人会尝试对每个窗口排序，再统计不同元素个数；
- 但排序会破坏子数组的「相对顺序 + 固定窗口位置」结构，且复杂度更高；
- 更严重的是：排序后就不是原窗口了，无法代表真实业务含义。

**3. 只维护 distinct，不维护 sum**

- 有人会先筛出所有满足 `distinct >= m` 的窗口，再在这些窗口上重新 O(k) 计算和；
- 这相当于退化回了 O(n·k)，失去了滑动窗口的优势。

**当前方案的优势**

- 使用一个哈希表同时维护 `distinct` 与 `window_sum`，全程单次扫描；
- 数据结构简单，容易在工程中 debug 和监控；
- 模式可以直接迁移到更复杂的场景（引入权重、标签、黑白名单等）。

---

### 常见坑点与注意事项

1. **窗口边界**  
   - 收缩条件：`i >= k` 时，需要移除 `nums[i-k]`  
   - 判断窗口完整：`i >= k-1` 时，窗口长度刚好为 `k`，才能参与答案比较。

2. **distinct 更新顺序**  
   - 先更新计数，再判断是否从 0 变 1 或从 1 变 0；  
   - 避免顺序写错导致 `distinct` 统计不准。

3. **m > k 的情况**  
   - 这意味着在长度为 k 的窗口内不可能有 m 个不同元素，答案必然为 0；  
   - 代码中可提前返回，也可以让逻辑自然返回 0。

4. **整数溢出（在 C / C++ / Go 中）**  
   - 如果 `nums` 元素较大，窗口和建议用 64 位整型（`long long` / `int64`）。

---

## S — Summary（总结）

- 本题本质是在固定长度为 k 的窗口中，寻找「至少有 m 个不同元素」且「窗口和最大」的子数组。
- 使用固定长度滑动窗口可以保证只扫描数组一次，避免 O(n·k) 的重复计算。
- 哈希表计数 + 一个 `distinct` 变量即可精确维护窗口内的不同元素个数。
- 通过同时维护窗口和与 distinct，能在 O(1) 时间内判断窗口是否可行并更新答案。
- 该模式在推荐系统、监控系统、用户行为分析等工程场景中有天然的对应关系。

---

## 参考与延伸阅读

- 各类「at least / at most k distinct elements」数组 / 字符串题  
  （例如 Longest Substring with At Most K Distinct Characters）
- LeetCode 904. Fruit Into Baskets（可变窗口 + 至多两种元素）
- LeetCode 159 / 340 等一系列「含至多 K 个不同字符的最长子串」问题
- 《算法导论》中关于哈希表与线性时间算法的章节

---

## 多语言完整实现（Python / C / C++ / Go / Rust / JS）

下面是多语言版本的完整实现，你可以根据自己主要使用的语言拷贝到相应项目中。

### Python 实现

```python
from collections import defaultdict
from typing import List


def max_sum_almost_unique(nums: List[int], m: int, k: int) -> int:
    n = len(nums)
    if k > n:
        return 0

    cnt = defaultdict(int)
    distinct = 0
    window_sum = 0
    ans = 0

    for i, x in enumerate(nums):
        window_sum += x
        if cnt[x] == 0:
            distinct += 1
        cnt[x] += 1

        if i >= k:
            y = nums[i - k]
            window_sum -= y
            cnt[y] -= 1
            if cnt[y] == 0:
                distinct -= 1

        if i >= k - 1 and distinct >= m:
            ans = max(ans, window_sum)

    return ans


if __name__ == "__main__":
    print(max_sum_almost_unique([1, 2, 1, 2, 3], 2, 3))  # 6
```

---

### C 实现（示例哈希表版）

> 说明：为了保持示例完整性，这里实现了一个简单的链地址哈希表。工程中建议直接使用成熟库或根据业务数据范围改成数组计数。

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int val;
    struct Node *next;
} Node;

typedef struct {
    Node **buckets;
    int size;
} HashMap;

static unsigned int hash_int(int key, int size) {
    unsigned int x = (unsigned int)key;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    return x % size;
}

HashMap *hm_create(int size) {
    HashMap *hm = (HashMap *)malloc(sizeof(HashMap));
    hm->size = size;
    hm->buckets = (Node **)calloc(size, sizeof(Node *));
    return hm;
}

int hm_get(HashMap *hm, int key) {
    unsigned int h = hash_int(key, hm->size);
    Node *cur = hm->buckets[h];
    while (cur) {
        if (cur->key == key) return cur->val;
        cur = cur->next;
    }
    return 0;
}

void hm_add(HashMap *hm, int key, int delta) {
    unsigned int h = hash_int(key, hm->size);
    Node *cur = hm->buckets[h];
    while (cur) {
        if (cur->key == key) {
            cur->val += delta;
            return;
        }
        cur = cur->next;
    }
    Node *node = (Node *)malloc(sizeof(Node));
    node->key = key;
    node->val = delta;
    node->next = hm->buckets[h];
    hm->buckets[h] = node;
}

void hm_free(HashMap *hm) {
    for (int i = 0; i < hm->size; ++i) {
        Node *cur = hm->buckets[i];
        while (cur) {
            Node *tmp = cur;
            cur = cur->next;
            free(tmp);
        }
    }
    free(hm->buckets);
    free(hm);
}

long long maxSumAlmostUnique(int *nums, int n, int m, int k) {
    if (k > n) return 0;

    HashMap *hm = hm_create(1024);
    int distinct = 0;
    long long windowSum = 0;
    long long ans = 0;

    for (int i = 0; i < n; ++i) {
        int x = nums[i];
        windowSum += x;
        int cx = hm_get(hm, x);
        if (cx == 0) distinct++;
        hm_add(hm, x, 1);

        if (i >= k) {
            int y = nums[i - k];
            windowSum -= y;
            int cy = hm_get(hm, y);
            hm_add(hm, y, -1);
            if (cy == 1) distinct--;
        }

        if (i >= k - 1 && distinct >= m && windowSum > ans) {
            ans = windowSum;
        }
    }

    hm_free(hm);
    return ans;
}

int main(void) {
    int nums[] = {1, 2, 1, 2, 3};
    int n = sizeof(nums) / sizeof(nums[0]);
    printf("%lld\n", maxSumAlmostUnique(nums, n, 2, 3)); // 6
    return 0;
}
```

---

### C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

long long maxSumAlmostUnique(const vector<int> &nums, int m, int k) {
    int n = (int)nums.size();
    if (k > n) return 0;

    unordered_map<int, int> cnt;
    int distinct = 0;
    long long windowSum = 0;
    long long ans = 0;

    for (int i = 0; i < n; ++i) {
        int x = nums[i];
        windowSum += x;
        if (cnt[x] == 0) distinct++;
        cnt[x]++;

        if (i >= k) {
            int y = nums[i - k];
            windowSum -= y;
            cnt[y]--;
            if (cnt[y] == 0) distinct--;
        }

        if (i >= k - 1 && distinct >= m) {
            ans = max(ans, windowSum);
        }
    }
    return ans;
}

int main() {
    vector<int> nums{1, 2, 1, 2, 3};
    cout << maxSumAlmostUnique(nums, 2, 3) << endl; // 6
    return 0;
}
```

---

### Go 实现

```go
package main

import "fmt"

func maxSumAlmostUnique(nums []int, m, k int) int64 {
	n := len(nums)
	if k > n {
		return 0
	}

	cnt := make(map[int]int)
	distinct := 0
	var windowSum int64
	var ans int64

	for i := 0; i < n; i++ {
		x := nums[i]
		windowSum += int64(x)
		if cnt[x] == 0 {
			distinct++
		}
		cnt[x]++

		if i >= k {
			y := nums[i-k]
			windowSum -= int64(y)
			cnt[y]--
			if cnt[y] == 0 {
				distinct--
			}
		}

		if i >= k-1 && distinct >= m {
			if windowSum > ans {
				ans = windowSum
			}
		}
	}
	return ans
}

func main() {
	fmt.Println(maxSumAlmostUnique([]int{1, 2, 1, 2, 3}, 2, 3)) // 6
}
```

---

### Rust 实现

```rust
use std::collections::HashMap;

fn max_sum_almost_unique(nums: &[i32], m: usize, k: usize) -> i64 {
    let n = nums.len();
    if k > n {
        return 0;
    }

    let mut cnt: HashMap<i32, i32> = HashMap::new();
    let mut distinct: i32 = 0;
    let mut window_sum: i64 = 0;
    let mut ans: i64 = 0;

    for i in 0..n {
        let x = nums[i];
        window_sum += x as i64;
        let entry = cnt.entry(x).or_insert(0);
        if *entry == 0 {
            distinct += 1;
        }
        *entry += 1;

        if i >= k {
            let y = nums[i - k];
            window_sum -= y as i64;
            if let Some(e) = cnt.get_mut(&y) {
                *e -= 1;
                if *e == 0 {
                    distinct -= 1;
                }
            }
        }

        if i + 1 >= k && (distinct as usize) >= m {
            if window_sum > ans {
                ans = window_sum;
            }
        }
    }
    ans
}

fn main() {
    let nums = vec![1, 2, 1, 2, 3];
    println!("{}", max_sum_almost_unique(&nums, 2, 3)); // 6
}
```

---

### JavaScript 实现

```js
function maxSumAlmostUnique(nums, m, k) {
  if (k > nums.length) return 0;

  const cnt = new Map();
  let distinct = 0;
  let windowSum = 0;
  let ans = 0;

  for (let i = 0; i < nums.length; i++) {
    const x = nums[i];
    windowSum += x;
    if (!cnt.has(x) || cnt.get(x) === 0) distinct++;
    cnt.set(x, (cnt.get(x) || 0) + 1);

    if (i >= k) {
      const y = nums[i - k];
      windowSum -= y;
      cnt.set(y, cnt.get(y) - 1);
      if (cnt.get(y) === 0) distinct--;
    }

    if (i >= k - 1 && distinct >= m) {
      ans = Math.max(ans, windowSum);
    }
  }

  return ans;
}

console.log(maxSumAlmostUnique([1, 2, 1, 2, 3], 2, 3)); // 6
```

---

## 行动号召（CTA）

- 把这道题在你最熟悉的语言里手写一遍，并加入到自己的「滑动窗口模板库」中。
- 找几道「at most k distinct」「at least k distinct」的题，试着用同一套窗口 + 哈希表框架解决。
- 回到你的业务代码中，思考是否存在类似「固定窗口 + 多样性约束 + 最大/最小某指标」的问题，尝试用本文的方法重构一处逻辑。

