---
title: "合并区间 Merge Intervals：排序扫描与区间合并的 ACERS 解析"
date: 2026-01-24T11:21:03+08:00
draft: false
categories: ["LeetCode"]
tags: ["区间", "排序", "扫描线", "合并区间", "ACERS"]
description: "围绕合并区间（LeetCode 56）讲清排序+线性扫描的核心思路、工程场景与多语言实现。"
keywords: ["Merge Intervals", "合并区间", "区间合并", "排序", "扫描线", "LeetCode 56"]
---

> **副标题 / 摘要**  
> 合并区间是最典型的“排序 + 线性扫描”问题：先按起点排序，再顺序合并重叠区间。本文按 ACERS 结构拆解题意、核心概念、工程迁移与多语言实现，帮助你形成可复用的区间处理模型。

- **预计阅读时长**：12~15 分钟
- **标签**：`区间`、`排序`、`扫描线`、`合并区间`
- **SEO 关键词**：Merge Intervals, 合并区间, 区间合并, 排序, 扫描线
- **元描述**：合并区间的排序扫描解法与工程应用解析，含复杂度对比与多语言实现。

---

## 目标读者

- 想掌握“区间合并”基础模型的初学者
- 需要把算法思路迁移到工程场景的中级开发者
- 正在准备算法面试、希望快速建立区间类题型的求职者

## 背景 / 动机

区间问题在日程排班、监控窗口、日志聚合、资源分配中非常常见。  
如果没有一个统一的合并策略，很容易产生重复统计、冲突判断错误或资源浪费。  
因此，“把重叠区间合成最少的不重叠集合”是工程与算法都高频出现的基础能力。

## A — Algorithm（题目与算法）

### 题目还原

给定一个区间数组 `intervals`，其中 `intervals[i] = [starti, endi]` 表示第 i 个区间。  
请合并所有重叠的区间，并返回一个**不重叠**的区间数组，且能完整覆盖输入中的所有区间。

### 输入输出

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| intervals | int[][] | 区间数组，元素为 `[start, end]` |
| 返回 | int[][] | 合并后的不重叠区间数组 |

### 基础示例（官方）

| 输入 | 输出 |
| --- | --- |
| [[1,3],[2,6],[8,10],[15,18]] | [[1,6],[8,10],[15,18]] |
| [[1,4],[4,5]] | [[1,5]] |

**合并示意（示例 1）**

```text
排序后: [1,3] [2,6] [8,10] [15,18]
合并:  [1,3] + [2,6] -> [1,6]
结果:  [1,6] [8,10] [15,18]
```

### 思路概览

1. 按区间起点升序排序（起点相同则按终点升序）。  
2. 线性扫描，维护当前合并区间 `[cur_start, cur_end]`。  
3. 如果下一个区间 `next_start <= cur_end`，则合并为 `cur_end = max(cur_end, next_end)`。  
4. 否则将当前区间放入结果，并以新起点开始下一段合并。

## C — Concepts（核心思想）

### 核心概念

| 概念 | 含义 | 作用 |
| --- | --- | --- |
| 重叠 | `next_start <= cur_end` | 判断是否需要合并 |
| 合并 | `cur_end = max(cur_end, next_end)` | 扩展当前区间 |
| 排序 | 按起点排序 | 让重叠区间相邻 |

### 方法类型

排序 + 线性扫描 + 贪心合并。

### 概念模型

```text
先排序 -> 扫描 -> 能合并则扩展 -> 不能合并则输出并重置
```

### 关键数据结构

使用数组/列表保存区间，结果数组按合并顺序追加即可。

## 实践指南 / 步骤

1. 按 `[start, end]` 排序区间数组。
2. 初始化结果数组 `merged`，先放入第一个区间。
3. 依次读取后续区间，与 `merged` 末尾比较并决定合并或新增。
4. 返回 `merged`。

运行方式示例：

```bash
python3 merge_intervals.py
```

## 可运行示例（Python）

```python
from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return merged


if __name__ == "__main__":
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
    print(merge([[1, 4], [4, 5]]))
```

## 解释与原理

排序让所有可能重叠的区间集中在一起：  
如果某个区间与当前合并区间不重叠（`next_start > cur_end`），  
那它也不可能与当前区间之前的任何区间重叠。  
因此“扫描 + 贪心合并”就能一次遍历完成合并，避免回头检查。

## E — Engineering（工程应用）

### 场景 1：日志时间窗聚合（Python，数据分析）

**背景**：日志分析中会出现大量连续活跃区间，需要合并以统计真实活跃时长。  
**为什么适用**：区间合并可直接消除重叠，得到最短覆盖集合。

```python
def merge_sessions(sessions):
    sessions.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e in sessions:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


print(merge_sessions([[0, 5], [3, 7], [10, 12]]))
```

### 场景 2：维护窗口合并（Go，后台服务）

**背景**：微服务系统里经常配置维护窗口，重叠窗口需要合并，避免重复停机。  
**为什么适用**：线性合并让配置校验更可靠、更易读。

```go
package main

import (
	"fmt"
	"sort"
)

type Interval struct{ Start, End int }

func mergeIntervals(intervals []Interval) []Interval {
	if len(intervals) == 0 {
		return nil
	}
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].Start != intervals[j].Start {
			return intervals[i].Start < intervals[j].Start
		}
		return intervals[i].End < intervals[j].End
	})
	merged := []Interval{intervals[0]}
	for _, it := range intervals[1:] {
		last := &merged[len(merged)-1]
		if it.Start <= last.End {
			if it.End > last.End {
				last.End = it.End
			}
		} else {
			merged = append(merged, it)
		}
	}
	return merged
}

func main() {
	fmt.Println(mergeIntervals([]Interval{{1, 3}, {2, 6}, {8, 10}}))
}
```

### 场景 3：前端日历高亮合并（JavaScript，前端）

**背景**：前端日历需要合并重叠高亮区间，避免重复渲染和冲突样式。  
**为什么适用**：合并后的区间更少、渲染更快。

```javascript
function mergeIntervals(intervals) {
  if (intervals.length === 0) return [];
  intervals.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const merged = [intervals[0].slice()];
  for (let i = 1; i < intervals.length; i += 1) {
    const [s, e] = intervals[i];
    const last = merged[merged.length - 1];
    if (s <= last[1]) {
      last[1] = Math.max(last[1], e);
    } else {
      merged.push([s, e]);
    }
  }
  return merged;
}

console.log(mergeIntervals([[1, 4], [4, 5], [10, 12]]));
```

## R — Reflection（反思与深入）

### 复杂度分析

时间复杂度：排序 O(n log n) + 扫描 O(n)。  
空间复杂度：O(n)（输出结果），若不计输出可视为 O(1) 额外空间。

### 替代方案与取舍

| 方案 | 时间 | 空间 | 说明 |
| --- | --- | --- | --- |
| 暴力两两合并 | O(n^2) | O(1) | 实现简单但易超时 |
| 扫描线/差分 | 取决于坐标 | 取决于坐标 | 适合坐标离散场景 |
| 排序 + 线性扫描 | O(n log n) | O(n) | 当前方法，稳定且易实现 |

### 常见问题与注意事项

- 必须先排序，否则无法保证相邻即为可合并对象
- 重叠判定要包含边界：`[1,4]` 与 `[4,5]` 需要合并
- 若不希望修改输入，先复制再排序
- 输入可能为空，需优雅返回空数组

### 为什么当前方法最优 / 最工程可行

排序让区间在数轴上按起点排好序，  
线性扫描保证每个区间只处理一次，整体简单、稳定、可维护，  
也是多数工程场景可接受的时间复杂度上界。

## 最佳实践与建议

- 使用 `start` 升序、`end` 次序的排序规则
- 合并时只维护一个“当前区间”，避免重复扫描
- 明确“边界相接是否合并”的业务定义（本题是合并）
- 单测覆盖空输入、完全包含、完全不重叠、边界相接

## S — Summary（总结）

- 合并区间的核心是：排序后线性扫描并贪心合并
- 重叠判定公式 `next_start <= cur_end` 是正确性的关键
- 排序把问题从“全局匹配”变成“局部合并”
- 工程上能显著降低冗余区间，提高统计与渲染效率

### 小结 / 结论

合并区间看似简单，但它是所有区间类题目的基础模型。  
掌握“排序 + 扫描”的范式，可以迁移到日程、监控、资源管理等各种场景。

## 参考与延伸阅读

- https://leetcode.com/problems/merge-intervals/
- https://en.cppreference.com/w/cpp/algorithm/sort
- https://docs.python.org/3/library/functions.html#sorted
- https://pkg.go.dev/sort

## 行动号召（CTA）

试着把本文方法应用到你负责的排班、监控或日志场景中，  
如果遇到更复杂的区间变体，欢迎留言交流你的方案与疑问。

## 多语言参考实现（Python / C / C++ / Go / Rust / JS）

```python
from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return merged


if __name__ == "__main__":
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
```

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start;
    int end;
} Interval;

static int cmp_interval(const void *a, const void *b) {
    const Interval *x = (const Interval *)a;
    const Interval *y = (const Interval *)b;
    if (x->start != y->start) {
        return x->start - y->start;
    }
    return x->end - y->end;
}

int merge_intervals(Interval *intervals, int n, Interval **out) {
    if (n == 0) {
        *out = NULL;
        return 0;
    }
    qsort(intervals, (size_t)n, sizeof(Interval), cmp_interval);
    Interval *res = (Interval *)malloc(sizeof(Interval) * (size_t)n);
    int size = 0;
    res[size++] = intervals[0];
    for (int i = 1; i < n; ++i) {
        if (intervals[i].start <= res[size - 1].end) {
            if (intervals[i].end > res[size - 1].end) {
                res[size - 1].end = intervals[i].end;
            }
        } else {
            res[size++] = intervals[i];
        }
    }
    *out = res;
    return size;
}

int main(void) {
    Interval intervals[] = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
    Interval *out = NULL;
    int n = merge_intervals(intervals, 4, &out);
    for (int i = 0; i < n; ++i) {
        printf("[%d,%d]%s", out[i].start, out[i].end, i + 1 == n ? "\n" : " ");
    }
    free(out);
    return 0;
}
```

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> merge_intervals(std::vector<std::vector<int>> intervals) {
    if (intervals.empty()) return {};
    std::sort(intervals.begin(), intervals.end(), [](const auto &a, const auto &b) {
        if (a[0] != b[0]) return a[0] < b[0];
        return a[1] < b[1];
    });
    std::vector<std::vector<int>> merged;
    merged.push_back(intervals[0]);
    for (size_t i = 1; i < intervals.size(); ++i) {
        auto &last = merged.back();
        if (intervals[i][0] <= last[1]) {
            if (intervals[i][1] > last[1]) {
                last[1] = intervals[i][1];
            }
        } else {
            merged.push_back(intervals[i]);
        }
    }
    return merged;
}

int main() {
    std::vector<std::vector<int>> intervals{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
    auto res = merge_intervals(intervals);
    for (const auto &it : res) {
        std::cout << "[" << it[0] << "," << it[1] << "] ";
    }
    std::cout << "\n";
    return 0;
}
```

```go
package main

import (
	"fmt"
	"sort"
)

func merge(intervals [][]int) [][]int {
	if len(intervals) == 0 {
		return nil
	}
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] != intervals[j][0] {
			return intervals[i][0] < intervals[j][0]
		}
		return intervals[i][1] < intervals[j][1]
	})
	merged := [][]int{append([]int{}, intervals[0]...)}
	for _, it := range intervals[1:] {
		last := merged[len(merged)-1]
		if it[0] <= last[1] {
			if it[1] > last[1] {
				last[1] = it[1]
			}
		} else {
			merged = append(merged, append([]int{}, it...))
		}
	}
	return merged
}

func main() {
	fmt.Println(merge([][]int{{1, 3}, {2, 6}, {8, 10}, {15, 18}}))
}
```

```rust
fn merge(mut intervals: Vec<[i32; 2]>) -> Vec<[i32; 2]> {
    if intervals.is_empty() {
        return vec![];
    }
    intervals.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    let mut merged: Vec<[i32; 2]> = Vec::new();
    merged.push(intervals[0]);
    for it in intervals.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if it[0] <= last[1] {
            if it[1] > last[1] {
                last[1] = it[1];
            }
        } else {
            merged.push(it);
        }
    }
    merged
}

fn main() {
    let res = merge(vec![[1, 3], [2, 6], [8, 10], [15, 18]]);
    for it in res {
        print!("[{},{}] ", it[0], it[1]);
    }
    println!();
}
```

```javascript
function merge(intervals) {
  if (intervals.length === 0) return [];
  intervals.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const merged = [intervals[0].slice()];
  for (let i = 1; i < intervals.length; i += 1) {
    const [s, e] = intervals[i];
    const last = merged[merged.length - 1];
    if (s <= last[1]) {
      last[1] = Math.max(last[1], e);
    } else {
      merged.push([s, e]);
    }
  }
  return merged;
}

console.log(merge([[1, 3], [2, 6], [8, 10], [15, 18]]));
```
