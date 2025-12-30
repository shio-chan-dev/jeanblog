---
title: "排序专题（四）：归并排序——稳定分治与外部排序的首选"
subtitle: "ACERS 拆解归并排序：自顶向下/自底向上、稳定性、外部排序与多语言实现"
date: 2025-12-04
summary: "系统讲解归并排序的分治原理、稳定性、空间取舍与工程场景，附 Python/C/C++/Go/Rust/JS 实现、外部排序思路与选型建议。"
tags: ["sorting", "merge-sort", "algorithms", "divide-and-conquer", "external-sort"]
categories: ["leetcode"]
keywords: ["归并排序", "稳定排序", "外部排序", "分治", "ACERS"]
readingTime: "约 15 分钟"
draft: false
---

> 排序系列第 4 篇聚焦归并排序：典型分治、稳定、时间 O(n log n)，代价是 O(n) 额外空间。它既是教科书算法，也是外部排序与语言内置稳定排序的基础。

## 目标读者
- 需要稳定排序且能接受 O(n) 额外空间的工程师。
- 学习分治思想、为快排/TimSort 打基础的同学。
- 要处理大文件、流式数据，想了解外部归并的人。

## 背景/动机
- 归并排序在任何输入上都有 O(n log n) 的稳定时间复杂度，不受枢轴退化影响。
- 代价：额外 O(n) 空间，原地版本复杂且常数大。
- 外部排序场景（数据大于内存）常用“分块排序 + 多路归并”——归并思想的直接应用。

# A — Algorithm（题目与算法）

**题目**：对可比较序列排序，要求稳定，时间 O(n log n)。

**步骤（自顶向下）**
1. 分：递归将数组拆成两半。
2. 治：分别排序左右半部分。
3. 合：用辅助数组按序合并两个有序子数组。

**基础示例**
数组 `[5,2,4,6,1,3]` 拆分合并流程：
- 拆成 `[5,2,4]` 与 `[6,1,3]`，各自再拆。
- 合并 `[2,4,5]` 与 `[1,3,6]` → `[1,2,3,4,5,6]`（稳定保持相对顺序）。

# C — Concepts（核心思想）

| 关键概念 | 说明 |
| --- | --- |
| 分治 | 递归拆分到子问题，再合并解决。|
| 稳定 | 合并时若元素相等，先取左边，保持原相对顺序。|
| 空间 | 典型实现需 O(n) 辅助数组；自底向上迭代仍需缓冲。|
| 变体 | 自底向上迭代归并、块归并、外部多路归并。|

**复杂度**
- 时间：T(n) = 2T(n/2) + O(n) ⇒ O(n log n)（最坏/平均/最好一致）。
- 空间：O(n) 辅助空间（外排时缓冲块大小相关）。

# E — Engineering（工程应用）

### 场景 1：需要稳定的多键排序（Python）
背景：日志按时间、再按 user_id 排序，需稳定保持同时间的原顺序。
为何：Python 内置排序是稳定归并系（TimSort），直接使用即可。
```python
from operator import itemgetter
logs = [("2025-11-21", "u2"), ("2025-11-21", "u1"), ("2025-11-20", "u3")]
logs.sort(key=itemgetter(0,1))
print(logs)
```

### 场景 2：外部排序的大文件（C++）
背景：对 10GB 整数文件排序，内存 512MB。
为何：用分块排序 + k 路归并，稳定且可控内存。
```cpp
// 伪代码骨架，展示思路
auto sort_chunk = [](vector<int>& buf, int id){
    sort(buf.begin(), buf.end());
    ofstream out("chunk"+to_string(id)+".tmp");
    for(int v:buf) out<<v<<'\n';
};
// 读取 -> 分块排序写盘 -> k 路归并（用优先队列最小堆）
```

### 场景 3：前端稳定排序（JavaScript）
背景：表格需保持同 key 的原顺序。
为何：现代浏览器排序多数稳定；如需保证，使用索引搭配归并实现。
```javascript
function mergeSort(arr){
  if(arr.length<=1) return arr;
  const mid = arr.length>>1;
  const left = mergeSort(arr.slice(0,mid));
  const right = mergeSort(arr.slice(mid));
  const res=[]; let i=0,j=0;
  while(i<left.length && j<right.length){
    if(left[i].key <= right[j].key) res.push(left[i++]);
    else res.push(right[j++]);
  }
  return res.concat(left.slice(i)).concat(right.slice(j));
}
console.log(mergeSort([{key:1},{key:1},{key:0}]));
```

### 场景 4：Go 后端稳定排序
背景：需要稳定地按多个字段排序结构体。
为何：`sort.SliceStable` 基于归并，直接可用。
```go
package main
import (
  "fmt"
  "sort"
)

type Item struct{ Date string; User string }
func main(){
  items := []Item{{"2025-11-21","u2"},{"2025-11-21","u1"},{"2025-11-20","u3"}}
  sort.SliceStable(items, func(i, j int) bool {
    if items[i].Date == items[j].Date { return items[i].User < items[j].User }
    return items[i].Date < items[j].Date
  })
  fmt.Println(items)
}
```

# R — Reflection（反思与深入）

- **复杂度分析**：时间 O(n log n)，空间 O(n)；外部排序空间与块大小相关，I/O 主导成本。
- **对比替代**：
  - vs 快排：快排原地、常数小但不稳定且可能退化；归并稳定且有固定上界。
  - vs 堆排：堆排原地、不稳定，缓存友好差；归并更适合需要稳定性或外部排序。
  - vs TimSort：TimSort 在近乎有序数据上更快且稳定，但实现复杂；归并是其基石。
- **为何可行/优选**：需要稳定性、可预测的 O(n log n)，或处理外部数据时，归并是默认选择。

# S — Summary（总结）

- 归并排序提供稳定、可预测的 O(n log n)，代价是 O(n) 额外空间。
- 外部排序、稳定多键排序、语言标准库的稳定排序都依赖归并思想。
- 自底向上迭代归并可避免递归开销，但仍需辅助缓冲。
- 若输入近乎有序且希望更快，可考虑 TimSort；若空间受限且不需稳定，可用快排/堆排。
- 评估时关注：稳定性需求、可用内存、数据规模与 I/O 成本。

## 实践指南 / 步骤
- 明确稳定性与空间预算：可用 O(n) 缓冲则选归并/稳定库；否则考虑快排/堆排。
- 选择实现：递归自顶向下简单；迭代自底向上适合避免深递归。
- 编写合并函数时确保稳定性：相等时取左侧元素。
- 边界测试：空数组、单元素、全相等、逆序、重复多，确保合并逻辑正确。

## 可运行示例：多语言实现

### Python（自顶向下）
```python
def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a)//2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    i=j=0; res=[]
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i+=1
        else:
            res.append(right[j]); j+=1
    res.extend(left[i:]); res.extend(right[j:])
    return res
print(merge_sort([5,2,4,6,1,3]))
```

### C（自底向上）
```c
#include <stdlib.h>
void merge(int *a, int *buf, int l, int m, int r){
    int i=l, j=m, k=l;
    while(i<m && j<r){
        if(a[i] <= a[j]) buf[k++] = a[i++];
        else buf[k++] = a[j++];
    }
    while(i<m) buf[k++] = a[i++];
    while(j<r) buf[k++] = a[j++];
    for(int t=l; t<r; ++t) a[t]=buf[t];
}
void merge_sort(int *a, int n){
    int *buf = malloc(sizeof(int)*n);
    for(int width=1; width<n; width*=2){
        for(int i=0; i<n; i+=2*width){
            int l=i, m=i+width< n? i+width: n, r=i+2*width< n? i+2*width: n;
            merge(a, buf, l, m, r);
        }
    }
    free(buf);
}
```

### C++（自顶向下）
```cpp
void merge(vector<int>& a, int l, int m, int r, vector<int>& buf){
    int i=l,j=m,k=l;
    while(i<m && j<r){
        if(a[i]<=a[j]) buf[k++]=a[i++];
        else buf[k++]=a[j++];
    }
    while(i<m) buf[k++]=a[i++];
    while(j<r) buf[k++]=a[j++];
    for(int t=l;t<r;++t) a[t]=buf[t];
}
void merge_sort(vector<int>& a, int l, int r, vector<int>& buf){
    if(r-l<=1) return;
    int m = l + (r-l)/2;
    merge_sort(a,l,m,buf); merge_sort(a,m,r,buf);
    merge(a,l,m,r,buf);
}
```

### Go（自顶向下）
```go
func mergeSort(a []int) []int {
    if len(a) <= 1 { return a }
    mid := len(a)/2
    left := mergeSort(a[:mid])
    right := mergeSort(a[mid:])
    res := make([]int, 0, len(a))
    i, j := 0, 0
    for i < len(left) && j < len(right) {
        if left[i] <= right[j] { res = append(res, left[i]); i++ } else { res = append(res, right[j]); j++ }
    }
    res = append(res, left[i:]...)
    res = append(res, right[j:]...)
    return res
}
```

### Rust（自顶向下，临时缓冲）
```rust
fn merge_sort(a: &mut [i32]) {
    let n = a.len();
    if n <= 1 { return; }
    let mid = n/2;
    merge_sort(&mut a[..mid]);
    merge_sort(&mut a[mid..]);
    let mut buf = a.to_vec();
    merge(&a[..mid], &a[mid..], &mut buf[..]);
    a.copy_from_slice(&buf);
}
fn merge(left: &[i32], right: &[i32], out: &mut [i32]) {
    let (mut i, mut j, mut k) = (0,0,0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] { out[k]=left[i]; i+=1; }
        else { out[k]=right[j]; j+=1; }
        k+=1;
    }
    if i < left.len() { out[k..k+left.len()-i].copy_from_slice(&left[i..]); }
    if j < right.len() { out[k..k+right.len()-j].copy_from_slice(&right[j..]); }
}
```

### JavaScript（自顶向下）
```javascript
function mergeSort(a){
  if(a.length<=1) return a;
  const mid = a.length>>1;
  const left = mergeSort(a.slice(0,mid));
  const right = mergeSort(a.slice(mid));
  const res=[]; let i=0,j=0;
  while(i<left.length && j<right.length){
    if(left[i] <= right[j]) res.push(left[i++]);
    else res.push(right[j++]);
  }
  return res.concat(left.slice(i)).concat(right.slice(j));
}
console.log(mergeSort([5,2,4,6,1,3]));
```

## 常见问题与注意事项
- 递归深度：对大 n 可用自底向上迭代或尾递归优化；某些语言需调栈或用迭代。
- 空间占用：在内存紧张场景需评估 O(n) 缓冲；外部排序要控制块大小与归并路数。
- 稳定性：合并时相等元素必须先取左侧，避免破坏稳定性。
- 性能：复制成本高时可用双缓冲、交替读写减少拷贝；注意缓存友好性。

## 最佳实践与建议
- 如果语言提供稳定排序（Python、Java `Arrays.sort` 对象版、Go `SliceStable`），优先使用库实现。
- 自定义实现时，抽出 `merge` 函数，保证稳定性；为大数据使用自底向上避免深递归。
- 外部排序：控制块大小以适配内存；使用优先队列做 k 路归并；批量写入减少 I/O 调用。
- 对近乎有序数据，考虑 TimSort；归并是理解 TimSort run 合并策略的基础。

## 小结 / 结论
- 归并排序以稳定性和固定的 O(n log n) 见长，适合稳定多键排序与外部排序。
- 额外空间是主要代价；原地变体复杂且常数大，工程中少用。
- 迭代归并可以避免递归深度问题；外部归并是处理超大数据的必备技能。

## 参考与延伸阅读
- CLRS《算法导论》归并排序
- TimSort 论文与 CPython/Java 源码（run 合并策略）
- PostgreSQL tuplesort 外部排序实现

## 元信息
- 阅读时长：约 15 分钟
- SEO 关键词：归并排序, 稳定排序, 外部排序, 分治, Merge Sort
- 元描述：排序专题第四篇，深入讲解归并排序的分治原理、稳定性、空间取舍与外部排序应用，附多语言实现与选型建议。

## 行动号召（CTA）
- 对你的数据集测试库内置稳定排序与自实现归并的性能差异。
- 若处理大文件，尝试实现分块 + k 路归并的外部排序原型，记录 I/O 成本。
- 关注后续系列：快排、堆排、非比较排序、TimSort/Introsort 与选型实战。
