---
title: "排序专题（六）：堆排序——原地 O(n log n) 的稳健方案"
subtitle: "ACERS 解读堆排序：建堆、下滤、top-k 与工程应用"
date: 2025-11-21
summary: "讲解堆排序的原理、复杂度与工程场景，对比快排/归并的取舍，附多语言实现和 top-k 应用示例。"
tags: ["sorting", "heap-sort", "algorithms", "heap", "priority-queue"]
categories: ["leetcode"]
keywords: ["堆排序", "heap sort", "原地排序", "top-k", "ACERS"]
readingTime: "约 14 分钟"
draft: false
---

> 排序系列第 6 篇聚焦堆排序：原地 O(n log n)、不稳定，常数略高但最坏时间有保障，也是流式 top-k 的基石。

## 目标读者
- 需要原地且有最坏 O(n log n) 保证的工程师。
- 想理解优先队列、top-k 与堆排序关系的学习者。
- 对比快排/归并/堆的选型者。

## 背景/动机
- 堆排序通过建堆 + 反复取堆顶实现排序，最坏/平均/最好都是 O(n log n)。
- 优势：原地、最坏有保障；劣势：不稳定，缓存友好性差，常数高于快排。
- 与优先队列/流式 top-k 共用核心结构，工程价值大。

# A — Algorithm（题目与算法）

**步骤**
1. 建最大堆（自底向上 O(n)）。
2. 反复交换堆顶与末尾，堆大小减一，对堆顶下滤恢复堆性质（O(log n)）。

**基础示例**
数组 `[4, 10, 3, 5, 1]`：
- 建堆后 `[10, 5, 3, 4, 1]`。
- 交换顶尾 → `[1,5,3,4,10]`，下滤恢复堆 → `[5,4,3,1,10]`。
- 重复直到有序。

# C — Concepts（核心思想）

| 概念 | 说明 |
| --- | --- |
| 堆性质 | 父节点 ≥ 子节点（最大堆），索引 i 的子为 2i+1, 2i+2。|
| 建堆 | 从最后一个非叶子节点向上下滤，O(n)。|
| 下滤 | 将节点向下交换到合适位置，单次 O(log n)。|
| 稳定性 | 不稳定；交换会打乱相对顺序。|
| 空间 | 原地 O(1) 额外空间。|

**复杂度**
- 时间：建堆 O(n) + n 次下滤 O(log n) ⇒ O(n log n)；最坏同样。
- 空间：O(1)；栈空间若递归实现下滤需 O(log n)，迭代则 O(1)。

# E — Engineering（工程应用）

### 场景 1：后端通用排序（C）
背景：需要原地且最坏有保障的排序。
```c
void heapify(int *a, int n, int i){
    while(1){
        int l=2*i+1, r=2*i+2, largest=i;
        if(l<n && a[l]>a[largest]) largest=l;
        if(r<n && a[r]>a[largest]) largest=r;
        if(largest==i) break;
        int t=a[i]; a[i]=a[largest]; a[largest]=t;
        i=largest;
    }
}
void heap_sort(int *a, int n){
    for(int i=n/2-1;i>=0;i--) heapify(a,n,i);
    for(int end=n-1; end>0; end--){
        int t=a[0]; a[0]=a[end]; a[end]=t;
        heapify(a,end,0);
    }
}
```

### 场景 2：流式 top-k（Python，小根堆）
背景：数据流中实时维护前 k 大。
```python
import heapq

def topk(stream, k):
    h=[]
    for x in stream:
        if len(h)<k:
            heapq.heappush(h, x)
        else:
            if x>h[0]:
                heapq.heapreplace(h, x)
    return sorted(h, reverse=True)

print(topk([5,1,9,3,12,4], 3))  # [12,9,5]
```

### 场景 3：Go 优先队列 + 排序
背景：已有 `container/heap`，演示构建堆排序。
```go
package main
import (
  "container/heap"
  "fmt"
)

type IntHeap []int
func (h IntHeap) Len() int { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *IntHeap) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *IntHeap) Pop() interface{} {
  old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x
}

func heapSort(a []int) []int {
  h := IntHeap(a)
  heap.Init(&h)
  res := make([]int, 0, len(a))
  for h.Len()>0 { res = append(res, heap.Pop(&h).(int)) }
  return res // 升序
}

func main(){ fmt.Println(heapSort([]int{4,10,3,5,1})) }
```

### 场景 4：Rust 原地堆排
```rust
pub fn heap_sort(a: &mut [i32]) {
    let n = a.len();
    // build max-heap
    for i in (0..n/2).rev() { sift_down(a, i, n); }
    for end in (1..n).rev() {
        a.swap(0, end);
        sift_down(a, 0, end);
    }
}
fn sift_down(a: &mut [i32], mut i: usize, n: usize) {
    loop {
        let l = 2*i+1; let r = l+1;
        let mut largest = i;
        if l < n && a[l] > a[largest] { largest = l; }
        if r < n && a[r] > a[largest] { largest = r; }
        if largest == i { break; }
        a.swap(i, largest);
        i = largest;
    }
}
```

### 场景 5：JavaScript 简洁版
```javascript
function heapify(a, n, i){
  while(true){
    let l=2*i+1, r=2*i+2, largest=i;
    if(l<n && a[l]>a[largest]) largest=l;
    if(r<n && a[r]>a[largest]) largest=r;
    if(largest===i) break;
    [a[i],a[largest]]=[a[largest],a[i]]; i=largest;
  }
}
function heapSort(a){
  const n=a.length;
  for(let i=Math.floor(n/2)-1;i>=0;i--) heapify(a,n,i);
  for(let end=n-1;end>0;end--){
    [a[0],a[end]]=[a[end],a[0]];
    heapify(a,end,0);
  }
  return a;
}
console.log(heapSort([4,10,3,5,1]));
```

# R — Reflection（反思与深入）

- **复杂度**：时间最坏/平均/最好均 O(n log n)；空间 O(1)。
- **替代方案**：
  - 稳定性需求 → 归并/TimSort。
  - 常数与缓存友好 → 快排更佳；堆排序常数较高。
  - 范围可知 → 计数/桶/基数更快。
- **为何可行**：
  - 最坏有保障，适用于不能容忍退化的场景。
  - 原地无额外内存，适合内存紧张环境。
  - 与优先队列/流式 top-k 共用堆结构，代码可复用。

# S — Summary（总结）

- 堆排序：原地、不稳定、最坏 O(n log n)，常数高于快排，缓存友好性稍差。
- 工程上常用堆来做 top-k/流式，而完整堆排序在标准库中较少直接暴露（C++ `std::make_heap/sort_heap`）。
- 若需稳定或近乎有序优化，用归并/TimSort；若追求低常数，用快排/Introsort；堆排序在“最坏有保障 + 原地”场景有价值。
- 建堆用自底向上 O(n)；下滤迭代实现避免递归栈。

## 实践指南 / 步骤
- 实现建堆（自底向上）与下滤（迭代），确保索引计算正确。
- 若只需 top-k，用小根堆维护 k 个元素，空间 O(k)。
- 对比性能：随机、逆序、重复多；记录交换次数与耗时，评估缓存影响。
- 如需稳定性，可在元素中加入原始索引作为第二关键字，但会增加常数。

## 常见问题与注意事项
- 易错点：子节点索引 2i+1/2i+2；交换后要继续下滤。
- 若用递归下滤，深度 O(log n)，大数组建议迭代避免栈风险。
- 堆排序不稳定，排序后相等元素相对顺序可能改变。

## 可运行示例：Python 最小版
```python
def heap_sort(a):
    n=len(a)
    def sift(i, size):
        while True:
            l,r=2*i+1,2*i+2; largest=i
            if l<size and a[l]>a[largest]: largest=l
            if r<size and a[r]>a[largest]: largest=r
            if largest==i: break
            a[i],a[largest]=a[largest],a[i]; i=largest
    for i in range(n//2-1,-1,-1): sift(i,n)
    for end in range(n-1,0,-1):
        a[0],a[end]=a[end],a[0]
        sift(0,end)
    return a
print(heap_sort([4,10,3,5,1]))
```

## 参考与延伸阅读
- CLRS《算法导论》堆排序章节
- C++ `std::make_heap` / `std::sort_heap` 实现
- William Cochran, "Heaps and Priority Queues" 技术笔记

## 元信息
- 阅读时长：约 14 分钟
- SEO 关键词：堆排序, heap sort, 原地排序, top-k, 优先队列
- 元描述：排序专题第六篇，讲解堆排序的建堆与下滤、复杂度与工程取舍，附多语言实现及 top-k 应用示例。

## 行动号召（CTA）
- 对比同一数据集的快排/堆排序耗时与交换次数，感受缓存友好度差异。
- 若有 top-k 需求，用小根堆实现一版并压测。
- 关注后续系列：非比较排序、TimSort/Introsort、排序选型实战篇。
