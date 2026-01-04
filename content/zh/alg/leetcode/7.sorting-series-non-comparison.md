---
title: "排序专题（七）：非比较排序——计数、桶、基数的范围与位数之战"
subtitle: "ACERS 拆解 O(n+k) 排序：何时可用、如何实现、工程取舍"
date: 2025-12-07
summary: "讲清非比较排序的适用前提、时间/空间复杂度、工程实现细节与常见坑，附计数/桶/基数排序的多语言示例。"
tags: ["sorting", "counting-sort", "bucket-sort", "radix-sort", "non-comparison"]
categories: ["leetcode"]
keywords: ["计数排序", "桶排序", "基数排序", "非比较排序", "O(n+k)"]
readingTime: "约 15 分钟"
draft: false
---

> 排序系列第 7 篇聚焦非比较排序：当数据范围或位数可控时，能把复杂度降到 O(n+k)，但需权衡空间、稳定性与工程可行性。

## 目标读者
- 处理整数键、范围/位数可知的工程师。
- 希望用更低复杂度处理大批量数据的同学。
- 想对比标准库比较排序与非比较排序取舍的人。

## 背景/动机
- 比较排序有 Ω(n log n) 下界；非比较排序利用键范围/位数信息绕过下界，实现 O(n+k)。
- 代价：额外空间，适用范围受限；实现需注意稳定性与内存占用。

# A — Algorithm（题目与算法）

**覆盖算法**：计数排序、桶排序、基数排序（LSD）。

**基础示例**
- 计数排序：`[4, 2, 2, 8, 3]`，范围 0..9，计数 → 前缀和 → 稳定回填。
- 基数排序：对整数按个位/十位/百位分组计数，逐位稳定排序。

# C — Concepts（核心思想）

| 算法 | 思路 | 时间 | 空间 | 稳定 |
| --- | --- | --- | --- | --- |
| 计数排序 | 统计频次 + 前缀和定位 | O(n+k) | O(k+n) | 可稳定 |
| 桶排序 | 按区间分桶，桶内用其他排序 | 期望 O(n+k) | O(n+k) | 取决于桶内排序 |
| 基数排序 | 按位稳定排序，多轮计数/桶 | O(d*(n+b)) | O(n+b) | 是（若每轮稳定） |

- **k**：范围大小；**d**：位数；**b**：基数（桶数）。
- 稳定性：计数排序天然可稳定；基数排序需每轮稳定；桶排序取决于桶内算法。

# E — Engineering（工程应用）

### 场景 1：小范围整数排序（Python 计数排序）
```python
def counting_sort(a, max_val):
    cnt = [0]*(max_val+1)
    for x in a: cnt[x]+=1
    # 前缀和定位
    for i in range(1, len(cnt)): cnt[i]+=cnt[i-1]
    out=[0]*len(a)
    for x in reversed(a):
        cnt[x]-=1
        out[cnt[x]] = x
    return out
print(counting_sort([4,2,2,8,3], 9))
```

### 场景 2：浮点分布已知的桶排序（JavaScript）
背景：0~1 均匀分布的小数。
```javascript
function bucketSort(arr, buckets=10){
  const B=Array.from({length:buckets},()=>[]);
  for(const x of arr){
    const idx = Math.min(buckets-1, Math.floor(x*buckets));
    B[idx].push(x);
  }
  for(const b of B) b.sort((a,b)=>a-b);
  return B.flat();
}
console.log(bucketSort([0.78,0.17,0.39,0.26,0.72,0.94,0.21,0.12,0.23,0.68]));
```

### 场景 3：大批量整数的基数排序（Go，LSD 基数）
```go
package main
import "fmt"

func radixLSD(a []int) {
    maxv := 0
    for _,v := range a { if v>maxv { maxv=v } }
    exp := 1
    buf := make([]int, len(a))
    for maxv/exp > 0 {
        cnt := make([]int, 10)
        for _,v := range a { digit := (v/exp)%10; cnt[digit]++ }
        for i:=1;i<10;i++ { cnt[i]+=cnt[i-1] }
        for i:=len(a)-1;i>=0;i-- {
            d := (a[i]/exp)%10
            cnt[d]--
            buf[cnt[d]] = a[i]
        }
        copy(a, buf)
        exp *= 10
    }
}

func main(){ a:=[]int{170,45,75,90,802,24,2,66}; radixLSD(a); fmt.Println(a) }
```

### 场景 4：C++ 计数排序（小范围）
```cpp
#include <bits/stdc++.h>
using namespace std;
vector<int> counting_sort(const vector<int>& a, int maxv){
    vector<int> cnt(maxv+1), out(a.size());
    for(int x:a) cnt[x]++;
    for(int i=1;i<=maxv;i++) cnt[i]+=cnt[i-1];
    for(int i=(int)a.size()-1;i>=0;i--){
        int x=a[i]; cnt[x]--; out[cnt[x]]=x;
    }
    return out;
}
```

### 场景 5：Rust 基数排序（LSD）
```rust
pub fn radix_lsd(a: &mut [u32]) {
    let mut maxv = *a.iter().max().unwrap();
    let mut exp = 1u32;
    let n = a.len();
    let mut buf = vec![0u32; n];
    while maxv/exp > 0 {
        let mut cnt = [0usize; 10];
        for &v in a.iter() { cnt[((v/exp)%10) as usize] += 1; }
        for i in 1..10 { cnt[i] += cnt[i-1]; }
        for &v in a.iter().rev() {
            let d = ((v/exp)%10) as usize;
            cnt[d] -= 1;
            buf[cnt[d]] = v;
        }
        a.copy_from_slice(&buf);
        exp *= 10;
    }
}
```

# R — Reflection（反思与深入）

- **复杂度与前提**：
  - 计数：O(n+k)，k 是范围；若 k ≫ n 不合算。
  - 桶：期望 O(n+k) 取决于分布假设，最坏仍可退化。
  - 基数：O(d*(n+b))，d 为位数，b 为基数；每轮需稳定排序，常用计数。
- **取舍**：
  - 内存：计数/桶需要 O(k) 或 O(n+k) 额外空间；范围大时不适用。
  - 稳定性：计数与基数可稳定，桶取决于桶内排序。
  - 数据类型：适合整数或可映射整数的键（日期、IP、定长字符串）。
- **为何可行**：
  - 当范围/位数可控时，非比较排序打破 n log n 下界，显著提速；
  - 在日志分桶、分段统计、批量整数排序等场景表现优异。

# S — Summary（总结）

- 非比较排序依赖“已知范围/位数/分布”前提，能实现 O(n+k) 时间。
- 计数排序简单稳定，适合小范围整数；基数排序适合多位整数/定长键；桶排序依赖分布假设。
- 核心风险：空间占用、分布假设不成立、稳定性需求未满足。
- 选型：范围小 → 计数；位数适中、需稳定 → 基数；均匀分布浮点 → 桶；否则回到比较排序。

## 实践指南 / 步骤
- 先估算范围/位数：若 k 接近 n 甚至更大，谨慎使用计数。
- 明确稳定性：基数需每轮稳定排序；桶内如需稳定，选稳定算法。
- 控制内存：计数数组长度 = max-min+1；基数的缓冲至少 O(n)。
- 准备测试：随机、全相等、范围极大、分布偏斜，评估性能与内存。

## 常见问题与注意事项
- 计数排序忘记偏移处理负数：需平移或分正负两段。
- 基数排序每轮若用不稳定排序，会破坏最终稳定性。
- 桶排序在分布偏斜时退化，可增加桶数或对大桶再用非比较/比较排序混合。
- 内存过大时需改用比较排序或分块处理。

## 可运行示例：Python 负数计数排序（带偏移）
```python
def counting_sort_with_neg(a):
    mn, mx = min(a), max(a)
    offset = -mn
    cnt = [0]*(mx - mn + 1)
    for x in a: cnt[x+offset]+=1
    for i in range(1,len(cnt)): cnt[i]+=cnt[i-1]
    out=[0]*len(a)
    for x in reversed(a):
        cnt[x+offset]-=1
        out[cnt[x+offset]] = x
    return out
print(counting_sort_with_neg([3,-1,2,-1,0]))
```

## 参考与延伸阅读
- CLRS《算法导论》非比较排序章节
- Donald Knuth, "The Art of Computer Programming, Vol. 3"（排序与查找）
- 关于整数排序下界与模型假设的讨论（word-RAM 模型）

## 元信息
- 阅读时长：约 15 分钟
- SEO 关键词：计数排序, 桶排序, 基数排序, 非比较排序, O(n+k)
- 元描述：排序专题第七篇，讲解非比较排序的适用前提、复杂度与工程实现，附多语言示例与取舍建议。

## 行动号召（CTA）
- 为你的数据估算范围/位数，尝试实现一版计数或基数排序并基准测试。
- 若分布偏斜，试调桶数或在大桶内改用基数/比较排序，记录效果。
- 关注后续系列：TimSort/Introsort 与排序选型实战篇。
