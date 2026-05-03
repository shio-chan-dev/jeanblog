---
title: "LeetCode 90：子集 II，从重复分支推出层内去重"
date: 2026-05-03T14:25:07+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "回溯", "子集", "去重", "LeetCode 90"]
---

## 从 `[1,2,2]` 的重复分支开始

题目给一个整数数组 `nums`，其中可能包含重复元素，要求返回所有不重复的子集。结果顺序不限。

最小能暴露问题的例子是：

```text
nums = [1,2,2]
```

如果直接照搬 `78. 子集` 的模板，搜索树里会出现两条不同分支：

```text
选择下标 1 的 2 -> [2]
选择下标 2 的 2 -> [2]
```

这两个分支路径不同，但值序列相同，最终答案重复。

所以这题真正新增的问题不是“会不会回溯”，而是：

> 怎样跳过重复分支，同时保留 `[2,2]` 这种合法答案？

这篇只用 Python，从一个能跑但浪费的版本，一步一步过渡到排序 + 层内去重。

## 题目事实

- 输入：`nums`，长度 `1 <= nums.length <= 10`
- 元素范围：`-10 <= nums[i] <= 10`
- `nums` 可能有重复元素
- 输出：所有不重复子集

示例：

```text
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

## Step 1：先复用 78 的状态，看看哪里会坏

当前部分答案仍然需要 `path`；当前层仍然需要 `start` 控制只能往右选。

先写出 78 的核心版本：

```python
def dfs(start: int) -> None:
    res.append(path.copy())

    for i in range(start, len(nums)):
        path.append(nums[i])
        dfs(i + 1)
        path.pop()
```

这个版本在 `nums` 全部互不相同时是正确的。

现在这个版本能做到：

- 枚举所有下标组合。
- 允许 `[2,2]`，因为两个 `2` 来自不同下标。

它还缺：

- 无法区分“合法使用两个重复值”和“同一层重复开启同一个值分支”。

## Step 2：先做一个正确但浪费的版本：收集时用 set 去重

在上一版基础上，先不急着优化。为了得到一个正确版本，可以把每条 `path` 转成 `tuple` 放入 `seen`。

新增 `seen`：

```python
seen: set[tuple[int, ...]] = set()
```

替换收集逻辑：

```python
state = tuple(path)
if state not in seen:
    seen.add(state)
    res.append(path.copy())
```

拼起来就是第一版完整正确代码：

```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    seen: set[tuple[int, ...]] = set()

    def dfs(start: int) -> None:
        state = tuple(path)
        if state not in seen:
            seen.add(state)
            res.append(path.copy())

        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res
```

现在这个版本能做到：

- 返回不重复子集。
- 保留 `[2,2]` 这种合法答案。

它还缺：

- 会先生成重复分支，再在收集时丢掉，搜索树没有变小。
- 如果输入是 `[2,1,2]`，相同值不相邻，后面很难在搜索阶段判断重复分支。

## Step 3：先排序，让相同值相邻

上一版的瓶颈是“重复分支已经生成出来了”。要在进入分支前跳过重复，至少要让相同值相邻。

在上一版基础上，进入 DFS 前新增排序：

```python
nums.sort()
```

现在以 `[2,1,2]` 为例，排序后变成：

```text
[1,2,2]
```

两个 `2` 相邻以后，当前层如果刚试过第一个 `2`，就能看见第二个 `2` 是同一层的重复候选。

现在这个版本能做到：

- 把相同值聚到一起，为“进入分支前去重”创造条件。
- 仍然可以用 `seen` 保证答案正确。

它还缺：

- 还没有真正跳过重复分支。

## Step 4：只跳过同一层的重复候选

现在要把“生成后去重”替换成“生成前跳过重复分支”。

关键判断是：

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

为什么是 `i > start`？

- `i > start` 表示当前值不是这一层第一个候选。
- `nums[i] == nums[i - 1]` 表示这个值已经在同一层被前一个相同值开过分支。

为什么不能写成 `i > 0`？

因为更深层允许选择第二个 `2` 来形成 `[2,2]`。

看 `[1,2,2]`：

```text
第一层：
  i = 1 选择第一个 2 -> 合法，生成 [2]
  i = 2 第二个 2 和前一个相同，且 i > start -> 跳过，避免重复 [2]

进入 [2] 的下一层：
  start = 2
  i = 2 选择第二个 2 -> i == start，不跳过，生成 [2,2]
```

现在可以移除 `seen`，恢复正常收集。

最终完整代码是：

```python
class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res: list[list[int]] = []
        path: list[int] = []

        def dfs(start: int) -> None:
            res.append(path.copy())

            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue

                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return res


if __name__ == "__main__":
    ans = Solution().subsetsWithDup([1, 2, 2])
    print(ans)
```

现在这个版本能做到：

- 在进入重复分支前直接跳过。
- 保留 `[2,2]`。
- 不需要额外的 `seen` 存储答案去重。

它还缺：

- 没有按特定顺序输出答案；题目本身不要求输出顺序。

## 慢速走一条分支

排序后 `nums = [1,2,2]`。

第一层 `dfs(0)`：

```text
收集 []
i = 0, 选 1 -> path = [1]
i = 1, 选第一个 2 -> path = [2]
i = 2, 第二个 2 与前一个相同，且 i > start，跳过
```

进入 `[2]` 对应的下一层 `dfs(2)`：

```text
start = 2
i = 2, 这里 i == start，不跳过
选第二个 2 -> path = [2,2]
```

所以这条规则的核心不是“重复值不能选”，而是：

> 同一层相同值只开一次分支；更深层仍然可以继续使用后面的重复值。

## 正确性

不变量：

- 排序后，相同值相邻。
- 在同一递归层，如果一个值已经由较小下标开过分支，后面相同值开出的分支会产生重复答案。
- `i > start` 限定“同一层”，不会影响更深层继续选择重复值。

为什么不会漏：

- 每种值在每一层至少保留第一个候选分支。
- 如果答案需要多个相同值，例如 `[2,2]`，第二个 `2` 会在更深层出现，此时 `i == start`，不会被跳过。

为什么不会重：

- 同一层相同值只允许第一个进入分支。
- 重复答案的来源正是同一层相同值开启的等价分支，因此会被消除。

## 复杂度

- 排序成本是 `O(n log n)`。
- 子集数量最多仍是 `2^n`。
- 收集答案的总复制成本是 `O(n * 2^n)`。
- 递归栈和 `path` 是 `O(n)`，输出空间是 `O(n * 2^n)`。

## 常见错误

- 把条件写成 `i > 0`，会误删 `[2,2]`。
- 忘记排序，导致相同值不相邻，无法用 `nums[i] == nums[i - 1]` 判断同层重复。
- 以为重复值不能选，实际应该跳过的是同一层重复分支。
- 仍然保留 `seen`，让代码同时存在两套去重逻辑。

## 小结

- `90. 子集 II` 是 `78. 子集` 加上重复值处理。
- 第一个正确版本可以用 `seen` 去重，但它会浪费搜索。
- 排序让相同值相邻，是进入分支前去重的前提。
- `if i > start and nums[i] == nums[i - 1]` 的含义是“同层去重”，不是“禁止重复值”。

## 参考与延伸

- LeetCode 78：Subsets
- LeetCode 90：Subsets II
- LeetCode 40：Combination Sum II

## Notes

- 题意、示例和约束来自当前仓库旧稿中的 LeetCode 90 摘要。
- 代码语言按本仓库当前 LeetCode 教程默认选择 Python。
