---
title: "LeetCode 547: Number of Provinces, Turn an Adjacency Matrix into Connected Components"
date: 2026-05-29T00:00:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "graph", "DFS", "BFS", "Union Find", "connected component", "LeetCode 547"]
description: "Turn LeetCode 547 Number of Provinces from direct connections in an adjacency matrix into connected component counting, then derive DFS, BFS, and Union Find solutions step by step."
keywords: ["LeetCode 547", "Number of Provinces", "connected components", "DFS", "BFS", "Union Find"]
---

## Problem Requirement

You are given an `n x n` matrix `isConnected`, where there are `n` cities.

If `isConnected[i][j] == 1`, city `i` and city `j` are **directly connected**. If two cities can reach each other through one or more directly connected cities, they belong to the same province.

The problem asks you to return the total number of provinces.

The easiest mistake is to think that the answer is the number of `1`s in the matrix, or only the number of directly connected pairs. That is not what the problem asks.

The real question is:

> How many groups of cities are connected, either directly or indirectly?

In graph language:

> Given an undirected graph represented by an adjacency matrix, return the number of connected components.

### Input / Output

- Input: `isConnected: List[List[int]]`
- Output: province count as an `int`
- City IDs can be understood as `0..n-1`.
- `isConnected[i][j] == 1` means there is an edge between city `i` and city `j`.
- `isConnected[i][j] == 0` means city `i` and city `j` are not directly connected.

### Example 1

```text
Input: isConnected = [
  [1, 1, 0],
  [1, 1, 0],
  [0, 0, 1]
]
Output: 2
```

City `0` and city `1` are directly connected, so they belong to the same province.

City `2` is only connected to itself. It cannot be reached from city `0` or city `1`, so it forms another province by itself.

There are two provinces:

```text
{0, 1}
{2}
```

The answer is `2`.

### Example 2

```text
Input: isConnected = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
]
Output: 3
```

No two different cities are directly connected. Each city can only reach itself, so every city is its own province:

```text
{0}
{1}
{2}
```

The answer is `3`.

### Constraints

- `1 <= n <= 200`
- `isConnected.length == n`
- `isConnected[i].length == n`
- `isConnected[i][j]` is either `0` or `1`
- `isConnected[i][i] == 1`
- `isConnected[i][j] == isConnected[j][i]`

### What This Section Freezes

We have translated the problem from a matrix problem into a connected component counting problem.

At this checkpoint, we can:

- Understand that `1` in the matrix means direct connection.
- Understand that a province may contain cities connected through intermediate cities.
- Understand that the target is not to count edges, but to count connected components.
- Explain why the two official examples return `2` and `3`.

It still lacks:

- How to find the full province that one city belongs to.
- How to avoid counting the same province more than once.
- How to turn the idea into runnable code.

## Step 1: Solve a Smaller Problem First

The current baseline is: we know the target is to count connected components, but we still do not know how to "get one province".

Asking "how many provinces are there in total?" is still too large. First shrink the problem:

> Given one city that has not been processed yet, how do we find all cities that belong to the same province?

Look at Example 1:

```text
isConnected = [
  [1, 1, 0],
  [1, 1, 0],
  [0, 0, 1]
]
```

If we start from city `0`:

- `isConnected[0][0] == 1`, so city `0` belongs to the current province.
- `isConnected[0][1] == 1`, so city `1` is directly connected to city `0` and also belongs to the current province.
- `isConnected[0][2] == 0`, so city `2` cannot be reached directly from city `0`.

Then continue from city `1`:

- `isConnected[1][0] == 1`, and city `0` is already in the current province.
- `isConnected[1][1] == 1`, and city `1` is already in the current province.
- `isConnected[1][2] == 0`, so city `2` still does not belong to the current province.

Starting from city `0`, we can find the full province:

```text
{0, 1}
```

The important part of this step is not the final answer. It is the intermediate capability:

> Starting from one city, mark every city that can be reached directly or indirectly.

To do that, we need a `visited` array:

```text
visited[i] == True  means city i has already been assigned to a discovered province
visited[i] == False means city i has not been assigned to any discovered province yet
```

If we start from an unvisited city and mark every reachable city as visited, that one traversal has found one complete province.

The smaller problem can be written as pseudocode:

```text
visit(city):
    mark city as visited
    for next_city in all cities:
        if city is connected to next_city and next_city is not visited:
            visit(next_city)
```

This is the entry point for graph traversal. It can be written with DFS or BFS.

For now, freeze only this smaller problem. Do not rush into the full answer yet.

At this checkpoint, we can:

- Start from "find one province" instead of jumping straight to the final count.
- Understand that `visited` prevents duplicate processing.
- Understand that one traversal marks every city in the same province.

It still lacks:

- How to write `visit(city)` in Python.
- How to scan all cities and count provinces.

## Step 2: Use DFS to Find Every Province

The previous baseline was:

```text
visit(city) can start from one city and mark every city in the same province.
```

But it still cannot answer the final question. The problem asks for the total number of provinces, not only the province containing city `0`.

This breaks in Example 1:

```text
{0, 1}
{2}
```

If we only start from city `0`, we only get `{0, 1}`. City `2` is never processed.

So we need to add an outer scan on top of the previous version:

> Check every city from left to right. If a city is not visited yet, we have found a new province. Increment the answer, then start DFS from that city to mark the whole province.

This can be written directly as LeetCode code:

```python
from typing import List


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n

        def dfs(city: int) -> None:
            visited[city] = True
            for next_city in range(n):
                if isConnected[city][next_city] == 1 and not visited[next_city]:
                    dfs(next_city)

        provinces = 0
        for city in range(n):
            if not visited[city]:
                provinces += 1
                dfs(city)

        return provinces
```

There are two actions in this code.

The first action is `dfs(city)`:

```python
def dfs(city: int) -> None:
    visited[city] = True
    for next_city in range(n):
        if isConnected[city][next_city] == 1 and not visited[next_city]:
            dfs(next_city)
```

It has one job: starting from `city`, mark every city in the same province as visited.

The second action is the outer scan:

```python
provinces = 0
for city in range(n):
    if not visited[city]:
        provinces += 1
        dfs(city)
```

The condition `if not visited[city]` is the key.

If a city has already been visited, it already belongs to a previously discovered province and must not be counted again.

If a city has not been visited, then none of the previously discovered provinces can reach it. It must start a new province. So we first do `provinces += 1`, then use `dfs(city)` to mark that entire new province.

Walk through Example 1:

```text
visited = [False, False, False]
provinces = 0

city = 0:
  not visited, provinces = 1
  dfs(0) marks 0 and 1
  visited = [True, True, False]

city = 1:
  already visited, skip

city = 2:
  not visited, provinces = 2
  dfs(2) marks 2
  visited = [True, True, True]
```

The final answer is `2`.

Now look at Example 2:

```text
isConnected = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
]
```

No city can reach any other city. The outer scan starts DFS at city `0`, city `1`, and city `2`, so the answer is `3`.

At this checkpoint, we can:

- Use DFS to find the full province starting from one city.
- Use an outer scan to count all provinces.
- Correctly handle connected cities and isolated cities.

It still lacks:

- Why this count never misses a province and never counts one twice.
- Time complexity and space complexity.

## Step 3: Why the DFS Count Misses Nothing and Counts Nothing Twice

Now we have a DFS solution that passes the examples. But passing examples is not enough. The tricky part of this problem is not syntax. It is this counting logic:

```python
for city in range(n):
    if not visited[city]:
        provinces += 1
        dfs(city)
```

Why can we increment `provinces` immediately when we see an unvisited city?

Start with a wrong idea: increment the answer every time we see a connected edge.

For example:

```text
0 -- 1 -- 2
```

The matrix may show that `0` is connected to `1`, and `1` is connected to `2`. If we count edges, we get two connections. But all three cities belong to one province, so the answer should be `1`.

A province is not the number of edges. A province is the whole group that can be reached from one entry point.

The DFS code depends on three invariants.

The first invariant:

> After `dfs(city)` finishes, every city reachable from `city` is marked as `visited = True`.

The reason is that `dfs` checks every possible `next_city` from `city`. If there is an edge and `next_city` has not been visited, it continues DFS from `next_city`. This expansion covers both direct and indirect connections.

The second invariant:

> If the outer scan sees `visited[city] == False`, then `city` does not belong to any province discovered earlier.

Every time we discover a province, we immediately call DFS and mark every city in that province. If `city` is still unvisited, all previous DFS traversals failed to reach it. It must start a new province.

The third invariant:

> One province is counted exactly once.

When the outer scan first reaches a province, the answer is incremented once, and DFS marks the whole province. Later, when the scan reaches other cities in that same province, they are already `visited = True` and are skipped.

Put together, these invariants prove the counting logic:

- No missing province: the outer scan checks every city, and any unvisited city starts a DFS.
- No duplicate province: once a city is marked by DFS, it cannot start another count later.
- The count is a province count: each increment corresponds to one full connected component, not one edge or one `1` in the matrix.

### Complexity

`isConnected` is an adjacency matrix. When DFS processes one city, it scans the entire row:

```python
for next_city in range(n):
```

Each city enters `dfs` at most once because it is marked as `visited`. Each entry scans `n` possible neighbors, so the total time complexity is:

```text
O(n * n) = O(n^2)
```

Do not write this as `O(n + m)`. `O(n + m)` is a better fit for adjacency lists. This problem gives an adjacency matrix, and the matrix itself has `n^2` cells, so matrix scanning dominates.

Space complexity comes from two parts:

- `visited` array: `O(n)`
- DFS recursion stack: worst-case depth `O(n)`

So the space complexity is:

```text
O(n)
```

At this checkpoint, we can:

- Explain why seeing an unvisited city means we can increment the province count.
- Explain why DFS does not miss indirectly connected cities.
- Explain why the same province is not counted twice.
- Give the matrix-based `O(n^2)` time complexity and `O(n)` space complexity.

It still lacks:

- How to write the same graph traversal with BFS if we do not want recursion.
- How to write the solution with Union Find if we think in terms of merging sets.

## Step 4: The Same Idea as BFS

The DFS version is already enough to solve the problem. BFS is not a new idea here. It is the same action, "expand from one entry city to the whole province", written with a different container.

The current baseline is:

```text
find an unvisited city -> provinces += 1 -> use DFS to mark the whole province
```

If you do not want recursion, put "cities to visit next" into a queue. Every time you pop one city from the queue, scan all neighbors it can reach. If a neighbor has not been visited yet, mark it and push it into the queue.

In other words, replace this part:

```text
dfs(city)
```

with:

```text
bfs(city)
```

Here is the BFS version:

```python
from collections import deque
from typing import List


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n

        def bfs(start: int) -> None:
            visited[start] = True
            queue = deque([start])

            while queue:
                city = queue.popleft()
                for next_city in range(n):
                    if isConnected[city][next_city] == 1 and not visited[next_city]:
                        visited[next_city] = True
                        queue.append(next_city)

        provinces = 0
        for city in range(n):
            if not visited[city]:
                provinces += 1
                bfs(city)

        return provinces
```

Notice that `visited[start] = True` happens before enqueueing the start node, and `visited[next_city] = True` happens when a neighbor is discovered.

This prevents the same city from being pushed into the queue multiple times. For this problem, duplicate queue entries may still produce the correct final answer, but they create unnecessary work. The cleaner rule is: once a city is handed to the queue, mark it immediately.

Walk through Example 1 with BFS:

```text
city = 0 is not visited:
  provinces = 1
  queue = [0]

pop 0:
  discover 1, mark it and enqueue it
  queue = [1]

pop 1:
  0 is visited, 1 is visited, 2 is not connected
  queue = []

city = 1 is already visited, skip

city = 2 is not visited:
  provinces = 2
  queue = [2]
  pop 2 and finish
```

BFS and DFS use exactly the same outer counting logic:

```python
for city in range(n):
    if not visited[city]:
        provinces += 1
        bfs(city)
```

So their correctness comes from the same fact: every time we start from an unvisited city, we mark one complete new province.

The complexity is also the same:

- Time complexity: `O(n^2)`
- Space complexity: `O(n)`

At this checkpoint, we can:

- Rewrite recursive DFS as queue-based BFS.
- Preserve the same outer scan and `visited` meaning.
- Explain why we mark a city as visited when it is discovered.

It still lacks:

- How to solve the same problem from the viewpoint of "connected cities should be merged into one set".

## Step 5: Merge Connected Cities with Union Find

DFS and BFS both do the same thing:

```text
Start from one city and find every city in the same province.
```

Union Find uses a different viewpoint:

```text
At first, every city is its own province.
Whenever we see a connection, merge the two cities' provinces.
The number of remaining sets is the number of provinces.
```

Use Example 1:

```text
Initial:
{0} {1} {2}

See that 0 and 1 are connected:
{0, 1} {2}

There is no other cross-set connection.
2 sets remain.
```

That is why Union Find fits this problem. The input gives connection relations, and Union Find is built for this question:

> If two elements are related, merge them into the same set.

The code needs three pieces.

First, `parent[i]` stores the representative of the set containing city `i`:

```python
parent = list(range(n))
```

At the beginning, every city is its own representative.

Second, `find(city)` finds the final representative of the set containing a city:

```python
def find(city: int) -> int:
    if parent[city] != city:
        parent[city] = find(parent[city])
    return parent[city]
```

The line `parent[city] = find(parent[city])` is path compression. It does not change the answer. It only makes later lookups faster.

Third, `union(a, b)` merges the sets containing two cities. If they already belong to the same set, do nothing. If they belong to different sets, merge them and reduce the province count by one:

```python
def union(a: int, b: int) -> None:
    nonlocal provinces
    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return

    parent[root_b] = root_a
    provinces -= 1
```

Here is the complete code:

```python
from typing import List


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        parent = list(range(n))
        provinces = n

        def find(city: int) -> int:
            if parent[city] != city:
                parent[city] = find(parent[city])
            return parent[city]

        def union(a: int, b: int) -> None:
            nonlocal provinces

            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return

            parent[root_b] = root_a
            provinces -= 1

        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    union(i, j)

        return provinces
```

The inner loop starts from `i + 1`:

```python
for j in range(i + 1, n):
```

The matrix is symmetric. `isConnected[i][j]` and `isConnected[j][i]` describe the same undirected edge. It is enough to scan the upper triangle. The diagonal `isConnected[i][i] == 1` means a city is connected to itself, so it does not need a merge.

The difference between Union Find and DFS/BFS is:

```text
DFS/BFS: find one entry point, then expand to the whole province.
Union Find: see one edge, then merge the two sets.
```

But they freeze the same core fact:

> A province is a connected component.

For complexity, this code still scans the upper triangle of the adjacency matrix, so the time complexity is:

```text
O(n^2)
```

Union Find with path compression makes `find/union` almost constant time. But in this problem, matrix scanning already costs `O(n^2)`, so total time is still dominated by the matrix.

Space complexity is the `parent` array:

```text
O(n)
```

At this checkpoint, we can:

- Understand province counting from the "merge connected sets" viewpoint.
- Write the full Union Find solution.
- Explain why scanning only the upper triangle is enough.
- Explain why one successful merge reduces the province count by one.

It still lacks:

- Compare the three methods and give a practical recommendation.
- Close with common mistakes so the reader does not solve the wrong problem again.

## Which Method Should You Use?

For this problem, learn DFS first.

The reason is simple: the essence of the problem is connected components, and DFS directly expresses "start from one city and expand to the whole province". Once you can write DFS, BFS is only replacing the recursion stack with a queue, and Union Find is only changing the viewpoint to set merging.

You can choose among the three methods like this:

| Method | When to use it | Core action |
|---|---|---|
| DFS | First time learning connected components, or writing the main interview solution quickly | Start from an unvisited city and recursively mark the whole province |
| BFS | You do not want recursion, or you want explicit control over the visiting queue | Start from an unvisited city and use a queue to mark the whole province |
| Union Find | You want to practice set merging, or the problem keeps giving connection relations | Every time you see a connection, merge the two cities' sets |

For LeetCode 547, DFS is usually the clearest answer. The constraint is only `n <= 200`, so recursion depth is at most 200 and is not a practical issue.

## Common Mistakes

First mistake: treating the number of `1`s in the matrix as the answer.

`1` only means direct connection. A province may be formed through indirect connections:

```text
0 -- 1 -- 2
```

These three cities form one province. The answer is not two edges and not three self-connections.

Second mistake: starting traversal only from city `0`.

If the graph is not fully connected, starting from `0` only finds the province containing city `0`. The outer scan is required:

```python
for city in range(n):
    if not visited[city]:
        provinces += 1
        dfs(city)
```

Third mistake: forgetting `visited`.

In an undirected graph, `0` can reach `1`, and `1` can go back to `0`. Without `visited`, DFS/BFS keeps walking back and forth.

Fourth mistake: incrementing the answer for an already visited city.

Only an unvisited city can start a new province:

```python
if not visited[city]:
    provinces += 1
```

If a city is already visited, it already belongs to a previous province and must not be counted again.

Fifth mistake: writing adjacency-list complexity for a matrix problem.

This problem gives an adjacency matrix. Even if the graph is sparse, the code scans matrix rows to check connections, so DFS, BFS, and Union Find can all be recorded as:

```text
Time complexity: O(n^2)
Space complexity: O(n)
```

## Final Recap

This problem can be compressed into one sentence:

> Scan all cities. Every time you find an unvisited city, you have found a new connected component. Use DFS or BFS to mark that entire component.

For DFS, the final idea is:

```text
visited records whether a city has already been assigned to a province
the outer loop scans every city
when an unvisited city appears, increment the province count
start DFS from that city and mark the whole province
```

For Union Find, the final idea is:

```text
at first, every city is its own province
whenever there is a connection, merge the two sets
each successful merge reduces the province count by one
the remaining set count is the answer
```

At this point, the tutorial has frozen three complete solutions:

- DFS: the main solution, and the one to learn first.
- BFS: the same traversal idea with an explicit queue.
- Union Find: the same connected component problem from a set-merging viewpoint.

The important thing to remember is not one specific template, but the translation:

```text
Number of provinces = number of connected components in an undirected graph
```
