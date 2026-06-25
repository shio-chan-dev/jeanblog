---
title: "LeetCode 208: Implement Trie (Prefix Tree) Template Guide"
date: 2026-06-25T13:58:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Trie", "prefix tree", "string", "LeetCode 208"]
description: "LeetCode 208 is the core Trie template problem: implement insert, search, and startsWith with the required class interface."
keywords: ["LeetCode 208", "Implement Trie", "Prefix Tree", "Trie", "startsWith"]
---

> **Subtitle / Summary**
> LeetCode 208 is not about a new trick. It asks you to fit the Trie template into a fixed interface: `insert` builds paths, `search` checks full words, and `startsWith` checks prefix paths.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `Trie`, `prefix tree`, `LeetCode 208`
- **SEO keywords**: LeetCode 208, Implement Trie, Prefix Tree, startsWith
- **Meta description**: A pressure-first guide to LeetCode 208 covering Trie nodes, children, is_end, insert, search, and startsWith.

---

## A - Algorithm

### Start with the smallest operation pressure

The key LeetCode 208 sequence is:

```text
Trie trie = new Trie()
trie.insert("apple")
trie.search("apple")    -> true
trie.search("app")      -> false
trie.startsWith("app")  -> true
trie.insert("app")
trie.search("app")      -> true
```

This example shows:

- `app` can be a prefix of `apple`
- `search("app")` should return `True` only after `app` itself has been inserted

So this problem is not solved by path existence alone.
We need to maintain both:

- whether a path exists
- whether that path is exactly a complete word

### Required interface

Design a Trie, also called a prefix tree, with three operations:

- `insert(word)`: insert a word
- `search(word)`: check whether a full word has been inserted
- `startsWith(prefix)`: check whether any inserted word starts with `prefix`

The platform requires this class and method shape:

```python
class Trie:
    def __init__(self):
        ...

    def insert(self, word: str) -> None:
        ...

    def search(self, word: str) -> bool:
        ...

    def startsWith(self, prefix: str) -> bool:
        ...
```

---

## Target Readers

- Learners who already understand the Trie template and want to submit LeetCode 208
- Readers who know hash-set word lookup but do not yet have a prefix-tree model
- Anyone trying to fix the boundary between `insert`, `search`, and `startsWith`

## Background / Motivation

LeetCode 208 is a pure template problem.
It does not require advanced pruning or prefix counting.
It checks whether you understand three basic questions:

- How do we create missing paths during insertion?
- Why is path existence not enough for full-word search?
- Why does prefix search not check `is_end`?

---

## C - Concepts

### Step 1: Fix the node structure

Each node needs two fields:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

- `children`: mapping from next character to child node
- `is_end`: whether the path from root to this node is a complete word

The problem only asks for a basic Trie, so no extra fields are needed.

### Step 2: insert builds missing paths

When inserting `word`, start at the root and scan each character.

Current loop invariant:

> Before processing character `i`, `node` points to the node for `word[:i]`, and that path already exists.

If the next character is missing, create it:

```python
if ch not in node.children:
    node.children[ch] = TrieNode()
node = node.children[ch]
```

After the loop, `node` points to the final node for `word`.
Then set:

```python
node.is_end = True
```

### Step 3: Extract path lookup

Both `search` and `startsWith` walk a string path.
They differ only in what they check after walking.

So use an internal helper:

```python
def _find_node(self, s: str):
    node = self.root
    for ch in s:
        if ch not in node.children:
            return None
        node = node.children[ch]
    return node
```

Lookup loop invariant:

> Before processing character `i`, `node` points to the node for `s[:i]`.

If a character is missing, the path is broken and the helper returns `None`.

### Step 4: search checks a full word

`search(word)` needs two conditions:

- the path for `word` exists
- the final node has `is_end == True`

So:

```python
node = self._find_node(word)
return node is not None and node.is_end
```

After inserting only `apple`, the path for `app` exists, but the `app` node is not marked as a full word.
Therefore `search("app")` must return `False`.

### Step 5: startsWith checks only the prefix path

`startsWith(prefix)` only cares whether the path exists:

```python
return self._find_node(prefix) is not None
```

It does not check `is_end`.
The prefix does not need to be a complete word; it only needs to be the beginning of some inserted word.

---

## Runnable Example (Python)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_node(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None


if __name__ == "__main__":
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.startsWith("app") is True
    trie.insert("app")
    assert trie.search("app") is True
```

### Reference Answer

For LeetCode submission, use the same implementation shape:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_node(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
```

This does not introduce new logic; it is the runnable version arranged as a submission-ready class.

---

## Explanation

### Why is LeetCode 208 a template problem?

The three methods cover the three basic Trie actions:

| Method | Meaning | Checks `is_end`? |
| --- | --- | --- |
| `insert` | create the path and mark the final node | set to `True` at the end |
| `search` | check a full-word path | yes |
| `startsWith` | check a prefix path | no |

Once these are stable, later Trie problems usually extend node fields:

- count how many words pass through a prefix
- maintain counts while deleting words
- use Trie as a pruning structure in DFS

### Why not use a hash set?

A hash set makes `search(word)` fast.
But `startsWith(prefix)` becomes awkward:

- scan all words
- or store every possible prefix separately

Trie naturally merges common prefixes, so prefix lookup only walks the characters of the prefix.

### The smallest counterexample for search vs startsWith

Insert only:

```text
apple
```

Then:

```text
search("app")     -> False
startsWith("app") -> True
```

This counterexample checks whether `is_end` is used correctly.

---

## R - Reflection

### Complexity

Let `L` be the string length:

- `insert`: `O(L)` time
- `search`: `O(L)` time
- `startsWith`: `O(L)` time
- space: `O(total_chars)`, more precisely the number of distinct prefix nodes

### Common mistakes

- Making `search` return `True` whenever the path exists
- Making `startsWith` check `is_end`, which rejects valid prefixes
- Rebuilding nodes even when a child already exists during insertion
- Forgetting to set `is_end` at the end of `insert`

### When should children be an array?

If the character set is fixed to lowercase English letters, you can write:

```python
self.children = [None] * 26
```

Then use `idx = ord(ch) - ord("a")` to access the child.
The array version has a smaller constant factor, but the index conversion can obscure the idea.
This guide uses `dict` because it directly expresses "character -> child node."

---

## S - Summary

- LeetCode 208 is the basic Trie template problem.
- The `insert` invariant is: `node` always points to the processed prefix, and missing paths are created.
- `search` requires path existence plus `is_end == True`.
- `startsWith` requires only prefix path existence.
- Once 208 is comfortable, Trie + DFS and prefix-counting problems become much easier.

### Further Practice

- Trie Template: Node Fields, Child Traversal, and Loop Invariants
- Word Search II: use Trie to prune impossible DFS branches
- Prefix-counting problems: add `pass_count` or `end_count` to nodes
