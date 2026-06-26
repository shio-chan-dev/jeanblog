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

### Build the required interface one behavior at a time

The platform asks for:

```python
insert(word)
search(word)
startsWith(prefix)
```

Those methods are not arbitrary names.
They come from the operation trace:

```text
insert("apple")
search("apple")    -> True
search("app")      -> False
startsWith("app")  -> True
```

We will build only the state needed to make these answers correct.

### Step 1: Start with paths for inserted words

The first behavior is `insert(word)`.
To insert a word, the structure must create one edge per character.

So start with the smallest node:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
```

Here `children` is a dictionary:

```text
character -> child node
```

Now add the root and the first version of `insert`:

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
```

This version can create the path for `apple`:

```text
root -> a -> p -> p -> l -> e
```

The loop invariant is:

> Before processing character `i`, `node` points to the node for `word[:i]`, and that path already exists.

This version can:

- make inserted words reachable as paths
- share prefixes between words

It still fails the trace:

```text
insert("apple")
search("app") should be False
startsWith("app") should be True
```

With only paths, `app` looks like it exists because the path `a -> p -> p` is present.

### Step 2: Add an end marker for exact word search

To make `search("app")` different from `startsWith("app")`, the final node of a complete word needs a marker.

Add `is_end` to each node:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

Then mark only the final node during insertion:

```python
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
```

After `insert("apple")`, only the `e` node is marked as a complete word.
The `app` node exists as a prefix, but it is not marked.

This version can:

- store the difference between a complete word and a prefix

It still lacks:

- a lookup operation that returns the node reached by a query

### Step 3: Extract one path lookup helper

Both `search` and `startsWith` walk a string path.
They differ only in what they check after walking.

Add `_find_node`:

```python
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
```

The lookup invariant is:

> Before processing character `i`, `node` points to the node for `s[:i]`.

If a character is missing, the helper returns `None`.
If the loop finishes, it returns the node for the whole query string.

This version can:

- tell whether a path exists
- recover the final node for a word or prefix

It still lacks:

- public methods that interpret the returned node according to the required interface

### Step 4: Implement search and startsWith from the same helper

Now `search(word)` is just:

```python
def search(self, word: str) -> bool:
    node = self._find_node(word)
    return node is not None and node.is_end
```

It requires:

- the path exists
- the final node is marked as a complete word

And `startsWith(prefix)` is:

```python
def startsWith(self, prefix: str) -> bool:
    return self._find_node(prefix) is not None
```

It requires only path existence.

Now the original trace works:

```text
insert("apple")
search("apple")    -> True
search("app")      -> False
startsWith("app")  -> True
insert("app")
search("app")      -> True
```

The complete runnable class is now earned.

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
