---
title: "Trie Template: Node Fields, Child Traversal, and Loop Invariants"
date: 2026-06-25T13:58:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Trie", "prefix tree", "string", "template"]
description: "Build a minimal Trie template from first principles: what each node stores, how children are traversed, why is_end matters, and the loop invariants behind insert and search."
keywords: ["Trie", "prefix tree", "children", "is_end", "string template", "Hot100"]
---

> **Subtitle / Summary**
> A Trie is not code to memorize. The key model is: a node represents a prefix reached by a path. Once that model is stable, insertion, full-word search, and prefix search become the same traversal with different final checks.

- **Reading time**: 8-10 min
- **Tags**: `Hot100`, `Trie`, `prefix tree`, `string`
- **SEO keywords**: Trie, prefix tree, children, is_end
- **Meta description**: A minimal Python Trie template with node fields, child traversal, end markers, and insert/search loop invariants.

---

## A - Algorithm: Start From A Small Task

### Tiny task: build a word index for lookup and autocomplete

Suppose we are building a small word index for a search box.
After adding:

```text
app
apple
bat
```

the index must answer two kinds of user-facing queries:

- `search("app")`: was `app` added as a complete word?
- `starts_with("ap")`: can autocomplete suggest words that begin with `ap`?
- `search("apply")`: should return false because this word was never added.

This is the smallest task that makes Trie useful:

- A hash set can answer full-word existence quickly, but it does not naturally answer prefix existence.
- A raw path check can confuse "this prefix exists" with "this full word exists."

A Trie solves both: it shares common prefixes for autocomplete while still distinguishing "prefix path exists" from "full word exists."

### Derive the operations from the pressure

Before caring about any platform interface, define the template directly:

- `insert(word)`: insert a word
- `search(word)`: check whether a full word exists
- `starts_with(prefix)`: check whether any inserted word starts with `prefix`

### Minimal structure picture

After inserting `app` and `apple`, the structure can be viewed as:

```text
root
 └─ a
    └─ p
       └─ p  [end]
          └─ l
             └─ e  [end]
```

The important points are:

- A node does not store the whole word.
- The path from `root` to a node forms a prefix.
- `[end]` means that this path is also a complete word.

---

## Target Readers

- Learners meeting Trie for the first time
- Readers who confuse `search` with prefix search
- Anyone preparing to implement LeetCode 208 after understanding the template

## Background / Motivation

If all we need is full-word membership, a hash set is enough.
But when the question becomes "does any word start with this prefix?", the hash set is no longer the natural structure.

Trie's value is that it does not store every string in isolation.
It compresses shared prefixes into shared paths.

---

## C - Concepts

### Build the template one capability at a time

The current pressure is the word index:

```text
add: app, apple, bat
ask: search("app"), starts_with("ap"), search("apply")
```

We will not jump straight to the final class.
Each step adds one missing capability to the current version.

### Step 1: Start with only outgoing edges

The first thing the index must remember is: from the current prefix, which next characters are possible?

That gives the smallest node:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
```

Here `children` is a dictionary:

```text
character -> child node
```

For example, after the root sees `"a"`, it can store:

```python
root.children["a"] = TrieNode()
```

Now this version can:

- represent a path from one prefix to the next prefix
- share the first character of `app` and `apple`

It still lacks:

- a root object that owns the whole index
- insertion logic that creates paths
- a way to tell full words from prefixes

### Step 2: Add the root and grow paths during insertion

The whole Trie needs one starting point.
That root represents the empty prefix `""`.

In the previous version, add a `Trie` class and an `insert` method:

```python
class TrieNode:
    def __init__(self):
        self.children = {}


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

The insertion loop invariant is:

> Before processing character `i`, `node` points to the node for prefix `word[:i]`, and the path for `word[:i]` already exists.

When we process `ch`, we create the missing child if needed, then move into it.

After inserting `apple`, the path exists:

```text
root -> a -> p -> p -> l -> e
```

Now this version can:

- create paths for inserted words
- share the `app` prefix between `app` and `apple`

It still lacks:

- a way to know whether a path is a complete word

This is exactly where the earlier task breaks:

```text
insert("apple")
search("app") should be False
starts_with("app") should be True
```

With paths only, both queries look the same.

### Step 3: Mark the end of a complete word

To separate a prefix from a complete word, each node needs one more field:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

Then `insert` marks only the final node:

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

This detail matters.
When inserting `apple`, we should not mark `a`, `ap`, `app`, or `appl` as full words.
Only the final `e` node represents the complete word `apple`.

Now this version can:

- store the difference between `apple` as a word and `app` as only a prefix

It still lacks:

- a lookup path that can answer whether a query reaches a node

### Step 4: Add one shared path lookup helper

Both full-word search and prefix search need the same traversal:

> Start at root, consume one character at a time, and stop if the path breaks.

In the previous version, add `_find_node`:

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

The lookup loop invariant is:

> Before processing character `i`, `node` points to the node for prefix `s[:i]`, and this prefix path has been found.

If a character is missing, the path is broken and `_find_node` returns `None`.
If the loop finishes, it returns the node for the whole query string.

Now this version can:

- tell whether a path exists
- return the exact node reached by a word or prefix

It still lacks:

- public methods that interpret that node differently for exact lookup and autocomplete

### Step 5: Split exact word search from prefix search

Now the final distinction becomes small.

For exact word lookup:

```python
def search(self, word: str) -> bool:
    node = self._find_node(word)
    return node is not None and node.is_end
```

For autocomplete prefix lookup:

```python
def starts_with(self, prefix: str) -> bool:
    return self._find_node(prefix) is not None
```

The difference is only the final check:

- `search` requires both path existence and `is_end`
- `starts_with` requires only path existence

Now this version can answer the original task:

```text
insert("app")
insert("apple")
search("app")        -> True
search("ap")         -> False
starts_with("ap")    -> True
search("apply")      -> False
```

At this point the full runnable code has been earned.

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

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None


if __name__ == "__main__":
    trie = Trie()
    trie.insert("app")
    trie.insert("apple")

    assert trie.search("app") is True
    assert trie.search("ap") is False
    assert trie.starts_with("ap") is True
    assert trie.search("apply") is False
```

---

## Explanation

### What does root mean?

`root` represents the empty prefix `""`.
Every word starts from the empty prefix and moves one character at a time.

That is why insertion and lookup both begin with:

```python
node = self.root
```

### Why use a dict for children?

In Python, `dict` is the clearest representation:

- key: next character
- value: child node
- lookup and insertion are average `O(1)`

If the problem guarantees lowercase English letters only, a 26-element array is also possible.
The array has a smaller constant factor, but index conversion can distract from the idea.
For a learning template, `dict` is more direct.

### is_end is set only after the last character

When inserting `apple`, we should not mark `a`, `ap`, `app`, and `appl` as full words.
Only after the loop finishes does the current node represent the whole word.

If we later insert `app`, we walk to the same `app` node and set its `is_end` to `True`.

---

## R - Reflection

### Complexity

Let `L` be the string length:

- `insert`: `O(L)` time, at most `L` new nodes
- `search`: `O(L)` time, `O(1)` extra space
- `starts_with`: `O(L)` time, `O(1)` extra space

Total space depends on the number of distinct prefixes across all inserted words.
The more prefixes are shared, the more useful the Trie structure becomes.

### Common mistakes

- Keeping only `children` and forgetting `is_end`, which treats prefixes as full words
- Continuing after a missing character during lookup
- Creating a fresh node every time during insertion and overwriting shared prefixes
- Making `starts_with` require `is_end`, which rejects valid prefixes

### Template memory hook

You can compress Trie into three sentences:

- A node means "we reached this prefix."
- `children[ch]` means "append character `ch` and move to that node."
- `is_end` means "this prefix is exactly a full word."

---

## S - Summary

- Trie is path-based; the character lives on the edge, not necessarily inside the node.
- `children` moves to the next prefix; `is_end` separates prefixes from full words.
- The insert and lookup invariants both say: `node` always points to the currently processed prefix.
- After this template is stable, LeetCode 208 is mostly a fixed-interface version of the same code.

### Further Practice

- `208. Implement Trie (Prefix Tree)`: fit this template into the required class and methods
- Word Search II: Trie + DFS pruning
- Prefix-counting problems: add counters such as `pass_count` or `end_count` to nodes
