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

### Tiny task: answer full-word and prefix questions

Suppose we have inserted:

```text
app
apple
bat
```

Now we need to answer:

- Is `app` a full word?
- Is `ap` a prefix of any inserted word?
- Does `apply` exist?

This tiny task exposes two missing capabilities:

- A hash set can answer full-word existence quickly, but it does not naturally answer prefix existence.
- A raw path check can confuse "this prefix exists" with "this full word exists."

A Trie solves both: it shares common prefixes while still distinguishing "prefix path exists" from "full word exists."

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

### Step 1: What does each node store?

A minimal Trie node needs two fields:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

Field meaning:

- `children`: possible next characters from the current prefix
- `is_end`: whether the path from the root to this node is a complete word

Usually the node does not need to store its own character.
That character is already represented by the edge used from its parent.

### Step 2: How do children move?

Suppose the current node represents prefix `ap`, and the next character is `p`:

```python
node = node.children["p"]
```

The meaning is:

> The current prefix changes from `ap` to `app`.

If `"p"` is not in `children`, then this path does not exist.
During insertion, create the missing node.
During lookup, fail immediately.

### Step 3: Why do we need is_end?

Path existence alone is not enough.

After inserting `apple`, the path `a -> p -> p` definitely exists.
But that does not mean `app` has been inserted as a word.

So:

- `starts_with("app")` only needs the path to exist
- `search("app")` also needs the final node to have `is_end == True`

That is the purpose of `is_end`: it separates "only a prefix" from "a complete word."

### Step 4: The insertion loop invariant

When inserting `word`, before processing character `i`, keep this invariant:

> `node` points to the node for prefix `word[:i]`, and the path for `word[:i]` already exists.

Process the current character `ch = word[i]`:

```python
if ch not in node.children:
    node.children[ch] = TrieNode()
node = node.children[ch]
```

After the loop:

> `node` points to the node for the whole `word`.

Now set `node.is_end = True`.
That marks this path as a complete word, not just a prefix.

### Step 5: The lookup loop invariant

When looking up `word` or `prefix`, before processing character `i`, keep this invariant:

> `node` points to the node for prefix `s[:i]`, and this prefix path has been found.

Process the current character:

```python
if ch not in node.children:
    return None
node = node.children[ch]
```

If a character is missing, the path is broken.
If all characters are consumed, return the last node.

That lets us extract `_find_node(s)`:

- missing path: return `None`
- found path: return the node for the last character

Then `search` and `starts_with` differ only in the final check:

```python
node = self._find_node(word)
return node is not None and node.is_end
```

```python
return self._find_node(prefix) is not None
```

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
