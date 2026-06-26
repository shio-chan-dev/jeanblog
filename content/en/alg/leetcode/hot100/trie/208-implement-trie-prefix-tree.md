---
title: "LeetCode 208: Implement Trie (Prefix Tree) in Python"
date: 2026-06-25T13:58:00+08:00
draft: false
categories: ["LeetCode"]
tags: ["Hot100", "Trie", "prefix tree", "string", "LeetCode 208"]
description: "Solve LeetCode 208 in Python by deriving Trie node fields, children traversal, is_end, insert, search, startsWith, and the exact interface LeetCode expects."
keywords: ["LeetCode 208", "Implement Trie", "Prefix Tree", "Python Trie", "Trie", "startsWith", "is_end", "Hot100"]
---

> **Subtitle / Summary**
> LeetCode 208 is the fixed-interface version of the Trie template. We will derive the node fields and lookup invariant first, then fit them into `Trie()`, `insert`, `search`, and `startsWith`.

- **Reading time**: 10-12 min
- **Tags**: `Hot100`, `Trie`, `prefix tree`, `LeetCode 208`
- **SEO keywords**: LeetCode 208, Implement Trie, Prefix Tree, Python Trie, startsWith, is_end
- **Meta description**: A pressure-first Python guide to LeetCode 208 covering Trie nodes, children, is_end, insert, search, startsWith, and submission checks.

---

## A - Algorithm

### Problem setup: design a prefix-tree word index

The task is to design a mutable word index.
The index starts empty, then receives words one by one, and must answer two kinds of lookup:

- exact lookup: was this complete word inserted?
- prefix lookup: does any inserted word start with this prefix?

That gives the required operations:

| Operation | Meaning | Example |
| --- | --- | --- |
| `Trie()` | create an empty index | `trie = Trie()` |
| `insert(word)` | add a word to the index | `insert("apple")` |
| `search(word)` | check exact word existence | `search("app")` |
| `startsWith(prefix)` | check whether a prefix exists | `startsWith("ap")` |

The important part is that exact lookup and prefix lookup are not the same question.

### Smallest operation pressure

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
To insert a word, the structure must create one edge per character and let different words reuse the same prefix.

For example, `app` and `apple` should not be stored as two unrelated strings:

```text
app
apple
```

They should share the path for `a -> p -> p`.
That means each position in the structure should represent a prefix reached so far, and from that prefix we need to know which next characters are available.

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

It is a dictionary rather than a set because we do not only need to know that a next character exists.
We also need to move to the child node that represents the longer prefix after taking that character.

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

But if we try to answer exact word search using only path existence, the first attempt would look like this:

```python
def search_by_path_only(word: str) -> bool:
    node = self.root
    for ch in word:
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True
```

Now test it against the required behavior:

```text
insert("apple")
search_by_path_only("app") -> True
expected search("app")     -> False
expected startsWith("app") -> True
```

This is the exact pressure for the next field.
The path `a -> p -> p` really exists, so prefix lookup should succeed.
But the complete word `app` was not inserted yet, so exact lookup should fail.

### Step 2: Add an end marker for exact word search

To fix the wrong `search_by_path_only("app")` result, the final node of a complete word needs a marker.

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

- the two required public lookup methods

### Step 3: Implement search and startsWith directly first

Now use the state we already have.

For `search(word)`, walk the path and then check `is_end`:

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

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end
```

The `search` lookup invariant is:

> Before processing character `i`, `node` points to the node for `word[:i]`.

If the path is missing, the word does not exist.
If the path exists, the final node still must be marked as a complete word.

For `startsWith(prefix)`, walk the same kind of path, but do not check `is_end`:

```python
def startsWith(self, prefix: str) -> bool:
    node = self.root
    for ch in prefix:
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True
```

The `startsWith` lookup invariant is:

> Before processing character `i`, `node` points to the node for `prefix[:i]`.

If the path exists, the prefix exists.
It does not matter whether the final node is the end of a complete word.

Now the required behavior works:

```text
insert("apple")
search("apple")    -> True
search("app")      -> False
startsWith("app")  -> True
```

But the two methods clearly repeat the same path-walking loop.

### Step 4: Extract one path lookup helper

The repeated part is:

- start at `root`
- walk one character at a time
- fail if the next edge is missing
- otherwise return the final node reached by the query

Extract that repeated path lookup into `_find_node`:

```python
def _find_node(self, s: str):
    node = self.root
    for ch in s:
        if ch not in node.children:
            return None
        node = node.children[ch]
    return node
```

The helper invariant is the same path invariant, only with a neutral name:

> Before processing character `i`, `node` points to the node for `s[:i]`.

If a character is missing, the helper returns `None`.
If the loop finishes, it returns the node for the whole query string.

Then the two public methods become the two different interpretations of that returned node:

```python
def search(self, word: str) -> bool:
    node = self._find_node(word)
    return node is not None and node.is_end

def startsWith(self, prefix: str) -> bool:
    return self._find_node(prefix) is not None
```

`search` requires:

- the path exists
- the final node is marked as a complete word

`startsWith` requires only path existence.

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

## E - Engineering: Submission checks and extension points

### LeetCode interface checklist

Before submitting, check the contract rather than only the algorithm idea:

| Requirement | Check |
| --- | --- |
| Class name | `class Trie:` |
| Constructor | `def __init__(self):` creates the root |
| Insert | `insert` mutates the tree and returns `None` |
| Exact lookup | `search` returns `True` only when the final node has `is_end == True` |
| Prefix lookup | `startsWith` returns `True` when the path exists, without checking `is_end` |

The most common accepted-code bug is not traversal.
It is mixing up the final check for `search` and `startsWith`.

### Edge tests to run before submission

With the reference class above, this block should pass:

```python
trie = Trie()

trie.insert("apple")
assert trie.search("apple") is True
assert trie.search("app") is False
assert trie.startsWith("app") is True
assert trie.startsWith("apply") is False

trie.insert("app")
assert trie.search("app") is True
assert trie.startsWith("apple") is True
assert trie.search("appl") is False
```

These cases cover:

- a full word that exists
- a prefix that is not yet a full word
- a prefix that should pass `startsWith`
- a longer string whose path breaks
- inserting a word that is already a prefix of another word

### How later Trie problems extend this solution

LeetCode 208 stores only the minimal state:

```text
children: next character -> child node
is_end:  whether this path is a complete word
```

Later problems usually keep the same traversal but add one field:

| Need | Typical extra field |
| --- | --- |
| Count words ending at a node | `end_count` |
| Count words passing through a prefix | `pass_count` |
| Support delete | counts plus careful decrement logic |
| Prune DFS by prefix | reuse `_find_node` idea or walk Trie nodes during DFS |

Do not add those fields to LeetCode 208.
They are useful extensions, but this problem only asks for the basic template.

### FAQ

**Why not use a hash set for LeetCode 208?**
A hash set handles `search(word)`, but `startsWith(prefix)` would need scanning all words or storing all prefixes separately.
Trie stores shared prefixes directly.

**Why keep `_find_node` if LeetCode does not require it?**
It names the shared traversal used by `search` and `startsWith`.
The public interface stays exactly the same, and the helper keeps the two final checks from drifting apart.

**Should `startsWith` check `is_end`?**
No.
`startsWith("app")` should be true after inserting only `"apple"`, even before `"app"` itself is inserted.

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
- Changing the required method name from `startsWith` to `starts_with` in the LeetCode submission
- Adding counters or delete logic before the basic interface is correct

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
- Delete operation: keep counts so removing one word does not break another shared prefix
