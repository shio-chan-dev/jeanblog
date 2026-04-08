# Derivation-First Tutorial Reference

Use this reference when writing the teaching-heavy part of an algorithm-problem
post.

## Core Rule

Teach the reader how to *arrive at* the solution from the problem itself.

Prefer this order:

1. evidence from the problem
2. a smaller manual example
3. the needed state or subproblem
4. the transition / choice rule
5. the stop condition or invariant
6. the final named technique
7. the polished code

Do not reverse that order unless the user explicitly wants a terse reference
note instead of a tutorial.

## Required Tutorial Ladder

For the tutorial section, walk through these questions in order whenever the
problem type allows:

1. **Shrink to a tiny example**
   - Use the smallest non-trivial input.
   - Manually enumerate 1-2 steps so the search space or bottleneck is visible.
2. **Define the partial answer**
   - State what a partial solution means.
   - Name the exact state the reader must track.
3. **Define the smaller subproblem**
   - Ask: “If this partial state is already fixed, what remains to solve?”
4. **Define the completion condition**
   - State exactly when the partial answer becomes complete, valid, or settled.
5. **Define the next choices or transitions**
   - List what can happen next from the current state.
6. **Explain state updates**
   - Show what changes when one choice is made.
   - If it is backtracking, show what must be undone.
   - If it is DP/greedy/BFS, show what must be carried forward.
7. **Walk one branch / trace slowly**
   - Follow one concrete branch or state evolution end to end.
8. **Only then present the final algorithm**
   - After the reader can predict the process, show the final compact version.

## Problem-Family Adapters

### Backtracking

Push these questions explicitly:

- What does `path` represent?
- Which choices are still available?
- When is `path` complete?
- What state must be undone before trying the next choice?

Use the visible pattern:

```text
choose
recurse
undo
```

### Dynamic Programming

Push these questions explicitly:

- What is the smallest subproblem worth naming?
- What does `dp[i]` / `dp[i][j]` mean in plain language?
- What transition creates the current state from earlier states?
- What are the base cases?
- Why is the iteration order valid?

### Greedy

Push these questions explicitly:

- What local decision is being made?
- Why is that local decision safe?
- What tempting alternative fails on a counterexample?
- What invariant stays true after each greedy step?

### Graph / BFS / DFS

Push these questions explicitly:

- What does one node/state represent?
- What is currently in the frontier / stack / queue?
- When is a node “done” or “visited”?
- What does one expansion step mean in the problem domain?

### Prefix Sum / Hash / Sliding Window

Push these questions explicitly:

- What does the naive scan recompute wastefully?
- What running summary should be carried forward?
- What lookup answers the current position instantly?
- What condition expands or shrinks the window?

## Output Cues

Include a section such as:

- `How To Build The Solution From Scratch`
- `Step-by-Step Derivation`
- `思路是怎么推出来的`

Place it before the final compact algorithm summary or reference implementation.

## Anti-Patterns

Do not do these in tutorial mode:

- “This is a standard backtracking / DP / greedy problem” as the opening move
- code before the reasoning path
- unexplained variables like `used`, `path`, `dp`, `leftMax`
- one-paragraph jumps from statement to final formula
- naming a template without showing why it fits

## Quick Check

Before delivery, confirm the article lets the reader answer these questions:

1. What does the partial state mean?
2. When is it complete or safe?
3. What choices or transitions are possible next?
4. What gets updated or undone after one move?
5. Why does this process lead to the final algorithm?
