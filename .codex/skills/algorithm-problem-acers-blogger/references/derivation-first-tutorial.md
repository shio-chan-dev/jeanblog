# Guided-Build Tutorial Reference

Use this reference when writing the teaching-heavy part of an algorithm-problem
post.

## Core Rule

Teach the reader how to *build* the solution from the problem itself.

Prefer this order:

1. evidence from the problem
2. a smaller manual example
3. the first necessary state
4. the first code fragment
5. the next missing rule
6. the next code fragment
7. one slow branch or state trace
8. assembled full code
9. reference answer

Do not reverse that order unless the user explicitly wants a terse reference
note instead of a tutorial.
Do not insert a standalone “naive idea”, “brute force first”, or
“naive-to-optimized” bridge section before the guided-build ladder. If that
contrast matters, place it inside Step 1 or the exact step that introduces the
missing state or rule.

## Required Guided-Build Ladder

For the tutorial section, walk through these steps in order whenever the
problem type allows.

Each step should do exactly one of these things:

- ask one concrete question
- answer it in plain language
- add one small code fragment or one state rule
- show one small state change

Do not let one step solve the whole problem at once.

### Step 1: Shrink to a tiny example
   - Use the smallest non-trivial input.
   - Manually enumerate 1-2 steps so the search space or bottleneck is visible.

### Step 2: Ask what information the partial solution must remember

- State what the partial answer means.
- Introduce only the first state variable that becomes necessary.
- Example: “That is why we need `path`.”

### Step 3: Define the smaller subproblem

- Ask: “If this partial state is already fixed, what remains to solve?”
- This is where recursion, transition, or state evolution becomes concrete.

### Step 4: Decide when the work is complete

- State exactly when the partial answer becomes complete, valid, or settled.
- Add the base case / completion check fragment here if code is appropriate.

### Step 5: Decide what can happen next

- List the valid next choices or transitions.
- Add the loop / transition fragment here.

### Step 6: Show how one move updates the state

- Show what changes when one choice is made.
- Add the “choose” fragment or update rule here.

### Step 7: Show what must be undone or carried forward

- If it is backtracking, show the undo fragment.
- If it is DP/BFS/greedy, show what summary, frontier, or invariant is carried forward.

### Step 8: Walk one branch or trace slowly

- Follow one concrete branch or state evolution end to end.
- This is the last place to slow down before presenting the complete solution.

### Step 9: Assemble the full code

- Combine the previously introduced fragments into one working implementation.
- This is the first full solution, not yet the final polished reference.

### Step 10: Present the reference answer

- Give the clean final answer after the assembly section.
- This version may be slightly cleaner, but it must not introduce new logic that was never explained.

## Required Section Labels

Use labels close to:

- `How To Build The Solution From Scratch`
- `Build It Step by Step`
- `Assemble the Full Code`
- `Reference Answer`

## Problem-Family Adapters

### Backtracking

Push these moves explicitly:

1. What does `path` represent?
2. Why do we need extra state like `used`?
3. When is `path` complete?
4. Which choices are still available?
5. What happens when we choose one?
6. What must be undone before trying the next choice?

Use the visible pattern:

```text
choose
recurse
undo
```

When code is introduced, prefer fragment order like:

```text
path = []
used = [...]

if complete:
    ...

for each candidate:
    if invalid:
        continue
    choose
    recurse
    undo
```

### Dynamic Programming

Push these moves explicitly:

1. What is the smallest subproblem worth naming?
2. What does `dp[i]` / `dp[i][j]` mean in plain language?
3. What base case must exist first?
4. What transition creates the next state?
5. Why is the iteration order valid?
6. Assemble the recurrence into code only after the state meaning is stable.

### Greedy

Push these moves explicitly:

1. What local decision are we making?
2. What tempting alternative should we reject?
3. Which counterexample breaks that alternative?
4. Why is the chosen local rule safe?
5. Then show the compact final loop.

### Graph / BFS / DFS

Push these moves explicitly:

1. What does one node/state represent?
2. What state container do we need first: stack, queue, or visited?
3. When is a node considered seen / done?
4. What does one expansion step mean?
5. Then assemble the traversal loop.

### Prefix Sum / Hash / Sliding Window

Push these moves explicitly:

1. What does the brute-force version recompute?
2. What running summary removes that repeated work?
3. What update rule changes the summary?
4. What query becomes O(1) after that?
5. Then assemble the final loop.

## Output Cues

Include a section such as:

- `How To Build The Solution From Scratch`
- `Build It Step by Step`
- `思路是怎么推出来的`

Place it before:

- `Assemble the Full Code`
- `Reference Answer`

## Anti-Patterns

Do not do these in tutorial mode:

- “This is a standard backtracking / DP / greedy problem” as the opening move
- a standalone `naive idea` / `naive-to-optimized` section before the guided-build steps
- code before the reasoning path
- unexplained variables like `used`, `path`, `dp`, `leftMax`
- one-paragraph jumps from statement to final formula
- naming a template without showing why it fits
- dumping the full code before the fragments have been introduced
- repeating the same content across “tutorial”, “steps”, and “implementation” in three separate sections

## Quick Check

Before delivery, confirm the article lets the reader answer these questions:

1. What does the partial state mean?
2. When is it complete or safe?
3. What choices or transitions are possible next?
4. What gets updated or undone after one move?
5. How do the small fragments combine into the first full code?
6. What is the final clean reference answer?
