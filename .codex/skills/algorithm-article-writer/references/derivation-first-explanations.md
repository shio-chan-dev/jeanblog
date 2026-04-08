# Derivation-First Explanations

Use this reference when an algorithm article should teach the method from
scratch instead of opening with the final conclusion.

## Core Rule

Do not open with “Use X algorithm” unless the article is intentionally a terse
reference note.

For tutorial-style articles, prefer:

1. concrete problem pressure
2. a tiny example or trace
3. naive baseline
4. bottleneck
5. key observation
6. final technique name
7. polished implementation

## Required Tutorial Flow

### 1. Start from a tiny task

- Show the smallest input or scenario where the method becomes interesting.
- Make the reader feel what must be decided at each step.

### 2. Show the baseline

- State the naive process in plain language.
- Give the baseline complexity if relevant.

### 3. Make the bottleneck visible

- Quantify what is recomputed, rechecked, or scanned wastefully.
- Show the exact pressure that forces a better method.

### 4. Name the key observation

- State the invariant, monotonicity, decomposition, or state meaning that
  unlocks the improved method.
- This is the bridge from “problem pain” to “algorithm idea.”

### 5. Only now name the technique

- After the reader can predict the shape of the solution, introduce the formal
  method or data structure.

### 6. Walk one concrete trace

- Follow one branch, one loop, one state table, or one queue evolution.
- Make state changes explicit.

## Adapter Questions By Algorithm Family

### Backtracking

- What does the partial answer mean?
- What choices are available next?
- When is the path complete?
- What must be undone?

### Dynamic Programming

- What smaller subproblem should be named?
- What does each state mean in plain language?
- What recurrence creates the next state?
- Why is the iteration order valid?

### Greedy

- What local choice is being made?
- Why is it safe?
- What counterexample breaks a tempting wrong choice?

### Sliding Window / Prefix Sum / Monotonic Structures

- What repeated work does the brute-force version perform?
- What running summary or invariant removes that repeated work?
- What boundary changes trigger updates?

## Anti-Patterns

Avoid:

- technique label in the first sentence
- code before the mental model
- unexplained state variables
- conclusion-first summaries with no derivation path

## Quick Check

Before delivery, confirm the article answers:

1. What is the baseline?
2. Why is it too slow or too fragile?
3. What observation changes the game?
4. How does that observation turn into the final method?
