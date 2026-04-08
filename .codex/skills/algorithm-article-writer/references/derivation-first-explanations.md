# Guided-Build Explanations

Use this reference when an algorithm article should teach the method by
constructing it step by step instead of opening with the final conclusion.

## Core Rule

Do not open with “Use X algorithm” unless the article is intentionally a terse
reference note.

For tutorial-style articles, prefer:

1. concrete problem pressure
2. a tiny example or trace
3. the first missing state
4. the first code fragment
5. the next missing rule
6. the next code fragment
7. assembled full code
8. clean reference answer

## Required Guided-Build Flow

### 1. Start from a tiny task

- Show the smallest input or scenario where the method becomes interesting.
- Make the reader feel what must be decided at each step.

### 2. Introduce the first necessary state

- Explain what the partial answer or running summary must remember.
- Only then introduce the first variable or structure.

### 3. Add one small code fragment

- Add only the fragment justified by the previous reasoning step.
- Do not jump to the full function.

### 4. Add the next missing rule

- This may be a base case, transition, loop, or invariant update.
- Introduce it as the answer to one concrete question.

### 5. Walk one branch or state trace

- Follow the partial build on one real example.
- Make the state changes explicit.

### 6. Assemble the full code

- Combine the fragments into one complete implementation.
- This is the first time the reader sees the whole code in one block.

### 7. Give the reference answer

- Present the clean final version after the assembly.
- The reference answer may be cleaner, but it must not depend on unexplained logic.

## Adapter Questions By Algorithm Family

### Backtracking

- What is the goal we are filling one slot at a time?
- What does the partial answer mean?
- Why do we need `used` or equivalent extra state?
- What choices are available next?
- When is the path complete?
- What must be undone?

### Dynamic Programming

- What smaller subproblem should be named?
- What does each state mean in plain language?
- What base case must exist before transitions make sense?
- What recurrence creates the next state?
- Why is the iteration order valid?

### Greedy

- What local choice is being made?
- Why is it safe?
- What counterexample breaks a tempting wrong choice?

### Sliding Window / Prefix Sum / Monotonic Structures

- What repeated work does the brute-force version perform?
- What running summary or invariant removes that repeated work?
- What is the first variable we need to introduce?
- What boundary changes trigger updates?

## Anti-Patterns

Avoid:

- technique label in the first sentence
- code before the mental model
- unexplained state variables
- conclusion-first summaries with no derivation path
- full code before fragment-by-fragment construction
- separate “steps”, “implementation”, and “code” sections that repeat the same material

## Quick Check

Before delivery, confirm the article answers:

1. What is the first state or fragment the reader needs?
2. What question does each step answer?
3. How do the fragments become full code?
4. What is the final clean reference answer?
