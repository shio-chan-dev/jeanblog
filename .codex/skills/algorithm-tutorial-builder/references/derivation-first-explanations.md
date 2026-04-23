# Guided-Build Explanations

Use this reference when an algorithm tutorial should teach the method by
constructing it step by step instead of opening with the final conclusion.

## Core Rule

Do not open with “Use X algorithm” unless the article is intentionally a terse
reference note.

For derivation-first algorithm tutorials, prefer:

1. concrete problem pressure
2. a tiny example or trace
3. the first missing state
4. the first code fragment
5. the next missing rule
6. the next code fragment or module
7. an explicit “what this version can do now / what it still lacks” connector
8. one final runnable complete implementation or minimal complete demo

If a brute-force contrast or common wrong instinct is useful, place it inside
the relevant numbered step. Do not create a standalone bridge section for it.

## Required Guided-Build Flow

### 1. Start from a tiny task

- Show the smallest input or scenario where the method becomes interesting.
- Make the reader feel what must be decided at each step.
- This should be the first teaching section after front matter.
- Do not insert `Target Audience`, `Background`, `Core Concepts`, or a formula summary before it.

### 2. Introduce the first necessary state

- Explain what the partial answer or running summary must remember.
- Only then introduce the first variable or structure.

### 3. Add one small code fragment

- Add only the fragment justified by the previous reasoning step.
- Do not jump to the full function.

### 4. Add the next missing rule

- This may be a base case, transition, loop, or invariant update.
- Introduce it as the answer to one concrete question.
- If the named term or formula first becomes necessary here, introduce it here rather than in an earlier glossary section.

### 5. Walk one branch or state trace

- Follow the partial build on one real example.
- Make the state changes explicit.

### 6. Converge to one final implementation

- Combine the fragments into one runnable complete implementation, end-to-end module, or minimal complete demo.
- This is the first time the reader sees the whole code in one block.
- Do not add a second duplicated “reference answer” section after it.

### 7. Keep late steps truly incremental

- Early steps are not the only ones that must grow code.
- If Step 5, Step 6, or Step 7 introduces a real new module, class, or block, show the full current version of that unit after the addition.
- Do not let late steps collapse into architecture talk plus a tiny local fragment when the reader actually needs the integrated code for that stage.

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
- `target audience`, `background`, or `core concepts` as the first teaching section
- opening summary that previews all later components or modules
- standalone `naive idea` / `naive-to-optimized` section before the guided-build steps
- standalone glossary or formula sheet before the first build step
- code before the mental model
- unexplained state variables
- conclusion-first summaries with no derivation path
- full code before fragment-by-fragment construction
- separate “steps”, “implementation”, and “code” sections that repeat the same material
- duplicated `assembled code` / `reference answer` endings
- late-step pseudo-growth where a new module is announced but the actual integrated block only appears in the final code

## Quick Check

Before delivery, confirm the article answers:

1. What is the first state or fragment the reader needs?
2. What question does each step answer?
3. After each addition, what can the current build already do and what still remains?
4. How do the fragments converge to one final runnable complete implementation or demo?
5. Are any terms, formulas, or modules introduced before the pressure actually requires them?
6. Do late steps still show the current real code growth rather than reverting to high-level explanation?
