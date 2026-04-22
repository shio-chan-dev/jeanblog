# LeetCode Guided-Build Quality Checks

Use this reference when the user wants a LeetCode tutorial that feels truly incremental rather than “explain some concepts, then drop the final code”.

## Core Rule

Make the code grow in front of the reader.

For LeetCode writeups, the guided build should usually let the reader see:

1. the first skeletal recursion / loop / state frame,
2. the first missing rule,
3. the first correct but still naive version,
4. the bottleneck,
5. the optimization that directly fixes that bottleneck.

## Preferred Teaching Order

1. Problem evidence from the example or constraint.
2. Smallest runnable skeleton.
3. Meaning of the first state variable.
4. Completion condition.
5. First correctness rule or legality check.
6. First fully correct version, even if slow.
7. Why that version is slow or clumsy.
8. Optimized helper state or formula.
9. Slow trace of one branch.
10. `Assemble the Full Code`.
11. `Reference Answer`.

## LeetCode-Specific Guardrails

- Do not ask “why do we need `X`?” before the reader knows what `X` is.
- Do not introduce helper arrays, sets, maps, bitmasks, or DP tables before the simpler version exposes the need for them.
- Do not jump from the problem statement to the final trick in one paragraph.
- Do not let a later step silently replace an earlier code fragment without saying what changed.
- Do not let the polished reference answer contain logic the guided build never taught.

## Backtracking Adapter

For LeetCode backtracking posts, prefer this progression:

1. Define what one recursion layer means.
2. Record the current choice.
3. Define when one branch is complete.
4. Add the simplest legality check.
5. Show the first complete correct DFS.
6. Name the repeated work inside that checker.
7. Introduce helper state that removes that repeated work.
8. Reuse the same choose / recurse / undo skeleton.

## Quick Review Questions

Before delivery, confirm the article lets the reader answer these questions:

1. What does the current partial solution mean?
2. What is the first version of the code that is already correct?
3. What exact pain point makes the next helper state necessary?
4. What changed from the previous step?
5. Why does the final optimized code still feel like the same solution?
