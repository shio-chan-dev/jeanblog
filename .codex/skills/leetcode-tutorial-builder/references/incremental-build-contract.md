# Incremental Build Contract

Use this reference when the user wants a tutorial that feels like the code is being forced into existence step by step, not merely explained after the fact.

## Core Rule

The tutorial must grow through connected code versions.

Before code growth starts, the reader must know the problem.

The first body section after front matter must state:

- what the input gives,
- what output is required,
- whether order, uniqueness, continuity, or other constraints matter,
- at least one concrete example,
- the relevant constraints when available.

Only after this problem block should the tutorial move to the tiny example,
conflict pattern, smaller subproblem, or first code skeleton.

Do not write:

- explanation fragment,
- unrelated code fragment,
- another explanation fragment,
- final answer.

Write instead:

- one problem,
- one code change,
- one new capability,
- one remaining gap.

## Required Step Template

For each numbered step, prefer this shape:

```text
### Step X: <one concrete problem>

Ask one concrete question.

Explain why this problem must be solved now.

In the previous version, add:
<small snippet>

or:

Replace this part of the previous version:
<old shape described briefly>

with:
<small snippet>

Now this version can:
- ...

It still lacks:
- ...
```

## Mandatory Connectors

The guided build should repeatedly use these connectors in substance:

1. `In the previous version, add ...`
2. `Replace this part with ...`
3. `Now this version can ...`
4. `It still lacks ...`

If these connectors are absent, the tutorial will usually drift back into explanation mode.

## Required Growth Landmarks

When the problem type allows it, the tutorial should contain:

1. a front-loaded problem requirement / input-output / example / constraints block,
2. a tiny example that exposes the conflict or bottleneck,
3. an explicit smaller-subproblem statement,
4. the smallest runnable skeleton,
5. the first partial-state variable,
6. the completion rule,
7. the first complete correct version,
8. at least one middle version if optimization is staged,
9. the final optimized version,
10. a slow branch trace,
11. one runnable complete code version.

Do not treat `Assemble the Full Code` or `Reference Answer` as mandatory landmarks.
They are optional only when they add real teaching value.

## Anti-Patterns

Avoid these:

- introducing `diag1`, `used`, `prefix`, `dp`, or similar helper state before the reader sees what pain it removes,
- opening with derivation, target-audience prose, background, or a tiny search/DP trace before the problem requirement is clear,
- placing the actual problem statement after the first derivation section,
- explaining the final design as if the code had already been written,
- making steps numerically sequential but not code-sequential,
- jumping from the first correct version to the final optimized version without an intermediate build when the bridge is large,
- adding a duplicate full-code section after the last step already produced a runnable complete solution,
- using `Reference Answer` to smuggle in unexplained logic.

## Full-Code Policy

Default policy:

- the last meaningful incremental step should already yield a runnable full solution,
- the tutorial should usually end there.

Only add a separate full-code or reference section if one of these is true:

- earlier steps never showed one complete runnable version,
- the user explicitly wants a platform wrapper such as LeetCode `class Solution`,
- the wrapper form has real delivery value and still contains no new logic.

## Family Adapters

### Backtracking

Prefer this progression:

1. tiny conflict example,
2. what one recursion layer means,
3. what one choice means,
4. when a branch is complete,
5. first legality check,
6. first correct DFS,
7. first helper-state optimization,
8. final helper-state optimization.

### Dynamic Programming

Prefer this progression:

1. tiny example,
2. smaller subproblem meaning,
3. base case,
4. first correct transition,
5. first full correct table/recurrence,
6. one optimization at a time.

### Graph

Prefer this progression:

1. tiny example graph/state,
2. what one node/state means,
3. what container we need,
4. first correct traversal,
5. optimization or pruning after the basic traversal works.
