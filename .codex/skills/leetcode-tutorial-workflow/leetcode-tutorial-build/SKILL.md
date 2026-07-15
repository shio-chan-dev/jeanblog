---
name: leetcode-tutorial-build
description: v0.1.2 - Build exactly one planned LeetCode tutorial increment through problem pressure, naive baseline, break, change, check, freeze, concept timing, and review handoff. Use when a task-first LeetCode tutorial plan is ready and the next checkpoint must be drafted without self-approval.
---

# LeetCode Tutorial Build

## Overview

Write one planned tutorial task, not the whole article. This skill drafts the
next checkpoint in a problem tutorial through a fixed increment cycle, then
stops for `leetcode-tutorial-review`.

Build may run self-checks, but self-checks are not acceptance. The next step is
blocked until an independent review returns `pass`.

## When to Use

- A `leetcode-tutorial-plan` task list exists.
- The user asks to draft the next LeetCode tutorial step.
- A problem tutorial needs one connected code checkpoint added to the current
  document.

**When NOT to use:** planning, reviewing, simplifying, standalone algorithm
concept tutorials, writing a full multi-step article in one pass, or enhancing
an already stable post.

## The Tutorial Increment Cycle

```text
+------------------------------------------------+
|                                                |
|  Pressure -> Naive baseline -> Break -> Change |
|      ^                                |        |
|      +------ Freeze <- Check <--------+        |
|                 |                              |
|                 v                              |
|          Review handoff                        |
|                 |                              |
|                 v                              |
|        Next step after review_pass             |
|                                                |
+------------------------------------------------+
```

For each LeetCode tutorial increment:

1. **Pressure** - show the problem example, failing case, trace, bottleneck,
   or constraint that makes the current baseline insufficient.
2. **Naive baseline** - show what the reader currently has: problem facts,
   current code, current recurrence, current trace, or current mental model.
3. **Break** - name exactly what fails: wrong answer, missing case, duplicate
   work, timeout risk, unclear state meaning, or unproved transition.
4. **Change** - add or replace one thing: a state variable, branch, recurrence,
   helper, loop, pruning rule, or code block.
   Any new variable, helper, invariant, recurrence, formula, or rule must be
   forced by the current break and either be operationally used in this
   checkpoint or explicitly marked as named-only until a later checkpoint.
5. **Check** - prove this change against the pressure with a runnable command,
   assertion, hand trace, table, or example walkthrough.
6. **Freeze** - state what this checkpoint can now do, what it still lacks, and
   stop with `needs_review`.

Write and self-check one numbered increment before starting another. Do not
draft multiple tutorial steps and then retrofit checks afterward. The next
increment is blocked until `leetcode-tutorial-review` returns `pass`.

## Build Contract

Use this contract when the tutorial should feel like code is being forced into
existence step by step, not explained after the final answer is already known.

### Core Rule

The tutorial must grow through connected code versions.

Before code growth starts, the reader must know the problem. The first body
section after front matter must state:

- what the input gives
- what output is required
- whether order, uniqueness, continuity, or other constraints matter
- at least one concrete example
- the relevant constraints when available

Only after this problem block should the tutorial move to the tiny example,
conflict pattern, smaller subproblem, or first code skeleton.

Do not write:

- explanation fragment
- unrelated code fragment
- another explanation fragment
- final answer

Write instead:

- one problem
- one code change
- one new capability
- one remaining gap

### Required Step Shape

For each numbered step, prefer this shape:

```text
### Step X: <one concrete problem>

Ask one concrete question.

Explain why this problem must be solved now.

Show the naive or previous baseline the reader currently has.

Name exactly what breaks in that baseline.

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

The guided build should repeatedly use these connectors in substance:

1. `The current baseline is ...`
2. `This breaks when ...`
3. `In the previous version, add ...`
4. `Replace this part with ...`
5. `Check this change with ...`
6. `Now this version can ...`
7. `It still lacks ...`

If these connectors are absent, the tutorial will usually drift back into
explanation mode.

### Concept Timing Rule

Do not introduce a variable, helper, invariant, recurrence, formula, or rule
just because the final answer will need it. A named concept must follow this
sequence:

```text
pressure -> name only if needed -> meaning -> first operational use -> capability claim
```

Operational use means the concept participates in a condition, update,
recurrence, helper call, return value, proof step, or decision. Before that
first operational use, the checkpoint may say only:

- the reader knows what the concept represents
- the reader knows why the concept will be needed
- the next checkpoint will show how it is used

It must not say:

- the reader can use the concept
- the algorithm can solve a case with that concept
- this version can apply the concept

If a concept would be named in one checkpoint but not used until the next,
prefer delaying the name. If delaying would make the prose awkward, explicitly
write the `still lacks` gap as "the first operational use of X".

### Growth Landmarks

When the problem type allows it, include:

1. a front-loaded problem requirement, input-output, example, and constraints
   block
2. a tiny example that exposes the conflict or bottleneck
3. an explicit smaller-subproblem statement
4. the smallest runnable skeleton
5. the first partial-state variable
6. the completion rule
7. the first complete correct version
8. at least one middle version if optimization is staged
9. the final optimized version
10. a slow branch trace
11. one runnable complete code version

Do not treat `Assemble the Full Code` or `Reference Answer` as mandatory
landmarks. They are optional only when they add real teaching value.

### Anti-Patterns

Avoid these:

- introducing `diag1`, `used`, `prefix`, `dp`, or similar helper state before
  the reader sees what pain it removes
- defining `current_end`, `dp`, `visited`, `stack`, `heap`, or similar state in
  one checkpoint and claiming it is usable before a condition, update,
  recurrence, return value, or proof step actually uses it
- opening with derivation, target-audience prose, background, or a tiny
  search/DP trace before the problem requirement is clear
- placing the actual problem statement after the first derivation section
- explaining the final design as if the code had already been written
- making steps numerically sequential but not code-sequential
- jumping from the first correct version to the final optimized version
  without an intermediate build when the bridge is large
- adding a duplicate full-code section after the last step already produced a
  runnable complete solution
- using `Reference Answer` to smuggle in unexplained logic

### Full-Code Policy

Default policy:

- the last meaningful incremental step should already yield a runnable full
  solution
- the tutorial should usually end there

Only add a separate full-code or reference section if one of these is true:

- earlier steps never showed one complete runnable version
- the user explicitly wants a platform wrapper such as LeetCode
  `class Solution`
- the wrapper form has real delivery value and still contains no new logic

### Family Adapters

Backtracking progression:

1. tiny conflict example
2. what one recursion layer means
3. what one choice means
4. when a branch is complete
5. first legality check
6. first correct DFS
7. first helper-state optimization
8. final helper-state optimization

Dynamic programming progression:

1. tiny example
2. smaller subproblem meaning
3. base case
4. first correct transition
5. first full correct table or recurrence
6. one optimization at a time

Graph progression:

1. tiny example graph or state
2. what one node or state means
3. what container we need
4. first correct traversal
5. optimization or pruning after the basic traversal works

## The Build Loop

1. Load Current Task
   - Read the task description, dependencies, acceptance criteria,
     verification, document target, code change role, and review gate.
   - Confirm previous tasks have review-pass evidence when they are
     dependencies.
   - Verify: exactly one task is in scope.
2. Confirm Problem Facts
   - Ensure the tutorial already states input, required output, examples, and
     relevant constraints before code growth starts.
   - If problem facts are missing, add or request them before drafting the
     increment.
   - Verify: the reader knows the problem before seeing derivation or helper
     state.
3. Create Pressure
   - Show the concrete example, failing case, trace, bottleneck, or constraint
     that forces this task.
   - Prefer a tiny input or table over a broad claim.
   - Verify: the pressure is visible before the fix appears.
4. Re-state the Naive Baseline
   - Identify what the reader already has from prior checkpoints.
   - Verify: the step starts from visible state, not final-answer memory.
5. Name the Break
   - Say exactly what the baseline cannot handle.
   - Use concrete failure language: wrong answer, missed case, duplicated
     work, timeout risk, unclear state, or unproved transition.
   - Verify: the break follows from the baseline, not from hidden final
     knowledge.
6. Make One Change
   - Add or replace exactly one state, rule, helper, recurrence, branch, or
     code block.
   - Keep code connected to the previous checkpoint.
   - Check concept timing: each new name must be operationally used now or
     explicitly described as named-only with its first use deferred.
   - Verify: the increment introduces one visible mechanism and no unrelated
     future logic.
7. Add Check Evidence
   - Add an assertion, trace, manual example walk-through, or command that
     verifies this one change.
   - Execute the runnable check when execution is feasible.
   - If the check is conceptual or manual, record concrete evidence, not a
     vague claim.
   - Verify: check evidence targets the named break.
8. Freeze and Stop
   - State what the checkpoint can now do and what it still lacks.
   - If a concept was only named but not yet operationally used, freeze may
     claim only that its meaning is known; `still lacks` must name its first
     operational use.
   - Output `needs_review`.
   - Do not continue to the next planned task.
   - Do not call the checkpoint accepted.
   - Verify: the next step can only start after review pass.

## Review Gate Contract

Every build output must end with:

```text
status: needs_review
review_skill: leetcode-tutorial-review
review_scope: <task / section / file>
self_checks:
- <check actually run or manual evidence>
next_step_blocked_until: review_pass
```

If a previous task does not have review-pass evidence, stop and ask for review
instead of writing the next task.

## Decision Points

- If no plan exists, use `leetcode-tutorial-plan` first.
- If the task asks for more than one numbered step, split it before writing.
- If the problem facts are incomplete, stop instead of inventing behavior.
- If the current draft is structurally broken, request review before adding
  more content.
- If the user explicitly asks for a full article in one pass, explain that this
  workflow requires checkpointed steps unless they opt out.

## Output Format

```markdown
## Build Result
- status: needs_review
- task:
- document_target:
- code_change_role:

## Drafted Content
<the single tutorial step or document section>

## Self-Checks
- ...
- increment_cycle:
  - pressure:
  - naive_baseline:
  - break:
  - change:
  - check:
  - freeze:

## Review Handoff
- review_skill: leetcode-tutorial-review
- review_scope:
- next_step_blocked_until: review_pass
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "I already ran checks, so I can continue." | Checks are evidence; review is the acceptance gate. |
| "The next step is obvious." | Continuing without review recreates writer-as-judge failure. |
| "One article pass is faster." | LeetCode tutorials fail when hidden leaps compound across steps. |
| "The final code will prove it." | Final code does not prove each teaching checkpoint was earned. |
| "The baseline is implied by the previous section." | The build step must make the current baseline visible before naming the break. |
| "The check can be added later." | Without a check, the freeze is not a trustworthy checkpoint. |

## Red Flags

- Build writes more than one planned task.
- Build says `pass`, `accepted`, or `done` without review.
- Step check is described but not executed or evidenced.
- The step introduces a final trick before the pressure appears.
- The step has a pressure and change but no explicit naive baseline.
- The step names a technique without showing what the current baseline cannot
  do.
- The step claims the reader can use a variable, helper, invariant, formula, or
  rule whose first operational use appears only in a later step.
- The freeze does not say what still lacks.
- The output proceeds to the next step after `Checkpoint`.

## Verification

- [ ] Exactly one task was drafted.
- [ ] The increment follows
      `Pressure -> Naive baseline -> Break -> Change -> Check -> Freeze`.
- [ ] Problem pressure appears before the change.
- [ ] The previous baseline and break are explicit.
- [ ] One change was added or replaced.
- [ ] New variables, helpers, invariants, recurrences, formulas, and rules obey
      concept timing and are not over-claimed before first operational use.
- [ ] Check evidence is concrete and tied to the break.
- [ ] Output ends with `needs_review`.
- [ ] The next step is blocked until `leetcode-tutorial-review` returns `pass`.

## Guardrails

- Do not self-approve.
- Do not continue to the next step.
- Do not invent problem facts.
- Do not add detached final code.
- Do not claim capability for a concept that has only been named.
- Do not hide missing runnable checks behind generic "verified" language.
- Do not collapse pressure, baseline, and break into one vague motivation
  paragraph.
