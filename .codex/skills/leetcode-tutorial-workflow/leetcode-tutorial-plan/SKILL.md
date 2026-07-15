---
name: leetcode-tutorial-plan
description: v0.1.1 - Plan one LeetCode or OJ-style tutorial as ordered, review-gated writing tasks before drafting. Use when a concrete problem needs a publishable guided-build tutorial with problem facts, examples, constraints, smaller subproblem, connected code checkpoints, checkpoint check requirements, concept timing, optimization bridge, verification, and checkpoint handoff.
---

# LeetCode Tutorial Plan

## Overview

Plan one problem tutorial before writing it. The output is not the tutorial
body. It is a task-first build plan that lets `leetcode-tutorial-build` write
one step at a time and lets `leetcode-tutorial-review` independently decide
whether that step may advance.

The plan preserves this tutorial path:

```text
problem facts
-> tiny example pressure
-> smaller subproblem
-> first correct baseline
-> optimization pressure
-> final runnable solution checkpoint
```

## When to Use

- The user supplied one concrete LeetCode, Hot100, Codeforces, AtCoder, Luogu,
  or OJ-style problem.
- The tutorial needs a teaching route before drafting.
- The problem requires staged reasoning, a first correct version, or an
  optimization bridge.
- The work should be checkpointed step by step instead of written in one pass.

**When NOT to use:** standalone algorithm concept tutorials, broad technique
explainers, post enhancement, SEO work, or final-code-only answers.

## The Planning Loop

1. Capture Problem Facts
   - Record statement, input, output, examples, constraints, and supplied
     platform wrapper requirements.
   - Separate supplied facts from inferred assumptions.
   - Verify: the first tutorial body section can be written without inventing
     problem behavior.
2. Classify the Problem Family
   - Identify likely family such as two pointers, binary search, backtracking,
     DP, graph, greedy, stack, heap, or simulation.
   - Treat the family as a hypothesis, not a template to reveal too early.
   - Verify: the first pressure still comes from the problem evidence.
3. Choose Path and Taxonomy
   - Pick path, slug, language, implementation language, front matter, tags,
     and category from repository conventions.
   - Verify: path is under the appropriate `content/<lang>/alg/...` lane.
4. Define Teaching Dependency Graph
   - Order the concepts from problem requirement to final runnable solution.
   - Include smaller subproblem, first state, completion condition, first
     correct baseline, optimization bridge, and final checkpoint when relevant.
   - Verify: the graph starts from a tiny example or conflict pattern.
5. Write Tutorial Build Tasks
   - Create ordered tasks with description, acceptance criteria, verification,
     dependencies, document target, code change role, and review gate.
   - Each tutorial-step task must include problem pressure, previous baseline,
     break, one change, check evidence, checkpoint check requirements, freeze,
     still lacks, and next gap.
   - Each task that introduces a state variable, helper, invariant, recurrence,
     formula, rule, or named concept must include a `concept_timing` check:
     what pressure forces the name, where it is first named, where it is first
     operationally used, and what capability claim is allowed before that first
     use.
   - Verify: every task can be written and reviewed in one focused pass.
6. Plan Review and Checkpoint Handoff
   - Mark tasks that require `leetcode-tutorial-review` before continuation.
   - Mark tasks that may use a document checkpoint commit after review pass.
   - Verify: no build task can self-approve.

## Tutorial Task Contract

Every tutorial-step task must include:

- `Problem pressure`: concrete example, trace, constraint, or conflict that
  forces the next step.
- `Previous baseline`: what the reader can already run or reason about.
- `Break`: what the baseline cannot yet answer, prove, optimize, or execute.
- `Change`: exactly one state, rule, helper, recurrence, branch, or code block.
- `Check evidence`: assertion, trace, example walk-through, or runnable command
  that must be executed or reviewed.
- `Checkpoint check requirements`: what the builder/reviewer must inspect for
  this checkpoint, what passes, what fails, what evidence is required, and
  whether concept timing is covered.
- `Freeze`: what the checkpoint can now do.
  Do not claim the reader can use a concept, variable, helper, rule, or
  invariant unless that checkpoint has shown its first operational use. Before
  first operational use, freeze may only say the reader knows what it
  represents or why it will be needed.
- `Still lacks`: the next defect that justifies the next task.
- `Review gate`: `leetcode-tutorial-review` must pass this step before the next
  build task starts.

## Concept Timing Contract

Plans must prevent named ideas from arriving before the current problem
pressure needs them.

For each new concept, field, helper, state variable, recurrence, invariant, or
rule introduced by a task, record:

- `introduced_concepts`: names introduced in this task.
- `pressure_for_introduction`: the unresolved problem that forces the name now.
- `first_operational_use`: the first task or checkpoint where the concept
  participates in a condition, update, recurrence, helper call, return value,
  proof step, or decision.
- `capability_claim_rule`: what the freeze may claim before and after that
  first operational use.

If a task names a concept but its first operational use belongs to a later
task, that task's `freeze` must use wording like "knows what X represents" or
"knows why X will be needed"; it must not say "can use X" or "can solve with
X". When possible, prefer moving the concept introduction into the same task as
its first operational use.

## Output Format

```markdown
# LeetCode Tutorial Plan: <Problem>

## Problem Facts
- Statement:
- Input / Output:
- Examples:
- Constraints:
- Supplied facts:
- Inferred assumptions:

## Placement
- Path:
- Slug:
- Tutorial language:
- Code language:
- Taxonomy:

## Teaching Dependency Graph
```text
problem requirement
-> tiny example pressure
-> smaller subproblem
-> first runnable baseline
-> first correct solution
-> optimization bridge
-> final runnable checkpoint
```

## Tutorial Build Task List

### Task 1: Establish problem requirement
**Description:**
**Acceptance criteria:**
- [ ] ...
**Verification:**
- [ ] ...
**Dependencies:** None
**Document target:**
**Code change role:** prose-only | patch | checkpoint
**Review gate:** required | not_required
**Checkpoint commit:** yes | no

### Task N: Draft <step>
**Description:**
**Acceptance criteria:**
- [ ] Problem pressure is visible before the fix.
- [ ] Previous baseline is explicit.
- [ ] Break is concrete.
- [ ] One change is introduced.
- [ ] Check evidence is executable or reviewable.
- [ ] Checkpoint check requirements specify inspect/pass/fail/evidence and
      concept timing coverage.
- [ ] Freeze and still-lacks are explicit.
- [ ] Concept timing is explicit: no introduced variable/helper/rule is claimed
      as usable before its first operational use.
**Verification:**
- [ ] ...
**Dependencies:**
**Document target:**
**Code change role:** patch | checkpoint | prose-only
**Review gate:** required
**Checkpoint commit:** yes | no
**Freeze fields for commit message:**
- Pressure:
- Baseline:
- Break:
- Change:
- Check:
- Checkpoint check requirements:
  - inspect:
  - pass_when:
  - fail_when:
  - required_evidence:
  - concept_timing_coverage:
- Freeze:
- Still lacks:
- Next:
- Concept timing:

## Review Gate Handoff
| Task | review_required | review_scope | pass_allows |
| --- | --- | --- | --- |

## Verification Matrix
| Case | What It Proves | Planned Task |
| --- | --- | --- |

## Build Handoff
- recommended_builder: leetcode-tutorial-build
- task_execution_order:
- notes:
```

## Decision Points

- If the user asks for a concept tutorial without one problem statement, route
  to `algorithm-tutorial-workflow`.
- If examples or constraints are missing and they affect the algorithm choice,
  ask before planning.
- If the route needs multiple optimization stages, make each stage a separate
  task.
- If a task cannot be reviewed independently, split or merge it before build.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The template is obvious." | The tutorial must derive the structure from problem evidence. |
| "The builder can decide the steps." | Build needs explicit tasks, dependencies, checks, and review gates. |
| "A self-check is enough." | Build self-check is evidence; review pass is the gate. |
| "The final answer can explain the earlier steps." | The reader must see connected checkpoints before final code. |

## Red Flags

- The plan starts from a named template before problem facts.
- No smaller subproblem is planned.
- No first correct baseline is planned before optimization.
- Tutorial tasks lack verification or review gate.
- Tutorial tasks introduce a variable, helper, invariant, or rule before the
  current pressure needs it.
- A task's freeze claims "can use X" even though X is not operationally used
  until a later task.
- A task asks build to write more than one numbered step.
- Final runnable code is planned as a detached reference answer.

## Verification

- [ ] Problem facts are separated from assumptions.
- [ ] Placement and taxonomy are planned.
- [ ] Dependency graph starts from problem evidence.
- [ ] Build tasks include acceptance criteria, verification, dependencies,
      document targets, review gates, and checkpoint handoff.
- [ ] Build tasks record concept timing and first operational use for new
      state, helpers, invariants, recurrences, formulas, and rules.
- [ ] First correct baseline and optimization bridge are planned when relevant.
- [ ] Final code appears as a connected checkpoint.

## Guardrails

- Do not write the tutorial body during planning.
- Do not invent constraints, examples, or results.
- Do not plan one giant build task for a multi-step tutorial.
- Do not allow build self-approval.
