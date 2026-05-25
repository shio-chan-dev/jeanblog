---
name: algorithm-tutorial-plan
description: v0.1.0 - Plan one standalone algorithm, data-structure, or method tutorial as review-gated writing tasks. Use when the subject is the technique itself and the plan must preserve derivation-first construction, runnable checkpoints, invariants, correctness, complexity, counterexamples, engineering tradeoffs, deepening tasks, and publishable Hugo placement before drafting.
---

# Algorithm Tutorial Plan

## Overview

Plan one standalone algorithm or method tutorial before drafting. The output is
a task-first plan, not the article body. It must make room for both guided
construction and algorithm depth.

The tutorial should grow through this path:

```text
real pressure
-> tiny task or trace
-> missing capability
-> first mechanism
-> invariant / correctness
-> complexity / tradeoff
-> final runnable demo
```

## When to Use

- The user wants to teach one algorithm, data structure, or method itself.
- The tutorial needs derivation-first construction and depth planning.
- The topic needs invariants, formulas, correctness, complexity, failure cases,
  or engineering tradeoffs.

**When NOT to use:** one concrete OJ-style problem tutorial, SEO enhancement,
thoughts/evaluation posts, or final-code-only answers.

## Reference Map

- `../../algorithm-tutorial-builder/references/derivation-first-explanations.md`
- `../../algorithm-tutorial-builder/references/depth-checklist.md`
- `../../algorithm-tutorial-builder/references/deepening-ladder.md`
- `../../algorithm-tutorial-builder/references/language-selection-rubric.md`

## The Planning Loop

1. Define Subject and Scope
   - Name the algorithm/method, target reader, implementation language, and
     output language.
   - Verify: the subject is not one judged problem statement.
2. Choose Real Pressure
   - Pick a tiny task, bottleneck, missing capability, or trace that makes the
     method necessary.
   - Verify: the opening does not start from a glossary or final formula.
3. Plan Derivation Checkpoints
   - Order missing state, mechanism, helper, invariant, code fragment, and final
     demo tasks.
   - Verify: every build task introduces one mechanism or rule.
4. Plan Algorithm Depth
   - Select 1-2 core concepts for PDKH deepening.
   - Plan invariant, formalization, correctness sketch, threshold/complexity,
     counterexample, and engineering reality tasks.
   - Verify: depth is not generic background.
5. Plan Placement
   - Choose Hugo path, taxonomy, front matter, and code language using project
     conventions and the language rubric.
6. Plan Review Gates
   - Mark build tasks that require `algorithm-tutorial-review`.
   - Mark depth tasks that require `algorithm-tutorial-deepen`.
   - Verify: build cannot self-approve.

## Output Format

```markdown
# Algorithm Tutorial Plan: <Topic>

## Subject and Scope
- Topic:
- Reader:
- Tutorial language:
- Code language:
- Output path:
- Taxonomy:

## Real Pressure
- Tiny task / trace:
- Missing capability:
- Why this pressure fits:

## Teaching Dependency Graph
```text
pressure
-> first representation
-> first mechanism
-> invariant
-> code checkpoint
-> correctness / complexity
-> final runnable demo
```

## Tutorial Build Task List
### Task N: <title>
**Description:**
**Acceptance criteria:**
- [ ] ...
**Verification:**
- [ ] ...
**Dependencies:**
**Document target:**
**Code change role:** prose-only | patch | checkpoint
**Review gate:** required | not_required
**Deepening gate:** required | not_required

## Deepening Plan
| Concept | PDKH steps required | Evidence target |
| --- | --- | --- |

## Verification Matrix
| Case | What It Proves | Planned Task |
| --- | --- | --- |

## Build Handoff
- recommended_builder: algorithm-tutorial-build
- review_skill: algorithm-tutorial-review
- deepen_skill: algorithm-tutorial-deepen
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The method name is enough." | The tutorial must show the pressure that makes the method necessary. |
| "Depth can be added later." | Invariants and correctness affect the build path. |
| "This can reuse LeetCode flow." | Algorithm tutorials teach the method, not one judged input/output task. |
| "Build can verify itself." | Build self-check is evidence, not acceptance. |

## Verification

- [ ] Subject is algorithm/method, not one OJ problem.
- [ ] Real pressure and dependency graph are explicit.
- [ ] Build tasks include acceptance, verification, dependencies, document
      target, and gates.
- [ ] Deepening plan selects 1-2 concepts.
- [ ] Final runnable demo/checkpoint is planned when implementation matters.

## Guardrails

- Do not write the article body during planning.
- Do not start from glossary, component preview, or final formula.
- Do not allow build self-approval.
- Do not flatten algorithm depth into generic background.
