---
name: leetcode-tutorial-simplify
description: v0.1.0 - Simplify a reviewed LeetCode tutorial without breaking the guided-build chain. Use when a problem tutorial has review-pass structure but needs tighter prose, less template noise, clearer transitions, or reader-facing language while preserving problem facts, step checks, checkpoints, and final runnable code.
---

# LeetCode Tutorial Simplify

## Overview

Simplify a structurally sound LeetCode tutorial. This skill is not a repair
gate for broken teaching flow. It tightens wording while preserving problem
facts, connected code checkpoints, check evidence, and final runnable code.

## When to Use

- A LeetCode tutorial step or full draft has passed review.
- The guide is correct but too repetitive, checklist-like, or heavy.
- The user wants more natural reader-facing prose without changing the solution.

**When NOT to use:** planning, first-draft building, accepting a checkpoint,
changing algorithm behavior, or fixing a broken teaching chain.

## The Simplification Loop

1. Confirm Review-Pass Baseline
   - Check that the target has review-pass evidence or is explicitly a local
     prose cleanup after review.
   - Verify: simplification is not hiding unresolved blockers.
2. Protect Required Content
   - Preserve problem requirement, input/output, examples, constraints,
     pressure, baseline, break, change, check evidence, freeze, and final code.
   - Verify: no checkpoint loses its proof or next gap.
3. Tighten Prose
   - Remove duplicated rationale and template noise.
   - Convert internal scaffolding into reader-facing paragraphs when safe.
   - Keep code change boundaries clear.
4. Preserve Code Behavior
   - Do not change code logic unless explicitly asked.
   - If code snippets are shortened, keep them runnable or clearly marked as
     partial additions.
5. Report Residual Risk
   - Name anything that still needs review rather than silently accepting it.

## Decision Points

- If review did not pass, route to `leetcode-tutorial-review` or build repair.
- If simplifying would remove check evidence, keep the evidence and shorten
  surrounding prose instead.
- If the tutorial needs algorithm changes, stop and ask whether this is a build
  task.

## Output Format

```markdown
## Simplification Summary
- ...

## Preserved Chain
- ...

## Remaining Risks
- ...
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "Shorter is clearer." | Removing checks or checkpoints makes the tutorial less reliable. |
| "The problem statement can be summarized away." | Problem facts anchor the whole problem tutorial. |
| "The final code is enough." | The guided-build chain is the teaching asset. |
| "Simplify can fix review blockers." | Broken structure needs build/review, not prose cleanup. |

## Red Flags

- Problem facts are removed.
- A step loses check evidence.
- A checkpoint no longer says what it can do or lacks.
- Code behavior changes during prose simplification.
- Final runnable code becomes a partial snippet.

## Verification

- [ ] Review-pass baseline was confirmed or limitation was reported.
- [ ] Problem facts remain intact.
- [ ] Step checks and checkpoints remain intact.
- [ ] Code behavior was not changed.
- [ ] Remaining risks are reported.

## Guardrails

- Do not self-approve.
- Do not delete check evidence.
- Do not change algorithm behavior.
- Do not use simplification to hide missing review.
