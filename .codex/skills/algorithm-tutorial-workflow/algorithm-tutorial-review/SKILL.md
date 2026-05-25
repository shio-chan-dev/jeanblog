---
name: algorithm-tutorial-review
description: v0.1.0 - Review one algorithm tutorial checkpoint or full draft as an independent gate. Use when checking derivation-first structure, pressure, mechanism, invariant, evidence, code continuity, depth readiness, and final runnable demo placement before allowing the tutorial to advance.
---

# Algorithm Tutorial Review

## Overview

Review a standalone algorithm/method tutorial as an independent gate. The
review decides whether the checkpoint can advance with `pass`, `revise`, or
`block`.

## When to Use

- `algorithm-tutorial-build` produced a checkpoint.
- A method tutorial needs structure/depth review before continuing.
- A checkpoint or full draft needs acceptance before simplification.

**When NOT to use:** writing, deepening, polishing, or LeetCode problem review.

## The Review Loop

1. Identify Scope
   - Confirm subject, reader, and whether implementation is part of the goal.
2. Check Derivation
   - Confirm the checkpoint starts from pressure, not glossary or final formula.
   - Confirm one mechanism follows from the pressure.
3. Check Algorithm Depth
   - Look for invariant, formalization, correctness, complexity,
     counterexample, or engineering reality when the task requires them.
4. Check Evidence
   - Confirm examples, traces, formulas, or runnable snippets are concrete.
5. Check Code Continuity
   - Confirm code grows from the previous checkpoint and final runnable demo
     does not introduce unexplained logic.
6. Produce Verdict
   - `pass`, `revise`, or `block`.

## Output Format

```markdown
## Verdict
- verdict: pass | revise | block
- review_scope:
- may_continue_next_step: yes | no
- needs_deepening: yes | no

## Findings
- [severity] [section]: [issue]
  Evidence:
  Required change:

## Evidence Review
- anchors_seen:
- missing_depth:

## Notes
- ...
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The explanation sounds plausible." | Algorithm tutorials need evidence, invariants, and checks. |
| "The final demo works." | The derivation chain can still be broken. |
| "Depth is for later." | Missing invariant/correctness may block the current checkpoint. |
| "Build already checked it." | Review is the acceptance gate. |

## Verification

- [ ] Scope was identified.
- [ ] Derivation-first structure was checked.
- [ ] Depth expectations were checked.
- [ ] Evidence quality was checked.
- [ ] Code continuity was checked.
- [ ] Verdict is `pass`, `revise`, or `block`.

## Guardrails

- Do not rewrite while reviewing.
- Do not pass checkpoints without concrete evidence.
- Do not let build self-check substitute for review pass.
