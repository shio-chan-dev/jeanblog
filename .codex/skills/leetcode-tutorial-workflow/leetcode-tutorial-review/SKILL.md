---
name: leetcode-tutorial-review
description: v0.1.1 - Review one LeetCode tutorial checkpoint or full draft as an independent gate. Use when deciding whether a problem tutorial step can advance, needs revision, or must block because pressure, break, change, concept timing, check evidence, code continuity, or final-code placement is weak.
---

# LeetCode Tutorial Review

## Overview

Review a LeetCode or OJ-style tutorial as an independent acceptance gate. The
reviewer is not the builder. It must decide whether the current checkpoint can
advance with `pass`, `revise`, or `block`.

The primary question is whether the reader can move from the previous visible
baseline to this checkpoint without hidden logic or unjustified template jumps.

## When to Use

- `leetcode-tutorial-build` produced a `needs_review` step.
- A problem tutorial draft needs structural review before the next step.
- A checkpoint commit is requested and needs review-pass evidence first.

**When NOT to use:** writing new tutorial content, SEO enhancement, standalone
algorithm concept review, or full rewrite requests.

## The Review Loop

1. Identify Scope
   - Determine whether the review target is one step or a full draft.
   - Identify the problem facts and previous baseline.
   - Verify: the target is an OJ-style problem tutorial.
2. Check Problem-First Discipline
   - Confirm problem requirement, input/output, examples, and constraints
     appear before derivation.
   - Verify: the tutorial does not start from a named template.
3. Check Step Gate
   - For the target step, inspect pressure, previous baseline, break, change,
     check evidence, freeze, and still-lacks.
   - Verify: the change follows from the break and changes only one core thing.
4. Check Concept Timing
   - For every newly named variable, helper, invariant, recurrence, formula, or
     rule, identify where it is first operationally used.
   - Verify: the concept is introduced only after pressure forces it, and the
     checkpoint does not claim the reader can use it before that first
     operational use.
   - If the concept is named now but used later, verify `still-lacks` says the
     first operational use is still missing.
5. Check Evidence
   - Confirm checks were executed or backed by concrete manual trace evidence.
   - Reject generic "checked" language.
   - Verify: evidence proves this checkpoint, not just the final answer.
6. Check Code Continuity
   - Confirm snippets add to or replace the previous baseline.
   - Confirm optimization stages preserve a bridge from first correct version
     to final version.
   - Verify: no hidden helper or final trick appears only at the end.
7. Produce Verdict
   - `pass`: the step may advance.
   - `revise`: targeted fixes are needed before advancing.
   - `block`: the teaching chain is broken or evidence is missing.

## Severity

- `block`: no concrete check evidence, hidden final logic, problem facts
  missing, or step cannot be derived from the previous baseline.
- `revise`: pressure is present but thin, break is generic, change is too broad,
  checkpoint/freeze is unclear, or a named-only concept is over-claimed before
  first operational use.
- `suggestion`: local wording or polish that does not block progression.

## Output Format

```markdown
## Verdict
- verdict: pass | revise | block
- review_scope:
- may_continue_next_step: yes | no
- may_checkpoint_commit: yes | no

## Findings
- [severity] [section]: [issue]
  Evidence:
  Required change:

## Check Evidence Review
- checks_seen:
- evidence_quality:

## Notes
- ...
```

## Decision Points

- If reviewing a full draft, still inspect step-by-step continuity.
- If a checkpoint commit is requested without review-pass evidence, return
  `block`.
- If the tutorial is actually a concept tutorial, route to
  `tutorial-reviewer` or the future algorithm workflow reviewer.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The code is correct, so the step passes." | Tutorial checkpoints must teach the transition, not only land correct code. |
| "The builder ran self-checks." | Self-checks are evidence to inspect, not acceptance. |
| "The missing bridge is obvious." | If the bridge is obvious, it should be visible in the tutorial. |
| "The final answer covers it." | Final answers do not repair hidden jumps in earlier checkpoints. |
| "The variable will be used in the next step." | Then this step may only claim the reader knows what it represents, not that they can use it. |

## Red Flags

- No problem requirement before derivation.
- No smaller subproblem when recursion, DP, or staged optimization needs one.
- Step check is missing or unexecuted.
- Optimization appears before a first correct baseline exposes the bottleneck.
- A checkpoint introduces a variable, helper, invariant, formula, or rule before
  the current pressure needs it.
- A freeze says "can use X" when X has not yet participated in a condition,
  update, recurrence, helper call, return value, proof step, or decision.
- Review output gives `pass` while listing blocking issues.

## Verification

- [ ] Review scope was identified.
- [ ] Problem-first discipline was checked.
- [ ] Step pressure, break, change, check, freeze, and still-lacks were checked.
- [ ] Concept timing and first operational use were checked.
- [ ] Check evidence quality was evaluated.
- [ ] Code continuity was checked.
- [ ] Verdict is one of `pass`, `revise`, or `block`.

## Guardrails

- Do not rewrite the tutorial during review.
- Do not let build self-check substitute for review pass.
- Do not pass a step with missing check evidence.
- Do not bury blockers under compliments.
