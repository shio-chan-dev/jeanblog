---
name: algorithm-tutorial-build
description: v0.1.0 - Build exactly one planned algorithm tutorial checkpoint and stop for review. Use when a task-first algorithm tutorial plan is ready and the next section must derive one mechanism, invariant, code fragment, or runnable checkpoint from concrete pressure without self-approval.
---

# Algorithm Tutorial Build

## Overview

Draft one planned algorithm tutorial checkpoint and stop. This skill writes the
next piece of the article, then hands it to `algorithm-tutorial-review` or
`algorithm-tutorial-deepen`.

Build may run self-checks, but self-checks are evidence, not acceptance.

## When to Use

- An `algorithm-tutorial-plan` task list exists.
- The next planned checkpoint should be drafted.
- The tutorial is about a method itself, not one OJ problem.

**When NOT to use:** planning, reviewing, deepening, simplifying, LeetCode
problem tutorials, or writing a whole multi-section article in one pass.

## Reference Map

- `../references/derivation-first-explanations.md`
- `../references/depth-checklist.md`

## The Build Loop

1. Load One Task
   - Read dependencies, acceptance criteria, verification, document target,
     code role, review gate, and deepening gate.
   - Verify: exactly one task is in scope.
2. Re-state Pressure and Baseline
   - Show the tiny task, trace, bottleneck, or missing capability.
   - Identify what the reader already has.
3. Draft One Mechanism
   - Add one representation, invariant, rule, helper, formula, or code block.
   - Introduce terms at first real use.
   - Keep code connected to the current checkpoint.
4. Add Evidence
   - Include a worked trace, invariant check, formula with variables, runnable
     snippet, or counterexample as required by the task.
   - Execute runnable checks when feasible.
5. Freeze and Stop
   - State what the checkpoint now explains or runs.
   - State what still lacks.
   - Output `needs_review` or `needs_deepening`.
   - Do not continue to the next task.

## Output Format

```markdown
## Build Result
- status: needs_review | needs_deepening
- task:
- document_target:
- code_change_role:

## Drafted Content
<one checkpoint only>

## Self-Checks
- ...

## Gate Handoff
- review_skill: algorithm-tutorial-review
- deepen_skill: algorithm-tutorial-deepen | n/a
- next_step_blocked_until: gate_pass
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "I can finish the whole article faster." | Hidden jumps compound across algorithm tutorials. |
| "The invariant is obvious." | The invariant is one of the teaching objects. |
| "The code works, so depth is optional." | Correctness and complexity are part of algorithm learning. |
| "Self-check passed." | Self-check is evidence; review or deepen gate accepts. |

## Verification

- [ ] Exactly one task was drafted.
- [ ] Concrete pressure appears before the mechanism.
- [ ] One mechanism or rule was introduced.
- [ ] Evidence is concrete.
- [ ] Output stops with a gate handoff.

## Guardrails

- Do not self-approve.
- Do not continue to the next checkpoint.
- Do not write LeetCode-style problem solution flow for concept tutorials.
- Do not add detached final code.
