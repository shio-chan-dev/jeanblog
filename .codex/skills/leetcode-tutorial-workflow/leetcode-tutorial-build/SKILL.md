---
name: leetcode-tutorial-build
description: v0.1.0 - Build exactly one planned LeetCode tutorial step and stop for independent review. Use when a task-first LeetCode tutorial plan is ready and the next checkpoint must be drafted with problem evidence, one change, check evidence, freeze, and a needs-review handoff.
---

# LeetCode Tutorial Build

## Overview

Write one planned tutorial task, not the whole article. This skill drafts the
next checkpoint in a problem tutorial and then stops for `leetcode-tutorial-review`.

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

## Bundled Resources

- `../../leetcode-tutorial-builder/references/incremental-build-contract.md`
  for the original repository-specific incremental build rules.

## The Build Loop

1. Load Current Task
   - Read the task description, dependencies, acceptance criteria,
     verification, document target, code change role, and review gate.
   - Confirm previous tasks have review-pass evidence when they are
     dependencies.
   - Verify: exactly one task is in scope.
2. Re-state the Current Baseline
   - Identify what the reader already has from prior checkpoints.
   - Identify the concrete problem evidence or trace that creates pressure.
   - Verify: the step starts from visible state, not final-answer memory.
3. Draft One Step
   - Show the problem pressure before the fix.
   - Name what breaks in the previous baseline.
   - Add or replace exactly one state, rule, helper, recurrence, branch, or
     code block.
   - Keep code connected to the previous checkpoint.
   - Verify: the step does not introduce unrelated future logic.
4. Add Check Evidence
   - Add an assertion, trace, manual example walk-through, or command that
     verifies this one change.
   - Execute the runnable check when execution is feasible.
   - If the check is conceptual or manual, record concrete evidence, not a
     vague claim.
   - Verify: check evidence targets this step's break.
5. Freeze and Stop
   - State what the checkpoint can now do and what it still lacks.
   - Output `needs_review`.
   - Do not continue to the next planned task.
   - Do not call the checkpoint accepted.

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

## Red Flags

- Build writes more than one planned task.
- Build says `pass`, `accepted`, or `done` without review.
- Step check is described but not executed or evidenced.
- The step introduces a final trick before the pressure appears.
- The output proceeds to the next step after `Checkpoint`.

## Verification

- [ ] Exactly one task was drafted.
- [ ] Problem pressure appears before the change.
- [ ] The previous baseline and break are explicit.
- [ ] One change was added or replaced.
- [ ] Check evidence is concrete and tied to the break.
- [ ] Output ends with `needs_review`.

## Guardrails

- Do not self-approve.
- Do not continue to the next step.
- Do not invent problem facts.
- Do not add detached final code.
- Do not hide missing runnable checks behind generic "verified" language.
