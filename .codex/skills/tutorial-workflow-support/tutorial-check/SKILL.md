---
name: tutorial-check
description: v0.1.1 - Check a tutorial draft against its saved plan and sketch before review. Use when a draft needs plan adherence, concept timing, first operational use, code continuity, runnable snippet, and Hugo placement checks before tutorial review.
---

# Tutorial Check

## Overview

Compare a tutorial draft with its saved plan and sketch before handing it to a
review skill. This is a consistency check, not an independent teaching review.
It verifies that the draft followed the planned article path, pressure-first
sequence, code-growth contract, and verification requirements.

Default inputs:

```text
.agent-runs/tutorials/<slug>/plan.md
.agent-runs/tutorials/<slug>/sketch.md
content/.../<article>.md
```

Default output:

```text
.agent-runs/tutorials/<slug>/check.md
```

## When to Use

- A tutorial draft or checkpoint has been written from a saved plan/sketch.
- Before `$leetcode-tutorial-review`, `$algorithm-tutorial-review`, or
  `$tutorial-reviewer`.
- When you need to catch drift such as premature helpers, missing freeze
  fields, detached final code, or changed article placement.

**When NOT to use:** creating a plan, writing the draft, replacing review,
enhancing SEO, or checking unrelated code changes.

## Reference Map

- `references/check-report-template.md`
  Use for the default check report shape.

## The Checking Process

### Step 1: Load The Contract Inputs

Read the three layers that define the checkpoint:

- plan record: `.agent-runs/tutorials/<slug>/plan.md`
- sketch: `.agent-runs/tutorials/<slug>/sketch.md`
- draft article or checkpoint file under `content/...`

Identify the target article path, expected language, current checkpoint scope,
planned review skill, and whether this check covers a partial checkpoint or a
full draft.

If the plan or sketch is missing, stop and route to `$tutorial-plan-record` or
`$tutorial-sketch`. Do not reconstruct missing contracts from memory.

### Step 2: Resolve The Check Boundary

Decide what exactly is being checked:

- Use the same slug as the plan and sketch.
- Write to `.agent-runs/tutorials/<slug>/check.md`.
- Check one article at a time.
- If the draft covers only one checkpoint, do not require later checkpoints to
  already exist.
- If the draft claims to be complete, require final runnable code placement and
  all planned evidence to be present or explicitly skipped with a reason.

The check boundary prevents two common errors: blocking an early checkpoint for
future work, or passing a full draft that only satisfies the first checkpoint.

### Step 3: Check Plan Adherence

Compare the article to `plan.md`:

- article path, language, taxonomy, and tutorial type match the plan
- current task order follows the recorded plan
- pressure, previous baseline, break, change, check, freeze, and still-lacks
  are visible for the checked checkpoint
- checkpoint check requirements from the plan are satisfied or explicitly
  marked missing: inspect targets, pass conditions, fail conditions, required
  evidence, and concept timing coverage
- concept timing from the plan is visible: newly named variables, helpers,
  invariants, recurrences, formulas, and rules are not claimed as usable before
  first operational use
- review gate and next recommended skill match the planned workflow
- any intentional divergence is named and routed back to plan/sketch revision

This is not a writing-quality review. It only asks whether the draft followed
the accepted route.

### Step 4: Check Sketch Adherence

Compare the article to `sketch.md`:

- first screen starts from the planned pressure, not from a detached template
- each introduced concept appears after its pressure
- each concept's first operational use appears at the planned checkpoint or is
  explicitly still missing for partial checkpoints
- freeze wording does not over-claim named-only concepts
- helpers, state variables, formulas, and invariants respect the forbidden
  early concept list
- code grows from the previous visible baseline to the allowed change
- final runnable code appears only at the planned location
- final code contains no logic that was never earned by an earlier checkpoint

If the draft reads well but violates helper timing or final-code placement,
mark the check `fail`; that is exactly the drift this skill exists to catch.

### Step 5: Check Evidence

Run or record concrete evidence:

- Run snippet or solution checks when the article contains runnable code and a
  local check is feasible.
- Run `git diff --check` for touched Markdown when available.
- Run Hugo build when front matter, content placement, Markdown structure, or
  links changed enough to risk site breakage and the command is feasible.
- If a check cannot run, record the operational reason instead of replacing it
  with a vague "manual check".

### Step 6: Write Report And Route

Write:

```text
.agent-runs/tutorials/<slug>/check.md
```

Use `references/check-report-template.md`. Return:

- `pass` only when there is no blocking plan/sketch drift
- `fail` when the draft exists but violates the plan or sketch
- `blocked` when required inputs are missing or verification cannot proceed

Name the next recommended skill:

- review skill when check passes
- build skill when a checkpoint needs rewrite
- plan or sketch skill when the contract itself must change

## Check Layer Boundaries

Belongs in `check.md`:

- pass/fail/blocked result
- plan adherence findings
- sketch adherence findings
- verification evidence
- next recommended skill

Belongs in the review skill:

- teaching quality judgment
- derivation strength
- explanation clarity
- whether the checkpoint is pedagogically acceptable

Belongs in the build skill:

- article rewriting
- adding missing sections
- changing code/prose

Belongs in the plan or sketch skill:

- accepted route changes
- checkpoint resequencing
- helper timing contract updates

## Checking Examples

Good sequence:

```text
$leetcode-tutorial-build
  -> writes one planned checkpoint

$tutorial-check
  -> verifies path, task order, helper timing, code growth, and evidence

$leetcode-tutorial-review
  -> judges whether the checkpoint teaches well enough to pass
```

Bad sequence:

```text
$tutorial-check
  -> reads the article only
  -> says pass because the prose looks fine
  -> skips plan/sketch drift and review handoff
```

The bad sequence turns check into a shallow review. Check must compare the
draft against the saved contract.

## Decision Points

- If plan or sketch is missing, route to `$tutorial-plan-record` or
  `$tutorial-sketch`.
- If the draft intentionally diverges from the plan, mark the check `fail` and
  route to the relevant plan/sketch update instead of normal review.
- If runnable checks cannot be run, record the operational reason.
- If only local run artifacts changed, do not recommend a Git commit.
- If the checked scope is only one checkpoint, report missing later sections as
  `still_lacks`, not as blocking drift.
- If the draft is complete but final code contains unexplained logic, mark the
  check `fail`.
- If the draft says a checkpoint can use a concept that was only defined and
  not operationally used, mark the check `fail` and route back to build.

## Common Rationalizations

| Rationalization | Reality |
| --- | --- |
| "Review will catch this." | Check catches mechanical drift before reviewer attention is spent. |
| "The article reads fine, so plan drift is okay." | Planned checkpoints are the contract for multi-turn build continuity. |
| "A concept is defined, so the checkpoint can claim it." | The checkpoint can claim use only after first operational use. |
| "A Hugo build is unnecessary for docs." | Markdown/front matter changes can break a static site. |
| "Check pass means review pass." | Check pass only means the draft is ready for the appropriate review skill. |

## Red Flags

- The report gives `pass` while listing blocking drift.
- The draft has a helper/state variable before the planned pressure.
- The draft introduces a concept without the planned first operational use or
  over-claims it in the freeze.
- The article path or language differs from the plan without explanation.
- Final code contains logic that never appeared in a checkpoint.
- Checks are described generically instead of naming concrete evidence.
- The check ignores whether it is checking a partial checkpoint or full draft.
- The report rewrites the article instead of routing back to build.

## Verification

- [ ] Plan, sketch, and draft were read or missing inputs were reported.
- [ ] Check boundary is identified as partial checkpoint or full draft.
- [ ] Plan adherence was checked.
- [ ] Sketch adherence was checked.
- [ ] Concept timing and first operational use were checked.
- [ ] Code/prose continuity was checked.
- [ ] Runnable or operational verification was recorded.
- [ ] Pass/fail/blocked status matches the findings.
- [ ] Next recommended skill is named.

## Output Format

```text
## Tutorial Check Result
- check_status: pass | fail | blocked
- plan_path:
- sketch_path:
- article_path:
- next_recommended_skill:

## Findings
- [blocking|warning|note] ...

## Verification
- ...
```

## Guardrails

- Do not rewrite the article during check.
- Do not replace tutorial review.
- Do not commit `.agent-runs/` artifacts.
- Do not mark `pass` with unresolved blocking drift.
