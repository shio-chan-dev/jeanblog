---
name: tutorial-plan-record
description: v0.1.0 - Record an approved tutorial plan into an ignored local run directory. Use when a LeetCode or algorithm tutorial plan must survive context compaction, multi-turn work, or handoff before build starts.
---

# Tutorial Plan Record

## Overview

Save an already-created tutorial plan as local execution state. This skill does
not plan, improve, or approve the tutorial. It preserves the plan's task order,
freeze fields, review gates, and article target so later build/check/review
turns can resume without relying on chat history.

Default output location:

```text
.agent-runs/tutorials/<slug>/plan.md
```

`.agent-runs/` is ignored by Git. Promote a plan to `docs/` only when the user
explicitly asks for a long-lived project document.

## When to Use

- A `$leetcode-tutorial-plan` or `$algorithm-tutorial-plan` result has been
  accepted as the working plan.
- A tutorial will be built across multiple turns or sessions.
- The next agent needs task order, pressure/break/change/check/freeze fields,
  or review gates without scanning chat history.

**When NOT to use:** creating a new plan, revising a weak plan, recording a
draft article, publishing user-facing documentation, or saving one-off notes
that will not guide later tutorial work.

## Reference Map

- `references/plan-record-template.md`
  Use for the default plan record shape.

## The Recording Process

### Step 1: Confirm The Accepted Source Plan

Before touching `.agent-runs/`, identify the final tutorial plan to record:

- Use the completed `$leetcode-tutorial-plan` or `$algorithm-tutorial-plan`
  output from the current conversation, a supplied plan document, or an
  approved revision of an existing run.
- Confirm the source includes target article path, tutorial type, article
  language, code language, ordered tasks, pressure/baseline/break/change/check/
  freeze fields, still-lacks, and review gates.
- Confirm the human has accepted the plan as the working route, or that the
  request explicitly says to record this plan.
- If the input is only a rough idea, draft article, review report, or chat
  fragment, stop and route to the appropriate plan skill.

**Do NOT infer a tutorial plan from an article draft or from scattered chat
history.** This skill records an approved plan; it does not create one.

### Step 2: Resolve The Tutorial Run Boundary

Choose whether this is a new tutorial run or an update to an existing one:

- Create a new slug for a new target article or new tutorial goal.
- Reuse `.agent-runs/tutorials/<slug>/` when recording a revision of the same
  article plan.
- Derive the slug from the target article filename when possible.
- If multiple existing tutorial runs could apply, inspect their `plan.md`
  targets and choose the one matching the same article; ask if the boundary is
  still unclear.

Run boundaries are part of the tutorial workbench. Do not create a fresh slug
only to avoid updating an existing record for the same article.

### Step 3: Prepare The Local Workbench

Create or reuse:

```text
.agent-runs/tutorials/<slug>/
```

Keep run artifacts out of commits:

- Verify `.agent-runs/` is ignored before finishing.
- Never stage `.agent-runs/` artifacts.
- Do not write run artifacts under `content/`, `docs/`, or `.codex/skills/`.
- Promote a plan record to tracked `docs/` only when the user explicitly asks
  for a durable project document.

### Step 4: Translate The Plan Into A Compact Record

Write:

```text
.agent-runs/tutorials/<slug>/plan.md
```

Use `references/plan-record-template.md` and preserve durable handoff data:

- source skill and source prompt
- target article path, tutorial type, article language, and code language
- supplied facts and inferred assumptions
- ordered task list
- each task's pressure, previous baseline, break, change, check evidence,
  freeze, still-lacks, review gate, and checkpoint commit guidance when present
- review gates and verification matrix
- next recommended skill

The record should compress the accepted plan without changing it. It should not
become a second tutorial plan and should not smooth over missing pressure,
checks, or review gates.

### Step 5: Keep Details In The Right Layer

Belongs in `plan.md`:

- article target, language, and tutorial type
- task order and checkpoint boundaries
- pressure/break/change/check/freeze summaries
- review gates and checkpoint commit guidance
- assumptions separated from supplied facts

Belongs in later `sketch.md`:

- first-screen teaching pressure
- section skeleton
- code-growth map
- forbidden early concepts
- helper extraction timing
- final runnable code placement

Belongs in later `check.md`:

- whether the draft followed the plan and sketch
- drift findings
- runnable snippet, Markdown, or Hugo verification evidence
- next recommended review or build action

Belongs in the article under `content/`:

- publishable tutorial prose
- incremental code explanations
- final runnable solution

Belongs in `.codex/skills/`:

- reusable workflow instructions only, not per-tutorial run state

### Step 6: Verify The Record

Before reporting success:

- Confirm `plan.md` reflects the final accepted plan, not an earlier draft.
- Confirm every recorded task maps to a source task or checkpoint.
- Confirm task order, pressure, break, change, check, freeze, still-lacks, and
  review gates were preserved or explicitly marked missing from the source.
- Confirm `.agent-runs/` is ignored and no run artifact is staged.
- Confirm no article body was modified.

## Recording Examples

Good source sequence:

```text
$leetcode-tutorial-plan
  -> accepted human-readable plan

$tutorial-plan-record
  -> .agent-runs/tutorials/<slug>/plan.md

$tutorial-sketch
  -> .agent-runs/tutorials/<slug>/sketch.md
```

Bad source sequence:

```text
$tutorial-plan-record
  -> invent plan.md from a rough request
  -> treat the local record as if planning was done
```

The bad sequence lets artifact formatting replace actual planning. The record
must follow the completed plan.

## Decision Points

- If the plan is missing target path or slug, infer from the planned article
  path when possible; otherwise ask for the smallest missing value.
- If `.agent-runs/` is not ignored, add or request the appropriate ignore rule
  before writing a local run record.
- If the plan is incomplete enough that build would be unsafe, stop and route
  back to the appropriate plan skill.
- If the user asks to commit the record, explain that run records are local by
  default and ask whether they want to promote it to `docs/`.
- If an existing run targets the same article, update that run instead of
  creating a duplicate slug.

## Common Rationalizations

| Rationalization | Reality |
| --- | --- |
| "I should improve the plan while recording it." | Recording preserves the accepted plan. Replanning belongs to the plan skill. |
| "Plans are useful, so they should go in docs." | Most plan records are execution state, not long-lived reader documentation. |
| "The chat already has the plan." | Context compaction and handoffs are exactly why this skill exists. |
| "A loose summary is enough." | Build and check need task-level pressure, break, change, check, and freeze fields. |

## Red Flags

- The record changes task order or hides missing acceptance criteria.
- The file is written under `content/` or `.codex/skills/`.
- The plan record is staged for commit without explicit promotion.
- The record lacks target article path, tutorial type, or review gates.
- A new slug is created for a revision of the same article plan.
- The output report presents `plan.md` as the plan itself.

## Verification

- [ ] Accepted source plan and target article path are identified.
- [ ] Tutorial run boundary is new or reused intentionally.
- [ ] Run path is `.agent-runs/tutorials/<slug>/plan.md`.
- [ ] `.agent-runs/` is ignored by Git.
- [ ] Every recorded task maps to the source plan.
- [ ] Task order, pressure, check, freeze, and review gates are preserved.
- [ ] No article body or skill definition was modified.
- [ ] The record states the next recommended skill.

## Output Format

```text
## Plan Record Result
- status: recorded | blocked
- source_skill:
- run_path:
- target_article:
- next_recommended_skill:
- notes:
```

## Guardrails

- Do not create or rewrite tutorial body content.
- Do not approve a plan.
- Do not commit `.agent-runs/` artifacts.
- Do not store generated run records in `.codex/skills/`.
