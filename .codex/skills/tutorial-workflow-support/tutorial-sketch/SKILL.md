---
name: tutorial-sketch
description: v0.1.1 - Create a teaching skeleton and code-growth contract from a saved tutorial plan. Use before tutorial build when concept timing, first operational use, helper extraction, final-code placement, or checkpoint continuity must be made explicit.
---

# Tutorial Sketch

## Overview

Turn a saved tutorial plan into a pre-build teaching skeleton. The sketch is
not the article body. It is a contract for how the article will grow: first
screen pressure, section sequence, code/prose checkpoints, forbidden early
concepts, runnable code placement, and verification anchors.

Default input:

```text
.agent-runs/tutorials/<slug>/plan.md
```

Default output:

```text
.agent-runs/tutorials/<slug>/sketch.md
```

## When to Use

- A tutorial plan has been recorded and build is about to start.
- The tutorial has risk of hidden jumps, premature helpers, or final-answer
  leakage.
- Multiple build turns need a shared article skeleton and code-growth map.

**When NOT to use:** drafting the article body, reviewing a finished draft,
rewriting the plan, or enhancing an already stable post.

## Reference Map

- `references/sketch-template.md`
  Use for the default sketch shape.

## The Sketching Process

### Step 1: Load The Recorded Plan

Before sketching, read the plan record or equivalent supplied plan:

- Prefer `.agent-runs/tutorials/<slug>/plan.md`.
- Confirm the plan names the target article path, tutorial type, language,
  ordered tasks, pressure/break/change/check/freeze fields, and review gates.
- Confirm the next work is article build, not plan revision.
- If the plan record is missing, route to `$tutorial-plan-record`.

Do not sketch from memory. The sketch is a contract for later build work, so it
must be anchored to a concrete recorded plan.

### Step 2: Resolve The Sketch Boundary

Choose exactly one article and one tutorial run:

- Write to `.agent-runs/tutorials/<slug>/sketch.md`.
- Reuse the same slug as the plan record.
- Sketch one target article at a time.
- If a prior `sketch.md` exists for the same run, update it only when the same
  plan has been revised or the user asks to refresh the build contract.

The sketch boundary should match the plan boundary. Do not split one article
into multiple unrelated sketches unless the source plan already defines
separate publishable articles.

### Step 3: Derive The Teaching Backbone

Translate the plan into section and checkpoint order:

- Start from the first screen pressure: the concrete reader problem, example,
  failing baseline, or question that forces the first concept.
- Map each planned task to one checkpoint or clearly explain why a task must
  share a checkpoint with its neighbor.
- Keep the source order unless the plan is unsafe; if the order is unsafe,
  block and route back to the plan skill.
- Name what the reader should understand after each checkpoint.

The backbone should explain how the article grows, not provide publishable
paragraphs.

### Step 4: Define The Code-Growth Contract

For each checkpoint, write the allowed growth rule:

- previous visible baseline
- pressure that forces the next change
- exactly what new code, field, helper, invariant, or proof idea may appear
- where each newly named concept first does real work
- what must not appear yet
- evidence that proves the checkpoint is ready to freeze

Helpers should be forced by repetition or pressure. If a helper is useful only
after two duplicated paths exist, the sketch must forbid that helper before the
duplication appears in the article.

Variables, helpers, invariants, formulas, and rules follow the same timing
discipline. They should be introduced only when the current checkpoint needs
them. If a name must be introduced before it is used, the sketch must mark it as
`named_only`, name the later `first_operational_use`, and forbid any freeze or
capability wording that says the reader can use it before that later
checkpoint.

### Step 5: Place Final Code And Evidence

Decide where complete code and checks belong:

- Name the first section where final runnable code may appear.
- List allowed final-code logic and the checkpoint that earned each piece.
- List disallowed unexplained logic so the builder cannot sneak in a template
  answer.
- Attach concrete verification evidence to each checkpoint.

### Step 6: Write And Handoff

Write:

```text
.agent-runs/tutorials/<slug>/sketch.md
```

Use `references/sketch-template.md`. Report the sketch path, target article,
known risks, and next recommended build skill.

## Sketch Layer Boundaries

Belongs in `sketch.md`:

- first-screen pressure
- section skeleton
- checkpoint contract
- code-growth map
- concept timing and first operational use
- forbidden early concepts
- final runnable code placement
- review handoff

Belongs in `plan.md`:

- accepted task order
- target article path and tutorial type
- source facts and assumptions
- review gates and checkpoint commit guidance

Belongs in the article under `content/`:

- prose, examples, diagrams, and runnable code
- reader-facing explanation
- final solution

Belongs in `check.md`:

- whether the draft obeyed the plan and sketch
- drift findings and verification evidence

## Sketching Examples

Good sequence:

```text
$tutorial-plan-record
  -> .agent-runs/tutorials/<slug>/plan.md

$tutorial-sketch
  -> first pressure, checkpoint contract, forbidden early concepts

$leetcode-tutorial-build
  -> article grows one checkpoint at a time
```

Bad sequence:

```text
$tutorial-sketch
  -> write final article outline from memory
  -> introduce helper and final code before any pressure exists
```

The bad sequence turns the sketch into an answer outline. A sketch must protect
the build from hidden jumps.

## Decision Points

- If no plan record exists, route to `$tutorial-plan-record`.
- If the plan itself has hidden jumps, stop and route back to the source plan
  skill instead of inventing a new route inside the sketch.
- If the user asks for the full article, explain that sketch only prepares the
  build contract.
- If multiple article paths are present, sketch one article at a time.
- If a prior sketch exists, update it only when it belongs to the same slug and
  target article.
- If a helper appears useful but not forced yet, list it under forbidden early
  concepts instead of allowing it.

## Common Rationalizations

| Rationalization | Reality |
| --- | --- |
| "The builder can decide helper timing." | The sketch exists to prevent premature helpers and hidden final logic. |
| "A section outline is enough." | Tutorial sketches need code/prose growth and forbidden-early-concept rules. |
| "Final code can appear anywhere." | Final code should appear only after the article has earned every piece. |
| "I can define a variable now and use it later." | A name that does no work yet needs explicit pressure and must be marked named-only until first operational use. |
| "This should fix the plan." | Unsafe plans should be revised by the plan skill, not silently patched in sketch. |

## Red Flags

- The sketch starts with a named template instead of problem pressure.
- A helper or state variable appears before the pressure that requires it.
- A concept is introduced in one checkpoint but its first operational use is
  neither in the same checkpoint nor explicitly scheduled later.
- A checkpoint's freeze says the reader can use a concept that was only named,
  not operationally used.
- The final runnable code contains logic not mapped to earlier checkpoints.
- The sketch writes article prose instead of a build contract.
- The sketch changes plan order without sending the plan back for revision.
- The sketch has sections but no code-growth map.

## Verification

- [ ] A plan record or equivalent accepted plan input was read.
- [ ] Sketch path matches the plan slug and target article.
- [ ] First screen pressure is explicit.
- [ ] Each planned task maps to a checkpoint or justified shared checkpoint.
- [ ] Each checkpoint introduces one mechanism, rule, helper, or invariant.
- [ ] Each introduced concept has a same-checkpoint or explicitly later first
      operational use.
- [ ] Freeze wording is constrained so named-only concepts are not claimed as
      usable.
- [ ] Forbidden early concepts and helper timing are listed.
- [ ] Code-growth map states previous baseline, allowed change, and blocked
      premature logic.
- [ ] Final runnable code placement is specified.
- [ ] No article body was written.
- [ ] Next recommended build skill is identified.

## Output Format

```text
## Tutorial Sketch Result
- status: sketched | blocked
- plan_path:
- sketch_path:
- target_article:
- next_recommended_skill:
- risks:
```

## Guardrails

- Do not write the tutorial body.
- Do not self-approve the sketch as review pass.
- Do not commit `.agent-runs/` artifacts.
- Do not create detached final code.
