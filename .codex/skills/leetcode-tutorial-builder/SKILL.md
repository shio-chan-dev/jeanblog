---
name: leetcode-tutorial-builder
description: v0.3.0 - Compatibility router for LeetCode and OJ-style tutorial work. Use when the user calls the legacy builder name; route new problem-tutorial work to leetcode-tutorial-workflow plan/build/review/simplify so build cannot self-approve or write a whole multi-step tutorial in one pass.
---

# LeetCode Tutorial Builder

## Overview

This legacy entrypoint now routes LeetCode and OJ-style problem tutorial work
to the review-gated `leetcode-tutorial-workflow` package.

The old behavior was a single builder that could plan, draft, and accept a
whole tutorial in one flow. New work must use separate phases:

```text
leetcode-tutorial-plan
-> leetcode-tutorial-build writes one step
-> leetcode-tutorial-review gates the step
-> optional checkpoint commit
-> next step
-> leetcode-tutorial-simplify after review pass
```

Build self-checks are useful evidence, but they are not acceptance. Only
`leetcode-tutorial-review` can allow the next step.

## When to Use

- The user explicitly invokes `$leetcode-tutorial-builder`.
- The user asks for a LeetCode, Hot100, Codeforces, AtCoder, Luogu, or
  OJ-style problem tutorial through the legacy skill name.
- A previous prompt or project config still points at this skill.

**When NOT to use:** standalone algorithm/method/data-structure tutorials,
stable post enhancement, SEO polish, or thoughts/evaluation posts.

## Routing

- Use `../leetcode-tutorial-workflow/leetcode-tutorial-plan/SKILL.md` when no
  task-first tutorial plan exists.
- Use `../leetcode-tutorial-workflow/leetcode-tutorial-build/SKILL.md` only for
  the next planned step, then stop with `needs_review`.
- Use `../leetcode-tutorial-workflow/leetcode-tutorial-review/SKILL.md` before
  advancing to the next step or checkpoint commit.
- Use `../leetcode-tutorial-workflow/leetcode-tutorial-simplify/SKILL.md` only
  after review has passed.

## Compatibility Loop

1. Classify the user request.
   - If it asks for a plan, route to `leetcode-tutorial-plan`.
   - If it asks to draft content and no plan exists, route to
     `leetcode-tutorial-plan` first.
   - If it asks to continue drafting from a plan, route to
     `leetcode-tutorial-build` for one step only.
   - If it asks whether a step is acceptable, route to
     `leetcode-tutorial-review`.
   - If it asks to polish an accepted draft, route to
     `leetcode-tutorial-simplify`.
2. Preserve problem-first discipline.
   - Problem requirement, input/output, examples, and constraints must appear
     before derivation.
3. Preserve review-gated progression.
   - Do not continue to Step N+1 until Step N has review-pass evidence.
4. Report the routed skill.
   - Say which workflow skill should handle the current phase.

## Guardrails

- Do not write a full multi-step tutorial directly from this legacy entrypoint.
- Do not self-approve a build step.
- Do not create a checkpoint commit without review-pass evidence.
- Do not use this skill for concept-first algorithm tutorials.
- Do not invent constraints, examples, or complexity claims.

## Verification

- [ ] The request was routed to one workflow phase.
- [ ] Problem tutorial scope was confirmed.
- [ ] Build self-check and review acceptance were kept separate.
- [ ] The legacy entrypoint did not produce a full tutorial body directly.
