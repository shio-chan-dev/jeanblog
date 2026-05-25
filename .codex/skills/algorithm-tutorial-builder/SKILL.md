---
name: algorithm-tutorial-builder
description: v0.3.0 - Compatibility router for standalone algorithm, data-structure, and method tutorials. Use when the user calls the legacy builder name; route new concept-tutorial work to algorithm-tutorial-workflow plan/build/review/deepen/simplify so derivation, depth, and review gates stay separate.
---

# Algorithm Tutorial Builder

## Overview

This legacy entrypoint now routes standalone algorithm, data-structure, and
method tutorial work to the review-gated `algorithm-tutorial-workflow` package.

The old builder combined planning, drafting, depth passes, validation, and
acceptance. New work should keep those phases separate:

```text
algorithm-tutorial-plan
-> algorithm-tutorial-build writes one checkpoint
-> algorithm-tutorial-review gates structure
-> algorithm-tutorial-deepen strengthens selected concepts
-> algorithm-tutorial-simplify after review/deepening
```

Build self-checks are evidence, not acceptance. Review/deepening gates decide
whether the tutorial can advance.

## When to Use

- The user explicitly invokes `$algorithm-tutorial-builder`.
- The subject is a concrete algorithm, data structure, technique, architecture,
  or method such as Union-Find, Segment Tree, Dijkstra, Bloom Filter,
  Transformer, or PageRank.
- A previous prompt or project config still points at this legacy skill.

**When NOT to use:** one concrete LeetCode/OJ-style problem tutorial, SEO
enhancement, thoughts/evaluation posts, or stable post polishing.

## Routing

- Use `../algorithm-tutorial-workflow/algorithm-tutorial-plan/SKILL.md` when no
  task-first tutorial plan exists.
- Use `../algorithm-tutorial-workflow/algorithm-tutorial-build/SKILL.md` only
  for the next planned checkpoint, then stop for review/deepening.
- Use `../algorithm-tutorial-workflow/algorithm-tutorial-review/SKILL.md`
  before advancing to another checkpoint.
- Use `../algorithm-tutorial-workflow/algorithm-tutorial-deepen/SKILL.md` when
  the plan or review calls for PDKH/depth work.
- Use `../algorithm-tutorial-workflow/algorithm-tutorial-simplify/SKILL.md`
  only after structure and depth are accepted.

## Preserved Resources

The workflow continues to rely on this package's resources:

- `references/derivation-first-explanations.md`
- `references/language-selection-rubric.md`
- `references/depth-checklist.md`
- `references/deepening-ladder.md`
- `assets/algorithm-tutorial-template.md`
- reinforcement resources when explicitly enabled

## Compatibility Loop

1. Classify the request.
   - If it asks for a plan, route to `algorithm-tutorial-plan`.
   - If it asks to draft and no plan exists, route to
     `algorithm-tutorial-plan` first.
   - If it asks to continue drafting from a plan, route to
     `algorithm-tutorial-build` for one checkpoint only.
   - If it asks for acceptance, route to `algorithm-tutorial-review`.
   - If it asks for deeper rigor, route to `algorithm-tutorial-deepen`.
   - If it asks for polish after acceptance, route to
     `algorithm-tutorial-simplify`.
2. Preserve concept-tutorial identity.
   - Do not force the article into LeetCode-style problem-solution framing.
3. Preserve depth gates.
   - Invariants, correctness, complexity, counterexamples, and engineering
     tradeoffs are part of the workflow, not optional filler.
4. Report the routed skill.

## Guardrails

- Do not write a full multi-section tutorial directly from this legacy
  entrypoint.
- Do not self-approve a build checkpoint.
- Do not route standalone method tutorials to the LeetCode workflow.
- Do not start from glossary, formula sheet, or component preview before a real
  pressure point.
- Do not invent benchmarks, constraints, or claims.

## Verification

- [ ] The request was routed to one workflow phase.
- [ ] The subject is a concept/method tutorial, not one OJ problem.
- [ ] Build, review, and deepening gates remain separate.
- [ ] The legacy entrypoint did not produce a full tutorial body directly.
