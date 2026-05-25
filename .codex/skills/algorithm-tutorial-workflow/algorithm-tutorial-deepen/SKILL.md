---
name: algorithm-tutorial-deepen
description: v0.1.0 - Deepen 1-2 selected concepts in a reviewed algorithm tutorial using PDKH standards. Use when a method tutorial needs stronger invariant, formalization, correctness, complexity, counterexample, or engineering-reality evidence without broadening into unrelated topics.
---

# Algorithm Tutorial Deepen

## Overview

Deepen selected concepts after the tutorial structure is in place. This skill
does not add broad new topics. It applies the PDKH ladder to 1-2 concepts so
the article becomes more rigorous.

## When to Use

- A plan selected concepts for deepening.
- Review passed structure but requested more depth.
- The tutorial needs stronger invariants, formulas, correctness, complexity,
  counterexamples, or engineering caveats.

**When NOT to use:** first drafting, review acceptance, broad expansion,
LeetCode problem optimization, or SEO enhancement.

## Reference Map

- `../../algorithm-tutorial-builder/references/deepening-ladder.md`
- `../../algorithm-tutorial-builder/references/depth-checklist.md`

## The Deepening Loop

1. Select Concepts
   - Use only 1-2 concepts named by the plan or review.
2. Apply PDKH
   - Problem reframe.
   - Minimal worked example.
   - Invariant or contract.
   - Formalization.
   - Correctness sketch.
   - Thresholds and complexity.
   - Counterexample or failure mode.
   - Engineering reality.
3. Keep Scope Narrow
   - Add evidence to existing sections or one targeted deepening section.
   - Do not create unrelated top-level topics.
4. Report Evidence
   - Name which PDKH steps were satisfied.

## Output Format

```markdown
## Deepening Result
- status: needs_review
- concepts:
- PDKH evidence:

## Added Or Revised Content
...

## Review Handoff
- review_skill: algorithm-tutorial-review
```

## Guardrails

- Do not broaden the article to meet length.
- Do not add generic background.
- Do not invent claims or benchmarks.
- Do not skip counterexamples when the concept has common failure modes.

## Verification

- [ ] 1-2 concepts were selected.
- [ ] PDKH evidence was recorded.
- [ ] Added content is concrete and bounded.
- [ ] Output returns to review instead of self-approving.
