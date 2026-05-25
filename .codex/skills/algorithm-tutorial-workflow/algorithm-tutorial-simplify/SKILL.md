---
name: algorithm-tutorial-simplify
description: v0.1.0 - Simplify a reviewed and sufficiently deep algorithm tutorial without weakening derivation, evidence, invariants, correctness, complexity, or final runnable demo. Use when the article is structurally accepted but too repetitive, checklist-like, or heavy.
---

# Algorithm Tutorial Simplify

## Overview

Tighten a structurally accepted algorithm tutorial while preserving its
derivation and depth. This is not a repair skill for broken teaching structure.

## When to Use

- Review has passed or only prose-level issues remain.
- Deepening evidence is already present when required.
- The user wants clearer prose without changing the method or code behavior.

**When NOT to use:** planning, first drafting, review acceptance, deepening, or
algorithm changes.

## Simplification Loop

1. Confirm Gate Status
   - Verify review/deepening status before editing.
2. Protect Required Content
   - Preserve pressure, mechanism, invariant, correctness, complexity,
     counterexample, engineering reality, checks, and final runnable demo.
3. Tighten Prose
   - Remove repetition.
   - Convert template-like scaffolding to reader-facing prose.
4. Preserve Code and Claims
   - Do not change algorithm behavior or complexity claims.
   - Do not remove evidence anchors.
5. Report Risks
   - Name any remaining weak depth or clarity issue.

## Output Format

```markdown
## Simplification Summary
- ...

## Preserved Evidence
- ...

## Remaining Risks
- ...
```

## Guardrails

- Do not simplify away invariants or correctness.
- Do not remove counterexamples or concrete anchors.
- Do not change code behavior.
- Do not hide missing review/deepening gates.

## Verification

- [ ] Review/deepening status was checked.
- [ ] Derivation and depth evidence remain.
- [ ] Code behavior was not changed.
- [ ] Remaining risks are reported.
