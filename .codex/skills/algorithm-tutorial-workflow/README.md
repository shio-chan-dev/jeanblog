# Algorithm Tutorial Workflow

This package builds standalone algorithm, data-structure, and method tutorials
through separate planning, writing, review, deepening, and simplification
gates.

Use it when the subject is the method itself, such as Union-Find, Segment Tree,
Dijkstra, Bloom Filter, Transformer, or PageRank. Use the LeetCode tutorial
workflow when the subject is one concrete OJ-style problem.

## Standard

Algorithm tutorials must preserve two layers:

- derivation-first construction: pressure -> missing capability -> one
  mechanism -> check -> checkpoint
- algorithm depth: invariant, formalization, correctness, complexity,
  counterexample, and engineering reality

Build and review remain separate:

```text
plan
-> build one section/checkpoint
-> review gate
-> deepen selected concepts
-> simplify after review/deepening
```

`algorithm-tutorial-build` may self-check, but it cannot self-approve. Review
or deepen gates decide whether the tutorial can advance.

## Skills

| Phase | Skill | Purpose |
| --- | --- | --- |
| Plan | [`algorithm-tutorial-plan`](algorithm-tutorial-plan/SKILL.md) | Plan one method tutorial with derivation and depth tasks. |
| Build | [`algorithm-tutorial-build`](algorithm-tutorial-build/SKILL.md) | Draft one planned tutorial checkpoint and stop for review. |
| Review | [`algorithm-tutorial-review`](algorithm-tutorial-review/SKILL.md) | Gate structure, derivation, code continuity, and final demo placement. |
| Deepen | [`algorithm-tutorial-deepen`](algorithm-tutorial-deepen/SKILL.md) | Apply PDKH/depth standards to 1-2 selected concepts. |
| Simplify | [`algorithm-tutorial-simplify`](algorithm-tutorial-simplify/SKILL.md) | Tighten a reviewed/deepened tutorial without weakening the teaching chain. |

The legacy `$algorithm-tutorial-builder` entrypoint is kept as a compatibility
router. New work should use this workflow directly.
