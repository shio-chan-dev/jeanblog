# LeetCode Tutorial Workflow

This package builds publishable LeetCode and OJ-style problem tutorials through
separate planning, writing, review, and simplification gates.

Use this workflow when the object is one concrete problem with input/output
behavior, examples, and constraints. Do not use it for standalone algorithm or
data-structure concept tutorials; those belong to the algorithm tutorial
workflow.

## Standard

The tutorial must grow from problem evidence into connected checkpoints:

```text
problem facts
-> tiny example pressure
-> smaller subproblem
-> first correct baseline
-> optimization pressure
-> final runnable solution checkpoint
```

Build and review are separate roles:

```text
plan
-> build one tutorial step
-> review that step
-> optional checkpoint commit
-> next step
```

`leetcode-tutorial-build` may run self-checks, but self-checks are evidence,
not acceptance. Only `leetcode-tutorial-review` can give the `pass` verdict
that allows the next step.

## Skills

| Phase | Skill | Purpose |
| --- | --- | --- |
| Plan | [`leetcode-tutorial-plan`](leetcode-tutorial-plan/SKILL.md) | Turn one problem into a task-first tutorial build plan. |
| Build | [`leetcode-tutorial-build`](leetcode-tutorial-build/SKILL.md) | Write exactly one planned tutorial step and stop for review. |
| Review | [`leetcode-tutorial-review`](leetcode-tutorial-review/SKILL.md) | Gate one step or a full draft with `pass`, `revise`, or `block`. |
| Simplify | [`leetcode-tutorial-simplify`](leetcode-tutorial-simplify/SKILL.md) | Tighten a reviewed tutorial without breaking the build chain. |

## Usage

```text
$leetcode-tutorial-plan
$leetcode-tutorial-build
$leetcode-tutorial-review
$leetcode-tutorial-simplify
```

The legacy `$leetcode-tutorial-builder` entrypoint is kept as a compatibility
router. New work should use this workflow directly.
