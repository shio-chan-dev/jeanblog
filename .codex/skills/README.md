# Skills Usage Guide

This directory contains local skills for writing, reviewing, improving, and
committing jeanblog content. Use a skill by naming it in the request, for
example:

```text
$leetcode-tutorial-plan Plan a tutorial for LeetCode 208.
$tutorial-reviewer Check whether this draft passes the teaching bar.
```

Each skill has its own `SKILL.md`. Workflow packages may also have a local
`README.md` with the full phase order.

## Quick Selection

| Scenario | Use | Notes |
| --- | --- | --- |
| Plan a tutorial for one concrete LeetCode/OJ problem | `$leetcode-tutorial-plan` | Use for one problem with input/output behavior, examples, and constraints. |
| Save an accepted tutorial plan locally | `$tutorial-plan-record` | Writes recoverable run state under ignored `.agent-runs/tutorials/<slug>/`. |
| Create a pre-build teaching skeleton | `$tutorial-sketch` | Turns a saved plan into section order, code-growth rules, and forbidden early concepts. |
| Draft the next planned LeetCode tutorial step | `$leetcode-tutorial-build` | Writes one checkpoint only, then stops for review. |
| Check a tutorial draft against plan/sketch | `$tutorial-check` | Mechanical drift check before review; not a replacement for review. |
| Review a LeetCode tutorial checkpoint | `$leetcode-tutorial-review` | Gives `pass`, `revise`, or `block`; build cannot self-approve. |
| Tighten a reviewed LeetCode tutorial | `$leetcode-tutorial-simplify` | Use after the structure has passed review. |
| Plan a standalone algorithm/data-structure tutorial | `$algorithm-tutorial-plan` | Use for the method itself, such as Trie, Union-Find, Dijkstra, or Segment Tree. |
| Draft the next planned algorithm tutorial checkpoint | `$algorithm-tutorial-build` | Writes one derivation checkpoint only, then stops for review/deepening. |
| Review an algorithm tutorial checkpoint | `$algorithm-tutorial-review` | Checks pressure, mechanism, invariant, evidence, and code continuity. |
| Deepen selected algorithm concepts | `$algorithm-tutorial-deepen` | Use for stronger invariants, correctness, complexity, counterexamples, or engineering reality. |
| Tighten a reviewed algorithm tutorial | `$algorithm-tutorial-simplify` | Simplifies prose without weakening derivation or proof. |
| Review any teaching-first tutorial draft | `$tutorial-reviewer` | Binary `Pass`/`Fail` structural review; does not rewrite. |
| Enhance a stable technical post | `$tech-post-enhancer` | Use for SEO, title, metadata, engineering add-ons, FAQs, or extra language code after the core is stable. |
| Write a tool/workflow/system evaluation post | `$thoughts-evaluation-writer` | Use for conclusion-first comparison or personal judgment posts, not tutorials. |

## Main Workflows

### LeetCode / OJ Problem Tutorial

Use this when the subject is one concrete judged problem.

```text
$leetcode-tutorial-plan
-> $tutorial-plan-record
-> $tutorial-sketch
-> $leetcode-tutorial-build
-> $tutorial-check
-> $leetcode-tutorial-review
-> optional commit/checkpoint
-> next build step
-> $leetcode-tutorial-simplify after review pass
```

Full workflow README:
[`leetcode-tutorial-workflow/README.md`](leetcode-tutorial-workflow/README.md)

### Algorithm / Data-Structure Tutorial

Use this when the subject is the technique itself, not one OJ problem.

```text
$algorithm-tutorial-plan
-> $tutorial-plan-record
-> $tutorial-sketch
-> $algorithm-tutorial-build
-> $tutorial-check
-> $algorithm-tutorial-review
-> $algorithm-tutorial-deepen when needed
-> $algorithm-tutorial-simplify after review/deepening
```

Full workflow README:
[`algorithm-tutorial-workflow/README.md`](algorithm-tutorial-workflow/README.md)

### Tutorial Workflow Support

Use this when a tutorial plan needs local process state before or during
multi-turn writing. These skills create run artifacts under ignored
`.agent-runs/tutorials/<slug>/`.

```text
$tutorial-plan-record
-> $tutorial-sketch
-> build skill
-> $tutorial-check
-> review skill
```

Full workflow README:
[`tutorial-workflow-support/README.md`](tutorial-workflow-support/README.md)

## Standalone Skills

### `$tutorial-reviewer`

Use when a draft already exists and you want a structural teaching review:

- Does it start from pressure or a tiny task?
- Does code grow through visible checkpoints?
- Does the final full code appear only after the build earns it?
- Should it pass or fail before publication?

This skill reports revision targets. It should not rewrite the whole tutorial.

### `$tutorial-plan-record`

Use after a LeetCode or algorithm tutorial plan is accepted and the work will
continue across turns. It records task order, pressure/break/change/check/freeze
fields, article path, and review gates under `.agent-runs/`.

It should not re-plan or create publishable docs.

### `$tutorial-sketch`

Use before build when helper timing, section order, final code placement, or
concept sequencing needs to be made explicit. It reads the local plan record and
creates a teaching skeleton, not the article body.

### `$tutorial-check`

Use after a build checkpoint or draft to compare the article with its plan and
sketch before sending it to the appropriate review skill. It catches mechanical
drift; it does not replace `$leetcode-tutorial-review`,
`$algorithm-tutorial-review`, or `$tutorial-reviewer`.

### `$tech-post-enhancer`

Use only after the core tutorial/post is stable. Good requests:

- strengthen title, description, and SEO metadata
- add FAQ or summary sections
- add engineering scenarios
- add more language implementations when the logic is already clear

Do not use it to fix a broken guided-build chain.

### `$thoughts-evaluation-writer`

Use for evaluation or judgment posts, for example:

- comparing two tools or workflows
- recording what a trial proved
- explaining what fits and what does not fit personal use

Do not use it for LeetCode, algorithm tutorials, or pure factual explainers.

## Common Decisions

- If the request is about **one problem**, prefer the LeetCode workflow.
- If the request is about **one method or data structure**, prefer the
  algorithm tutorial workflow.
- If the request asks **"does this pass?"**, use `$tutorial-reviewer`.
- If a tutorial will span **multiple turns or agents**, use
  `$tutorial-plan-record` and `$tutorial-sketch` before build.
- If a draft needs **plan/sketch adherence checking**, use `$tutorial-check`
  before the review skill.
- If the request asks for **SEO, polish, title, FAQ, or extra sections**, use
  `$tech-post-enhancer` only after the teaching structure is already sound.
- If the request asks for a **tool/workflow opinion**, use
  `$thoughts-evaluation-writer`.

## Guardrails

- Keep planning, building, reviewing, deepening, and simplifying as separate
  phases when using the workflow packages.
- Keep `.agent-runs/` artifacts local and untracked by default.
- Build skills may run self-checks, but review skills decide whether a
  checkpoint passes.
- Do not use enhancement skills to hide unresolved tutorial-structure issues.
- Read the selected `SKILL.md` before acting; it may require additional
  reference files.
