---
name: algorithm-tutorial-builder
description: v0.1.3 - Build teaching-first algorithm tutorials when the user wants the solution to be constructed step by step from problem evidence, with strict incremental code growth from one working version to the next and the final step yielding one runnable complete code version before any ACERS formatting, blog packaging, or post enhancement.
---

# Algorithm Tutorial Builder

## Trigger
Use when the user wants to learn or publish an algorithm solution as a true guided build rather than as a polished final article. Use this before blog formatting when the main need is to make the reasoning path teachable and incremental.

## Bundled Resources
- `docs/leetcode_std.md` for algorithm-post section expectations in this repo.
- `.codex/skills/algorithm-tutorial-builder/references/incremental-build-contract.md` for the strict step template and anti-pattern checks.

## Workflow
1. Gather only the supplied problem facts: statement, constraints, examples, target language, and any required implementation language.
2. Start from a tiny example or conflict pattern that exposes the pressure behind the solution. Do not start from a finished DFS/DP template unless the tiny example has already established why that template fits.
3. Define the smaller subproblem explicitly in plain language before writing code. The reader should be able to answer: “if this partial state is already fixed, what remains to solve?”
4. Build the tutorial as a sequence of working versions, not as disconnected explanation fragments. For each numbered step:
   - solve exactly one concrete problem,
   - add exactly one new state, rule, or helper when possible,
   - say whether the new code is an addition or a replacement,
   - show the exact snippet being added or replaced,
   - state what the current version can do now,
   - state what it still lacks.
5. Force the growth path to include these milestones when the problem family allows them:
   - problem evidence,
   - smaller subproblem,
   - smallest runnable skeleton,
   - first state variable,
   - completion condition,
   - first correctness rule,
   - first complete correct but still naive version,
   - at least one middle version if optimization is multi-stage,
   - final optimized version,
   - one slow trace,
   - one runnable complete code version.
6. When moving from a naive version to an optimized version, preserve the bridge:
   - first name the exact bottleneck,
   - then add the smallest helper state that removes one part of that bottleneck,
   - then wire that helper state back into the current code immediately,
   - then reassess what still remains slow.
7. Keep the output teaching-first:
   - prioritize reasoning and code growth,
   - do not require ACERS structure by default,
   - do not add Hugo front matter unless explicitly requested,
   - do not spend tokens on SEO, CTA, or marketing polish.
8. If the problem family is clear, adapt the ladder:
   - backtracking: tiny example -> layer meaning -> current choice -> completion -> first legality check -> first correct DFS -> one partial optimization -> full helper-state optimization,
   - DP: tiny example -> subproblem meaning -> base case -> first correct recurrence/table -> one optimization at a time,
   - graph: tiny example -> node/state meaning -> frontier container -> expansion rule -> first correct traversal -> optimization.
9. By default, let the final incremental step become the runnable complete code. Do not add a separate `Assemble the Full Code` section unless the user explicitly asks for a post format that requires it.
10. Do not add a separate `Reference Answer` section unless the user explicitly asks for a platform-specific wrapper such as a LeetCode `class Solution` shell. Even then, it must not introduce new logic.
11. End with assumptions, missing facts, and what was inferred versus supplied.

## Required Inputs
- Problem statement or faithful summary.
- At least one example.
- Target language for the tutorial (`zh` or `en`).
- Preferred code language if not obvious.

## Defaults
- teaching mode: derivation-first
- default implementation language: follow user request; otherwise prefer Python
- output shape: tutorial draft, not final publishable blog post
- structure policy: ACERS not required at this stage; publication formatting belongs to `acers-blog-formatter`
- step policy: every step must explicitly connect to the previous version
- final-code policy: default to one final runnable code block, not duplicated full-code sections

## Output Format
- Working Title: `<draft title>`
- Scope: `<problem + language + code language>`
- Tutorial Draft: `<guided-build markdown>`
- Notes: `<assumptions, missing facts, inferred pieces>`

## Guardrails
- Do not write a “teacher already knows the answer” retrospective explanation and call it a guided build.
- Do not skip the tiny example or the explicit smaller-subproblem step when they are needed to justify the recursion or state design.
- Do not ask “why do we need X?” before defining what `X` is.
- Do not introduce optimized helper state before the slower correct version exposes the need for it.
- Do not jump directly from the first correct version to the final optimized version when an intermediate version would make the evolution materially clearer.
- Do not leave steps as isolated concept fragments; each step must specify what was added or replaced in the previous code.
- Do not hide the bridge from “first correct version” to “optimized version”; if the optimization is staged, show the stages.
- Do not add ritual `Assemble the Full Code` or `Reference Answer` sections when the last step already yields a runnable complete solution.
- Do not jump from the statement to a named template or final trick in one paragraph.
- Do not force the result into ACERS structure unless the user explicitly asks for that in this stage.
- Do not automatically invoke blog-formatting or enhancement skills.
- Do not invent constraints, examples, or complexity claims.

## Verification
- Confirm the tutorial starts from problem evidence, not from a pre-decided final template.
- Confirm the smaller subproblem is stated explicitly in plain language.
- Confirm the tutorial contains a first correct version before the optimized version.
- Confirm each step says what was added or what was replaced from the previous version.
- Confirm each step says what the current version can do and what it still lacks.
- Confirm every optimization is justified by a pain point in the previous step.
- Confirm there is a middle version when the optimization naturally happens in more than one stage.
- Confirm the final step yields one runnable complete code version by default.
- Confirm no duplicate full-code section was added unless the user explicitly needed a separate wrapper form.
- Confirm the output is still a teaching-first draft rather than a packaging-heavy blog post.
- Confirm ACERS-only publishing concerns were left to the formatting stage unless explicitly requested.
