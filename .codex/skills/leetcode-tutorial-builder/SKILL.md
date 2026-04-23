---
name: leetcode-tutorial-builder
description: v0.2.0 - Build publishable teaching-first LeetCode and algorithm-problem tutorials when the user wants one concrete problem solution to be constructed step by step from problem evidence, with strict incremental code growth, minimal front matter, and one final runnable complete code version.
---

# LeetCode Tutorial Builder

## Trigger
Use when the user asks for a concrete algorithm problem tutorial: LeetCode, Hot100, Codeforces, AtCoder, Luogu, or a custom OJ-style problem with explicit input/output behavior. Use it when the main need is to make one problem solution teachable, incremental, and directly publishable in this Hugo blog. Do not use it for tutorials about an algorithm, data structure, or technique itself when there is no single problem statement to solve.

## Bundled Resources
- `docs/leetcode_std.md` for algorithm-post section expectations in this repo.
- `.codex/skills/leetcode-tutorial-builder/references/incremental-build-contract.md` for the strict step template and anti-pattern checks.

## Workflow
1. Gather only the supplied problem facts: statement, constraints, examples, target language, and any required implementation language.
2. Choose the output path and slug before drafting the post body.
   - Default to `content/<lang>/alg/leetcode/<slug>.md`.
   - If the request clearly belongs to an existing series folder such as `hot100` or `binary-search` and that folder already exists, place it there instead.
   - Keep ASCII kebab-case filenames; preserve the slug once chosen.
3. Start from a tiny example or conflict pattern that exposes the pressure behind the solution. Do not start from a finished DFS/DP template unless the tiny example has already established why that template fits.
4. Define the smaller subproblem explicitly in plain language before writing code. The reader should be able to answer: “if this partial state is already fixed, what remains to solve?”
5. Build the tutorial as a sequence of working versions, not as disconnected explanation fragments. For each numbered step:
   - solve exactly one concrete problem,
   - add exactly one new state, rule, or helper when possible,
   - say whether the new code is an addition or a replacement,
   - show the exact snippet being added or replaced,
   - state what the current version can do now,
   - state what it still lacks.
6. Force the growth path to include these milestones when the problem family allows them:
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
7. When moving from a naive version to an optimized version, preserve the bridge:
   - first name the exact bottleneck,
   - then add the smallest helper state that removes one part of that bottleneck,
   - then wire that helper state back into the current code immediately,
   - then reassess what still remains slow.
8. Add minimal publishable front matter and repo-aligned taxonomy:
   - `title`, `date`, `draft`, `categories`, `tags`
   - Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for `date`.
   - Default `draft` to `false` unless the user says otherwise.
   - Default `categories` to `["LeetCode"]` unless nearby repo examples or the requested series clearly imply a different existing category.
   - Keep tags basic and topic-driven; do not expand into SEO keyword lists.
9. Keep the output teaching-first:
   - prioritize reasoning and code growth,
   - produce publishable Hugo Markdown in the chosen `content/` path,
   - use a clear reader-facing structure without requiring a separate formatting stage,
   - do not spend tokens on SEO, CTA, or marketing polish.
10. If the problem family is clear, adapt the ladder:
   - backtracking: tiny example -> layer meaning -> current choice -> completion -> first legality check -> first correct DFS -> one partial optimization -> full helper-state optimization,
   - DP: tiny example -> subproblem meaning -> base case -> first correct recurrence/table -> one optimization at a time,
   - graph: tiny example -> node/state meaning -> frontier container -> expansion rule -> first correct traversal -> optimization.
11. By default, let the final incremental step become the runnable complete code. Do not add a separate `Assemble the Full Code` section unless the user explicitly asks for a post format that requires it.
12. Do not add a separate `Reference Answer` section unless the user explicitly asks for a platform-specific wrapper such as a LeetCode `class Solution` shell. Even then, it must not introduce new logic.
13. End with assumptions, missing facts, and what was inferred versus supplied.

## Required Inputs
- Problem statement or faithful summary.
- At least one example.
- Target language for the tutorial (`zh` or `en`).
- Preferred code language if not obvious.

## Defaults
- teaching mode: derivation-first
- default implementation language: follow user request; otherwise prefer Python
- output shape: publishable tutorial post, not a draft-only reasoning note
- output path: `content/<lang>/alg/leetcode/<slug>.md` unless an existing sub-series folder clearly fits better
- front matter policy: minimal Hugo front matter only
- step policy: every step must explicitly connect to the previous version
- final-code policy: default to one final runnable code block, not duplicated full-code sections
- metadata policy: keep only minimal publishable metadata; enhancement belongs to `tech-post-enhancer`
- domain policy: this skill is for problem-solving tutorials, not for standalone algorithm tutorials

## Output Format
- Path: `<file path>`
- Scope: `<problem + language + code language>`
- Taxonomy: `<categories/tags>`
- Publishable Tutorial: `<guided-build markdown with minimal front matter>`
- Notes: `<assumptions, missing facts, inferred pieces>`

## Guardrails
- Do not write a “teacher already knows the answer” retrospective explanation and call it a guided build.
- Do not skip the tiny example or the explicit smaller-subproblem step when they are needed to justify the recursion or state design.
- Do not use this skill for a concept-first algorithm tutorial such as Transformer, Union-Find, Segment Tree, Bloom Filter, or PageRank when the user is not solving one concrete problem.
- Do not ask “why do we need X?” before defining what `X` is.
- Do not introduce optimized helper state before the slower correct version exposes the need for it.
- Do not jump directly from the first correct version to the final optimized version when an intermediate version would make the evolution materially clearer.
- Do not leave steps as isolated concept fragments; each step must specify what was added or replaced in the previous code.
- Do not hide the bridge from “first correct version” to “optimized version”; if the optimization is staged, show the stages.
- Do not add ritual `Assemble the Full Code` or `Reference Answer` sections when the last step already yields a runnable complete solution.
- Do not jump from the statement to a named template or final trick in one paragraph.
- Do not add `readingTime`, `keywords`, CTA sections, or other enhancement-layer metadata by default.
- Do not assume a separate formatting stage exists.
- Do not automatically invoke enhancement skills.
- Do not invent constraints, examples, or complexity claims.

## Verification
- Confirm the input is one concrete problem rather than a general algorithm/topic tutorial request.
- Confirm the tutorial starts from problem evidence, not from a pre-decided final template.
- Confirm the smaller subproblem is stated explicitly in plain language.
- Confirm the tutorial contains a first correct version before the optimized version.
- Confirm each step says what was added or what was replaced from the previous version.
- Confirm each step says what the current version can do and what it still lacks.
- Confirm every optimization is justified by a pain point in the previous step.
- Confirm there is a middle version when the optimization naturally happens in more than one stage.
- Confirm the final step yields one runnable complete code version by default.
- Confirm no duplicate full-code section was added unless the user explicitly needed a separate wrapper form.
- Confirm the output is publishable Hugo Markdown with minimal front matter.
- Confirm the path, slug, and basic taxonomy follow existing repo conventions.
